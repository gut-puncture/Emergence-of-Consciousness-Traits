"""Chunk-level DeepSeek relabeling for rejected final-35k documents.

This module is deliberately narrow: it turns document-level reject labels from
the existing final 35k teacher run into private, natural-block-backed chunk
labeling work. Public artifacts contain hashes and counts only; raw text exists
only in local variables during refetch and request construction.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from twc.data.normalize import normalize_text
from twc.data.teacher_labeling import (
    FINAL_LABEL_POLICY_VERSION,
    PROMPT_V3_DEEPSEEK_TEACHER_MAX_OUTPUT_TOKENS,
    TEXT_HASH_ALGORITHM_UTF8_SHA256_V1,
    DeepSeekLabelingConfig,
    PriceSnapshot,
    _adaptive_missing_vote_plan,
    _append_label_row_known_unique,
    _config_payload,
    _finish_reason,
    _hash,
    _latency_summary_from_rows,
    _price_from_official_snapshot,
    _private_refetch_text_hash,
    _public_config_payload,
    _read_jsonl_objects,
    _reject_live_artifact_leakage,
    _row_cost_sum,
    _rows_by_candidate,
    _schedule_entry_for_candidate,
    _status_counts,
    _token_usage_summary,
    _utc_now_iso,
    _write_json_atomic,
    actual_label_cost_usd,
    build_deepseek_request,
    call_deepseek_live,
    estimate_label_cost_usd,
    parse_teacher_response,
    prepare_deepseek_live_call,
    run_with_retries,
    teacher_label_parse_error_record,
)
from twc.pretraining.chunked_clean_run import (
    ChunkedSegmentationConfig,
    SourceDocumentRecord,
    segment_source_document,
    segmentation_policy_hash,
)


CHUNK_RELABEL_MANIFEST_SCHEMA_VERSION = 1
CHUNK_PRIVATE_MAP_SCHEMA_VERSION = 1
CHUNK_RELABEL_REPORT_SCHEMA_VERSION = 1
CHUNK_RELABEL_PROGRESS_SCHEMA_VERSION = 1
CHUNK_RELABEL_REPAIR_QUEUE_SCHEMA_VERSION = 1
CHUNK_RELABEL_MANIFEST_SCOPE = "rejected_35k_chunk_relabel_candidates"
CHUNK_RELABEL_POLICY_VERSION = "rejected_35k_chunk_relabel_v1"
CHUNK_RELABEL_PRIVATE_MAP_VERSION = "rejected_35k_chunk_private_map_v1"
INHERITED_SAFE_PARENT_PRIVATE_MAP_KIND = "twc_inherited_safe_parent_private_map"
INHERITED_SAFE_PARENT_PRIVATE_MAP_ROUTE = "safe_parent_source_refetch"
INHERITED_SAFE_PARENT_POLICY_VERSION = "inherited_parent_safe_v2"
INHERITED_SAFE_PARENT_LABEL_SOURCE = "final_35k_bof3_parent_label"
FRESH_CHUNK_REPAIR_CONSOLIDATION_RULE = "first_parsed_fresh_vote_per_required_vote_index"
FRESH_CHUNK_REPAIR_LABEL_SOURCE = "fresh_repair_overlay"
CANONICAL_CHUNK_LABEL_SOURCE = "canonical"

_SHA256_LENGTH = 64
_PUBLIC_FALSE_FLAGS = (
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "accepted_manifest",
    "labels_are_corpus_approval",
    "training_shards_created",
    "classifier_training_performed",
)


class ChunkRelabelBuildArtifacts:
    """Public build artifacts plus an explicit private in-memory operator cache."""

    __slots__ = ("manifest", "private_map", "report", "_private_parent_rows")

    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        private_map: dict[str, Any],
        report: dict[str, Any],
        private_parent_rows: dict[str, dict[str, Any]],
    ) -> None:
        self.manifest = manifest
        self.private_map = private_map
        self.report = report
        self._private_parent_rows = private_parent_rows

    def __getitem__(self, key: str) -> dict[str, Any]:
        if key == "manifest":
            return self.manifest
        if key == "private_map":
            return self.private_map
        if key == "report":
            return self.report
        raise KeyError(key)

    def private_parent_rows_for_operator(self) -> dict[str, dict[str, Any]]:
        return self._private_parent_rows


@dataclass(frozen=True)
class ChunkRelabelPolicy:
    required_vote_count: int = 3
    budget_cap_usd: float = 100.0
    minimum_balance_reserve_usd: float = 0.0
    target_concurrency: int = 64
    tranche_chunks: int = 512
    private_refetch_concurrency: int = 64
    policy_version: str = CHUNK_RELABEL_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.required_vote_count != 3:
            raise ValueError("chunk relabel requires exactly three BoF votes")
        if self.budget_cap_usd <= 0 or self.budget_cap_usd > 100.0:
            raise ValueError("chunk relabel budget cap must be positive and no more than USD 100.00")
        if self.minimum_balance_reserve_usd < 0:
            raise ValueError("chunk relabel minimum balance reserve must be non-negative")
        if self.target_concurrency < 1 or self.target_concurrency > 500:
            raise ValueError("chunk relabel target_concurrency must be between 1 and 500")
        if self.tranche_chunks < 1:
            raise ValueError("chunk relabel tranche_chunks must be positive")
        if self.private_refetch_concurrency < 1 or self.private_refetch_concurrency > 500:
            raise ValueError("chunk relabel private_refetch_concurrency must be between 1 and 500")
        if self.policy_version != CHUNK_RELABEL_POLICY_VERSION:
            raise ValueError(f"chunk relabel policy_version must be {CHUNK_RELABEL_POLICY_VERSION}")


def build_rejected_chunk_relabel_artifacts(
    *,
    source_manifest: dict[str, Any],
    final_labels: list[dict[str, Any]],
    source_schedule: dict[str, Any],
    row_provider: Any,
    output_dir: str | Path,
    target_proxy_tokens: int = 1024,
    max_parent_docs: int | None = None,
    skip_parent_docs: int = 0,
    segmentation_config: ChunkedSegmentationConfig | None = None,
) -> ChunkRelabelBuildArtifacts:
    """Build no-raw chunk relabel artifacts from rejected final-35k parents."""

    started = time.monotonic()
    target_proxy_tokens = _require_positive_int(target_proxy_tokens, "target_proxy_tokens")
    _validate_source_schedule_no_raw(source_schedule)
    if max_parent_docs is not None:
        max_parent_docs = _require_positive_int(max_parent_docs, "max_parent_docs")
    skip_parent_docs = _require_nonnegative_int(skip_parent_docs, "skip_parent_docs")
    segmentation_config = segmentation_config or ChunkedSegmentationConfig()
    parent_candidates = _rejected_parent_candidates(
        source_manifest=source_manifest,
        final_labels=final_labels,
        max_parent_docs=max_parent_docs,
        skip_parent_docs=skip_parent_docs,
    )
    parent_rows = _fetch_parent_rows(parent_candidates, schedule=source_schedule, row_provider=row_provider)

    candidates: list[dict[str, Any]] = []
    private_chunks: list[dict[str, Any]] = []
    parent_reports: list[dict[str, Any]] = []
    chunk_index_global = 0
    for parent in parent_candidates:
        parent_id = str(parent["candidate_id"])
        private_text = _private_parent_text(parent, parent_rows[parent_id])
        split = "validation" if str(parent.get("dataset_split") or "") == "validation" else "train"
        segmentation = segment_source_document(
            SourceDocumentRecord(
                source_document_id_hash=parent_id,
                split=split,
                raw_text=private_text,
                source_lineage_hash=str(parent.get("candidate_hash") or "") or None,
            ),
            config=segmentation_config,
        )
        chunks = _plan_chunks(
            parent=parent,
            private_text=private_text,
            blocks=list(segmentation.blocks),
            target_proxy_tokens=target_proxy_tokens,
            chunk_index_start=chunk_index_global,
            segmentation_hash=segmentation.receipt.segmentation_policy_hash,
        )
        if not chunks:
            raise ValueError("rejected parent produced zero chunks")
        candidates.extend(chunk["candidate"] for chunk in chunks)
        private_chunks.extend(chunk["private"] for chunk in chunks)
        chunk_index_global += len(chunks)
        parent_reports.append(
            {
                "parent_candidate_id": parent_id,
                "document_status": segmentation.receipt.document_status,
                "natural_block_count": len(segmentation.blocks),
                "chunk_count": len(chunks),
                "warning_code_counts": dict(sorted(segmentation.receipt.warning_code_counts.items())),
                "raw_text_serialized": False,
                "raw_url_serialized": False,
                "token_serialized": False,
            }
        )

    chunk_kind_counts = Counter()
    warning_counts = Counter()
    for candidate in candidates:
        bundle = candidate["window_bundle"]
        chunk_kind_counts.update(bundle["block_kind_counts"])
        warning_counts.update(bundle["warning_code_counts"])

    private_map = {
        "schema_version": CHUNK_PRIVATE_MAP_SCHEMA_VERSION,
        "artifact_kind": "chunk_relabel_private_map",
        "private_map_version": CHUNK_RELABEL_PRIVATE_MAP_VERSION,
        "policy_version": CHUNK_RELABEL_POLICY_VERSION,
        "source_candidate_manifest_hash": source_manifest["manifest_hash"],
        "source_schedule_hash": source_schedule.get("schedule_hash"),
        "segmentation_policy_hash": segmentation_policy_hash(segmentation_config),
        "target_proxy_tokens": target_proxy_tokens,
        "parent_document_skip_count": skip_parent_docs,
        "chunk_count": len(private_chunks),
        "chunks": private_chunks,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    manifest = {
        "schema_version": CHUNK_RELABEL_MANIFEST_SCHEMA_VERSION,
        "manifest_scope": CHUNK_RELABEL_MANIFEST_SCOPE,
        "policy_version": CHUNK_RELABEL_POLICY_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "source_candidate_manifest_hash": source_manifest["manifest_hash"],
        "source_final_labels_sha256": _hash(final_labels),
        "source_schedule_hash": source_schedule.get("schedule_hash"),
        "segmentation_policy_hash": segmentation_policy_hash(segmentation_config),
        "private_map_hash": _private_map_hash(private_map),
        "target_proxy_tokens": target_proxy_tokens,
        "parent_document_skip_count": skip_parent_docs,
        "parent_document_count": len(parent_candidates),
        "chunk_count": len(candidates),
        "label_unit": "window_bundle",
        "required_vote_count": 3,
        "ready_for_paid_chunk_labeling": len(candidates) > 0,
        "chunk_kind_counts": dict(sorted(chunk_kind_counts.items())),
        "warning_code_counts": dict(sorted(warning_counts.items())),
        "candidates": candidates,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    manifest["manifest_hash"] = _manifest_hash(manifest)
    private_map["manifest_hash"] = manifest["manifest_hash"]
    private_map["private_map_hash"] = manifest["private_map_hash"]

    validate_chunk_relabel_manifest(manifest)
    validate_chunk_private_map(private_map, manifest=manifest)
    report = _build_manifest_report(
        manifest=manifest,
        private_map=private_map,
        parent_reports=parent_reports,
        elapsed_seconds=time.monotonic() - started,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "chunk_relabel_manifest.json", manifest)
    _write_json_private(output / "chunk_relabel_private_map.json", private_map)
    _write_json_atomic(output / "chunk_relabel_build_report.json", report)
    return ChunkRelabelBuildArtifacts(
        manifest=manifest,
        private_map=private_map,
        report=report,
        private_parent_rows=parent_rows,
    )


def build_inherited_safe_parent_chunk_artifacts(
    *,
    source_manifest: dict[str, Any],
    final_labels: list[dict[str, Any]],
    source_schedule: dict[str, Any],
    row_provider: Any,
    output_candidates_jsonl: str | Path,
    output_private_map_json: str | Path,
    target_proxy_tokens: int = 1024,
    max_parent_docs: int | None = None,
    skip_parent_docs: int = 0,
    segmentation_config: ChunkedSegmentationConfig | None = None,
    output_report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Build v2 inherited-safe chunk candidates and a private span map.

    This is the safe-parent counterpart to rejected-parent chunk relabeling. It
    creates no teacher-label votes and no accepted corpus: it deterministically
    chunks final-35k safe parent documents so the v2 classifier trains on the
    full final-35k label universe.
    """

    started = time.monotonic()
    target_proxy_tokens = _require_positive_int(target_proxy_tokens, "target_proxy_tokens")
    _validate_source_schedule_no_raw(source_schedule)
    if max_parent_docs is not None:
        max_parent_docs = _require_positive_int(max_parent_docs, "max_parent_docs")
    skip_parent_docs = _require_nonnegative_int(skip_parent_docs, "skip_parent_docs")
    segmentation_config = segmentation_config or ChunkedSegmentationConfig()
    parent_candidates = _parent_candidates_for_final_label(
        source_manifest=source_manifest,
        final_labels=final_labels,
        final_label="safe",
        max_parent_docs=max_parent_docs,
        skip_parent_docs=skip_parent_docs,
    )
    parent_rows = _fetch_parent_rows(parent_candidates, schedule=source_schedule, row_provider=row_provider)

    candidates: list[dict[str, Any]] = []
    private_chunks: list[dict[str, Any]] = []
    parent_reports: list[dict[str, Any]] = []
    chunk_index_global = 0
    for parent in parent_candidates:
        parent_id = str(parent["candidate_id"])
        private_text = _private_parent_text(parent, parent_rows[parent_id])
        split = "validation" if str(parent.get("dataset_split") or "") == "validation" else "train"
        segmentation = segment_source_document(
            SourceDocumentRecord(
                source_document_id_hash=parent_id,
                split=split,
                raw_text=private_text,
                source_lineage_hash=str(parent.get("candidate_hash") or "") or None,
            ),
            config=segmentation_config,
        )
        chunks = _plan_chunks(
            parent=parent,
            private_text=private_text,
            blocks=list(segmentation.blocks),
            target_proxy_tokens=target_proxy_tokens,
            chunk_index_start=chunk_index_global,
            segmentation_hash=segmentation.receipt.segmentation_policy_hash,
            candidate_policy_version=INHERITED_SAFE_PARENT_POLICY_VERSION,
            teacher_label_use=INHERITED_SAFE_PARENT_POLICY_VERSION,
            source_chunk_manifest_hash=str(source_manifest["manifest_hash"]),
            parent_label_source=INHERITED_SAFE_PARENT_LABEL_SOURCE,
            include_classifier_v2_fields=True,
        )
        if not chunks:
            raise ValueError("safe parent produced zero chunks")
        candidates.extend(chunk["candidate"] for chunk in chunks)
        private_chunks.extend(chunk["private"] for chunk in chunks)
        chunk_index_global += len(chunks)
        parent_reports.append(
            {
                "parent_candidate_id": parent_id,
                "document_status": segmentation.receipt.document_status,
                "natural_block_count": len(segmentation.blocks),
                "chunk_count": len(chunks),
                "warning_code_counts": dict(sorted(segmentation.receipt.warning_code_counts.items())),
                "raw_text_serialized": False,
                "raw_url_serialized": False,
                "token_serialized": False,
            }
        )

    private_map = {
        "schema_version": CHUNK_PRIVATE_MAP_SCHEMA_VERSION,
        "artifact_kind": INHERITED_SAFE_PARENT_PRIVATE_MAP_KIND,
        "private_map_route": INHERITED_SAFE_PARENT_PRIVATE_MAP_ROUTE,
        "policy_version": INHERITED_SAFE_PARENT_POLICY_VERSION,
        "source_candidate_manifest_hash": source_manifest["manifest_hash"],
        "source_final_labels_sha256": _hash(final_labels),
        "source_schedule_hash": source_schedule.get("schedule_hash"),
        "segmentation_policy_hash": segmentation_policy_hash(segmentation_config),
        "target_proxy_tokens": target_proxy_tokens,
        "parent_document_skip_count": skip_parent_docs,
        "parent_document_count": len(parent_candidates),
        "chunk_count": len(private_chunks),
        "chunks": private_chunks,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    private_map["private_map_hash"] = _private_map_hash(private_map)
    validate_inherited_safe_parent_private_map(private_map, candidates=candidates)

    report = {
        "artifact_kind": "twc_inherited_safe_parent_chunk_build_report",
        "schema_version": 1,
        "status": "complete",
        "policy_version": INHERITED_SAFE_PARENT_POLICY_VERSION,
        "private_map_route": INHERITED_SAFE_PARENT_PRIVATE_MAP_ROUTE,
        "source_candidate_manifest_hash": source_manifest["manifest_hash"],
        "source_final_labels_sha256": _hash(final_labels),
        "source_schedule_hash": source_schedule.get("schedule_hash"),
        "segmentation_policy_hash": segmentation_policy_hash(segmentation_config),
        "target_proxy_tokens": target_proxy_tokens,
        "parent_document_skip_count": skip_parent_docs,
        "parent_document_count": len(parent_candidates),
        "candidate_count": len(candidates),
        "private_map_hash": private_map["private_map_hash"],
        "candidate_rows_hash": _hash(candidates),
        "parent_report_hash": _hash(parent_reports),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "paid_api_calls_performed": False,
        "gpu_rental_used": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    report["report_hash"] = _hash(
        {key: value for key, value in report.items() if key not in {"elapsed_seconds", "report_hash"}}
    )
    _reject_live_artifact_leakage(report)
    _write_label_rows(output_candidates_jsonl, candidates)
    _write_json_private(output_private_map_json, private_map)
    if output_report_json is not None:
        _write_json_atomic(output_report_json, report)
    return {
        "candidates": candidates,
        "private_map": private_map,
        "report": report,
    }


def validate_chunk_relabel_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("chunk relabel manifest must be an object")
    if manifest.get("schema_version") != CHUNK_RELABEL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported chunk relabel manifest schema")
    if manifest.get("manifest_scope") != CHUNK_RELABEL_MANIFEST_SCOPE:
        raise ValueError("unsupported chunk relabel manifest scope")
    if manifest.get("policy_version") != CHUNK_RELABEL_POLICY_VERSION:
        raise ValueError("unsupported chunk relabel policy")
    for field in _PUBLIC_FALSE_FLAGS:
        if manifest.get(field) is not False:
            raise ValueError(f"chunk relabel manifest must set {field}=false")
    _require_sha256(str(manifest.get("source_candidate_manifest_hash") or ""), "source_candidate_manifest_hash")
    _require_sha256(str(manifest.get("segmentation_policy_hash") or ""), "segmentation_policy_hash")
    _require_sha256(str(manifest.get("private_map_hash") or ""), "private_map_hash")
    if manifest.get("source_schedule_hash") is not None:
        _require_sha256(str(manifest["source_schedule_hash"]), "source_schedule_hash")
    target_proxy_tokens = _require_positive_int(manifest.get("target_proxy_tokens"), "target_proxy_tokens")
    _require_nonnegative_int(manifest.get("parent_document_skip_count"), "parent_document_skip_count")
    if manifest.get("label_unit") != "window_bundle":
        raise ValueError("chunk relabel manifest label_unit must be window_bundle")
    if int(manifest.get("required_vote_count") or 0) != 3:
        raise ValueError("chunk relabel manifest requires exactly three votes")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("chunk relabel manifest candidates must be a list")
    if _require_nonnegative_int(manifest.get("chunk_count"), "chunk_count") != len(candidates):
        raise ValueError("chunk relabel manifest chunk_count mismatch")
    if bool(manifest.get("ready_for_paid_chunk_labeling")) != bool(candidates):
        raise ValueError("chunk relabel readiness must match non-empty candidates")
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        _validate_chunk_candidate(candidate, index=index, target_proxy_tokens=target_proxy_tokens)
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in seen_ids:
            raise ValueError("chunk relabel manifest contains duplicate candidate_id")
        seen_ids.add(candidate_id)
    if str(manifest.get("manifest_hash") or "") != _manifest_hash(manifest):
        raise ValueError("chunk relabel manifest hash mismatch")
    _reject_live_artifact_leakage(manifest)


def validate_chunk_private_map(private_map: dict[str, Any], *, manifest: dict[str, Any]) -> None:
    validate_chunk_relabel_manifest(manifest)
    if not isinstance(private_map, dict):
        raise ValueError("chunk private map must be an object")
    if private_map.get("schema_version") != CHUNK_PRIVATE_MAP_SCHEMA_VERSION:
        raise ValueError("unsupported chunk private map schema")
    if private_map.get("private_map_version") != CHUNK_RELABEL_PRIVATE_MAP_VERSION:
        raise ValueError("unsupported chunk private map version")
    if private_map.get("policy_version") != CHUNK_RELABEL_POLICY_VERSION:
        raise ValueError("unsupported chunk private map policy")
    if private_map.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("private chunk map manifest hash mismatch")
    if private_map.get("parent_document_skip_count") != manifest.get("parent_document_skip_count"):
        raise ValueError("chunk private map parent skip count mismatch")
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if private_map.get(field) is not False:
            raise ValueError(f"chunk private map must set {field}=false")
    if str(private_map.get("private_map_hash") or "") != _private_map_hash(private_map):
        raise ValueError("private chunk map hash mismatch")
    if private_map["private_map_hash"] != manifest["private_map_hash"]:
        raise ValueError("private chunk map hash does not match manifest")
    chunks = private_map.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("chunk private map chunks must be a list")
    if _require_nonnegative_int(private_map.get("chunk_count"), "chunk_count") != len(chunks):
        raise ValueError("chunk private map chunk_count mismatch")
    by_candidate = {candidate["candidate_id"]: candidate for candidate in manifest["candidates"]}
    if set(by_candidate) != {str(chunk.get("candidate_id") or "") for chunk in chunks}:
        raise ValueError("chunk private map candidate set mismatch")
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("chunk private map entry must be an object")
        candidate = by_candidate[str(chunk.get("candidate_id") or "")]
        if chunk.get("parent_candidate_id") != candidate["parent_candidate_id"]:
            raise ValueError("chunk private map parent candidate mismatch")
        if chunk.get("chunk_text_sha256") != candidate["text_hash_sha256"]:
            raise ValueError("chunk private map text hash mismatch")
        _require_nonnegative_int(chunk.get("raw_char_start"), "raw_char_start")
        end = _require_positive_int(chunk.get("raw_char_end"), "raw_char_end")
        if end <= int(chunk["raw_char_start"]):
            raise ValueError("chunk private map raw span is empty")
        _require_nonnegative_int(chunk.get("block_start_index"), "block_start_index")
        block_end = _require_positive_int(chunk.get("block_end_index_exclusive"), "block_end_index_exclusive")
        if block_end <= int(chunk["block_start_index"]):
            raise ValueError("chunk private map block span is empty")
    _reject_live_artifact_leakage(private_map)


def validate_inherited_safe_parent_private_map(
    private_map: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
) -> None:
    if not isinstance(private_map, dict):
        raise ValueError("inherited safe private map must be an object")
    if private_map.get("schema_version") != CHUNK_PRIVATE_MAP_SCHEMA_VERSION:
        raise ValueError("unsupported inherited safe private map schema")
    if private_map.get("artifact_kind") != INHERITED_SAFE_PARENT_PRIVATE_MAP_KIND:
        raise ValueError("unsupported inherited safe private map kind")
    if private_map.get("private_map_route") != INHERITED_SAFE_PARENT_PRIVATE_MAP_ROUTE:
        raise ValueError("inherited safe private map route mismatch")
    if private_map.get("policy_version") != INHERITED_SAFE_PARENT_POLICY_VERSION:
        raise ValueError("unsupported inherited safe private map policy")
    for field in _PUBLIC_FALSE_FLAGS:
        if private_map.get(field) is not False:
            raise ValueError(f"inherited safe private map must set {field}=false")
    _require_sha256(str(private_map.get("source_candidate_manifest_hash") or ""), "source_candidate_manifest_hash")
    _require_sha256(str(private_map.get("source_final_labels_sha256") or ""), "source_final_labels_sha256")
    _require_sha256(str(private_map.get("segmentation_policy_hash") or ""), "segmentation_policy_hash")
    if private_map.get("source_schedule_hash") is not None:
        _require_sha256(str(private_map["source_schedule_hash"]), "source_schedule_hash")
    target_proxy_tokens = _require_positive_int(private_map.get("target_proxy_tokens"), "target_proxy_tokens")
    _require_nonnegative_int(private_map.get("parent_document_skip_count"), "parent_document_skip_count")
    _require_nonnegative_int(private_map.get("parent_document_count"), "parent_document_count")
    if str(private_map.get("private_map_hash") or "") != _private_map_hash(private_map):
        raise ValueError("inherited safe private map hash mismatch")
    chunks = private_map.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("inherited safe private map chunks must be a list")
    if _require_nonnegative_int(private_map.get("chunk_count"), "chunk_count") != len(chunks):
        raise ValueError("inherited safe private map chunk_count mismatch")
    if not isinstance(candidates, list):
        raise ValueError("inherited safe candidates must be a list")
    by_candidate: dict[str, dict[str, Any]] = {}
    parent_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        _validate_inherited_safe_parent_candidate(
            candidate,
            index=index,
            target_proxy_tokens=target_proxy_tokens,
            source_manifest_hash=str(private_map["source_candidate_manifest_hash"]),
        )
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in by_candidate:
            raise ValueError("duplicate inherited safe candidate_id")
        by_candidate[candidate_id] = candidate
        parent_ids.add(str(candidate["parent_candidate_id"]))
    if len(parent_ids) != int(private_map["parent_document_count"]):
        raise ValueError("inherited safe parent_document_count mismatch")
    chunk_ids = [str(chunk.get("candidate_id") or "") for chunk in chunks if isinstance(chunk, dict)]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("duplicate inherited safe private chunk candidate_id")
    if set(by_candidate) != set(chunk_ids):
        raise ValueError("inherited safe private map candidate set mismatch")
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("inherited safe private map entry must be an object")
        candidate = by_candidate[str(chunk.get("candidate_id") or "")]
        if chunk.get("parent_candidate_id") != candidate["parent_candidate_id"]:
            raise ValueError("inherited safe private map parent candidate mismatch")
        if chunk.get("chunk_text_sha256") != candidate["text_hash_sha256"]:
            raise ValueError("inherited safe private map chunk hash drift")
        if chunk.get("parent_text_hash_sha256") != candidate["parent_text_hash_sha256"]:
            raise ValueError("inherited safe private map parent hash drift")
        if str(chunk.get("parent_text_hash_algorithm") or "") != str(candidate.get("parent_text_hash_algorithm") or ""):
            raise ValueError("inherited safe private map parent hash algorithm drift")
        if str(chunk.get("source_file_id") or "") != str(candidate.get("source_file_id") or ""):
            raise ValueError("inherited safe private map source_file_id drift")
        if str(chunk.get("source_schedule_hash") or "") != str(candidate.get("source_schedule_hash") or ""):
            raise ValueError("inherited safe private map source_schedule_hash drift")
        if str(chunk.get("schedule_entry_hash") or "") != str(candidate.get("schedule_entry_hash") or ""):
            raise ValueError("inherited safe private map schedule_entry_hash drift")
        if int(chunk.get("row_group_index", -1)) != int(candidate.get("row_group_index", -2)):
            raise ValueError("inherited safe private map row_group_index drift")
        if int(chunk.get("row_group_local_row_index", -1)) != int(candidate.get("row_group_local_row_index", -2)):
            raise ValueError("inherited safe private map row_group_local_row_index drift")
        _require_nonnegative_int(chunk.get("raw_char_start"), "raw_char_start")
        end = _require_positive_int(chunk.get("raw_char_end"), "raw_char_end")
        if end <= int(chunk["raw_char_start"]):
            raise ValueError("inherited safe private map raw span is empty")
        _require_nonnegative_int(chunk.get("block_start_index"), "block_start_index")
        block_end = _require_positive_int(chunk.get("block_end_index_exclusive"), "block_end_index_exclusive")
        if block_end <= int(chunk["block_start_index"]):
            raise ValueError("inherited safe private map block span is empty")
        for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
            if chunk.get(field) is not False:
                raise ValueError(f"inherited safe private map chunk must set {field}=false")
    _reject_live_artifact_leakage(private_map)


def run_chunk_label_batch(
    *,
    manifest: dict[str, Any],
    private_map: dict[str, Any],
    source_schedule: dict[str, Any],
    row_provider: Any,
    vote_jsonl: str | Path,
    output_jsonl: str | Path,
    report_path: str | Path,
    repair_queue_path: str | Path,
    progress_path: str | Path,
    live: bool,
    max_chunks: int | None,
    current_balance_usd: float | None,
    budget_cap_usd: float,
    minimum_balance_reserve_usd: float = 0.0,
    target_concurrency: int = 64,
    private_refetch_concurrency: int = 64,
    tranche_chunks: int = 512,
    env: dict[str, str] | None = None,
    call: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    price_snapshot: PriceSnapshot | None = None,
    provider_config: DeepSeekLabelingConfig | None = None,
    avg_input_tokens: int = 1_400,
    avg_output_tokens: int = 512,
    stop_requested: Callable[[], bool] | None = None,
    private_parent_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    validate_chunk_private_map(private_map, manifest=manifest)
    _validate_source_schedule_no_raw(source_schedule)
    if max_chunks is None or int(max_chunks) < 1:
        raise ValueError("chunk label batch requires explicit --max-chunks")
    if int(max_chunks) > int(manifest["chunk_count"]):
        raise ValueError("chunk label max_chunks cannot exceed manifest chunk_count")
    if current_balance_usd is None:
        raise ValueError("chunk label batch requires current balance for available-balance policy")
    policy = ChunkRelabelPolicy(
        budget_cap_usd=budget_cap_usd,
        minimum_balance_reserve_usd=minimum_balance_reserve_usd,
        target_concurrency=target_concurrency,
        private_refetch_concurrency=private_refetch_concurrency,
        tranche_chunks=tranche_chunks,
    )
    if float(current_balance_usd) < policy.minimum_balance_reserve_usd:
        raise ValueError("chunk label current balance is below required reserve")
    provider_config = provider_config or DeepSeekLabelingConfig()
    _validate_provider_config(provider_config)
    if policy.target_concurrency > provider_config.concurrency_limit:
        raise ValueError("chunk relabel target_concurrency cannot exceed provider concurrency_limit")
    candidates = list(manifest["candidates"][: int(max_chunks)])
    price = price_snapshot or _price_from_official_snapshot(provider_config.model)
    provider_config_hash = _hash(_config_payload(provider_config))
    provider_payload = _public_config_payload(provider_config)
    stop_requested = stop_requested or (lambda: False)
    budget_check = estimate_label_cost_usd(
        label_count=len(candidates) * policy.required_vote_count,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=price,
        safety_multiplier=1.25,
        budget_cap_usd=policy.budget_cap_usd,
        reserve_usd=0.0,
    )
    per_vote_estimate = estimate_label_cost_usd(
        label_count=1,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=price,
        safety_multiplier=1.0,
        budget_cap_usd=policy.budget_cap_usd,
        reserve_usd=0.0,
    )["estimated_cost_usd"]
    per_vote_budget_estimate = estimate_label_cost_usd(
        label_count=1,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=price,
        safety_multiplier=1.25,
        budget_cap_usd=policy.budget_cap_usd,
        reserve_usd=0.0,
    )["estimated_cost_usd"]
    call = call or call_deepseek_live
    using_live_client = call is call_deepseek_live
    if live is True and using_live_client:
        prepare_deepseek_live_call(
            {"base_url": provider_config.base_url, "endpoint": provider_config.endpoint_path},
            live=True,
            max_docs=int(max_chunks),
            budget_cap_usd=policy.budget_cap_usd,
            env=env,
        )

    rows_for_report = _read_existing_chunk_vote_rows(
        vote_jsonl,
        manifest=manifest,
        provider_config=provider_config,
        candidates=candidates,
        policy=policy,
    )
    existing_keys = {str(row["idempotency_key"]) for row in rows_for_report}
    observed_cost = _row_cost_sum(rows_for_report, "actual_cost_usd")
    estimated_cost = _row_cost_sum(rows_for_report, "estimated_cost_usd")
    paid_calls_made = 0
    provider_vote_rows_recorded = 0
    transport_failures: list[dict[str, Any]] = []
    run_status = "complete" if int(max_chunks) == int(manifest["chunk_count"]) else "resumable_partial"
    blocked_reason: str | None = None
    if live is not True:
        run_status = "dry_run_no_paid_calls"
        blocked_reason = "explicit live flag not set"

    def budget_allows(next_call_count: int) -> bool:
        next_call_estimate = float(per_vote_budget_estimate) * next_call_count
        if observed_cost + next_call_estimate > float(policy.budget_cap_usd):
            return False
        return float(current_balance_usd) - next_call_estimate >= float(policy.minimum_balance_reserve_usd)

    def append_new_rows(new_rows: list[dict[str, Any]]) -> None:
        nonlocal observed_cost, estimated_cost, paid_calls_made, provider_vote_rows_recorded, rows_for_report
        for row in new_rows:
            key = str(row["idempotency_key"])
            if key in existing_keys:
                continue
            _append_label_row_known_unique(vote_jsonl, row)
            existing_keys.add(key)
            rows_for_report.append(row)
            provider_vote_rows_recorded += 1
            if live is True and using_live_client:
                paid_calls_made += 1
            observed_cost += float(row.get("actual_cost_usd") or 0.0)
            estimated_cost += float(row.get("estimated_cost_usd") or 0.0)

    if run_status in {"complete", "resumable_partial"}:
        for start_index in range(0, len(candidates), policy.tranche_chunks):
            if stop_requested():
                run_status = "stopped_by_operator"
                blocked_reason = "stop requested before scheduling next chunk tranche"
                break
            tranche = candidates[start_index : start_index + policy.tranche_chunks]
            retry_lookup = _parse_error_retry_lookup(rows_for_report)
            missing_plan = _chunk_missing_vote_plan(
                tranche,
                existing_rows=rows_for_report,
                wanted_vote_indexes=(0, 1, 2),
            )
            missing_count = sum(len(missing) for _candidate, missing in missing_plan)
            if missing_count and not budget_allows(missing_count):
                run_status = "resumable_partial_budget_exhausted"
                blocked_reason = "next planned chunk BoF call would exceed budget cap or available balance reserve"
                break
            request_plan = _chunk_vote_request_tasks(
                manifest=manifest,
                private_map=private_map,
                schedule=source_schedule,
                row_provider=row_provider,
                missing_plan=missing_plan,
                provider_config=provider_config,
                policy=policy,
                refetch_concurrency=policy.private_refetch_concurrency,
                private_parent_rows=private_parent_rows,
                retry_lookup=retry_lookup,
            )
            tasks = list(request_plan["tasks"])
            tranche_transport_failures = list(request_plan["transport_failures"])
            if tasks:
                result = _run_chunk_vote_tasks(
                    tasks,
                    live=live,
                    max_chunks=int(max_chunks),
                    budget_cap_usd=policy.budget_cap_usd,
                    env=env,
                    call=call,
                    provider_config=provider_config,
                    price=price,
                    per_vote_estimate=float(per_vote_estimate),
                    concurrency=policy.target_concurrency,
                    on_row=append_new_rows,
                    stop_requested=stop_requested,
                )
                tranche_transport_failures.extend(result["transport_failures"])
                if result.get("stopped"):
                    run_status = "stopped_by_operator"
                    blocked_reason = "stop requested after preserving completed chunk votes"
                    break
            if tranche_transport_failures:
                transport_failures.extend(tranche_transport_failures)
                if any(item.get("http_status_code") == 402 for item in tranche_transport_failures):
                    run_status = "resumable_partial_account_exhausted"
                    blocked_reason = "provider reported insufficient balance; partial chunk labels preserved"
                else:
                    run_status = "resumable_partial_transport_failures"
                    blocked_reason = "chunk label transport or private refetch failures queued after retry exhaustion"
                break
            if run_status == "stopped_by_operator":
                break

    grouped_rows = _rows_by_candidate(rows_for_report)
    grouped_transport: dict[str, list[dict[str, Any]]] = {}
    for item in transport_failures:
        grouped_transport.setdefault(str(item.get("candidate_id") or ""), []).append(item)
    aggregations = [
        _build_chunk_label_aggregation(
            candidate,
            grouped_rows.get(str(candidate["candidate_id"]), []),
            transport_failures=grouped_transport.get(str(candidate["candidate_id"]), []),
            manifest=manifest,
            policy=policy,
            provider_config_hash=provider_config_hash,
        )
        for candidate in candidates
    ]
    label_rows = [
        _chunk_label_row_from_aggregation(aggregation, candidate)
        for aggregation, candidate in zip(aggregations, candidates, strict=False)
        if aggregation.get("chunk_label_row_ready") is True
    ]
    if label_rows or live is True:
        _write_label_rows(output_jsonl, label_rows)
    parse_error_rows = [row for row in rows_for_report if row.get("parse_status") == "parse_error"]
    repair_queue = _write_chunk_repair_queue(
        repair_queue_path,
        parse_error_rows=parse_error_rows,
        transport_failures=transport_failures,
        aggregations=aggregations,
    )
    unfinished = [row for row in aggregations if row.get("chunk_label_row_ready") is not True]
    if run_status == "complete" and unfinished:
        run_status = "resumable_partial"
        blocked_reason = "some chunk label candidates require repair or rerun before completion"
    progress = _write_chunk_progress(
        progress_path,
        manifest=manifest,
        policy=policy,
        provider_config_hash=provider_config_hash,
        rows=rows_for_report,
        aggregations=aggregations,
        observed_cost=observed_cost,
        estimated_cost=estimated_cost,
        run_status=run_status,
        blocked_reason=blocked_reason,
        max_chunks=int(max_chunks),
        elapsed_seconds=time.monotonic() - started,
    )
    report = {
        "chunk_label_batch_report_schema_version": CHUNK_RELABEL_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "run_status": run_status,
        "blocked_reason": blocked_reason,
        "source_chunk_manifest_hash": manifest["manifest_hash"],
        "source_candidate_manifest_hash": manifest["source_candidate_manifest_hash"],
        "chunk_count": int(manifest["chunk_count"]),
        "max_chunks": int(max_chunks),
        "chunk_relabel_policy_version": policy.policy_version,
        "ready_for_live_chunk_labeling": manifest["ready_for_paid_chunk_labeling"],
        "available_balance_policy": "proceed_with_available_balance_preserve_resumable_partials_v1",
        "votes_recorded": len(rows_for_report),
        "provider_vote_rows_recorded": provider_vote_rows_recorded,
        "paid_api_calls_made": paid_calls_made,
        "paid_api_calls_performed": bool(paid_calls_made > 0),
        "chunk_label_rows_ready": len(label_rows),
        "unfinished_chunk_count": len(unfinished),
        "progress_telemetry": {
            "phase": progress["phase"],
            "completed_chunk_count": progress["completed_chunk_count"],
            "unfinished_chunk_count": progress["unfinished_chunk_count"],
            "final_safe_count": progress["final_safe_count"],
            "final_reject_count": progress["final_reject_count"],
            "active_concurrency": progress["active_concurrency"],
            "private_refetch_concurrency": progress["private_refetch_concurrency"],
            "elapsed_seconds": progress["elapsed_seconds"],
            "vote_throughput_per_second": progress["vote_throughput_per_second"],
            "budget_cap_used_fraction": progress["budget_cap_used_fraction"],
            "stop_reason": progress["stop_reason"],
        },
        "transport_failure_count": len(transport_failures),
        "per_vote_parse_status_counts": _status_counts(rows_for_report, "parse_status"),
        "finish_reason_counts": _status_counts(rows_for_report, "finish_reason"),
        "semantic_retry_counts": {},
        "main_batch_semantic_retry_performed": False,
        "provider_config": provider_payload,
        "provider_config_hash": provider_config_hash,
        "price_snapshot": asdict(price),
        "budget_cap_usd": policy.budget_cap_usd,
        "minimum_balance_reserve_usd": policy.minimum_balance_reserve_usd,
        "current_balance_usd": float(current_balance_usd),
        "budget_check": budget_check,
        "total_estimated_cost_usd": estimated_cost,
        "total_actual_cost_usd": observed_cost,
        "target_concurrency": policy.target_concurrency,
        "private_refetch_concurrency": policy.private_refetch_concurrency,
        "tranche_chunks": policy.tranche_chunks,
        "latency_summary": _latency_summary_from_rows(rows_for_report),
        "token_usage_summary": _token_usage_summary(rows_for_report),
        "chunk_label_aggregations": aggregations,
        "repair_queue_summary": {
            "queued_count": repair_queue["queued_count"],
            "parse_error_vote_count": repair_queue["parse_error_vote_count"],
            "transport_failure_count": repair_queue["transport_failure_count"],
            "by_finish_reason": repair_queue["by_finish_reason"],
        },
        "repair_queue_hash": repair_queue["repair_queue_hash"],
        "progress_hash": progress["chunk_label_progress_hash"],
        "vote_jsonl_path_sha256": _hash(str(vote_jsonl)),
        "output_jsonl_path_sha256": _hash(str(output_jsonl)),
        "one_chunk_per_request": True,
        "transport_http_retries_allowed": True,
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "commits_or_pushes_performed": False,
    }
    report["chunk_label_batch_report_hash"] = _hash({key: value for key, value in report.items() if key != "generated_at_utc"})
    _reject_live_artifact_leakage(report)
    _write_json_atomic(report_path, report)
    return report


def _rejected_parent_candidates(
    *,
    source_manifest: dict[str, Any],
    final_labels: list[dict[str, Any]],
    max_parent_docs: int | None,
    skip_parent_docs: int,
) -> list[dict[str, Any]]:
    return _parent_candidates_for_final_label(
        source_manifest=source_manifest,
        final_labels=final_labels,
        final_label="reject",
        max_parent_docs=max_parent_docs,
        skip_parent_docs=skip_parent_docs,
    )


def _parent_candidates_for_final_label(
    *,
    source_manifest: dict[str, Any],
    final_labels: list[dict[str, Any]],
    final_label: str,
    max_parent_docs: int | None,
    skip_parent_docs: int,
) -> list[dict[str, Any]]:
    if final_label not in {"safe", "reject"}:
        raise ValueError("final_label selector must be safe or reject")
    if source_manifest.get("manifest_scope") != "final_35k_teacher_label_freeze_candidates":
        raise ValueError("source manifest must be the final 35k teacher candidate manifest")
    source_hash = str(source_manifest.get("manifest_hash") or "")
    _require_sha256(source_hash, "source manifest hash")
    candidates = source_manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("source manifest candidates must be a non-empty list")
    seen_source_candidate_ids: set[str] = set()
    source_candidates_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("source manifest candidate must be an object")
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("source manifest candidate is missing candidate_id")
        if candidate_id in seen_source_candidate_ids:
            raise ValueError("duplicate source candidate_id in final manifest")
        seen_source_candidate_ids.add(candidate_id)
        source_candidates_by_id[candidate_id] = candidate
    labels_by_id: dict[str, dict[str, Any]] = {}
    for row in final_labels:
        if not isinstance(row, dict):
            raise ValueError("final label rows must be objects")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("final label row is missing candidate_id")
        if candidate_id in labels_by_id:
            raise ValueError("duplicate final label row for candidate")
        if str(row.get("source_candidate_manifest_hash") or "") != source_hash:
            raise ValueError("final label source manifest hash mismatch")
        source_candidate = source_candidates_by_id.get(candidate_id)
        if source_candidate is None:
            raise ValueError("final label row references a candidate outside the source manifest")
        if str(row.get("candidate_hash") or "") != str(source_candidate.get("candidate_hash") or ""):
            raise ValueError("final label candidate hash mismatch")
        if str(row.get("final_label_policy_version") or "") != FINAL_LABEL_POLICY_VERSION:
            raise ValueError("final label policy mismatch")
        if row.get("label_unit") != source_candidate.get("label_unit", "document"):
            raise ValueError("final label row label_unit drift")
        if int(row.get("required_vote_count") or 0) != 3:
            raise ValueError("final label row must require exactly three votes")
        parsed_votes = int(row.get("parsed_vote_count") or 0)
        parse_error_votes = int(row.get("parse_error_vote_count") or 0)
        if parsed_votes + parse_error_votes != 3:
            raise ValueError("final label row vote accounting must total three")
        if int(row.get("transport_failure_count") or 0) != 0:
            raise ValueError("final label row must not have transport failures")
        for field in _PUBLIC_FALSE_FLAGS:
            if row.get(field) is not False:
                raise ValueError(f"final label row must set {field}=false")
        if row.get("parse_status") != "final_label":
            raise ValueError("final label row must have parse_status=final_label")
        if row.get("final_label") not in {"safe", "reject"}:
            raise ValueError("final label row must be safe or reject")
        labels_by_id[candidate_id] = row
    candidate_ids = [str(candidate.get("candidate_id") or "") for candidate in candidates]
    missing_labels = [candidate_id for candidate_id in candidate_ids if candidate_id not in labels_by_id]
    if missing_labels:
        raise ValueError("final labels do not cover every source manifest candidate")
    selected = [
        candidate
        for candidate in candidates
        if labels_by_id[str(candidate["candidate_id"])]["final_label"] == final_label
    ]
    selected = selected[skip_parent_docs:]
    if max_parent_docs is not None:
        selected = selected[:max_parent_docs]
    return selected


def _fetch_parent_rows(
    parent_candidates: list[dict[str, Any]],
    *,
    schedule: dict[str, Any],
    row_provider: Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for candidate in parent_candidates:
        entry = _schedule_entry_for_candidate(candidate, schedule)
        key = str(entry.get("entry_hash") or _hash(entry))
        grouped.setdefault(key, (entry, []))[1].append(candidate)
    rows_by_parent: dict[str, dict[str, Any]] = {}
    if hasattr(row_provider, "fetch_row_groups"):
        entry_offsets = [
            (
                entry,
                sorted({int(candidate["row_group_local_row_index"]) for candidate in group_candidates}),
            )
            for entry, group_candidates in grouped.values()
        ]
        fetched_groups = row_provider.fetch_row_groups(entry_offsets)
        if not isinstance(fetched_groups, dict):
            raise ValueError("row provider must return a row-group mapping")
        for entry, group_candidates in grouped.values():
            key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
            fetched = fetched_groups.get(key)
            if not isinstance(fetched, dict):
                raise ValueError("row provider did not return a requested row group")
            for candidate in group_candidates:
                offset = int(candidate["row_group_local_row_index"])
                row = fetched.get(offset)
                if isinstance(row, BaseException):
                    raise ValueError(f"private refetch failed for parent row: {type(row).__name__}")
                if not isinstance(row, dict):
                    raise ValueError("private refetch returned a non-object parent row")
                rows_by_parent[str(candidate["candidate_id"])] = row
        return rows_by_parent
    for entry, group_candidates in grouped.values():
        offsets = sorted({int(candidate["row_group_local_row_index"]) for candidate in group_candidates})
        fetched = row_provider.fetch_rows(entry, offsets)
        if not isinstance(fetched, dict):
            raise ValueError("row provider must return an offset->row mapping")
        for candidate in group_candidates:
            offset = int(candidate["row_group_local_row_index"])
            row = fetched.get(offset)
            if isinstance(row, BaseException):
                raise ValueError(f"private refetch failed for parent row: {type(row).__name__}")
            if not isinstance(row, dict):
                raise ValueError("private refetch returned a non-object parent row")
            rows_by_parent[str(candidate["candidate_id"])] = row
    return rows_by_parent


def _private_parent_text(parent: dict[str, Any], row: dict[str, Any]) -> str:
    private_text = row.get("text")
    if not isinstance(private_text, str) or not private_text:
        raise ValueError("private refetch returned missing or empty parent text")
    computed_hash, _algorithm, had_expected_hash = _private_refetch_text_hash(parent, private_text)
    expected_hash = parent.get("text_hash") or parent.get("text_hash_sha256")
    if had_expected_hash and expected_hash != computed_hash:
        raise ValueError("private refetch parent text hash drift")
    return private_text


def _plan_chunks(
    *,
    parent: dict[str, Any],
    private_text: str,
    blocks: list[Any],
    target_proxy_tokens: int,
    chunk_index_start: int,
    segmentation_hash: str,
    candidate_policy_version: str = CHUNK_RELABEL_POLICY_VERSION,
    teacher_label_use: str = CHUNK_RELABEL_POLICY_VERSION,
    source_chunk_manifest_hash: str | None = None,
    parent_label_source: str | None = None,
    include_classifier_v2_fields: bool = False,
) -> list[dict[str, dict[str, Any]]]:
    if not blocks:
        return []
    planned: list[list[Any]] = []
    current: list[Any] = []
    current_tokens = 0
    for block in blocks:
        block_tokens = int(block.proxy_token_count)
        if current and current_tokens + block_tokens > target_proxy_tokens:
            planned.append(current)
            current = []
            current_tokens = 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        planned.append(current)

    output: list[dict[str, dict[str, Any]]] = []
    parent_id = str(parent["candidate_id"])
    parent_hash = str(parent.get("candidate_hash") or "")
    parent_text_hash = str(parent.get("text_hash") or parent.get("text_hash_sha256") or "")
    parent_text_hash_algorithm = str(parent.get("text_hash_algorithm") or TEXT_HASH_ALGORITHM_UTF8_SHA256_V1)
    if not parent_text_hash:
        parent_text_hash = _sha256_text(private_text)
        parent_text_hash_algorithm = TEXT_HASH_ALGORITHM_UTF8_SHA256_V1
    for local_index, chunk_blocks in enumerate(planned):
        first = chunk_blocks[0]
        last = chunk_blocks[-1]
        raw_start = int(first.owned_char_start)
        raw_end = int(last.owned_char_end)
        chunk_text = private_text[raw_start:raw_end]
        if not chunk_text:
            raise ValueError("planned chunk produced empty private text")
        chunk_hash = _sha256_text(chunk_text)
        normalized_hash = _sha256_text(normalize_text(chunk_text).canonical_text)
        proxy_count = sum(int(block.proxy_token_count) for block in chunk_blocks)
        block_kind_counts = Counter(str(block.block_kind) for block in chunk_blocks)
        warning_counts = Counter(warning for block in chunk_blocks for warning in block.warning_codes)
        if any(int(block.proxy_token_count) > target_proxy_tokens for block in chunk_blocks):
            warning_counts["oversized_natural_block_in_chunk"] += 1
        chunk_index = chunk_index_start + len(output)
        local_chunk_id_payload = {
            "policy_version": candidate_policy_version,
            "parent_candidate_id": parent_id,
            "parent_candidate_hash": parent_hash,
            "local_chunk_index": local_index,
            "global_chunk_index": chunk_index,
            "block_start_index": int(first.block_index),
            "block_end_index_exclusive": int(last.block_index) + 1,
            "chunk_text_sha256": chunk_hash,
            "segmentation_policy_hash": segmentation_hash,
        }
        candidate_id = _hash(local_chunk_id_payload)
        candidate = {
            "candidate_schema_version": 1,
            "candidate_id": candidate_id,
            "parent_candidate_id": parent_id,
            "parent_candidate_hash": parent_hash,
            "parent_text_hash_sha256": parent_text_hash,
            "parent_text_hash_algorithm": parent_text_hash_algorithm,
            "global_chunk_index": chunk_index,
            "parent_chunk_index": local_index,
            "parent_chunk_count": len(planned),
            "label_unit": "window_bundle",
            "teacher_label_use": teacher_label_use,
            "source_doc_id": parent_id,
            "dataset_repo_id": parent.get("dataset_repo_id"),
            "dataset_config": parent.get("dataset_config"),
            "dataset_split": parent.get("dataset_split"),
            "resolved_repo_sha": parent.get("resolved_repo_sha"),
            "source_metadata_hash": parent.get("source_metadata_hash"),
            "source_schedule_hash": parent.get("source_schedule_hash"),
            "schedule_entry_hash": parent.get("schedule_entry_hash"),
            "source_file_id": parent.get("source_file_id"),
            "row_group_index": parent.get("row_group_index"),
            "row_group_local_row_index": parent.get("row_group_local_row_index"),
            "global_schedule_index": parent.get("global_schedule_index"),
            "schedule_pass_index": parent.get("schedule_pass_index"),
            "physical_source_row_key_hash": parent.get("physical_source_row_key_hash"),
            "text_hash_sha256": chunk_hash,
            "text_hash_algorithm": TEXT_HASH_ALGORITHM_UTF8_SHA256_V1,
            "normalized_text_hash_sha256": normalized_hash,
            "normalized_text_hash_algorithm": "normalize_text_sha256_v1",
            "window_bundle": {
                "bundle_schema_version": 1,
                "bundle_kind": "natural_block_chunk",
                "parent_candidate_id": parent_id,
                "chunk_index": local_index,
                "chunk_count": len(planned),
                "target_proxy_tokens": target_proxy_tokens,
                "proxy_token_count": proxy_count,
                "block_start_index": int(first.block_index),
                "block_end_index_exclusive": int(last.block_index) + 1,
                "block_count": len(chunk_blocks),
                "block_kind_counts": dict(sorted(block_kind_counts.items())),
                "warning_code_counts": dict(sorted(warning_counts.items())),
                "segmentation_policy_hash": segmentation_hash,
                "chunk_text_sha256": chunk_hash,
            },
            "labels_are_corpus_approval": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
            "accepted_manifest": False,
            "training_shards_created": False,
            "classifier_training_performed": False,
        }
        if teacher_label_use == CHUNK_RELABEL_POLICY_VERSION:
            candidate["chunk_relabel_policy_version"] = CHUNK_RELABEL_POLICY_VERSION
        else:
            candidate["classifier_candidate_policy_version"] = candidate_policy_version
        if source_chunk_manifest_hash is not None:
            candidate["source_chunk_manifest_hash"] = source_chunk_manifest_hash
        if parent_label_source is not None:
            candidate["parent_label_source"] = parent_label_source
        if include_classifier_v2_fields:
            candidate["source_bucket_hash"] = _source_bucket_hash(parent)
            candidate["token_count_bucket"] = _proxy_token_count_bucket(proxy_count)
        candidate["candidate_hash"] = _chunk_candidate_hash(candidate)
        private = {
            "candidate_id": candidate_id,
            "parent_candidate_id": parent_id,
            "chunk_text_sha256": chunk_hash,
            "parent_text_hash_sha256": parent_text_hash,
            "parent_text_hash_algorithm": parent_text_hash_algorithm,
            "raw_char_start": raw_start,
            "raw_char_end": raw_end,
            "block_start_index": int(first.block_index),
            "block_end_index_exclusive": int(last.block_index) + 1,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        if include_classifier_v2_fields:
            private.update(
                {
                    "source_file_id": parent.get("source_file_id"),
                    "source_schedule_hash": parent.get("source_schedule_hash"),
                    "schedule_entry_hash": parent.get("schedule_entry_hash"),
                    "row_group_index": parent.get("row_group_index"),
                    "row_group_local_row_index": parent.get("row_group_local_row_index"),
                }
            )
        _reject_live_artifact_leakage(candidate)
        _reject_live_artifact_leakage(private)
        output.append({"candidate": candidate, "private": private})
    return output


def _chunk_missing_vote_plan(
    candidates: list[dict[str, Any]],
    *,
    existing_rows: list[dict[str, Any]],
    wanted_vote_indexes: tuple[int, ...],
) -> list[tuple[dict[str, Any], tuple[int, ...]]]:
    grouped = _rows_by_candidate(existing_rows)
    plan: list[tuple[dict[str, Any], tuple[int, ...]]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_rows = grouped.get(candidate_id, [])
        parsed_indexes = {
            int(row.get("vote_index", row.get("bof_call_index", -1)))
            for row in candidate_rows
            if row.get("parse_status") == "parsed"
        }
        missing = tuple(index for index in wanted_vote_indexes if index not in parsed_indexes)
        if missing:
            plan.append((candidate, missing))
    return plan


def _parse_error_retry_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    parsed = {
        (str(row.get("candidate_id") or ""), int(row.get("vote_index", -1)))
        for row in rows
        if row.get("parse_status") == "parsed"
    }
    lookup: dict[tuple[str, int], str] = {}
    for row in rows:
        if row.get("parse_status") != "parse_error":
            continue
        key = (str(row.get("candidate_id") or ""), int(row.get("vote_index", -1)))
        if key in parsed:
            continue
        idempotency_key = str(row.get("idempotency_key") or "")
        if idempotency_key:
            lookup[key] = idempotency_key
    return lookup


def _chunk_vote_request_tasks(
    *,
    manifest: dict[str, Any],
    private_map: dict[str, Any],
    schedule: dict[str, Any],
    row_provider: Any,
    missing_plan: list[tuple[dict[str, Any], tuple[int, ...]]],
    provider_config: DeepSeekLabelingConfig,
    policy: ChunkRelabelPolicy,
    refetch_concurrency: int,
    private_parent_rows: dict[str, dict[str, Any]] | None = None,
    retry_lookup: dict[tuple[str, int], str] | None = None,
) -> dict[str, Any]:
    private_by_candidate = {str(chunk["candidate_id"]): chunk for chunk in private_map["chunks"]}
    private_parent_rows = private_parent_rows or {}
    retry_lookup = retry_lookup or {}

    def vote_request(candidate: dict[str, Any], base_request: dict[str, Any], vote_index: int) -> dict[str, Any]:
        failed_key = retry_lookup.get((str(candidate["candidate_id"]), int(vote_index)))
        request = _with_chunk_vote_identity(
            base_request,
            vote_index=vote_index,
            policy=policy,
            retry_of_idempotency_key=failed_key,
            retry_reason="parse_error_vote_retry" if failed_key else None,
        )
        request["refetch_audit_hash"] = base_request["refetch_audit_hash"]
        return request

    def build_base_request(candidate: dict[str, Any], missing: tuple[int, ...], row: Any) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
        chunk_text, refetch_audit = _chunk_text_and_audit(candidate, private_by_candidate[str(candidate["candidate_id"])], row)
        base_request = build_deepseek_request(
            candidate,
            raw_text=chunk_text,
            config=provider_config,
            manifest_hash=manifest["manifest_hash"],
        )
        base_request["candidate_hash"] = candidate.get("candidate_hash")
        base_request["parent_candidate_id"] = candidate.get("parent_candidate_id")
        base_request["label_unit"] = candidate.get("label_unit")
        base_request["global_chunk_index"] = candidate.get("global_chunk_index")
        base_request["parent_chunk_index"] = candidate.get("parent_chunk_index")
        base_request["parent_chunk_count"] = candidate.get("parent_chunk_count")
        base_request["refetch_audit_hash"] = _hash(refetch_audit)
        return candidate, missing, base_request

    grouped: dict[str, tuple[dict[str, Any], list[tuple[int, dict[str, Any], tuple[int, ...]]]]] = {}
    for index, (candidate, missing) in enumerate(missing_plan):
        entry = _schedule_entry_for_candidate(candidate, schedule)
        group_key = str(entry.get("entry_hash") or _hash(entry))
        grouped.setdefault(group_key, (entry, []))[1].append((index, candidate, missing))

    def build_group_requests(entry: dict[str, Any], indexed_candidates: list[tuple[int, dict[str, Any], tuple[int, ...]]]) -> list[tuple[int, dict[str, Any], tuple[int, ...], dict[str, Any]]]:
        offsets = sorted({int(candidate["row_group_local_row_index"]) for _index, candidate, _missing in indexed_candidates})
        rows = row_provider.fetch_rows(entry, offsets)
        built = []
        for index, candidate, missing in indexed_candidates:
            offset = int(candidate["row_group_local_row_index"])
            built_candidate, built_missing, base_request = build_base_request(candidate, missing, rows.get(offset))
            built.append((index, built_candidate, built_missing, base_request))
        return built

    fetched: list[tuple[dict[str, Any], tuple[int, ...], dict[str, Any]] | None] = [None] * len(missing_plan)
    failed_indexes: set[int] = set()
    failures: list[dict[str, Any]] = []

    def record_failure(indexed_candidates: list[tuple[int, dict[str, Any], tuple[int, ...]]], exc: BaseException) -> None:
        for index, candidate, missing in indexed_candidates:
            failed_indexes.add(index)
            failures.append(_chunk_refetch_failure_item(candidate, missing=missing, exc=exc, manifest=manifest, provider_config=provider_config, policy=policy))

    for index, (candidate, missing) in enumerate(missing_plan):
        parent_row = private_parent_rows.get(str(candidate.get("parent_candidate_id") or ""))
        if parent_row is None:
            continue
        try:
            built_candidate, built_missing, base_request = build_base_request(candidate, missing, parent_row)
        except Exception as exc:
            record_failure([(index, candidate, missing)], exc)
            continue
        fetched[index] = (built_candidate, built_missing, base_request)

    if len(fetched) == len(missing_plan) and all(item is not None for item in fetched):
        tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for fetched_item in fetched:
            if fetched_item is None:
                raise ValueError("chunk label private refetch plan produced an incomplete task set")
            candidate, missing, base_request = fetched_item
            for vote_index in missing:
                tasks.append((candidate, vote_request(candidate, base_request, vote_index)))
        return {"tasks": tasks, "transport_failures": failures}

    missing_plan = [
        (candidate, missing)
        for index, (candidate, missing) in enumerate(missing_plan)
        if fetched[index] is None and index not in failed_indexes
    ]
    original_indexes = [
        index
        for index, fetched_item in enumerate(fetched)
        if fetched_item is None and index not in failed_indexes
    ]
    group_items = list(grouped.values())
    grouped = {}
    for local_index, (candidate, missing) in enumerate(missing_plan):
        entry = _schedule_entry_for_candidate(candidate, schedule)
        group_key = str(entry.get("entry_hash") or _hash(entry))
        grouped.setdefault(group_key, (entry, []))[1].append((original_indexes[local_index], candidate, missing))
    group_items = list(grouped.values())
    if group_items and hasattr(row_provider, "fetch_row_groups"):
        try:
            entry_offsets = [
                (
                    entry,
                    sorted(
                        {
                            int(candidate["row_group_local_row_index"])
                            for _index, candidate, _missing in indexed_candidates
                        }
                    ),
                )
                for entry, indexed_candidates in group_items
            ]
            fetched_groups = row_provider.fetch_row_groups(entry_offsets)
            if not isinstance(fetched_groups, dict):
                raise ValueError("row provider must return a row-group mapping")
            for group_index, (entry, indexed_candidates) in enumerate(group_items):
                key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
                rows = fetched_groups.get(key)
                if not isinstance(rows, dict):
                    raise ValueError("row provider did not return a requested row group")
                try:
                    built_requests = []
                    for index, candidate, missing in indexed_candidates:
                        offset = int(candidate["row_group_local_row_index"])
                        built_candidate, built_missing, base_request = build_base_request(candidate, missing, rows.get(offset))
                        built_requests.append((index, built_candidate, built_missing, base_request))
                except Exception as exc:
                    record_failure(indexed_candidates, exc)
                    continue
                for index, candidate, missing, base_request in built_requests:
                    fetched[index] = (candidate, missing, base_request)
        except Exception as exc:
            for _entry, indexed_candidates in group_items:
                record_failure(indexed_candidates, exc)
    elif group_items and refetch_concurrency > 1:
        with ThreadPoolExecutor(max_workers=max(1, int(refetch_concurrency))) as executor:
            future_map = {
                executor.submit(build_group_requests, entry, indexed_candidates): group_index
                for group_index, (entry, indexed_candidates) in enumerate(group_items)
            }
            for future in as_completed(future_map):
                group_index = future_map[future]
                _entry, indexed_candidates = group_items[group_index]
                try:
                    built_requests = future.result()
                except Exception as exc:
                    record_failure(indexed_candidates, exc)
                    continue
                for index, candidate, missing, base_request in built_requests:
                    fetched[index] = (candidate, missing, base_request)
    else:
        for entry, indexed_candidates in group_items:
            try:
                built_requests = build_group_requests(entry, indexed_candidates)
            except Exception as exc:
                record_failure(indexed_candidates, exc)
                continue
            for index, candidate, missing, base_request in built_requests:
                fetched[index] = (candidate, missing, base_request)

    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, fetched_item in enumerate(fetched):
        if fetched_item is None:
            if index in failed_indexes:
                continue
            raise ValueError("chunk label private refetch plan produced an incomplete task set")
        candidate, missing, base_request = fetched_item
        for vote_index in missing:
            tasks.append((candidate, vote_request(candidate, base_request, vote_index)))
    return {"tasks": tasks, "transport_failures": failures}


def _chunk_text_and_audit(candidate: dict[str, Any], private_chunk: dict[str, Any], row: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(row, BaseException):
        raise ValueError(f"private refetch failed for scheduled row: {type(row).__name__}")
    if not isinstance(row, dict):
        raise ValueError("private refetch returned a non-object row")
    private_text = row.get("text")
    if not isinstance(private_text, str) or not private_text:
        raise ValueError("private refetch returned missing or empty parent text")
    parent_probe = {
        "text_hash_sha256": private_chunk["parent_text_hash_sha256"],
        "text_hash_algorithm": private_chunk["parent_text_hash_algorithm"],
    }
    computed_parent_hash, _algorithm, had_expected_hash = _private_refetch_text_hash(parent_probe, private_text)
    if had_expected_hash and computed_parent_hash != private_chunk["parent_text_hash_sha256"]:
        raise ValueError("private refetch parent text hash drift")
    start = int(private_chunk["raw_char_start"])
    end = int(private_chunk["raw_char_end"])
    if start < 0 or end <= start or end > len(private_text):
        raise ValueError("private chunk span is outside parent text")
    chunk_text = private_text[start:end]
    if _sha256_text(chunk_text) != private_chunk["chunk_text_sha256"]:
        raise ValueError("private chunk text hash drift")
    audit = {
        "candidate_id": candidate["candidate_id"],
        "parent_candidate_id": candidate["parent_candidate_id"],
        "source_identity_hash": _hash(
            {
                "parent_candidate_id": candidate["parent_candidate_id"],
                "candidate_id": candidate["candidate_id"],
                "source_file_id": candidate.get("source_file_id"),
                "row_group_index": candidate.get("row_group_index"),
                "row_group_local_row_index": candidate.get("row_group_local_row_index"),
            }
        ),
        "source_schedule_hash": candidate.get("source_schedule_hash"),
        "schedule_entry_hash": candidate.get("schedule_entry_hash"),
        "source_file_id": candidate.get("source_file_id"),
        "row_group_index": int(candidate["row_group_index"]),
        "row_group_local_row_index": int(candidate["row_group_local_row_index"]),
        "text_hash": private_chunk["chunk_text_sha256"],
        "text_hash_algorithm": TEXT_HASH_ALGORITHM_UTF8_SHA256_V1,
        "text_hash_status": "computed_from_private_chunk",
        "parent_text_hash_status": "computed_from_private_refetch",
        "text_byte_length": len(chunk_text.encode("utf-8", errors="surrogatepass")),
        "raw_text_serialized": False,
    }
    _reject_live_artifact_leakage(audit)
    return chunk_text, audit


def _run_chunk_vote_tasks(
    tasks: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    live: bool,
    max_chunks: int,
    budget_cap_usd: float,
    env: dict[str, str] | None,
    call: Callable[[dict[str, Any]], dict[str, Any]],
    provider_config: DeepSeekLabelingConfig,
    price: PriceSnapshot,
    per_vote_estimate: float,
    concurrency: int,
    on_row: Callable[[list[dict[str, Any]]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    using_live_client = call is call_deepseek_live

    def run_one(candidate: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
        if using_live_client:
            prepare_deepseek_live_call(request_payload, live=live, max_docs=max_chunks, budget_cap_usd=budget_cap_usd, env=env)
        start = time.monotonic()
        response_body = run_with_retries(
            lambda prepared_request: call(
                prepared_request,
                live=live,
                max_docs=max_chunks,
                budget_cap_usd=budget_cap_usd,
                env=env,
            )
            if call is call_deepseek_live
            else call(prepared_request),
            request_payload,
            max_attempts=provider_config.max_retries + 1,
        )
        latency_seconds = time.monotonic() - start
        finish_value = _finish_reason(response_body)
        actual_cost_value = actual_label_cost_usd(response_body, price)
        try:
            row = parse_teacher_response(
                response_body,
                candidate=candidate,
                request=request_payload,
                price_snapshot=price,
                estimated_cost_usd=float(per_vote_estimate),
                actual_cost_usd=actual_cost_value,
                latency_seconds=latency_seconds,
            )
        except ValueError as exc:
            row = teacher_label_parse_error_record(
                error_code="provider_response_rejected",
                error_message=str(exc),
                candidate=candidate,
                request=request_payload,
                response_body=response_body,
                finish_reason=finish_value,
                usage=response_body.get("usage", {}),
                estimated_cost_usd=float(per_vote_estimate),
                actual_cost_usd=actual_cost_value,
                latency_seconds=latency_seconds,
            )
        return _attach_chunk_vote_metadata(row, request_payload)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stopped = False
    stop_requested = stop_requested or (lambda: False)
    if not tasks:
        return {"rows": rows, "transport_failures": failures, "stopped": stopped}
    wave_size = max(1, int(concurrency))
    for start in range(0, len(tasks), wave_size):
        if stop_requested():
            stopped = True
            break
        wave = tasks[start : start + wave_size]
        with ThreadPoolExecutor(max_workers=wave_size) as executor:
            future_map = {executor.submit(run_one, candidate, request): (candidate, request) for candidate, request in wave}
            for future in as_completed(future_map):
                candidate, request = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001 - converted to no-raw retry item.
                    failures.append(_chunk_transport_failure_item(candidate, request, exc))
                    continue
                _reject_live_artifact_leakage(row)
                rows.append(row)
                if on_row is not None:
                    on_row([row])
                if stop_requested():
                    stopped = True
        if any(item.get("http_status_code") in {401, 402} for item in failures):
            break
        if stopped:
            break
    rows.sort(key=lambda row: (str(row["candidate_id"]), int(row["vote_index"])))
    failures.sort(key=lambda item: (str(item["candidate_id"]), int(item["vote_index"])))
    return {"rows": rows, "transport_failures": failures, "stopped": stopped}


def _build_chunk_label_aggregation(
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    transport_failures: list[dict[str, Any]],
    manifest: dict[str, Any],
    policy: ChunkRelabelPolicy,
    provider_config_hash: str,
) -> dict[str, Any]:
    parsed = [row for row in rows if row.get("parse_status") == "parsed"]
    parse_errors = [row for row in rows if row.get("parse_status") == "parse_error"]
    required_indexes = set(range(policy.required_vote_count))
    parsed_by_index: dict[int, dict[str, Any]] = {}
    for row in parsed:
        vote_index = int(row.get("vote_index", -1))
        if vote_index in required_indexes:
            parsed_by_index[vote_index] = row
    parsed_required = [parsed_by_index[index] for index in sorted(parsed_by_index)]
    parse_error_required = [row for row in parse_errors if int(row.get("vote_index", -1)) in required_indexes]
    unrepaired_parse_errors = [
        row for row in parse_error_required if int(row.get("vote_index", -1)) not in parsed_by_index
    ]
    vote_indexes = sorted({int(row.get("vote_index", -1)) for row in rows})
    attempted_required = {int(row.get("vote_index", -1)) for row in rows if int(row.get("vote_index", -1)) in required_indexes}
    label_counts: dict[str, int] = {}
    borderline_true = 0
    borderline_false = 0
    for row in parsed_required:
        label = str(row.get("label") or "missing")
        label_counts[label] = label_counts.get(label, 0) + 1
        if row.get("borderline") is True:
            borderline_true += 1
        elif row.get("borderline") is False:
            borderline_false += 1
    complete = (
        not transport_failures
        and not unrepaired_parse_errors
        and len(parsed_required) == policy.required_vote_count
        and sorted(int(row.get("vote_index", -1)) for row in parsed_required) == sorted(required_indexes)
    )
    if complete:
        if label_counts == {"safe": policy.required_vote_count} and borderline_true == 0:
            final_label = "safe"
            route = "unanimous_safe"
        elif label_counts == {"reject": policy.required_vote_count} and borderline_true == 0:
            final_label = "reject"
            route = "unanimous_reject"
        elif borderline_true > 0:
            final_label = "reject"
            route = "borderline_fail_closed_reject"
        else:
            final_label = "reject"
            route = "mixed_fail_closed_reject"
        ready = True
    else:
        final_label = None
        route = "parse_or_transport_repair_required"
        ready = False
    aggregation = {
        "chunk_label_aggregation_schema_version": 1,
        "candidate_id": candidate["candidate_id"],
        "parent_candidate_id": candidate["parent_candidate_id"],
        "source_chunk_manifest_hash": manifest["manifest_hash"],
        "candidate_hash": candidate.get("candidate_hash"),
        "provider_config_hash": provider_config_hash,
        "chunk_relabel_policy_version": policy.policy_version,
        "required_vote_count": policy.required_vote_count,
        "vote_count": len(rows),
        "parsed_vote_count": len(parsed),
        "parse_error_vote_count": len(parse_errors),
        "transport_failure_count": len(transport_failures),
        "vote_indexes": vote_indexes,
        "label_counts": dict(sorted(label_counts.items())),
        "borderline_true_count": borderline_true,
        "borderline_false_count": borderline_false,
        "aggregation_route": route,
        "final_label": final_label,
        "chunk_label_row_ready": ready,
        "labels_are_corpus_approval": False,
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
    }
    aggregation["chunk_label_aggregation_hash"] = _hash(aggregation)
    _reject_live_artifact_leakage(aggregation)
    return aggregation


def _chunk_label_row_from_aggregation(aggregation: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if aggregation.get("chunk_label_row_ready") is not True or aggregation.get("final_label") not in {"safe", "reject"}:
        raise ValueError("chunk label row requires ready safe/reject aggregation")
    row = {
        "chunk_label_row_schema_version": 1,
        "parse_status": "chunk_label",
        "candidate_id": aggregation["candidate_id"],
        "parent_candidate_id": aggregation["parent_candidate_id"],
        "source_chunk_manifest_hash": aggregation["source_chunk_manifest_hash"],
        "candidate_hash": aggregation.get("candidate_hash"),
        "chunk_relabel_policy_version": aggregation["chunk_relabel_policy_version"],
        "chunk_label_aggregation_hash": aggregation["chunk_label_aggregation_hash"],
        "label": aggregation["final_label"],
        "final_label": aggregation["final_label"],
        "aggregation_route": aggregation["aggregation_route"],
        "required_vote_count": aggregation["required_vote_count"],
        "parsed_vote_count": aggregation["parsed_vote_count"],
        "parse_error_vote_count": aggregation["parse_error_vote_count"],
        "transport_failure_count": aggregation["transport_failure_count"],
        "label_unit": candidate["label_unit"],
        "global_chunk_index": candidate["global_chunk_index"],
        "parent_chunk_index": candidate["parent_chunk_index"],
        "parent_chunk_count": candidate["parent_chunk_count"],
        "source_schedule_hash": candidate.get("source_schedule_hash"),
        "schedule_entry_hash": candidate.get("schedule_entry_hash"),
        "source_file_id": candidate.get("source_file_id"),
        "row_group_index": candidate.get("row_group_index"),
        "row_group_local_row_index": candidate.get("row_group_local_row_index"),
        "labels_are_corpus_approval": False,
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    row["chunk_label_row_hash"] = _hash(row)
    _reject_live_artifact_leakage(row)
    return row


def load_fresh_chunk_repair_overlay(labels_jsonl: str | Path, provenance_json: str | Path) -> dict[str, Any]:
    """Load a no-raw fresh chunk repair overlay without mutating canonical stages."""

    label_rows = _read_jsonl_objects(labels_jsonl)
    provenance = _load_json(provenance_json)
    _reject_live_artifact_leakage(provenance)
    _require_false_flags(
        provenance,
        ("raw_text_serialized", "raw_url_serialized", "token_serialized", "accepted_manifest", "training_shards_created", "classifier_training_performed"),
        context="fresh chunk repair provenance",
    )
    if provenance.get("schema_version") != 1:
        raise ValueError("fresh chunk repair provenance schema version mismatch")
    if provenance.get("consolidation_rule") != FRESH_CHUNK_REPAIR_CONSOLIDATION_RULE:
        raise ValueError("fresh chunk repair provenance consolidation rule mismatch")
    if int(provenance.get("unlabelled_count") or 0) != 0 or provenance.get("unlabelled") not in ([], None):
        raise ValueError("fresh chunk repair overlay cannot contain unlabelled chunks")
    items = provenance.get("items")
    if not isinstance(items, list):
        raise ValueError("fresh chunk repair provenance items must be a list")
    if int(provenance.get("target_chunk_count") or -1) != len(items):
        raise ValueError("fresh chunk repair provenance target count mismatch")
    if int(provenance.get("ready_label_count") or -1) != len(label_rows):
        raise ValueError("fresh chunk repair provenance ready label count mismatch")

    label_candidate_ids: set[str] = set()
    for row in label_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in label_candidate_ids:
            raise ValueError("fresh chunk repair overlay contains duplicate label candidate_id")
        label_candidate_ids.add(candidate_id)

    provenance_by_candidate: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("fresh chunk repair provenance item must be an object")
        _validate_fresh_repair_provenance_item(item)
        candidate_id = str(item["candidate_id"])
        if candidate_id in provenance_by_candidate:
            raise ValueError("fresh chunk repair overlay contains duplicate provenance candidate_id")
        provenance_by_candidate[candidate_id] = item
    if set(provenance_by_candidate) != label_candidate_ids:
        raise ValueError("fresh chunk repair overlay label/provenance candidate mismatch")

    labels_by_candidate: dict[str, dict[str, Any]] = {}
    for row in label_rows:
        _validate_chunk_label_evidence_row(row, source=FRESH_CHUNK_REPAIR_LABEL_SOURCE)
        candidate_id = str(row["candidate_id"])
        if candidate_id in labels_by_candidate:
            raise ValueError("fresh chunk repair overlay contains duplicate label candidate_id")
        item = provenance_by_candidate.get(candidate_id)
        if item is None:
            raise ValueError("fresh chunk repair overlay label is missing provenance")
        if row["source_chunk_manifest_hash"] != item["source_chunk_manifest_hash"]:
            raise ValueError("fresh chunk repair overlay manifest hash mismatch")
        if row["final_label"] != item["final_label"]:
            raise ValueError("fresh chunk repair overlay final label mismatch")
        if row["aggregation_route"] != item["aggregation_route"]:
            raise ValueError("fresh chunk repair overlay aggregation route mismatch")
        if int(row.get("parse_error_vote_count") or 0) != 0:
            raise ValueError("fresh chunk repair overlay must not count unparsed attempts as selected votes")
        labels_by_candidate[candidate_id] = row

    overlay = {
        "fresh_chunk_repair_overlay_schema_version": 1,
        "label_source": FRESH_CHUNK_REPAIR_LABEL_SOURCE,
        "consolidation_rule": FRESH_CHUNK_REPAIR_CONSOLIDATION_RULE,
        "label_count": len(labels_by_candidate),
        "labels_by_candidate_id": labels_by_candidate,
        "provenance_by_candidate_id": provenance_by_candidate,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    overlay["fresh_chunk_repair_overlay_hash"] = _hash(overlay)
    _reject_live_artifact_leakage(overlay)
    return overlay


def build_effective_chunk_label_evidence(
    canonical_label_rows: list[dict[str, Any]],
    *,
    repair_overlay: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build classifier-label evidence with fresh repair rows taking precedence."""

    overlay_labels: dict[str, dict[str, Any]] = {}
    if repair_overlay is not None:
        if repair_overlay.get("label_source") != FRESH_CHUNK_REPAIR_LABEL_SOURCE:
            raise ValueError("repair overlay label source mismatch")
        overlay_labels = dict(repair_overlay.get("labels_by_candidate_id") or {})
        for candidate_id, row in overlay_labels.items():
            if str(row.get("candidate_id") or "") != str(candidate_id):
                raise ValueError("fresh repair overlay row key/candidate mismatch")
            _validate_chunk_label_evidence_row(row, source=FRESH_CHUNK_REPAIR_LABEL_SOURCE)
            if int(row.get("parse_error_vote_count") or 0) != 0:
                raise ValueError("fresh chunk repair overlay must not count unparsed attempts as selected votes")

    canonical_by_candidate: dict[str, dict[str, Any]] = {}
    for row in canonical_label_rows:
        candidate_id = str(row.get("candidate_id") or "")
        _validate_chunk_label_evidence_row(
            row,
            source=CANONICAL_CHUNK_LABEL_SOURCE,
            require_complete_votes=candidate_id not in overlay_labels,
        )
        candidate_id = str(row["candidate_id"])
        if candidate_id in canonical_by_candidate:
            raise ValueError("canonical chunk label evidence contains duplicate candidate_id")
        canonical_by_candidate[candidate_id] = row

    if overlay_labels:
        unknown_overlay_ids = sorted(set(overlay_labels) - set(canonical_by_candidate))
        if unknown_overlay_ids:
            raise ValueError("fresh repair overlay contains candidate_id outside canonical evidence")

    effective_by_candidate: dict[str, dict[str, Any]] = {}
    for candidate_id in sorted(canonical_by_candidate):
        canonical = canonical_by_candidate[candidate_id]
        repair = overlay_labels.get(candidate_id)
        effective = repair or canonical
        source = FRESH_CHUNK_REPAIR_LABEL_SOURCE if repair is not None else CANONICAL_CHUNK_LABEL_SOURCE
        evidence_row = {
            "chunk_label_evidence_schema_version": 1,
            "candidate_id": candidate_id,
            "parent_candidate_id": effective.get("parent_candidate_id"),
            "source_chunk_manifest_hash": effective.get("source_chunk_manifest_hash"),
            "candidate_hash": effective.get("candidate_hash"),
            "final_label": effective["final_label"],
            "label": effective["final_label"],
            "label_source": source,
            "canonical_final_label": canonical["final_label"],
            "canonical_chunk_label_row_hash": canonical.get("chunk_label_row_hash"),
            "repair_overlay_final_label": None if repair is None else repair["final_label"],
            "repair_overlay_chunk_label_row_hash": None if repair is None else repair.get("chunk_label_row_hash"),
            "labels_are_corpus_approval": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
            "accepted_manifest": False,
            "training_shards_created": False,
            "classifier_training_performed": False,
        }
        evidence_row["chunk_label_evidence_hash"] = _hash(evidence_row)
        _reject_live_artifact_leakage(evidence_row)
        effective_by_candidate[candidate_id] = evidence_row

    evidence = {
        "chunk_label_evidence_view_schema_version": 1,
        "canonical_label_count": len(canonical_by_candidate),
        "repair_overlay_label_count": len(overlay_labels),
        "effective_label_count": len(effective_by_candidate),
        "labels_by_candidate_id": effective_by_candidate,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    evidence["chunk_label_evidence_view_hash"] = _hash(evidence)
    _reject_live_artifact_leakage(evidence)
    return evidence


def _validate_chunk_label_evidence_row(row: dict[str, Any], *, source: str, require_complete_votes: bool = True) -> None:
    if not isinstance(row, dict):
        raise ValueError(f"{source} chunk label evidence row must be an object")
    _reject_live_artifact_leakage(row)
    if row.get("parse_status") != "chunk_label":
        raise ValueError(f"{source} chunk label evidence row must have parse_status=chunk_label")
    if row.get("chunk_label_row_schema_version") != 1:
        raise ValueError(f"{source} chunk label row schema version mismatch")
    _require_false_flags(row, _PUBLIC_FALSE_FLAGS, context=f"{source} chunk label evidence row")
    _require_sha256(str(row.get("candidate_id") or ""), "candidate_id")
    _require_sha256(str(row.get("parent_candidate_id") or ""), "parent_candidate_id")
    _require_sha256(str(row.get("source_chunk_manifest_hash") or ""), "source_chunk_manifest_hash")
    _require_sha256(str(row.get("candidate_hash") or ""), "candidate_hash")
    if row.get("chunk_relabel_policy_version") != CHUNK_RELABEL_POLICY_VERSION:
        raise ValueError(f"{source} chunk label row policy version mismatch")
    if row.get("label") not in {"safe", "reject"} or row.get("final_label") != row.get("label"):
        raise ValueError(f"{source} chunk label row final_label mismatch")
    if int(row.get("required_vote_count") or 0) != 3:
        raise ValueError(f"{source} chunk label row must require three votes")
    if require_complete_votes and int(row.get("parsed_vote_count") or 0) != 3:
        raise ValueError(f"{source} chunk label row must contain three parsed votes")
    if require_complete_votes and int(row.get("transport_failure_count") or 0) != 0:
        raise ValueError(f"{source} chunk label row must not contain transport failures")
    if row.get("label_unit") != "window_bundle":
        raise ValueError(f"{source} chunk label row label_unit must be window_bundle")
    row_hash = str(row.get("chunk_label_row_hash") or "")
    _require_sha256(row_hash, "chunk_label_row_hash")
    without_hash = dict(row)
    without_hash.pop("chunk_label_row_hash", None)
    if row_hash != _hash(without_hash):
        raise ValueError(f"{source} chunk label row hash mismatch")


def _validate_fresh_repair_provenance_item(item: dict[str, Any]) -> None:
    _reject_live_artifact_leakage(item)
    _require_false_flags(item, ("raw_text_serialized", "raw_url_serialized", "token_serialized"), context="fresh chunk repair provenance item")
    _require_sha256(str(item.get("candidate_id") or ""), "candidate_id")
    _require_sha256(str(item.get("source_chunk_manifest_hash") or ""), "source_chunk_manifest_hash")
    if item.get("consolidation_rule") != FRESH_CHUNK_REPAIR_CONSOLIDATION_RULE:
        raise ValueError("fresh chunk repair provenance item consolidation rule mismatch")
    if item.get("final_label") not in {"safe", "reject"}:
        raise ValueError("fresh chunk repair provenance item final_label mismatch")
    indexes = [int(value) for value in item.get("selected_vote_indexes") or []]
    sources = item.get("selected_vote_sources")
    if item.get("selected_fresh_vote_count") != 3 or sorted(indexes) != [0, 1, 2]:
        raise ValueError("fresh chunk repair provenance item must select vote indexes 0,1,2")
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("fresh chunk repair provenance item selected vote sources mismatch")
    source_indexes = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("fresh chunk repair selected vote source must be an object")
        source_index = int(source.get("vote_index", -1))
        source_indexes.append(source_index)
        _require_sha256(str(source.get("idempotency_key") or ""), "idempotency_key")
        _require_sha256(str(source.get("response_body_sha256") or ""), "response_body_sha256")
        _require_sha256(str(source.get("label_row_hash") or ""), "label_row_hash")
        _require_positive_int(source.get("pass_index"), "pass_index")
        if not str(source.get("repair_dir") or ""):
            raise ValueError("fresh chunk repair selected vote source requires repair_dir")
    if sorted(source_indexes) != [0, 1, 2]:
        raise ValueError("fresh chunk repair provenance item selected vote source indexes mismatch")


def _require_false_flags(payload: dict[str, Any], fields: tuple[str, ...], *, context: str) -> None:
    for field in fields:
        if payload.get(field) is not False:
            raise ValueError(f"{context} must set {field}=false")


def _read_existing_chunk_vote_rows(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    provider_config: DeepSeekLabelingConfig,
    candidates: list[dict[str, Any]],
    policy: ChunkRelabelPolicy,
) -> list[dict[str, Any]]:
    output = Path(path)
    if not output.exists():
        return []
    rows = _read_jsonl_objects(output)
    expected_manifest_hash = str(manifest["manifest_hash"])
    expected_config_hash = _hash(_config_payload(provider_config))
    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    seen_idempotency: set[str] = set()
    rows_by_vote: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        _reject_live_artifact_leakage(row)
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id not in candidate_ids:
            raise ValueError("existing chunk vote row contains a candidate outside the current run")
        if str(row.get("source_chunk_manifest_hash") or row.get("source_candidate_manifest_hash") or "") != expected_manifest_hash:
            raise ValueError("existing chunk vote row belongs to a different chunk manifest")
        if str(row.get("provider_config_hash") or "") != expected_config_hash:
            raise ValueError("existing chunk vote row belongs to a different provider config")
        if str(row.get("teacher_prompt_version") or "") != provider_config.prompt_version:
            raise ValueError("existing chunk vote row belongs to a different teacher prompt")
        if str(row.get("teacher_model") or "") != provider_config.model:
            raise ValueError("existing chunk vote row belongs to a different teacher model")
        if str(row.get("chunk_relabel_policy_version") or "") != policy.policy_version:
            raise ValueError("existing chunk vote row belongs to a different chunk relabel policy")
        candidate = next(candidate for candidate in candidates if str(candidate["candidate_id"]) == candidate_id)
        row_label_unit = row.get("label_unit")
        expected_label_unit = candidate.get("label_unit")
        if row_label_unit != expected_label_unit and not (
            row.get("parse_status") == "parse_error" and row_label_unit is None
        ):
            raise ValueError("existing chunk vote row label_unit drift")
        if str(row.get("candidate_hash") or "") != str(candidate.get("candidate_hash") or ""):
            raise ValueError("existing chunk vote row candidate hash drift")
        if str(row.get("parent_candidate_id") or "") != str(candidate.get("parent_candidate_id") or ""):
            raise ValueError("existing chunk vote row parent candidate drift")
        if str(row.get("text_hash") or "") != str(candidate.get("text_hash_sha256") or ""):
            raise ValueError("existing chunk vote row text hash drift")
        _require_sha256(str(row.get("prompt_hash") or ""), "existing chunk vote prompt_hash")
        _require_sha256(str(row.get("request_body_sha256") or ""), "existing chunk vote request_body_sha256")
        vote_index = int(row.get("vote_index", -1))
        if vote_index < 0 or vote_index >= policy.required_vote_count:
            raise ValueError("existing chunk vote row has invalid vote_index")
        idempotency_key = str(row.get("idempotency_key") or "")
        if not idempotency_key:
            raise ValueError("existing chunk vote row is missing idempotency_key")
        if idempotency_key in seen_idempotency:
            raise ValueError("existing chunk vote rows contain duplicate idempotency_key")
        group_key = _hash(
            [
                "chunk_relabel_vote_group",
                policy.policy_version,
                expected_manifest_hash,
                candidate_id,
                row["text_hash"],
                row["prompt_hash"],
                expected_config_hash,
                row["request_body_sha256"],
            ]
        )
        if str(row.get("chunk_relabel_group_key") or "") != group_key:
            raise ValueError("existing chunk vote row group key drift")
        if str(row.get("bof_group_key") or "") != group_key:
            raise ValueError("existing chunk vote row BoF group key drift")
        retry_attempt = int(row.get("retry_attempt") or 0)
        retry_of = str(row.get("retry_of_idempotency_key") or "")
        retry_reason = str(row.get("retry_reason") or "")
        base_idempotency = _hash(["chunk_relabel_vote", group_key, vote_index])
        retry_idempotency = _hash(
            ["chunk_relabel_vote_retry", group_key, vote_index, retry_attempt, retry_of, retry_reason]
        )
        if retry_attempt == 0 and idempotency_key != base_idempotency:
            raise ValueError("existing chunk vote row idempotency drift")
        if retry_attempt > 0 and (retry_of != base_idempotency or idempotency_key != retry_idempotency):
            raise ValueError("existing chunk vote row retry idempotency drift")
        seen_idempotency.add(idempotency_key)
        vote_key = (candidate_id, vote_index)
        rows_by_vote.setdefault(vote_key, []).append(row)
        if row.get("parse_status") not in {"parsed", "parse_error"}:
            raise ValueError("existing chunk vote row has unsupported parse_status")
    for vote_key, vote_rows in rows_by_vote.items():
        parsed_count = sum(1 for row in vote_rows if row.get("parse_status") == "parsed")
        base_count = sum(1 for row in vote_rows if int(row.get("retry_attempt") or 0) == 0)
        if parsed_count > 1 or base_count != 1 or len(vote_rows) > 2:
            raise ValueError("existing chunk vote rows contain invalid parse-error retry history")
        if len(vote_rows) == 2:
            if not any(row.get("parse_status") == "parse_error" and int(row.get("retry_attempt") or 0) == 0 for row in vote_rows):
                raise ValueError("existing chunk vote retry requires one original parse-error row")
            if not any(row.get("parse_status") == "parsed" and int(row.get("retry_attempt") or 0) == 1 for row in vote_rows):
                raise ValueError("existing chunk vote retry requires one parsed retry row")
    return rows


def _with_chunk_vote_identity(
    request_payload: dict[str, Any],
    *,
    vote_index: int,
    policy: ChunkRelabelPolicy,
    retry_of_idempotency_key: str | None = None,
    retry_reason: str | None = None,
) -> dict[str, Any]:
    if vote_index < 0 or vote_index >= policy.required_vote_count:
        raise ValueError("chunk relabel vote_index must be between 0 and 2")
    updated = dict(request_payload)
    group_key = _chunk_bof_group_key(request_payload, policy=policy)
    retry_attempt = 1 if retry_of_idempotency_key else 0
    updated["bof_n"] = policy.required_vote_count
    updated["bof_call_index"] = int(vote_index)
    updated["vote_index"] = int(vote_index)
    updated["chunk_relabel_group_key"] = group_key
    updated["bof_group_key"] = group_key
    updated["chunk_relabel_policy_version"] = policy.policy_version
    updated["chunk_relabel_stage"] = "bof3"
    updated["retry_attempt"] = retry_attempt
    updated["retry_of_idempotency_key"] = str(retry_of_idempotency_key or "")
    updated["retry_reason"] = str(retry_reason or "")
    if retry_attempt:
        updated["idempotency_key"] = _hash(
            [
                "chunk_relabel_vote_retry",
                group_key,
                vote_index,
                retry_attempt,
                updated["retry_of_idempotency_key"],
                updated["retry_reason"],
            ]
        )
    else:
        updated["idempotency_key"] = _hash(["chunk_relabel_vote", group_key, vote_index])
    return updated


def _chunk_bof_group_key(request_payload: dict[str, Any], *, policy: ChunkRelabelPolicy) -> str:
    return _hash(
        [
            "chunk_relabel_vote_group",
            policy.policy_version,
            request_payload["source_candidate_manifest_hash"],
            request_payload["candidate_id"],
            request_payload["text_hash"],
            request_payload["prompt_hash"],
            request_payload["config_hash"],
            request_payload["request_body_sha256"],
        ]
    )


def _attach_chunk_vote_metadata(row: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["source_chunk_manifest_hash"] = str(request_payload["source_candidate_manifest_hash"])
    updated["candidate_hash"] = str(request_payload.get("candidate_hash") or "")
    updated["parent_candidate_id"] = str(request_payload.get("parent_candidate_id") or "")
    updated["label_unit"] = str(request_payload.get("label_unit") or "")
    updated["global_chunk_index"] = int(request_payload.get("global_chunk_index", -1))
    updated["parent_chunk_index"] = int(request_payload.get("parent_chunk_index", -1))
    updated["parent_chunk_count"] = int(request_payload.get("parent_chunk_count", -1))
    updated["prompt_hash"] = str(request_payload.get("prompt_hash") or "")
    updated["bof_n"] = int(request_payload["bof_n"])
    updated["bof_call_index"] = int(request_payload["bof_call_index"])
    updated["vote_index"] = int(request_payload["vote_index"])
    updated["chunk_relabel_group_key"] = str(request_payload["chunk_relabel_group_key"])
    updated["bof_group_key"] = str(request_payload["chunk_relabel_group_key"])
    updated["chunk_relabel_policy_version"] = str(request_payload["chunk_relabel_policy_version"])
    updated["chunk_relabel_stage"] = str(request_payload["chunk_relabel_stage"])
    updated["refetch_audit_hash"] = str(request_payload.get("refetch_audit_hash") or "")
    updated["bof_vote_role"] = "independent_same_prompt_vote"
    updated["semantic_retry_performed"] = False
    if "label_row_hash" in updated:
        updated.pop("label_row_hash", None)
        updated["label_row_hash"] = _hash(updated)
    if "label_parse_error_hash" in updated:
        updated.pop("label_parse_error_hash", None)
        updated["label_parse_error_hash"] = _hash(updated)
    _reject_live_artifact_leakage(updated)
    return updated


def _chunk_refetch_failure_item(
    candidate: dict[str, Any],
    *,
    missing: tuple[int, ...],
    exc: BaseException,
    manifest: dict[str, Any],
    provider_config: DeepSeekLabelingConfig,
    policy: ChunkRelabelPolicy,
) -> dict[str, Any]:
    item = {
        "work_item_schema_version": 1,
        "work_item_type": "chunk_label_private_refetch_failure_retry",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "parent_candidate_id": str(candidate.get("parent_candidate_id") or ""),
        "source_chunk_manifest_hash": manifest["manifest_hash"],
        "provider_config_hash": _hash(_config_payload(provider_config)),
        "teacher_model": provider_config.model,
        "teacher_prompt_version": provider_config.prompt_version,
        "chunk_relabel_policy_version": policy.policy_version,
        "missing_vote_indexes": [int(value) for value in missing],
        "error_class": exc.__class__.__name__,
        "private_refetch_failed_before_provider_call": True,
        "private_refetch_or_relabel_required": True,
        "raw_text_serialized": False,
        "raw_model_response_serialized": False,
    }
    item["transport_failure_work_item_hash"] = _hash(item)
    _reject_live_artifact_leakage(item)
    return item


def _chunk_transport_failure_item(candidate: dict[str, Any], request_payload: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    http_status_code = getattr(exc, "status_code", None)
    item = {
        "work_item_schema_version": 1,
        "work_item_type": "chunk_label_transport_failure_retry",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "parent_candidate_id": str(candidate.get("parent_candidate_id") or ""),
        "source_chunk_manifest_hash": str(request_payload.get("source_candidate_manifest_hash") or ""),
        "provider_config_hash": str(request_payload.get("config_hash") or ""),
        "teacher_model": str((request_payload.get("body") or {}).get("model") or ""),
        "teacher_prompt_version": str(request_payload.get("prompt_version") or ""),
        "prompt_hash": str(request_payload.get("prompt_hash") or ""),
        "request_body_sha256": str(request_payload.get("request_body_sha256") or ""),
        "idempotency_key": str(request_payload.get("idempotency_key") or ""),
        "chunk_relabel_group_key": str(request_payload.get("chunk_relabel_group_key") or ""),
        "vote_index": int(request_payload.get("vote_index", -1)),
        "bof_call_index": int(request_payload.get("bof_call_index", -1)),
        "chunk_relabel_stage": str(request_payload.get("chunk_relabel_stage") or ""),
        "source_doc_id": str(request_payload.get("source_doc_id") or ""),
        "source_identity_hash": str(request_payload.get("source_identity_hash") or ""),
        "text_hash": str(request_payload.get("text_hash") or ""),
        "refetch_audit_hash": str(request_payload.get("refetch_audit_hash") or ""),
        "error_class": exc.__class__.__name__,
        "http_status_code": http_status_code,
        "private_refetch_or_relabel_required": True,
        "raw_text_serialized": False,
        "raw_model_response_serialized": False,
    }
    item["transport_failure_work_item_hash"] = _hash(item)
    _reject_live_artifact_leakage(item)
    return item


def _write_chunk_repair_queue(
    path: str | Path,
    *,
    parse_error_rows: list[dict[str, Any]],
    transport_failures: list[dict[str, Any]],
    aggregations: list[dict[str, Any]],
) -> dict[str, Any]:
    by_finish_reason: dict[str, int] = {}
    by_error_code: dict[str, int] = {}
    by_schema_rule: dict[str, int] = {}
    for row in parse_error_rows:
        finish = str(row.get("finish_reason") or "missing")
        by_finish_reason[finish] = by_finish_reason.get(finish, 0) + 1
        error_code = str(row.get("error_code") or "missing")
        by_error_code[error_code] = by_error_code.get(error_code, 0) + 1
        schema_rule = str(row.get("schema_failure_rule") or "missing")
        by_schema_rule[schema_rule] = by_schema_rule.get(schema_rule, 0) + 1
    candidate_ids = sorted(
        {str(row.get("candidate_id") or "") for row in parse_error_rows if row.get("candidate_id")}
        | {str(item.get("candidate_id") or "") for item in transport_failures if item.get("candidate_id")}
        | {
            str(row.get("candidate_id") or "")
            for row in aggregations
            if row.get("aggregation_route") == "parse_or_transport_repair_required"
        }
    )
    work_items = []
    for candidate_id in candidate_ids:
        candidate_parse_rows = [row for row in parse_error_rows if str(row.get("candidate_id") or "") == candidate_id]
        candidate_transport = [item for item in transport_failures if str(item.get("candidate_id") or "") == candidate_id]
        aggregation = next((row for row in aggregations if str(row.get("candidate_id") or "") == candidate_id), {})
        item = {
            "repair_work_item_schema_version": 1,
            "repair_mode": "chunk_label_explicit_rerun_or_private_adjudication",
            "candidate_id": candidate_id,
            "parent_candidate_id": aggregation.get("parent_candidate_id"),
            "aggregation_route": aggregation.get("aggregation_route", "parse_or_transport_repair_required"),
            "required_vote_count": int(aggregation.get("required_vote_count") or 0),
            "parsed_vote_count": int(aggregation.get("parsed_vote_count") or 0),
            "parse_error_vote_count": len(candidate_parse_rows),
            "transport_failure_count": len(candidate_transport),
            "failed_idempotency_keys": sorted(str(row.get("idempotency_key") or "") for row in candidate_parse_rows if row.get("idempotency_key")),
            "transport_failure_hashes": sorted(str(row.get("transport_failure_work_item_hash") or "") for row in candidate_transport),
            "private_refetch_or_relabel_required": True,
            "semantic_retry_in_main_batch": False,
            "raw_text_serialized": False,
            "raw_model_response_serialized": False,
        }
        item["repair_work_item_hash"] = _hash(item)
        _reject_live_artifact_leakage(item)
        work_items.append(item)
    queue = {
        "repair_queue_schema_version": CHUNK_RELABEL_REPAIR_QUEUE_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "repair_queue_mode": "chunk_label_repair_or_rerun",
        "queued_count": len(work_items),
        "parse_error_vote_count": len(parse_error_rows),
        "transport_failure_count": len(transport_failures),
        "by_error_code": dict(sorted(by_error_code.items())),
        "by_finish_reason": dict(sorted(by_finish_reason.items())),
        "by_schema_failure_rule": dict(sorted(by_schema_rule.items())),
        "work_items": work_items,
        "transport_failure_work_items": transport_failures,
        "raw_text_serialized": False,
        "raw_model_response_serialized": False,
        "paid_api_calls_performed": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    queue["repair_queue_hash"] = _hash({key: value for key, value in queue.items() if key != "generated_at_utc"})
    _reject_live_artifact_leakage(queue)
    _write_json_atomic(path, queue)
    return queue


def _write_chunk_progress(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    policy: ChunkRelabelPolicy,
    provider_config_hash: str,
    rows: list[dict[str, Any]],
    aggregations: list[dict[str, Any]],
    observed_cost: float,
    estimated_cost: float,
    run_status: str,
    blocked_reason: str | None,
    max_chunks: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    final_safe_count = sum(1 for row in aggregations if row.get("final_label") == "safe")
    final_reject_count = sum(1 for row in aggregations if row.get("final_label") == "reject")
    unfinished = [row for row in aggregations if row.get("chunk_label_row_ready") is not True]
    progress = {
        "chunk_label_progress_schema_version": CHUNK_RELABEL_PROGRESS_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "phase": "chunk_label_dry_run" if run_status == "dry_run_no_paid_calls" else "chunk_label_bof3_batch",
        "run_status": run_status,
        "blocked_reason": blocked_reason,
        "source_chunk_manifest_hash": manifest["manifest_hash"],
        "chunk_relabel_policy_version": policy.policy_version,
        "provider_config_hash": provider_config_hash,
        "max_chunks": max_chunks,
        "total_chunk_count": int(manifest["chunk_count"]),
        "completed_chunk_count": sum(1 for row in aggregations if row.get("chunk_label_row_ready") is True),
        "unfinished_chunk_count": len(unfinished),
        "final_safe_count": final_safe_count,
        "final_reject_count": final_reject_count,
        "votes_recorded": len(rows),
        "parse_status_counts": _status_counts(rows, "parse_status"),
        "finish_reason_counts": _status_counts(rows, "finish_reason"),
        "observed_cost_usd": observed_cost,
        "estimated_cost_usd": estimated_cost,
        "active_concurrency": policy.target_concurrency,
        "private_refetch_concurrency": policy.private_refetch_concurrency,
        "elapsed_seconds": elapsed_seconds,
        "vote_throughput_per_second": (len(rows) / elapsed_seconds) if elapsed_seconds > 0 else None,
        "budget_cap_used_fraction": observed_cost / policy.budget_cap_usd if policy.budget_cap_usd else None,
        "stop_reason": blocked_reason or run_status,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    progress["chunk_label_progress_hash"] = _hash({key: value for key, value in progress.items() if key != "generated_at_utc"})
    _reject_live_artifact_leakage(progress)
    _write_json_atomic(path, progress)
    return progress


def _build_manifest_report(
    *,
    manifest: dict[str, Any],
    private_map: dict[str, Any],
    parent_reports: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    report = {
        "chunk_relabel_build_report_schema_version": CHUNK_RELABEL_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "status": "ready_for_chunk_labeling" if manifest["chunk_count"] else "blocked_no_rejected_chunks",
        "manifest_hash": manifest["manifest_hash"],
        "private_map_hash": private_map["private_map_hash"],
        "policy_version": CHUNK_RELABEL_POLICY_VERSION,
        "source_candidate_manifest_hash": manifest["source_candidate_manifest_hash"],
        "source_schedule_hash": manifest.get("source_schedule_hash"),
        "segmentation_policy_hash": manifest["segmentation_policy_hash"],
        "target_proxy_tokens": manifest["target_proxy_tokens"],
        "parent_document_skip_count": manifest.get("parent_document_skip_count", 0),
        "parent_document_count": manifest["parent_document_count"],
        "chunk_count": manifest["chunk_count"],
        "chunk_kind_counts": manifest["chunk_kind_counts"],
        "warning_code_counts": manifest["warning_code_counts"],
        "parent_status_summaries": parent_reports,
        "elapsed_seconds": elapsed_seconds,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    report["chunk_relabel_build_report_hash"] = _hash({key: value for key, value in report.items() if key != "generated_at_utc"})
    _reject_live_artifact_leakage(report)
    return report


def _write_label_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(output)


def _write_json_private(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_live_artifact_leakage(payload)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(output)
    output.chmod(0o600)


def _validate_provider_config(provider_config: DeepSeekLabelingConfig) -> None:
    default_config = DeepSeekLabelingConfig()
    if provider_config.model != default_config.model or provider_config.temperature != 0.2:
        raise ValueError("chunk relabel batch must use canonical non-thinking DeepSeek V4 Pro settings")
    if provider_config.max_output_tokens != PROMPT_V3_DEEPSEEK_TEACHER_MAX_OUTPUT_TOKENS:
        raise ValueError("chunk relabel batch must use prompt-v3 max_output_tokens=512")
    if provider_config.thinking != "disabled" or provider_config.reasoning_effort:
        raise ValueError("chunk relabel batch must use thinking disabled")
    if provider_config.stream is not False or provider_config.response_format != {"type": "json_object"}:
        raise ValueError("chunk relabel batch must use JSON Output and stream=false")


def _validate_source_schedule_no_raw(schedule: dict[str, Any]) -> None:
    if not isinstance(schedule, dict):
        raise ValueError("source schedule must be an object")
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if schedule.get(field) is not False:
            raise ValueError(f"source schedule must set {field}=false")


def _validate_chunk_candidate(candidate: dict[str, Any], *, index: int, target_proxy_tokens: int) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("chunk relabel candidate must be an object")
    if candidate.get("label_unit") != "window_bundle":
        raise ValueError("chunk relabel candidate label_unit must be window_bundle")
    if candidate.get("teacher_label_use") != CHUNK_RELABEL_POLICY_VERSION:
        raise ValueError("chunk relabel candidate teacher_label_use mismatch")
    for field in ("candidate_id", "candidate_hash", "parent_candidate_id", "text_hash_sha256"):
        _require_sha256(str(candidate.get(field) or ""), field)
    if candidate.get("text_hash_algorithm") != TEXT_HASH_ALGORITHM_UTF8_SHA256_V1:
        raise ValueError("chunk relabel candidate text hash algorithm must be utf8_sha256_v1")
    if int(candidate.get("global_chunk_index")) != index:
        raise ValueError("chunk relabel candidate global indexes must be contiguous")
    for field in _PUBLIC_FALSE_FLAGS:
        if candidate.get(field) is not False:
            raise ValueError(f"chunk relabel candidate must set {field}=false")
    bundle = candidate.get("window_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("chunk relabel candidate requires window_bundle metadata")
    if bundle.get("bundle_kind") != "natural_block_chunk":
        raise ValueError("chunk relabel window_bundle kind mismatch")
    if int(bundle.get("target_proxy_tokens") or 0) != target_proxy_tokens:
        raise ValueError("chunk relabel target proxy token drift")
    if bundle.get("chunk_text_sha256") != candidate["text_hash_sha256"]:
        raise ValueError("chunk relabel bundle text hash mismatch")
    _require_positive_int(bundle.get("proxy_token_count"), "proxy_token_count")
    if str(candidate.get("candidate_hash") or "") != _chunk_candidate_hash(candidate):
        raise ValueError("chunk relabel candidate hash mismatch")


def _validate_inherited_safe_parent_candidate(
    candidate: dict[str, Any],
    *,
    index: int,
    target_proxy_tokens: int,
    source_manifest_hash: str,
) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("inherited safe candidate must be an object")
    if candidate.get("label_unit") != "window_bundle":
        raise ValueError("inherited safe candidate label_unit must be window_bundle")
    if candidate.get("teacher_label_use") != INHERITED_SAFE_PARENT_POLICY_VERSION:
        raise ValueError("inherited safe candidate teacher_label_use mismatch")
    if candidate.get("classifier_candidate_policy_version") != INHERITED_SAFE_PARENT_POLICY_VERSION:
        raise ValueError("inherited safe candidate policy mismatch")
    if candidate.get("parent_label_source") != INHERITED_SAFE_PARENT_LABEL_SOURCE:
        raise ValueError("inherited safe candidate parent label source mismatch")
    if candidate.get("source_chunk_manifest_hash") != source_manifest_hash:
        raise ValueError("inherited safe candidate source manifest hash mismatch")
    for field in (
        "candidate_id",
        "candidate_hash",
        "parent_candidate_id",
        "parent_candidate_hash",
        "parent_text_hash_sha256",
        "source_chunk_manifest_hash",
        "text_hash_sha256",
        "normalized_text_hash_sha256",
        "source_bucket_hash",
    ):
        _require_sha256(str(candidate.get(field) or ""), field)
    if candidate.get("text_hash_algorithm") != TEXT_HASH_ALGORITHM_UTF8_SHA256_V1:
        raise ValueError("inherited safe candidate text hash algorithm must be utf8_sha256_v1")
    if int(candidate.get("global_chunk_index")) != index:
        raise ValueError("inherited safe candidate global indexes must be contiguous")
    if not str(candidate.get("token_count_bucket") or ""):
        raise ValueError("inherited safe candidate requires token_count_bucket")
    for field in _PUBLIC_FALSE_FLAGS:
        if candidate.get(field) is not False:
            raise ValueError(f"inherited safe candidate must set {field}=false")
    bundle = candidate.get("window_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("inherited safe candidate requires window_bundle metadata")
    if bundle.get("bundle_kind") != "natural_block_chunk":
        raise ValueError("inherited safe window_bundle kind mismatch")
    if int(bundle.get("target_proxy_tokens") or 0) != target_proxy_tokens:
        raise ValueError("inherited safe target proxy token drift")
    if bundle.get("chunk_text_sha256") != candidate["text_hash_sha256"]:
        raise ValueError("inherited safe bundle text hash mismatch")
    _require_positive_int(bundle.get("proxy_token_count"), "proxy_token_count")
    if str(candidate.get("candidate_hash") or "") != _chunk_candidate_hash(candidate):
        raise ValueError("inherited safe candidate hash mismatch")


def _manifest_hash(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key not in {"generated_at_utc", "manifest_hash"}}
    return _hash(payload)


def _private_map_hash(private_map: dict[str, Any]) -> str:
    payload = {key: value for key, value in private_map.items() if key not in {"private_map_hash", "manifest_hash"}}
    return _hash(payload)


def _chunk_candidate_hash(candidate: dict[str, Any]) -> str:
    return _hash({key: value for key, value in candidate.items() if key != "candidate_hash"})


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _source_bucket_hash(parent: dict[str, Any]) -> str:
    dataset_payload = {
        "dataset_repo_id": str(parent.get("dataset_repo_id") or ""),
        "dataset_config": str(parent.get("dataset_config") or ""),
        "dataset_split": str(parent.get("dataset_split") or ""),
        "resolved_repo_sha": str(parent.get("resolved_repo_sha") or ""),
    }
    return _hash(
        {
            "dataset": dataset_payload,
            "source_file_id": str(parent.get("source_file_id") or ""),
            "source_metadata_hash": str(parent.get("source_metadata_hash") or ""),
        }
    )


def _proxy_token_count_bucket(proxy_count: int) -> str:
    proxy_count = _require_positive_int(proxy_count, "proxy_count")
    if proxy_count <= 256:
        return "0001_0256"
    if proxy_count <= 512:
        return "0257_0512"
    if proxy_count <= 1024:
        return "0513_1024"
    if proxy_count <= 2048:
        return "1025_2048"
    if proxy_count <= 4096:
        return "2049_4096"
    return "4097_plus"


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase sha256 hex string")


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _build_row_provider(args: argparse.Namespace, *, default_cache_dir: str) -> Any:
    columns = tuple(column.strip() for column in str(args.columns).split(",") if column.strip())
    if not columns or "text" not in columns or any(column not in {"text", "id"} for column in columns):
        raise ValueError("chunk relabel row-provider columns must be limited to text,id and include text")
    if args.row_provider == "local-parquet":
        from twc.data.decision_log import LocalParquetRowProvider

        return LocalParquetRowProvider(columns=columns or None)
    from twc.data.decision_log import HFFsspecParquetRowProvider

    return HFFsspecParquetRowProvider(
        cache_dir=args.hf_cache_dir or default_cache_dir,
        cache_delta_cap_bytes=args.cache_delta_cap_bytes,
        block_size_bytes=args.block_size_bytes,
        columns=columns or ("text", "id"),
        max_cache_blocks=args.max_cache_blocks,
        manage_hf_cache_env=False,
        row_group_batch_size=args.row_group_batch_size,
        source_concurrency=args.source_concurrency,
        source_retry_count=args.source_retry_count,
        source_retry_backoff_seconds=args.source_retry_backoff_seconds,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TWC rejected-document chunk relabeling utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-rejected-35k-chunk-manifest")
    build.add_argument("--manifest", required=True)
    build.add_argument("--labels-jsonl", required=True)
    build.add_argument("--schedule", required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--target-proxy-tokens", type=int, default=1024)
    build.add_argument("--max-parent-docs", type=int, default=None)
    build.add_argument("--skip-parent-docs", type=int, default=0)
    build.add_argument("--row-provider", choices=["local-parquet", "hf-fsspec"], required=True)
    build.add_argument("--hf-cache-dir", default="reports/teacher_labeling/rejected_chunk_row_cache")
    build.add_argument("--cache-delta-cap-bytes", type=int, default=256 * 1024 * 1024)
    build.add_argument("--block-size-bytes", type=int, default=16 * 1024 * 1024)
    build.add_argument("--max-cache-blocks", type=int, default=4)
    build.add_argument("--source-concurrency", type=int, default=32)
    build.add_argument("--row-group-batch-size", type=int, default=1)
    build.add_argument("--source-retry-count", type=int, default=3)
    build.add_argument("--source-retry-backoff-seconds", type=float, default=1.0)
    build.add_argument("--columns", default="text,id")

    run = subparsers.add_parser("run-chunk-label-batch")
    run.add_argument("--manifest", required=True)
    run.add_argument("--private-map", required=True)
    run.add_argument("--schedule", required=True)
    run.add_argument("--out-votes", required=True)
    run.add_argument("--out-labels", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--repair-queue", required=True)
    run.add_argument("--progress", required=True)
    run.add_argument("--live", action="store_true")
    run.add_argument("--max-chunks", type=int, required=True)
    run.add_argument("--current-balance-usd", type=float, required=True)
    run.add_argument("--budget-cap-usd", type=float, required=True)
    run.add_argument("--minimum-balance-reserve-usd", type=float, default=0.0)
    run.add_argument("--target-concurrency", type=int, default=64)
    run.add_argument("--tranche-chunks", type=int, default=512)
    run.add_argument("--private-refetch-concurrency", type=int, default=64)
    run.add_argument("--row-provider", choices=["local-parquet", "hf-fsspec"], required=True)
    run.add_argument("--hf-cache-dir", default="reports/teacher_labeling/rejected_chunk_row_cache")
    run.add_argument("--cache-delta-cap-bytes", type=int, default=256 * 1024 * 1024)
    run.add_argument("--block-size-bytes", type=int, default=16 * 1024 * 1024)
    run.add_argument("--max-cache-blocks", type=int, default=4)
    run.add_argument("--source-concurrency", type=int, default=32)
    run.add_argument("--row-group-batch-size", type=int, default=1)
    run.add_argument("--source-retry-count", type=int, default=3)
    run.add_argument("--source-retry-backoff-seconds", type=float, default=1.0)
    run.add_argument("--columns", default="text,id")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-rejected-35k-chunk-manifest":
        result = build_rejected_chunk_relabel_artifacts(
            source_manifest=_load_json(args.manifest),
            final_labels=_read_jsonl_objects(args.labels_jsonl),
            source_schedule=_load_json(args.schedule),
            row_provider=_build_row_provider(args, default_cache_dir="reports/teacher_labeling/rejected_chunk_row_cache"),
            output_dir=args.out_dir,
            target_proxy_tokens=args.target_proxy_tokens,
            max_parent_docs=args.max_parent_docs,
            skip_parent_docs=args.skip_parent_docs,
        )
        print(
            json.dumps(
                {
                    "out_dir": args.out_dir,
                    "manifest_hash": result["manifest"]["manifest_hash"],
                    "private_map_hash": result["private_map"]["private_map_hash"],
                    "parent_document_count": result["manifest"]["parent_document_count"],
                    "chunk_count": result["manifest"]["chunk_count"],
                    "ready_for_paid_chunk_labeling": result["manifest"]["ready_for_paid_chunk_labeling"],
                    "paid_api_calls_performed": False,
                    "raw_text_serialized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-chunk-label-batch":
        report = run_chunk_label_batch(
            manifest=_load_json(args.manifest),
            private_map=_load_json(args.private_map),
            source_schedule=_load_json(args.schedule),
            row_provider=_build_row_provider(args, default_cache_dir="reports/teacher_labeling/rejected_chunk_row_cache"),
            vote_jsonl=args.out_votes,
            output_jsonl=args.out_labels,
            report_path=args.report,
            repair_queue_path=args.repair_queue,
            progress_path=args.progress,
            live=bool(args.live),
            max_chunks=args.max_chunks,
            current_balance_usd=args.current_balance_usd,
            budget_cap_usd=args.budget_cap_usd,
            minimum_balance_reserve_usd=args.minimum_balance_reserve_usd,
            target_concurrency=args.target_concurrency,
            private_refetch_concurrency=args.private_refetch_concurrency,
            tranche_chunks=args.tranche_chunks,
            env=os.environ,
        )
        print(
            json.dumps(
                {
                    "report": args.report,
                    "out_votes": args.out_votes,
                    "out_labels": args.out_labels,
                    "run_status": report["run_status"],
                    "blocked_reason": report["blocked_reason"],
                    "votes_recorded": report["votes_recorded"],
                    "chunk_label_rows_ready": report["chunk_label_rows_ready"],
                    "unfinished_chunk_count": report["unfinished_chunk_count"],
                    "paid_api_calls_made": report["paid_api_calls_made"],
                    "paid_api_calls_performed": report["paid_api_calls_performed"],
                    "chunk_label_batch_report_hash": report["chunk_label_batch_report_hash"],
                },
                sort_keys=True,
            )
        )
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
