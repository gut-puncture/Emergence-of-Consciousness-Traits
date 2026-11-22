"""Candidate records, survivor queues, and prefilter receipt validation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from twc.classifiers import chunk_contracts, chunk_inference
from twc.data.risk_ontology import RiskOntology
from twc.repro.hashing import stable_json_hash, self_hash_mismatches
from twc.pretraining.filtering import artifacts as filtering_artifacts
from twc.pretraining.filtering import policy as filtering_policy
from twc.pretraining.filtering import stage_state_v1
from twc.pretraining.filtering.prefilter_metrics import _prefilter_runtime_receipt_fields
from twc.pretraining.chunked_clean_run import (
    SegmentationResult,
    _estimate_proxy_tokens_from_normalized,
)
from twc.pretraining.deterministic_filter import (
    DeterministicFilterRuntime, evaluate_chunk_candidate_deterministic_filter,
    evaluate_chunk_candidate_deterministic_filter_with_runtime,
)

def _is_w1_compact_prefilter_receipt(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("compact_start_state_artifact_kind") == filtering_artifacts.W1_COMPACT_START_STATE_KIND

def _validate_w1_compact_ready_db(
    conn: stage_state_v1.FilteringStageStateV1,
    run_path: Path,
    config_payload: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    queue_rows: Sequence[Mapping[str, Any]],
    require_empty_w1_scores: bool = True,
) -> None:
    if receipt.get("compact_start_state_status") != "compact_fresh_import":
        raise ValueError("compact start state status mismatch")
    inventory = conn.compact_inventory(
        [str(row["private_evidence_row_id"]) for row in queue_rows]
    )
    compact_candidate_count = inventory["candidate_count"]
    compact_survivor_count = inventory["survivor_count"]
    compact_evidence_count = inventory["private_evidence_count"]
    compact_score_count = inventory["w1_score_count"]
    survivor_count = len(queue_rows)
    if compact_candidate_count != survivor_count or compact_survivor_count != survivor_count:
        raise ValueError("compact start state survivor candidate count mismatch")
    if compact_evidence_count != survivor_count:
        raise ValueError("compact start state private evidence count mismatch")
    if require_empty_w1_scores and compact_score_count != 0:
        raise ValueError("compact start state must have empty w1_scores")
    if int(receipt.get("compact_candidate_count", -1)) != compact_candidate_count:
        raise ValueError("compact start state compact candidate count mismatch")
    if int(receipt.get("compact_private_evidence_count", -1)) != compact_evidence_count:
        raise ValueError("compact start state compact private evidence count mismatch")
    if int(receipt.get("survivor_count", -1)) != survivor_count:
        raise ValueError("compact start state survivor count mismatch")
    db_queue_rows = _prefilter_survivor_queue_rows(conn, config_payload)
    if not _survivor_queue_rows_match_live_db(queue_rows, db_queue_rows):
        raise ValueError("compact start state survivor queue export does not match SQLite state")
    filtering_artifacts._require_sha256(receipt.get("source_prefilter_stage_receipt_hash"), "source_prefilter_stage_receipt_hash")
    filtering_artifacts._require_sha256(receipt.get("compact_sqlite_sha256"), "compact_sqlite_sha256")

def _prefilter_source_row_payload(
    *,
    entry: Mapping[str, Any],
    offset: int,
    row_status: str,
    error_class: str | None,
    segmentation_receipt_hash: str | None,
    candidate_count: int,
    survivor_count: int,
) -> dict[str, Any]:
    payload = {
        "artifact_kind": "twc_chunk_filter_prefilter_source_row_receipt_v1",
        "schema_version": 1,
        "physical_row_key_hash": _physical_row_key(entry, offset),
        "source_schedule_hash": entry.get("source_schedule_hash"),
        "schedule_entry_hash": entry.get("entry_hash"),
        "source_file_id_hash": stable_json_hash(str(entry.get("source_file_id"))),
        "row_group_index": int(entry.get("row_group_index")),
        "row_group_local_offset": int(offset),
        "source_document_id_hash": _source_document_id_for_schedule_row(entry, offset),
        "row_status": row_status,
        "error_class": error_class,
        "segmentation_receipt_hash": segmentation_receipt_hash,
        "candidate_count": int(candidate_count),
        "survivor_count": int(survivor_count),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["row_receipt_hash"] = stable_json_hash(payload)
    filtering_artifacts.validate_public_no_raw_payload(payload)
    return payload

def _prefilter_terminal_offsets_for_entry(
    conn: stage_state_v1.FilteringStageStateV1,
    entry: Mapping[str, Any],
    offsets: Sequence[int],
) -> set[int]:
    key_to_offset = {_physical_row_key(entry, int(offset)): int(offset) for offset in offsets}
    return conn.terminal_offsets(key_to_offset)

def _prefilter_candidate_insert_record(
    candidate: Mapping[str, Any],
    *,
    prefilter_route: str,
    deterministic_result: Mapping[str, Any],
    s1_score: float | None,
    private_evidence_text: str | None,
) -> tuple[tuple[Any, ...], tuple[Any, ...] | None]:
    if prefilter_route not in {"deterministic_reject", "s1_reject", "prefilter_survivor", "prefilter_error", "prefilter_quarantine"}:
        raise ValueError("unsupported prefilter route")
    filtering_artifacts.validate_public_no_raw_payload(candidate)
    filtering_artifacts.validate_public_no_raw_payload(deterministic_result)
    candidate_id = filtering_artifacts._require_sha256(candidate.get("candidate_id_hash"), "candidate_id_hash")
    candidate_json = json.dumps(dict(candidate), sort_keys=True)
    deterministic_json = json.dumps(dict(deterministic_result), sort_keys=True)
    candidate_row = (
        candidate_id,
        candidate["source_document_id_hash"],
        candidate["segmentation_receipt_hash"],
        candidate_json,
        deterministic_json,
        prefilter_route,
        s1_score,
        time.time(),
    )
    evidence_row: tuple[Any, ...] | None = None
    if prefilter_route == "prefilter_survivor":
        if private_evidence_text is None:
            raise ValueError("prefilter survivor requires private evidence text")
        evidence_hash = filtering_artifacts._sha256_text(private_evidence_text)
        if evidence_hash != candidate["evidence_text_sha256"]:
            raise ValueError("private W1 evidence hash mismatch")
        private_evidence_id = stable_json_hash(
            {
                "candidate_id_hash": candidate_id,
                "evidence_text_sha256": evidence_hash,
                "config_hash": candidate["candidate_policy_hash"],
            }
        )
        evidence_row = (private_evidence_id, candidate_id, evidence_hash, private_evidence_text)
    return candidate_row, evidence_row

def _insert_prefilter_candidate(
    conn: stage_state_v1.FilteringStageStateV1,
    candidate: Mapping[str, Any],
    *,
    prefilter_route: str,
    deterministic_result: Mapping[str, Any],
    s1_score: float | None,
    private_evidence_text: str | None,
) -> None:
    record = _prefilter_candidate_insert_record(
        candidate,
        prefilter_route=prefilter_route,
        deterministic_result=deterministic_result,
        s1_score=s1_score,
        private_evidence_text=private_evidence_text,
    )
    conn.store_candidate_with_evidence(*record)

def _prefilter_segmented_result(
    conn: stage_state_v1.FilteringStageStateV1,
    *,
    result: SegmentationResult,
    raw_text: str,
    config_payload: Mapping[str, Any],
    sparse_scorer: chunk_contracts.ChunkFilteringScorer,
    deterministic_policy_payload: Mapping[str, Any],
    deterministic_ontology: RiskOntology,
    deterministic_runtime: DeterministicFilterRuntime | None = None,
) -> None:
    candidates = filtering_policy.build_filtering_candidates(
        result,
        raw_text=raw_text,
        config_hash=str(config_payload["config_hash"]),
        target_proxy_tokens=int(config_payload["target_proxy_tokens"]),
        halo_chars=int(config_payload["halo_chars"]),
        deterministic_policy_hash=str(deterministic_policy_payload["deterministic_filter_policy_hash"]),
    )
    scorer_items: list[tuple[Mapping[str, Any], Mapping[str, Any], str]] = []
    scorer_texts: list[str] = []
    scorer_metadata: list[dict[str, Any]] = []
    sparse_scorer_metadata_required = _sparse_scorer_requires_metadata(sparse_scorer)
    deterministic_normalization_cache: dict[tuple[Any, ...], Any] = {}
    for candidate in candidates:
        existing = conn.candidate_json(str(candidate["candidate_id_hash"]))
        if existing is not None:
            if existing != dict(candidate):
                raise ValueError("conflicting prefilter candidate row for existing id")
            continue
        if deterministic_runtime is None:
            deterministic_result = evaluate_chunk_candidate_deterministic_filter(
                candidate,
                raw_text=raw_text,
                ontology=deterministic_ontology,
                policy_payload=deterministic_policy_payload,
            )
        else:
            deterministic_result = evaluate_chunk_candidate_deterministic_filter_with_runtime(
                candidate,
                raw_text=raw_text,
                runtime=deterministic_runtime,
                normalization_cache=deterministic_normalization_cache,
            )
        deterministic_decision = filtering_policy._deterministic_candidate_decision(candidate)
        if deterministic_result.get("terminal_decision") == "reject":
            _insert_prefilter_candidate(
                conn,
                candidate,
                prefilter_route="deterministic_reject",
                deterministic_result=deterministic_result,
                s1_score=None,
                private_evidence_text=None,
            )
            continue
        if deterministic_decision == "quarantine":
            _insert_prefilter_candidate(
                conn,
                candidate,
                prefilter_route="prefilter_quarantine",
                deterministic_result=deterministic_result,
                s1_score=None,
                private_evidence_text=None,
            )
            continue
        evidence_text = raw_text[int(candidate["evidence_char_start"]) : int(candidate["evidence_char_end"])]
        scorer_items.append((candidate, deterministic_result, evidence_text))
        scorer_texts.append(evidence_text)
        if sparse_scorer_metadata_required:
            scorer_metadata.append(_sparse_scorer_metadata(sparse_scorer, candidate))
    if not scorer_items:
        return
    scores = list(
        sparse_scorer.score_many(
            scorer_texts,
            scorer_metadata if sparse_scorer_metadata_required else [],
        )
    )
    if len(scores) != len(scorer_items):
        scores = [math.nan] * len(scorer_items)
    for (candidate, deterministic_result, evidence_text), score in zip(scorer_items, scores, strict=True):
        parsed = chunk_contracts.coerce_optional_risk_score(score)
        if parsed is None:
            route = "prefilter_error"
        elif parsed >= float(config_payload["sparse_threshold"]):
            route = "s1_reject"
        else:
            route = "prefilter_survivor"
        _insert_prefilter_candidate(
            conn,
            candidate,
            prefilter_route=route,
            deterministic_result=deterministic_result,
            s1_score=parsed,
            private_evidence_text=evidence_text if route == "prefilter_survivor" else None,
        )

def _iter_prefilter_survivor_queue_rows(
    conn: stage_state_v1.FilteringStageStateV1,
    config_payload: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    for row in conn.iter_survivor_inputs():
        candidate = row.candidate
        deterministic_result = row.deterministic_result
        core_proxy_token_count = int(candidate["core_proxy_token_count"])
        evidence_proxy_token_count = candidate.get("evidence_proxy_token_count")
        if evidence_proxy_token_count is None:
            evidence_proxy_token_count = _estimate_proxy_tokens_from_normalized(row.private_evidence_text)
        evidence_proxy_token_count = max(core_proxy_token_count, int(evidence_proxy_token_count))
        payload = {
            "artifact_kind": "twc_chunk_filter_w1_survivor_queue_row_v1",
            "candidate_id_hash": row.candidate_id_hash,
            "source_document_id_hash": candidate["source_document_id_hash"],
            "segmentation_receipt_hash": candidate["segmentation_receipt_hash"],
            "candidate_policy_hash": candidate["candidate_policy_hash"],
            "source_metadata_hash": config_payload["source_metadata_hash"],
            "config_hash": config_payload["config_hash"],
            "selected_policy_hash": config_payload["selected_policy_hash"],
            "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
            "deterministic_filter_policy_hash": deterministic_result["deterministic_filter_policy_hash"],
            "deterministic_filter_result_hash": deterministic_result["deterministic_filter_result_hash"],
            "s1_score": chunk_contracts.coerce_risk_score(row.s1_score),
            "s1_threshold": filtering_policy._threshold(config_payload["sparse_threshold"], "s1_threshold"),
            "s1_report_hash": config_payload["sparse_report_hash"],
            "s1_model_file_sha256": config_payload["sparse_model_file_sha256"],
            "w1_threshold": filtering_policy._threshold(config_payload["window_mil_threshold"], "w1_threshold"),
            "w1_report_hash": config_payload["window_mil_report_hash"],
            "w1_model_file_sha256": config_payload["window_mil_model_file_sha256"],
            "core_text_sha256": candidate["core_text_sha256"],
            "evidence_text_sha256": candidate["evidence_text_sha256"],
            "core_proxy_token_count": core_proxy_token_count,
            "evidence_proxy_token_count": evidence_proxy_token_count,
            "estimated_w1_window_count": max(1, math.ceil(evidence_proxy_token_count / 512)),
            "private_evidence_row_id": row.private_evidence_row_id,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        payload["stage_receipt_hash"] = stable_json_hash(payload)
        filtering_artifacts.validate_public_no_raw_payload(payload)
        yield payload

def _legacy_core_estimated_survivor_queue_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.pop("stage_receipt_hash", None)
    payload.pop("evidence_proxy_token_count", None)
    payload["estimated_w1_window_count"] = max(1, math.ceil(int(payload["core_proxy_token_count"]) / 512))
    payload["stage_receipt_hash"] = stable_json_hash(payload)
    return payload

def _survivor_queue_rows_match_live_db(
    exported_rows: Sequence[Mapping[str, Any]],
    db_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if list(exported_rows) == list(db_rows):
        return True
    legacy_db_rows = [_legacy_core_estimated_survivor_queue_row(row) for row in db_rows]
    return list(exported_rows) == legacy_db_rows

def _prefilter_survivor_queue_rows(
    conn: stage_state_v1.FilteringStageStateV1,
    config_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return list(_iter_prefilter_survivor_queue_rows(conn, config_payload))

def _build_prefilter_receipt(
    conn: stage_state_v1.FilteringStageStateV1,
    config_payload: Mapping[str, Any],
    *,
    run_path: Path,
    status: str,
    stop_reason: str,
    elapsed_seconds: float,
    cache_dir: Path | None,
    min_free_bytes: int,
    survivor_queue_ready: bool | None = None,
    survivor_queue_part: bool | None = None,
    survivor_queue_hash: str | None = None,
    survivor_queue_row_count: int | None = None,
    survivor_queue_id_index_hash: str | None = None,
    survivor_queue_id_index_row_count: int | None = None,
    survivor_candidate_id_set_hash: str | None = None,
    survivor_private_evidence_id_set_hash: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    route_counts = conn.candidate_route_counts()
    survivor_count = int(route_counts.get("prefilter_survivor", 0))
    queue_ready = run_path / "w1_survivor_queue.jsonl.ready"
    queue_part = run_path / "w1_survivor_queue.jsonl.part"
    disk_bytes = _directory_size(run_path)
    cache_bytes = _directory_size(cache_dir) if cache_dir is not None else 0
    row_counts = conn.prefilter_summary_facts()
    accounted_rows = int(row_counts["terminal_rows"]) + int(row_counts["pending_rows"])
    expected_rows = int(row_counts["expected_physical_rows"])
    if accounted_rows > expected_rows:
        raise ValueError("prefilter expected source row accounting mismatch")
    unvisited_rows = expected_rows - accounted_rows
    ready_status = status in {"prefilter_ready_for_w1", "prefilter_shard_ready_for_merge"}
    if ready_status and int(row_counts["unaccounted_row_count"]) != 0:
        raise ValueError("prefilter readiness blocked by unaccounted source rows")
    if ready_status and int(row_counts["pending_rows"]) != 0:
        raise ValueError("prefilter readiness blocked by pending source rows")
    if ready_status and unvisited_rows != 0:
        raise ValueError("prefilter readiness blocked by unvisited source rows")
    if (
        ready_status
        and row_counts["expected_physical_row_key_set_hash"] is not None
        and row_counts["terminal_physical_row_key_set_hash"] != row_counts["expected_physical_row_key_set_hash"]
    ):
        raise ValueError("prefilter readiness blocked by expected physical row key-set mismatch")
    if ready_status and row_counts["row_error_counts"]:
        raise ValueError("prefilter readiness blocked by terminal row errors")
    next_phase = "w1-score" if status == "prefilter_ready_for_w1" else "resume_prefilter_cpu"
    if status == "prefilter_shard_ready_for_merge":
        next_phase = "build-sharded-w1-handoff"
    if status == "prefilter_blocked_errors":
        next_phase = "review_prefilter_row_errors"
    if status == "prefilter_diagnostic_complete":
        next_phase = "review_prefilter_diagnostic_only"
    if status == "prefilter_ready_for_w1" and survivor_count == 0:
        next_phase = "w1-score_or_assemble-report_zero_survivors"
    payload = {
        "artifact_kind": filtering_artifacts.PREFILTER_CPU_STAGE_KIND,
        "schema_version": 1,
        "phase": "prefilter-cpu",
        "status": status,
        "stop_reason": stop_reason,
        "next_required_phase": next_phase,
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "selected_eval_deterministic_filter_policy_hash": config_payload[
            "selected_eval_deterministic_filter_policy_hash"
        ],
        "executed_deterministic_filter_policy_hash": config_payload.get("deterministic_filter_policy_hash"),
        "candidate_count": int(sum(route_counts.values())),
        "survivor_count": survivor_count,
        "input_schedule_hash": row_counts["input_schedule_hash"],
        "expected_physical_rows": row_counts["expected_physical_rows"],
        "expected_physical_row_key_set_hash": row_counts["expected_physical_row_key_set_hash"],
        "scheduled_rows": row_counts["scheduled_rows"],
        "scheduled_physical_row_key_set_hash": row_counts["scheduled_physical_row_key_set_hash"],
        "terminal_rows": row_counts["terminal_rows"],
        "terminal_physical_row_key_set_hash": row_counts["terminal_physical_row_key_set_hash"],
        "pending_rows": row_counts["pending_rows"],
        "pending_physical_row_key_set_hash": row_counts["pending_physical_row_key_set_hash"],
        "unvisited_rows": int(unvisited_rows),
        "fetched_rows": row_counts["fetched_rows"],
        "segmented_rows": row_counts["segmented_rows"],
        "row_error_counts": row_counts["row_error_counts"],
        "unaccounted_row_count": row_counts["unaccounted_row_count"],
        "route_counts": route_counts,
        "survivor_queue_ready": queue_ready.exists() if survivor_queue_ready is None else bool(survivor_queue_ready),
        "survivor_queue_part": queue_part.exists() if survivor_queue_part is None else bool(survivor_queue_part),
        "survivor_queue_hash": (
            filtering_artifacts._sha256_file(queue_ready) if queue_ready.exists() else None
        )
        if survivor_queue_hash is None
        else filtering_artifacts._require_sha256(survivor_queue_hash, "survivor_queue_hash"),
        "survivor_queue_row_count": None if survivor_queue_row_count is None else filtering_artifacts._nonnegative_int(
            survivor_queue_row_count,
            "survivor_queue_row_count",
        ),
        "survivor_queue_id_index_hash": None
        if survivor_queue_id_index_hash is None
        else filtering_artifacts._require_sha256(survivor_queue_id_index_hash, "survivor_queue_id_index_hash"),
        "survivor_queue_id_index_row_count": None
        if survivor_queue_id_index_row_count is None
        else filtering_artifacts._nonnegative_int(
            survivor_queue_id_index_row_count,
            "survivor_queue_id_index_row_count",
        ),
        "survivor_candidate_id_set_hash": None
        if survivor_candidate_id_set_hash is None
        else filtering_artifacts._require_sha256(survivor_candidate_id_set_hash, "survivor_candidate_id_set_hash"),
        "survivor_private_evidence_id_set_hash": None
        if survivor_private_evidence_id_set_hash is None
        else filtering_artifacts._require_sha256(survivor_private_evidence_id_set_hash, "survivor_private_evidence_id_set_hash"),
        "survivor_queue_proof_kind": (
            "jsonl_sha256_and_id_set_hashes_v1"
            if (
                survivor_queue_row_count is not None
                and survivor_candidate_id_set_hash is not None
                and survivor_private_evidence_id_set_hash is not None
            )
            else None
        ),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "candidates_per_second": round(float(sum(route_counts.values()) / max(float(elapsed_seconds), 1e-9)), 6),
        "private_run_dir_disk_bytes": int(disk_bytes),
        "source_cache_disk_bytes": int(cache_bytes),
        "disk_free_bytes": min(filtering_artifacts._path_free_bytes(run_path), filtering_artifacts._path_free_bytes(cache_dir)) if cache_dir is not None else filtering_artifacts._path_free_bytes(run_path),
        "disk_min_free_bytes": int(min_free_bytes),
        "disk_watermark_guard_configured": True,
        "disk_watermark_guard_enforced": True,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    if extra_fields:
        filtering_artifacts.validate_public_no_raw_payload(extra_fields)
        payload.update(dict(extra_fields))
    payload["stage_receipt_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "stage_receipt_hash"})
    filtering_artifacts.validate_public_no_raw_payload(payload)
    return payload

def _sparse_scorer_metadata(sparse_scorer: chunk_contracts.ChunkFilteringScorer, candidate: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(sparse_scorer, chunk_inference.ChunkSparseRiskScorer):
        return {}
    return filtering_policy._candidate_public_metadata(candidate)

def _sparse_scorer_requires_metadata(sparse_scorer: chunk_contracts.ChunkFilteringScorer) -> bool:
    return type(sparse_scorer) is not chunk_inference.ChunkSparseRiskScorer

def _physical_row_key(entry: Mapping[str, Any], offset: int) -> str:
    return stable_json_hash(
        {
            "dataset_repo_id": entry.get("dataset_repo_id"),
            "config": entry.get("config"),
            "split": entry.get("split"),
            "resolved_repo_sha": entry.get("resolved_repo_sha"),
            "source_file_id": entry.get("source_file_id"),
            "row_group_index": int(entry.get("row_group_index")),
            "offset": int(offset),
        }
    )

def _source_document_id_for_schedule_row(entry: Mapping[str, Any], offset: int) -> str:
    return stable_json_hash(
        {
            "kind": "chunk_filter_source_document",
            "physical_row_key_hash": _physical_row_key(entry, offset),
        }
    )

def _metadata_token_count(row: Mapping[str, Any]) -> int | None:
    try:
        value = int(row.get("token_count"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None

@dataclass(frozen=True)
class _SurvivorQueueWriteProof:
    survivor_queue_hash: str
    survivor_queue_row_count: int
    survivor_queue_id_index_hash: str
    survivor_queue_id_index_row_count: int
    survivor_candidate_id_set_hash: str
    survivor_private_evidence_id_set_hash: str

def _survivor_queue_id_index_base_path(base_path: str | Path) -> Path:
    base = Path(base_path)
    return base.with_suffix(".ids.tsv")

def _survivor_queue_id_index_part_path(base_path: str | Path) -> Path:
    base = _survivor_queue_id_index_base_path(base_path)
    return base.with_suffix(base.suffix + ".part")

def _survivor_queue_id_index_ready_path(base_path: str | Path) -> Path:
    base = _survivor_queue_id_index_base_path(base_path)
    return base.with_suffix(base.suffix + ".ready")

def _acquire_survivor_queue_writer_lock(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
        expected = os.fstat(fd)
        filtering_artifacts._validate_private_partial_metadata(expected)
    except FileExistsError:
        expected = filtering_artifacts._validate_owned_private_partial(path)
        open_flags = os.O_RDWR | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
        fd = os.open(path, open_flags)
    try:
        opened = os.fstat(fd)
        filtering_artifacts._validate_private_partial_metadata(opened)
        current = filtering_artifacts._validate_owned_private_partial(path)
        if (
            (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise RuntimeError("survivor queue writer lock changed while opening")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError("survivor queue writer is already active") from None
    except BaseException:
        os.close(fd)
        raise

def _write_survivor_queue_partial_streaming(
    base_path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[str, int]:
    """Durably write a private diagnostic queue without publishing readiness."""

    base = Path(base_path)
    part = base.with_suffix(base.suffix + ".part")
    ready = base.with_suffix(base.suffix + ".ready")
    index_ready = _survivor_queue_id_index_ready_path(base)
    if ready.exists() or index_ready.exists():
        raise ValueError("partial survivor queue conflicts with published readiness")
    base.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = _acquire_survivor_queue_writer_lock(
        base.with_suffix(base.suffix + ".writer.lock")
    )
    digest = hashlib.sha256()
    candidate_ids: set[str] = set()
    evidence_ids: set[str] = set()
    count = 0
    try:
        queue_fd = filtering_artifacts._create_private_partial(part)
        with os.fdopen(queue_fd, "w", encoding="utf-8") as handle:
            for row in rows:
                filtering_artifacts.validate_public_no_raw_payload(row)
                candidate_id = filtering_artifacts._require_sha256(
                    row.get("candidate_id_hash"), "survivor queue candidate_id_hash"
                )
                evidence_id = filtering_artifacts._require_sha256(
                    row.get("private_evidence_row_id"),
                    "survivor queue private_evidence_row_id",
                )
                if candidate_id in candidate_ids:
                    raise ValueError("survivor queue contains duplicate candidate ids")
                if evidence_id in evidence_ids:
                    raise ValueError("survivor queue contains duplicate private evidence ids")
                candidate_ids.add(candidate_id)
                evidence_ids.add(evidence_id)
                line = json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
                count += 1
            filtering_artifacts._flush_and_fsync(handle)
        filtering_artifacts._fsync_directory(part.parent)
        return digest.hexdigest(), count
    finally:
        os.close(lock_fd)

def _write_survivor_queue_ready_streaming(
    base_path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    stage_state: stage_state_v1.FilteringStageStateV1,
    config_payload: Mapping[str, Any],
    _iter_live_rows: Callable[
        [stage_state_v1.FilteringStageStateV1, Mapping[str, Any]],
        Iterator[dict[str, Any]],
    ] = _iter_prefilter_survivor_queue_rows,
) -> _SurvivorQueueWriteProof:
    base = Path(base_path)
    part = base.with_suffix(base.suffix + ".part")
    ready = base.with_suffix(base.suffix + ".ready")
    id_index_part = _survivor_queue_id_index_part_path(base)
    id_index_ready = _survivor_queue_id_index_ready_path(base)
    part.parent.mkdir(parents=True, exist_ok=True)
    lock_path = base.with_suffix(base.suffix + ".writer.lock")
    lock_fd = _acquire_survivor_queue_writer_lock(lock_path)
    digest = hashlib.sha256()
    id_index_digest = hashlib.sha256()
    candidate_ids: set[str] = set()
    evidence_ids: set[str] = set()
    count = 0
    try:
        queue_fd = filtering_artifacts._create_private_partial(part)
        try:
            index_fd = filtering_artifacts._create_private_partial(id_index_part)
        except BaseException:
            os.close(queue_fd)
            raise
        with os.fdopen(queue_fd, "w", encoding="utf-8") as handle, os.fdopen(
            index_fd, "w", encoding="ascii"
        ) as id_handle:
            for row in rows:
                filtering_artifacts.validate_public_no_raw_payload(row)
                candidate_id = filtering_artifacts._require_sha256(row.get("candidate_id_hash"), "survivor queue candidate_id_hash")
                evidence_id = filtering_artifacts._require_sha256(row.get("private_evidence_row_id"), "survivor queue private_evidence_row_id")
                if candidate_id in candidate_ids:
                    raise ValueError("survivor queue contains duplicate candidate ids")
                if evidence_id in evidence_ids:
                    raise ValueError("survivor queue contains duplicate private evidence ids")
                candidate_ids.add(candidate_id)
                evidence_ids.add(evidence_id)
                line = json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
                id_line = f"{candidate_id}\t{evidence_id}\n"
                id_handle.write(id_line)
                id_index_digest.update(id_line.encode("ascii"))
                count += 1
            filtering_artifacts._flush_and_fsync(handle)
            filtering_artifacts._flush_and_fsync(id_handle)
        proof = _SurvivorQueueWriteProof(
            survivor_queue_hash=digest.hexdigest(),
            survivor_queue_row_count=count,
            survivor_queue_id_index_hash=id_index_digest.hexdigest(),
            survivor_queue_id_index_row_count=count,
            survivor_candidate_id_set_hash=stable_json_hash(sorted(candidate_ids)),
            survivor_private_evidence_id_set_hash=stable_json_hash(sorted(evidence_ids)),
        )
        live_queue = hashlib.sha256()
        live_index = hashlib.sha256()
        live_candidate_ids: set[str] = set()
        live_evidence_ids: set[str] = set()
        live_count = 0
        for live_row in _iter_live_rows(stage_state, config_payload):
            candidate_id = str(live_row["candidate_id_hash"])
            evidence_id = str(live_row["private_evidence_row_id"])
            live_candidate_ids.add(candidate_id)
            live_evidence_ids.add(evidence_id)
            live_queue.update(
                (json.dumps(dict(live_row), ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")
            )
            live_index.update(f"{candidate_id}\t{evidence_id}\n".encode("ascii"))
            live_count += 1
        if proof != _SurvivorQueueWriteProof(
            survivor_queue_hash=live_queue.hexdigest(),
            survivor_queue_row_count=live_count,
            survivor_queue_id_index_hash=live_index.hexdigest(),
            survivor_queue_id_index_row_count=live_count,
            survivor_candidate_id_set_hash=stable_json_hash(sorted(live_candidate_ids)),
            survivor_private_evidence_id_set_hash=stable_json_hash(sorted(live_evidence_ids)),
        ):
            raise ValueError("survivor queue does not match live stage state")
        if ready.exists() and id_index_ready.exists():
            if (
                filtering_artifacts._sha256_file(ready) != proof.survivor_queue_hash
                or filtering_artifacts._sha256_file(id_index_ready)
                != proof.survivor_queue_id_index_hash
            ):
                raise ValueError("completed survivor queue conflicts with requested rows")
            part.unlink()
            id_index_part.unlink()
            return proof
        filtering_artifacts._promote_private_partial(id_index_part, id_index_ready)
        filtering_artifacts._promote_private_partial(part, ready)
        return proof
    finally:
        os.close(lock_fd)

def _validate_survivor_queue_rows(
    rows: Sequence[Mapping[str, Any]],
    config_payload: Mapping[str, Any],
    *,
    allow_empty: bool = False,
) -> None:
    if not rows and not allow_empty:
        raise ValueError("survivor queue is empty or missing survivors")
    seen: set[str] = set()
    for row in rows:
        _validate_survivor_queue_row(row, config_payload, seen)

def _validate_survivor_queue_ready_file(
    base_path: str | Path,
    config_payload: Mapping[str, Any],
    *,
    allow_empty: bool = False,
    seen_survivor_candidate_ids: set[str] | None = None,
    seen_private_evidence_ids: set[str] | None = None,
) -> int:
    seen: set[str] = set()
    count = 0
    for row in filtering_artifacts._iter_ready_jsonl(base_path, "prefilter survivor queue"):
        _validate_survivor_queue_row(row, config_payload, seen)
        candidate_id = filtering_artifacts._require_sha256(row.get("candidate_id_hash"), "candidate_id_hash")
        evidence_id = filtering_artifacts._require_sha256(row.get("private_evidence_row_id"), "private_evidence_row_id")
        if seen_survivor_candidate_ids is not None:
            if candidate_id in seen_survivor_candidate_ids:
                raise ValueError("sharded W1 handoff contains duplicate survivor candidate ids")
            seen_survivor_candidate_ids.add(candidate_id)
        if seen_private_evidence_ids is not None:
            if evidence_id in seen_private_evidence_ids:
                raise ValueError("sharded W1 handoff contains duplicate private evidence ids")
            seen_private_evidence_ids.add(evidence_id)
        count += 1
    if count == 0 and not allow_empty:
        raise ValueError("survivor queue is empty or missing survivors")
    return count

def _validate_survivor_queue_id_index_ready_file(
    base_path: str | Path,
    *,
    allow_empty: bool = False,
    expected_hash: str | None = None,
    expected_row_count: int | None = None,
    expected_candidate_id_set_hash: str | None = None,
    expected_private_evidence_id_set_hash: str | None = None,
    seen_survivor_candidate_ids: set[str] | None = None,
    seen_private_evidence_ids: set[str] | None = None,
) -> int:
    part = _survivor_queue_id_index_part_path(base_path)
    ready = _survivor_queue_id_index_ready_path(base_path)
    if part.exists():
        raise ValueError("prefilter survivor queue ID index has untrusted .part export")
    if not ready.exists():
        raise ValueError("prefilter survivor queue ID index is missing ready export")
    if expected_hash is not None and filtering_artifacts._require_sha256(expected_hash, "survivor_queue_id_index_hash") != filtering_artifacts._sha256_file(ready):
        raise ValueError("survivor queue ID index hash mismatch")
    local_candidates: set[str] = set()
    local_evidence: set[str] = set()
    count = 0
    with ready.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(f"prefilter survivor queue ID index row {line_number} is malformed")
            candidate_id = filtering_artifacts._require_sha256(parts[0], "survivor queue ID index candidate_id_hash")
            evidence_id = filtering_artifacts._require_sha256(parts[1], "survivor queue ID index private_evidence_row_id")
            if candidate_id in local_candidates:
                raise ValueError("survivor queue ID index contains duplicate candidate ids")
            if evidence_id in local_evidence:
                raise ValueError("survivor queue ID index contains duplicate private evidence ids")
            local_candidates.add(candidate_id)
            local_evidence.add(evidence_id)
            if seen_survivor_candidate_ids is not None:
                if candidate_id in seen_survivor_candidate_ids:
                    raise ValueError("sharded W1 handoff contains duplicate survivor candidate ids")
                seen_survivor_candidate_ids.add(candidate_id)
            if seen_private_evidence_ids is not None:
                if evidence_id in seen_private_evidence_ids:
                    raise ValueError("sharded W1 handoff contains duplicate private evidence ids")
                seen_private_evidence_ids.add(evidence_id)
            count += 1
    if count == 0 and not allow_empty:
        raise ValueError("survivor queue ID index is empty or missing survivors")
    if expected_row_count is not None and count != int(expected_row_count):
        raise ValueError("survivor queue ID index row count mismatch")
    if (
        expected_candidate_id_set_hash is not None
        and filtering_artifacts._require_sha256(expected_candidate_id_set_hash, "survivor_candidate_id_set_hash")
        != stable_json_hash(sorted(local_candidates))
    ):
        raise ValueError("survivor queue ID index candidate set hash mismatch")
    if (
        expected_private_evidence_id_set_hash is not None
        and filtering_artifacts._require_sha256(expected_private_evidence_id_set_hash, "survivor_private_evidence_id_set_hash")
        != stable_json_hash(sorted(local_evidence))
    ):
        raise ValueError("survivor queue ID index private evidence set hash mismatch")
    return count

def _validate_survivor_queue_row(
    row: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    seen: set[str],
) -> None:
    filtering_artifacts.validate_public_no_raw_payload(row)
    candidate_id = filtering_artifacts._require_sha256(row.get("candidate_id_hash"), "candidate_id_hash")
    if candidate_id in seen:
        raise ValueError("survivor queue contains duplicate candidate ids")
    seen.add(candidate_id)
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "s1_report_hash",
        "s1_model_file_sha256",
        "w1_report_hash",
        "w1_model_file_sha256",
    ):
        expected = {
            "s1_report_hash": "sparse_report_hash",
            "s1_model_file_sha256": "sparse_model_file_sha256",
            "w1_report_hash": "window_mil_report_hash",
            "w1_model_file_sha256": "window_mil_model_file_sha256",
        }.get(field, field)
        if row.get(field) != config_payload.get(expected):
            raise ValueError(f"survivor queue {field} mismatch")
    if float(row.get("s1_threshold")) != float(config_payload["sparse_threshold"]):
        raise ValueError("survivor queue S1 threshold mismatch")
    if float(row.get("w1_threshold")) != float(config_payload["window_mil_threshold"]):
        raise ValueError("survivor queue W1 threshold mismatch")
    core_proxy_token_count = filtering_artifacts._nonnegative_int(row.get("core_proxy_token_count"), "survivor queue core_proxy_token_count")
    if row.get("evidence_proxy_token_count") is None:
        evidence_proxy_token_count = core_proxy_token_count
    else:
        evidence_proxy_token_count = filtering_artifacts._nonnegative_int(
            row.get("evidence_proxy_token_count"),
            "survivor queue evidence_proxy_token_count",
        )
        if evidence_proxy_token_count < core_proxy_token_count:
            raise ValueError("survivor queue evidence_proxy_token_count must cover core_proxy_token_count")
    expected_window_count = max(1, math.ceil(evidence_proxy_token_count / 512))
    if filtering_artifacts._positive_int(row.get("estimated_w1_window_count"), "survivor queue estimated_w1_window_count") != expected_window_count:
        raise ValueError("survivor queue estimated_w1_window_count mismatch")
    filtering_artifacts._require_sha256(row.get("private_evidence_row_id"), "private_evidence_row_id")
    filtering_artifacts._require_sha256(row.get("stage_receipt_hash"), "stage_receipt_hash")
    if self_hash_mismatches(row, "stage_receipt_hash"):
        raise ValueError("survivor queue row receipt hash mismatch")

def _prefilter_receipt_has_survivor_queue_id_proof(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("survivor_queue_proof_kind") != "jsonl_sha256_and_id_set_hashes_v1":
        return False
    return (
        receipt.get("survivor_queue_row_count") is not None
        and receipt.get("survivor_candidate_id_set_hash") is not None
        and receipt.get("survivor_private_evidence_id_set_hash") is not None
    )

def _prefilter_receipt_has_survivor_queue_id_index_proof(receipt: Mapping[str, Any]) -> bool:
    return (
        _prefilter_receipt_has_survivor_queue_id_proof(receipt)
        and receipt.get("survivor_queue_id_index_hash") is not None
        and receipt.get("survivor_queue_id_index_row_count") is not None
    )

def _validate_prefilter_ready_receipt_payload(
    receipt: Mapping[str, Any],
    run_path: Path,
    config_payload: Mapping[str, Any],
    *,
    queue_row_count: int | None = None,
    expected_status: str = "prefilter_ready_for_w1",
) -> None:
    filtering_artifacts.validate_public_no_raw_payload(receipt)
    if receipt.get("artifact_kind") != filtering_artifacts.PREFILTER_CPU_STAGE_KIND:
        raise ValueError("prefilter ready receipt kind mismatch")
    if expected_status not in {"prefilter_ready_for_w1", "prefilter_shard_ready_for_merge", "compact_fresh_import"}:
        raise ValueError("unsupported prefilter ready receipt status expectation")
    if receipt.get("status") != expected_status:
        raise ValueError("prefilter stage is not ready for the requested next phase")
    if self_hash_mismatches(receipt, "stage_receipt_hash"):
        raise ValueError("prefilter ready receipt hash mismatch")
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "selected_eval_deterministic_filter_policy_hash",
    ):
        if receipt.get(field) != config_payload.get(field):
            raise ValueError(f"prefilter ready receipt {field} mismatch")
    if receipt.get("executed_deterministic_filter_policy_hash") != config_payload.get("deterministic_filter_policy_hash"):
        raise ValueError("prefilter ready receipt deterministic policy mismatch")
    route_counts = {str(key): int(value) for key, value in dict(receipt.get("route_counts") or {}).items()}
    if int(receipt.get("candidate_count", -1)) != sum(route_counts.values()):
        raise ValueError("prefilter ready receipt candidate count does not match route_counts")
    if int(receipt.get("survivor_count", -1)) != int(route_counts.get("prefilter_survivor", 0)):
        raise ValueError("prefilter ready receipt survivor count does not match route_counts")
    if queue_row_count is not None and int(receipt.get("survivor_count", -1)) != int(queue_row_count):
        raise ValueError("prefilter ready receipt survivor count mismatch")
    if receipt.get("survivor_queue_ready") is not True or receipt.get("survivor_queue_part") is not False:
        raise ValueError("prefilter ready receipt does not prove a ready survivor queue")
    ready_queue_path = run_path / "w1_survivor_queue.jsonl.ready"
    if receipt.get("survivor_queue_hash") != filtering_artifacts._sha256_file(ready_queue_path):
        raise ValueError("prefilter ready receipt survivor queue hash mismatch")
    if _prefilter_receipt_has_survivor_queue_id_proof(receipt):
        if int(receipt.get("survivor_queue_row_count", -1)) != int(receipt.get("survivor_count", -2)):
            raise ValueError("prefilter ready receipt survivor queue row count mismatch")
        filtering_artifacts._require_sha256(receipt.get("survivor_candidate_id_set_hash"), "survivor_candidate_id_set_hash")
        filtering_artifacts._require_sha256(receipt.get("survivor_private_evidence_id_set_hash"), "survivor_private_evidence_id_set_hash")
        if _prefilter_receipt_has_survivor_queue_id_index_proof(receipt):
            ready_index_path = _survivor_queue_id_index_ready_path(run_path / "w1_survivor_queue.jsonl")
            if receipt.get("survivor_queue_id_index_hash") != filtering_artifacts._sha256_file(ready_index_path):
                raise ValueError("prefilter ready receipt survivor queue ID index hash mismatch")
            if int(receipt.get("survivor_queue_id_index_row_count", -1)) != int(receipt.get("survivor_count", -2)):
                raise ValueError("prefilter ready receipt survivor queue ID index row count mismatch")

def _validate_prefilter_ready_receipt(
    conn: stage_state_v1.FilteringStageStateV1,
    run_path: Path,
    config_payload: Mapping[str, Any],
    *,
    queue_rows: Sequence[Mapping[str, Any]] | None = None,
    queue_row_count: int | None = None,
    expected_status: str = "prefilter_ready_for_w1",
    require_compact_w1_scores_empty: bool = True,
) -> None:
    receipt_path = run_path / "prefilter_cpu_receipt.json"
    if not receipt_path.exists():
        raise ValueError("prefilter ready receipt is missing")
    receipt = filtering_artifacts._load_json(receipt_path)
    if not isinstance(receipt, Mapping):
        raise ValueError("prefilter ready receipt must be an object")
    if queue_rows is None and queue_row_count is None:
        raise ValueError("prefilter ready receipt validation requires survivor queue evidence")
    survivor_queue_count = len(queue_rows) if queue_rows is not None else int(queue_row_count or 0)
    _validate_prefilter_ready_receipt_payload(
        receipt,
        run_path,
        config_payload,
        queue_row_count=survivor_queue_count,
        expected_status=expected_status,
    )
    if _is_w1_compact_prefilter_receipt(receipt):
        if queue_rows is None:
            raise ValueError("compact start state validation requires survivor queue rows")
        _validate_w1_compact_ready_db(
            conn,
            run_path,
            config_payload,
            receipt=receipt,
            queue_rows=queue_rows,
            require_empty_w1_scores=require_compact_w1_scores_empty,
        )
        return
    route_counts = conn.candidate_route_counts()
    candidate_counts = conn.candidate_counts()
    if int(receipt.get("candidate_count", -1)) != int(candidate_counts["candidate_count"]):
        raise ValueError("prefilter ready receipt candidate count mismatch")
    if int(receipt.get("survivor_count", -1)) != int(candidate_counts["survivor_count"]):
        raise ValueError("prefilter ready receipt DB survivor count mismatch")
    if dict(receipt.get("route_counts") or {}) != route_counts:
        raise ValueError("prefilter ready receipt route_counts mismatch")
    row_counts = conn.prefilter_summary_facts()
    if receipt.get("input_schedule_hash") != row_counts["input_schedule_hash"]:
        raise ValueError("prefilter ready receipt input schedule hash mismatch")
    for field in (
        "expected_physical_rows",
        "expected_physical_row_key_set_hash",
        "scheduled_rows",
        "scheduled_physical_row_key_set_hash",
        "terminal_rows",
        "terminal_physical_row_key_set_hash",
        "pending_rows",
        "pending_physical_row_key_set_hash",
        "fetched_rows",
        "segmented_rows",
        "row_error_counts",
        "unaccounted_row_count",
    ):
        if receipt.get(field) != row_counts[field]:
            raise ValueError(f"prefilter ready receipt {field} mismatch")
    accounted_rows = int(row_counts["terminal_rows"]) + int(row_counts["pending_rows"])
    unvisited_rows = int(row_counts["expected_physical_rows"]) - accounted_rows
    if int(receipt.get("unvisited_rows", -1)) != unvisited_rows:
        raise ValueError("prefilter ready receipt unvisited row mismatch")
    if int(row_counts["pending_rows"]) != 0 or int(row_counts["unaccounted_row_count"]) != 0 or unvisited_rows != 0:
        raise ValueError("prefilter ready receipt does not prove all expected rows reached terminal state")
    if (
        row_counts["expected_physical_row_key_set_hash"] is not None
        and row_counts["terminal_physical_row_key_set_hash"] != row_counts["expected_physical_row_key_set_hash"]
    ):
        raise ValueError("prefilter ready receipt physical row key-set mismatch")

def _directory_size(path: str | Path) -> int:
    root = Path(path)
    total = 0
    if not root.exists():
        return 0
    for item in root.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total

def _w1_compact_forbidden_fresh_files(run_path: Path) -> list[str]:
    forbidden = []
    for name in (
        "STOP_REQUESTED",
        "w1_score_receipt.json",
        "w1_scores.jsonl.ready",
        "w1_scores.jsonl.part",
    ):
        if (run_path / name).exists():
            forbidden.append(name)
    return forbidden

def _validate_w1_compact_receipt_proof_payload(
    receipt: Mapping[str, Any],
    run_path: Path,
    config_payload: Mapping[str, Any],
    *,
    queue_row_count: int | None = None,
    expected_status: str = "compact_fresh_import",
    require_empty_w1_scores: bool = False,
) -> None:
    """Validate compact W1-start receipt proof without rereading full private state.

    Compact export already performs queue/SQLite/private-evidence validation while
    building each shard. Global handoff and launch preflight can rely on the
    compact receipt self-hash plus the compact ID-index proof; shard scoring
    still performs deep queue/SQLite validation before reading private evidence.
    """
    filtering_artifacts.validate_public_no_raw_payload(receipt)
    if receipt.get("artifact_kind") != filtering_artifacts.PREFILTER_CPU_STAGE_KIND:
        raise ValueError("compact start receipt kind mismatch")
    if receipt.get("status") != expected_status:
        raise ValueError("compact start receipt status mismatch")
    if receipt.get("compact_start_state_artifact_kind") != filtering_artifacts.W1_COMPACT_START_STATE_KIND:
        raise ValueError("compact start receipt artifact kind mismatch")
    if receipt.get("compact_start_state_status") != "compact_fresh_import":
        raise ValueError("compact start state status mismatch")
    if self_hash_mismatches(receipt, "stage_receipt_hash"):
        raise ValueError("compact start receipt hash mismatch")
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "selected_eval_deterministic_filter_policy_hash",
    ):
        if receipt.get(field) != config_payload.get(field):
            raise ValueError(f"compact start receipt {field} mismatch")
    if receipt.get("executed_deterministic_filter_policy_hash") != config_payload.get("deterministic_filter_policy_hash"):
        raise ValueError("compact start receipt deterministic policy mismatch")
    route_counts = {str(key): int(value) for key, value in dict(receipt.get("route_counts") or {}).items()}
    survivor_count = int(receipt.get("survivor_count", -1))
    if int(receipt.get("candidate_count", -1)) != sum(route_counts.values()):
        raise ValueError("compact start receipt candidate count does not match route_counts")
    if survivor_count != int(route_counts.get("prefilter_survivor", 0)):
        raise ValueError("compact start receipt survivor count does not match route_counts")
    if queue_row_count is not None and survivor_count != int(queue_row_count):
        raise ValueError("compact start receipt survivor count mismatch")
    if int(receipt.get("compact_candidate_count", -1)) != survivor_count:
        raise ValueError("compact start receipt compact candidate count mismatch")
    if int(receipt.get("compact_private_evidence_count", -1)) != survivor_count:
        raise ValueError("compact start receipt compact private evidence count mismatch")
    if bool(require_empty_w1_scores) and int(receipt.get("compact_w1_scores_count", -1)) != 0:
        raise ValueError("compact start receipt must prove empty w1_scores")
    if receipt.get("compact_db_contains_only_survivor_prefilter_candidates") is not True:
        raise ValueError("compact start receipt survivor-only DB flag mismatch")
    if receipt.get("compact_prefilter_source_rows_omitted") is not True:
        raise ValueError("compact start receipt source-row omission flag mismatch")
    if receipt.get("survivor_queue_ready") is not True or receipt.get("survivor_queue_part") is not False:
        raise ValueError("compact start receipt does not prove a ready survivor queue")
    if require_empty_w1_scores and _w1_compact_forbidden_fresh_files(run_path):
        raise ValueError("compact start state contains stale W1 output or STOP_REQUESTED")
    ready_queue_path = run_path / "w1_survivor_queue.jsonl.ready"
    if not ready_queue_path.exists():
        raise ValueError("compact start survivor queue is missing")
    if not _prefilter_receipt_has_survivor_queue_id_index_proof(receipt):
        raise ValueError("compact start receipt requires survivor queue ID-index proof")
    ready_index_path = _survivor_queue_id_index_ready_path(run_path / "w1_survivor_queue.jsonl")
    if receipt.get("survivor_queue_id_index_hash") != filtering_artifacts._sha256_file(ready_index_path):
        raise ValueError("compact start receipt survivor queue ID index hash mismatch")
    if int(receipt.get("survivor_queue_id_index_row_count", -1)) != survivor_count:
        raise ValueError("compact start receipt survivor queue ID index row count mismatch")
    if int(receipt.get("survivor_queue_row_count", -1)) != survivor_count:
        raise ValueError("compact start receipt survivor queue row count mismatch")
    for field in (
        "source_prefilter_stage_receipt_hash",
        "compact_sqlite_sha256",
        "survivor_queue_hash",
        "survivor_candidate_id_set_hash",
        "survivor_private_evidence_id_set_hash",
    ):
        filtering_artifacts._require_sha256(receipt.get(field), field)
