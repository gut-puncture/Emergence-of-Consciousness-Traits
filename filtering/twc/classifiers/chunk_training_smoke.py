"""Local smoke checks for chunk classifier private-text and feature paths.

This is not the final classifier trainer. It verifies the production-facing
inputs that a real trainer will depend on: frozen no-raw chunk manifests, private
stage maps, source-schedule refetch, core/halo extraction, sparse feature
construction, window/MIL pooling shape, thresholding, checkpoint round-trip, and
no-raw reporting.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from twc.classifiers.active_learning_private_maps import load_active_learning_private_index
from twc.classifiers.chunk_dataset import validate_chunk_classifier_training_manifest
from twc.classifiers.feature_extraction import hashed_ngram_feature_counts
from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.data.chunk_relabeling import (
    CHUNK_PRIVATE_MAP_SCHEMA_VERSION,
    CHUNK_RELABEL_POLICY_VERSION,
    CHUNK_RELABEL_PRIVATE_MAP_VERSION,
    _private_map_hash,
    validate_chunk_private_map,
    validate_chunk_relabel_manifest,
)
from twc.data.decision_log import (
    DEFAULT_HF_ROW_READ_BLOCK_SIZE_BYTES,
    HFFsspecParquetRowProvider,
)
from twc.data.decision_schema import stable_json_hash
from twc.data.teacher_labeling import _private_refetch_text_hash


CHUNK_LOCAL_SMOKE_REPORT_KIND = "twc_chunk_classifier_local_smoke_report"
CHUNK_LOCAL_SMOKE_CHECKPOINT_KIND = "twc_chunk_classifier_local_smoke_checkpoint"
SPARSE_SMOKE_MODEL_ID = "chunk_hashed_ngram_centroid_smoke"
WINDOW_SMOKE_MODEL_ID = "chunk_window_mil_hash_proxy_smoke"
DEFAULT_HALO_CHARS = 512
DEFAULT_MAX_ROWS = 24
DEFAULT_N_FEATURES = 2**12


def build_chunk_classifier_local_smoke_report(
    *,
    classifier_manifest: Mapping[str, Any],
    stage_dirs: Iterable[str | Path],
    active_learning_private_maps: Iterable[str | Path] = (),
    source_schedule: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    row_provider: Any,
    max_rows: int = DEFAULT_MAX_ROWS,
    halo_chars: int = DEFAULT_HALO_CHARS,
    n_features: int = DEFAULT_N_FEATURES,
    target_recall: float = 0.995,
    real_entrypoint_command: Sequence[str] | None = None,
    output_checkpoint_json: str | Path | None = None,
    require_active_learning_batches: Iterable[str] = (),
) -> dict[str, Any]:
    """Run a bounded local smoke over private chunk core+halo text.

    Raw text is kept in local variables only. The returned report and optional
    checkpoint are public/no-raw metadata and hashes.
    """

    validate_chunk_classifier_training_manifest(classifier_manifest)
    validate_public_no_raw_payload(classifier_manifest)
    if max_rows < 2:
        raise ValueError("max_rows must be at least 2")
    if halo_chars < 0:
        raise ValueError("halo_chars must be nonnegative")
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")

    stage_dirs = list(stage_dirs)
    stage_index = _load_stage_private_index(stage_dirs) if stage_dirs else {}
    active_learning_index = load_active_learning_private_index(
        active_learning_private_maps,
        classifier_manifest=classifier_manifest,
    )
    if _manifest_has_active_learning_rows(classifier_manifest) and not active_learning_index:
        raise ValueError("chunk local smoke missing active-learning private map")
    private_index = _merge_private_indexes(stage_index, active_learning_index)
    schedule_hash, schedule_by_entry, template_by_source_file = _load_source_schedule_index(source_schedule)
    _augment_source_schedule_index_from_private_maps(
        schedule_by_entry,
        template_by_source_file=template_by_source_file,
        private_index=private_index,
    )
    selected_rows = _select_smoke_rows(
        classifier_manifest=classifier_manifest,
        stage_index=private_index,
        schedule_by_entry=schedule_by_entry,
        max_rows=max_rows,
        required_active_batches={str(batch) for batch in require_active_learning_batches},
    )
    if _manifest_has_active_learning_rows(classifier_manifest) and not any(
        str(row.get("evidence_tier") or "") == "active_learning_bof3_chunk_label"
        for row in selected_rows
    ):
        raise ValueError("chunk local smoke must exercise active-learning private rows")
    extracted = _fetch_core_halo_texts(
        selected_rows,
        stage_index=private_index,
        schedule_by_entry=schedule_by_entry,
        row_provider=row_provider,
        halo_chars=halo_chars,
    )

    sparse_result = _run_sparse_smoke(
        extracted,
        n_features=n_features,
        target_recall=target_recall,
    )
    window_result = _run_window_mil_proxy_smoke(extracted)
    checkpoint = _smoke_checkpoint_payload(
        classifier_manifest_hash=str(classifier_manifest["training_manifest_hash"]),
        source_schedule_hash=schedule_hash,
        max_rows=max_rows,
        halo_chars=halo_chars,
        n_features=n_features,
        sparse_result=sparse_result,
        window_result=window_result,
    )
    checkpoint_path_hash = None
    checkpoint_file_sha256 = None
    if output_checkpoint_json is not None:
        checkpoint_path = Path(output_checkpoint_json)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        checkpoint_path_hash = stable_json_hash({"checkpoint_path": str(checkpoint_path)})
        checkpoint_file_sha256 = _sha256_file(checkpoint_path)
    reloaded_checkpoint = json.loads(json.dumps(checkpoint, ensure_ascii=True, sort_keys=True))
    checkpoint_roundtrip_passed = reloaded_checkpoint == checkpoint

    label_counts = Counter(item["classifier_label"] for item in extracted)
    split_counts = Counter(item["split"] for item in extracted)
    evidence_tier_counts = Counter(str(row.get("evidence_tier") or "") for row in selected_rows)
    private_route_counts = Counter(str(row.get("private_cache_route") or "") for row in selected_rows)
    active_learning_batch_counts = Counter(
        str(row.get("active_learning_batch_id") or "")
        for row in selected_rows
        if str(row.get("evidence_tier") or "") == "active_learning_bof3_chunk_label"
    )
    report = {
        "artifact_kind": CHUNK_LOCAL_SMOKE_REPORT_KIND,
        "schema_version": 1,
        "status": "passed",
        "classifier_manifest_hash": str(classifier_manifest["training_manifest_hash"]),
        "source_schedule_hash": schedule_hash,
        "selected_row_count": len(extracted),
        "requested_max_rows": int(max_rows),
        "label_counts": _sorted_counter(label_counts),
        "split_counts": _sorted_counter(split_counts),
        "evidence_tier_counts": _sorted_counter(evidence_tier_counts),
        "private_cache_route_counts": _sorted_counter(private_route_counts),
        "active_learning_batch_counts": _sorted_counter(active_learning_batch_counts),
        "required_active_learning_batches": sorted(str(batch) for batch in require_active_learning_batches),
        "halo_chars": int(halo_chars),
        "n_features": int(n_features),
        "core_text_hashes_validated": True,
        "parent_text_hashes_validated": True,
        "core_halo_extraction_validated": True,
        "private_text_fetched": True,
        "sparse_smoke": sparse_result["public"],
        "window_mil_proxy_smoke": window_result["public"],
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "checkpoint_path_hash": checkpoint_path_hash,
        "checkpoint_roundtrip_passed": checkpoint_roundtrip_passed,
        "real_entrypoint_command_hash": None if real_entrypoint_command is None else stable_json_hash([str(part) for part in real_entrypoint_command]),
        "prime_gpu_training_decision": {
            "status": "blocked",
            "paid_action_recommended_now": False,
            "blocking_reasons": [
                "final_chunk_semantic_training_runner_not_yet_frozen",
                "frozen_chunk_classifier_model_absent",
            ],
            "local_private_text_smoke_passed": True,
            "gpu_rental_performed": False,
            "paid_action_performed": False,
        },
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "paid_api_used": False,
        "gpu_rental_used": False,
    }
    report["report_hash"] = stable_json_hash(report)
    validate_chunk_classifier_local_smoke_report(report)
    return report


def validate_chunk_classifier_local_smoke_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_LOCAL_SMOKE_REPORT_KIND:
        raise ValueError("unsupported chunk classifier local smoke report kind")
    if report.get("status") != "passed":
        raise ValueError("chunk classifier local smoke report must pass")
    for flag in (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "accepted_manifest_created",
        "tokenizer_shards_created",
        "training_shards_created",
        "classifier_training_performed",
        "base_pretraining_started",
        "pretraining_started",
        "paid_api_used",
        "gpu_rental_used",
    ):
        if report.get(flag) is not False:
            raise ValueError(f"chunk classifier local smoke report must keep {flag}=false")
    if report.get("private_text_fetched") is not True:
        raise ValueError("chunk classifier local smoke must fetch private text")
    if report.get("core_halo_extraction_validated") is not True:
        raise ValueError("chunk classifier local smoke must validate core/halo extraction")
    if report.get("checkpoint_roundtrip_passed") is not True:
        raise ValueError("chunk classifier local smoke checkpoint roundtrip failed")
    active_batch_counts = dict(report.get("active_learning_batch_counts") or {})
    for batch_id in report.get("required_active_learning_batches") or []:
        if int(active_batch_counts.get(str(batch_id), 0)) < 1:
            raise ValueError("chunk classifier local smoke missing required active-learning batch")
    decision = report.get("prime_gpu_training_decision")
    if not isinstance(decision, Mapping):
        raise ValueError("chunk classifier local smoke report requires Prime decision")
    if decision.get("paid_action_recommended_now") is not False:
        raise ValueError("local smoke report cannot recommend immediate paid Prime action")
    if report.get("report_hash") != stable_json_hash({key: value for key, value in report.items() if key != "report_hash"}):
        raise ValueError("chunk classifier local smoke report hash mismatch")


def _load_stage_private_index(stage_dirs: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    seen_stage_paths: set[str] = set()
    for stage_dir in sorted((Path(path) for path in stage_dirs), key=lambda item: str(item)):
        stage_key = str(stage_dir.resolve(strict=False))
        if stage_key in seen_stage_paths:
            raise ValueError("duplicate chunk stage directory")
        seen_stage_paths.add(stage_key)
        manifest_path = stage_dir / "chunk_relabel_manifest.json"
        private_map_path = stage_dir / "chunk_relabel_private_map.json"
        if not manifest_path.exists() or not private_map_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        private_map = json.loads(private_map_path.read_text(encoding="utf-8"))
        _validate_stage_manifest_for_smoke(manifest, stage_dir=stage_dir)
        _validate_private_map_for_smoke(private_map, manifest=manifest, stage_dir=stage_dir)
        private_by_candidate = {str(item["candidate_id"]): item for item in private_map["chunks"]}
        for candidate in manifest["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            private_chunk = private_by_candidate.get(candidate_id)
            if private_chunk is None:
                raise ValueError("chunk private map missing candidate")
            if candidate_id in index:
                raise ValueError("duplicate chunk candidate across stage directories")
            index[candidate_id] = {
                "candidate": candidate,
                "private_chunk": private_chunk,
                "stage_manifest_hash": manifest["manifest_hash"],
                "private_map_hash": private_map["private_map_hash"],
            }
    if not index:
        raise ValueError("no chunk stage private maps found")
    return index


def _merge_private_indexes(
    *indexes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    merged: dict[str, Mapping[str, Any]] = {}
    for source in indexes:
        for candidate_id, bundle in source.items():
            if candidate_id in merged:
                raise ValueError("duplicate private smoke candidate across routes")
            merged[candidate_id] = bundle
    return merged


def _manifest_has_active_learning_rows(classifier_manifest: Mapping[str, Any]) -> bool:
    return any(
        row.get("fit_eligible") is not False
        and str(row.get("split") or "") != "audit"
        and str(row.get("evidence_tier") or "") == "active_learning_bof3_chunk_label"
        for row in classifier_manifest.get("rows") or []
    )


def _validate_stage_manifest_for_smoke(manifest: dict[str, Any], *, stage_dir: Path) -> None:
    try:
        validate_chunk_relabel_manifest(manifest)
    except ValueError as exc:
        if "parent_document_skip_count" not in str(exc) or not _legacy_missing_parent_skip_allowed(manifest, stage_dir=stage_dir):
            raise


def _validate_private_map_for_smoke(private_map: dict[str, Any], *, manifest: dict[str, Any], stage_dir: Path) -> None:
    try:
        validate_chunk_private_map(private_map, manifest=manifest)
        return
    except ValueError as exc:
        if "parent_document_skip_count" not in str(exc) or not _legacy_missing_parent_skip_allowed(manifest, stage_dir=stage_dir):
            raise
    if private_map.get("artifact_kind") != "chunk_relabel_private_map":
        raise ValueError("chunk private map artifact kind mismatch")
    if private_map.get("schema_version") != CHUNK_PRIVATE_MAP_SCHEMA_VERSION:
        raise ValueError("chunk private map schema version mismatch")
    if private_map.get("private_map_version") != CHUNK_RELABEL_PRIVATE_MAP_VERSION:
        raise ValueError("chunk private map version mismatch")
    if private_map.get("policy_version") != CHUNK_RELABEL_POLICY_VERSION:
        raise ValueError("chunk private map policy version mismatch")
    if private_map.get("manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("private chunk map manifest hash mismatch")
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if private_map.get(field) is not False:
            raise ValueError(f"private chunk map must keep {field}=false")
    if str(private_map.get("private_map_hash") or "") != _private_map_hash(private_map):
        raise ValueError("private chunk map hash mismatch")
    if private_map["private_map_hash"] != manifest["private_map_hash"]:
        raise ValueError("private chunk map hash does not match manifest")
    chunks = private_map.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("chunk private map chunks must be a list")
    if int(private_map.get("chunk_count") or -1) != len(chunks):
        raise ValueError("chunk private map chunk_count mismatch")
    by_candidate = {candidate["candidate_id"]: candidate for candidate in manifest["candidates"]}
    if set(by_candidate) != {str(chunk.get("candidate_id") or "") for chunk in chunks}:
        raise ValueError("chunk private map candidate coverage mismatch")
    for chunk in chunks:
        candidate = by_candidate[str(chunk["candidate_id"])]
        if chunk.get("parent_candidate_id") != candidate["parent_candidate_id"]:
            raise ValueError("chunk private map parent candidate mismatch")
        if chunk.get("chunk_text_sha256") != candidate["text_hash_sha256"]:
            raise ValueError("chunk private map text hash mismatch")
        if not chunk.get("parent_text_hash_algorithm"):
            raise ValueError("chunk private map parent text hash algorithm missing")
        for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
            if chunk.get(field) is not False:
                raise ValueError(f"private chunk map chunk must keep {field}=false")


def _legacy_missing_parent_skip_allowed(manifest: Mapping[str, Any], *, stage_dir: Path) -> bool:
    return (
        "parent_document_skip_count" not in manifest
        and "stage0000_0100" in stage_dir.name
        and int(manifest.get("parent_document_count") or -1) == 100
        and int(manifest.get("chunk_count") or -1) >= 1
    )


def _load_source_schedule_index(
    source_schedule: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    schedules: list[Mapping[str, Any]]
    if isinstance(source_schedule, Mapping):
        schedules = [source_schedule]
    else:
        schedules = list(source_schedule)
    if not schedules:
        raise ValueError("at least one source schedule is required")
    schedule_hashes: list[str] = []
    by_entry: dict[str, Mapping[str, Any]] = {}
    template_by_source_file: dict[str, Mapping[str, Any]] = {}
    for schedule in schedules:
        schedule_hash = _required_hash(schedule.get("schedule_hash"), "source schedule hash")
        schedule_hashes.append(schedule_hash)
        entries = schedule.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError("source schedule must contain entries")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("source schedule entry must be an object")
            entry_hash = _required_hash(entry.get("entry_hash"), "schedule entry hash")
            existing = by_entry.get(entry_hash)
            if existing is not None and stable_json_hash(existing) != stable_json_hash(entry):
                raise ValueError("conflicting duplicate source schedule entry_hash")
            by_entry[entry_hash] = entry
            source_file_id = str(entry.get("source_file_id") or "")
            if source_file_id and source_file_id not in template_by_source_file:
                template_by_source_file[source_file_id] = entry
    combined_hash = (
        schedule_hashes[0]
        if len(schedule_hashes) == 1
        else stable_json_hash({"source_schedule_hashes": sorted(schedule_hashes), "source_schedule_count": len(schedule_hashes)})
    )
    return combined_hash, by_entry, template_by_source_file


def _augment_source_schedule_index_from_private_maps(
    schedule_by_entry: dict[str, Mapping[str, Any]],
    *,
    template_by_source_file: Mapping[str, Mapping[str, Any]],
    private_index: Mapping[str, Mapping[str, Any]],
) -> None:
    for bundle in private_index.values():
        candidate = bundle.get("candidate")
        private_chunk = bundle.get("private_chunk")
        if not isinstance(candidate, Mapping) or not isinstance(private_chunk, Mapping):
            continue
        entry_hash = str(candidate.get("schedule_entry_hash") or private_chunk.get("schedule_entry_hash") or "")
        if not entry_hash:
            continue
        source_file_id = str(candidate.get("source_file_id") or private_chunk.get("source_file_id") or "")
        row_group_index = int(candidate.get("row_group_index", private_chunk.get("row_group_index", -1)))
        offset = int(candidate.get("row_group_local_row_index", private_chunk.get("row_group_local_row_index", -1)))
        existing = schedule_by_entry.get(entry_hash)
        if existing is not None:
            _validate_smoke_source_entry(existing, source_file_id=source_file_id, row_group_index=row_group_index, offset=offset)
            continue
        template = template_by_source_file.get(source_file_id)
        if template is None:
            raise ValueError("missing source schedule template for active-learning smoke row")
        derived = dict(template)
        derived["entry_hash"] = entry_hash
        derived["source_file_id"] = source_file_id
        derived["row_group_index"] = row_group_index
        derived["row_group_local_offsets"] = [offset]
        derived.pop("row_group_num_rows", None)
        derived["derived_from_same_source_file_template"] = True
        _validate_smoke_source_entry(derived, source_file_id=source_file_id, row_group_index=row_group_index, offset=offset)
        schedule_by_entry[entry_hash] = derived


def _validate_smoke_source_entry(
    entry: Mapping[str, Any],
    *,
    source_file_id: str,
    row_group_index: int,
    offset: int,
) -> None:
    if str(entry.get("source_file_id") or "") != source_file_id:
        raise ValueError("chunk local smoke source schedule source_file_id drift")
    if int(entry.get("row_group_index", -1)) != int(row_group_index):
        raise ValueError("chunk local smoke source schedule row group drift")
    offsets = entry.get("row_group_local_offsets")
    if isinstance(offsets, Sequence) and not isinstance(offsets, (str, bytes)):
        if int(offset) not in {int(value) for value in offsets}:
            raise ValueError("chunk local smoke source schedule offset drift")


def _select_smoke_rows(
    *,
    classifier_manifest: Mapping[str, Any],
    stage_index: Mapping[str, Mapping[str, Any]],
    schedule_by_entry: Mapping[str, Mapping[str, Any]],
    max_rows: int,
    required_active_batches: set[str] | None = None,
) -> list[Mapping[str, Any]]:
    required_active_batches = set(required_active_batches or set())
    candidates_by_id = {
        candidate_id: bundle["candidate"]
        for candidate_id, bundle in stage_index.items()
    }
    available: list[tuple[tuple[str, int, str, str], Mapping[str, Any]]] = []
    for row in classifier_manifest.get("rows") or []:
        if row.get("fit_eligible") is False or str(row.get("split")) == "audit":
            continue
        candidate_id = str(row.get("candidate_id") or "")
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None:
            continue
        if str(candidate.get("schedule_entry_hash") or "") not in schedule_by_entry:
            continue
        key = (
            str(candidate.get("source_file_id") or ""),
            int(candidate.get("row_group_index") or 0),
            str(row.get("classifier_label") or ""),
            candidate_id,
        )
        available.append((key, row))
    if len(available) < 2:
        raise ValueError("not enough eligible chunk rows for local smoke")
    by_label: dict[str, list[tuple[tuple[str, int, str, str], Mapping[str, Any]]]] = defaultdict(list)
    for item in sorted(available, key=lambda item: item[0]):
        by_label[str(item[1]["classifier_label"])].append(item)
    if not by_label.get("safe") or not by_label.get("reject"):
        raise ValueError("chunk local smoke requires both safe and reject labels")

    selected: list[Mapping[str, Any]] = []
    used: set[str] = set()
    for batch_id in sorted(required_active_batches):
        for _key, row in sorted(available, key=lambda item: item[0]):
            candidate_id = str(row["candidate_id"])
            if candidate_id in used:
                continue
            if str(row.get("active_learning_batch_id") or "") != batch_id:
                continue
            selected.append(row)
            used.add(candidate_id)
            break
        else:
            raise ValueError(f"chunk local smoke missing required active-learning batch: {batch_id}")
    if len(selected) > max_rows:
        raise ValueError("chunk local smoke max_rows too small for required active-learning batches")
    label_order = ["reject", "safe"]
    while len(selected) < max_rows:
        added = False
        for label in label_order:
            while by_label[label] and str(by_label[label][0][1]["candidate_id"]) in used:
                by_label[label].pop(0)
            if not by_label[label]:
                continue
            _key, row = by_label[label].pop(0)
            selected.append(row)
            used.add(str(row["candidate_id"]))
            added = True
            if len(selected) >= max_rows:
                break
        if not added:
            break
    if len({str(row["classifier_label"]) for row in selected}) < 2:
        raise ValueError("chunk local smoke selected rows lost label coverage")
    return selected


def _fetch_core_halo_texts(
    selected_rows: Sequence[Mapping[str, Any]],
    *,
    stage_index: Mapping[str, Mapping[str, Any]],
    schedule_by_entry: Mapping[str, Mapping[str, Any]],
    row_provider: Any,
    halo_chars: int,
) -> list[dict[str, Any]]:
    grouped_offsets: dict[tuple[str, int], set[int]] = defaultdict(set)
    entry_by_group: dict[tuple[str, int], Mapping[str, Any]] = {}
    selected_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    for row in selected_rows:
        candidate_id = str(row["candidate_id"])
        bundle = stage_index[candidate_id]
        candidate = bundle["candidate"]
        private_chunk = bundle["private_chunk"]
        entry = schedule_by_entry[str(candidate["schedule_entry_hash"])]
        group_key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
        grouped_offsets[group_key].add(int(candidate["row_group_local_row_index"]))
        entry_by_group[group_key] = entry
        selected_pairs.append((row, candidate, private_chunk, entry))

    fetched: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    fetch_inputs = [
        (key, entry_by_group[key], sorted(offsets))
        for key, offsets in sorted(grouped_offsets.items())
    ]
    if hasattr(row_provider, "fetch_row_groups"):
        provider_results = row_provider.fetch_row_groups([(dict(entry), list(offsets)) for _key, entry, offsets in fetch_inputs])
        for key, _entry, offsets in fetch_inputs:
            rows_by_offset = provider_results[(key[0], key[1])]
            for offset in offsets:
                fetched[(key[0], key[1], offset)] = rows_by_offset[offset]
    else:
        for key, entry, offsets in fetch_inputs:
            rows_by_offset = row_provider.fetch_rows(dict(entry), list(offsets))
            for offset in offsets:
                fetched[(key[0], key[1], offset)] = rows_by_offset[offset]

    extracted: list[dict[str, Any]] = []
    for row, candidate, private_chunk, entry in selected_pairs:
        key = (str(entry["source_file_id"]), int(entry["row_group_index"]), int(candidate["row_group_local_row_index"]))
        source_row = fetched[key]
        if isinstance(source_row, BaseException):
            raise ValueError(f"private chunk refetch failed: {type(source_row).__name__}")
        if not isinstance(source_row, Mapping):
            raise ValueError("private chunk refetch returned non-object row")
        parent_text = source_row.get("text")
        if not isinstance(parent_text, str) or not parent_text:
            raise ValueError("private chunk refetch returned missing parent text")
        parent_probe = {
            "text_hash_sha256": private_chunk.get("parent_text_hash_sha256"),
            "text_hash_algorithm": private_chunk.get("parent_text_hash_algorithm"),
        }
        parent_hash, _parent_hash_algorithm, had_expected_parent_hash = _private_refetch_text_hash(parent_probe, parent_text)
        expected_parent_hash = str(private_chunk.get("parent_text_hash_sha256") or "")
        if had_expected_parent_hash and parent_hash != expected_parent_hash:
            raise ValueError("private chunk parent text hash drift")
        start = int(private_chunk["raw_char_start"])
        end = int(private_chunk["raw_char_end"])
        if start < 0 or end <= start or end > len(parent_text):
            raise ValueError("private chunk span outside parent text")
        core_text = parent_text[start:end]
        if _sha256_text(core_text) != str(private_chunk["chunk_text_sha256"]):
            raise ValueError("private chunk core text hash drift")
        if _sha256_text(core_text) != str(row["text_hash_sha256"]):
            raise ValueError("classifier manifest chunk text hash drift")
        if "evidence_char_start" in private_chunk or "evidence_char_end" in private_chunk:
            halo_start = int(private_chunk["evidence_char_start"])
            halo_end = int(private_chunk["evidence_char_end"])
            if halo_start < 0 or halo_end <= halo_start or halo_end > len(parent_text):
                raise ValueError("private chunk evidence span outside parent text")
        else:
            halo_start = max(0, start - int(halo_chars))
            halo_end = min(len(parent_text), end + int(halo_chars))
        evidence_text = parent_text[halo_start:halo_end]
        if private_chunk.get("evidence_text_sha256") and _sha256_text(evidence_text) != str(private_chunk["evidence_text_sha256"]):
            raise ValueError("private chunk evidence text hash drift")
        if row.get("teacher_input_text_hash_sha256") and _sha256_text(evidence_text) != str(row["teacher_input_text_hash_sha256"]):
            raise ValueError("classifier manifest teacher-input text hash drift")
        extracted.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "classifier_label": str(row["classifier_label"]),
                "split": str(row["split"]),
                "core_text": core_text,
                "evidence_text": evidence_text,
                "core_text_sha256": _sha256_text(core_text),
                "evidence_text_sha256": _sha256_text(evidence_text),
                "core_char_length": len(core_text),
                "evidence_char_length": len(evidence_text),
                "halo_left_chars": start - halo_start,
                "halo_right_chars": halo_end - end,
            }
        )
    return extracted


def _run_sparse_smoke(
    extracted: Sequence[Mapping[str, Any]],
    *,
    n_features: int,
    target_recall: float,
) -> dict[str, Any]:
    rows = []
    for item in extracted:
        counts = hashed_ngram_feature_counts(str(item["core_text"]), n_features=n_features)
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "label": item["classifier_label"],
                "counts": counts,
                "nonzero": len(counts),
            }
        )
    model = _centroid_model(rows)
    scores = [_centroid_score(row["counts"], model) for row in rows]
    labels = [1 if row["label"] == "reject" else 0 for row in rows]
    threshold = _select_threshold_for_recall(labels=labels, scores=scores, target_recall=target_recall)
    predictions = [score >= threshold for score in scores]
    true_positives = sum(1 for label, pred in zip(labels, predictions, strict=True) if label == 1 and pred)
    positives = sum(labels)
    recall = true_positives / positives if positives else 0.0
    model_state_hash = stable_json_hash(
        {
            "positive_hash": stable_json_hash(model["positive"]),
            "negative_hash": stable_json_hash(model["negative"]),
            "threshold": threshold,
        }
    )
    public = {
        "model_id": SPARSE_SMOKE_MODEL_ID,
        "status": "passed",
        "row_count": len(rows),
        "feature_config_hash": stable_json_hash({"n_features": n_features, "feature_family": "word_char_hashed_ngrams"}),
        "feature_nonzero_min": min(row["nonzero"] for row in rows),
        "feature_nonzero_max": max(row["nonzero"] for row in rows),
        "threshold": threshold,
        "target_recall": float(target_recall),
        "observed_recall": recall,
        "model_state_hash": model_state_hash,
    }
    return {"public": public, "model": model}


def _run_window_mil_proxy_smoke(extracted: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pooled_scores: list[float] = []
    window_counts: list[int] = []
    for item in extracted:
        windows = _char_windows(str(item["evidence_text"]), width=512, stride=256)
        if not windows:
            raise ValueError("window-MIL smoke produced no windows")
        scores = [_hash_score(window) for window in windows]
        pooled_scores.append(max(scores))
        window_counts.append(len(windows))
    public = {
        "model_id": WINDOW_SMOKE_MODEL_ID,
        "status": "passed",
        "backend": "deterministic_window_hash_proxy_local_smoke_only",
        "mil_pooling": "max_window_score",
        "row_count": len(extracted),
        "window_count_min": min(window_counts),
        "window_count_max": max(window_counts),
        "pooled_score_min": min(pooled_scores),
        "pooled_score_max": max(pooled_scores),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    return {"public": public}


def _centroid_model(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    by_label = {"reject": Counter(), "safe": Counter()}
    counts_by_label = Counter(str(row["label"]) for row in rows)
    if counts_by_label["reject"] == 0 or counts_by_label["safe"] == 0:
        raise ValueError("sparse smoke requires both labels")
    for row in rows:
        label = str(row["label"])
        by_label[label].update(dict(row["counts"]))
    return {
        "positive": {feature: count / counts_by_label["reject"] for feature, count in sorted(by_label["reject"].items())},
        "negative": {feature: count / counts_by_label["safe"] for feature, count in sorted(by_label["safe"].items())},
    }


def _centroid_score(counts: Mapping[str, int], model: Mapping[str, Mapping[str, float]]) -> float:
    positive = model["positive"]
    negative = model["negative"]
    total = sum(int(value) for value in counts.values()) or 1
    margin = 0.0
    for feature, value in counts.items():
        margin += int(value) * (float(positive.get(feature, 0.0)) - float(negative.get(feature, 0.0)))
    return 1.0 / (1.0 + math.exp(max(-30.0, min(30.0, -margin / total))))


def _select_threshold_for_recall(*, labels: Sequence[int], scores: Sequence[float], target_recall: float) -> float:
    positives = sum(1 for label in labels if label == 1)
    if positives < 1:
        raise ValueError("threshold selection requires positive labels")
    candidates = sorted(set(float(score) for score in scores))
    best = candidates[0]
    for threshold in candidates:
        predicted = [score >= threshold for score in scores]
        true_positives = sum(1 for label, pred in zip(labels, predicted, strict=True) if label == 1 and pred)
        recall = true_positives / positives
        if recall >= target_recall:
            best = threshold
        else:
            break
    return float(best)


def _smoke_checkpoint_payload(
    *,
    classifier_manifest_hash: str,
    source_schedule_hash: str,
    max_rows: int,
    halo_chars: int,
    n_features: int,
    sparse_result: Mapping[str, Any],
    window_result: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = {
        "artifact_kind": CHUNK_LOCAL_SMOKE_CHECKPOINT_KIND,
        "schema_version": 1,
        "classifier_manifest_hash": classifier_manifest_hash,
        "source_schedule_hash": source_schedule_hash,
        "max_rows": int(max_rows),
        "halo_chars": int(halo_chars),
        "n_features": int(n_features),
        "sparse_model_state_hash": sparse_result["public"]["model_state_hash"],
        "sparse_threshold": sparse_result["public"]["threshold"],
        "window_proxy_backend": window_result["public"]["backend"],
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "classifier_training_performed": False,
    }
    checkpoint["checkpoint_hash"] = stable_json_hash(checkpoint)
    validate_public_no_raw_payload(checkpoint)
    return checkpoint


def _char_windows(text: str, *, width: int, stride: int) -> list[str]:
    if not text:
        return []
    if len(text) <= width:
        return [text]
    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + width)
        windows.append(text[start:end])
        if end == len(text):
            break
        start += stride
    return windows


def _hash_score(text: str) -> float:
    digest = stable_json_hash({"window": text})
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _stage_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    dirs = [Path(path) for path in args.stage_dir]
    if args.stage_root:
        dirs.extend(_complete_stage_dirs_from_root(Path(args.stage_root), args.stage_glob))
    return dirs


def _complete_stage_dirs_from_root(stage_root: Path, stage_glob: str) -> list[Path]:
    return [
        path for path in sorted(stage_root.glob(stage_glob))
        if (path / "chunk_relabel_manifest.json").exists()
        and (path / "chunk_relabel_private_map.json").exists()
        and (path / "chunk_labels_live.jsonl").exists()
    ]


def _build_hf_provider(args: argparse.Namespace) -> HFFsspecParquetRowProvider:
    return HFFsspecParquetRowProvider(
        columns=("text",),
        cache_dir=Path(args.cache_dir),
        block_size_bytes=int(args.block_size_bytes),
        cache_delta_cap_bytes=int(args.cache_delta_cap_bytes),
        manage_hf_cache_env=int(args.source_concurrency) == 1,
        row_group_batch_size=int(args.row_group_batch_size),
        source_concurrency=int(args.source_concurrency),
        source_retry_count=int(args.source_retry_count),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local chunk classifier smoke checks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("local-smoke")
    smoke.add_argument("--classifier-manifest", required=True)
    smoke.add_argument("--source-schedule", action="append", required=True)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--checkpoint-output")
    smoke.add_argument("--stage-dir", action="append", default=[])
    smoke.add_argument("--stage-root")
    smoke.add_argument("--stage-glob", default="rejected_35k_chunk_relabel_stage*")
    smoke.add_argument("--active-learning-private-map", action="append", default=[])
    smoke.add_argument("--require-active-learning-batch", action="append", default=[])
    smoke.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    smoke.add_argument("--halo-chars", type=int, default=DEFAULT_HALO_CHARS)
    smoke.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    smoke.add_argument("--target-recall", type=float, default=0.995)
    smoke.add_argument("--cache-dir", default="reports/cache_probe/chunk_classifier_local_smoke")
    smoke.add_argument("--block-size-bytes", type=int, default=DEFAULT_HF_ROW_READ_BLOCK_SIZE_BYTES)
    smoke.add_argument("--cache-delta-cap-bytes", type=int, default=2_000_000_000)
    smoke.add_argument("--row-group-batch-size", type=int, default=8)
    smoke.add_argument("--source-concurrency", type=int, default=4)
    smoke.add_argument("--source-retry-count", type=int, default=3)
    args = parser.parse_args(argv)

    if args.command == "local-smoke":
        command = [
            "python",
            "-m",
            "twc.classifiers.chunk_training_smoke",
            "local-smoke",
            "--classifier-manifest",
            str(args.classifier_manifest),
            "--output",
            str(args.output),
        ]
        for path in args.source_schedule:
            command.extend(["--source-schedule", str(path)])
        for path in args.active_learning_private_map:
            command.extend(["--active-learning-private-map", str(path)])
        for batch_id in args.require_active_learning_batch:
            command.extend(["--require-active-learning-batch", str(batch_id)])
        manifest = json.loads(Path(args.classifier_manifest).read_text(encoding="utf-8"))
        schedules = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.source_schedule]
        checkpoint_output = args.checkpoint_output or str(Path(args.output).with_name(Path(args.output).stem + "_checkpoint.json"))
        report = build_chunk_classifier_local_smoke_report(
            classifier_manifest=manifest,
            stage_dirs=_stage_dirs_from_args(args),
            active_learning_private_maps=args.active_learning_private_map,
            source_schedule=schedules,
            row_provider=_build_hf_provider(args),
            max_rows=args.max_rows,
            halo_chars=args.halo_chars,
            n_features=args.n_features,
            target_recall=args.target_recall,
            real_entrypoint_command=command,
            output_checkpoint_json=checkpoint_output,
            require_active_learning_batches=args.require_active_learning_batch,
        )
        _write_json(args.output, report)
        return 0
    raise AssertionError(args.command)


def _required_hash(value: Any, field_name: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return candidate


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
