"""Prefilter runtime and row-group telemetry helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path
import time
from typing import Any

from twc.repro.hashing import stable_json_hash, self_hash_mismatches
from twc.pretraining.filtering import artifacts as filtering_artifacts

PREFILTER_TIMING_SECOND_FIELDS = (
    "row_group_iteration", "parquet_fetch", "segmentation", "candidate_construction",
    "deterministic_filter", "s1_feature_score", "sqlite_write",
    "survivor_queue_export", "receipt_build",
)
PREFILTER_TIMING_COUNT_FIELDS = (
    "row_groups_considered", "row_groups_fetched", "rows_fetched",
    "segmentation_sampled_validations", "candidates_built",
    "deterministic_candidates", "s1_scored_candidates",
)
PREFILTER_ROW_GROUP_METRICS_FILENAME = "prefilter_row_group_metrics.jsonl"

def _new_prefilter_runtime_metrics() -> dict[str, Any]:
    return {
        "seconds": {field: 0.0 for field in PREFILTER_TIMING_SECOND_FIELDS},
        "counts": Counter(),
    }

def _record_prefilter_elapsed(metrics: dict[str, Any], field: str, start: float) -> float:
    if field not in PREFILTER_TIMING_SECOND_FIELDS:
        raise ValueError(f"unsupported prefilter timing field: {field}")
    elapsed = time.monotonic() - start
    metrics["seconds"][field] = float(metrics["seconds"].get(field, 0.0)) + elapsed
    return elapsed

def _increment_prefilter_count(metrics: dict[str, Any], field: str, count: int = 1) -> None:
    if field not in PREFILTER_TIMING_COUNT_FIELDS:
        raise ValueError(f"unsupported prefilter count field: {field}")
    metrics["counts"][field] += int(count)

def _prefilter_runtime_receipt_fields(metrics: Mapping[str, Any]) -> dict[str, Any]:
    seconds = metrics.get("seconds", {})
    counts = metrics.get("counts", {})
    payload: dict[str, Any] = {"prefilter_timing_schema_version": 1}
    for field in PREFILTER_TIMING_SECOND_FIELDS:
        payload[f"prefilter_{field}_seconds"] = round(float(seconds.get(field, 0.0)), 6)
    for field in PREFILTER_TIMING_COUNT_FIELDS:
        payload[f"prefilter_{field}"] = int(counts.get(field, 0))
    payload["prefilter_private_w1_evidence_write_included_in_sqlite_write_seconds"] = True
    return payload

def _attach_prefilter_runtime_receipt_fields(report: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    payload.update(_prefilter_runtime_receipt_fields(metrics))
    if (
        payload.get("segmentation_validation_mode") == "sampled"
        and int(payload.get("segmented_rows", 0)) > 0
        and int(payload.get("prefilter_segmentation_sampled_validations", 0)) == 0
    ):
        raise ValueError("sampled segmentation validation produced zero validations")
    payload.pop("stage_receipt_hash", None)
    payload["stage_receipt_hash"] = stable_json_hash(payload)
    filtering_artifacts.validate_public_no_raw_payload(payload)
    return payload

def _prefilter_row_group_metric_key(entry: Mapping[str, Any]) -> tuple[str, int]:
    return (stable_json_hash(str(entry.get("source_file_id") or "")), int(entry.get("row_group_index", 0)))

def _new_prefilter_row_group_metric_state(entry: Mapping[str, Any]) -> dict[str, Any]:
    source_file_id_hash, row_group_index = _prefilter_row_group_metric_key(entry)
    return {
        "entry": dict(entry),
        "source_schedule_hash": str(
            entry.get("row_group_metric_source_schedule_hash")
            or entry.get("source_schedule_hash")
            or ""
        ),
        "input_schedule_hash": str(entry.get("source_schedule_hash") or ""),
        "schedule_entry_hash": str(entry.get("entry_hash") or ""),
        "source_file_id_hash": source_file_id_hash,
        "row_group_index": int(row_group_index),
        "row_group_num_rows": int(entry.get("row_group_num_rows", 0)),
        "rows_fetched": 0,
        "terminal_rows": 0,
        "candidate_count": 0,
        "survivor_count": 0,
        "route_counts": Counter(),
        "seconds": {field: 0.0 for field in PREFILTER_TIMING_SECOND_FIELDS},
    }

def _add_prefilter_row_group_elapsed(state: dict[str, Any], field: str, elapsed: float) -> None:
    if field not in PREFILTER_TIMING_SECOND_FIELDS:
        raise ValueError(f"unsupported prefilter row-group timing field: {field}")
    seconds = state.setdefault("seconds", {})
    seconds[field] = float(seconds.get(field, 0.0)) + float(elapsed)

def _prefilter_row_group_metric_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "artifact_kind": "twc_chunk_filter_prefilter_row_group_metrics_v1",
        "schema_version": 1,
        "row_group_metric_id": stable_json_hash(
            {
                "source_schedule_hash": state["source_schedule_hash"],
                "input_schedule_hash": state["input_schedule_hash"],
                "source_file_id_hash": state["source_file_id_hash"],
                "row_group_index": int(state["row_group_index"]),
                "row_group_num_rows": int(state["row_group_num_rows"]),
            }
        ),
        "source_schedule_hash": filtering_artifacts._require_sha256(state["source_schedule_hash"], "row group source_schedule_hash"),
        "input_schedule_hash": filtering_artifacts._require_sha256(state["input_schedule_hash"], "row group input_schedule_hash"),
        "schedule_entry_hash": filtering_artifacts._require_sha256(state["schedule_entry_hash"], "row group schedule_entry_hash"),
        "source_file_id_hash": filtering_artifacts._require_sha256(state["source_file_id_hash"], "row group source_file_id_hash"),
        "row_group_index": int(state["row_group_index"]),
        "row_group_num_rows": int(state["row_group_num_rows"]),
        "rows_fetched": int(state.get("rows_fetched", 0)),
        "terminal_rows": int(state.get("terminal_rows", 0)),
        "candidate_count": int(state.get("candidate_count", 0)),
        "survivor_count": int(state.get("survivor_count", 0)),
        "route_counts": filtering_artifacts._sorted_counter(Counter(dict(state.get("route_counts") or {}))),
        "stage_seconds": {
            field: round(float((state.get("seconds") or {}).get(field, 0.0)), 6)
            for field in PREFILTER_TIMING_SECOND_FIELDS
        },
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["row_group_metric_hash"] = stable_json_hash(payload)
    filtering_artifacts.validate_public_no_raw_payload(payload)
    return payload

def _load_prefilter_row_group_metrics(run_path: str | Path) -> dict[str, dict[str, Any]]:
    ready = Path(run_path) / f"{PREFILTER_ROW_GROUP_METRICS_FILENAME}.ready"
    if not ready.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with ready.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"prefilter row-group metrics row {line_number} is not an object")
            filtering_artifacts.validate_public_no_raw_payload(payload)
            if payload.get("artifact_kind") != "twc_chunk_filter_prefilter_row_group_metrics_v1":
                raise ValueError("prefilter row-group metrics kind mismatch")
            if self_hash_mismatches(payload, "row_group_metric_hash"):
                raise ValueError("prefilter row-group metrics hash mismatch")
            metric_id = str(payload.get("row_group_metric_id") or "")
            filtering_artifacts._require_sha256(metric_id, "row_group_metric_id")
            if metric_id in rows:
                raise ValueError("prefilter row-group metrics contain duplicate metric ids")
            rows[metric_id] = dict(payload)
    return rows

def _write_prefilter_row_group_metrics(
    run_path: str | Path,
    metrics_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = sorted(
        (dict(row) for row in metrics_by_id.values()),
        key=lambda row: (str(row["source_file_id_hash"]), int(row["row_group_index"])),
    )
    base_path = Path(run_path) / PREFILTER_ROW_GROUP_METRICS_FILENAME
    digest, count = filtering_artifacts._write_jsonl_ready_streaming(base_path, rows)
    return {
        "prefilter_row_group_metrics_filename": f"{PREFILTER_ROW_GROUP_METRICS_FILENAME}.ready",
        "prefilter_row_group_metrics_row_count": int(count),
        "prefilter_row_group_metrics_sha256": digest,
    }
