"""Validation helpers for immutable sharded W1 handoffs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twc.repro.hashing import stable_json_hash, self_hash_mismatches
from twc.pretraining.filtering import artifacts as filtering_artifacts
from twc.pretraining.filtering import policy as filtering_policy
from twc.pretraining.filtering import stage_state_v1
from twc.pretraining.filtering.prefilter_records import (
    _is_w1_compact_prefilter_receipt,
    _prefilter_receipt_has_survivor_queue_id_index_proof,
    _prefilter_receipt_has_survivor_queue_id_proof,
    _validate_prefilter_ready_receipt,
    _validate_prefilter_ready_receipt_payload,
    _validate_survivor_queue_id_index_ready_file,
    _validate_survivor_queue_ready_file,
    _validate_w1_compact_receipt_proof_payload,
)
from twc.pretraining.filtering.prefilter_supervisor import (
    _prefilter_supervisor_manifest_hash_payload,
    _prefilter_supervisor_run_dir_sha256,
    _prefilter_supervisor_shard_dir,
    _w1_start_prefilter_expected_status,
)

def _elapsed_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"max": None, "p50": None, "p90": None}
    ordered = sorted(float(value) for value in values)
    return {
        "max": round(ordered[-1], 6),
        "p50": round(ordered[len(ordered) // 2], 6),
        "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 6),
    }

def _validate_sharded_w1_handoff_payload(
    handoff: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    manifest: Mapping[str, Any] | None = None,
    require_live_shard_state: bool = True,
    require_compact_w1_scores_empty: bool = False,
) -> None:
    if handoff.get("artifact_kind") != filtering_artifacts.SHARDED_W1_HANDOFF_KIND:
        raise ValueError("sharded W1 handoff kind mismatch")
    if int(handoff.get("schema_version") or 0) != 1:
        raise ValueError("sharded W1 handoff schema_version mismatch")
    if self_hash_mismatches(handoff, "handoff_hash"):
        raise ValueError("sharded W1 handoff hash mismatch")
    if handoff.get("status") != "sharded_prefilter_ready_for_w1":
        raise ValueError("sharded W1 handoff is not ready")
    if handoff.get("supervisor_run_dir_sha256") != _prefilter_supervisor_run_dir_sha256(supervisor_run_dir):
        raise ValueError("sharded W1 handoff supervisor run dir mismatch")
    supervisor_path = Path(supervisor_run_dir)
    prefilter_config_payload = filtering_policy._prefilter_input_config_payload_for_hash(
        config_payload,
        handoff.get("config_hash"),
        label="sharded W1 handoff",
    )
    for field in (
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "deterministic_filter_policy_hash",
        "sparse_report_hash",
        "sparse_model_file_sha256",
    ):
        if handoff.get(field) != prefilter_config_payload.get(field):
            raise ValueError(f"sharded W1 handoff {field} mismatch")
    if handoff.get("w1_report_hash") != prefilter_config_payload["window_mil_report_hash"]:
        raise ValueError("sharded W1 handoff W1 report hash mismatch")
    if handoff.get("w1_model_file_sha256") != prefilter_config_payload["window_mil_model_file_sha256"]:
        raise ValueError("sharded W1 handoff W1 model hash mismatch")
    shards = handoff.get("shards")
    if not isinstance(shards, list) or len(shards) != int(handoff.get("shard_count", -1)):
        raise ValueError("sharded W1 handoff shard set mismatch")
    live_manifest = manifest
    if live_manifest is None and require_live_shard_state:
        manifest_path = supervisor_path / "prefilter_supervisor_manifest.json"
        if not manifest_path.exists():
            raise ValueError("sharded W1 handoff requires the supervisor manifest in the supervisor run dir")
        loaded_manifest = filtering_artifacts._load_json(manifest_path)
        if not isinstance(loaded_manifest, Mapping):
            raise ValueError("sharded W1 handoff supervisor manifest must be an object")
        live_manifest = loaded_manifest
    manifest_shards_by_index: dict[int, Mapping[str, Any]] = {}
    if live_manifest is not None:
        if live_manifest.get("manifest_hash") != stable_json_hash(_prefilter_supervisor_manifest_hash_payload(live_manifest)):
            raise ValueError("sharded W1 handoff supervisor manifest hash mismatch")
        if live_manifest.get("manifest_hash") != handoff.get("supervisor_manifest_hash"):
            raise ValueError("sharded W1 handoff supervisor_manifest_hash mismatch")
        if live_manifest.get("prefilter_plan_hash") != handoff.get("prefilter_plan_hash"):
            raise ValueError("sharded W1 handoff prefilter_plan_hash mismatch")
        if live_manifest.get("prefilter_plan_kind") != handoff.get("prefilter_plan_kind"):
            raise ValueError("sharded W1 handoff prefilter_plan_kind mismatch")
        if live_manifest.get("supervisor_run_dir_sha256") != handoff.get("supervisor_run_dir_sha256"):
            raise ValueError("sharded W1 handoff supervisor run dir mismatch")
        if int(live_manifest.get("shard_count", -1)) != int(handoff.get("shard_count", -1)):
            raise ValueError("sharded W1 handoff manifest shard_count mismatch")
        manifest_shards = live_manifest.get("shards")
        if not isinstance(manifest_shards, list):
            raise ValueError("sharded W1 handoff supervisor manifest shard set mismatch")
        manifest_shards_by_index = {int(shard["shard_index"]): shard for shard in manifest_shards}
    seen: set[int] = set()
    expected_physical_rows = 0
    terminal_rows = 0
    candidate_count = 0
    survivor_count = 0
    route_counts: Counter[str] = Counter()
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("sharded W1 handoff shard must be an object")
        index = int(shard.get("shard_index", -1))
        if index < 0 or index in seen:
            raise ValueError("sharded W1 handoff duplicate shard index")
        seen.add(index)
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_path, shard)
        filtering_artifacts._require_sha256(shard.get("prefilter_stage_receipt_hash"), "prefilter_stage_receipt_hash")
        filtering_artifacts._require_sha256(shard.get("survivor_queue_hash"), "survivor_queue_hash")
        expected_physical_rows += int(shard.get("expected_physical_rows", 0))
        terminal_rows += int(shard.get("terminal_rows", 0))
        candidate_count += int(shard.get("candidate_count", 0))
        survivor_count += int(shard.get("survivor_count", 0))
        route_counts.update({str(key): int(value) for key, value in dict(shard.get("route_counts") or {}).items()})
        manifest_shard = manifest_shards_by_index.get(index)
        if manifest_shard is not None:
            if manifest_shard.get("shard_run_dir_name") != shard.get("shard_run_dir_name"):
                raise ValueError("sharded W1 handoff shard_run_dir_name mismatch")
            if int(manifest_shard.get("expected_physical_rows", -1)) != int(shard.get("expected_physical_rows", -2)):
                raise ValueError("sharded W1 handoff shard expected_physical_rows mismatch")
        if require_live_shard_state:
            receipt_path = shard_dir / "prefilter_cpu_receipt.json"
            if not receipt_path.exists():
                raise ValueError("sharded W1 handoff shard prefilter receipt is missing")
            receipt = filtering_artifacts._load_json(receipt_path)
            if not isinstance(receipt, Mapping):
                raise ValueError("sharded W1 handoff shard prefilter receipt must be an object")
            expected_status = _w1_start_prefilter_expected_status(receipt)
            shard_prefilter_config_payload = filtering_policy._prefilter_input_config_payload_for_hash(
                prefilter_config_payload,
                receipt.get("config_hash"),
                label="sharded W1 handoff shard prefilter receipt",
            )
            if _is_w1_compact_prefilter_receipt(receipt):
                if _prefilter_receipt_has_survivor_queue_id_index_proof(receipt):
                    queue_count = _validate_survivor_queue_id_index_ready_file(
                        shard_dir / "w1_survivor_queue.jsonl",
                        allow_empty=True,
                        expected_hash=receipt.get("survivor_queue_id_index_hash"),
                        expected_row_count=int(
                            receipt.get("survivor_queue_id_index_row_count", receipt.get("survivor_count", 0))
                        ),
                        expected_candidate_id_set_hash=receipt.get("survivor_candidate_id_set_hash"),
                        expected_private_evidence_id_set_hash=receipt.get("survivor_private_evidence_id_set_hash"),
                    )
                    _validate_w1_compact_receipt_proof_payload(
                        receipt,
                        shard_dir,
                        shard_prefilter_config_payload,
                        queue_row_count=queue_count,
                        expected_status=expected_status,
                        require_empty_w1_scores=require_compact_w1_scores_empty,
                    )
                else:
                    queue_rows = filtering_artifacts._load_ready_jsonl(shard_dir / "w1_survivor_queue.jsonl", "prefilter survivor queue")
                    conn = stage_state_v1.open_stage_state_readonly(shard_dir, shard_prefilter_config_payload)
                    try:
                        _validate_prefilter_ready_receipt(
                            conn,
                            shard_dir,
                            shard_prefilter_config_payload,
                            queue_rows=queue_rows,
                            expected_status=expected_status,
                            require_compact_w1_scores_empty=require_compact_w1_scores_empty,
                        )
                    finally:
                        conn.close()
            elif _prefilter_receipt_has_survivor_queue_id_proof(receipt):
                _validate_prefilter_ready_receipt_payload(
                    receipt,
                    shard_dir,
                    shard_prefilter_config_payload,
                    queue_row_count=int(shard.get("survivor_count", 0)),
                    expected_status=expected_status,
                )
            else:
                queue_count = _validate_survivor_queue_ready_file(
                    shard_dir / "w1_survivor_queue.jsonl",
                    shard_prefilter_config_payload,
                    allow_empty=True,
                )
                conn = stage_state_v1.open_stage_state_readonly(shard_dir, shard_prefilter_config_payload)
                try:
                    _validate_prefilter_ready_receipt(
                        conn,
                        shard_dir,
                        shard_prefilter_config_payload,
                        queue_row_count=queue_count,
                        expected_status=expected_status,
                    )
                finally:
                    conn.close()
            if receipt.get("stage_receipt_hash") != shard.get("prefilter_stage_receipt_hash"):
                raise ValueError("sharded W1 handoff shard prefilter_stage_receipt_hash mismatch")
            for field in ("expected_physical_rows", "terminal_rows", "candidate_count", "survivor_count", "route_counts"):
                if receipt.get(field) != shard.get(field):
                    raise ValueError(f"sharded W1 handoff shard {field} mismatch")
            for field in (
                "survivor_queue_row_count",
                "survivor_queue_id_index_hash",
                "survivor_queue_id_index_row_count",
                "survivor_candidate_id_set_hash",
                "survivor_private_evidence_id_set_hash",
            ):
                if shard.get(field) is not None and receipt.get(field) != shard.get(field):
                    raise ValueError(f"sharded W1 handoff shard {field} mismatch")
    if seen != set(range(int(handoff["shard_count"]))):
        raise ValueError("sharded W1 handoff shard indexes are not contiguous")
    aggregate_expected = {
        "expected_physical_rows": int(expected_physical_rows),
        "terminal_rows": int(terminal_rows),
        "candidate_count": int(candidate_count),
        "survivor_count": int(survivor_count),
        "route_counts": filtering_artifacts._sorted_counter(route_counts),
    }
    for field, expected_value in aggregate_expected.items():
        if handoff.get(field) != expected_value:
            raise ValueError(f"sharded W1 handoff {field} mismatch")
    filtering_artifacts._require_false_flags(handoff, filtering_artifacts.PUBLIC_FALSE_FLAGS, "sharded W1 handoff")
    filtering_artifacts.validate_public_no_raw_payload(handoff)
