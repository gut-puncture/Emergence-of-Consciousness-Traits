"""Prefilter supervisor manifests, shard commands, and child validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from twc.repro.hashing import stable_json_hash
from twc.pretraining.filtering import artifacts as filtering_artifacts
from twc.pretraining.filtering import policy as filtering_policy
from twc.pretraining.filtering import source_prep
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

PREFILTER_SUPERVISOR_SCHEMA_VERSION = 1
PREFILTER_SUPERVISOR_MODES = {"mergeable_tranche", "diagnostic_sweep"}
SEGMENTATION_VALIDATION_MODES = {"full", "sampled"}
DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS = 1000
PREFILTER_SUPERVISOR_THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

def _validate_segmentation_validation_mode(mode: str) -> str:
    value = str(mode)
    if value not in SEGMENTATION_VALIDATION_MODES:
        raise ValueError("segmentation validation mode must be full or sampled")
    return value

def _segmentation_sample_modulus(value: int) -> int:
    return filtering_artifacts._positive_int(value, "segmentation_validation_sample_modulus")

def _prefilter_supervisor_manifest_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}

def _prefilter_supervisor_shard_command_hash(
    *,
    manifest_mode: str,
    shard_count: int,
    shard_index: int,
    row_group_batch_size: int,
    shard_max_rows: int | None,
    min_free_bytes: int,
    row_group_sharding_policy: str = source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> str:
    payload = {
        "command": "prefilter-local-parquet-plan",
        "manifest_mode": manifest_mode,
        "row_group_shard_count": int(shard_count),
        "row_group_shard_index": int(shard_index),
        "row_group_batch_size": int(row_group_batch_size),
        "max_rows": None if shard_max_rows is None else int(shard_max_rows),
        "min_free_bytes": int(min_free_bytes),
        "segmentation_validation_mode": _validate_segmentation_validation_mode(segmentation_validation_mode),
        "segmentation_validation_sample_modulus": _segmentation_sample_modulus(
            segmentation_validation_sample_modulus
        ),
        "thread_env": PREFILTER_SUPERVISOR_THREAD_ENV,
    }
    policy = source_prep._validate_row_group_sharding_policy(row_group_sharding_policy)
    if policy != source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL:
        payload["row_group_sharding_policy"] = policy
    return stable_json_hash(payload)

def _prefilter_supervisor_shard_dir(supervisor_run_dir: str | Path, shard: Mapping[str, Any]) -> Path:
    name = str(shard.get("shard_run_dir_name") or "")
    relative = Path(name)
    if not name or relative.is_absolute() or relative.name != name:
        raise ValueError("prefilter supervisor shard_run_dir_name must be a simple relative name")
    return Path(supervisor_run_dir) / relative

def _prefilter_supervisor_run_dir_sha256(path: str | Path) -> str:
    return filtering_artifacts._sha256_text(str(Path(path).expanduser().resolve(strict=False)))

def _allocate_prefilter_supervisor_diagnostic_row_caps(
    shard_expected_rows: Sequence[int],
    total_max_rows: int | None,
) -> list[int | None]:
    if total_max_rows is None:
        return [None for _ in shard_expected_rows]
    budget = filtering_artifacts._positive_int(total_max_rows, "max_rows")
    caps: list[int | None] = []
    remaining_budget = budget
    remaining_shards = len(shard_expected_rows)
    for expected_rows in shard_expected_rows:
        if remaining_budget <= 0:
            cap = 0
        else:
            cap = min(int(expected_rows), math.ceil(remaining_budget / max(1, remaining_shards)))
        caps.append(cap)
        remaining_budget -= int(cap)
        remaining_shards -= 1
    if sum(int(cap or 0) for cap in caps) > budget:
        raise ValueError("prefilter supervisor diagnostic row budget allocation overflow")
    return caps

def _expected_prefilter_supervisor_shards(
    *,
    prefilter_plan: Mapping[str, Any],
    shard_count: int,
    row_group_batch_size: int,
    manifest_mode: str,
    max_rows: int | None,
    min_free_bytes: int,
    row_group_sharding_policy: str = source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> list[dict[str, Any]]:
    policy = source_prep._validate_row_group_sharding_policy(row_group_sharding_policy)
    plan = dict(prefilter_plan)
    source_prep._validate_local_parquet_prefilter_plan(plan)
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("local all-row prefilter plan has no entries to shard")
    assigned_ordinals = source_prep._assigned_row_group_ordinals(
        entries,
        shard_count=shard_count,
        row_group_sharding_policy=policy,
    )
    shard_plans = [
        source_prep._shard_local_parquet_prefilter_plan_from_assigned_ordinals(
            plan,
            entries=entries,
            assigned_ordinals=assigned_ordinals,
            shard_count=shard_count,
            shard_index=shard_index,
            row_group_sharding_policy=policy,
        )
        for shard_index in range(shard_count)
    ]
    shard_caps = _allocate_prefilter_supervisor_diagnostic_row_caps(
        [int(shard_plan["expected_physical_rows"]) for shard_plan in shard_plans],
        max_rows if manifest_mode == "diagnostic_sweep" else None,
    )
    shards: list[dict[str, Any]] = []
    for shard_index, (shard_plan, shard_cap) in enumerate(zip(shard_plans, shard_caps, strict=True)):
        shards.append(
            {
                "shard_index": shard_index,
                "shard_run_dir_name": f"shard_{shard_index:05d}",
                "row_group_shard_count": shard_count,
                "expected_row_group_count": int(shard_plan["row_group_count"]),
                "expected_physical_rows": int(
                    shard_plan["expected_physical_rows"]
                    if shard_cap is None
                    else min(shard_cap, int(shard_plan["expected_physical_rows"]))
                ),
                "source_plan_expected_physical_rows": int(shard_plan["expected_physical_rows"]),
                "max_rows": shard_cap,
                "command_hash": _prefilter_supervisor_shard_command_hash(
                    manifest_mode=manifest_mode,
                    shard_count=shard_count,
                    shard_index=shard_index,
                    row_group_batch_size=row_group_batch_size,
                    row_group_sharding_policy=policy,
                    shard_max_rows=shard_cap,
                    min_free_bytes=min_free_bytes,
                    segmentation_validation_mode=segmentation_validation_mode,
                    segmentation_validation_sample_modulus=segmentation_validation_sample_modulus,
                ),
            }
        )
    return shards

def validate_prefilter_local_parquet_plan_supervisor_manifest(
    manifest: Mapping[str, Any],
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    supervisor_run_dir: str | Path | None = None,
    expect_merge_allowed: bool | None = None,
) -> None:
    if manifest.get("artifact_kind") != filtering_artifacts.PREFILTER_SUPERVISOR_MANIFEST_KIND:
        raise ValueError("prefilter supervisor manifest kind mismatch")
    if int(manifest.get("schema_version") or 0) != PREFILTER_SUPERVISOR_SCHEMA_VERSION:
        raise ValueError("prefilter supervisor manifest schema_version mismatch")
    if manifest.get("manifest_hash") != stable_json_hash(_prefilter_supervisor_manifest_hash_payload(manifest)):
        raise ValueError("prefilter supervisor manifest hash mismatch")
    filtering_policy._validate_prefilter_input_config_payload(config_payload)
    plan = dict(prefilter_plan)
    source_prep._validate_local_parquet_prefilter_plan(plan)
    plan_hash = filtering_artifacts._require_sha256(plan.get("schedule_hash"), "prefilter_plan_hash")
    if manifest.get("prefilter_plan_hash") != plan_hash:
        raise ValueError("prefilter supervisor plan hash mismatch")
    if supervisor_run_dir is not None and manifest.get("supervisor_run_dir_sha256") != _prefilter_supervisor_run_dir_sha256(
        supervisor_run_dir
    ):
        raise ValueError("prefilter supervisor run dir mismatch")
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "deterministic_filter_policy_hash",
        "sparse_report_hash",
        "sparse_model_file_sha256",
    ):
        if manifest.get(field) != config_payload.get(field):
            raise ValueError(f"prefilter supervisor {field} mismatch")
    if float(manifest.get("sparse_threshold")) != float(config_payload["sparse_threshold"]):
        raise ValueError("prefilter supervisor sparse_threshold mismatch")
    mode = str(manifest.get("manifest_mode") or "")
    if mode not in PREFILTER_SUPERVISOR_MODES:
        raise ValueError("prefilter supervisor manifest_mode mismatch")
    merge_allowed = bool(manifest.get("merge_allowed"))
    if expect_merge_allowed is not None and merge_allowed != bool(expect_merge_allowed):
        raise ValueError("prefilter supervisor merge_allowed mismatch")
    if mode == "mergeable_tranche":
        if not merge_allowed:
            raise ValueError("prefilter supervisor mergeable manifest is not merge_allowed")
        if manifest.get("max_rows") is not None:
            raise ValueError("mergeable prefilter supervisor manifests cannot use max_rows")
        if manifest.get("claims_scope") != "mergeable_prefilter_tranche":
            raise ValueError("prefilter supervisor claims_scope mismatch")
    else:
        if merge_allowed:
            raise ValueError("diagnostic prefilter supervisor manifest is not mergeable")
        if manifest.get("claims_scope") != "throughput_diagnostic_only":
            raise ValueError("prefilter supervisor claims_scope mismatch")
    count = filtering_artifacts._positive_int(manifest.get("shard_count"), "shard_count")
    workers = filtering_artifacts._positive_int(manifest.get("worker_count"), "worker_count")
    if workers > count:
        raise ValueError("worker_count cannot exceed shard_count")
    batch_size = filtering_artifacts._positive_int(manifest.get("row_group_batch_size"), "row_group_batch_size")
    free_bytes = filtering_artifacts._positive_int(manifest.get("min_free_bytes"), "min_free_bytes")
    shard_policy = source_prep._validate_row_group_sharding_policy(
        manifest.get("row_group_sharding_policy", source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL)
    )
    segment_mode = _validate_segmentation_validation_mode(str(manifest.get("segmentation_validation_mode", "full")))
    segment_sample_modulus = _segmentation_sample_modulus(
        int(manifest.get("segmentation_validation_sample_modulus", DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS))
    )
    row_cap = None if manifest.get("max_rows") is None else filtering_artifacts._positive_int(manifest.get("max_rows"), "max_rows")
    if int(manifest.get("expected_row_group_count") or -1) != int(plan["row_group_count"]):
        raise ValueError("prefilter supervisor expected row-group count mismatch")
    if int(manifest.get("expected_physical_rows") or -1) != int(plan["expected_physical_rows"]):
        raise ValueError("prefilter supervisor expected physical rows mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != count:
        raise ValueError("prefilter supervisor shard set mismatch")
    expected_shards_by_index = {
        int(shard["shard_index"]): shard
        for shard in _expected_prefilter_supervisor_shards(
            prefilter_plan=plan,
            shard_count=count,
            row_group_batch_size=batch_size,
            row_group_sharding_policy=shard_policy,
            manifest_mode=mode,
            max_rows=row_cap,
            min_free_bytes=free_bytes,
            segmentation_validation_mode=segment_mode,
            segmentation_validation_sample_modulus=segment_sample_modulus,
        )
    }
    seen: set[int] = set()
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("prefilter supervisor shard must be an object")
        shard_index = int(shard.get("shard_index", -1))
        if shard_index < 0 or shard_index >= count:
            raise ValueError("prefilter supervisor shard index mismatch")
        if shard_index in seen:
            raise ValueError("prefilter supervisor duplicate shard index")
        seen.add(shard_index)
        _prefilter_supervisor_shard_dir("supervisor", shard)
        expected = expected_shards_by_index[shard_index]
        for field, expected_value in expected.items():
            if shard.get(field) != expected_value:
                raise ValueError(f"prefilter supervisor shard {field} mismatch")
    if seen != set(range(count)):
        raise ValueError("prefilter supervisor shard set mismatch")
    if row_cap is not None and sum(int(shard.get("max_rows") or 0) for shard in shards) > row_cap:
        raise ValueError("prefilter supervisor diagnostic row budget exceeds max_rows")
    filtering_artifacts._require_false_flags(manifest, filtering_artifacts.PUBLIC_FALSE_FLAGS, "prefilter supervisor manifest")
    filtering_artifacts.validate_public_no_raw_payload(manifest)

def _validate_prefilter_supervisor_child_binding(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    supervisor_manifest_path: str | Path,
    supervisor_manifest_hash: str | None,
    supervisor_command_hash: str | None,
    supervisor_run_dir_sha256: str | None,
    row_group_shard_count: int,
    row_group_shard_index: int,
    row_group_batch_size: int,
    max_rows: int | None,
    min_free_bytes: int,
    row_group_sharding_policy: str = source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> dict[str, Any]:
    claimed_manifest_hash = filtering_artifacts._require_sha256(supervisor_manifest_hash, "supervisor_manifest_hash")
    claimed_command_hash = filtering_artifacts._require_sha256(supervisor_command_hash, "supervisor_command_hash")
    claimed_run_dir_hash = filtering_artifacts._require_sha256(supervisor_run_dir_sha256, "supervisor_run_dir_sha256")
    manifest_path = Path(supervisor_manifest_path)
    manifest = filtering_artifacts._load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("prefilter supervisor manifest must be an object")
    if manifest.get("manifest_hash") != claimed_manifest_hash:
        raise ValueError("prefilter supervisor manifest hash mismatch")
    if manifest.get("supervisor_run_dir_sha256") != claimed_run_dir_hash:
        raise ValueError("prefilter supervisor run dir hash mismatch")
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=prefilter_plan,
        config_payload=config_payload,
        supervisor_run_dir=manifest_path.parent,
    )
    count = filtering_artifacts._positive_int(row_group_shard_count, "row_group_shard_count")
    index = filtering_artifacts._nonnegative_int(row_group_shard_index, "row_group_shard_index")
    shard_policy = source_prep._validate_row_group_sharding_policy(row_group_sharding_policy)
    if count != int(manifest["shard_count"]):
        raise ValueError("prefilter supervisor command hash shard count mismatch")
    if index >= count:
        raise ValueError("prefilter supervisor command hash shard index mismatch")
    shards_by_index = {int(shard["shard_index"]): shard for shard in manifest["shards"]}
    shard = shards_by_index.get(index)
    if shard is None:
        raise ValueError("prefilter supervisor command hash shard missing from manifest")
    expected_command_hash = _prefilter_supervisor_shard_command_hash(
        manifest_mode=str(manifest["manifest_mode"]),
        shard_count=count,
        shard_index=index,
        row_group_batch_size=filtering_artifacts._positive_int(row_group_batch_size, "row_group_batch_size"),
        shard_max_rows=max_rows,
        min_free_bytes=min_free_bytes,
        row_group_sharding_policy=shard_policy,
        segmentation_validation_mode=segmentation_validation_mode,
        segmentation_validation_sample_modulus=segmentation_validation_sample_modulus,
    )
    if claimed_command_hash != expected_command_hash:
        raise ValueError("prefilter supervisor command hash mismatch for child arguments")
    if shard.get("command_hash") != expected_command_hash:
        raise ValueError("prefilter supervisor command hash mismatch for manifest shard")
    return {
        "supervisor_manifest_hash": claimed_manifest_hash,
        "supervisor_command_hash": claimed_command_hash,
        "supervisor_run_dir_sha256": claimed_run_dir_hash,
    }

def _local_parquet_prefilter_plan_receipt_fields(plan: Mapping[str, Any]) -> dict[str, Any]:
    source_prep._validate_local_parquet_prefilter_plan(plan, allow_shard=bool(plan.get("row_group_sharding_enabled", False)))
    plan_hash = filtering_artifacts._require_sha256(plan.get("schedule_hash"), "prefilter_plan_hash")
    fields = {
        "local_parquet_prefilter_plan_hash": plan_hash,
        "local_parquet_prefilter_plan_kind": str(plan["artifact_kind"]),
        "local_all_rows_prefilter_plan_hash": plan_hash,
        "all_rows_prefilter_plan_hash": filtering_artifacts._require_sha256(
            plan.get("all_rows_prefilter_plan_hash") or plan_hash,
            "all_rows_prefilter_plan_hash",
        ),
        "row_group_sharding_enabled": bool(plan.get("row_group_sharding_enabled", False)),
        "row_group_shard_count": int(plan.get("row_group_shard_count", 1)),
        "row_group_shard_index": int(plan.get("row_group_shard_index", 0)),
        "row_group_shard_row_group_count": int(plan["row_group_count"]),
        "row_group_shard_expected_physical_rows": int(plan["expected_physical_rows"]),
        "all_rows_row_group_count": int(plan.get("all_rows_row_group_count", plan["row_group_count"])),
        "all_rows_expected_physical_rows": int(
            plan.get("all_rows_expected_physical_rows", plan["expected_physical_rows"])
        ),
    }
    if plan["artifact_kind"] == filtering_artifacts.LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND:
        source_plan_hash = filtering_artifacts._require_sha256(
            plan.get("token_target_tranche_plan_hash") or plan.get("local_parquet_prefilter_plan_hash") or plan_hash,
            "token_target_tranche_plan_hash",
        )
        sampling_policy = plan.get("sampling_policy")
        if not isinstance(sampling_policy, Mapping):
            raise ValueError("token tranche plan missing sampling_policy")
        fields.update(
            {
                "local_all_rows_prefilter_plan_hash": None,
                "all_rows_prefilter_plan_hash": None,
                "all_rows_row_group_count": int(plan.get("population_row_group_count", plan["row_group_count"])),
                "all_rows_expected_physical_rows": int(
                    plan.get("population_expected_physical_rows", plan["expected_physical_rows"])
                ),
                "token_target_tranche_plan_hash": source_plan_hash,
                "token_target_tranche_source_plan_hash": source_plan_hash,
                "token_target_tranche_mode": source_prep.TOKEN_TARGET_TRANCHE_MODE,
                "token_target_tranche_seed_hash": stable_json_hash(str(sampling_policy.get("seed") or "")),
                "token_target_tranche_estimated_source_token_target": int(
                    sampling_policy["estimated_source_token_target"]
                ),
                "token_target_tranche_selected_estimated_source_tokens": int(
                    sampling_policy["selected_estimated_source_tokens_midpoint"]
                ),
                "token_target_tranche_population_estimated_source_tokens": int(
                    sampling_policy["population_estimated_source_tokens_midpoint"]
                ),
                "token_target_tranche_row_group_count": int(
                    plan.get("token_target_tranche_row_group_count", plan["row_group_count"])
                ),
                "token_target_tranche_expected_physical_rows": int(
                    plan.get("token_target_tranche_expected_physical_rows", plan["expected_physical_rows"])
                ),
                "population_row_group_count": int(plan.get("population_row_group_count", plan["row_group_count"])),
                "population_expected_physical_rows": int(
                    plan.get("population_expected_physical_rows", plan["expected_physical_rows"])
                ),
                "full_corpus_census": False,
                "selected_equals_population": False,
                "claims_scope": "tranche_only",
            }
        )
    return fields

def build_prefilter_local_parquet_plan_supervisor_manifest(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    shard_count: int,
    worker_count: int,
    row_group_batch_size: int,
    manifest_mode: str,
    max_rows: int | None = None,
    min_free_bytes: int = 1,
    row_group_sharding_policy: str = source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> dict[str, Any]:
    filtering_policy.validate_chunk_filtering_config_payload(config_payload)
    plan = dict(prefilter_plan)
    source_prep._validate_local_parquet_prefilter_plan(plan)
    if plan["source_metadata_hash"] != config_payload["source_metadata_hash"]:
        raise ValueError("prefilter supervisor source metadata hash mismatch")
    mode = str(manifest_mode)
    if mode not in PREFILTER_SUPERVISOR_MODES:
        raise ValueError("prefilter supervisor manifest_mode mismatch")
    count = filtering_artifacts._positive_int(shard_count, "shard_count")
    workers = filtering_artifacts._positive_int(worker_count, "worker_count")
    if workers > count:
        raise ValueError("worker_count cannot exceed shard_count")
    batch_size = filtering_artifacts._positive_int(row_group_batch_size, "row_group_batch_size")
    free_bytes = filtering_artifacts._positive_int(min_free_bytes, "min_free_bytes")
    shard_policy = source_prep._validate_row_group_sharding_policy(row_group_sharding_policy)
    segment_mode = _validate_segmentation_validation_mode(segmentation_validation_mode)
    segment_sample_modulus = _segmentation_sample_modulus(segmentation_validation_sample_modulus)
    row_cap = None if max_rows is None else filtering_artifacts._positive_int(max_rows, "max_rows")
    if mode == "mergeable_tranche" and row_cap is not None:
        raise ValueError("mergeable prefilter supervisor manifests cannot use max_rows")
    merge_allowed = mode == "mergeable_tranche"
    claims_scope = "mergeable_prefilter_tranche" if merge_allowed else "throughput_diagnostic_only"
    plan_hash = filtering_artifacts._require_sha256(plan.get("schedule_hash"), "prefilter_plan_hash")
    shards = _expected_prefilter_supervisor_shards(
        prefilter_plan=plan,
        shard_count=count,
        row_group_batch_size=batch_size,
        row_group_sharding_policy=shard_policy,
        manifest_mode=mode,
        max_rows=row_cap,
        min_free_bytes=free_bytes,
        segmentation_validation_mode=segment_mode,
        segmentation_validation_sample_modulus=segment_sample_modulus,
    )
    manifest = {
        "artifact_kind": filtering_artifacts.PREFILTER_SUPERVISOR_MANIFEST_KIND,
        "schema_version": PREFILTER_SUPERVISOR_SCHEMA_VERSION,
        "manifest_mode": mode,
        "status": "planned",
        "claims_scope": claims_scope,
        "merge_allowed": merge_allowed,
        "prefilter_plan_hash": plan_hash,
        "prefilter_plan_kind": str(plan["artifact_kind"]),
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "deterministic_filter_policy_hash": config_payload["deterministic_filter_policy_hash"],
        "sparse_report_hash": config_payload["sparse_report_hash"],
        "sparse_model_file_sha256": config_payload["sparse_model_file_sha256"],
        "sparse_threshold": float(config_payload["sparse_threshold"]),
        "supervisor_run_dir_sha256": _prefilter_supervisor_run_dir_sha256(supervisor_run_dir),
        "shard_count": count,
        "worker_count": workers,
        "row_group_batch_size": batch_size,
        "row_group_sharding_policy": shard_policy,
        "segmentation_validation_mode": segment_mode,
        "segmentation_validation_sample_modulus": segment_sample_modulus,
        "max_rows": row_cap,
        "min_free_bytes": free_bytes,
        "expected_row_group_count": int(plan["row_group_count"]),
        "expected_physical_rows": int(plan["expected_physical_rows"]),
        "shards": shards,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    manifest["manifest_hash"] = stable_json_hash(_prefilter_supervisor_manifest_hash_payload(manifest))
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
    )
    return manifest

def build_prefilter_local_parquet_plan_supervisor_shard_commands(
    *,
    manifest: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    config_path: str | Path,
    selected_policy_report_path: str | Path,
    plan_path: str | Path,
    cache_dir: str | Path,
    sparse_report_path: str | Path,
    sparse_model_path: str | Path,
    deterministic_ontology_path: str | Path,
    python_executable: str | Path | None = None,
) -> list[dict[str, Any]]:
    if manifest.get("artifact_kind") != filtering_artifacts.PREFILTER_SUPERVISOR_MANIFEST_KIND:
        raise ValueError("prefilter supervisor manifest kind mismatch")
    if manifest.get("manifest_hash") != stable_json_hash(_prefilter_supervisor_manifest_hash_payload(manifest)):
        raise ValueError("prefilter supervisor manifest hash mismatch")
    filtering_artifacts._require_false_flags(manifest, filtering_artifacts.PUBLIC_FALSE_FLAGS, "prefilter supervisor manifest")
    filtering_artifacts.validate_public_no_raw_payload(manifest)
    executable = str(python_executable or sys.executable)
    shard_count = filtering_artifacts._positive_int(manifest.get("shard_count"), "shard_count")
    row_group_batch_size = filtering_artifacts._positive_int(manifest.get("row_group_batch_size"), "row_group_batch_size")
    min_free_bytes = filtering_artifacts._positive_int(manifest.get("min_free_bytes"), "min_free_bytes")
    row_group_sharding_policy = source_prep._validate_row_group_sharding_policy(
        manifest.get("row_group_sharding_policy", source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL)
    )
    segmentation_validation_mode = _validate_segmentation_validation_mode(
        str(manifest.get("segmentation_validation_mode", "full"))
    )
    segmentation_validation_sample_modulus = _segmentation_sample_modulus(
        int(manifest.get("segmentation_validation_sample_modulus", DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS))
    )
    commands: list[dict[str, Any]] = []
    for shard in manifest.get("shards") or []:
        if not isinstance(shard, Mapping):
            raise ValueError("prefilter supervisor shard must be an object")
        shard_index = int(shard["shard_index"])
        shard_max_rows = None if shard.get("max_rows") is None else filtering_artifacts._nonnegative_int(shard.get("max_rows"), "shard max_rows")
        if shard_max_rows == 0:
            continue
        shard_run_dir = _prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
        command = [
            executable,
            "-m",
            "twc.pretraining.chunk_filtering",
            "prefilter-local-parquet-plan",
            "--config",
            str(config_path),
            "--selected-policy-report",
            str(selected_policy_report_path),
            "--plan",
            str(plan_path),
            "--supervisor-manifest",
            str(Path(supervisor_run_dir) / "prefilter_supervisor_manifest.json"),
            "--run-dir",
            str(shard_run_dir),
            "--cache-dir",
            str(cache_dir),
            "--sparse-report",
            str(sparse_report_path),
            "--sparse-model",
            str(sparse_model_path),
            "--deterministic-ontology",
            str(deterministic_ontology_path),
            "--min-free-bytes",
            str(min_free_bytes),
            "--row-group-batch-size",
            str(row_group_batch_size),
            "--row-group-shard-count",
            str(shard_count),
            "--row-group-shard-index",
            str(shard_index),
            "--row-group-sharding-policy",
            row_group_sharding_policy,
            "--segmentation-validation-mode",
            segmentation_validation_mode,
            "--segmentation-validation-sample-modulus",
            str(segmentation_validation_sample_modulus),
            "--supervisor-manifest-hash",
            str(manifest["manifest_hash"]),
            "--supervisor-command-hash",
            str(shard["command_hash"]),
            "--supervisor-run-dir-sha256",
            str(manifest["supervisor_run_dir_sha256"]),
        ]
        if shard_max_rows is not None:
            command.extend(["--max-rows", str(filtering_artifacts._positive_int(shard_max_rows, "shard max_rows"))])
        commands.append(
            {
                "shard_index": shard_index,
                "shard_run_dir": shard_run_dir,
                "command": command,
                "command_hash": str(shard["command_hash"]),
                "env_overrides": dict(PREFILTER_SUPERVISOR_THREAD_ENV),
            }
        )
    expected_command_count = sum(
        1
        for shard in manifest.get("shards") or []
        if shard.get("max_rows") is None or int(shard.get("max_rows") or 0) > 0
    )
    if len(commands) != expected_command_count:
        raise ValueError("prefilter supervisor command set mismatch")
    return sorted(commands, key=lambda item: int(item["shard_index"]))

def _append_prefilter_supervisor_event(path: Path, payload: Mapping[str, Any]) -> None:
    event = dict(payload)
    event.setdefault("schema_version", 1)
    event.setdefault("timestamp_unix", round(time.time(), 6))
    filtering_artifacts.validate_public_no_raw_payload(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
    filtering_artifacts._chmod_private_file(path)

def _prefilter_supervisor_shard_terminal_status(shard_run_dir: str | Path) -> str:
    receipt_path = Path(shard_run_dir) / "prefilter_cpu_receipt.json"
    if not receipt_path.exists():
        return "missing_receipt"
    try:
        receipt = filtering_artifacts._load_json(receipt_path)
    except (OSError, json.JSONDecodeError):
        return "invalid_receipt"
    if not isinstance(receipt, Mapping):
        return "invalid_receipt"
    status = str(receipt.get("status") or "")
    if status in {"prefilter_shard_ready_for_merge", "compact_fresh_import"}:
        return "ready"
    if status == "stopped":
        return "stopped"
    if status == "prefilter_blocked_errors":
        return "blocked_errors"
    return status or "unknown_receipt_status"

def _w1_start_prefilter_expected_status(receipt: Mapping[str, Any]) -> str:
    if _is_w1_compact_prefilter_receipt(receipt):
        return "compact_fresh_import"
    return "prefilter_shard_ready_for_merge"

def _validate_prefilter_supervisor_ready_shard(
    *,
    shard_run_dir: str | Path,
    shard: Mapping[str, Any],
    manifest: Mapping[str, Any],
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    seen_survivor_candidate_ids: set[str] | None = None,
    seen_private_evidence_ids: set[str] | None = None,
    audit_sqlite_state: bool = True,
) -> dict[str, Any]:
    shard_path = Path(shard_run_dir)
    filtering_artifacts.validate_private_filtering_run_dir(shard_path)
    receipt = filtering_artifacts._load_json(shard_path / "prefilter_cpu_receipt.json")
    if not isinstance(receipt, Mapping):
        raise ValueError("prefilter supervisor shard receipt must be an object")
    expected_status = _w1_start_prefilter_expected_status(receipt)
    compact_receipt = expected_status == "compact_fresh_import"
    prefilter_config_payload = (
        filtering_policy._prefilter_input_config_payload_for_hash(
            config_payload,
            receipt.get("config_hash"),
            label="compact prefilter supervisor shard receipt",
        )
        if compact_receipt
        else config_payload
    )
    proof_ready = _prefilter_receipt_has_survivor_queue_id_proof(receipt)
    id_index_ready = _prefilter_receipt_has_survivor_queue_id_index_proof(receipt)
    needs_cross_shard_id_scan = seen_survivor_candidate_ids is not None or seen_private_evidence_ids is not None
    queue_rows: list[dict[str, Any]] | None = None
    if compact_receipt:
        if not audit_sqlite_state and id_index_ready:
            queue_row_count = _validate_survivor_queue_id_index_ready_file(
                shard_path / "w1_survivor_queue.jsonl",
                allow_empty=True,
                expected_hash=receipt.get("survivor_queue_id_index_hash"),
                expected_row_count=int(receipt.get("survivor_queue_id_index_row_count", receipt.get("survivor_count", 0))),
                expected_candidate_id_set_hash=receipt.get("survivor_candidate_id_set_hash"),
                expected_private_evidence_id_set_hash=receipt.get("survivor_private_evidence_id_set_hash"),
                seen_survivor_candidate_ids=seen_survivor_candidate_ids,
                seen_private_evidence_ids=seen_private_evidence_ids,
            )
        else:
            queue_row_count = _validate_survivor_queue_ready_file(
                shard_path / "w1_survivor_queue.jsonl",
                prefilter_config_payload,
                allow_empty=True,
                seen_survivor_candidate_ids=seen_survivor_candidate_ids,
                seen_private_evidence_ids=seen_private_evidence_ids,
            )
            queue_rows = filtering_artifacts._load_ready_jsonl(shard_path / "w1_survivor_queue.jsonl", "prefilter survivor queue")
            if len(queue_rows) != queue_row_count:
                raise ValueError("compact start state survivor queue count changed during validation")
    elif audit_sqlite_state or not proof_ready:
        queue_row_count = _validate_survivor_queue_ready_file(
            shard_path / "w1_survivor_queue.jsonl",
            config_payload,
            allow_empty=True,
            seen_survivor_candidate_ids=seen_survivor_candidate_ids,
            seen_private_evidence_ids=seen_private_evidence_ids,
        )
    elif needs_cross_shard_id_scan and id_index_ready:
        queue_row_count = _validate_survivor_queue_id_index_ready_file(
            shard_path / "w1_survivor_queue.jsonl",
            allow_empty=True,
            expected_hash=receipt.get("survivor_queue_id_index_hash"),
            expected_row_count=int(receipt.get("survivor_queue_id_index_row_count", receipt.get("survivor_count", 0))),
            expected_candidate_id_set_hash=receipt.get("survivor_candidate_id_set_hash"),
            expected_private_evidence_id_set_hash=receipt.get("survivor_private_evidence_id_set_hash"),
            seen_survivor_candidate_ids=seen_survivor_candidate_ids,
            seen_private_evidence_ids=seen_private_evidence_ids,
        )
    elif needs_cross_shard_id_scan:
        queue_row_count = _validate_survivor_queue_ready_file(
            shard_path / "w1_survivor_queue.jsonl",
            prefilter_config_payload,
            allow_empty=True,
            seen_survivor_candidate_ids=seen_survivor_candidate_ids,
            seen_private_evidence_ids=seen_private_evidence_ids,
        )
    else:
        queue_row_count = int(receipt.get("survivor_queue_row_count", receipt.get("survivor_count", 0)) or 0)
    if (compact_receipt and queue_rows is not None) or audit_sqlite_state or not proof_ready:
        conn = stage_state_v1.open_stage_state_readonly(
            shard_path,
            prefilter_config_payload,
            legacy_retained_config_payload=config_payload,
        )
    else:
        conn = None
    try:
        if conn is None:
            if compact_receipt:
                _validate_w1_compact_receipt_proof_payload(
                    receipt,
                    shard_path,
                    prefilter_config_payload,
                    queue_row_count=queue_row_count,
                    expected_status=expected_status,
                    require_empty_w1_scores=False,
                )
            else:
                _validate_prefilter_ready_receipt_payload(
                    receipt,
                    shard_path,
                    prefilter_config_payload,
                    queue_row_count=queue_row_count,
                    expected_status=expected_status,
                )
        else:
            _validate_prefilter_ready_receipt(
                conn,
                shard_path,
                prefilter_config_payload,
                queue_rows=queue_rows,
                queue_row_count=None if queue_rows is not None else queue_row_count,
                expected_status=expected_status,
            )
        shard_index = int(shard["shard_index"])
        shard_count = int(manifest["shard_count"])
        shard_policy = source_prep._validate_row_group_sharding_policy(
            manifest.get("row_group_sharding_policy", source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL)
        )
        expected_shard_plan = source_prep._shard_local_parquet_prefilter_plan(
            prefilter_plan,
            shard_count=shard_count,
            shard_index=shard_index,
            row_group_sharding_policy=shard_policy,
        )
        expected_fields = _local_parquet_prefilter_plan_receipt_fields(expected_shard_plan)
        for field, expected_value in expected_fields.items():
            if receipt.get(field) != expected_value:
                raise ValueError(f"prefilter supervisor shard receipt {field} mismatch")
        if receipt.get("input_schedule_hash") != expected_shard_plan["schedule_hash"]:
            raise ValueError("prefilter supervisor shard input schedule hash mismatch")
        if int(receipt.get("terminal_rows", -1)) != int(shard["expected_physical_rows"]):
            raise ValueError("prefilter supervisor shard terminal row count mismatch")
        if receipt.get("supervisor_manifest_hash") != manifest["manifest_hash"]:
            raise ValueError("prefilter supervisor shard supervisor manifest hash mismatch")
        if receipt.get("supervisor_command_hash") != shard["command_hash"]:
            raise ValueError("prefilter supervisor shard command hash mismatch")
        if receipt.get("supervisor_run_dir_sha256") != manifest["supervisor_run_dir_sha256"]:
            raise ValueError("prefilter supervisor shard run dir hash mismatch")
        return dict(receipt)
    finally:
        if conn is not None:
            conn.close()

def _request_prefilter_supervisor_shard_stop(shard_run_dir: str | Path) -> None:
    stop_path = Path(shard_run_dir) / "STOP_REQUESTED"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text("prefilter supervisor stop requested\n", encoding="utf-8")
    filtering_artifacts._chmod_private_file(stop_path)

def _clear_prefilter_supervisor_owned_shard_stop(shard_run_dir: str | Path) -> None:
    stop_path = Path(shard_run_dir) / "STOP_REQUESTED"
    if not stop_path.exists():
        return
    try:
        text = stop_path.read_text(encoding="utf-8")
    except OSError:
        return
    if text == "prefilter supervisor stop requested\n":
        stop_path.unlink()

def _prefilter_supervisor_summary(
    *,
    manifest: Mapping[str, Any],
    status: str,
    elapsed_seconds: float,
    shard_status_counts: Mapping[str, int],
    launched_shards: int,
    skipped_ready_shards: int,
    merge_receipt_hash: str | None = None,
) -> dict[str, Any]:
    summary = {
        "artifact_kind": filtering_artifacts.PREFILTER_SUPERVISOR_SUMMARY_KIND,
        "schema_version": 1,
        "status": status,
        "manifest_hash": manifest["manifest_hash"],
        "manifest_mode": manifest["manifest_mode"],
        "merge_allowed": bool(manifest["merge_allowed"]),
        "claims_scope": manifest["claims_scope"],
        "shard_count": int(manifest["shard_count"]),
        "worker_count": int(manifest["worker_count"]),
        "row_group_batch_size": int(manifest["row_group_batch_size"]),
        "row_group_sharding_policy": source_prep._validate_row_group_sharding_policy(
            manifest.get("row_group_sharding_policy", source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL)
        ),
        "segmentation_validation_mode": str(manifest.get("segmentation_validation_mode", "full")),
        "segmentation_validation_sample_modulus": int(
            manifest.get("segmentation_validation_sample_modulus", DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS)
        ),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "launched_shards": int(launched_shards),
        "skipped_ready_shards": int(skipped_ready_shards),
        "shard_status_counts": {str(key): int(value) for key, value in sorted(shard_status_counts.items())},
        "merge_receipt_hash": merge_receipt_hash,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    summary["summary_hash"] = stable_json_hash({key: value for key, value in summary.items() if key != "summary_hash"})
    filtering_artifacts.validate_public_no_raw_payload(summary)
    return summary
