"""Resumable prefilter shard-merge preparation and validation."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import shutil
import stat
from typing import Any

from twc.repro.hashing import stable_json_hash, self_hash_mismatches
from twc.pretraining.filtering import artifacts as filtering_artifacts
from twc.pretraining.filtering import stage_state_v1
from twc.pretraining.filtering.prefilter_records import (
    _validate_prefilter_ready_receipt,
    _validate_survivor_queue_ready_file,
)
from twc.pretraining.filtering.prefilter_supervisor import (
    _local_parquet_prefilter_plan_receipt_fields,
    _prefilter_supervisor_shard_dir,
    _validate_prefilter_supervisor_ready_shard,
    validate_prefilter_local_parquet_plan_supervisor_manifest,
)

PREFILTER_MERGE_SPEC_FILENAME = "prefilter_merge_spec.json"

def _prefilter_supervisor_merge_source_snapshot(
    *,
    shard_dir: Path,
    shard_index: int,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    queue = shard_dir / "w1_survivor_queue.jsonl.ready"
    index = shard_dir / "w1_survivor_queue.ids.tsv.ready"
    receipt_path = shard_dir / "prefilter_cpu_receipt.json"
    snapshot = {
        "shard_index": int(shard_index),
        "stage_receipt_hash": filtering_artifacts._require_sha256(
            receipt.get("stage_receipt_hash"), "stage_receipt_hash"
        ),
        "receipt_file_sha256": filtering_artifacts._sha256_file(receipt_path),
        "survivor_queue_sha256": filtering_artifacts._sha256_file(queue),
        "survivor_queue_id_index_sha256": filtering_artifacts._sha256_file(index),
    }
    for field in (
        "receipt_file_sha256",
        "survivor_queue_sha256",
        "survivor_queue_id_index_sha256",
    ):
        filtering_artifacts._require_sha256(snapshot[field], field)
    if snapshot["survivor_queue_sha256"] != receipt.get("survivor_queue_hash"):
        raise ValueError("prefilter merge source queue hash mismatch")
    if snapshot["survivor_queue_id_index_sha256"] != receipt.get(
        "survivor_queue_id_index_hash"
    ):
        raise ValueError("prefilter merge source queue index hash mismatch")
    return snapshot

def _prefilter_supervisor_merge_spec(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    shard_snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    spec = {
        "artifact_kind": "twc_prefilter_supervisor_merge_spec_v1",
        "schema_version": 1,
        "prefilter_plan_hash": filtering_artifacts._require_sha256(
            prefilter_plan.get("schedule_hash"), "prefilter_plan_hash"
        ),
        "config_hash": filtering_artifacts._require_sha256(
            config_payload.get("config_hash"), "config_hash"
        ),
        "supervisor_manifest_hash": filtering_artifacts._require_sha256(
            manifest.get("manifest_hash"), "supervisor_manifest_hash"
        ),
        "source_shards": [dict(item) for item in shard_snapshots],
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    spec["merge_spec_hash"] = stable_json_hash(spec)
    filtering_artifacts.validate_public_no_raw_payload(spec)
    return spec

def _validate_prefilter_merge_directory(path: Path) -> None:
    try:
        root_metadata = path.lstat()
    except FileNotFoundError:
        raise ValueError("prefilter merge incoming directory is missing") from None
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("prefilter merge incoming path must be a directory")
    if os.name == "posix":
        if int(root_metadata.st_uid) != int(os.geteuid()):
            raise ValueError("prefilter merge incoming directory has the wrong owner")
        if stat.S_IMODE(root_metadata.st_mode) != 0o700:
            raise ValueError("prefilter merge incoming directory must use mode 0700")
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs:
            metadata = (Path(root) / name).lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("prefilter merge incoming tree contains an unsafe directory")
            if os.name == "posix" and (
                int(metadata.st_uid) != int(os.geteuid())
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise ValueError("prefilter merge incoming tree contains an unsafe directory")
        for name in files:
            metadata = (Path(root) / name).lstat()
            filtering_artifacts._validate_private_partial_metadata(metadata)

def _validate_prefilter_merge_spec_file(
    directory: Path,
    expected_spec: Mapping[str, Any],
) -> None:
    spec_path = directory / PREFILTER_MERGE_SPEC_FILENAME
    try:
        filtering_artifacts._validate_owned_private_partial(spec_path)
    except ValueError as exc:
        raise ValueError("prefilter merge spec is missing or unsafe") from exc
    loaded = filtering_artifacts._load_json(spec_path)
    if not isinstance(loaded, Mapping):
        raise ValueError("prefilter merge spec must be an object")
    if self_hash_mismatches(loaded, "merge_spec_hash"):
        raise ValueError("prefilter merge spec hash mismatch")
    if dict(loaded) != dict(expected_spec):
        raise ValueError("prefilter merge spec conflicts with requested merge")

def _prepare_prefilter_merge_incoming(
    incoming_path: Path,
    merge_spec: Mapping[str, Any],
) -> None:
    if incoming_path.exists() or incoming_path.is_symlink():
        if incoming_path.is_symlink() or not incoming_path.is_dir():
            raise ValueError("prefilter merge incoming path is untrusted")
        _validate_prefilter_merge_directory(incoming_path)
        _validate_prefilter_merge_spec_file(incoming_path, merge_spec)
        shutil.rmtree(incoming_path)
        filtering_artifacts._fsync_directory(incoming_path.parent)
    filtering_artifacts._ensure_private_dir(incoming_path)
    filtering_artifacts._write_json_atomic(
        incoming_path / PREFILTER_MERGE_SPEC_FILENAME,
        merge_spec,
    )

def _promote_prefilter_merge_directory(incoming_path: Path, final_path: Path) -> None:
    _validate_prefilter_merge_directory(incoming_path)
    if final_path.exists() or final_path.is_symlink():
        raise ValueError("prefilter merge final path appeared before promotion")
    os.replace(incoming_path, final_path)
    filtering_artifacts._fsync_directory(final_path.parent)

def _prefilter_merged_plan_receipt_fields(
    plan: Mapping[str, Any],
    receipts_by_index: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    shard_count = len(receipts_by_index)
    extra_fields = _local_parquet_prefilter_plan_receipt_fields(plan)
    extra_fields.update({
        "merged_prefilter_shards": True,
        "merged_shard_count": shard_count,
        "merged_shard_indexes": sorted(receipts_by_index),
        "merged_shard_receipt_hashes": sorted(str(receipt["stage_receipt_hash"]) for receipt in receipts_by_index.values()),
        "row_group_sharding_enabled": True,
        "row_group_shard_count": shard_count,
        "row_group_shard_row_group_count": int(plan["row_group_count"]),
        "row_group_shard_expected_physical_rows": int(plan["expected_physical_rows"]),
        "all_rows_row_group_count": int(plan["row_group_count"]),
        "all_rows_expected_physical_rows": int(plan["expected_physical_rows"]),
    })
    if plan["artifact_kind"] == filtering_artifacts.LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND:
        extra_fields.update(
            {
                "all_rows_row_group_count": int(plan["population_row_group_count"]),
                "all_rows_expected_physical_rows": int(plan["population_expected_physical_rows"]),
                "full_corpus_census": False,
                "selected_equals_population": False,
                "claims_scope": "tranche_only",
            }
        )
    return extra_fields

def _validate_existing_prefilter_supervisor_merge(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    filtering_artifacts.validate_private_filtering_run_dir(run_path)
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=prefilter_plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
        expect_merge_allowed=True,
    )
    shard_receipts_by_index: dict[int, dict[str, Any]] = {}
    shard_dirs = []
    for shard in manifest["shards"]:
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
        shard_dirs.append(shard_dir)
        receipt = _validate_prefilter_supervisor_ready_shard(
            shard_run_dir=shard_dir,
            shard=shard,
            manifest=manifest,
            prefilter_plan=prefilter_plan,
            config_payload=config_payload,
        )
        shard_receipts_by_index[int(shard["shard_index"])] = receipt
    if any(path.resolve(strict=False) == run_path.resolve(strict=False) for path in shard_dirs):
        raise ValueError("merged prefilter run dir must be separate from shard run dirs")
    queue_row_count = _validate_survivor_queue_ready_file(
        run_path / "w1_survivor_queue.jsonl",
        config_payload,
        allow_empty=True,
    )
    conn = stage_state_v1.open_stage_state_readonly(run_path, config_payload)
    try:
        _validate_prefilter_ready_receipt(
            conn,
            run_path,
            config_payload,
            queue_row_count=queue_row_count,
        )
        receipt = filtering_artifacts._load_json(run_path / "prefilter_cpu_receipt.json")
    finally:
        conn.close()
    if not isinstance(receipt, Mapping):
        raise ValueError("merged prefilter receipt must be an object")
    filtering_artifacts.validate_public_no_raw_payload(receipt)
    expected_fields = _prefilter_merged_plan_receipt_fields(prefilter_plan, shard_receipts_by_index)
    for field, expected_value in expected_fields.items():
        if receipt.get(field) != expected_value:
            raise ValueError(f"merged prefilter receipt {field} mismatch")
    if receipt.get("input_schedule_hash") != prefilter_plan["schedule_hash"]:
        raise ValueError("merged prefilter input schedule hash mismatch")
    if int(receipt.get("terminal_rows", -1)) != int(prefilter_plan["expected_physical_rows"]):
        raise ValueError("merged prefilter terminal row count mismatch")
    if int(receipt.get("survivor_count", -1)) != sum(
        int(shard_receipt.get("survivor_count", 0)) for shard_receipt in shard_receipts_by_index.values()
    ):
        raise ValueError("merged prefilter survivor count mismatch")
    return dict(receipt)
