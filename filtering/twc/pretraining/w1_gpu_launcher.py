from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.classifiers.window_mil import (
    W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2,
    W1_ATTN_IMPLEMENTATION_SDPA,
    SUPPORTED_W1_ATTN_IMPLEMENTATIONS,
    normalize_w1_attn_implementation,
)
from twc.data.decision_schema import stable_json_hash
from twc.pretraining.chunk_filtering import (
    ASSEMBLED_STAGE_REPORT_KIND,
    W1_PRIVATE_SCORE_STATE_ARCHIVE_KIND,
    W1_PERSIST_STRATEGY_PER_BATCH,
    W1_PERSIST_STRATEGY_ROW_THRESHOLD,
    _load_and_validate_selected_policy_report,
    _validate_sharded_w1_handoff_payload,
    build_chunk_filtering_config_payload,
    validate_locked_chunk_filtering_config_payload,
    validate_w1_fa2_runtime_compatibility_manifest,
    validate_w1_score_run_comparison_report,
)
from twc.pretraining.w1_gpu_canary import (
    BOOTSTRAP_STATUS_KIND,
    CommandResult,
    CommandRunner,
)


W1_PRODUCTION_LAUNCH_PLAN_KIND = "twc_w1_production_launch_plan_v1"
W1_PRODUCTION_LAUNCH_RECEIPT_KIND = "twc_w1_production_launch_receipt_v1"
W1_PRODUCTION_STOP_REPORT_KIND = "twc_w1_production_stop_report_v1"
W1_PRODUCTION_COPYBACK_MANIFEST_KIND = "twc_w1_production_copyback_manifest_v1"
W1_PRODUCTION_SUMMARY_KIND = "twc_w1_production_projection_summary_v1"

FALSE_FLAGS = {
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

FORBIDDEN_COPYBACK_SUFFIXES = {
    ".sqlite",
    ".db",
    ".parquet",
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".arrow",
    ".zst",
}
FORBIDDEN_COPYBACK_NAMES = {
    "filtering_stage.sqlite",
    "filtering_stage_state.sqlite",
    "w1_survivor_queue.jsonl",
    "w1_scores.jsonl",
    "w1_scores.jsonl.ready",
    "w1_scores.jsonl.part",
}

W1_TARGET_FULL_SECONDS = 4 * 60 * 60
W1_ENGINEERING_DOCS_PER_SECOND = 2000.0
RANK_TAIL_IMBALANCE_RATIO_THRESHOLD = 1.5
SUMMARY_COMMAND_ELAPSED_TOLERANCE_RATIO = 1.05
SUMMARY_COMMAND_ELAPSED_TOLERANCE_SECONDS = 1.0
W1_ASSEMBLY_DEFERRED_STATUSES = {
    "accepted_token_target_reached_partial",
    "queue_exhausted_before_token_target",
}
MODULO_RANK_ASSIGNMENT_STRATEGY = "shard_index_modulo_v1"
LPT_RANK_ASSIGNMENT_STRATEGY = "shard_survivor_count_lpt_v1"
EXPLICIT_SHARD_SELECTION_MODE = "explicit_shard_indexes_v1"


@dataclass(frozen=True)
class W1ProductionLaunchConfig:
    code_root: Path
    expected_commit: str
    run_root: Path
    supervisor_run_dir: Path
    handoff_path: Path
    config_path: Path
    selected_policy_report: Path
    window_mil_report: Path
    window_mil_model: Path
    output_report_path: Path
    private_archive_dir: Path | None = None
    gpu_devices: tuple[str, ...] = ("0",)
    batch_size: int = 512
    max_windows_per_device_batch: int | None = None
    synchronize_cuda_telemetry: bool = False
    w1_attn_implementation: str = W1_ATTN_IMPLEMENTATION_SDPA
    w1_prepare_pipeline: str = "serial"
    max_estimated_windows_per_prepare_batch: int | None = None
    w1_prepare_queue_depth: int = 1
    w1_pin_memory: str = "auto"
    w1_tokenizers_parallelism: str = "inherit"
    w1_persist_commit_row_threshold: int | None = None
    enforce_accepted_token_target: bool = True
    w1_backend_runtime_compatibility_manifest: Path | None = None
    w1_score_comparator_report: Path | None = None
    bootstrap_status: Path | None = None
    python_executable: str = sys.executable
    run_id: str | None = None
    dry_run: bool = False
    rank_assignment_strategy: str = MODULO_RANK_ASSIGNMENT_STRATEGY
    required_persistent_output_root: Path | None = None


def plan_w1_production_launch(config: W1ProductionLaunchConfig) -> dict[str, Any]:
    code_root = config.code_root.expanduser().resolve(strict=False)
    run_root = config.run_root.expanduser().resolve(strict=False)
    supervisor_run_dir = config.supervisor_run_dir.expanduser().resolve(strict=False)
    _require_repo_root(code_root)
    _require_not_under_public_code(run_root, code_root, "run_root")
    if not config.gpu_devices:
        raise ValueError("at least one GPU device is required")
    if len(set(config.gpu_devices)) != len(config.gpu_devices):
        raise ValueError("gpu_devices must be unique")
    if int(config.batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if config.max_windows_per_device_batch is not None and int(config.max_windows_per_device_batch) < 1:
        raise ValueError("max_windows_per_device_batch must be positive")
    w1_attn_implementation = normalize_w1_attn_implementation(config.w1_attn_implementation)
    if config.w1_prepare_pipeline not in {"serial", "bounded-tokenize-overlap"}:
        raise ValueError("w1_prepare_pipeline must be serial or bounded-tokenize-overlap")
    if (
        config.max_estimated_windows_per_prepare_batch is not None
        and int(config.max_estimated_windows_per_prepare_batch) < 1
    ):
        raise ValueError("max_estimated_windows_per_prepare_batch must be positive")
    if int(config.w1_prepare_queue_depth) != 1:
        raise ValueError("w1_prepare_queue_depth must be 1")
    if config.w1_pin_memory not in {"auto", "on", "off"}:
        raise ValueError("w1_pin_memory must be auto, on, or off")
    if config.w1_tokenizers_parallelism not in {"inherit", "false", "true"}:
        raise ValueError("w1_tokenizers_parallelism must be inherit, false, or true")
    if config.w1_persist_commit_row_threshold is not None and int(config.w1_persist_commit_row_threshold) < 1:
        raise ValueError("w1_persist_commit_row_threshold must be positive")
    if config.rank_assignment_strategy not in {MODULO_RANK_ASSIGNMENT_STRATEGY, LPT_RANK_ASSIGNMENT_STRATEGY}:
        raise ValueError("rank_assignment_strategy must be shard_index_modulo_v1 or shard_survivor_count_lpt_v1")
    w1_persist_strategy = _w1_persist_strategy_for_threshold(config.w1_persist_commit_row_threshold)

    config_payload = build_chunk_filtering_config_payload(yaml.safe_load(Path(config.config_path).read_text(encoding="utf-8")))
    validate_locked_chunk_filtering_config_payload(config_payload)
    selected_policy = _load_and_validate_selected_policy_report(config.selected_policy_report, config_payload, required=True)
    handoff = _load_json(config.handoff_path)
    _validate_sharded_w1_handoff_payload(
        handoff,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
        require_live_shard_state=False,
    )
    if (supervisor_run_dir / "STOP_REQUESTED").exists():
        raise ValueError("stale STOP_REQUESTED exists in supervisor_run_dir")
    shard_count = int(handoff["shard_count"])
    if len(config.gpu_devices) > shard_count:
        raise ValueError("gpu_count exceeds shard_count")

    run_root.mkdir(parents=True, exist_ok=True)
    private_archive_dir = (
        config.private_archive_dir.expanduser().resolve(strict=False)
        if config.private_archive_dir is not None
        else run_root / "w1_private_score_state_archive"
    )
    accepted_token_ledger_path = (
        run_root / "w1_accepted_token_ledger.sqlite"
        if bool(config.enforce_accepted_token_target) and len(config.gpu_devices) > 1
        else None
    )
    required_persistent_output_root = _validate_required_persistent_output_root(
        config.required_persistent_output_root,
        mutable_output_paths={
            "run_root": run_root,
            "supervisor_run_dir": supervisor_run_dir,
            "output_report_path": config.output_report_path.expanduser().resolve(strict=False),
            "private_archive_dir": private_archive_dir,
            **({"accepted_token_ledger_path": accepted_token_ledger_path} if accepted_token_ledger_path is not None else {}),
        },
    )
    run_id = config.run_id or stable_json_hash(
        {
            "code_commit": config.expected_commit,
            "handoff_hash": handoff["handoff_hash"],
            "gpu_devices": list(config.gpu_devices),
            "batch_size": int(config.batch_size),
            "w1_attn_implementation": w1_attn_implementation,
            "w1_prepare_pipeline": config.w1_prepare_pipeline,
            "max_estimated_windows_per_prepare_batch": config.max_estimated_windows_per_prepare_batch,
            "w1_prepare_queue_depth": int(config.w1_prepare_queue_depth),
            "w1_pin_memory": config.w1_pin_memory,
            "w1_tokenizers_parallelism": config.w1_tokenizers_parallelism,
            "w1_persist_commit_row_threshold": config.w1_persist_commit_row_threshold,
            "enforce_accepted_token_target": bool(config.enforce_accepted_token_target),
            "rank_assignment_strategy": config.rank_assignment_strategy,
        }
    )[:16]
    all_shards = sorted(list(handoff["shards"]), key=lambda item: int(item["shard_index"]))
    rank_assignments = _rank_shard_assignments(
        all_shards,
        gpu_count=len(config.gpu_devices),
        strategy=config.rank_assignment_strategy,
    )
    rank_assignment_hash = _rank_assignment_hash(
        strategy=config.rank_assignment_strategy,
        handoff_hash=str(handoff["handoff_hash"]),
        gpu_devices=config.gpu_devices,
        rank_assignments=rank_assignments,
    )
    modulo_assignments = _rank_shard_assignments(
        all_shards,
        gpu_count=len(config.gpu_devices),
        strategy=MODULO_RANK_ASSIGNMENT_STRATEGY,
    )
    commands = _production_command_specs(
        config,
        run_root=run_root,
        handoff=handoff,
        rank_assignments=rank_assignments,
        rank_assignment_hash=rank_assignment_hash,
        accepted_token_ledger_path=accepted_token_ledger_path,
    )
    rank_survivor_counts = [
        int(command["expected_assigned_survivor_count"])
        for command in commands
        if command.get("role") == "rank"
    ]
    rank_shard_counts = [
        int(command["expected_assigned_shard_count"])
        for command in commands
        if command.get("role") == "rank"
    ]
    archive_command = _archive_private_score_state_command_spec(
        config,
        private_archive_dir=private_archive_dir,
        run_root=run_root,
        accepted_token_ledger_path=accepted_token_ledger_path,
    )
    commands_with_archive = (
        [*commands[:-1], archive_command, commands[-1]]
        if commands and commands[-1].get("role") == "assemble"
        else [*commands, archive_command]
    )
    payload: dict[str, Any] = {
        "artifact_kind": W1_PRODUCTION_LAUNCH_PLAN_KIND,
        "schema_version": 1,
        "status": "w1_production_launch_plan_ready",
        "run_id": run_id,
        "code_commit": config.expected_commit,
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "selected_policy_freeze_report_hash": selected_policy["report_hash"],
        "w1_report_hash": config_payload["window_mil_report_hash"],
        "w1_model_file_sha256": config_payload["window_mil_model_file_sha256"],
        "window_mil_precision": config_payload["window_mil_precision"],
        "w1_attn_implementation": w1_attn_implementation,
        "w1_backend_runtime_compatibility_required": (
            w1_attn_implementation == W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2
        ),
        "torch_compile": "off_deferred",
        "w1_threshold": float(config_payload["window_mil_threshold"]),
        "sharded_w1_handoff_hash": handoff["handoff_hash"],
        "shard_count": shard_count,
        "survivor_count": int(handoff["survivor_count"]),
        "gpu_devices": list(config.gpu_devices),
        "gpu_count": len(config.gpu_devices),
        "expected_rank_count": len(rank_survivor_counts),
        "partition_strategy": config.rank_assignment_strategy,
        "rank_assignment_strategy": config.rank_assignment_strategy,
        "rank_assignment_hash": rank_assignment_hash,
        "rank_assignment_weight_field": "survivor_count",
        "rank_assignment_survivor_count_skew": _numeric_skew(rank_survivor_counts),
        "rank_assignment_shard_count_skew": _numeric_skew(rank_shard_counts),
        "modulo_rank_assignment_survivor_count_skew": _numeric_skew(
            [_assigned_survivor_count(assigned) for assigned in modulo_assignments]
        ),
        "modulo_rank_assignment_shard_count_skew": _numeric_skew(
            [len(assigned) for assigned in modulo_assignments]
        ),
        "batch_size": int(config.batch_size),
        "max_windows_per_device_batch": (
            None if config.max_windows_per_device_batch is None else int(config.max_windows_per_device_batch)
        ),
        "synchronize_cuda_telemetry": bool(config.synchronize_cuda_telemetry),
        "w1_prepare_pipeline": config.w1_prepare_pipeline,
        "max_estimated_windows_per_prepare_batch": (
            None
            if config.max_estimated_windows_per_prepare_batch is None
            else int(config.max_estimated_windows_per_prepare_batch)
        ),
        "w1_prepare_queue_depth": int(config.w1_prepare_queue_depth),
        "w1_pin_memory": config.w1_pin_memory,
        "w1_tokenizers_parallelism": config.w1_tokenizers_parallelism,
        "w1_persist_strategy": w1_persist_strategy,
        "w1_persist_commit_row_threshold": (
            None
            if config.w1_persist_commit_row_threshold is None
            else int(config.w1_persist_commit_row_threshold)
        ),
        "accepted_token_target_enforced": bool(config.enforce_accepted_token_target),
        "accepted_token_target_lower_bound": int(config_payload["accepted_token_target_lower_bound"]),
        "accepted_token_ledger_enabled": accepted_token_ledger_path is not None,
        "accepted_token_ledger_path_hash": _path_hash(accepted_token_ledger_path),
        "required_persistent_output_root_hash": _path_hash(required_persistent_output_root),
        "mutable_output_storage": {
            "status": (
                "persistent_output_root_enforced"
                if required_persistent_output_root is not None
                else "persistent_output_root_not_required"
            ),
            "required_persistent_output_root_hash": _path_hash(required_persistent_output_root),
            "path_hashes": {
                "run_root": _path_hash(run_root),
                "supervisor_run_dir": _path_hash(supervisor_run_dir),
                "output_report_path": _path_hash(config.output_report_path),
                "private_archive_dir": _path_hash(private_archive_dir),
                **(
                    {"accepted_token_ledger_path": _path_hash(accepted_token_ledger_path)}
                    if accepted_token_ledger_path is not None
                    else {}
                ),
            },
            "private_w1_score_state_persists_on_required_root": required_persistent_output_root is not None,
            "operator_outputs_persist_on_required_root": required_persistent_output_root is not None,
        },
        "w1_runtime_control_envelope": {
            "w1_attn_implementation_requested": w1_attn_implementation,
            "w1_tokenizers_parallelism_requested": config.w1_tokenizers_parallelism,
            "w1_persist_strategy": w1_persist_strategy,
            "w1_persist_commit_row_threshold": (
                None
                if config.w1_persist_commit_row_threshold is None
                else int(config.w1_persist_commit_row_threshold)
            ),
            "w1_prepare_pipeline_requested": config.w1_prepare_pipeline,
            "torch_compile": "off_deferred",
        },
        "w1_backend_runtime_compatibility_manifest_path_hash": _path_hash(
            config.w1_backend_runtime_compatibility_manifest
        ),
        "w1_score_comparator_report_path_hash": _path_hash(config.w1_score_comparator_report),
        "supervisor_run_dir_hash": _path_hash(supervisor_run_dir),
        "run_root_hash": _path_hash(run_root),
        "output_report_path_hash": _path_hash(config.output_report_path),
        "private_score_state_archive_dir_hash": _path_hash(private_archive_dir),
        "private_score_state_archive_command": archive_command["argv"],
        "commands": commands_with_archive,
        "command_count": len(commands_with_archive),
        "stop_command": f"touch {supervisor_run_dir / 'STOP_REQUESTED'}",
        "copyback_allowlist": [
            "launch_plan.json",
            "launcher_receipt.json",
            "commands.jsonl",
            "rank_*_w1_summary.json",
            "w1_sharded_summary.json",
            "w1_assembled_report.json",
        ],
        "teardown_reminder": "terminate paid GPU pod after no-raw copyback; retained disks remain billable until explicitly terminated",
        **FALSE_FLAGS,
    }
    payload["plan_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "plan_hash"})
    validate_public_no_raw_payload(payload)
    _write_json_atomic(run_root / "launch_plan.json", payload)
    return payload


def launch_w1_production(
    config: W1ProductionLaunchConfig,
    *,
    bootstrap_status: str | Path | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    run_root = config.run_root.expanduser().resolve(strict=False)
    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "process_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_log = run_root / "commands.jsonl"
    plan = plan_w1_production_launch(config)
    bootstrap_source = bootstrap_status or config.bootstrap_status
    if bootstrap_source is None:
        raise ValueError("bootstrap_status is required for W1 production launch")
    bootstrap_path = Path(bootstrap_source)
    _validate_bootstrap_status(bootstrap_path, config=config, plan=plan)
    if not config.dry_run and normalize_w1_attn_implementation(config.w1_attn_implementation) != W1_ATTN_IMPLEMENTATION_SDPA:
        config_payload = build_chunk_filtering_config_payload(
            yaml.safe_load(Path(config.config_path).read_text(encoding="utf-8"))
        )
        readiness_blockers: list[str] = []
        _validate_w1_score_comparator_evidence(
            config,
            config_payload=config_payload,
            blocking_reasons=readiness_blockers,
        )
        if readiness_blockers:
            raise ValueError(
                "W1 backend score comparator readiness evidence is required before non-default backend launch: "
                + ",".join(sorted(set(readiness_blockers)))
            )
    _verify_clean_commit(config, command_runner=command_runner, command_log=command_log, logs_dir=logs_dir)

    start = time.monotonic()
    executed: list[dict[str, Any]] = []
    if config.dry_run:
        for command in plan["commands"]:
            _log_planned_command(command_log, command)
        status = "w1_production_launch_dry_run"
    else:
        rank_commands = [command for command in plan["commands"] if command["role"] == "rank"]
        tail_commands = [command for command in plan["commands"] if command["role"] != "rank"]
        if command_runner is None and len(rank_commands) > 1:
            executed.extend(_run_parallel_command_specs(rank_commands, command_log=command_log, logs_dir=logs_dir, cwd=config.code_root))
        else:
            for command in rank_commands:
                executed.append(
                    _run_command_spec(
                        command,
                        command_runner=command_runner,
                        command_log=command_log,
                        logs_dir=logs_dir,
                        cwd=config.code_root,
                    )
                )
        for command in tail_commands:
            if command.get("role") == "assemble" and _run_root_deferred_assembly_status(run_root) is not None:
                _log_skipped_command(command_log, command, reason="w1_assembly_deferred_for_token_target_review")
                executed.append(
                    {
                        "label": str(command.get("label") or command["role"]),
                        "role": command["role"],
                        "rank": command.get("rank"),
                        "returncode": 0,
                        "elapsed_seconds": 0.0,
                        "skipped": True,
                    }
                )
                continue
            executed.append(
                _run_command_spec(
                    command,
                    command_runner=command_runner,
                    command_log=command_log,
                    logs_dir=logs_dir,
                    cwd=config.code_root,
                )
            )
        status = "w1_production_launch_complete"

    payload: dict[str, Any] = {
        "artifact_kind": W1_PRODUCTION_LAUNCH_RECEIPT_KIND,
        "schema_version": 1,
        "status": status,
        "run_id": plan["run_id"],
        "code_commit": config.expected_commit,
        "plan_hash": plan["plan_hash"],
        "command_count": len(plan["commands"]),
        "executed_command_count": len(executed),
        "planned_command_hashes": {
            str(command["label"]): str(command["argv_hash"])
            for command in plan["commands"]
        },
        "command_log_hash": _sha256_file(command_log) if command_log.exists() else None,
        "elapsed_seconds": round(time.monotonic() - start, 6),
        "dry_run": bool(config.dry_run),
        **FALSE_FLAGS,
    }
    payload["receipt_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "receipt_hash"})
    validate_public_no_raw_payload(payload)
    _write_json_atomic(run_root / "launcher_receipt.json", payload)
    return payload


def stop_w1_production(*, run_root: str | Path, supervisor_run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve(strict=False)
    supervisor = Path(supervisor_run_dir).expanduser().resolve(strict=False)
    stop_path = supervisor / "STOP_REQUESTED"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text("W1 production stop requested\n", encoding="utf-8")
    payload: dict[str, Any] = {
        "artifact_kind": W1_PRODUCTION_STOP_REPORT_KIND,
        "schema_version": 1,
        "status": "stop_requested",
        "run_root_hash": _path_hash(root),
        "supervisor_run_dir_hash": _path_hash(supervisor),
        "stop_path_hash": _path_hash(stop_path),
        **FALSE_FLAGS,
    }
    payload["report_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "report_hash"})
    validate_public_no_raw_payload(payload)
    if root.exists():
        _write_json_atomic(root / "stop_report.json", payload)
    return payload


def collect_w1_production_copyback(config: W1ProductionLaunchConfig) -> dict[str, Any]:
    run_root = config.run_root.expanduser().resolve(strict=False)
    _raise_for_forbidden_copyback_files(run_root)
    candidates = [
        run_root / "launch_plan.json",
        run_root / "launcher_receipt.json",
        run_root / "commands.jsonl",
        run_root / "stop_report.json",
        run_root / "w1_sharded_summary.json",
        config.output_report_path.expanduser().resolve(strict=False),
    ]
    candidates.extend(sorted(run_root.glob("rank_*_w1_summary.json")))
    files = []
    for path in candidates:
        if path.exists():
            _require_safe_copyback_path(path)
            _validate_safe_copyback_artifact(path)
            files.append(
                {
                    "relative_path": _copyback_relative_path(path, run_root=run_root),
                    "file_name": path.name,
                    "path_hash": _path_hash(path),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    payload: dict[str, Any] = {
        "artifact_kind": W1_PRODUCTION_COPYBACK_MANIFEST_KIND,
        "schema_version": 1,
        "status": "w1_production_copyback_manifest_ready",
        "run_root_hash": _path_hash(run_root),
        "file_count": len(files),
        "files": files,
        **FALSE_FLAGS,
    }
    payload["manifest_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    validate_public_no_raw_payload(payload)
    _write_json_atomic(run_root / "copyback_manifest.json", payload)
    return payload


def summarize_w1_production_run(
    config: W1ProductionLaunchConfig,
    *,
    hourly_gpu_cost_usd: float,
    min_scored_docs: int = 1,
    max_input_wait_fraction: float = 0.5,
) -> dict[str, Any]:
    run_root = config.run_root.expanduser().resolve(strict=False)
    handoff = _load_json(config.handoff_path)
    config_payload = build_chunk_filtering_config_payload(yaml.safe_load(Path(config.config_path).read_text(encoding="utf-8")))
    report_set = _load_w1_summary_report_set(run_root)
    plan = _load_launch_plan_if_present(run_root)
    command_rows = _load_command_log_rows(run_root)
    command_rows_by_label = _command_rows_by_label(command_rows)
    blocking_reasons: list[str] = []
    reports: list[Mapping[str, Any]] = []
    rank_reports: list[Mapping[str, Any]] = []
    final_report: Mapping[str, Any] | None = None
    expected_rank_count = len(config.gpu_devices)
    observed_rank_ids: list[int] = []
    missing_rank_ids: list[int] = []
    unexpected_rank_summary_paths: list[str] = []
    rank_command_elapsed_values: list[float] = []
    rank_summary_elapsed_values: list[float] = []
    rank_input_wait_fractions: list[float] = []
    rank_assignment_strategy = MODULO_RANK_ASSIGNMENT_STRATEGY
    rank_assignment_hash: str | None = None
    expected_all_shard_indexes = sorted(int(shard["shard_index"]) for shard in list(handoff.get("shards") or []))

    if plan is None:
        blocking_reasons.append("missing_launch_plan_contract")
        rank_reports = list(report_set["rank_reports"])
        final_report = report_set["final_report"]
        reports = rank_reports if len(config.gpu_devices) > 1 and rank_reports else ([final_report] if final_report else [])
        rank_summary_elapsed_values = [
            float(report.get("elapsed_seconds", 0.0) or 0.0)
            for report in reports
        ]
    else:
        if plan.get("artifact_kind") != W1_PRODUCTION_LAUNCH_PLAN_KIND:
            blocking_reasons.append("launch_plan_contract_mismatch")
        if plan.get("sharded_w1_handoff_hash") != handoff.get("handoff_hash"):
            blocking_reasons.append("launch_plan_contract_mismatch")
        rank_assignment_strategy = str(plan.get("rank_assignment_strategy") or plan.get("partition_strategy") or "")
        rank_assignment_hash = str(plan.get("rank_assignment_hash") or "") or None
        planned_commands = list(plan.get("commands") or [])
        _validate_launcher_receipt(
            run_root,
            plan=plan,
            config=config,
            planned_commands=planned_commands,
            blocking_reasons=blocking_reasons,
        )
        rank_commands = [command for command in planned_commands if command.get("role") == "rank"]
        final_commands = [command for command in planned_commands if command.get("role") == "final_resume"]
        tail_commands = [command for command in planned_commands if command.get("role") not in {"rank", "final_resume"}]
        expected_rank_count = len(rank_commands)
        expected_rank_paths = {str(Path(str(command["summary_out"])).resolve(strict=False)) for command in rank_commands}
        observed_rank_paths = {
            str(Path(str(item["path"])).resolve(strict=False))
            for item in report_set["rank_report_records"]
        }
        unexpected_rank_summary_paths = sorted(observed_rank_paths - expected_rank_paths)
        if unexpected_rank_summary_paths:
            blocking_reasons.append("unexpected_rank_summary_receipts")
        for command in rank_commands:
            rank = int(command.get("rank", -1))
            summary_path = Path(str(command["summary_out"]))
            report = _load_json(summary_path) if summary_path.exists() else None
            if report is None:
                missing_rank_ids.append(rank)
                blocking_reasons.append("missing_rank_summary_receipts")
                continue
            observed_rank_ids.append(rank)
            rank_reports.append(report)
            reports.append(report)
            _validate_planned_summary_report(
                report,
                command=command,
                handoff=handoff,
                config_payload=config_payload,
                blocking_reasons=blocking_reasons,
                role_prefix="rank",
            )
            command_elapsed = _planned_command_elapsed(
                command,
                command_rows_by_label=command_rows_by_label,
                blocking_reasons=blocking_reasons,
                command_kind="scoring",
            )
            summary_elapsed = float(report.get("elapsed_seconds", 0.0) or 0.0)
            rank_summary_elapsed_values.append(summary_elapsed)
            if command_elapsed is not None:
                rank_command_elapsed_values.append(command_elapsed)
                _validate_summary_elapsed_not_larger_than_command(
                    summary_elapsed,
                    command_elapsed,
                    blocking_reasons=blocking_reasons,
                )
            denominator = command_elapsed if command_elapsed and command_elapsed > 0 else summary_elapsed
            if denominator > 0:
                rank_input_wait_fractions.append(
                    float(dict(report.get("w1_runtime_telemetry") or {}).get("score_input_fetch_seconds", 0.0) or 0.0)
                    / denominator
                )

        if len(set(observed_rank_ids)) != len(observed_rank_ids):
            blocking_reasons.append("duplicate_rank_summary_receipts")
        observed_partition_indexes = [
            int(item)
            for report in rank_reports
            for item in list(report.get("partition_shard_indexes") or [])
        ]
        if len(set(observed_partition_indexes)) != len(observed_partition_indexes):
            blocking_reasons.append("rank_partition_duplicate_shards")
        if sorted(set(observed_partition_indexes)) != expected_all_shard_indexes:
            blocking_reasons.append("rank_partition_missing_shards")

        if final_commands:
            final_command = final_commands[0]
            final_path = Path(str(final_command["summary_out"]))
            final_report = _load_json(final_path) if final_path.exists() else None
            if final_report is None:
                blocking_reasons.append("missing_final_resume_summary")
            else:
                _validate_planned_summary_report(
                    final_report,
                    command=final_command,
                    handoff=handoff,
                    config_payload=config_payload,
                    blocking_reasons=blocking_reasons,
                    role_prefix="final_resume",
                )
                final_elapsed = _planned_command_elapsed(
                    final_command,
                    command_rows_by_label=command_rows_by_label,
                    blocking_reasons=blocking_reasons,
                    command_kind="scoring",
                )
                final_summary_elapsed = float(final_report.get("elapsed_seconds", 0.0) or 0.0)
                if final_elapsed is not None:
                    _validate_summary_elapsed_not_larger_than_command(
                        final_summary_elapsed,
                        final_elapsed,
                        blocking_reasons=blocking_reasons,
                    )
                rank_scored_total = sum(int(report.get("scored_count", 0) or 0) for report in rank_reports)
                if int(final_report.get("scored_count", 0) or 0) > rank_scored_total:
                    blocking_reasons.append("final_resume_repaired_partition_gap")

        deferred_assembly_status = _w1_report_deferred_assembly_status(final_report)
        if deferred_assembly_status is None:
            deferred_assembly_status = next(
                (
                    status
                    for status in (_w1_report_deferred_assembly_status(report) for report in rank_reports)
                    if status is not None
                ),
                None,
            )
        if deferred_assembly_status is not None:
            blocking_reasons.append("w1_assembly_deferred_for_token_target_review")

        for command in tail_commands:
            if command.get("role") == "assemble" and deferred_assembly_status is not None:
                continue
            command_elapsed = _planned_command_elapsed(
                command,
                command_rows_by_label=command_rows_by_label,
                blocking_reasons=blocking_reasons,
                command_kind="launch",
            )
            if command_elapsed is None:
                continue
            if command.get("role") == "assemble":
                _validate_assembled_report(
                    config.output_report_path.expanduser().resolve(strict=False),
                    handoff=handoff,
                    config_payload=config_payload,
                    blocking_reasons=blocking_reasons,
                )
            elif command.get("role") == "archive_private_score_state":
                _validate_private_score_state_archive_manifest(
                    command,
                    config_payload=config_payload,
                    blocking_reasons=blocking_reasons,
                )

    scored = sum(int(report.get("scored_count", 0) or 0) for report in reports)
    elapsed_candidates = list(rank_command_elapsed_values)
    elapsed_candidates.extend(rank_summary_elapsed_values)
    elapsed = max(elapsed_candidates) if elapsed_candidates else 0.0
    telemetry = _aggregate_summary_telemetry(reports)
    envelope_hashes = sorted(
        {
            str(item)
            for report in reports
            for item in list(report.get("w1_runtime_envelope_hashes") or [])
            if item is not None
        }
    )
    if not envelope_hashes:
        envelope_hash = next(
            (str(report.get("w1_runtime_envelope_hash")) for report in reports if report.get("w1_runtime_envelope_hash")),
            None,
        )
        envelope_hashes = [envelope_hash] if envelope_hash else []
    input_wait_fraction = max(rank_input_wait_fractions) if rank_input_wait_fractions else (
        float(telemetry.get("score_input_fetch_seconds", 0.0) or 0.0) / elapsed
        if elapsed > 0
        else None
    )
    if scored < int(min_scored_docs):
        blocking_reasons.append("too_few_scored_docs")
    if elapsed <= 0:
        blocking_reasons.append("missing_elapsed_seconds")
    if len(envelope_hashes) != 1:
        blocking_reasons.append("mixed_runtime_envelopes")
    if int(telemetry.get("score_window_forward_window_count", 0) or 0) <= 0:
        blocking_reasons.append("zero_forward_windows")
    if int(telemetry.get("score_oom_split_count", 0) or 0) > 0 or int(
        telemetry.get("score_oom_failed_attempt_count", 0) or 0
    ) > 0:
        blocking_reasons.append("oom_backoff_observed")
    if not all(_summary_report_observed_cuda(report) for report in reports):
        blocking_reasons.append("cpu_fallback_observed")
    if not all(_summary_report_observed_locked_bf16(report) for report in reports):
        blocking_reasons.append("missing_bf16_precision_proof")
    if input_wait_fraction is not None and input_wait_fraction > float(max_input_wait_fraction):
        blocking_reasons.append("input_wait_fraction_high")
    _validate_w1_score_comparator_evidence(
        config,
        config_payload=config_payload,
        blocking_reasons=blocking_reasons,
    )

    docs_per_second = scored / elapsed if elapsed > 0 else 0.0
    projected_full_seconds = int(handoff["survivor_count"]) / docs_per_second if docs_per_second > 0 else None
    projected_full_hours = projected_full_seconds / 3600.0 if projected_full_seconds is not None else None
    required_docs_per_second = int(handoff["survivor_count"]) / float(W1_TARGET_FULL_SECONDS)
    projected_four_hour_pass = (
        projected_full_seconds is not None and projected_full_seconds <= float(W1_TARGET_FULL_SECONDS)
    )
    projected_engineering_pass = docs_per_second >= float(W1_ENGINEERING_DOCS_PER_SECOND)
    if projected_full_seconds is None:
        blocking_reasons.append("missing_four_hour_projection")
    elif not projected_four_hour_pass:
        blocking_reasons.append("four_hour_projection_not_met")
    if not projected_engineering_pass:
        blocking_reasons.append("projected_engineering_throughput_not_met")
    rank_elapsed_skew = _numeric_skew(rank_command_elapsed_values or rank_summary_elapsed_values)
    rank_elapsed_ratio = rank_elapsed_skew.get("ratio")
    if (
        isinstance(rank_elapsed_ratio, float)
        and rank_elapsed_ratio > RANK_TAIL_IMBALANCE_RATIO_THRESHOLD
        and (not projected_four_hour_pass or not projected_engineering_pass)
    ):
        blocking_reasons.append("rank_tail_imbalance_high")
    projected_cost = (
        projected_full_seconds / 3600.0 * float(hourly_gpu_cost_usd) * len(config.gpu_devices)
        if projected_full_seconds is not None
        else None
    )
    payload: dict[str, Any] = {
        "artifact_kind": W1_PRODUCTION_SUMMARY_KIND,
        "schema_version": 1,
        "status": "w1_production_projection_ready" if not blocking_reasons else "w1_production_projection_blocked",
        "blocking_reasons": sorted(set(blocking_reasons)),
        "sharded_w1_handoff_hash": handoff.get("handoff_hash"),
        "handoff_survivor_count": int(handoff["survivor_count"]),
        "observed_scored_count": int(scored),
        "observed_elapsed_seconds": round(elapsed, 6),
        "observed_docs_per_second": docs_per_second,
        "target_full_seconds": W1_TARGET_FULL_SECONDS,
        "required_docs_per_second": required_docs_per_second,
        "engineering_docs_per_second": W1_ENGINEERING_DOCS_PER_SECOND,
        "projected_full_hours": projected_full_hours,
        "projected_four_hour_pass": bool(projected_four_hour_pass),
        "projected_engineering_pass": bool(projected_engineering_pass),
        "rank_assignment_strategy": rank_assignment_strategy,
        "rank_assignment_hash": rank_assignment_hash,
        "rank_assignment_survivor_count_skew": plan.get("rank_assignment_survivor_count_skew") if plan else None,
        "rank_assignment_shard_count_skew": plan.get("rank_assignment_shard_count_skew") if plan else None,
        "expected_rank_count": int(expected_rank_count),
        "observed_rank_ids": sorted(observed_rank_ids),
        "missing_rank_ids": sorted(missing_rank_ids),
        "rank_summary_count": len(rank_reports),
        "final_resume_summary_present": final_report is not None,
        "rank_command_elapsed_seconds_max": max(rank_command_elapsed_values) if rank_command_elapsed_values else None,
        "rank_summary_elapsed_seconds_max": max(rank_summary_elapsed_values) if rank_summary_elapsed_values else None,
        "rank_elapsed_skew": rank_elapsed_skew,
        "unexpected_rank_summary_receipt_count": len(unexpected_rank_summary_paths),
        "input_wait_fraction": input_wait_fraction,
        "gpu_count": len(config.gpu_devices),
        "gpu_cost_per_hour_dollars": float(hourly_gpu_cost_usd),
        "projected_full_seconds": projected_full_seconds,
        "projected_full_cost_usd": projected_cost,
        "w1_runtime_envelope_hashes": envelope_hashes,
        "w1_runtime_telemetry": telemetry,
        **FALSE_FLAGS,
    }
    payload["summary_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "summary_hash"})
    validate_public_no_raw_payload(payload)
    _write_json_atomic(run_root / "projection_summary.json", payload)
    return payload


def _production_command_specs(
    config: W1ProductionLaunchConfig,
    *,
    run_root: Path,
    handoff: Mapping[str, Any],
    rank_assignments: Sequence[Sequence[Mapping[str, Any]]],
    rank_assignment_hash: str,
    accepted_token_ledger_path: Path | None,
) -> list[dict[str, Any]]:
    gpu_count = len(config.gpu_devices)
    all_shards = sorted(list(handoff["shards"]), key=lambda item: int(item["shard_index"]))
    commands: list[dict[str, Any]] = []
    if gpu_count == 1:
        commands.append(
            _w1_score_command_spec(
                config,
                label="rank_00",
                role="rank",
                rank=0,
                gpu_device=config.gpu_devices[0],
                summary_path=run_root / "w1_sharded_summary.json",
                handoff_hash=str(handoff["handoff_hash"]),
                assigned_shards=all_shards,
                shard_index_modulus=1,
                shard_index_remainder=0,
                selection_mode=MODULO_RANK_ASSIGNMENT_STRATEGY,
                rank_assignment_hash=rank_assignment_hash,
                expected_status="w1_sharded_scored",
                accepted_token_ledger_path=accepted_token_ledger_path,
            )
        )
    else:
        for rank, gpu_device in enumerate(config.gpu_devices):
            assigned = list(rank_assignments[rank])
            selection_mode = (
                EXPLICIT_SHARD_SELECTION_MODE
                if config.rank_assignment_strategy == LPT_RANK_ASSIGNMENT_STRATEGY
                else MODULO_RANK_ASSIGNMENT_STRATEGY
            )
            commands.append(
                _w1_score_command_spec(
                    config,
                    label=f"rank_{rank:02d}",
                    role="rank",
                    rank=rank,
                    gpu_device=gpu_device,
                    summary_path=run_root / f"rank_{rank:02d}_w1_summary.json",
                    handoff_hash=str(handoff["handoff_hash"]),
                    assigned_shards=assigned,
                    shard_index_modulus=(gpu_count if selection_mode == MODULO_RANK_ASSIGNMENT_STRATEGY else None),
                    shard_index_remainder=(rank if selection_mode == MODULO_RANK_ASSIGNMENT_STRATEGY else None),
                    selection_mode=selection_mode,
                    rank_assignment_hash=rank_assignment_hash,
                    expected_status="w1_sharded_partition_scored",
                    accepted_token_ledger_path=accepted_token_ledger_path,
                )
            )
        commands.append(
            _w1_score_command_spec(
                config,
                label="final_resume",
                role="final_resume",
                rank=None,
                gpu_device=config.gpu_devices[0],
                summary_path=run_root / "w1_sharded_summary.json",
                handoff_hash=str(handoff["handoff_hash"]),
                assigned_shards=all_shards,
                shard_index_modulus=1,
                shard_index_remainder=0,
                selection_mode=MODULO_RANK_ASSIGNMENT_STRATEGY,
                rank_assignment_hash=rank_assignment_hash,
                expected_status="w1_sharded_scored",
                accepted_token_ledger_path=accepted_token_ledger_path,
            )
        )
    commands.append(_assemble_command_spec(config, gpu_device=config.gpu_devices[0]))
    return commands


def _w1_persist_strategy_for_threshold(threshold: int | None) -> str:
    return W1_PERSIST_STRATEGY_ROW_THRESHOLD if threshold is not None else W1_PERSIST_STRATEGY_PER_BATCH


def _w1_score_command_spec(
    config: W1ProductionLaunchConfig,
    *,
    label: str,
    role: str,
    rank: int | None,
    gpu_device: str,
    summary_path: Path,
    handoff_hash: str,
    assigned_shards: Sequence[Mapping[str, Any]],
    shard_index_modulus: int | None,
    shard_index_remainder: int | None,
    selection_mode: str,
    rank_assignment_hash: str,
    expected_status: str,
    accepted_token_ledger_path: Path | None,
) -> dict[str, Any]:
    w1_persist_strategy = _w1_persist_strategy_for_threshold(config.w1_persist_commit_row_threshold)
    argv = [
        str(config.python_executable),
        "-m",
        "twc.pretraining.chunk_filtering",
        "w1-score-sharded",
        "--config",
        str(config.config_path),
        "--selected-policy-report",
        str(config.selected_policy_report),
        "--handoff",
        str(config.handoff_path),
        "--supervisor-run-dir",
        str(config.supervisor_run_dir),
        "--window-mil-report",
        str(config.window_mil_report),
        "--window-mil-model",
        str(config.window_mil_model),
        "--summary-out",
        str(summary_path),
        "--device",
        "cuda:0",
        "--require-cuda",
        "--batch-size",
        str(int(config.batch_size)),
    ]
    if config.max_windows_per_device_batch is not None:
        argv.extend(["--max-windows-per-device-batch", str(int(config.max_windows_per_device_batch))])
    if config.synchronize_cuda_telemetry:
        argv.append("--synchronize-cuda-telemetry")
    w1_attn_implementation = normalize_w1_attn_implementation(config.w1_attn_implementation)
    argv.extend(["--w1-attn-implementation", w1_attn_implementation])
    if config.w1_prepare_pipeline != "serial":
        argv.extend(["--w1-prepare-pipeline", config.w1_prepare_pipeline])
        argv.extend(["--w1-prepare-queue-depth", str(int(config.w1_prepare_queue_depth))])
        argv.extend(["--w1-pin-memory", config.w1_pin_memory])
    if config.max_estimated_windows_per_prepare_batch is not None:
        argv.extend([
            "--w1-max-estimated-windows-per-prepare-batch",
            str(int(config.max_estimated_windows_per_prepare_batch)),
        ])
    if config.w1_tokenizers_parallelism != "inherit":
        argv.extend(["--w1-tokenizers-parallelism", config.w1_tokenizers_parallelism])
    if config.w1_persist_commit_row_threshold is not None:
        argv.extend(["--w1-persist-commit-row-threshold", str(int(config.w1_persist_commit_row_threshold))])
    if config.enforce_accepted_token_target:
        argv.append("--enforce-accepted-token-target")
    if accepted_token_ledger_path is not None:
        argv.extend(["--accepted-token-ledger", str(accepted_token_ledger_path)])
    assigned_indexes = [int(shard["shard_index"]) for shard in assigned_shards]
    if selection_mode == EXPLICIT_SHARD_SELECTION_MODE:
        for shard_index in assigned_indexes:
            argv.extend(["--assigned-shard-index", str(int(shard_index))])
    elif shard_index_modulus != 1 or shard_index_remainder != 0:
        argv.extend(["--shard-index-modulus", str(int(shard_index_modulus))])
        argv.extend(["--shard-index-remainder", str(int(shard_index_remainder))])
    argv.extend(["--rank-assignment-hash", rank_assignment_hash])
    assigned_survivors = sum(int(shard.get("survivor_count", 0)) for shard in assigned_shards)
    expected_statuses = [expected_status]
    if config.enforce_accepted_token_target:
        expected_statuses.extend(
            [
                "accepted_token_target_reached_partial",
                "queue_exhausted_before_token_target",
            ]
        )
    expected_statuses = sorted(set(expected_statuses))
    return {
        "label": label,
        "role": role,
        "rank": None if rank is None else int(rank),
        "gpu_device": str(gpu_device),
        "env": {
            **{"CUDA_VISIBLE_DEVICES": str(gpu_device)},
            **(
                {"TOKENIZERS_PARALLELISM": config.w1_tokenizers_parallelism}
                if config.w1_tokenizers_parallelism != "inherit"
                else {}
            ),
        },
        "summary_out": str(summary_path),
        "expected_status": expected_status,
        "expected_statuses": expected_statuses,
        "expected_assigned_shard_indexes": assigned_indexes,
        "expected_assigned_shard_count": len(assigned_indexes),
        "expected_assigned_survivor_count": int(assigned_survivors),
        "sharded_w1_handoff_hash": handoff_hash,
        "partition_selection_mode": selection_mode,
        "shard_index_modulus": None if shard_index_modulus is None else int(shard_index_modulus),
        "shard_index_remainder": None if shard_index_remainder is None else int(shard_index_remainder),
        "rank_assignment_hash": rank_assignment_hash,
        "w1_attn_implementation": w1_attn_implementation,
        "torch_compile": "off_deferred",
        "w1_prepare_pipeline": config.w1_prepare_pipeline,
        "max_estimated_windows_per_prepare_batch": (
            None
            if config.max_estimated_windows_per_prepare_batch is None
            else int(config.max_estimated_windows_per_prepare_batch)
        ),
        "w1_prepare_queue_depth": int(config.w1_prepare_queue_depth),
        "w1_pin_memory": config.w1_pin_memory,
        "w1_tokenizers_parallelism": config.w1_tokenizers_parallelism,
        "w1_persist_strategy": w1_persist_strategy,
        "w1_persist_commit_row_threshold": (
            None
            if config.w1_persist_commit_row_threshold is None
            else int(config.w1_persist_commit_row_threshold)
        ),
        "accepted_token_target_enforced": bool(config.enforce_accepted_token_target),
        "accepted_token_ledger_enabled": accepted_token_ledger_path is not None,
        "accepted_token_ledger_path_hash": _path_hash(accepted_token_ledger_path),
        "w1_runtime_control_envelope": {
            "w1_attn_implementation_requested": w1_attn_implementation,
            "w1_tokenizers_parallelism_requested": config.w1_tokenizers_parallelism,
            "w1_persist_strategy": w1_persist_strategy,
            "w1_persist_commit_row_threshold": (
                None
                if config.w1_persist_commit_row_threshold is None
                else int(config.w1_persist_commit_row_threshold)
            ),
            "w1_prepare_pipeline_requested": config.w1_prepare_pipeline,
            "torch_compile": "off_deferred",
        },
        "argv": argv,
        "argv_hash": stable_json_hash(argv),
    }


def _assemble_command_spec(config: W1ProductionLaunchConfig, *, gpu_device: str) -> dict[str, Any]:
    argv = [
        str(config.python_executable),
        "-m",
        "twc.pretraining.chunk_filtering",
        "assemble-sharded-report",
        "--config",
        str(config.config_path),
        "--selected-policy-report",
        str(config.selected_policy_report),
        "--handoff",
        str(config.handoff_path),
        "--supervisor-run-dir",
        str(config.supervisor_run_dir),
        "--report-out",
        str(config.output_report_path),
    ]
    return {
        "label": "assemble",
        "role": "assemble",
        "rank": None,
        "gpu_device": str(gpu_device),
        "env": {"CUDA_VISIBLE_DEVICES": str(gpu_device)},
        "summary_out": None,
        "skip_if_w1_summary_statuses": sorted(W1_ASSEMBLY_DEFERRED_STATUSES),
        "argv": argv,
        "argv_hash": stable_json_hash(argv),
    }


def _archive_private_score_state_command_spec(
    config: W1ProductionLaunchConfig,
    *,
    private_archive_dir: Path,
    run_root: Path,
    accepted_token_ledger_path: Path | None,
) -> dict[str, Any]:
    argv = [
        str(config.python_executable),
        "-m",
        "twc.pretraining.chunk_filtering",
        "archive-w1-private-score-state",
        "--config",
        str(config.config_path),
        "--selected-policy-report",
        str(config.selected_policy_report),
        "--run-dir",
        str(config.supervisor_run_dir),
        "--archive-dir",
        str(private_archive_dir),
        "--out",
        str(run_root / "w1_private_score_state_manifest.json"),
    ]
    if accepted_token_ledger_path is not None:
        argv.extend(["--extra-state-file", str(accepted_token_ledger_path)])
    return {
        "label": "archive_private_score_state",
        "role": "archive_private_score_state",
        "rank": None,
        "gpu_device": None,
        "env": {},
        "summary_out": str(run_root / "w1_private_score_state_manifest.json"),
        "expected_status": "w1_private_score_state_archived",
        "private_score_state_archive_dir_hash": _path_hash(private_archive_dir),
        "accepted_token_ledger_archive_required": accepted_token_ledger_path is not None,
        "accepted_token_ledger_path_hash": _path_hash(accepted_token_ledger_path),
        "argv": argv,
        "argv_hash": stable_json_hash(argv),
    }


def _validate_bootstrap_status(
    bootstrap_status: Path,
    *,
    config: W1ProductionLaunchConfig,
    plan: Mapping[str, Any],
) -> None:
    if not str(bootstrap_status):
        raise ValueError("bootstrap_status is required for W1 production launch")
    payload = _load_json(bootstrap_status)
    if payload.get("artifact_kind") != BOOTSTRAP_STATUS_KIND:
        raise ValueError("bootstrap status kind mismatch")
    if payload.get("status") != "w1_gpu_bootstrap_ready":
        raise ValueError("bootstrap status is not ready")
    if payload.get("code_commit") != config.expected_commit or payload.get("expected_commit") != config.expected_commit:
        raise ValueError("bootstrap status commit mismatch")
    if payload.get("require_cuda") is not True:
        raise ValueError("bootstrap status must require CUDA")
    cuda = payload.get("cuda")
    if not isinstance(cuda, Mapping) or cuda.get("cuda_available") is not True:
        raise ValueError("bootstrap status CUDA unavailable")
    if str(cuda.get("device_type") or "").lower() != "cuda":
        raise ValueError("bootstrap status device is not CUDA")
    if cuda.get("bf16_supported") is not True:
        raise ValueError("bootstrap status BF16 support missing")
    if payload.get("model_load_smoke_skipped") is not False:
        raise ValueError("bootstrap status model load smoke was skipped")
    bootstrap_attn = payload.get("w1_attn_implementation")
    expected_attn = plan.get("w1_attn_implementation") or W1_ATTN_IMPLEMENTATION_SDPA
    if bootstrap_attn is None and expected_attn == W1_ATTN_IMPLEMENTATION_SDPA:
        bootstrap_attn = W1_ATTN_IMPLEMENTATION_SDPA
    try:
        if normalize_w1_attn_implementation(str(bootstrap_attn)) != normalize_w1_attn_implementation(str(expected_attn)):
            raise ValueError("bootstrap status attention backend mismatch")
    except ValueError as exc:
        if "attention backend mismatch" in str(exc):
            raise
        raise ValueError("bootstrap status attention backend mismatch") from exc
    artifact_validation = payload.get("artifact_validation")
    if not isinstance(artifact_validation, Mapping) or artifact_validation.get("status") != "w1_artifacts_validated":
        raise ValueError("bootstrap artifact validation missing")
    expected = {
        "selected_policy_hash": plan["selected_policy_hash"],
        "selected_policy_report_hash": plan["selected_policy_report_hash"],
        "w1_report_hash": plan["w1_report_hash"],
        "w1_model_file_sha256": plan["w1_model_file_sha256"],
    }
    for key, value in expected.items():
        if artifact_validation.get(key) != value:
            raise ValueError(f"bootstrap artifact validation {key} mismatch")
    if artifact_validation.get("window_mil_precision") not in (None, plan["window_mil_precision"]):
        raise ValueError("bootstrap artifact validation window_mil_precision mismatch")
    validate_public_no_raw_payload(payload)


def _verify_clean_commit(
    config: W1ProductionLaunchConfig,
    *,
    command_runner: CommandRunner | None,
    command_log: Path,
    logs_dir: Path,
) -> None:
    rev = _run_logged_argv(
        ["git", "rev-parse", "HEAD"],
        command_runner=command_runner,
        command_log=command_log,
        logs_dir=logs_dir,
        cwd=config.code_root,
        label="git_rev_parse",
        env={},
    ).stdout.strip()
    if rev != config.expected_commit:
        raise ValueError("launcher code commit mismatch")
    dirty = _run_logged_argv(
        ["git", "status", "--short"],
        command_runner=command_runner,
        command_log=command_log,
        logs_dir=logs_dir,
        cwd=config.code_root,
        label="git_status",
        env={},
    ).stdout.strip()
    if dirty:
        raise ValueError("launcher public code repo is dirty")


def _run_command_spec(
    command: Mapping[str, Any],
    *,
    command_runner: CommandRunner | None,
    command_log: Path,
    logs_dir: Path,
    cwd: Path,
) -> dict[str, Any]:
    result = _run_logged_argv(
        list(command["argv"]),
        command_runner=command_runner,
        command_log=command_log,
        logs_dir=logs_dir,
        cwd=cwd,
        label=str(command.get("label") or command["role"]),
        env=dict(command.get("env") or {}),
    )
    return {
        "label": str(command.get("label") or command["role"]),
        "role": command["role"],
        "rank": command.get("rank"),
        "returncode": result.returncode,
        "elapsed_seconds": result.elapsed_seconds,
    }


def _run_logged_argv(
    argv: list[str],
    *,
    command_runner: CommandRunner | None,
    command_log: Path,
    logs_dir: Path,
    cwd: Path,
    label: str,
    env: Mapping[str, str],
) -> CommandResult:
    safe_label = _safe_label(label)
    stdout_path = logs_dir / f"{safe_label}.out.log"
    stderr_path = logs_dir / f"{safe_label}.err.log"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if command_runner is None:
        start = time.monotonic()
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env={**os.environ, **dict(env)},
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        result = CommandResult(
            argv=list(argv),
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_seconds=round(time.monotonic() - start, 6),
        )
    else:
        result = command_runner(
            list(argv),
            cwd=cwd,
            env={**os.environ, **dict(env)},
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            label=label,
        )
    _append_command_log(command_log, label=label, argv=argv, cwd=cwd, env=env, started=started, result=result, stdout_path=stdout_path, stderr_path=stderr_path)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}: {' '.join(argv)}")
    return result


def _run_parallel_command_specs(
    commands: Sequence[Mapping[str, Any]],
    *,
    command_log: Path,
    logs_dir: Path,
    cwd: Path,
) -> list[dict[str, Any]]:
    processes = []
    for command in commands:
        label = str(command.get("label") or command["role"])
        safe_label = _safe_label(label)
        stdout_path = logs_dir / f"{safe_label}.out.log"
        stderr_path = logs_dir / f"{safe_label}.err.log"
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        start = time.monotonic()
        process = subprocess.Popen(
            list(command["argv"]),
            cwd=str(cwd),
            env={**os.environ, **dict(command.get("env") or {})},
            text=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        processes.append((command, label, started, start, process, stdout_handle, stderr_handle, stdout_path, stderr_path))
    results = []
    failures = []
    for command, label, started, start, process, stdout_handle, stderr_handle, stdout_path, stderr_path in processes:
        returncode = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        result = CommandResult(
            argv=list(command["argv"]),
            returncode=int(returncode),
            elapsed_seconds=round(time.monotonic() - start, 6),
        )
        _append_command_log(command_log, label=label, argv=list(command["argv"]), cwd=cwd, env=dict(command.get("env") or {}), started=started, result=result, stdout_path=stdout_path, stderr_path=stderr_path)
        results.append(
            {
                "label": label,
                "role": command["role"],
                "rank": command.get("rank"),
                "returncode": int(returncode),
                "elapsed_seconds": result.elapsed_seconds,
            }
        )
        if returncode != 0:
            failures.append(label)
    if failures:
        raise RuntimeError(f"parallel W1 rank command(s) failed: {', '.join(failures)}")
    return results


def _log_planned_command(command_log: Path, command: Mapping[str, Any]) -> None:
    row = {
        "label": str(command.get("label") or command["role"]),
        "execution_mode": "planned_only",
        "argv": list(command["argv"]),
        "argv_hash": command["argv_hash"],
        "env": dict(command.get("env") or {}),
        "returncode": None,
        "elapsed_seconds": 0.0,
    }
    _append_jsonl(command_log, row)


def _log_skipped_command(command_log: Path, command: Mapping[str, Any], *, reason: str) -> None:
    _append_jsonl(
        command_log,
        {
            "label": str(command.get("label") or command["role"]),
            "execution_mode": "skipped",
            "skip_reason": str(reason),
            "argv": list(command["argv"]),
            "argv_hash": command["argv_hash"],
            "env": dict(command.get("env") or {}),
            "returncode": 0,
            "elapsed_seconds": 0.0,
        },
    )


def _append_command_log(
    command_log: Path,
    *,
    label: str,
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    started: str,
    result: CommandResult,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    _append_jsonl(
        command_log,
        {
            "label": label,
            "argv": list(argv),
            "argv_hash": stable_json_hash(list(argv)),
            "env": dict(env),
            "cwd_hash": _path_hash(cwd),
            "started_utc": started,
            "returncode": int(result.returncode),
            "elapsed_seconds": float(result.elapsed_seconds),
            "stdout_log_hash": _sha256_file(stdout_path) if stdout_path.exists() else None,
            "stderr_log_hash": _sha256_file(stderr_path) if stderr_path.exists() else None,
        },
    )


def _load_w1_summary_report_set(run_root: Path) -> dict[str, Any]:
    final = run_root / "w1_sharded_summary.json"
    rank_records = [
        {"path": path, "report": _load_json(path)}
        for path in sorted(run_root.glob("rank_*_w1_summary.json"))
    ]
    return {
        "final_report": _load_json(final) if final.exists() else None,
        "rank_reports": [dict(record["report"]) for record in rank_records],
        "rank_report_records": rank_records,
    }


def _w1_report_deferred_assembly_status(report: Mapping[str, Any] | None) -> str | None:
    if not isinstance(report, Mapping):
        return None
    status = str(report.get("status") or "")
    return status if status in W1_ASSEMBLY_DEFERRED_STATUSES else None


def _run_root_deferred_assembly_status(run_root: Path) -> str | None:
    final = run_root / "w1_sharded_summary.json"
    if not final.exists():
        return None
    try:
        report = _load_json(final)
    except ValueError:
        return None
    return _w1_report_deferred_assembly_status(report)


def _load_launch_plan_if_present(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "launch_plan.json"
    return _load_json(path) if path.exists() else None


def _load_command_log_rows(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "commands.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _command_rows_by_label(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    by_label: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = str(row.get("label") or "")
        if not label:
            continue
        by_label.setdefault(label, []).append(row)
    return by_label


def _planned_command_elapsed(
    command: Mapping[str, Any],
    *,
    command_rows_by_label: Mapping[str, Sequence[Mapping[str, Any]]],
    blocking_reasons: list[str],
    command_kind: str,
) -> float | None:
    label = str(command.get("label") or command.get("role") or "")
    rows = list(command_rows_by_label.get(label) or [])
    missing_reason = f"missing_{command_kind}_command_elapsed"
    failed_reason = f"{command_kind}_command_failed"
    if not rows:
        blocking_reasons.append(missing_reason)
        return None
    matching_rows = [
        row
        for row in rows
        if row.get("argv_hash") == command.get("argv_hash")
    ]
    if not matching_rows:
        blocking_reasons.append("command_log_argv_hash_mismatch")
        return None
    successful_rows = [
        row
        for row in matching_rows
        if int(row.get("returncode", 1) or 0) == 0 and float(row.get("elapsed_seconds", 0.0) or 0.0) > 0.0
    ]
    elapsed_values = [
        float(row.get("elapsed_seconds", 0.0) or 0.0)
        for row in matching_rows
        if float(row.get("elapsed_seconds", 0.0) or 0.0) > 0.0
    ]
    if not successful_rows:
        if any(int(row.get("returncode", 1) or 0) != 0 for row in matching_rows):
            blocking_reasons.append(failed_reason)
        blocking_reasons.append(missing_reason)
        return None
    elapsed = sum(elapsed_values)
    if elapsed <= 0.0:
        blocking_reasons.append(missing_reason)
        return None
    return elapsed


def _validate_summary_elapsed_not_larger_than_command(
    summary_elapsed: float,
    command_elapsed: float,
    *,
    blocking_reasons: list[str],
) -> None:
    allowed = (
        float(command_elapsed) * SUMMARY_COMMAND_ELAPSED_TOLERANCE_RATIO
        + SUMMARY_COMMAND_ELAPSED_TOLERANCE_SECONDS
    )
    if float(summary_elapsed) > allowed:
        blocking_reasons.append("summary_elapsed_mismatch")


def _validate_planned_summary_report(
    report: Mapping[str, Any],
    *,
    command: Mapping[str, Any],
    handoff: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    blocking_reasons: list[str],
    role_prefix: str,
) -> None:
    expected_statuses = set(str(item) for item in list(command.get("expected_statuses") or []))
    if not expected_statuses and command.get("expected_status") is not None:
        expected_statuses.add(str(command.get("expected_status")))
    if str(report.get("status")) not in expected_statuses:
        blocking_reasons.append(f"{role_prefix}_summary_status_mismatch")
    if report.get("sharded_w1_handoff_hash") != handoff.get("handoff_hash"):
        blocking_reasons.append("rank_partition_coverage_mismatch")
    if report.get("config_hash") not in (None, config_payload.get("config_hash")):
        blocking_reasons.append("runtime_config_mismatch")
    if report.get("selected_policy_hash") not in (None, config_payload.get("selected_policy_hash")):
        blocking_reasons.append("runtime_config_mismatch")
    if report.get("selected_policy_report_hash") not in (None, config_payload.get("selected_policy_report_hash")):
        blocking_reasons.append("runtime_config_mismatch")
    _validate_w1_runtime_envelope_binding(
        report,
        config_payload=config_payload,
        blocking_reasons=blocking_reasons,
    )
    envelope = report.get("w1_runtime_envelope")
    expected_prepare_pipeline = command.get("w1_prepare_pipeline")
    if (
        expected_prepare_pipeline == "bounded-tokenize-overlap"
        and isinstance(envelope, Mapping)
        and envelope.get("w1_prepare_pipeline") != "bounded_tokenize_overlap_v1"
    ):
        blocking_reasons.append("prepare_pipeline_disabled")
    expected_attn_implementation = normalize_w1_attn_implementation(
        str(command.get("w1_attn_implementation") or W1_ATTN_IMPLEMENTATION_SDPA)
    )
    if isinstance(envelope, Mapping):
        requested_attn_raw = envelope.get("w1_attn_implementation_requested")
        observed_attn_raw = envelope.get("w1_attn_implementation_observed")
        if requested_attn_raw is None and expected_attn_implementation == W1_ATTN_IMPLEMENTATION_SDPA:
            requested_attn = W1_ATTN_IMPLEMENTATION_SDPA
        else:
            try:
                requested_attn = normalize_w1_attn_implementation(str(requested_attn_raw))
            except ValueError:
                requested_attn = None
        if observed_attn_raw is None and requested_attn is not None:
            observed_attn = requested_attn
        else:
            try:
                observed_attn = normalize_w1_attn_implementation(str(observed_attn_raw))
            except ValueError:
                observed_attn = None
        if requested_attn != expected_attn_implementation or observed_attn != expected_attn_implementation:
            blocking_reasons.append("attention_backend_mismatch")
        if expected_attn_implementation == W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2 and (
            requested_attn != W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2
            or observed_attn != W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2
        ):
            blocking_reasons.append("attention_backend_unavailable")
    expected_indexes = [int(item) for item in list(command.get("expected_assigned_shard_indexes") or [])]
    observed_indexes = [int(item) for item in list(report.get("partition_shard_indexes") or [])]
    if observed_indexes != expected_indexes:
        blocking_reasons.append("rank_partition_coverage_mismatch")
    if report.get("partition_selection_mode") != command.get("partition_selection_mode"):
        blocking_reasons.append("rank_partition_coverage_mismatch")
    if command.get("rank_assignment_hash") is not None and report.get("rank_assignment_hash") != command.get(
        "rank_assignment_hash"
    ):
        blocking_reasons.append("rank_assignment_hash_mismatch")
    if _int_value(report.get("selected_shard_count"), default=-1) != int(command.get("expected_assigned_shard_count", -2)):
        blocking_reasons.append("rank_partition_coverage_mismatch")
    if _int_value(report.get("selected_survivor_count"), default=-1) != int(
        command.get("expected_assigned_survivor_count", -2)
    ):
        blocking_reasons.append("rank_partition_coverage_mismatch")
    if not _optional_int_matches(report.get("shard_index_modulus"), command.get("shard_index_modulus")):
        blocking_reasons.append("rank_partition_coverage_mismatch")
    if not _optional_int_matches(report.get("shard_index_remainder"), command.get("shard_index_remainder")):
        blocking_reasons.append("rank_partition_coverage_mismatch")
    if _int_value(report.get("scored_count"), default=-1) != int(command.get("expected_assigned_survivor_count", -2)):
        blocking_reasons.append(
            "rank_scored_count_mismatch"
            if role_prefix == "rank"
            else "final_resume_scored_count_mismatch"
        )


def _validate_w1_score_comparator_evidence(
    config: W1ProductionLaunchConfig,
    *,
    config_payload: Mapping[str, Any],
    blocking_reasons: list[str],
) -> None:
    expected_attn_implementation = normalize_w1_attn_implementation(config.w1_attn_implementation)
    if expected_attn_implementation == W1_ATTN_IMPLEMENTATION_SDPA:
        return
    compatibility_hash = _validate_w1_backend_runtime_compatibility_evidence(
        config,
        config_payload=config_payload,
        blocking_reasons=blocking_reasons,
    )
    report_path = config.w1_score_comparator_report
    if report_path is None:
        blocking_reasons.append("missing_w1_score_comparator_report")
        return
    resolved = Path(report_path).expanduser().resolve(strict=False)
    if not resolved.exists():
        blocking_reasons.append("missing_w1_score_comparator_report")
        return
    try:
        report = _load_json(resolved)
        validate_w1_score_run_comparison_report(report, config_payload=config_payload)
    except ValueError:
        blocking_reasons.append("w1_score_comparator_report_invalid")
        return
    if report.get("status") != "w1_score_run_comparison_ready" or report.get("score_stability_ready") is not True:
        blocking_reasons.append("w1_score_comparator_failed")
        for reason in list(report.get("blocking_reasons") or []):
            blocking_reasons.append(f"w1_score_comparator_{reason}")
        return
    comparison = report.get("comparison")
    if not isinstance(comparison, Mapping):
        blocking_reasons.append("w1_score_comparator_report_invalid")
        return
    try:
        requested = normalize_w1_attn_implementation(
            str(comparison.get("w1_attn_implementation_requested") or "")
        )
        observed = normalize_w1_attn_implementation(
            str(comparison.get("w1_attn_implementation_observed") or "")
        )
    except ValueError:
        blocking_reasons.append("w1_score_comparator_backend_mismatch")
        return
    if requested != expected_attn_implementation or observed != expected_attn_implementation:
        blocking_reasons.append("w1_score_comparator_backend_mismatch")
    compatibility = report.get("comparison_backend_runtime_compatibility")
    if not isinstance(compatibility, Mapping):
        blocking_reasons.append("w1_score_comparator_runtime_compatibility_missing")
    elif compatibility_hash is not None and compatibility.get("manifest_hash") != compatibility_hash:
        blocking_reasons.append("w1_score_comparator_runtime_compatibility_mismatch")


def _validate_w1_backend_runtime_compatibility_evidence(
    config: W1ProductionLaunchConfig,
    *,
    config_payload: Mapping[str, Any],
    blocking_reasons: list[str],
) -> str | None:
    expected_attn_implementation = normalize_w1_attn_implementation(config.w1_attn_implementation)
    if expected_attn_implementation == W1_ATTN_IMPLEMENTATION_SDPA:
        return None
    manifest_path = config.w1_backend_runtime_compatibility_manifest
    if manifest_path is None:
        blocking_reasons.append("missing_w1_backend_runtime_compatibility_manifest")
        return None
    resolved = Path(manifest_path).expanduser().resolve(strict=False)
    if not resolved.exists():
        blocking_reasons.append("missing_w1_backend_runtime_compatibility_manifest")
        return None
    try:
        manifest = _load_json(resolved)
        validate_w1_fa2_runtime_compatibility_manifest(
            manifest,
            config_payload=config_payload,
            expected_code_commit=config.expected_commit,
        )
    except ValueError:
        blocking_reasons.append("w1_backend_runtime_compatibility_manifest_invalid")
        return None
    try:
        requested = normalize_w1_attn_implementation(
            str(manifest.get("w1_attn_implementation_requested") or "")
        )
        observed = normalize_w1_attn_implementation(
            str(manifest.get("w1_attn_implementation_observed") or "")
        )
    except ValueError:
        blocking_reasons.append("w1_backend_runtime_compatibility_backend_mismatch")
        return None
    if requested != expected_attn_implementation or observed != expected_attn_implementation:
        blocking_reasons.append("w1_backend_runtime_compatibility_backend_mismatch")
        return None
    return str(manifest["manifest_hash"])


def _int_value(value: Any, *, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _optional_int_matches(observed: Any, expected: Any) -> bool:
    if expected is None:
        return observed is None
    try:
        return int(observed) == int(expected)
    except (TypeError, ValueError):
        return False


def _summary_report_observed_cuda(report: Mapping[str, Any]) -> bool:
    telemetry = report.get("w1_runtime_telemetry")
    if not isinstance(telemetry, Mapping):
        return False
    observed_types: list[str] = []
    if telemetry.get("cuda_device_observed") is True:
        observed_types.append("cuda")
    elif telemetry.get("cuda_device_observed") is False:
        observed_types.append("non_cuda")
    device_types = telemetry.get("scorer_device_types")
    if isinstance(device_types, Sequence) and not isinstance(device_types, (str, bytes)):
        observed_types.extend(str(item).lower() for item in device_types if str(item))
    scorer_device = telemetry.get("scorer_device")
    if isinstance(scorer_device, Mapping):
        device_type = str(scorer_device.get("device_type") or "").lower()
        if device_type:
            observed_types.append(device_type)
    return bool(observed_types) and all(item == "cuda" for item in observed_types)


def _summary_report_observed_locked_bf16(report: Mapping[str, Any]) -> bool:
    envelope = report.get("w1_runtime_envelope")
    if not isinstance(envelope, Mapping):
        return False
    if str(envelope.get("window_mil_precision") or "").lower() != "amp_bf16":
        return False
    runtime = envelope.get("window_mil_precision_runtime")
    if isinstance(runtime, Mapping):
        runtime_mode = str(runtime.get("mode") or runtime.get("precision_mode") or "").lower()
        if runtime_mode and runtime_mode != "amp_bf16":
            return False
    return bool(envelope.get("window_mil_precision_policy_hash"))


def _validate_w1_runtime_envelope_binding(
    report: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
    blocking_reasons: list[str],
) -> None:
    envelope = report.get("w1_runtime_envelope")
    if not isinstance(envelope, Mapping):
        blocking_reasons.append("missing_bf16_precision_proof")
        return
    expected_values = {
        "config_hash": config_payload.get("config_hash"),
        "selected_policy_hash": config_payload.get("selected_policy_hash"),
        "selected_policy_report_hash": config_payload.get("selected_policy_report_hash"),
        "window_mil_report_hash": config_payload.get("window_mil_report_hash"),
        "window_mil_model_file_sha256": config_payload.get("window_mil_model_file_sha256"),
        "window_mil_precision": config_payload.get("window_mil_precision"),
        "window_mil_input_mode": config_payload.get("window_mil_input_mode"),
        "window_mil_pooling_mode": config_payload.get("window_mil_pooling_mode"),
    }
    for key, expected in expected_values.items():
        if envelope.get(key) != expected:
            blocking_reasons.append("runtime_config_mismatch")
    precision_policy = config_payload.get("window_mil_precision_policy")
    if not isinstance(precision_policy, Mapping) or envelope.get("window_mil_precision_policy_hash") != precision_policy.get(
        "precision_policy_hash"
    ):
        blocking_reasons.append("runtime_config_mismatch")
    if _int_value(envelope.get("window_mil_pooling_top_k"), default=-1) != int(
        config_payload.get("window_mil_pooling_top_k", -2)
    ):
        blocking_reasons.append("runtime_config_mismatch")
    try:
        observed_threshold = float(envelope.get("window_mil_threshold"))
        expected_threshold = float(config_payload.get("window_mil_threshold"))
    except (TypeError, ValueError):
        blocking_reasons.append("runtime_config_mismatch")
    else:
        if abs(observed_threshold - expected_threshold) > 1e-12:
            blocking_reasons.append("runtime_config_mismatch")


def _validate_launcher_receipt(
    run_root: Path,
    *,
    plan: Mapping[str, Any],
    config: W1ProductionLaunchConfig,
    planned_commands: Sequence[Mapping[str, Any]],
    blocking_reasons: list[str],
) -> None:
    receipt_path = run_root / "launcher_receipt.json"
    if not receipt_path.exists():
        blocking_reasons.append("missing_launcher_receipt")
        return
    try:
        receipt = _load_json(receipt_path)
        validate_public_no_raw_payload(receipt)
    except ValueError:
        blocking_reasons.append("launcher_receipt_contract_mismatch")
        return
    if receipt.get("artifact_kind") != W1_PRODUCTION_LAUNCH_RECEIPT_KIND:
        blocking_reasons.append("launcher_receipt_contract_mismatch")
    if receipt.get("status") != "w1_production_launch_complete" or receipt.get("dry_run") is not False:
        blocking_reasons.append("launcher_receipt_status_mismatch")
    if receipt.get("code_commit") != config.expected_commit or receipt.get("plan_hash") != plan.get("plan_hash"):
        blocking_reasons.append("launcher_receipt_contract_mismatch")
    if _int_value(receipt.get("command_count"), default=-1) != len(planned_commands):
        blocking_reasons.append("launcher_receipt_contract_mismatch")
    planned_hashes = receipt.get("planned_command_hashes")
    if not isinstance(planned_hashes, Mapping):
        blocking_reasons.append("launcher_receipt_contract_mismatch")
    else:
        for command in planned_commands:
            label = str(command.get("label") or command.get("role") or "")
            if planned_hashes.get(label) != command.get("argv_hash"):
                blocking_reasons.append("launcher_receipt_contract_mismatch")
    command_log = run_root / "commands.jsonl"
    if not command_log.exists():
        blocking_reasons.append("missing_command_log")
    elif receipt.get("command_log_hash") != _sha256_file(command_log):
        blocking_reasons.append("command_log_hash_mismatch")


def _validate_assembled_report(
    report_path: Path,
    *,
    handoff: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    blocking_reasons: list[str],
) -> None:
    if not report_path.exists():
        blocking_reasons.append("missing_assembled_report")
        return
    try:
        report = _load_json(report_path)
        validate_public_no_raw_payload(report)
    except ValueError:
        blocking_reasons.append("assembled_report_contract_mismatch")
        return
    if report.get("artifact_kind") != ASSEMBLED_STAGE_REPORT_KIND:
        blocking_reasons.append("assembled_report_contract_mismatch")
    if report.get("status") != "assembled_ready_for_review":
        blocking_reasons.append("assembled_report_status_mismatch")
    if report.get("assembly_input_mode") != "sharded_w1_handoff":
        blocking_reasons.append("assembled_report_contract_mismatch")
    if report.get("sharded_w1_handoff_hash") != handoff.get("handoff_hash"):
        blocking_reasons.append("assembled_report_contract_mismatch")
    for key in ("config_hash", "selected_policy_hash", "selected_policy_report_hash"):
        if report.get(key) != config_payload.get(key):
            blocking_reasons.append("assembled_report_contract_mismatch")
    if report.get("w1_consumer_config_hash") != config_payload.get("config_hash"):
        blocking_reasons.append("assembled_report_contract_mismatch")
    if _int_value(report.get("survivor_count"), default=-1) != int(handoff.get("survivor_count", -2)):
        blocking_reasons.append("assembled_report_contract_mismatch")
    if _int_value(report.get("w1_scored_count"), default=-1) != int(handoff.get("survivor_count", -2)):
        blocking_reasons.append("assembled_report_contract_mismatch")
    if report.get("survivor_w1_set_equality_validated") is not True:
        blocking_reasons.append("assembled_report_contract_mismatch")
    _validate_w1_runtime_envelope_binding(
        report,
        config_payload=config_payload,
        blocking_reasons=blocking_reasons,
    )


def _validate_private_score_state_archive_manifest(
    command: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
    blocking_reasons: list[str],
) -> None:
    manifest_path = Path(str(command.get("summary_out") or "")).expanduser().resolve(strict=False)
    if not manifest_path.exists():
        blocking_reasons.append("missing_private_score_state_archive_manifest")
        return
    try:
        manifest = _load_json(manifest_path)
        validate_public_no_raw_payload(manifest)
    except ValueError:
        blocking_reasons.append("private_score_state_archive_manifest_invalid")
        return
    if manifest.get("artifact_kind") != W1_PRIVATE_SCORE_STATE_ARCHIVE_KIND:
        blocking_reasons.append("private_score_state_archive_manifest_invalid")
    if manifest.get("status") != "w1_private_score_state_archived":
        blocking_reasons.append("private_score_state_archive_status_mismatch")
    if manifest.get("archive_manifest_hash") != stable_json_hash(
        {key: value for key, value in manifest.items() if key != "archive_manifest_hash"}
    ):
        blocking_reasons.append("private_score_state_archive_manifest_hash_mismatch")
    if manifest.get("config_hash") != config_payload["config_hash"]:
        blocking_reasons.append("private_score_state_archive_config_mismatch")
    if int(manifest.get("file_count", 0) or 0) < 1 or manifest.get("private_archive_contains_sqlite_state") is not True:
        blocking_reasons.append("private_score_state_archive_incomplete")
    if manifest.get("private_archive_may_contain_private_evidence_text") is not True:
        blocking_reasons.append("private_score_state_archive_incomplete")
    relative_paths = {
        str(item.get("relative_path") or "")
        for item in list(manifest.get("files") or [])
        if isinstance(item, Mapping)
    }
    if command.get("accepted_token_ledger_archive_required") is True:
        if manifest.get("private_archive_contains_accepted_token_ledger_state") is not True:
            blocking_reasons.append("private_score_state_archive_incomplete")
        if "extra_state/w1_accepted_token_ledger.sqlite" not in relative_paths:
            blocking_reasons.append("private_score_state_archive_incomplete")
    argv = list(command.get("argv") or [])
    if "--archive-dir" not in argv:
        blocking_reasons.append("private_score_state_archive_manifest_invalid")
        return
    archive_dir = Path(str(argv[argv.index("--archive-dir") + 1])).expanduser().resolve(strict=False)
    archive_file = manifest.get("archive_file")
    archive_sha = manifest.get("archive_sha256")
    if not isinstance(archive_file, str) or not archive_file:
        blocking_reasons.append("private_score_state_archive_manifest_invalid")
        return
    archive_path = archive_dir / archive_file
    if not archive_path.exists():
        blocking_reasons.append("missing_private_score_state_archive_file")
        return
    if archive_sha != _sha256_file(archive_path):
        blocking_reasons.append("private_score_state_archive_hash_mismatch")


def _aggregate_summary_telemetry(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for report in reports:
        telemetry = report.get("w1_runtime_telemetry")
        if not isinstance(telemetry, Mapping):
            continue
        for key, value in telemetry.items():
            if isinstance(value, (int, float)):
                result[key] = result.get(key, 0) + value
            elif key == "scorer_device" and key not in result:
                result[key] = dict(value) if isinstance(value, Mapping) else value
    return result


def _modulo_assigned_shards(
    shards: Sequence[Mapping[str, Any]],
    *,
    modulus: int,
    remainder: int,
) -> list[Mapping[str, Any]]:
    return [
        shard
        for shard in shards
        if int(shard["shard_index"]) % int(modulus) == int(remainder)
    ]


def _rank_shard_assignments(
    shards: Sequence[Mapping[str, Any]],
    *,
    gpu_count: int,
    strategy: str,
) -> list[list[Mapping[str, Any]]]:
    if int(gpu_count) < 1:
        raise ValueError("gpu_count must be positive")
    ordered = sorted(list(shards), key=lambda item: int(item["shard_index"]))
    if int(gpu_count) > len(ordered):
        raise ValueError("gpu_count exceeds shard_count")
    if strategy == MODULO_RANK_ASSIGNMENT_STRATEGY:
        return [
            _modulo_assigned_shards(ordered, modulus=int(gpu_count), remainder=rank)
            for rank in range(int(gpu_count))
        ]
    if strategy != LPT_RANK_ASSIGNMENT_STRATEGY:
        raise ValueError("unknown rank assignment strategy")
    assignments: list[list[Mapping[str, Any]]] = [[] for _ in range(int(gpu_count))]
    assigned_survivors = [0 for _ in range(int(gpu_count))]
    for shard in sorted(ordered, key=lambda item: (-int(item.get("survivor_count", 0)), int(item["shard_index"]))):
        rank = min(
            range(int(gpu_count)),
            key=lambda item: (assigned_survivors[item], len(assignments[item]), item),
        )
        assignments[rank].append(shard)
        assigned_survivors[rank] += int(shard.get("survivor_count", 0))
    return [sorted(assignment, key=lambda item: int(item["shard_index"])) for assignment in assignments]


def _assigned_survivor_count(shards: Sequence[Mapping[str, Any]]) -> int:
    return sum(int(shard.get("survivor_count", 0)) for shard in shards)


def _rank_assignment_hash(
    *,
    strategy: str,
    handoff_hash: str,
    gpu_devices: Sequence[str],
    rank_assignments: Sequence[Sequence[Mapping[str, Any]]],
) -> str:
    return stable_json_hash(
        {
            "strategy": strategy,
            "handoff_hash": handoff_hash,
            "gpu_devices": [str(item) for item in gpu_devices],
            "rank_assignments": [
                {
                    "rank": rank,
                    "gpu_device": str(gpu_devices[rank]),
                    "assigned_shard_indexes": [int(shard["shard_index"]) for shard in assignment],
                    "assigned_survivor_count": _assigned_survivor_count(assignment),
                }
                for rank, assignment in enumerate(rank_assignments)
            ],
        }
    )


def _numeric_skew(values: Sequence[int | float]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {"min": 0, "max": 0, "ratio": None}
    minimum = min(numeric)
    maximum = max(numeric)
    if minimum <= 0:
        ratio = 1.0 if maximum <= 0 else None
    else:
        ratio = maximum / minimum
    return {
        "min": int(minimum) if minimum.is_integer() else minimum,
        "max": int(maximum) if maximum.is_integer() else maximum,
        "ratio": None if ratio is None else float(ratio),
    }


def _raise_for_forbidden_copyback_files(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and _is_forbidden_copyback_path(path):
            raise ValueError(f"forbidden copyback file in run_root: {path.name}")


def _validate_safe_copyback_artifact(path: Path) -> None:
    try:
        if path.suffix.lower() == ".json":
            validate_public_no_raw_payload(_load_json(path))
        elif path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row {line_number} must be an object")
                validate_public_no_raw_payload(payload, path=f"$.row[{line_number}]")
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"unsafe copyback artifact {path.name}: {exc}") from exc


def _copyback_relative_path(path: Path, *, run_root: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return str(resolved.relative_to(run_root.resolve(strict=False)))
    except ValueError:
        return path.name


def _require_safe_copyback_path(path: Path) -> None:
    if _is_forbidden_copyback_path(path):
        raise ValueError(f"forbidden copyback path: {path.name}")


def _is_forbidden_copyback_path(path: Path) -> bool:
    return path.name in FORBIDDEN_COPYBACK_NAMES or path.suffix.lower() in FORBIDDEN_COPYBACK_SUFFIXES


def _require_repo_root(code_root: Path) -> None:
    if not (code_root / "pyproject.toml").exists() or not (code_root / "uv.lock").exists():
        raise ValueError("code_root must be the public code repo")


def _require_not_under_public_code(path: Path, code_root: Path, label: str) -> None:
    if _is_relative_to(path, code_root):
        raise ValueError(f"{label} must not be inside the public code root")


def _validate_required_persistent_output_root(
    root: str | Path | None,
    *,
    mutable_output_paths: Mapping[str, Path],
) -> Path | None:
    if root is None:
        return None
    resolved_root = Path(root).expanduser().resolve(strict=False)
    for label, path in mutable_output_paths.items():
        if path is None:
            continue
        resolved = Path(path).expanduser().resolve(strict=False)
        if not _is_relative_to(resolved, resolved_root):
            raise ValueError(f"{label} must be under required_persistent_output_root")
    return resolved_root


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _path_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return stable_json_hash(str(Path(path).expanduser().resolve(strict=False)))


def _sha256_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        Path(tmp_name).replace(output)
    except Exception:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _safe_label(label: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in label)[:80]


def _add_common_launch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--supervisor-run-dir", required=True)
    parser.add_argument("--handoff", dest="handoff_path", required=True)
    parser.add_argument("--config", dest="config_path", required=True)
    parser.add_argument("--selected-policy-report", required=True)
    parser.add_argument("--window-mil-report", required=True)
    parser.add_argument("--window-mil-model", required=True)
    parser.add_argument("--output-report", dest="output_report_path", required=True)
    parser.add_argument("--gpu-device", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-windows-per-device-batch", type=int)
    parser.add_argument("--synchronize-cuda-telemetry", action="store_true")
    parser.add_argument(
        "--w1-attn-implementation",
        choices=sorted(SUPPORTED_W1_ATTN_IMPLEMENTATIONS),
        default=W1_ATTN_IMPLEMENTATION_SDPA,
    )
    parser.add_argument("--w1-prepare-pipeline", choices=("serial", "bounded-tokenize-overlap"), default="serial")
    parser.add_argument("--w1-max-estimated-windows-per-prepare-batch", type=int)
    parser.add_argument("--w1-prepare-queue-depth", type=int, default=1)
    parser.add_argument("--w1-pin-memory", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--w1-tokenizers-parallelism", choices=("inherit", "false", "true"), default="inherit")
    parser.add_argument("--w1-persist-commit-row-threshold", type=int)
    parser.add_argument(
        "--no-enforce-accepted-token-target",
        dest="enforce_accepted_token_target",
        action="store_false",
        default=True,
    )
    parser.add_argument("--private-archive-dir")
    parser.add_argument("--w1-backend-runtime-compatibility-manifest")
    parser.add_argument("--w1-score-comparator-report")
    parser.add_argument("--bootstrap-status")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rank-assignment-strategy",
        choices=[MODULO_RANK_ASSIGNMENT_STRATEGY, LPT_RANK_ASSIGNMENT_STRATEGY],
        default=MODULO_RANK_ASSIGNMENT_STRATEGY,
    )
    parser.add_argument("--required-persistent-output-root")


def _config_from_args(args: argparse.Namespace) -> W1ProductionLaunchConfig:
    return W1ProductionLaunchConfig(
        code_root=Path(args.code_root),
        expected_commit=str(args.expected_commit),
        run_root=Path(args.run_root),
        supervisor_run_dir=Path(args.supervisor_run_dir),
        handoff_path=Path(args.handoff_path),
        config_path=Path(args.config_path),
        selected_policy_report=Path(args.selected_policy_report),
        window_mil_report=Path(args.window_mil_report),
        window_mil_model=Path(args.window_mil_model),
        output_report_path=Path(args.output_report_path),
        gpu_devices=tuple(args.gpu_device or ["0"]),
        batch_size=int(args.batch_size),
        max_windows_per_device_batch=args.max_windows_per_device_batch,
        synchronize_cuda_telemetry=bool(args.synchronize_cuda_telemetry),
        w1_attn_implementation=str(args.w1_attn_implementation),
        w1_prepare_pipeline=str(args.w1_prepare_pipeline),
        max_estimated_windows_per_prepare_batch=args.w1_max_estimated_windows_per_prepare_batch,
        w1_prepare_queue_depth=int(args.w1_prepare_queue_depth),
        w1_pin_memory=str(args.w1_pin_memory),
        w1_tokenizers_parallelism=str(args.w1_tokenizers_parallelism),
        w1_persist_commit_row_threshold=args.w1_persist_commit_row_threshold,
        enforce_accepted_token_target=bool(args.enforce_accepted_token_target),
        private_archive_dir=Path(args.private_archive_dir) if args.private_archive_dir else None,
        w1_backend_runtime_compatibility_manifest=(
            Path(args.w1_backend_runtime_compatibility_manifest)
            if args.w1_backend_runtime_compatibility_manifest
            else None
        ),
        w1_score_comparator_report=(
            Path(args.w1_score_comparator_report) if args.w1_score_comparator_report else None
        ),
        bootstrap_status=Path(args.bootstrap_status) if args.bootstrap_status else None,
        python_executable=str(args.python_executable),
        run_id=args.run_id,
        dry_run=bool(args.dry_run),
        rank_assignment_strategy=str(args.rank_assignment_strategy),
        required_persistent_output_root=(
            Path(args.required_persistent_output_root) if args.required_persistent_output_root else None
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TWC W1 production GPU launch planner")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    _add_common_launch_args(plan)
    launch = sub.add_parser("launch")
    _add_common_launch_args(launch)
    stop = sub.add_parser("stop")
    stop.add_argument("--run-root", required=True)
    stop.add_argument("--supervisor-run-dir", required=True)
    copyback = sub.add_parser("copyback-manifest")
    _add_common_launch_args(copyback)
    summarize = sub.add_parser("summarize")
    _add_common_launch_args(summarize)
    summarize.add_argument("--hourly-gpu-cost-usd", type=float, required=True)
    summarize.add_argument("--min-scored-docs", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "plan":
        payload = plan_w1_production_launch(_config_from_args(args))
    elif args.command == "launch":
        payload = launch_w1_production(_config_from_args(args))
    elif args.command == "stop":
        payload = stop_w1_production(run_root=args.run_root, supervisor_run_dir=args.supervisor_run_dir)
    elif args.command == "copyback-manifest":
        payload = collect_w1_production_copyback(_config_from_args(args))
    elif args.command == "summarize":
        payload = summarize_w1_production_run(
            _config_from_args(args),
            hourly_gpu_cost_usd=float(args.hourly_gpu_cost_usd),
            min_scored_docs=int(args.min_scored_docs),
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps({"status": payload["status"], "hash": payload.get("plan_hash") or payload.get("receipt_hash") or payload.get("summary_hash") or payload.get("manifest_hash") or payload.get("report_hash")}, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
