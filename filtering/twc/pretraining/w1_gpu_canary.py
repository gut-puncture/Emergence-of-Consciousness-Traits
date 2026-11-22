from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from twc.classifiers.window_mil import (
    W1_ATTN_IMPLEMENTATION_SDPA,
    SUPPORTED_W1_ATTN_IMPLEMENTATIONS,
    normalize_w1_attn_implementation,
)
from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.data.decision_schema import stable_json_hash
from twc.pretraining.chunk_filtering import (
    build_chunk_filtering_config_payload,
    validate_locked_chunk_filtering_config_payload,
    validate_private_filtering_run_dir,
    validate_window_mil_inference_artifacts,
    _load_and_validate_selected_policy_report,
)


BOOTSTRAP_STATUS_KIND = "twc_w1_gpu_bootstrap_status_v1"
CANARY_LAUNCHER_RECEIPT_KIND = "twc_w1_runtime_cap_canary_launcher_receipt_v1"
CANARY_LAUNCHER_OWNED_ROOT_KIND = "twc_w1_gpu_canary_launcher_owned_root_v1"
CANARY_STOP_REPORT_KIND = "twc_w1_runtime_cap_canary_stop_report_v1"
CANARY_RESET_REPORT_KIND = "twc_w1_runtime_cap_canary_reset_report_v1"
CLAIMS_SCOPE = "throughput_diagnostic_only"

TORCH_PROFILES: dict[str, list[str]] = {
    "locked": [],
    "cu124-torch260": [
        "uv",
        "pip",
        "install",
        "torch==2.6.0+cu124",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
    ],
}

MODEL_LOAD_SMOKE_CODE = (
    "from types import SimpleNamespace; "
    "import sys, yaml; "
    "from twc.pretraining.chunk_filtering import "
    "build_chunk_filtering_config_payload, validate_locked_chunk_filtering_config_payload, _load_w1_scorer; "
    "cfg=build_chunk_filtering_config_payload(yaml.safe_load(open(sys.argv[1], encoding='utf-8').read())); "
    "validate_locked_chunk_filtering_config_payload(cfg); "
    "args=SimpleNamespace(window_mil_report=sys.argv[2], window_mil_model=sys.argv[3], "
    "device=sys.argv[4], require_cuda=True, max_windows_per_device_batch=None, "
    "synchronize_cuda_telemetry=False, w1_attn_implementation=sys.argv[5]); "
    "_load_w1_scorer(args, cfg); "
    "print('w1_model_load_smoke_ready')"
)

W1_SCORE_RESET_FILES = {
    "w1_score_receipt.json",
    "w1_scores.jsonl.ready",
    "w1_scores.jsonl.part",
}
W1_SCORE_STATE_DB_NAMES = ("filtering_stage_state.sqlite", "filtering_stage.sqlite")
STALE_STOP_MARKER_NAME = "STOP_REQUESTED.stale_source_marker"

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


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0


CommandRunner = Callable[..., CommandResult]
TorchProbe = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True)
class BootstrapConfig:
    code_root: Path
    expected_commit: str
    disk_root: Path
    run_root: Path
    cache_root: Path
    config_path: Path
    selected_policy_report: Path
    window_mil_report: Path
    window_mil_model: Path
    torch_profile: str = "locked"
    device: str = "cuda:0"
    require_cuda: bool = True
    min_free_bytes: int = 1
    write_env: Path | None = None
    skip_artifact_validation: bool = False
    skip_model_load: bool = False
    allow_dirty: bool = False
    w1_attn_implementation: str = W1_ATTN_IMPLEMENTATION_SDPA


@dataclass(frozen=True)
class CanaryLaunchConfig:
    source_supervisor_root: Path
    diagnostic_root: Path
    plan: Path
    manifest: Path
    config_path: Path
    selected_policy_report: Path
    window_mil_report: Path
    window_mil_model: Path
    shard_indexes: list[int]
    override_cap: int
    code_commit: str
    code_root: Path | None = None
    batch_size: int = 64
    python_executable: str = sys.executable
    synchronize_cuda_telemetry: bool = True
    dry_run: bool = False
    reset_copied_w1_state: bool = True
    run_id: str | None = None


def default_command_runner(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    label: str = "command",
) -> CommandResult:
    start = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = round(time.monotonic() - start, 6)
    if stdout_path is not None:
        _write_text_private(stdout_path, completed.stdout)
    if stderr_path is not None:
        _write_text_private(stderr_path, completed.stderr)
    return CommandResult(
        argv=list(argv),
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_seconds=elapsed,
    )


def default_torch_probe() -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on remote bootstrap state.
        raise RuntimeError("torch is not importable after dependency setup") from exc
    cuda_available = bool(torch.cuda.is_available())
    result: dict[str, Any] = {
        "torch_version": str(getattr(torch, "__version__", "")),
        "cuda_available": cuda_available,
        "torch_cuda_runtime": str(getattr(torch.version, "cuda", "") or ""),
        "device_type": "cuda" if cuda_available else "cpu",
    }
    if cuda_available:
        index = int(torch.cuda.current_device())
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        result.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": str(torch.cuda.get_device_name(index)),
                "cuda_device_count": int(torch.cuda.device_count()),
                "bf16_supported": bool(is_bf16_supported()) if callable(is_bf16_supported) else False,
            }
        )
    validate_public_no_raw_payload(result)
    return result


def bootstrap_w1_gpu_env(
    config: BootstrapConfig,
    *,
    command_runner: CommandRunner = default_command_runner,
    torch_probe: TorchProbe = default_torch_probe,
) -> dict[str, Any]:
    code_root = config.code_root.expanduser().resolve(strict=False)
    disk_root = config.disk_root.expanduser().resolve(strict=False)
    run_root = config.run_root.expanduser().resolve(strict=False)
    cache_root = config.cache_root.expanduser().resolve(strict=False)
    run_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "process_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_log = run_root / "bootstrap_commands.jsonl"

    _require_repo_root(code_root)
    _require_not_under_public_code(run_root, code_root, "run_root")
    _require_not_under_public_code(cache_root, code_root, "cache_root")
    _require_disk_ready(disk_root, min_free_bytes=config.min_free_bytes)
    _require_disk_ready(run_root, min_free_bytes=1)
    w1_attn_implementation = normalize_w1_attn_implementation(config.w1_attn_implementation)

    commit = _run_logged_command(
        ["git", "rev-parse", "HEAD"],
        command_runner=command_runner,
        command_log=command_log,
        cwd=code_root,
        logs_dir=logs_dir,
        label="git_rev_parse",
    ).stdout.strip()
    if commit != config.expected_commit:
        raise ValueError("code commit mismatch")
    status = _run_logged_command(
        ["git", "status", "--short"],
        command_runner=command_runner,
        command_log=command_log,
        cwd=code_root,
        logs_dir=logs_dir,
        label="git_status",
    ).stdout.strip()
    if status and not config.allow_dirty:
        raise ValueError("public code repo is dirty")
    if sys.version_info[:2] != (3, 11):
        raise ValueError("W1 GPU setup must run under Python 3.11")
    _run_logged_command(
        ["uv", "lock", "--check"],
        command_runner=command_runner,
        command_log=command_log,
        cwd=code_root,
        logs_dir=logs_dir,
        label="uv_lock_check",
    )
    _run_logged_command(
        ["uv", "sync", "--locked", "--all-extras", "--group", "dev"],
        command_runner=command_runner,
        command_log=command_log,
        cwd=code_root,
        logs_dir=logs_dir,
        label="uv_sync",
    )
    torch_install = TORCH_PROFILES.get(config.torch_profile)
    if torch_install is None:
        raise ValueError("unsupported torch profile")
    if torch_install:
        _run_logged_command(
            list(torch_install),
            command_runner=command_runner,
            command_log=command_log,
            cwd=code_root,
            logs_dir=logs_dir,
            label="torch_profile_overlay",
        )
    _run_logged_command(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        command_runner=command_runner,
        command_log=command_log,
        cwd=code_root,
        logs_dir=logs_dir,
        label="nvidia_smi",
    )
    cuda = dict(torch_probe())
    validate_public_no_raw_payload(cuda)
    if config.require_cuda and not bool(cuda.get("cuda_available")):
        raise RuntimeError("CUDA is required for W1 GPU bootstrap")
    if config.require_cuda and str(cuda.get("device_type") or "") != "cuda":
        raise RuntimeError("CUDA device did not resolve for W1 GPU bootstrap")

    artifact_report: dict[str, Any] = {"status": "artifact_validation_skipped"}
    if not config.skip_artifact_validation:
        artifact_report = _validate_w1_artifacts_for_bootstrap(config)
    if not config.skip_model_load:
        _run_logged_command(
            [
                sys.executable,
                "-c",
                MODEL_LOAD_SMOKE_CODE,
                str(config.config_path),
                str(config.window_mil_report),
                str(config.window_mil_model),
                str(config.device),
                w1_attn_implementation,
            ],
            command_runner=command_runner,
            command_log=command_log,
            cwd=code_root,
            logs_dir=logs_dir,
            label="model_load_smoke",
        )

    env_path = config.write_env.expanduser().resolve(strict=False) if config.write_env is not None else run_root / "w1_env.sh"
    _write_env_file(
        env_path,
        {
            "TWC_CODE_ROOT": str(code_root),
            "TWC_RUN_ROOT": str(run_root),
            "TWC_CONFIG": str(config.config_path),
            "TWC_SELECTED_POLICY_REPORT": str(config.selected_policy_report),
            "TWC_W1_REPORT": str(config.window_mil_report),
            "TWC_W1_MODEL": str(config.window_mil_model),
            "UV_CACHE_DIR": str(cache_root / "uv"),
            "HF_HOME": str(cache_root / "hf"),
            "HF_HUB_CACHE": str(cache_root / "hf" / "hub"),
            "TRANSFORMERS_CACHE": str(cache_root / "hf" / "transformers"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    payload: dict[str, Any] = {
        "artifact_kind": BOOTSTRAP_STATUS_KIND,
        "schema_version": 1,
        "status": "w1_gpu_bootstrap_ready",
        "code_commit": commit,
        "expected_commit": config.expected_commit,
        "torch_profile": config.torch_profile,
        "device": config.device,
        "w1_attn_implementation": w1_attn_implementation,
        "require_cuda": bool(config.require_cuda),
        "cuda": cuda,
        "artifact_validation": artifact_report,
        "artifact_validation_skipped": bool(config.skip_artifact_validation),
        "model_load_smoke_skipped": bool(config.skip_model_load),
        "code_root_hash": _path_hash(code_root),
        "disk_root_hash": _path_hash(disk_root),
        "run_root_hash": _path_hash(run_root),
        "cache_root_hash": _path_hash(cache_root),
        "command_log_hash": _sha256_file(command_log),
        "env_file_hash": _sha256_file(env_path),
        **FALSE_FLAGS,
    }
    payload["copyback_safe_summary"] = _copyback_summary(payload)
    payload["status_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "status_hash"})
    validate_public_no_raw_payload(payload["copyback_safe_summary"])
    _write_json_atomic(run_root / "bootstrap_status.json", payload)
    _write_sha256s(run_root, [run_root / "bootstrap_status.json", command_log, env_path])
    return payload


def launch_w1_runtime_cap_canary(
    config: CanaryLaunchConfig,
    *,
    command_runner: CommandRunner = default_command_runner,
) -> dict[str, Any]:
    if not config.shard_indexes:
        raise ValueError("at least one shard index is required")
    if int(config.override_cap) <= 64:
        raise ValueError("override_cap must exceed the artifact default cap 64")
    if int(config.batch_size) < 1:
        raise ValueError("batch_size must be positive")
    root = config.diagnostic_root.expanduser().resolve(strict=False)
    code_root = (
        config.code_root.expanduser().resolve(strict=False)
        if config.code_root is not None
        else Path.cwd().expanduser().resolve(strict=False)
    )
    source_root = config.source_supervisor_root.expanduser().resolve(strict=False)
    _require_safe_diagnostic_root(root, code_root=code_root, source_supervisor_root=source_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = config.run_id or stable_json_hash(
        {"code_commit": config.code_commit, "shards": list(config.shard_indexes), "override_cap": int(config.override_cap)}
    )[:16]
    _write_owner_marker(root, run_id=run_id)
    logs_dir = root / "process_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_log = root / "commands.jsonl"
    _require_repo_root(code_root)
    commit = _run_logged_command(
        ["git", "rev-parse", "HEAD"],
        command_runner=command_runner,
        command_log=command_log,
        cwd=code_root,
        logs_dir=logs_dir,
        label="git_rev_parse",
    ).stdout.strip()
    if commit != config.code_commit:
        raise ValueError("launcher code commit mismatch")
    dirty = _run_logged_command(
        ["git", "status", "--short"],
        command_runner=command_runner,
        command_log=command_log,
        cwd=code_root,
        logs_dir=logs_dir,
        label="git_status",
    ).stdout.strip()
    if dirty:
        raise ValueError("launcher public code repo is dirty")
    manifest = _load_json(config.manifest)
    shard_map = _manifest_shard_map(manifest)
    selected = [_selected_shard(shard_map, index) for index in config.shard_indexes]

    baseline_supervisor = root / "baseline_cap64" / "supervisor"
    override_supervisor = root / f"override_cap{int(config.override_cap)}" / "supervisor"
    for supervisor in (baseline_supervisor, override_supervisor):
        supervisor.mkdir(parents=True, exist_ok=True)
        for shard in selected:
            if config.dry_run:
                _create_dry_run_shard_placeholder(supervisor, shard)
            else:
                _copy_selected_shard(
                    source_root,
                    supervisor,
                    shard,
                    reset_w1_state=config.reset_copied_w1_state,
                )

    baseline_handoff = root / "baseline_cap64" / "partial_handoff.json"
    baseline_summary = root / "baseline_cap64" / "partial_summary.json"
    override_handoff = root / f"override_cap{int(config.override_cap)}" / "partial_handoff.json"
    override_summary = root / f"override_cap{int(config.override_cap)}" / "partial_summary.json"
    comparison_dir = root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        _build_partial_handoff_command(config, baseline_supervisor, baseline_handoff),
        _score_partial_diagnostic_command(config, baseline_supervisor, baseline_handoff, baseline_summary, None),
        _build_partial_handoff_command(config, override_supervisor, override_handoff),
        _score_partial_diagnostic_command(
            config,
            override_supervisor,
            override_handoff,
            override_summary,
            int(config.override_cap),
        ),
    ]
    comparison_reports: list[dict[str, Any]] = []
    for shard in selected:
        name = _shard_dir_name(shard)
        report = comparison_dir / f"{name}_runtime_cap_comparison.json"
        comparison_reports.append(
            {
                "shard_index": int(shard["shard_index"]),
                "baseline_shard_dir_hash": _path_hash(baseline_supervisor / name),
                "comparison_shard_dir_hash": _path_hash(override_supervisor / name),
                "report_hash": _path_hash(report),
            }
        )
        commands.append(
            _compare_runtime_cap_command(
                config,
                baseline_supervisor / name,
                override_supervisor / name,
                report,
            )
        )

    start = time.monotonic()
    for ordinal, command in enumerate(commands):
        label = f"{ordinal:02d}_{_command_label(command)}"
        if config.dry_run:
            _log_planned_command(command_log, argv=command, cwd=code_root, label=label)
        else:
            _run_logged_command(
                command,
                command_runner=command_runner,
                command_log=command_log,
                cwd=code_root,
                logs_dir=logs_dir,
                label=label,
            )
    elapsed = round(time.monotonic() - start, 6)
    comparison_statuses: list[str] = []
    blocking_reasons: list[str] = []
    if not config.dry_run:
        for item in comparison_dir.glob("*_runtime_cap_comparison.json"):
            report = _load_json(item)
            comparison_statuses.append(str(report.get("status") or ""))
            blocking_reasons.extend(str(reason) for reason in list(report.get("blocking_reasons") or []))
    ready = bool(comparison_statuses) and all(status == "w1_runtime_cap_comparison_ready" for status in comparison_statuses)
    status = "w1_runtime_cap_canary_dry_run" if config.dry_run else (
        "w1_runtime_cap_canary_ready" if ready else "w1_runtime_cap_canary_blocked"
    )
    payload: dict[str, Any] = {
        "artifact_kind": CANARY_LAUNCHER_RECEIPT_KIND,
        "schema_version": 1,
        "status": status,
        "claims_scope": CLAIMS_SCOPE,
        "run_id": run_id,
        "code_commit": commit,
        "code_root_hash": _path_hash(code_root),
        "selected_shard_indexes": [int(index) for index in config.shard_indexes],
        "baseline_cap": 64,
        "override_cap": int(config.override_cap),
        "batch_size": int(config.batch_size),
        "synchronize_cuda_telemetry": bool(config.synchronize_cuda_telemetry),
        "dry_run": bool(config.dry_run),
        "source_supervisor_root_hash": _path_hash(config.source_supervisor_root),
        "diagnostic_root_hash": _path_hash(root),
        "baseline_supervisor_hash": _path_hash(baseline_supervisor),
        "override_supervisor_hash": _path_hash(override_supervisor),
        "command_log_hash": _sha256_file(command_log),
        "command_count": len(commands),
        "comparison_reports": comparison_reports,
        "comparison_statuses": comparison_statuses,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "elapsed_seconds": elapsed,
        **FALSE_FLAGS,
    }
    payload["copyback_safe_summary"] = _copyback_summary(payload)
    payload["launcher_receipt_hash"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "launcher_receipt_hash"}
    )
    validate_public_no_raw_payload(payload["copyback_safe_summary"])
    _write_json_atomic(root / "launcher_receipt.json", payload)
    return payload


def stop_w1_runtime_cap_canary(canary_root: str | Path) -> dict[str, Any]:
    root = Path(canary_root).expanduser().resolve(strict=False)
    marker = _load_owner_marker(root)
    supervisors = [path for path in root.glob("*_cap*/supervisor") if path.is_dir()]
    supervisors.extend(path for path in root.glob("baseline_cap*/supervisor") if path.is_dir())
    unique_supervisors = sorted(set(supervisors))
    for supervisor in unique_supervisors:
        _write_text_private(supervisor / "STOP_REQUESTED", "stop requested by W1 runtime cap canary launcher\n")
    payload: dict[str, Any] = {
        "artifact_kind": CANARY_STOP_REPORT_KIND,
        "schema_version": 1,
        "status": "stop_requested",
        "claims_scope": CLAIMS_SCOPE,
        "run_id": marker["run_id"],
        "canary_root_hash": _path_hash(root),
        "stop_file_count": len(unique_supervisors),
        **FALSE_FLAGS,
    }
    payload["copyback_safe_summary"] = _copyback_summary(payload)
    payload["report_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "report_hash"})
    validate_public_no_raw_payload(payload["copyback_safe_summary"])
    _write_json_atomic(root / "stop_report.json", payload)
    return payload


def reset_launcher_owned_w1_state(canary_root: str | Path, *, run_id: str) -> dict[str, Any]:
    root = Path(canary_root).expanduser().resolve(strict=False)
    marker = _load_owner_marker(root)
    if str(marker.get("run_id") or "") != str(run_id):
        raise ValueError("launcher-owned root run_id mismatch")
    removed = _remove_w1_score_files(root)
    cleared_tables = _clear_w1_score_tables(root)
    renamed_stop_markers = _quarantine_stop_markers(root)
    payload: dict[str, Any] = {
        "artifact_kind": CANARY_RESET_REPORT_KIND,
        "schema_version": 1,
        "status": "launcher_owned_w1_state_reset",
        "claims_scope": CLAIMS_SCOPE,
        "run_id": str(run_id),
        "canary_root_hash": _path_hash(root),
        "removed_file_count": removed,
        "cleared_w1_score_table_count": cleared_tables,
        "renamed_stop_marker_count": renamed_stop_markers,
        **FALSE_FLAGS,
    }
    payload["copyback_safe_summary"] = _copyback_summary(payload)
    payload["report_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "report_hash"})
    validate_public_no_raw_payload(payload["copyback_safe_summary"])
    _write_json_atomic(root / "reset_report.json", payload)
    return payload


def _require_safe_diagnostic_root(root: Path, *, code_root: Path, source_supervisor_root: Path) -> None:
    if root.exists() and not root.is_dir():
        raise ValueError("diagnostic_root must be a directory")
    if _is_relative_to(root, code_root):
        raise ValueError("diagnostic_root must not be inside the public code root")
    if _paths_overlap(root, source_supervisor_root):
        raise ValueError("diagnostic_root must not overlap source_supervisor_root")
    existing_marker = root / ".twc_w1_canary_launcher_owned.json"
    if root.exists() and any(root.iterdir()) and not existing_marker.exists():
        raise ValueError("diagnostic_root must be empty or launcher-owned")


def _paths_overlap(left: Path, right: Path) -> bool:
    return _is_relative_to(left, right) or _is_relative_to(right, left)


def _validate_w1_artifacts_for_bootstrap(config: BootstrapConfig) -> dict[str, Any]:
    config_payload = build_chunk_filtering_config_payload(yaml.safe_load(config.config_path.read_text(encoding="utf-8")))
    validate_locked_chunk_filtering_config_payload(config_payload)
    selected = _load_and_validate_selected_policy_report(
        config.selected_policy_report,
        config_payload,
        required=True,
    )
    w1_report = _load_json(config.window_mil_report)
    validate_window_mil_inference_artifacts(w1_report, config.window_mil_model)
    if w1_report.get("report_hash") != config_payload["window_mil_report_hash"]:
        raise ValueError("W1 report hash mismatch")
    if w1_report.get("model_file_sha256") != config_payload["window_mil_model_file_sha256"]:
        raise ValueError("W1 model hash mismatch")
    payload = {
        "status": "w1_artifacts_validated",
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": selected["selected_policy_hash"],
        "selected_policy_report_hash": selected["report_hash"],
        "w1_report_hash": w1_report["report_hash"],
        "w1_model_file_sha256": w1_report["model_file_sha256"],
        "w1_threshold": float(config_payload["window_mil_threshold"]),
        "w1_input_mode": str(config_payload["window_mil_input_mode"]),
        "w1_pooling_mode": str(config_payload["window_mil_pooling_mode"]),
        "w1_pooling_top_k": int(config_payload["window_mil_pooling_top_k"]),
        "window_mil_precision": str(config_payload["window_mil_precision"]),
        "window_mil_precision_policy_hash": str(
            config_payload["window_mil_precision_policy"]["precision_policy_hash"]
        ),
    }
    validate_public_no_raw_payload(payload)
    return payload


def _build_partial_handoff_command(config: CanaryLaunchConfig, supervisor: Path, handoff: Path) -> list[str]:
    command = [
        config.python_executable,
        "-m",
        "twc.pretraining.chunk_filtering",
        "build-partial-sharded-w1-diagnostic-handoff",
        "--config",
        str(config.config_path),
        "--selected-policy-report",
        str(config.selected_policy_report),
        "--plan",
        str(config.plan),
        "--manifest",
        str(config.manifest),
        "--supervisor-run-dir",
        str(supervisor),
        "--handoff-out",
        str(handoff),
    ]
    for index in config.shard_indexes:
        command.extend(["--shard-index", str(int(index))])
    return command


def _score_partial_diagnostic_command(
    config: CanaryLaunchConfig,
    supervisor: Path,
    handoff: Path,
    summary: Path,
    max_windows_per_device_batch: int | None,
) -> list[str]:
    command = [
        config.python_executable,
        "-m",
        "twc.pretraining.chunk_filtering",
        "w1-score-partial-sharded-diagnostic",
        "--config",
        str(config.config_path),
        "--selected-policy-report",
        str(config.selected_policy_report),
        "--handoff",
        str(handoff),
        "--supervisor-run-dir",
        str(supervisor),
        "--window-mil-report",
        str(config.window_mil_report),
        "--window-mil-model",
        str(config.window_mil_model),
        "--summary-out",
        str(summary),
        "--batch-size",
        str(int(config.batch_size)),
    ]
    if max_windows_per_device_batch is not None:
        command.extend(["--max-windows-per-device-batch", str(int(max_windows_per_device_batch))])
    if config.synchronize_cuda_telemetry:
        command.append("--synchronize-cuda-telemetry")
    return command


def _compare_runtime_cap_command(
    config: CanaryLaunchConfig,
    baseline_run_dir: Path,
    comparison_run_dir: Path,
    report: Path,
) -> list[str]:
    return [
        config.python_executable,
        "-m",
        "twc.pretraining.chunk_filtering",
        "compare-w1-runtime-cap-runs",
        "--config",
        str(config.config_path),
        "--selected-policy-report",
        str(config.selected_policy_report),
        "--baseline-run-dir",
        str(baseline_run_dir),
        "--comparison-run-dir",
        str(comparison_run_dir),
        "--report-out",
        str(report),
    ]


def _copy_selected_shard(
    source_supervisor: Path,
    destination_supervisor: Path,
    shard: Mapping[str, Any],
    *,
    reset_w1_state: bool,
) -> None:
    name = _shard_dir_name(shard)
    source = source_supervisor / name
    destination = destination_supervisor / name
    if not source.exists():
        raise FileNotFoundError(f"selected shard directory is missing: {name}")
    if destination.exists():
        raise ValueError("diagnostic shard destination already exists")
    validate_private_filtering_run_dir(source)
    shutil.copytree(source, destination)
    validate_private_filtering_run_dir(destination)
    if reset_w1_state:
        _remove_w1_score_files(destination)
        _clear_w1_score_tables(destination)
        _quarantine_stop_markers(destination)


def _remove_w1_score_files(root: Path) -> int:
    removed = 0
    for path in root.rglob("*"):
        if path.is_file() and path.name in W1_SCORE_RESET_FILES:
            path.unlink()
            removed += 1
    return removed


def _clear_w1_score_tables(root: Path) -> int:
    cleared_tables = 0
    seen: set[Path] = set()
    for db_name in W1_SCORE_STATE_DB_NAMES:
        for db_path in root.rglob(db_name):
            resolved = db_path.resolve(strict=False)
            if resolved in seen:
                continue
            seen.add(resolved)
            with sqlite3.connect(db_path) as conn:
                exists = conn.execute(
                    "select 1 from sqlite_master where type = 'table' and name = 'w1_scores'"
                ).fetchone()
                if exists is None:
                    continue
                conn.execute("delete from w1_scores")
                conn.commit()
                cleared_tables += 1
    return cleared_tables


def _quarantine_stop_markers(root: Path) -> int:
    renamed = 0
    for stop_path in root.rglob("STOP_REQUESTED"):
        if not stop_path.is_file():
            continue
        target = _unique_sibling(stop_path.with_name(STALE_STOP_MARKER_NAME))
        stop_path.replace(target)
        renamed += 1
    return renamed


def _unique_sibling(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}.{index}")
        if not candidate.exists():
            return candidate
    raise ValueError("could not choose unique stale stop marker path")


def _create_dry_run_shard_placeholder(destination_supervisor: Path, shard: Mapping[str, Any]) -> None:
    destination = destination_supervisor / _shard_dir_name(shard)
    destination.mkdir(parents=True, exist_ok=True)


def _run_logged_command(
    argv: list[str],
    *,
    command_runner: CommandRunner,
    command_log: Path,
    cwd: Path | None,
    logs_dir: Path,
    label: str,
) -> CommandResult:
    safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)[:80]
    stdout_path = logs_dir / f"{safe_label}.out.log"
    stderr_path = logs_dir / f"{safe_label}.err.log"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result = command_runner(
        list(argv),
        cwd=cwd,
        env=None,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        label=label,
    )
    row = {
        "label": label,
        "argv": list(argv),
        "argv_hash": stable_json_hash(list(argv)),
        "cwd": str(cwd) if cwd is not None else None,
        "cwd_hash": _path_hash(cwd) if cwd is not None else None,
        "started_utc": started,
        "returncode": int(result.returncode),
        "elapsed_seconds": float(result.elapsed_seconds),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_log_hash": _sha256_file(stdout_path) if stdout_path.exists() else None,
        "stderr_log_hash": _sha256_file(stderr_path) if stderr_path.exists() else None,
    }
    _append_jsonl(command_log, row)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}: {' '.join(argv)}")
    return result


def _log_planned_command(command_log: Path, *, argv: list[str], cwd: Path | None, label: str) -> None:
    row = {
        "label": label,
        "execution_mode": "planned_only",
        "argv": list(argv),
        "argv_hash": stable_json_hash(list(argv)),
        "cwd": str(cwd) if cwd is not None else None,
        "cwd_hash": _path_hash(cwd) if cwd is not None else None,
        "returncode": None,
        "elapsed_seconds": 0.0,
    }
    _append_jsonl(command_log, row)


def _command_label(command: Sequence[str]) -> str:
    for token in command:
        if token in {
            "build-partial-sharded-w1-diagnostic-handoff",
            "w1-score-partial-sharded-diagnostic",
            "compare-w1-runtime-cap-runs",
        }:
            return token
    return Path(str(command[0])).name


def _copyback_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        "status": str(payload.get("status") or ""),
        "artifact_kind": str(payload.get("artifact_kind") or ""),
        "claims_scope": str(payload.get("claims_scope") or ""),
        "code_commit": payload.get("code_commit"),
        "selected_shard_indexes": payload.get("selected_shard_indexes"),
        "baseline_cap": payload.get("baseline_cap"),
        "override_cap": payload.get("override_cap"),
        "batch_size": payload.get("batch_size"),
        "synchronize_cuda_telemetry": payload.get("synchronize_cuda_telemetry"),
        "blocking_reasons": payload.get("blocking_reasons"),
        "cuda": payload.get("cuda"),
        "torch_profile": payload.get("torch_profile"),
        **FALSE_FLAGS,
    }
    compact = {key: value for key, value in summary.items() if value is not None}
    validate_public_no_raw_payload(compact)
    return compact


def _manifest_shard_map(manifest: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("manifest missing shards")
    result: dict[int, Mapping[str, Any]] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("manifest shard must be an object")
        index = int(shard.get("shard_index"))
        result[index] = shard
    return result


def _selected_shard(shard_map: Mapping[int, Mapping[str, Any]], index: int) -> Mapping[str, Any]:
    try:
        return shard_map[int(index)]
    except KeyError as exc:
        raise ValueError(f"selected shard {index} is not present in manifest") from exc


def _shard_dir_name(shard: Mapping[str, Any]) -> str:
    name = str(shard.get("shard_run_dir_name") or "")
    if not name or Path(name).name != name or Path(name).is_absolute():
        raise ValueError("shard_run_dir_name must be a simple relative name")
    return name


def _require_repo_root(code_root: Path) -> None:
    if not (code_root / "pyproject.toml").exists():
        raise ValueError("code_root must contain pyproject.toml")
    if not (code_root / "uv.lock").exists():
        raise ValueError("code_root must contain uv.lock")


def _require_not_under_public_code(path: Path, code_root: Path, label: str) -> None:
    if _is_relative_to(path, code_root):
        raise ValueError(f"{label} must not be inside public code root")


def _require_disk_ready(path: Path, *, min_free_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".twc_write_probe"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    if shutil.disk_usage(path).free < int(min_free_bytes):
        raise ValueError("insufficient free disk space")


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


def _write_text_private(path: str | Path, value: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(value, encoding="utf-8")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _write_env_file(path: Path, env: Mapping[str, str]) -> None:
    lines = ["# Generated by twc.pretraining.w1_gpu_canary; private operator file.\n"]
    for key in sorted(env):
        value = str(env[key]).replace("'", "'\"'\"'")
        lines.append(f"export {key}='{value}'\n")
    _write_text_private(path, "".join(lines))
    try:
        path.chmod(0o600)
    except PermissionError:
        pass


def _write_sha256s(root: Path, paths: Sequence[Path]) -> None:
    rows = [f"{_sha256_file(path)}  {path.name}\n" for path in paths if path.exists()]
    _write_text_private(root / "SHA256SUMS", "".join(rows))


def _write_owner_marker(root: Path, *, run_id: str) -> None:
    marker = {
        "artifact_kind": CANARY_LAUNCHER_OWNED_ROOT_KIND,
        "schema_version": 1,
        "run_id": str(run_id),
        "claims_scope": CLAIMS_SCOPE,
        "canary_root_hash": _path_hash(root),
        **FALSE_FLAGS,
    }
    marker["marker_hash"] = stable_json_hash({key: value for key, value in marker.items() if key != "marker_hash"})
    _write_json_atomic(root / ".twc_w1_canary_launcher_owned.json", marker)


def _load_owner_marker(root: Path) -> dict[str, Any]:
    marker_path = root / ".twc_w1_canary_launcher_owned.json"
    if not marker_path.exists():
        raise ValueError("path is not a launcher-owned W1 canary root")
    marker = _load_json(marker_path)
    if marker.get("artifact_kind") != CANARY_LAUNCHER_OWNED_ROOT_KIND:
        raise ValueError("path is not a launcher-owned W1 canary root")
    if marker.get("claims_scope") != CLAIMS_SCOPE:
        raise ValueError("launcher-owned root claims_scope mismatch")
    expected_hash = stable_json_hash({key: value for key, value in marker.items() if key != "marker_hash"})
    if marker.get("marker_hash") is not None and marker.get("marker_hash") != expected_hash:
        raise ValueError("launcher-owned root marker hash mismatch")
    if marker.get("canary_root_hash") != _path_hash(root):
        raise ValueError("launcher-owned root path hash mismatch")
    return marker


def _add_bootstrap_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("bootstrap-w1-gpu-env")
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--disk-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--config", dest="config_path", required=True)
    parser.add_argument("--selected-policy-report", required=True)
    parser.add_argument("--window-mil-report", required=True)
    parser.add_argument("--window-mil-model", required=True)
    parser.add_argument("--torch-profile", choices=sorted(TORCH_PROFILES), default="locked")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--w1-attn-implementation",
        choices=sorted(SUPPORTED_W1_ATTN_IMPLEMENTATIONS),
        default=W1_ATTN_IMPLEMENTATION_SDPA,
    )
    parser.add_argument("--require-cuda", action="store_true", default=True)
    parser.add_argument("--min-free-bytes", type=int, default=1)
    parser.add_argument("--write-env")
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")


def _add_launch_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("launch-w1-runtime-cap-canary")
    parser.add_argument("--code-root")
    parser.add_argument("--source-supervisor-root", required=True)
    parser.add_argument("--diagnostic-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", dest="config_path", required=True)
    parser.add_argument("--selected-policy-report", required=True)
    parser.add_argument("--window-mil-report", required=True)
    parser.add_argument("--window-mil-model", required=True)
    parser.add_argument("--shard-index", type=int, action="append", required=True)
    parser.add_argument("--override-cap", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--synchronize-cuda-telemetry", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-reset-copied-w1-state", dest="reset_copied_w1_state", action="store_false")
    parser.add_argument("--run-id")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TWC W1 GPU bootstrap and runtime-cap canary wrappers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_bootstrap_parser(subparsers)
    _add_launch_parser(subparsers)
    stop = subparsers.add_parser("stop-w1-runtime-cap-canary")
    stop.add_argument("--canary-root", required=True)
    reset = subparsers.add_parser("reset-launcher-owned-w1-state")
    reset.add_argument("--canary-root", required=True)
    reset.add_argument("--run-id", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "bootstrap-w1-gpu-env":
        result = bootstrap_w1_gpu_env(
            BootstrapConfig(
                code_root=Path(args.code_root),
                expected_commit=args.expected_commit,
                disk_root=Path(args.disk_root),
                run_root=Path(args.run_root),
                cache_root=Path(args.cache_root),
                config_path=Path(args.config_path),
                selected_policy_report=Path(args.selected_policy_report),
                window_mil_report=Path(args.window_mil_report),
                window_mil_model=Path(args.window_mil_model),
                torch_profile=args.torch_profile,
                device=args.device,
                w1_attn_implementation=args.w1_attn_implementation,
                require_cuda=bool(args.require_cuda),
                min_free_bytes=int(args.min_free_bytes),
                write_env=Path(args.write_env) if args.write_env else None,
                skip_model_load=bool(args.skip_model_load),
                allow_dirty=bool(args.allow_dirty),
            )
        )
    elif args.command == "launch-w1-runtime-cap-canary":
        result = launch_w1_runtime_cap_canary(
            CanaryLaunchConfig(
                code_root=Path(args.code_root) if args.code_root else None,
                source_supervisor_root=Path(args.source_supervisor_root),
                diagnostic_root=Path(args.diagnostic_root),
                plan=Path(args.plan),
                manifest=Path(args.manifest),
                config_path=Path(args.config_path),
                selected_policy_report=Path(args.selected_policy_report),
                window_mil_report=Path(args.window_mil_report),
                window_mil_model=Path(args.window_mil_model),
                shard_indexes=[int(index) for index in args.shard_index],
                override_cap=int(args.override_cap),
                batch_size=int(args.batch_size),
                code_commit=str(args.code_commit),
                python_executable=str(args.python_executable),
                synchronize_cuda_telemetry=bool(args.synchronize_cuda_telemetry),
                dry_run=bool(args.dry_run),
                reset_copied_w1_state=bool(args.reset_copied_w1_state),
                run_id=args.run_id,
            )
        )
    elif args.command == "stop-w1-runtime-cap-canary":
        result = stop_w1_runtime_cap_canary(args.canary_root)
    elif args.command == "reset-launcher-owned-w1-state":
        result = reset_launcher_owned_w1_state(args.canary_root, run_id=args.run_id)
    else:  # pragma: no cover - argparse enforces valid choices.
        raise AssertionError(args.command)
    print(json.dumps(result["copyback_safe_summary"], sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
