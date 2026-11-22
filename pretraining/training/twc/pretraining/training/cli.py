"""File-driven, provider-neutral ``twc-pretrain`` operator commands."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from decimal import Decimal
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Callable, Iterator, Mapping

from twc.pretraining.training.checkpoints import CheckpointManager, CheckpointReference
from twc.pretraining.training.configuration import load_run_root
from twc.pretraining.training.contracts import (
    ExecutionAttempt,
    RunContract,
    SelectedCheckpoint,
    load_execution_attempt,
    load_training_config,
    load_worker_contract,
    training_config_contract,
    validate_resume_compatibility,
)
from twc.pretraining.training.data import (
    build_verified_placement_receipt,
    load_sequence_evaluation_artifacts,
    load_sequence_training_artifacts,
)
from twc.pretraining.training.deployment import (
    build_environment_receipt,
    build_release,
    validate_environment_receipt,
    verify_offline_bootstrap,
    verify_release_receipt,
)
from twc.pretraining.gpu_campaign_deployment import (
    validate_gpu_capability_receipt,
    verify_thin_runtime_release,
)
from twc.pretraining.configuration_semantic_code import (
    build_configuration_semantic_code_receipt,
)
from twc.pretraining.campaign_store import (
    CampaignStore,
    campaign_plan_artifact,
    load_campaign_plan,
)
from twc.pretraining.training.planning import build_device_execution_plan, device_execution_plan_artifact, global_plan_for_sequence_artifacts
from twc.pretraining.training.operator import BackupReceipt, TrainingOperator, build_backup_receipt
from twc.pretraining.training.storage import StorageLayout, StorageRole
from twc.pretraining.training.run_fetch import fetch_run, load_fetch_receipt
from twc.pretraining.training.startup import (
    ReadyPublicationInProgress,
    create_startup_namespace,
    load_watchdog_ready,
    load_worker_ready,
    read_ready_once,
)
from twc.pretraining.training.worker import (
    _load_device_plan,
    _load_global_plan,
    load_worker_spec,
    validate_worker_storage_bindings,
)
from twc.pretraining.configuration_shape_probe import (
    build_configuration_shape_selection,
)
from twc.pretraining.configuration_shape_campaign import (
    build_configuration_shape_probe_bundle,
    build_configuration_shape_probe_campaign,
    build_configuration_shape_probe_execution_matrix,
    load_configuration_shape_probe_bundle,
)
from twc.pretraining.configuration_study_execution import (
    build_configuration_execution_path_receipt,
    build_configuration_shape_probe_execution_path_receipt,
    load_configuration_shape_probe_execution_path_receipt,
)
from twc.pretraining.configuration_shape_runtime import (
    run_configuration_shape_probe_point,
)
from twc.pretraining.configuration_study import load_configuration_study_plan
from twc.pretraining.configuration_study_compiler import (
    DATA_ORDER_POLICY_V2,
    SCHEDULE_POLICY_V2,
    load_configuration_compiler_input_receipt,
)
from twc.repro.hashing import stable_json_hash
from twc.runtime.processes import (
    capture_process_identity,
    observe_process,
    provisional_process_identity,
)
from twc.runtime.host import host_identity


def _read_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _latest_execution_attempt(
    operator_root: Path, *, expected_attempt_hash: object
) -> ExecutionAttempt:
    try:
        lines = (operator_root / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        payload = json.loads(lines[-1])
    except (FileNotFoundError, IndexError, json.JSONDecodeError) as error:
        raise ValueError("operator execution-attempt ledger is unavailable") from error
    attempt = load_execution_attempt(payload)
    if attempt.identity != expected_attempt_hash:
        raise ValueError("operator active attempt differs from its attempt ledger")
    return attempt


def _write_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    if (
        path.exists()
        or path.is_symlink()
        or partial.exists()
        or partial.is_symlink()
    ):
        raise FileExistsError(f"artifact output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _write_artifact_or_verify(path: Path, payload: dict[str, Any]) -> None:
    """Create an immutable artifact, or accept an identical interrupted retry."""

    if path.is_symlink():
        raise ValueError(f"artifact path cannot be a symlink: {path}")
    if path.exists():
        if _read_json(str(path)) != payload:
            raise ValueError(f"immutable artifact conflicts with retry: {path}")
        return
    _write_artifact(path, payload)
    parent_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _reserve_artifact_output(path: Path) -> None:
    """Reserve create-once output before an expensive operation begins."""

    partial = path.with_name(path.name + ".partial")
    if path.exists() or path.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(f"artifact output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)


def _write_reserved_artifact(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"artifact output already exists: {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.is_symlink() or not partial.is_file() or partial.stat().st_size != 0:
        raise ValueError("reserved artifact output is missing or has changed")
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


@contextmanager
def _active_shape_campaign(
    *,
    campaign_root: Path,
    bundle_payload: Mapping[str, Any],
    wave_id: str,
    attempt_id: str,
) -> Iterator[Any]:
    """Hold the campaign writer lock while one paid shape point executes."""

    store = CampaignStore(campaign_root)
    with store.writer():
        ledger = store.load()
        bundle = load_configuration_shape_probe_bundle(
            bundle_payload, campaign_plan=ledger.plan
        )
        if (
            bundle.campaign_wave_id != wave_id
            or bundle.campaign_attempt_id != attempt_id
        ):
            raise ValueError("shape bundle differs from the requested campaign slot")
        wave = ledger.plan.wave(wave_id)
        started = ledger.attempt_start(attempt_id, required=False)
        stopped = ledger.attempt_stop(attempt_id, required=False)
        if (
            wave.purpose != "configuration_shape_probe"
            or wave_id not in ledger.open_wave_ids
            or started is None
            or started.wave_id != wave_id
            or stopped is not None
        ):
            raise ValueError(
                "shape probe requires its exact active, open, unclosed diagnostic attempt"
            )
        yield ledger.plan


def _configuration_shape_execution_matrices(
    *, stage: str, payloads: tuple[dict[str, Any], ...]
) -> tuple[Any, ...]:
    profiles = {
        "c2": ("m_wide", "m_prior_deep"),
        "c3": ("l_wide", "l_prior_deep"),
    }.get(stage)
    if profiles is None:
        raise ValueError("configuration shape stage is unsupported")
    if len(payloads) != 6:
        raise ValueError("shape campaign requires exactly six execution paths")
    parsed = tuple(
        load_configuration_shape_probe_execution_path_receipt(payload)
        for payload in payloads
    )
    matrices = []
    for profile in profiles:
        points = tuple(
            sorted(
                (path for path in parsed if path.probe_profile_id == profile),
                key=lambda path: path.probe_max_sequences,
            )
        )
        matrices.append(
            build_configuration_shape_probe_execution_matrix(paths=points)
        )
    if sum(len(matrix.paths) for matrix in matrices) != len(parsed):
        raise ValueError("shape campaign execution paths contain a wrong profile")
    return tuple(matrices)


_CONFIGURATION_SHAPE_STAGE_V1 = {
    "c2": {
        "phase": "m_transfer",
        "seed": 17,
        "profiles": ("m_wide", "m_prior_deep"),
        "train_view_index": 0,
    },
    "c3": {
        "phase": "l_confirmation",
        "seed": 29,
        "profiles": ("l_wide", "l_prior_deep"),
        "train_view_index": 1,
    },
}
_CONFIGURATION_GLOBAL_STEP_CAPS_V2 = (262_144, 1_073_741_824, 4_096)
_CONFIGURATION_RUNTIME_PATH_V1 = {
    "device": "cuda:0",
    "precision": "bf16",
    "attention_backend": "native_varlen",
    "optimizer_backend": "fused",
    "compile_mode": "default",
    "allow_tf32": True,
}


def _validated_configuration_placement_receipt(
    payload: Mapping[str, Any], *, data_manifest_hash: str
) -> str:
    """Validate the no-I/O placement boundary used to construct a path.

    Full file readback occurs when the placement receipt is built and again on
    worker launch.  This builder validates the immutable receipt semantics and
    exact manifest binding without repeating the large readback six times.
    """

    expected = {
        "artifact_kind",
        "schema_version",
        "pack_manifest_hash",
        "verified_file_count",
        "full_destination_readback",
        "destination_root",
        "destination_device_id",
        "file_identities",
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "receipt_hash",
    }
    unsigned = {key: value for key, value in payload.items() if key != "receipt_hash"}
    identities = payload.get("file_identities")
    if (
        set(payload) != expected
        or payload.get("artifact_kind") != "twc_data_placement_receipt_v1"
        or payload.get("schema_version") != 1
        or payload.get("receipt_hash") != stable_json_hash(unsigned)
        or payload.get("pack_manifest_hash") != data_manifest_hash
        or payload.get("full_destination_readback") is not True
        or payload.get("raw_text_serialized") is not False
        or payload.get("raw_url_serialized") is not False
        or payload.get("token_serialized") is not False
        or not isinstance(identities, list)
        or not identities
        or payload.get("verified_file_count") != len(identities)
    ):
        raise ValueError(
            "configuration shape execution path placement receipt is invalid or stale"
        )
    destination = payload.get("destination_root")
    destination_device = payload.get("destination_device_id")
    if (
        not isinstance(destination, str)
        or not Path(destination).is_absolute()
        or type(destination_device) is not int
        or destination_device < 0
    ):
        raise ValueError("configuration shape execution path placement identity is invalid")
    identity_fields = {
        "relative_path",
        "device_id",
        "inode",
        "byte_count",
        "mtime_ns",
        "sha256",
    }
    for row in identities:
        relative = row.get("relative_path") if isinstance(row, Mapping) else None
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != identity_fields
            or path is None
            or path.is_absolute()
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
            or type(row.get("device_id")) is not int
            or row.get("device_id") != destination_device
            or type(row.get("inode")) is not int
            or int(row["inode"]) < 0
            or type(row.get("byte_count")) is not int
            or int(row["byte_count"]) < 0
            or type(row.get("mtime_ns")) is not int
            or int(row["mtime_ns"]) < 0
            or not isinstance(row.get("sha256"), str)
            or len(str(row["sha256"])) != 64
            or any(character not in "0123456789abcdef" for character in str(row["sha256"]))
        ):
            raise ValueError(
                "configuration shape execution path placement file identity is invalid"
            )
    return str(payload["receipt_hash"])


def _build_configuration_shape_execution_path_from_artifacts(
    *,
    stage: str,
    profile_id: str,
    max_sequences: int,
    study_payload: Mapping[str, Any],
    compiler_input_payload: Mapping[str, Any],
    release_payload: Mapping[str, Any],
    environment_payload: Mapping[str, Any],
    placement_payload: Mapping[str, Any],
    capability_payload: Mapping[str, Any],
):
    """Derive one diagnostic path solely from validated canonical evidence."""

    spec = _CONFIGURATION_SHAPE_STAGE_V1.get(stage)
    if spec is None or profile_id not in spec["profiles"]:
        raise ValueError("configuration shape execution path stage/profile mismatch")
    study = load_configuration_study_plan(study_payload)
    inputs = load_configuration_compiler_input_receipt(compiler_input_payload)
    matches = tuple(
        condition
        for condition in study.comparison_spec.conditions
        if condition.phase == spec["phase"]
        and condition.seed == spec["seed"]
        and condition.profile_id == profile_id
    )
    if len(matches) != 1:
        raise ValueError(
            "configuration shape execution path study lacks its exact stage/profile"
        )
    condition = matches[0]
    if (
        inputs.tokenizer_adoption_receipt_hash
        != study.tokenizer_adoption_receipt_hash
        or inputs.data_plan_hash != study.data_plan_hash
        or inputs.prefix_continuity_receipt_hash
        != study.prefix_continuity_receipt_hash
        or inputs.train_view_hash
        != study.train_view_hashes[int(spec["train_view_index"])]
        or inputs.validation_view_hash != study.validation_view_hash
        or inputs.train_view_loss_tokens != condition.target_loss_tokens
        or inputs.global_step_plan_loss_tokens != condition.target_loss_tokens
    ):
        raise ValueError(
            "configuration shape execution path compiler input/study stage lineage mismatch"
        )
    if (
        inputs.optimizer_step_token_cap,
        inputs.optimizer_step_attention_cell_cap,
        inputs.optimizer_step_sequence_cap,
    ) != _CONFIGURATION_GLOBAL_STEP_CAPS_V2:
        raise ValueError("configuration shape execution path global caps mismatch")
    if (
        study.data_order_policy_hash != stable_json_hash(DATA_ORDER_POLICY_V2)
        or study.schedule_policy_hash != stable_json_hash(SCHEDULE_POLICY_V2)
    ):
        raise ValueError("configuration shape execution path study policy mismatch")

    verified_release = verify_thin_runtime_release(release_payload)
    environment = validate_environment_receipt(
        environment_payload, require_cuda_ready=True
    )
    mismatches = [
        field
        for field, expected in _CONFIGURATION_RUNTIME_PATH_V1.items()
        if environment.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            f"configuration shape execution path environment mismatch: {sorted(mismatches)}"
        )
    placement_hash = _validated_configuration_placement_receipt(
        placement_payload, data_manifest_hash=inputs.data_manifest_hash
    )
    validate_gpu_capability_receipt(capability_payload, require_ready=True)
    capability_mismatches = []
    expected_capability = {
        "release_hash": verified_release["release_hash"],
        "dependency_lock_sha256": release_payload.get("dependency_lock_sha256"),
        "torch_version": environment["torch_version"],
        "cuda_runtime_version": environment["cuda_runtime_version"],
        "compile_mode": environment["compile_mode"],
        "supports_bf16": True,
        "supports_tf32": True,
        "native_varlen_available": True,
        "fused_optimizer_available": True,
    }
    for field, expected in expected_capability.items():
        if capability_payload.get(field) != expected:
            capability_mismatches.append(field)
    if capability_mismatches:
        raise ValueError(
            "configuration shape execution path capability/release/environment "
            f"mismatch: {sorted(capability_mismatches)}"
        )

    return build_configuration_shape_probe_execution_path_receipt(
        stage=stage,
        probe_profile_id=profile_id,
        study_plan_hash=study.identity,
        comparison_spec_hash=study.comparison_spec.identity,
        data_manifest_hash=inputs.data_manifest_hash,
        train_view_hash=inputs.train_view_hash,
        global_step_plan_hash=inputs.global_step_plan_hash,
        release_hash=str(verified_release["release_hash"]),
        environment_hash=str(environment["environment_hash"]),
        placement_hash=placement_hash,
        device_identity_hash=str(capability_payload["capability_hash"]),
        probe_max_sequences=max_sequences,
    )


def _validate_worker_storage_from_spec(worker_spec: Path) -> None:
    spec = load_worker_spec(worker_spec)
    validate_worker_storage_bindings(
        pack_root=spec.pack_root,
        placement_receipt=_read_json(str(spec.data_placement_receipt_path)),
        storage_layout_receipt=_read_json(str(spec.storage_layout_receipt_path)),
        checkpoint_root=spec.checkpoint_root,
        persistent_checkpoint_root=spec.persistent_checkpoint_root,
        checkpoint_topology=spec.checkpoint_topology,
    )


def _validate_static_launch_inputs(
    *, worker_spec: Path, mode: str, startup_timeout_seconds: int
) -> None:
    if mode != "run":
        raise ValueError("launch mode must be run")
    if not 1 <= startup_timeout_seconds <= 3600:
        raise ValueError("startup timeout must be between 1 and 3600 seconds")
    _validate_worker_storage_from_spec(worker_spec)
    spec = load_worker_spec(worker_spec)
    contract = load_worker_contract(_read_json(str(spec.contract_path)))
    if not isinstance(contract, RunContract):
        raise ValueError("generic launch requires a run contract")
    release_root, environment = _release_launch_context(worker_spec)
    if (
        environment.get("PYTHONPATH") != str(release_root / "training")
        or environment.get("PYTHONNOUSERSITE") != "1"
        or environment.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise ValueError("release launch environment is not canonical")
    for module in ("worker.py", "watchdog.py"):
        path = release_root / "training/twc/pretraining/training" / module
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"release lacks fixed child module: {module}")


def _release_launch_context(worker_spec: Path) -> tuple[Path, dict[str, str]]:
    """Revalidate and launch from the exact content-addressed release."""
    spec_path = worker_spec.resolve(strict=True)
    spec = _read_json(str(spec_path))
    receipt_path = Path(str(spec["release_receipt_path"]))
    if not receipt_path.is_absolute():
        receipt_path = spec_path.parent / receipt_path
    release_receipt = _read_json(str(receipt_path))
    verified = (
        verify_thin_runtime_release(release_receipt)
        if release_receipt.get("artifact_kind") == "twc_gpu_runtime_release_v1"
        else verify_release_receipt(release_receipt)
    )
    release_root = Path(str(verified["path"]))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(release_root / "training")
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return release_root, environment


_EXECUTION_FIELDS = {
    "artifact_kind",
    "schema_version",
    "pack_root",
    "checkpoint_root",
    "checkpoint_topology",
    "device",
    "read_mode",
    "pin_memory",
    "telemetry_every_steps",
    "checkpoint_every_loss_tokens",
    "checkpoint_max_interval_seconds",
    "durable_every_loss_tokens",
    "durable_max_interval_seconds",
}


def _load_run_execution(run_root: Path) -> dict[str, Any]:
    payload = _read_json(str(run_root / "execution.json"))
    if set(payload) != _EXECUTION_FIELDS:
        raise ValueError("execution settings fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_pretraining_execution_v1"
        or payload.get("schema_version") != 1
        or payload.get("checkpoint_topology") not in {
            "local_only",
            "direct_persistent",
        }
        or payload.get("device") not in {"cpu", "cuda:0"}
        or payload.get("read_mode") not in {"local_mmap", "buffered_pread"}
        or type(payload.get("pin_memory")) is not bool
    ):
        raise ValueError("execution settings are invalid")
    for field in (
        "telemetry_every_steps",
        "checkpoint_every_loss_tokens",
        "checkpoint_max_interval_seconds",
        "durable_every_loss_tokens",
        "durable_max_interval_seconds",
    ):
        if type(payload.get(field)) is not int or payload[field] <= 0:
            raise ValueError(f"execution setting {field} must be a positive integer")
    if payload["durable_every_loss_tokens"] < payload["checkpoint_every_loss_tokens"]:
        raise ValueError("durable cadence cannot be tighter than checkpoint cadence")
    if payload["durable_max_interval_seconds"] < payload["checkpoint_max_interval_seconds"]:
        raise ValueError("durable interval cannot be shorter than checkpoint interval")
    return payload


def _public_run_context(
    run_root: Path,
    *,
    load_desired_execution: bool = True,
) -> tuple[TrainingOperator, Path, dict[str, Any]]:
    run_root = run_root.expanduser().resolve(strict=True)
    config = load_run_root(run_root)
    execution = (
        _load_run_execution(run_root) if load_desired_execution else None
    )
    operator_root = run_root / "operator"
    state_path = operator_root / "state.json"
    active_attempt: ExecutionAttempt | None = None
    if state_path.exists():
        state = TrainingOperator(operator_root).status()
        active_attempt = _latest_execution_attempt(
            operator_root, expected_attempt_hash=state["attempt_hash"]
        )
        if (run_root / "fetch-receipt.json").is_file():
            load_fetch_receipt(run_root)
            checkpoint_root_value = str(run_root / "checkpoints")
            topology = "direct_persistent"
            persistent_value = None
        else:
            checkpoint_root_value = active_attempt.checkpoint_root
            topology = active_attempt.checkpoint_topology
            persistent_value = active_attempt.persistent_checkpoint_root
        if checkpoint_root_value is None or topology is None:
            raise ValueError("active attempt lacks an exact execution snapshot")
    else:
        if execution is None:
            raise ValueError("an uninitialized run requires execution settings")
        checkpoint_root_value = str(execution["checkpoint_root"])
        topology = str(execution["checkpoint_topology"])
        persistent_value = None
    checkpoint_root = Path(checkpoint_root_value).expanduser().resolve(strict=True)
    persistent_root = (
        Path(persistent_value).expanduser().resolve(strict=True)
        if persistent_value is not None
        else None
    )
    operator = TrainingOperator(
        operator_root,
        checkpoint_manager=CheckpointManager(
            checkpoint_root,
            persistent_root=persistent_root,
            topology=topology,
            create=False,
        ),
    )
    if operator.state_path.exists():
        state = operator.status()
        if training_config_contract(config).identity != state.get("run_identity"):
            raise ValueError("training configuration changed after initialization")
        binding = _read_json(str(operator.root / "configuration-binding.json"))
        if binding != {
            "configuration_hash": config["configuration_hash"],
            "run_identity": state["run_identity"],
        }:
            raise ValueError("training configuration changed after initialization")
    return operator, run_root, {
        "config": config,
        "execution": execution,
        "active_attempt": active_attempt,
    }


def _initialize_public_run(
    operator: TrainingOperator,
    run_root: Path,
    context: Mapping[str, Any],
) -> Path:
    """Prepare one initial attempt from the run directory, once."""

    operator_root = operator.root
    if operator.state_path.exists():
        state = operator.status()
        spec_path = (
            operator_root / "worker-specs" / f"{state['attempt_hash']}.json"
        )
        if not spec_path.is_file():
            raise ValueError("initialized run is missing its internal worker specification")
        return spec_path
    config = dict(context["config"])
    execution = dict(context["execution"])
    contract = training_config_contract(config)
    runtime = run_root / "runtime"
    artifacts = {
        "records_path": runtime / "train-records.json",
        "validation_records_path": runtime / "validation-records.json",
        "global_step_plan_path": runtime / "global-plan.json",
        "device_execution_plan_path": runtime / "device-plan.json",
        "release_receipt_path": runtime / "release.json",
        "environment_receipt_path": runtime / "environment.json",
        "data_placement_receipt_path": runtime / "placement.json",
        "storage_layout_receipt_path": runtime / "storage.json",
    }
    if any(not path.is_file() for path in artifacts.values()):
        missing = sorted(name for name, path in artifacts.items() if not path.is_file())
        raise ValueError(f"run runtime artifacts are missing: {missing}")
    records = _read_json(str(artifacts["records_path"]))
    validation_records = _read_json(str(artifacts["validation_records_path"]))
    device_plan = _read_json(str(artifacts["device_execution_plan_path"]))
    release = _read_json(str(artifacts["release_receipt_path"]))
    environment = _read_json(str(artifacts["environment_receipt_path"]))
    placement = _read_json(str(artifacts["data_placement_receipt_path"]))
    storage = _read_json(str(artifacts["storage_layout_receipt_path"]))
    expected = {
        "global_step_plan_hash": _read_json(str(artifacts["global_step_plan_path"]))["plan_hash"],
        "release_hash": release["release_hash"],
        "environment_hash": environment["environment_hash"],
        "data_placement_hash": placement["receipt_hash"],
        "storage_layout_hash": storage["receipt_hash"],
        "device_execution_plan_hash": device_plan["plan_hash"],
    }
    if expected["global_step_plan_hash"] != contract.global_step_plan_hash:
        raise ValueError("runtime global plan differs from the training configuration")
    try:
        semantic_code_hash = build_configuration_semantic_code_receipt(
            Path(str(release["path"]))
        )["semantic_code_hash"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("runtime release semantic projection is invalid") from error
    if semantic_code_hash != contract.code_hash:
        raise ValueError("runtime release changes the training implementation")
    operator_root.mkdir(mode=0o700, exist_ok=True)
    _write_artifact_or_verify(
        operator_root / "configuration-binding.json",
        {
            "configuration_hash": config["configuration_hash"],
            "run_identity": contract.identity,
        },
    )
    _write_artifact_or_verify(operator_root / "execution.json", execution)
    model_path = operator_root / "model.json"
    _write_artifact_or_verify(model_path, dict(config["model"]))
    spec = {
        "artifact_kind": "twc_pretraining_worker_spec_v2",
        "schema_version": 2,
        "contract_path": str(operator.contract_path),
        "model_path": str(model_path),
        **{name: str(path) for name, path in artifacts.items()},
        "pack_root": str(Path(str(execution["pack_root"])).expanduser().resolve(strict=True)),
        "checkpoint_root": str(operator.checkpoint_manager.root),
        "persistent_checkpoint_root": None,
        "checkpoint_topology": execution["checkpoint_topology"],
        "device": execution["device"],
        "read_mode": execution["read_mode"],
        "removable_data": True,
    }
    spec["spec_hash"] = stable_json_hash(spec)
    attempt = ExecutionAttempt(
        run_identity=contract.identity,
        attempt_id=stable_json_hash(
            {
                "run_identity": contract.identity,
                "execution": execution,
                "runtime": expected,
                "host_id": host_identity(),
            }
        )[:32],
        parent_attempt_hash=None,
        device_execution_plan_hash=str(expected["device_execution_plan_hash"]),
        release_hash=str(expected["release_hash"]),
        environment_hash=str(expected["environment_hash"]),
        data_placement_hash=str(expected["data_placement_hash"]),
        storage_layout_hash=str(expected["storage_layout_hash"]),
        host_id=host_identity(),
        fencing_epoch=1,
        prefetch_batches=1,
        pin_memory=bool(execution["pin_memory"]),
        telemetry_every_steps=int(execution["telemetry_every_steps"]),
        checkpoint_every_loss_tokens=int(execution["checkpoint_every_loss_tokens"]),
        checkpoint_max_interval_seconds=int(execution["checkpoint_max_interval_seconds"]),
        durable_every_loss_tokens=int(execution["durable_every_loss_tokens"]),
        durable_max_interval_seconds=int(execution["durable_max_interval_seconds"]),
        evaluation_every_loss_tokens=None,
        train_records_hash=str(records["records_hash"]),
        validation_records_hash=str(validation_records["records_hash"]),
        pack_root=spec["pack_root"],
        checkpoint_root=spec["checkpoint_root"],
        persistent_checkpoint_root=spec["persistent_checkpoint_root"],
        checkpoint_topology=str(spec["checkpoint_topology"]),
        device=str(spec["device"]),
        read_mode=str(spec["read_mode"]),
        selected_checkpoint=None,
    )
    spec_path = operator_root / "worker-specs" / f"{attempt.identity}.json"
    _write_artifact_or_verify(spec_path, spec)
    try:
        operator.plan(contract, attempt)
        operator.prepare_runtime(
            runtime_safety={
                "heartbeat_timeout_seconds": 180,
                "cooperative_stop_grace_seconds": 60,
            },
            storage_layout_receipt=storage,
        )
    finally:
        operator.release_writer()
    return spec_path


def _finalize_public_result(
    *, operator: TrainingOperator, run_root: Path, state: Mapping[str, Any]
) -> None:
    if state.get("state") != "COMPLETED":
        return
    outcome_path = (
        operator.root
        / "attempt-outcomes"
        / f"{state['attempt_hash']}.json"
    )
    outcome = _read_json(str(outcome_path))
    if (
        outcome.get("result_hash")
        != stable_json_hash(
            {key: value for key, value in outcome.items() if key != "result_hash"}
        )
        or outcome.get("completed") is not True
        or outcome.get("attempt_hash") != state["attempt_hash"]
        or outcome.get("run_identity") != state["run_identity"]
    ):
        raise ValueError("completed attempt outcome is invalid")
    checkpoint_hashes = (
        outcome.get("checkpoint_manifest_sha256"),
        outcome.get("checkpoint_state_sha256"),
    )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in checkpoint_hashes
    ):
        raise ValueError("completed attempt outcome checkpoint identity is invalid")
    if operator.checkpoint_manager is None:
        raise RuntimeError("checkpoint manager is required for completed result")
    terminal = operator.checkpoint_manager.validate_reference(
        CheckpointReference(
            int(outcome["checkpoint_generation"]),
            operator.checkpoint_manager.root
            / f"generation-{int(outcome['checkpoint_generation']):08d}",
            str(outcome["checkpoint_durability"]),
            str(checkpoint_hashes[0]),
            str(checkpoint_hashes[1]),
        )
    )
    if (
        terminal.run_identity != state["run_identity"]
        or terminal.attempt_hash != state["attempt_hash"]
        or terminal.generation != state["checkpoint_generation"]
        or terminal.durability != state["checkpoint_durability"]
        or terminal.step != state["step"]
        or terminal.consumed_loss_tokens != outcome["consumed_loss_tokens"]
        or terminal.next_global_ordinal != state["next_global_ordinal"]
    ):
        raise ValueError("completed result terminal checkpoint binding is invalid")
    config = load_training_config(_read_json(str(run_root / "run.json")))
    result = {
        "artifact_kind": "twc_pretraining_result_v1",
        "schema_version": 1,
        "configuration_hash": config["configuration_hash"],
        "run_identity": state["run_identity"],
        "attempt_hash": state["attempt_hash"],
        "state": "COMPLETED",
        "step": state["step"],
        "consumed_loss_tokens": outcome["consumed_loss_tokens"],
        "next_global_ordinal": state["next_global_ordinal"],
        "checkpoint_generation": state["checkpoint_generation"],
        "checkpoint_durability": state["checkpoint_durability"],
        "checkpoint_manifest_sha256": checkpoint_hashes[0],
        "checkpoint_state_sha256": checkpoint_hashes[1],
        "evaluation_loss": outcome["evaluation_loss"],
        "evaluation_loss_tokens": outcome["evaluation_loss_tokens"],
        "evaluation_receipt_hash": outcome["evaluation_receipt_hash"],
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "tokens_serialized": False,
        "secrets_serialized": False,
    }
    result["result_hash"] = stable_json_hash(result)
    path = run_root / "result.json"
    if path.exists():
        if _read_json(str(path)) != result:
            raise ValueError("generic result is immutable")
        return
    _write_artifact(path, result)


def _checkpoint_request_id(
    operator: TrainingOperator, state: Mapping[str, Any]
) -> str:
    pending: list[str] = []
    requests_root = operator.root / "requests"
    for path in sorted(requests_root.glob("*.json")):
        request = _read_json(str(path))
        if (
            request.get("action") != "checkpoint"
            or request.get("run_identity") != state["run_identity"]
            or request.get("attempt_hash") != state["attempt_hash"]
            or request.get("fencing_epoch") != state["fencing_epoch"]
        ):
            continue
        request_id = request.get("request_id")
        if (
            not isinstance(request_id, str)
            or path.stem != request_id
            or request.get("request_hash")
            != stable_json_hash(
                {key: value for key, value in request.items() if key != "request_hash"}
            )
        ):
            raise ValueError("checkpoint request identity is invalid")
        ack_path = operator.root / "request-acks" / f"{request_id}.json"
        if not ack_path.is_file():
            pending.append(request_id)
            continue
        ack = _read_json(str(ack_path))
        if (
            ack.get("ack_hash")
            != stable_json_hash(
                {key: value for key, value in ack.items() if key != "ack_hash"}
            )
            or ack.get("request_id") != request_id
            or ack.get("request_hash") != request["request_hash"]
        ):
            raise ValueError("checkpoint acknowledgement identity is invalid")
        committed_generation = state.get("checkpoint_generation")
        committed = (
            type(committed_generation) is int
            and type(ack.get("generation")) is int
            and ack["generation"] <= committed_generation
            and type(ack.get("step")) is int
            and ack["step"] <= state["step"]
            and type(ack.get("next_global_ordinal")) is int
            and ack["next_global_ordinal"] <= state["next_global_ordinal"]
        )
        if not committed:
            pending.append(request_id)
    if len(pending) > 1:
        raise ValueError("multiple uncommitted checkpoint requests require reconciliation")
    return pending[0] if pending else uuid.uuid4().hex


def _checkpoint_manager(args: argparse.Namespace) -> CheckpointManager | None:
    if not getattr(args, "checkpoints_root", None):
        return None
    persistent = Path(args.persistent_root) if getattr(args, "persistent_root", None) else None
    return CheckpointManager(
        Path(args.checkpoints_root),
        persistent_root=persistent,
        topology=getattr(args, "checkpoint_topology", None),
        create=False,
    )


def _operator(args: argparse.Namespace) -> TrainingOperator:
    if getattr(args, "run", None):
        return _public_run_context(
            Path(args.run),
            load_desired_execution=args.command in {"start", "resume"},
        )[0]
    return TrainingOperator(Path(args.root), checkpoint_manager=_checkpoint_manager(args))


def _mutate(operator: TrainingOperator, action: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    operator.claim_writer()
    try:
        return action()
    finally:
        operator.release_writer()


@contextmanager
def _checkpoint_writer(
    operator: TrainingOperator, *, deadline_monotonic: float
) -> Iterator[None]:
    while True:
        try:
            operator.claim_writer()
            break
        except FileExistsError as error:
            if str(error) != "writer lease already exists":
                raise
            if time.monotonic() >= deadline_monotonic:
                raise TimeoutError("checkpoint writer lease timed out") from error
            time.sleep(0.01)
    try:
        yield
    finally:
        operator.release_writer()


def _require_reserved_launch(
    operator: TrainingOperator, *, command_id: str, mode: str,
    kind: str, attempt_hash: str, fencing_epoch: int,
) -> None:
    state = operator.status()
    reserved = state.get("start_command")
    expected = {
        "command_id": command_id, "mode": mode, "kind": kind,
        "fencing_epoch": fencing_epoch,
    }
    if (
        state.get("state") != "STARTING"
        or state.get("attempt_hash") != attempt_hash
        or state.get("fencing_epoch") != fencing_epoch
        or not isinstance(reserved, dict)
        or any(reserved.get(field) != value for field, value in expected.items())
    ):
        raise RuntimeError("launcher no longer owns the exact STARTING reservation")


class _LaunchFailure(RuntimeError):
    def __init__(self, code: str, role: str, exit_code: int | None = None) -> None:
        super().__init__(f"{role} startup failed: {code}")
        self.code = code
        self.role = role
        self.exit_code = exit_code


def _await_ready(
    namespace: Any,
    *,
    role: str,
    process: subprocess.Popen[Any],
    identity: Mapping[str, Any],
    deadline: float,
) -> dict[str, Any]:
    loader = load_worker_ready if role == "worker" else load_watchdog_ready
    while True:
        exit_code = process.poll()
        if exit_code is not None:
            raise _LaunchFailure("child_pre_ready_exit", role, exit_code)
        if observe_process(identity) != "confirmed_alive":
            raise _LaunchFailure("child_identity_unavailable", role)
        try:
            return loader(read_ready_once(namespace, role))
        except FileNotFoundError:
            pass
        except ReadyPublicationInProgress:
            pass
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise _LaunchFailure("child_ready_invalid", role) from error
        if time.monotonic() >= deadline:
            exit_code = process.poll()
            if exit_code is not None:
                raise _LaunchFailure("child_pre_ready_exit", role, exit_code)
            try:
                return loader(read_ready_once(namespace, role))
            except FileNotFoundError as error:
                raise _LaunchFailure("child_ready_timeout", role) from error
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise _LaunchFailure("child_ready_invalid", role) from error
        time.sleep(0.025)


def _cleanup_reserved_children(children: list[dict[str, Any]]) -> bool:
    certain = True
    for child in reversed(children):
        process = child["process"]
        identity = child.get("identity")
        try:
            if identity is None:
                if process.poll() is None:
                    process.terminate()
            else:
                observation = observe_process(identity)
                if observation == "confirmed_alive":
                    os.killpg(int(identity["process_group_id"]), signal.SIGTERM)
                elif observation != "exited":
                    certain = False
        except (OSError, ValueError):
            certain = False
    for child in reversed(children):
        process = child["process"]
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            identity = child.get("identity")
            try:
                if identity is None:
                    process.kill()
                elif observe_process(identity) == "confirmed_alive":
                    os.killpg(int(identity["process_group_id"]), signal.SIGKILL)
                else:
                    certain = False
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired, ValueError):
                certain = False
        except OSError:
            certain = False
        if process.poll() is None:
            certain = False
        pgid = child.get("process_group_id")
        if type(pgid) is not int:
            certain = False
        else:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                pass
            except (PermissionError, OSError):
                certain = False
            else:
                certain = False
    return certain


def _launch_reserved_runtime(
    *,
    operator: TrainingOperator,
    root: Path,
    worker_spec: Path,
    release_root: Path,
    environment: Mapping[str, str],
    command_id: str,
    command_kind: str,
    mode: str,
    startup_timeout_seconds: int,
) -> dict[str, Any]:
    ready_deadline = time.monotonic() + startup_timeout_seconds
    activation_deadline = ready_deadline + 15.0
    logs: list[Any] = []
    children: list[dict[str, Any]] = []
    attempt_hash = str(operator.status()["attempt_hash"])
    namespace = None
    try:
        namespace = create_startup_namespace(root, command_id)
        spec = load_worker_spec(worker_spec)
        sanitized_environment = {
            "PYTHONPATH": str(environment["PYTHONPATH"]),
            "PYTHONNOUSERSITE": str(environment["PYTHONNOUSERSITE"]),
        }
    except Exception:
        if namespace is not None:
            namespace.close()
        operator.fail_start(
            command_id=command_id,
            failure_code="launcher_preflight_failed",
            child_role="launcher",
        )
        raise

    def spawn(role: str, argv: list[str]) -> tuple[subprocess.Popen[Any], dict[str, Any]]:
        label, process_role = (
            ("r7-pretraining-worker", "trainer")
            if role == "worker"
            else ("r7-pretraining-watchdog", "guard")
        )
        command = {
            "argv": argv,
            "env": sanitized_environment,
            "label": label,
            "role": process_role,
        }
        try:
            log = namespace.open_log_once(role)
        except Exception as error:
            raise _LaunchFailure("child_launch_failed", role) from error
        logs.append(log)
        try:
            process = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, shell=False,
                cwd=release_root, env=dict(environment),
            )
        except Exception as error:
            raise _LaunchFailure("child_launch_failed", role) from error
        provisional = provisional_process_identity(
            process.pid, command=command, attempt_id=attempt_hash
        )
        child = {
            "role": role,
            "process": process,
            "process_group_id": provisional.get("process_group_id"),
            "provisional": provisional,
            "identity": None,
        }
        children.append(child)
        if provisional.get("process_group_id") != process.pid:
            raise _LaunchFailure("child_identity_unavailable", role)
        try:
            identity = capture_process_identity(
                process.pid, command=command, attempt_id=attempt_hash
            )
        except Exception as error:
            raise _LaunchFailure("child_identity_unavailable", role) from error
        if identity.get("process_group_id") != process.pid:
            raise _LaunchFailure("child_identity_unavailable", role)
        child["identity"] = identity
        return process, identity

    failure: _LaunchFailure | None = None
    try:
        root_value = str(root.resolve())
        deadline_value = repr(activation_deadline)
        watchdog_argv = [
            sys.executable, "-m", "twc.pretraining.training.watchdog",
            "--operator-root", root_value,
            "--checkpoint-root", str(spec.checkpoint_root.resolve()),
            "--worker-spec", str(worker_spec),
            "--launch-id", command_id,
            "--activation-deadline-monotonic", deadline_value,
        ]
        watchdog_process, watchdog_identity = spawn("watchdog", watchdog_argv)
        watchdog_ready = _await_ready(
            namespace, role="watchdog", process=watchdog_process,
            identity=watchdog_identity,
            deadline=min(time.monotonic() + 30.0, ready_deadline),
        )
        worker_argv = [
            sys.executable, "-m", "twc.pretraining.training.worker",
            "--spec", str(worker_spec), "--operator-root", root_value,
            "--launch-id", command_id,
            "--activation-deadline-monotonic", deadline_value,
        ]
        worker_process, worker_identity = spawn("worker", worker_argv)
        worker_ready = _await_ready(
            namespace, role="worker", process=worker_process,
            identity=worker_identity, deadline=ready_deadline,
        )
        try:
            _require_reserved_launch(
                operator, command_id=command_id, mode=mode, kind=command_kind,
                attempt_hash=attempt_hash,
                fencing_epoch=int(operator.status()["fencing_epoch"]),
            )
        except Exception as error:
            raise _LaunchFailure("confirmation_rejected", "launcher") from error
        final_ready: dict[str, dict[str, Any]] = {}
        for role, loader in (
            ("watchdog", load_watchdog_ready), ("worker", load_worker_ready)
        ):
            try:
                final_ready[role] = loader(read_ready_once(namespace, role))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise _LaunchFailure("child_ready_invalid", role) from error
        watchdog_ready = final_ready["watchdog"]
        worker_ready = final_ready["worker"]
        try:
            for log in logs:
                log.close()
            logs.clear()
            namespace.close()
            namespace = None
        except Exception as error:
            raise _LaunchFailure("confirmation_rejected", "launcher") from error
        try:
            return operator.confirm_start(
                command_id=command_id,
                command_kind=command_kind,
                worker_process_identity=worker_identity,
                watchdog_process_identity=watchdog_identity,
                worker_ready_receipt=worker_ready,
                watchdog_ready_receipt=watchdog_ready,
            )
        except Exception as error:
            state = operator.status()
            if state.get("state") != "STARTING":
                return operator.confirm_start(
                    command_id=command_id,
                    command_kind=command_kind,
                    worker_process_identity=worker_identity,
                    watchdog_process_identity=watchdog_identity,
                    worker_ready_receipt=worker_ready,
                    watchdog_ready_receipt=watchdog_ready,
                )
            raise _LaunchFailure("confirmation_rejected", "launcher") from error
    except _LaunchFailure as error:
        failure = error
        raise
    finally:
        if failure is not None:
            current = operator.status()
            reservation = current.get("start_command")
            exact = (
                current.get("state") == "STARTING"
                and current.get("attempt_hash") == attempt_hash
                and isinstance(reservation, dict)
                and reservation.get("command_id") == command_id
                and reservation.get("kind") == command_kind
            )
            absent = _cleanup_reserved_children(children) if exact else False
            if exact and absent:
                operator.fail_start(
                    command_id=command_id, failure_code=failure.code,
                    child_role=failure.role, exit_code=failure.exit_code,
                )
            elif exact:
                operator.mark_orphaned(
                    reason="startup_cleanup_uncertain",
                    command_id=command_id,
                    failure_code=failure.code,
                    child_role=failure.role,
                    exit_code=failure.exit_code,
                    startup_orphan_processes={
                        child["role"]: child.get("identity") or child["provisional"]
                        for child in children
                    },
                )
        for log in logs:
            try:
                log.close()
            except OSError:
                pass
        if namespace is not None:
            try:
                namespace.close()
            except OSError:
                pass


def _add_root(parser: argparse.ArgumentParser, *, checkpoints: bool = False) -> None:
    parser.add_argument("--root", required=True)
    if checkpoints:
        parser.add_argument("--checkpoints-root", required=True)
        parser.add_argument("--persistent-root")
        parser.add_argument(
            "--checkpoint-topology",
            choices=("local_only", "direct_persistent", "local_then_replicate"),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="twc-pretrain")
    sub = parser.add_subparsers(dest="command", required=True)
    compile_base = sub.add_parser("compile-base")
    for name in (
        "run",
        "run-id",
        "model-config",
        "training-config",
        "pack-manifest",
        "tokenizer-manifest",
        "train-view",
        "validation-view",
        "global-plan",
        "semantic-code-receipt",
        "evaluation-manifest",
        "evaluation-implementation",
    ):
        compile_base.add_argument(f"--{name}", required=True)
    check = sub.add_parser("check")
    check.add_argument("--run", required=True)
    verify_backup = sub.add_parser("verify-backup")
    verify_backup.add_argument("--checkpoints-root", required=True)
    verify_backup.add_argument("--persistent-root")
    verify_backup.add_argument(
        "--topology", choices=("direct_persistent", "local_then_replicate"), required=True
    )
    verify_backup.add_argument("--generation", required=True, type=int)
    verify_backup.add_argument("--destination-root", required=True)
    verify_backup.add_argument("--output", required=True)
    start = sub.add_parser("start")
    start.add_argument("--run", required=True)
    start.add_argument("--startup-timeout-seconds", type=int, default=600)
    status = sub.add_parser("status")
    status.add_argument("--run", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--run", required=True)
    fetch.add_argument("--destination", required=True)
    for name in ("checkpoint", "stop"):
        command = sub.add_parser(name)
        command.add_argument("--run", required=True)
        command.add_argument("--timeout-seconds", type=float, default=300.0)
    resume = sub.add_parser("resume")
    resume.add_argument("--run", required=True)
    resume.add_argument("--startup-timeout-seconds", type=int, default=600)
    teardown = sub.add_parser("teardown-readiness")
    _add_root(teardown)
    teardown.add_argument("--scope", required=True, choices=("pod", "disk"))
    backup = sub.add_parser("record-backup")
    _add_root(backup, checkpoints=True)
    backup.add_argument("--receipt", required=True)
    backup.add_argument("--destination-root", required=True)
    repair = sub.add_parser("repair-checkpoints")
    repair.add_argument("--checkpoints-root", required=True)
    repair.add_argument("--persistent-root")
    repair.add_argument(
        "--topology",
        choices=("local_only", "direct_persistent", "local_then_replicate"),
    )
    release = sub.add_parser("build-release")
    release.add_argument("--source-root", required=True)
    release.add_argument("--release-root", required=True)
    bootstrap = sub.add_parser("verify-bootstrap")
    bootstrap.add_argument("--release-receipt", required=True)
    bootstrap.add_argument("--python", default=sys.executable)
    bootstrap.add_argument("--python-major", type=int, required=True)
    bootstrap.add_argument("--python-minor", type=int, required=True)
    environment = sub.add_parser("capture-environment")
    environment.add_argument("--device", required=True)
    environment.add_argument("--precision", required=True)
    environment.add_argument("--attention-backend", required=True)
    environment.add_argument("--optimizer-backend", required=True)
    environment.add_argument("--compile-mode", required=True)
    environment.add_argument("--allow-tf32", action="store_true")
    environment.add_argument("--output", required=True)
    prepare = sub.add_parser("prepare-worker-data")
    prepare.add_argument("--pack-root", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--view", required=True)
    prepare.add_argument("--validation-view", required=True)
    prepare.add_argument("--global-plan", required=True)
    prepare.add_argument("--placement-receipt", required=True)
    prepare.add_argument("--tokenizer-hash", required=True)
    prepare.add_argument("--max-loss-tokens", type=int, required=True)
    prepare.add_argument("--max-attention-cells", type=int, required=True)
    prepare.add_argument("--max-sequences", type=int, required=True)
    prepare.add_argument("--records-output", required=True)
    prepare.add_argument("--validation-records-output", required=True)
    prepare.add_argument("--device-plan-output", required=True)
    placement = sub.add_parser("verify-data-placement")
    placement.add_argument("--pack-root", required=True)
    placement.add_argument("--manifest", required=True)
    placement.add_argument("--output", required=True)
    storage = sub.add_parser("validate-storage")
    for name in ("control-root", "data-root"):
        storage.add_argument(f"--{name}", required=True)
    storage.add_argument(
        "--checkpoint-topology",
        choices=("local_only", "direct_persistent", "local_then_replicate"),
        default="direct_persistent",
    )
    storage.add_argument("--durable-root")
    storage.add_argument("--local-root")
    storage.add_argument("--copyback-root")
    storage.add_argument("--data-sentinel", required=True)
    storage.add_argument("--durable-sentinel")
    storage.add_argument("--local-sentinel")
    storage.add_argument("--copyback-sentinel")
    for name in ("data", "durable", "local", "copyback"):
        storage.add_argument(f"--{name}-require-mount", action="store_true")
    storage.add_argument("--required-checkpoint-bytes", type=int, required=True)
    storage.add_argument("--reserve-bytes", type=int, default=0)
    storage.add_argument("--output", required=True)
    for name in ("orphan", "takeover"):
        command = sub.add_parser(name)
        _add_root(command)
        command.add_argument("--command-id", required=True)
        if name == "takeover":
            command.add_argument("--former-host-termination-receipt", required=True)
    request = sub.add_parser("request")
    _add_root(request)
    request.add_argument("--request-id", required=True)
    request.add_argument(
        "--action", choices=("checkpoint", "pause", "stop", "emergency_stop"), required=True
    )
    request.add_argument("--deadline-seconds", type=int, required=True)
    shape_execution_path = sub.add_parser(
        "build-configuration-shape-execution-path"
    )
    shape_execution_path.add_argument(
        "--stage", choices=("c2", "c3"), required=True
    )
    shape_execution_path.add_argument(
        "--profile-id",
        choices=("m_wide", "m_prior_deep", "l_wide", "l_prior_deep"),
        required=True,
    )
    shape_execution_path.add_argument(
        "--max-sequences", type=int, choices=(16, 20, 24), required=True
    )
    for name in (
        "study-plan",
        "compiler-input",
        "release-receipt",
        "environment-receipt",
        "placement-receipt",
        "capability-receipt",
        "output",
    ):
        shape_execution_path.add_argument(f"--{name}", required=True)
    execution_path = sub.add_parser("build-configuration-execution-path")
    execution_path.add_argument("--stage", choices=("c2", "c3"), required=True)
    execution_path.add_argument("--shape-selection", required=True)
    execution_path.add_argument("--output", required=True)
    shape_campaign = sub.add_parser("build-configuration-shape-campaign")
    shape_campaign.add_argument("--stage", choices=("c2", "c3"), required=True)
    shape_campaign.add_argument("--study-plan", required=True)
    shape_campaign.add_argument("--compiler-input", required=True)
    shape_campaign.add_argument("--release-hash", required=True)
    shape_campaign.add_argument("--execution-path", action="append", required=True)
    shape_campaign.add_argument(
        "--max-cost-per-attempt-usd", type=Decimal, required=True
    )
    shape_campaign.add_argument("--reserve-usd", type=Decimal, required=True)
    shape_campaign.add_argument("--output", required=True)
    shape_bundle = sub.add_parser("build-configuration-shape-probe-bundle")
    shape_bundle.add_argument("--stage", choices=("c2", "c3"), required=True)
    shape_bundle.add_argument(
        "--profile-id",
        choices=("m_wide", "m_prior_deep", "l_wide", "l_prior_deep"),
        required=True,
    )
    shape_bundle.add_argument("--study-plan", required=True)
    shape_bundle.add_argument("--compiler-input", required=True)
    shape_bundle.add_argument("--campaign-plan", required=True)
    shape_bundle.add_argument("--execution-path", action="append", required=True)
    shape_bundle.add_argument("--output", required=True)
    shape_probe = sub.add_parser("run-configuration-shape-probe")
    for name in (
        "bundle",
        "campaign-root",
        "campaign-wave-id",
        "campaign-attempt-id",
        "pack-root",
        "manifest",
        "train-view",
        "global-plan",
        "placement-receipt",
        "release-receipt",
        "environment-receipt",
        "capability-receipt",
        "output",
    ):
        shape_probe.add_argument(f"--{name}", required=True)
    shape_probe.add_argument(
        "--max-sequences", type=int, choices=(16, 20, 24), required=True
    )
    shape_probe.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    shape_selection = sub.add_parser("select-configuration-shape-plan")
    shape_selection.add_argument("--stage", choices=("c2", "c3"), required=True)
    shape_selection.add_argument("--result", action="append", required=True)
    shape_selection.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compile-base":
        from twc.pretraining.base_recipe_compiler import (
            compile_base_pretraining_root,
        )

        config = compile_base_pretraining_root(
            run_root=Path(args.run),
            run_id=args.run_id,
            model_config_path=Path(args.model_config),
            training_config_path=Path(args.training_config),
            pack_manifest_path=Path(args.pack_manifest),
            tokenizer_manifest_path=Path(args.tokenizer_manifest),
            train_view_path=Path(args.train_view),
            validation_view_path=Path(args.validation_view),
            global_plan_path=Path(args.global_plan),
            semantic_code_receipt_path=Path(args.semantic_code_receipt),
            evaluation_manifest_path=Path(args.evaluation_manifest),
            evaluation_implementation_path=Path(
                args.evaluation_implementation
            ),
        )
        _write_result(
            {
                "run_id": config["run_id"],
                "configuration_hash": config["configuration_hash"],
                "run": args.run,
            }
        )
        return 0
    if args.command == "check":
        run = load_run_root(Path(args.run))
        _write_result(
            {
                "run_id": run["run_id"],
                "configuration_hash": run["configuration_hash"],
                "status": "ok",
            }
        )
        return 0
    if args.command == "build-configuration-shape-execution-path":
        receipt = _build_configuration_shape_execution_path_from_artifacts(
            stage=args.stage,
            profile_id=args.profile_id,
            max_sequences=args.max_sequences,
            study_payload=_read_json(args.study_plan),
            compiler_input_payload=_read_json(args.compiler_input),
            release_payload=_read_json(args.release_receipt),
            environment_payload=_read_json(args.environment_receipt),
            placement_payload=_read_json(args.placement_receipt),
            capability_payload=_read_json(args.capability_receipt),
        )
        _write_artifact(Path(args.output), receipt.to_dict())
        _write_result(
            {"execution_path_hash": receipt.identity, "output": args.output}
        )
        return 0
    if args.command == "build-configuration-execution-path":
        receipt = build_configuration_execution_path_receipt(
            stage=args.stage,
            shape_selection=_read_json(args.shape_selection),
        )
        _write_artifact(Path(args.output), receipt.to_dict())
        _write_result(
            {"execution_path_hash": receipt.identity, "output": args.output}
        )
        return 0
    if args.command == "build-configuration-shape-campaign":
        payloads = tuple(_read_json(path) for path in args.execution_path)
        matrices = _configuration_shape_execution_matrices(
            stage=args.stage, payloads=payloads
        )
        plan = build_configuration_shape_probe_campaign(
            stage=args.stage,
            study_plan=_read_json(args.study_plan),
            compiler_input=_read_json(args.compiler_input),
            release_hash=args.release_hash,
            execution_matrices=matrices,
            max_cost_per_attempt_usd=args.max_cost_per_attempt_usd,
            reserve_usd=args.reserve_usd,
        )
        artifact = campaign_plan_artifact(plan)
        _write_artifact(Path(args.output), artifact)
        _write_result(
            {"campaign_hash": artifact["campaign_hash"], "output": args.output}
        )
        return 0
    if args.command == "build-configuration-shape-probe-bundle":
        if len(args.execution_path) != 3:
            raise ValueError("shape bundle requires exactly three execution paths")
        matrix = build_configuration_shape_probe_execution_matrix(
            paths=tuple(_read_json(path) for path in args.execution_path)
        )
        bundle = build_configuration_shape_probe_bundle(
            stage=args.stage,
            profile_id=args.profile_id,
            study_plan=_read_json(args.study_plan),
            compiler_input=_read_json(args.compiler_input),
            campaign_plan=load_campaign_plan(Path(args.campaign_plan)),
            execution_matrix=matrix,
        )
        artifact = bundle.to_dict()
        _write_artifact(Path(args.output), artifact)
        _write_result(
            {"bundle_hash": artifact["bundle_hash"], "output": args.output}
        )
        return 0
    if args.command == "run-configuration-shape-probe":
        output = Path(args.output)
        _reserve_artifact_output(output)
        bundle_payload = _read_json(args.bundle)
        with _active_shape_campaign(
            campaign_root=Path(args.campaign_root),
            bundle_payload=bundle_payload,
            wave_id=args.campaign_wave_id,
            attempt_id=args.campaign_attempt_id,
        ) as campaign_plan:
            result = run_configuration_shape_probe_point(
                bundle_payload=bundle_payload,
                campaign_plan=campaign_plan,
                pack_root=Path(args.pack_root),
                manifest_path=Path(args.manifest),
                train_view_path=Path(args.train_view),
                global_plan_path=Path(args.global_plan),
                placement_receipt=_read_json(args.placement_receipt),
                release_receipt=_read_json(args.release_receipt),
                environment_receipt=_read_json(args.environment_receipt),
                capability_receipt=_read_json(args.capability_receipt),
                max_sequences=args.max_sequences,
                device=args.device,
                progress_callback=lambda payload: print(
                    json.dumps({**payload, "campaign_wave_id": args.campaign_wave_id, "campaign_attempt_id": args.campaign_attempt_id, "max_sequences": args.max_sequences}, sort_keys=True, separators=(",", ":")),
                    file=sys.stderr,
                    flush=True,
                ),
            )
            _write_reserved_artifact(output, result)
        _write_result({"receipt_hash": result["receipt_hash"], "output": args.output})
        return 0 if result.get("status") == "success" else 2
    if args.command == "select-configuration-shape-plan":
        if len(args.result) != 6:
            raise ValueError("shape selection requires exactly six result files")
        result = build_configuration_shape_selection(
            stage=args.stage,
            results=tuple(_read_json(path) for path in args.result),
        )
        _write_artifact(Path(args.output), result)
        _write_result({"selection_hash": result["selection_hash"], "output": args.output})
        return 0
    if args.command == "repair-checkpoints":
        manager = CheckpointManager(
            Path(args.checkpoints_root),
            persistent_root=Path(args.persistent_root) if args.persistent_root else None,
            topology=args.topology,
            create=False,
        )
        _write_result(manager.repair())
        return 0
    if args.command == "build-release":
        _write_result(build_release(Path(args.source_root), Path(args.release_root)))
        return 0
    if args.command == "verify-bootstrap":
        _write_result(
            verify_offline_bootstrap(
                release_receipt=_read_json(args.release_receipt),
                python_executable=Path(args.python),
                expected_python=(args.python_major, args.python_minor),
            )
        )
        return 0
    if args.command == "capture-environment":
        receipt = build_environment_receipt(
            device=args.device,
            precision=args.precision,
            attention_backend=args.attention_backend,
            optimizer_backend=args.optimizer_backend,
            compile_mode=args.compile_mode,
            allow_tf32=args.allow_tf32,
        )
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("environment receipt output already exists")
        output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        _write_result({"environment_hash": receipt["environment_hash"], "status": receipt["status"], "output": str(output)})
        return 0
    if args.command == "prepare-worker-data":
        placement = _read_json(args.placement_receipt)
        artifacts = load_sequence_training_artifacts(
            pack_root=Path(args.pack_root),
            manifest_path=Path(args.manifest),
            view_path=Path(args.view),
            global_plan_path=Path(args.global_plan),
            placement_receipt=placement,
        )
        semantic = global_plan_for_sequence_artifacts(artifacts)
        validation = load_sequence_evaluation_artifacts(
            pack_root=Path(args.pack_root), manifest_path=Path(args.manifest),
            view_path=Path(args.validation_view), placement_receipt=placement,
        )
        device = build_device_execution_plan(
            semantic,
            artifacts.loss_token_lengths,
            max_loss_tokens=args.max_loss_tokens,
            max_attention_cells=args.max_attention_cells,
            max_sequences=args.max_sequences,
        )
        records_payload = artifacts.records_artifact(tokenizer_hash=args.tokenizer_hash)
        validation_payload = validation.records_artifact(tokenizer_hash=args.tokenizer_hash)
        device_payload = device_execution_plan_artifact(device)
        for output, payload in (
            (Path(args.records_output), records_payload),
            (Path(args.validation_records_output), validation_payload),
            (Path(args.device_plan_output), device_payload),
        ):
            if output.exists():
                raise FileExistsError(f"worker-data output already exists: {output}")
            output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        _write_result({"records_hash": records_payload["records_hash"], "validation_records_hash": validation_payload["records_hash"], "device_execution_plan_hash": device.plan_hash})
        return 0
    if args.command == "verify-data-placement":
        receipt = build_verified_placement_receipt(
            pack_root=Path(args.pack_root), manifest_path=Path(args.manifest)
        )
        _write_artifact(Path(args.output), receipt)
        _write_result({"receipt_hash": receipt["receipt_hash"], "output": args.output})
        return 0
    if args.command == "validate-storage":
        if args.checkpoint_topology == "local_only":
            if args.local_root is None or args.local_sentinel is None or args.durable_root is not None:
                raise ValueError("local_only storage requires only --local-root")
        elif args.checkpoint_topology == "direct_persistent":
            if args.durable_root is None or args.durable_sentinel is None or args.local_root is not None:
                raise ValueError("direct_persistent storage requires only --durable-root")
        elif (
            args.local_root is None or args.local_sentinel is None
            or args.durable_root is None or args.durable_sentinel is None
        ):
            raise ValueError("local_then_replicate storage requires local and durable roots")
        role = lambda path, sentinel, require_mount: None if path is None else StorageRole(Path(path), require_mount=require_mount, sentinel=sentinel)
        layout = StorageLayout(
            control=StorageRole(Path(args.control_root), require_mount=False),
            data=role(args.data_root, args.data_sentinel, args.data_require_mount),
            durable_checkpoints=role(args.durable_root, args.durable_sentinel, args.durable_require_mount),
            local_checkpoints=role(args.local_root, args.local_sentinel, args.local_require_mount),
            copyback=role(args.copyback_root, args.copyback_sentinel, args.copyback_require_mount),
        )
        receipt = layout.validate(required_checkpoint_bytes=args.required_checkpoint_bytes, reserve_bytes=args.reserve_bytes)
        _write_artifact(Path(args.output), receipt)
        _write_result({"storage_layout_hash": receipt["receipt_hash"], "output": args.output})
        return 0
    if args.command == "verify-backup":
        if args.topology == "local_then_replicate" and not args.persistent_root:
            raise ValueError("local_then_replicate backup verification requires --persistent-root")
        manager = CheckpointManager(
            Path(args.checkpoints_root),
            persistent_root=Path(args.persistent_root) if args.persistent_root else None,
            topology=args.topology,
            create=False,
        )
        receipt = build_backup_receipt(
            checkpoint_manager=manager,
            generation=args.generation,
            destination_root=Path(args.destination_root),
        )
        output = Path(args.output)
        output.write_text(json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        _write_result({"receipt_hash": receipt.receipt_hash, "output": str(output)})
        return 0
    if args.command == "fetch":
        operator, run_root, _context = _public_run_context(
            Path(args.run), load_desired_execution=False
        )
        state = operator.status()
        if state.get("state") == "COMPLETED":
            state = _mutate(operator, operator.recover)
        _finalize_public_result(operator=operator, run_root=run_root, state=state)
        _write_result(fetch_run(run_root, Path(args.destination), state))
        return 0
    operator = _operator(args)
    if args.command == "start":
        operator, run_root, context = _public_run_context(Path(args.run))
        root = operator.root
        if not 1 <= args.startup_timeout_seconds <= 3600:
            raise ValueError("startup timeout must be between 1 and 3600 seconds")
        worker_spec = _initialize_public_run(operator, run_root, context)
        operator.claim_writer()
        try:
            existing = operator.status()
            command_id = stable_json_hash(
                {"action": "start", "attempt_hash": existing["attempt_hash"]}
            )[:32]
            reserved = operator.reserve_start(command_id=command_id, mode="run")
            try:
                _validate_static_launch_inputs(
                    worker_spec=worker_spec,
                    mode="run",
                    startup_timeout_seconds=args.startup_timeout_seconds,
                )
                release_root, launch_environment = _release_launch_context(worker_spec)
            except Exception:
                operator.fail_start(
                    command_id=command_id,
                    failure_code="launcher_preflight_failed",
                    child_role="launcher",
                )
                raise
            _require_reserved_launch(
                operator, command_id=command_id, mode="run", kind="start",
                attempt_hash=str(existing["attempt_hash"]),
                fencing_epoch=int(reserved["fencing_epoch"]),
            )
            result = _launch_reserved_runtime(
                operator=operator, root=root, worker_spec=worker_spec,
                release_root=release_root, environment=launch_environment,
                command_id=command_id, command_kind="start", mode="run",
                startup_timeout_seconds=args.startup_timeout_seconds,
            )
        finally:
            operator.release_writer()
    elif args.command == "status":
        operator, run_root, context = _public_run_context(
            Path(args.run), load_desired_execution=False
        )
        result = operator.status()
        if result.get("state") == "COMPLETED" or (
            result.get("state") in {"RUNNING", "QUIESCING"}
            and observe_process(result["process_identity"]) == "exited"
            and observe_process(result["watchdog_process_identity"]) == "exited"
        ):
            result = _mutate(operator, operator.recover)
        _finalize_public_result(operator=operator, run_root=run_root, state=result)
    elif args.command == "resume":
        operator, run_root, context = _public_run_context(Path(args.run))
        if not 1 <= args.startup_timeout_seconds <= 3600:
            raise ValueError("startup timeout must be between 1 and 3600 seconds")
        existing = operator.status()
        if (
            existing.get("state") != "STOPPED"
            or existing.get("checkpoint_durability") != "persistent_verified"
        ):
            raise ValueError("resume requires a persistent verified stopped checkpoint")
        selected_generation = int(existing["checkpoint_generation"])
        selected = operator.checkpoint_manager.validate_reference(
            CheckpointReference(
                selected_generation,
                operator.checkpoint_manager.root
                / f"generation-{selected_generation:08d}",
                "persistent_verified",
            )
        )
        desired_checkpoint_root = Path(
            str(context["execution"]["checkpoint_root"])
        ).expanduser().resolve(strict=True)
        selected = CheckpointManager(
            desired_checkpoint_root,
            topology=str(context["execution"]["checkpoint_topology"]),
            create=False,
        ).validate_reference(
            CheckpointReference(
                selected.generation,
                desired_checkpoint_root
                / f"generation-{selected.generation:08d}",
                selected.durability,
                selected.manifest_sha256,
                selected.state_sha256,
            )
        )
        runtime = run_root / "runtime"
        release = _read_json(str(runtime / "release.json"))
        environment = _read_json(str(runtime / "environment.json"))
        placement = _read_json(str(runtime / "placement.json"))
        storage = _read_json(str(runtime / "storage.json"))
        device_plan = _read_json(str(runtime / "device-plan.json"))
        contract = training_config_contract(context["config"])
        release_semantics_verified = (
            build_configuration_semantic_code_receipt(
                Path(str(release["path"]))
            )["semantic_code_hash"]
            == contract.code_hash
        )
        if not release_semantics_verified:
            raise ValueError("resume release changes the training implementation")
        previous = _latest_execution_attempt(
            operator.root, expected_attempt_hash=existing["attempt_hash"]
        )
        successor = replace(
            previous,
            attempt_id=stable_json_hash(
                {
                    "parent_attempt_hash": previous.identity,
                    "execution": context["execution"],
                    "selected_checkpoint": {
                        "generation": selected.generation,
                        "manifest_sha256": selected.manifest_sha256,
                        "state_sha256": selected.state_sha256,
                    },
                }
            )[:32],
            parent_attempt_hash=previous.identity,
            device_execution_plan_hash=str(device_plan["plan_hash"]),
            release_hash=str(release["release_hash"]),
            environment_hash=str(environment["environment_hash"]),
            data_placement_hash=str(placement["receipt_hash"]),
            storage_layout_hash=str(storage["receipt_hash"]),
            pin_memory=bool(context["execution"]["pin_memory"]),
            telemetry_every_steps=int(
                context["execution"]["telemetry_every_steps"]
            ),
            checkpoint_every_loss_tokens=int(
                context["execution"]["checkpoint_every_loss_tokens"]
            ),
            checkpoint_max_interval_seconds=int(
                context["execution"]["checkpoint_max_interval_seconds"]
            ),
            durable_every_loss_tokens=int(
                context["execution"]["durable_every_loss_tokens"]
            ),
            durable_max_interval_seconds=int(
                context["execution"]["durable_max_interval_seconds"]
            ),
            pack_root=str(
                Path(str(context["execution"]["pack_root"])).expanduser().resolve(
                    strict=True
                )
            ),
            checkpoint_root=str(desired_checkpoint_root),
            checkpoint_topology=str(context["execution"]["checkpoint_topology"]),
            device=str(context["execution"]["device"]),
            read_mode=str(context["execution"]["read_mode"]),
            selected_checkpoint=SelectedCheckpoint(
                generation=selected.generation,
                durability=selected.durability,
                manifest_sha256=selected.manifest_sha256,
                state_sha256=selected.state_sha256,
                run_identity=selected.run_identity,
                attempt_hash=selected.attempt_hash,
                step=selected.step,
                consumed_loss_tokens=selected.consumed_loss_tokens,
                next_global_ordinal=selected.next_global_ordinal,
            ),
        )
        validate_resume_compatibility(
            previous,
            successor,
            release_semantics_verified=release_semantics_verified,
        )
        previous_spec = _read_json(
            str(operator.root / "worker-specs" / f"{previous.identity}.json")
        )
        successor_spec = {
            **previous_spec,
            "pack_root": successor.pack_root,
            "checkpoint_root": successor.checkpoint_root,
            "persistent_checkpoint_root": successor.persistent_checkpoint_root,
            "checkpoint_topology": successor.checkpoint_topology,
            "device": successor.device,
            "read_mode": successor.read_mode,
        }
        successor_spec.pop("spec_hash", None)
        successor_spec["spec_hash"] = stable_json_hash(successor_spec)
        worker_spec = (
            operator.root / "worker-specs" / f"{successor.identity}.json"
        )
        _write_artifact_or_verify(worker_spec, successor_spec)
        _validate_static_launch_inputs(
            worker_spec=worker_spec,
            mode="run",
            startup_timeout_seconds=args.startup_timeout_seconds,
        )
        command_id = stable_json_hash(
            {"action": "resume", "attempt_hash": successor.identity}
        )[:32]
        operator.claim_writer()
        try:
            reserved = operator.reserve_resume(
                command_id=command_id,
                attempt=successor,
                mode="run",
            )
            try:
                _validate_static_launch_inputs(
                    worker_spec=worker_spec,
                    mode="run",
                    startup_timeout_seconds=args.startup_timeout_seconds,
                )
                release_root, launch_environment = _release_launch_context(worker_spec)
            except Exception:
                operator.fail_start(
                    command_id=command_id,
                    failure_code="launcher_preflight_failed",
                    child_role="launcher",
                )
                raise
            _require_reserved_launch(
                operator,
                command_id=command_id,
                mode="run",
                kind="resume",
                attempt_hash=successor.identity,
                fencing_epoch=int(reserved["fencing_epoch"]),
            )
            result = _launch_reserved_runtime(
                operator=operator,
                root=operator.root,
                worker_spec=worker_spec,
                release_root=release_root,
                environment=launch_environment,
                command_id=command_id,
                command_kind="resume",
                mode="run",
                startup_timeout_seconds=args.startup_timeout_seconds,
            )
            result = {
                **result,
                "resumed_from_checkpoint_generation": selected_generation,
            }
        finally:
            operator.release_writer()
    elif args.command == "request":
        result = operator.request_action(
            args.request_id, action=args.action, deadline_seconds=args.deadline_seconds
        )
    elif args.command == "orphan":
        result = operator.mark_orphaned(reason=f"operator command {args.command_id}")
    elif args.command == "takeover":
        proof = _read_json(args.former_host_termination_receipt)
        proof_unhashed = {key: value for key, value in proof.items() if key != "receipt_hash"}
        if (
            proof.get("artifact_kind") != "twc_former_host_termination_receipt_v1"
            or proof.get("terminated") is not True
            or proof.get("run_identity") != operator.status()["run_identity"]
            or proof.get("receipt_hash") != stable_json_hash(proof_unhashed)
        ):
            raise ValueError("former-host termination proof is invalid")
        result = operator.takeover(command_id=args.command_id, former_host_terminated=True)
    elif args.command == "checkpoint":
        observed_state = operator.status()
        deadline = time.monotonic() + args.timeout_seconds
        with _checkpoint_writer(operator, deadline_monotonic=deadline):
            state = operator.status()
            if (
                state["state"] != "RUNNING"
                or observe_process(state["process_identity"]) != "confirmed_alive"
                or state["run_identity"] != observed_state["run_identity"]
                or state["attempt_hash"] != observed_state["attempt_hash"]
                or state["fencing_epoch"] != observed_state["fencing_epoch"]
            ):
                raise RuntimeError("checkpoint requires the exact live training worker")
            request_id = _checkpoint_request_id(operator, observed_state)
            request_deadline_seconds = max(1, int(args.timeout_seconds))
            request_path = operator.root / "requests" / f"{request_id}.json"
            if request_path.is_file():
                request_deadline_seconds = int(
                    _read_json(str(request_path))["deadline_seconds"]
                )
            operator.request_action(
                request_id,
                action="checkpoint",
                deadline_seconds=request_deadline_seconds,
            )
        ack_path = operator.root / "request-acks" / f"{request_id}.json"
        while time.monotonic() < deadline and not ack_path.is_file():
            time.sleep(0.05)
        if not ack_path.is_file():
            raise TimeoutError("training worker did not acknowledge checkpoint request")
        ack = _read_json(str(ack_path))
        if ack.get("ack_hash") != stable_json_hash(
            {key: value for key, value in ack.items() if key != "ack_hash"}
        ):
            raise ValueError("checkpoint acknowledgement hash mismatch")
        reference = CheckpointReference(int(ack["generation"]), Path(ack["path"]), str(ack["durability"]))
        with _checkpoint_writer(operator, deadline_monotonic=deadline):
            result = operator.checkpoint(
                step=int(ack["step"]), next_global_ordinal=int(ack["next_global_ordinal"]), checkpoint=reference
            )
    elif args.command in {"pause", "stop"}:
        state = operator.status()
        if state["state"] != "RUNNING" or observe_process(state["process_identity"]) != "confirmed_alive":
            raise RuntimeError(f"{args.command} requires the exact live training worker")
        operator.request_action(
            uuid.uuid4().hex,
            action=args.command,
            deadline_seconds=max(1, int(args.timeout_seconds)),
        )
        deadline = time.monotonic() + args.timeout_seconds
        identities = (
            state["process_identity"],
            state["watchdog_process_identity"],
        )
        while time.monotonic() < deadline and any(
            observe_process(identity) == "confirmed_alive" for identity in identities
        ):
            time.sleep(0.05)
        if any(observe_process(identity) != "exited" for identity in identities):
            raise TimeoutError(f"training processes did not complete {args.command}")
        result = _mutate(operator, operator.recover)
    elif args.command == "teardown-readiness":
        result = operator.teardown_readiness(args.scope)
    elif args.command == "record-backup":
        receipt = BackupReceipt.from_dict(_read_json(args.receipt))
        result = _mutate(
            operator,
            lambda: operator.record_off_resource_backup(receipt, destination_root=Path(args.destination_root)),
        )
    else:  # pragma: no cover - argparse owns command exhaustiveness
        raise AssertionError(args.command)
    _write_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
