"""R7 single-device worker driven by immutable semantic and physical plans."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import stat
import statistics
import subprocess
import sys
from threading import Event
from threading import Thread
import time
from typing import Any, Callable, Iterator, Mapping

import torch
import twc

from twc.pretraining.configuration_semantic_code import (
    build_configuration_semantic_code_receipt,
)
from twc.pretraining.configuration_performance import (
    ConfigurationPerformanceWindowAccumulator,
    load_configuration_performance_policy,
)
from twc.pretraining.configuration_study import configuration_evaluation_due
from twc.pretraining.configuration_study_execution import (
    ConfigurationExecutionPathReceiptV1,
    ExperimentTrainingContractV2,
    load_configuration_execution_path_receipt,
    validate_configuration_execution_path_for_training,
)
from twc.pretraining.training.checkpoints import CheckpointManager, CheckpointPayload, CheckpointReference
from twc.pretraining.training.contracts import ExecutionAttempt, load_execution_attempt, load_worker_contract
from twc.pretraining.training.data import BatchSource, SequenceRecord
from twc.pretraining.training.engine import EngineConfig, TrainingEngine
from twc.pretraining.training.model import DecoderConfig, PackedCausalBatch, estimate_parameter_count, initialize_decoder
from twc.pretraining.training.deployment import build_environment_receipt, verify_release_receipt
from twc.pretraining.gpu_campaign_deployment import (
    capture_live_gpu_capability_receipt,
    validate_gpu_capability_receipt,
    verify_thin_runtime_release,
)
from twc.pretraining.training.operator import TrainingOperator
from twc.pretraining.training.storage import checkpoint_filesystem_requirements
from twc.pretraining.training.startup import (
    activation_state,
    build_worker_ready,
    open_startup_namespace,
    publish_ready_once,
)
from twc.pretraining.training.planning import DeviceExecutionPlan, DeviceSlice, GlobalStepPlan, PlannedWork
from twc.repro.hashing import stable_json_hash
from twc.runtime.processes import observe_process, process_start_token


def _checkpoint_filesystem_observations(spec: "WorkerSpec") -> list[dict[str, int]]:
    receipt = _json(spec.storage_layout_receipt_path)
    requirements = checkpoint_filesystem_requirements(receipt)
    checkpoint_roles = {"local_checkpoints", "durable_checkpoints"}
    roots_by_device: dict[int, Path] = {}
    for row in receipt["roles"]:
        if row["role"] in checkpoint_roles:
            roots_by_device.setdefault(
                int(row["device_id"]), Path(str(row["resolved_root"]))
            )
    observations = []
    for row in requirements:
        device = int(row["device_id"])
        try:
            root = roots_by_device[device]
        except KeyError as error:
            raise ValueError("checkpoint filesystem lacks a live root") from error
        stats = os.statvfs(root)
        observations.append(
            {"device_id": device, "free_bytes": stats.f_bavail * stats.f_frsize}
        )
    return observations


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    contract_path: Path
    model_path: Path
    records_path: Path
    validation_records_path: Path
    pack_root: Path
    global_step_plan_path: Path
    device_execution_plan_path: Path
    release_receipt_path: Path
    environment_receipt_path: Path
    data_placement_receipt_path: Path
    storage_layout_receipt_path: Path
    checkpoint_root: Path
    persistent_checkpoint_root: Path | None
    checkpoint_topology: str
    device: str
    read_mode: str
    removable_data: bool
    configuration_execution_path_receipt_path: Path | None = None
    gpu_capability_receipt_path: Path | None = None


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _real_storage_directory(path: Path, *, label: str) -> tuple[Path, os.stat_result]:
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be an absolute unaliased path")
    current = Path(path.anchor)
    fingerprints: list[tuple[Path, os.stat_result]] = []
    for part in path.parts[1:]:
        current /= part
        try:
            observed = os.lstat(current)
        except OSError as error:
            raise ValueError(f"{label} does not exist: {current}") from error
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"{label} contains a symlink component: {current}")
        fingerprints.append((current, observed))
    resolved = path.resolve(strict=True)
    try:
        unchanged = all(
            os.path.samestat(before, after := os.lstat(current))
            and before.st_mode == after.st_mode
            for current, before in fingerprints
        )
    except OSError as error:
        raise ValueError(f"{label} changed during validation") from error
    if not unchanged or not resolved.is_dir():
        raise ValueError(f"{label} must be a real directory")
    return resolved, fingerprints[-1][1] if fingerprints else os.lstat(path)


def _storage_sentinel(
    *, root: Path, root_stat: os.stat_result, role: str
) -> Mapping[str, Any]:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise ValueError("worker storage validation lacks required no-follow primitives")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if not os.path.samestat(root_stat, os.fstat(directory)):
            raise ValueError(f"worker {role} storage root changed during validation")
        sentinel = os.open(
            ".twc-storage.json", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory
        )
        with os.fdopen(sentinel, "r", encoding="utf-8") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError(f"worker {role} sentinel is not a real file")
            payload = json.load(handle)
    finally:
        os.close(directory)
    if not isinstance(payload, dict):
        raise ValueError(f"worker {role} sentinel must be a JSON object")
    return payload


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def validate_worker_storage_bindings(
    *,
    pack_root: Path,
    placement_receipt: Mapping[str, Any],
    storage_layout_receipt: Mapping[str, Any],
    checkpoint_root: Path,
    persistent_checkpoint_root: Path | None,
    checkpoint_topology: str,
) -> None:
    """Validate exact live roots without performing full pack-file readback."""

    receipts = (
        (placement_receipt, "twc_data_placement_receipt_v1"),
        (storage_layout_receipt, "twc_storage_layout_receipt_v1"),
    )
    for receipt, kind in receipts:
        if receipt.get("artifact_kind") != kind or receipt.get("schema_version") != 1:
            raise ValueError("worker storage receipt schema is invalid")
        unhashed = {key: value for key, value in receipt.items() if key != "receipt_hash"}
        if receipt.get("receipt_hash") != stable_json_hash(unhashed):
            raise ValueError("worker storage receipt hash is invalid")
    placement_fields = {
        "artifact_kind", "schema_version", "pack_manifest_hash",
        "verified_file_count", "full_destination_readback", "destination_root",
        "destination_device_id", "file_identities", "raw_text_serialized",
        "raw_url_serialized", "token_serialized", "receipt_hash",
    }
    identity_fields = {
        "relative_path", "device_id", "inode", "byte_count", "mtime_ns", "sha256"
    }
    if set(placement_receipt) != placement_fields:
        raise ValueError("worker placement receipt fields are not canonical")

    pack, pack_stat = _real_storage_directory(pack_root, label="worker pack root")
    checkpoint, _ = _real_storage_directory(checkpoint_root, label="worker checkpoint root")
    placement, _ = _real_storage_directory(
        Path(str(placement_receipt.get("destination_root"))),
        label="worker placement root",
    )
    device = pack_stat.st_dev
    if placement != pack:
        raise ValueError("worker placement does not bind the exact data root")
    if placement_receipt.get("destination_device_id") != device:
        raise ValueError("worker placement data device differs from the pack root")
    identities = placement_receipt.get("file_identities")
    if (
        placement_receipt.get("full_destination_readback") is not True
        or not isinstance(identities, list)
        or not identities
        or placement_receipt.get("verified_file_count") != len(identities)
        or not _sha256(placement_receipt.get("pack_manifest_hash"))
        or any(
            not isinstance(row, dict)
            or set(row) != identity_fields
            or not isinstance(row.get("relative_path"), str)
            or not row["relative_path"]
            or PurePosixPath(row["relative_path"]).is_absolute()
            or PurePosixPath(row["relative_path"]).as_posix() != row["relative_path"]
            or any(part in {"", ".", ".."} for part in PurePosixPath(row["relative_path"]).parts)
            or row.get("device_id") != device
            or type(row.get("inode")) is not int or row["inode"] <= 0
            or type(row.get("byte_count")) is not int or row["byte_count"] < 0
            or type(row.get("mtime_ns")) is not int or row["mtime_ns"] < 0
            or not _sha256(row.get("sha256"))
            for row in identities
        )
        or any(placement_receipt.get(field) is not False for field in (
            "raw_text_serialized", "raw_url_serialized", "token_serialized"
        ))
    ):
        raise ValueError("worker placement receipt lacks canonical safe readback evidence")

    rows = storage_layout_receipt.get("roles")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("worker storage roles are invalid")
    roles: dict[str, list[tuple[dict[str, Any], Path]]] = {}
    for row in rows:
        role = row.get("role")
        if role not in {"control", "data", "durable_checkpoints", "local_checkpoints", "copyback"}:
            raise ValueError("worker storage role name is invalid")
        root, root_stat = _real_storage_directory(
            Path(str(row.get("resolved_root"))), label=f"worker {role} storage root"
        )
        if row.get("device_id") != root_stat.st_dev:
            raise ValueError(f"worker {role} storage device is invalid")
        sentinel = row.get("sentinel")
        sentinel_hash = row.get("sentinel_hash")
        if sentinel is None:
            if sentinel_hash is not None:
                raise ValueError(f"worker {role} sentinel hash is unexpected")
        else:
            try:
                payload = _storage_sentinel(root=root, root_stat=root_stat, role=role)
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"worker {role} sentinel is missing or malformed") from error
            if (
                not isinstance(payload, dict)
                or payload.get("storage_id") != sentinel
                or payload.get("role") not in {None, role}
                or stable_json_hash(payload) != sentinel_hash
            ):
                raise ValueError(f"worker {role} sentinel identity changed")
        roles.setdefault(role, []).append((row, root))

    if any(len(values) != 1 for values in roles.values()) or len(roles.get("control", [])) != 1:
        raise ValueError("worker storage layout has missing or duplicate canonical roles")

    data_rows = roles.get("data", [])
    if len(data_rows) != 1:
        raise ValueError("worker storage layout requires exactly one data role")
    data_row, data_root = data_rows[0]
    if data_root != pack or data_row.get("device_id") != device or not isinstance(
        data_row.get("sentinel"), str
    ):
        raise ValueError("worker storage layout does not bind the exact data root/device")
    expected = {"local_only": ("local_checkpoints", False),
                "direct_persistent": ("durable_checkpoints", False),
                "local_then_replicate": ("local_checkpoints", True)}.get(checkpoint_topology)
    if expected is None:
        raise ValueError("worker checkpoint topology is invalid")
    checkpoint_role, requires_persistent = expected
    checkpoint_rows = roles.get(checkpoint_role, [])
    if (
        len(checkpoint_rows) != 1
        or checkpoint_rows[0][1] != checkpoint
        or not isinstance(checkpoint_rows[0][0].get("sentinel"), str)
    ):
        raise ValueError("worker checkpoint topology does not match storage role evidence")
    durable_rows = roles.get("durable_checkpoints", [])
    local_rows = roles.get("local_checkpoints", [])
    if requires_persistent:
        if persistent_checkpoint_root is None or len(durable_rows) != 1:
            raise ValueError("worker checkpoint topology requires one persistent root")
        persistent = _real_storage_directory(persistent_checkpoint_root, label="worker persistent checkpoint root")
        if durable_rows[0][1] != persistent[0] or not isinstance(
            durable_rows[0][0].get("sentinel"), str
        ):
            raise ValueError("worker persistent root does not match storage role evidence")
    elif persistent_checkpoint_root is not None or (
        checkpoint_topology == "local_only" and durable_rows
    ) or (checkpoint_topology == "direct_persistent" and local_rows):
        raise ValueError("worker checkpoint topology has contradictory storage roles")


def load_worker_spec(path: Path) -> WorkerSpec:
    payload = _json(path)
    base_expected = {
        "artifact_kind", "schema_version", "contract_path", "model_path", "records_path", "validation_records_path", "pack_root",
        "global_step_plan_path", "device_execution_plan_path", "release_receipt_path",
        "environment_receipt_path", "data_placement_receipt_path", "storage_layout_receipt_path",
        "checkpoint_root", "persistent_checkpoint_root", "checkpoint_topology", "device", "read_mode",
        "removable_data", "spec_hash",
    }
    physical_fields = {
        "configuration_execution_path_receipt_path",
        "gpu_capability_receipt_path",
    }
    version = (payload.get("artifact_kind"), payload.get("schema_version"))
    if version == ("twc_pretraining_worker_spec_v2", 2):
        expected = base_expected
    elif version == ("twc_pretraining_worker_spec_v3", 3):
        expected = base_expected | physical_fields
    else:
        raise ValueError("unsupported worker spec; R6 specs must be rebuilt")
    if set(payload) != expected:
        raise ValueError(f"worker spec fields invalid; unknown={sorted(set(payload)-expected)} missing={sorted(expected-set(payload))}")
    unhashed = {key: value for key, value in payload.items() if key != "spec_hash"}
    if payload["spec_hash"] != stable_json_hash(unhashed):
        raise ValueError("worker spec hash mismatch")
    if payload["checkpoint_topology"] not in {"local_only", "direct_persistent", "local_then_replicate"}:
        raise ValueError("worker checkpoint topology is invalid")
    if payload["checkpoint_topology"] == "local_then_replicate":
        if payload["persistent_checkpoint_root"] is None:
            raise ValueError("replicated worker checkpoints require a persistent root")
    elif payload["persistent_checkpoint_root"] is not None:
        raise ValueError("only replicated worker checkpoints accept a persistent root")
    if payload["read_mode"] not in {"local_mmap", "buffered_pread"}:
        raise ValueError("worker read mode is invalid")
    if type(payload["removable_data"]) is not bool:
        raise ValueError("worker removable_data must be a boolean")
    root = path.resolve().parent

    def artifact_path(value: object) -> Path:
        candidate = Path(str(value))
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if not resolved.is_file():
            raise ValueError(f"worker artifact is missing: {resolved}")
        return resolved

    def storage_path(value: object, *, optional: bool = False) -> Path | None:
        if value is None and optional:
            return None
        candidate = Path(str(value))
        raw = candidate if candidate.is_absolute() else root / candidate
        if not raw.exists():
            raise ValueError(f"worker storage root must already exist: {raw}")
        return raw

    return WorkerSpec(
        contract_path=artifact_path(payload["contract_path"]),
        model_path=artifact_path(payload["model_path"]),
        records_path=artifact_path(payload["records_path"]),
        validation_records_path=artifact_path(payload["validation_records_path"]),
        pack_root=storage_path(payload["pack_root"]),  # type: ignore[arg-type]
        global_step_plan_path=artifact_path(payload["global_step_plan_path"]),
        device_execution_plan_path=artifact_path(payload["device_execution_plan_path"]),
        release_receipt_path=artifact_path(payload["release_receipt_path"]),
        environment_receipt_path=artifact_path(payload["environment_receipt_path"]),
        data_placement_receipt_path=artifact_path(payload["data_placement_receipt_path"]),
        storage_layout_receipt_path=artifact_path(payload["storage_layout_receipt_path"]),
        checkpoint_root=storage_path(payload["checkpoint_root"]),  # type: ignore[arg-type]
        persistent_checkpoint_root=storage_path(payload["persistent_checkpoint_root"], optional=True),
        checkpoint_topology=str(payload["checkpoint_topology"]),
        device=str(payload["device"]),
        read_mode=str(payload["read_mode"]),
        removable_data=payload["removable_data"],
        configuration_execution_path_receipt_path=(
            artifact_path(payload["configuration_execution_path_receipt_path"])
            if version[1] == 3
            else None
        ),
        gpu_capability_receipt_path=(
            artifact_path(payload["gpu_capability_receipt_path"])
            if version[1] == 3
            else None
        ),
    )


def _load_attempts(operator_root: Path) -> tuple[ExecutionAttempt, ...]:
    try:
        rows = (operator_root / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, IndexError, json.JSONDecodeError) as error:
        raise ValueError("worker execution-attempt ledger is missing") from error
    if not rows:
        raise ValueError("worker execution-attempt ledger is empty")
    return tuple(load_execution_attempt(json.loads(row)) for row in rows)


def _load_model(path: Path, expected_hash: str) -> DecoderConfig:
    payload = _json(path)
    if payload.get("artifact_kind") != "twc_base_model_spec_v1" or payload.get("schema_version") != 1:
        raise ValueError("unsupported R7 model artifact")
    if payload.get("model_hash") != expected_hash or stable_json_hash(
        {key: value for key, value in payload.items() if key != "model_hash"}
    ) != expected_hash:
        raise ValueError("model artifact hash mismatch")
    names = {field.name for field in fields(DecoderConfig)}
    missing = names - set(payload)
    if missing:
        raise ValueError(f"model artifact is missing dimensions: {sorted(missing)}")
    return DecoderConfig(**{name: payload[name] for name in names})


def _load_records(
    path: Path,
    contract,
    *,
    pack_root: Path,
    placement_hash: str,
    split: str,
    expected_records_hash: str | None = None,
) -> tuple[Path, tuple[SequenceRecord, ...]]:
    payload = _json(path)
    if payload.get("artifact_kind") != "twc_batch_source_records_v1" or payload.get("schema_version") != 1:
        raise ValueError("unsupported BatchSource records artifact")
    expected_hash = stable_json_hash({key: value for key, value in payload.items() if key != "records_hash"})
    if payload.get("records_hash") != expected_hash:
        raise ValueError("BatchSource records hash mismatch")
    if (
        expected_records_hash is not None
        and payload.get("records_hash") != expected_records_hash
    ):
        raise ValueError(
            f"BatchSource {split} records do not match the execution attempt"
        )
    expected_view_hash = contract.view_hash if split == "train" else contract.validation_view_hash
    for field, expected in (
        ("data_manifest_hash", contract.data_manifest_hash),
        ("view_hash", expected_view_hash),
        ("tokenizer_hash", contract.tokenizer_hash),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"BatchSource records {field} binding mismatch")
    if payload.get("split") != split:
        raise ValueError(f"BatchSource records split must be {split}")
    if payload.get("placement_hash") != placement_hash:
        raise ValueError("BatchSource records placement binding mismatch")
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("BatchSource records are empty")
    records = tuple(
        SequenceRecord(
            relative_path=str(row["relative_path"]),
            token_offset_bytes=row["token_offset_bytes"],
            token_count=row["token_count"],
        )
        for row in rows
    )
    return pack_root, records


def _nvidia_smi_utilization(device: str) -> float:
    if not device.startswith("cuda:") or not device.removeprefix("cuda:").isdigit():
        raise RuntimeError("nvidia-smi sampler requires an indexed CUDA device")
    completed = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
            "-i", device.removeprefix("cuda:"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    lines = completed.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("nvidia-smi utilization output is ambiguous")
    return float(lines[0].strip())


class CudaUtilizationSampler:
    """Provider-neutral bounded sampler with explicit unavailable status."""

    def __init__(self, device: str, *, sample: Callable[[], float] | None = None, interval_seconds: float = 0.2) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self._sample = sample
        self._stop = Event()
        self._thread: Thread | None = None
        self._values: list[float] = []
        self._error: str | None = None

    def start(self) -> None:
        if not self.device.startswith("cuda"):
            return
        if self._sample is None:
            candidate = getattr(torch.cuda, "utilization", None)
            def sample() -> float:
                if callable(candidate):
                    try:
                        return float(candidate())
                    except (ImportError, ModuleNotFoundError):
                        pass
                return _nvidia_smi_utilization(self.device)
            self._sample = sample

        def run() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    value = float(self._sample())
                    if 0 <= value <= 100:
                        self._values.append(value)
                except (OSError, RuntimeError, ValueError, TypeError, subprocess.SubprocessError):
                    self._error = "sampling_failed"
                    return

        self._thread = Thread(target=run, name="twc-cuda-utilization", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def sample_count(self) -> int:
        return len(self._values)

    def evidence_since(self, sample_index: int) -> dict[str, Any]:
        if type(sample_index) is not int or not 0 <= sample_index <= len(self._values):
            raise ValueError("GPU utilization sample index is invalid")
        return self._evidence(self._values[sample_index:])

    def evidence(self) -> dict[str, Any]:
        return self._evidence(self._values)

    def _evidence(self, values: list[float]) -> dict[str, Any]:
        if not self.device.startswith("cuda"):
            return {"gpu_utilization_status": "not_requested", "median_gpu_utilization": None, "gpu_utilization_samples": 0}
        if not values:
            return {"gpu_utilization_status": self._error or "unavailable", "median_gpu_utilization": None, "gpu_utilization_samples": 0}
        return {
            "gpu_utilization_status": "measured",
            "median_gpu_utilization": statistics.median(values),
            "gpu_utilization_samples": len(values),
        }


def _terminal_utilization_evidence(
    last_training_telemetry: Mapping[str, Any],
    terminal_sampler_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep terminal checkpoint time out of the training-utilization metric."""

    fields = (
        "gpu_utilization_status",
        "median_gpu_utilization",
        "gpu_utilization_samples",
    )
    source = (
        last_training_telemetry
        if all(field in last_training_telemetry for field in fields)
        else terminal_sampler_evidence
    )
    return {field: source[field] for field in fields}


class CancellationSignal(Event):
    def __init__(self) -> None:
        super().__init__()
        self.requested_at_epoch: float | None = None

    def request(self) -> None:
        if self.requested_at_epoch is None:
            self.requested_at_epoch = time.time()
        self.set()


def _load_global_plan(path: Path) -> GlobalStepPlan:
    payload = _json(path)
    if payload.get("artifact_kind") != "twc_global_step_plan_v1" or payload.get("schema_version") != 1:
        raise ValueError("unsupported global-step plan")
    plan_hash = payload.get("plan_hash")
    if plan_hash != stable_json_hash({key: value for key, value in payload.items() if key != "plan_hash"}):
        raise ValueError("global-step plan hash mismatch")
    rows = payload["steps"]
    if rows and "first_segment_index" in rows[0]:
        steps = tuple(
            PlannedWork(
                sequence_start=row["first_segment_index"],
                sequence_stop=row["first_segment_index"] + row["sequence_count"],
                loss_tokens=row["loss_token_count"],
                attention_cells=row["attention_cells"],
            )
            for row in rows
        )
        sequence_count = sum(row["sequence_count"] for row in rows)
        total_loss_tokens = payload["loss_token_count"]
    else:
        steps = tuple(PlannedWork(**row) for row in rows)
        sequence_count = payload["sequence_count"]
        total_loss_tokens = payload["total_loss_tokens"]
    cursor = 0
    for step in steps:
        if step.sequence_start != cursor or step.sequence_stop <= step.sequence_start:
            raise ValueError("global-step plan is not contiguous")
        cursor = step.sequence_stop
    if (
        cursor != sequence_count
        or sum(step.loss_tokens for step in steps) != total_loss_tokens
    ):
        raise ValueError("global-step plan totals are inconsistent")
    return GlobalStepPlan(
        steps=steps,
        sequence_count=sequence_count,
        total_loss_tokens=total_loss_tokens,
        max_loss_tokens=payload["max_loss_tokens"],
        max_attention_cells=payload["max_attention_cells"],
        max_sequences=payload["max_sequences"],
        plan_hash=plan_hash,
    )


def _load_device_plan(path: Path) -> DeviceExecutionPlan:
    payload = _json(path)
    if payload.get("artifact_kind") != "twc_device_execution_plan_v1" or payload.get("schema_version") != 1:
        raise ValueError("unsupported device-execution plan")
    plan_hash = payload.get("plan_hash")
    if plan_hash != stable_json_hash({key: value for key, value in payload.items() if key != "plan_hash"}):
        raise ValueError("device-execution plan hash mismatch")
    slices = tuple(DeviceSlice(**row) for row in payload["slices"])
    if not slices:
        raise ValueError("device-execution plan has no slices")
    return DeviceExecutionPlan(
        global_plan_hash=payload["global_step_plan_hash"],
        slices=slices,
        max_loss_tokens=payload["max_loss_tokens"],
        max_attention_cells=payload["max_attention_cells"],
        max_sequences=payload["max_sequences"],
        plan_hash=plan_hash,
    )


def _validate_plan_contract(global_plan: GlobalStepPlan, device_plan: DeviceExecutionPlan, contract) -> None:
    """Reject semantically false plans before model/device initialization."""

    expected_caps = (
        contract.optimizer_step_token_cap,
        contract.optimizer_step_attention_cell_cap,
        contract.optimizer_step_sequence_cap,
    )
    if (global_plan.max_loss_tokens, global_plan.max_attention_cells, global_plan.max_sequences) != expected_caps:
        raise ValueError("global-step plan caps do not equal the run contract")
    if global_plan.total_loss_tokens != contract.target_loss_tokens:
        raise ValueError("global-step plan must cover exactly the contracted target loss tokens")
    device_caps = (device_plan.max_loss_tokens, device_plan.max_attention_cells, device_plan.max_sequences)
    if any(actual > maximum for actual, maximum in zip(device_caps, expected_caps, strict=True)):
        raise ValueError("device plan caps exceed the run contract")
    by_step: dict[int, list[DeviceSlice]] = {}
    for item in device_plan.slices:
        if not 0 <= item.global_step_index < len(global_plan.steps):
            raise ValueError("device slice references an unknown global step")
        if item.loss_tokens > device_plan.max_loss_tokens or item.attention_cells > device_plan.max_attention_cells or item.sequence_stop - item.sequence_start > device_plan.max_sequences:
            raise ValueError("device slice exceeds declared device caps")
        by_step.setdefault(item.global_step_index, []).append(item)
    for index, step in enumerate(global_plan.steps):
        slices = by_step.get(index, [])
        cursor = step.sequence_start
        for item in slices:
            if item.sequence_start != cursor or item.sequence_stop <= item.sequence_start or item.sequence_stop > step.sequence_stop:
                raise ValueError(f"device slices do not exactly partition global step {index}")
            cursor = item.sequence_stop
        if cursor != step.sequence_stop or sum(item.loss_tokens for item in slices) != step.loss_tokens or sum(item.attention_cells for item in slices) != step.attention_cells:
            raise ValueError(f"device slices do not exactly partition global step {index}")


def _evaluation_groups(records: tuple[SequenceRecord, ...], contract) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    start = 0
    while start < len(records):
        stop, tokens, cells = start, 0, 0
        while stop < len(records):
            length = records[stop].token_count - 1
            if length > contract.optimizer_step_token_cap or length * length > contract.optimizer_step_attention_cell_cap:
                raise ValueError("validation sequence exceeds evaluation caps")
            if stop > start and (
                tokens + length > contract.optimizer_step_token_cap
                or cells + length * length > contract.optimizer_step_attention_cell_cap
                or stop - start + 1 > contract.optimizer_step_sequence_cap
            ):
                break
            tokens += length
            cells += length * length
            stop += 1
        groups.append(tuple(range(start, stop)))
        start = stop
    return tuple(groups)


def _evaluation_component_group_hashes(
    records: tuple[SequenceRecord, ...],
    groups: tuple[tuple[int, ...], ...],
    *,
    validation_records_hash: str,
) -> tuple[str, ...]:
    """Bind every paired evaluation component to its exact ordered records."""

    if len(validation_records_hash) != 64 or any(
        character not in "0123456789abcdef" for character in validation_records_hash
    ):
        raise ValueError("validation records hash is invalid")
    identities: list[str] = []
    for group in groups:
        identities.append(
            stable_json_hash(
                {
                    "artifact_kind": "twc_validation_component_group_v1",
                    "schema_version": 1,
                    "validation_records_hash": validation_records_hash,
                    "ordinals": list(group),
                    "record_hashes": [
                        stable_json_hash(
                            {
                                "relative_path": records[ordinal].relative_path,
                                "token_offset_bytes": records[ordinal].token_offset_bytes,
                                "token_count": records[ordinal].token_count,
                            }
                        )
                        for ordinal in group
                    ],
                }
            )
        )
    if len(identities) != len(set(identities)):
        raise ValueError("validation component group identities must be unique")
    return tuple(identities)


def _identity(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field, payload.get("receipt_hash"))
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"runtime artifact {field} identity is invalid")
    return value


def _validate_release_code_binding(release: dict[str, Any], contract: object) -> None:
    """Prove v2 semantic bytes without conflating them with physical packaging."""

    binding = getattr(contract, "execution_binding", None)
    experiment = getattr(contract, "experiment", None)
    semantic_hash = getattr(experiment, "semantic_code_hash", None)
    if binding is None or semantic_hash is None:
        contract_code_hash = getattr(contract, "code_hash", None)
        if (
            release.get("artifact_kind") != "twc_gpu_runtime_release_v1"
            and release.get("code_hash") == contract_code_hash
        ):
            return
        try:
            observed = build_configuration_semantic_code_receipt(
                Path(str(release["path"]))
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("worker semantic source projection is invalid") from error
        if observed["semantic_code_hash"] != contract_code_hash:
            raise ValueError("worker semantic source projection hash mismatch")
        return
    if release.get("artifact_kind") != "twc_gpu_runtime_release_v1":
        raise ValueError("v2 configuration requires a thin physical release")
    if release.get("release_hash") != getattr(binding, "release_hash", None):
        raise ValueError("v2 worker physical release binding mismatch")
    try:
        observed = build_configuration_semantic_code_receipt(
            Path(str(release["path"]))
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "v2 worker semantic source projection is incomplete or invalid"
        ) from error
    if observed["semantic_code_hash"] != semantic_hash:
        raise ValueError("v2 worker semantic source projection hash mismatch")


def _validate_v2_attempt_cadence(contract: object, attempt: ExecutionAttempt) -> None:
    """Reject attempts whose configured interval changes v2 scientific cadence."""

    experiment = getattr(contract, "experiment", None)
    condition = getattr(experiment, "condition", None)
    if condition is None:
        return
    expected = getattr(condition, "evaluation_interval_loss_tokens", None)
    if expected is not None and attempt.evaluation_every_loss_tokens != expected:
        raise ValueError("v2 attempt evaluation cadence does not match its condition")


def _validate_v2_physical_checkpoint_cadence(
    contract: object, attempt: ExecutionAttempt
) -> None:
    """Keep operational checkpoint frequency outside scientific identity."""

    if getattr(getattr(contract, "experiment", None), "condition", None) is None:
        return
    if attempt.checkpoint_max_interval_seconds > 300:
        raise ValueError("v2 attempt local checkpoint interval exceeds 300 seconds")


def validate_configuration_worker_physical_binding(
    *,
    spec: Any,
    training_contract: ExperimentTrainingContractV2,
    execution_path: ConfigurationExecutionPathReceiptV1,
    attempt: ExecutionAttempt,
    global_plan: GlobalStepPlan,
    device_plan: DeviceExecutionPlan,
    release_receipt: Mapping[str, Any],
    environment_receipt: Mapping[str, Any],
    placement_receipt: Mapping[str, Any],
    capability_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete v2 physical launch chain without live recapture.

    ``execution_path.identity`` is a selected-policy identity.  It is never
    compared to, or substituted for, ``device_plan.plan_hash``.
    """

    path = validate_configuration_execution_path_for_training(
        execution_path=execution_path,
        training_contract=training_contract,
    )
    validate_gpu_capability_receipt(capability_receipt, require_ready=True)
    _validate_plan_contract(global_plan, device_plan, training_contract)
    if attempt.run_identity != training_contract.identity:
        raise ValueError("configuration attempt does not bind the training contract")
    if (
        attempt.attempt_id != training_contract.execution_binding.attempt_id
        or attempt.parent_attempt_hash is not None
    ):
        raise ValueError(
            "configuration execution must use its exact fresh attempt from zero"
        )
    if attempt.device_execution_plan_hash != device_plan.plan_hash:
        raise ValueError("configuration attempt does not bind the real device plan")
    if device_plan.global_plan_hash != global_plan.plan_hash:
        raise ValueError("configuration device plan does not bind the global plan")
    if global_plan.plan_hash != path.global_step_plan_hash:
        raise ValueError("configuration global plan does not bind the selected path")
    global_caps = (
        path.global_step_max_loss_tokens,
        path.global_step_max_attention_cells,
        path.global_step_max_sequences,
    )
    if (
        global_plan.max_loss_tokens,
        global_plan.max_attention_cells,
        global_plan.max_sequences,
    ) != global_caps:
        raise ValueError("configuration global plan caps differ from the selected path")
    if (
        device_plan.max_loss_tokens,
        device_plan.max_attention_cells,
    ) != global_caps[:2]:
        raise ValueError("configuration device plan global caps differ from the selected path")
    if device_plan.max_sequences != path.microbatch_max_sequences:
        raise ValueError("configuration device plan microbatch cap differs from the selected path")
    if (
        attempt.release_hash != path.release_hash
        or release_receipt.get("release_hash") != path.release_hash
        or training_contract.execution_binding.release_hash != path.release_hash
    ):
        raise ValueError("configuration release lineage mismatch")
    if (
        attempt.environment_hash != path.environment_hash
        or environment_receipt.get("environment_hash") != path.environment_hash
    ):
        raise ValueError("configuration environment lineage mismatch")
    if (
        attempt.data_placement_hash != path.placement_hash
        or placement_receipt.get("receipt_hash") != path.placement_hash
    ):
        raise ValueError("configuration placement lineage mismatch")
    if (
        capability_receipt.get("capability_hash") != path.device_identity_hash
        or capability_receipt.get("release_hash") != path.release_hash
        or capability_receipt.get("host_identity_digest") != attempt.host_id
        or capability_receipt.get("compile_mode") != path.compile_mode
        or capability_receipt.get("torch_version")
        != environment_receipt.get("torch_version")
        or capability_receipt.get("cuda_runtime_version")
        != environment_receipt.get("cuda_runtime_version")
        or path.allow_tf32
        and capability_receipt.get("supports_tf32") is not True
    ):
        raise ValueError("configuration device capability lineage mismatch")
    if spec.device != "cuda:0" or environment_receipt.get("device") != spec.device:
        raise ValueError("configuration selected device must be the exact cuda:0 path")
    if spec.read_mode != path.read_mode or spec.removable_data != path.removable_data:
        raise ValueError("configuration local data-read policy mismatch")
    if attempt.prefetch_batches != path.prefetch_batches:
        raise ValueError("configuration prefetch policy mismatch")
    if attempt.pin_memory != path.pin_memory:
        raise ValueError("configuration pinned-memory policy mismatch")
    observed_nonblocking = bool(attempt.pin_memory and spec.device.startswith("cuda"))
    if observed_nonblocking != path.nonblocking_device_transfer:
        raise ValueError("configuration nonblocking transfer policy mismatch")
    if (
        attempt.train_records_hash is None
        or attempt.validation_records_hash is None
    ):
        raise ValueError("configuration attempt lacks prepared records bindings")
    return {
        "execution_path_hash": path.identity,
        "device_execution_plan_hash": device_plan.plan_hash,
        "capability_hash": capability_receipt["capability_hash"],
        "train_records_hash": attempt.train_records_hash,
        "validation_records_hash": attempt.validation_records_hash,
        "execution_path_spot_policy": "non_spot_required",
        # A selected path and a capability receipt do not prove provider billing
        # class. Provider offer/pod evidence closes that claim outside the worker.
        "provider_spot_evidence_status": "not_verified_by_worker",
    }


def _validate_runtime_bindings(spec: WorkerSpec, contract, attempt: ExecutionAttempt) -> None:
    _validate_v2_attempt_cadence(contract, attempt)
    _validate_v2_physical_checkpoint_cadence(contract, attempt)
    bindings = (
        (_json(spec.release_receipt_path), "release_hash", attempt.release_hash),
        (_json(spec.environment_receipt_path), "environment_hash", attempt.environment_hash),
        (_json(spec.data_placement_receipt_path), "receipt_hash", attempt.data_placement_hash),
        (_json(spec.storage_layout_receipt_path), "receipt_hash", attempt.storage_layout_hash),
    )
    for payload, field, expected in bindings:
        if field == "receipt_hash" and payload.get(field) != stable_json_hash(
            {key: value for key, value in payload.items() if key != field}
        ):
            raise ValueError(f"worker runtime {field} artifact hash mismatch")
        if _identity(payload, field) != expected:
            raise ValueError(f"worker runtime {field} binding mismatch")
    validate_worker_storage_bindings(
        pack_root=spec.pack_root,
        placement_receipt=bindings[2][0],
        storage_layout_receipt=bindings[3][0],
        checkpoint_root=spec.checkpoint_root,
        persistent_checkpoint_root=spec.persistent_checkpoint_root,
        checkpoint_topology=spec.checkpoint_topology,
    )
    release = bindings[0][0]
    if release.get("artifact_kind") == "twc_gpu_runtime_release_v1":
        verify_thin_runtime_release(release)
    else:
        verify_release_receipt(release)
    _validate_release_code_binding(release, contract)
    release_training = Path(str(release["path"])).resolve(strict=True) / "training"
    package_file = getattr(twc, "__file__", None)
    if package_file is None or any(
        not Path(candidate).resolve(strict=True).is_relative_to(release_training)
        for candidate in (__file__, package_file)
    ):
        raise ValueError("worker imports are outside the verified release training tree")
    environment = bindings[1][0]
    if environment.get("environment_hash") != stable_json_hash(
        {key: value for key, value in environment.items() if key != "environment_hash"}
    ) or environment.get("status") != "ready":
        raise ValueError("worker environment receipt is not hash-valid and ready")
    expected_environment = {
        "device": spec.device,
        "precision": contract.precision,
        "attention_backend": contract.attention_backend,
        "optimizer_backend": contract.optimizer_backend,
        "compile_mode": contract.compile_mode,
        "allow_tf32": contract.allow_tf32,
    }
    mismatches = [
        key for key, value in expected_environment.items() if environment.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"worker environment/contract mismatch: {sorted(mismatches)}")
    live_environment = build_environment_receipt(**expected_environment)
    if live_environment != environment:
        raise ValueError("worker live Python/Torch/CUDA environment differs from the captured receipt")
    placement = bindings[2][0]
    live_root = spec.pack_root.resolve(strict=True)
    if placement.get("destination_root") != str(live_root) or placement.get("destination_device_id") != live_root.stat().st_dev:
        raise ValueError("worker live data root differs from the verified placement")
    identities = placement.get("file_identities")
    if not isinstance(identities, list) or not identities:
        raise ValueError("worker placement lacks live file identities")
    for expected in identities:
        relative = expected.get("relative_path") if isinstance(expected, dict) else None
        if not isinstance(relative, str):
            raise ValueError("worker placement file identity is malformed")
        path = live_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("worker placed data file is missing or replaced")
        stat = path.stat()
        observed = {
            "relative_path": relative,
            "device_id": stat.st_dev,
            "inode": stat.st_ino,
            "byte_count": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": expected.get("sha256"),
        }
        if observed != expected:
            raise ValueError("worker placed data file identity changed after readback")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected.get("sha256"):
            raise ValueError("worker placed data content changed after readback")


def _validate_configuration_runtime_bindings(
    *,
    spec: WorkerSpec,
    contract: object,
    attempt: ExecutionAttempt,
    global_plan: GlobalStepPlan,
    device_plan: DeviceExecutionPlan,
) -> dict[str, Any] | None:
    if not isinstance(contract, ExperimentTrainingContractV2):
        return None
    if (
        spec.configuration_execution_path_receipt_path is None
        or spec.gpu_capability_receipt_path is None
    ):
        raise ValueError(
            "v2 configuration worker requires execution-path and capability receipts"
        )
    path = load_configuration_execution_path_receipt(
        _json(spec.configuration_execution_path_receipt_path)
    )
    capability = _json(spec.gpu_capability_receipt_path)
    release = _json(spec.release_receipt_path)
    environment = _json(spec.environment_receipt_path)
    placement = _json(spec.data_placement_receipt_path)
    evidence = validate_configuration_worker_physical_binding(
        spec=spec,
        training_contract=contract,
        execution_path=path,
        attempt=attempt,
        global_plan=global_plan,
        device_plan=device_plan,
        release_receipt=release,
        environment_receipt=environment,
        placement_receipt=placement,
        capability_receipt=capability,
    )
    live_capability = capture_live_gpu_capability_receipt(
        provider_label=str(capability["provider_label"]),
        release_receipt=release,
        storage_root=spec.pack_root,
        compile_mode=path.compile_mode,
        device_index=0,
    )
    if live_capability != capability:
        raise ValueError("worker live GPU capability differs from selected device evidence")
    return evidence


def _prepare_cuda_batch(batch: PackedCausalBatch, *, device: str, pin_memory: bool) -> PackedCausalBatch:
    if not device.startswith("cuda"):
        if pin_memory:
            raise ValueError("pin_memory is valid only for a CUDA attempt")
        return batch
    inputs = batch.input_ids.pin_memory() if pin_memory else batch.input_ids
    targets = batch.target_ids.pin_memory() if pin_memory else batch.target_ids
    offsets = batch.cu_seqlens.pin_memory() if pin_memory else batch.cu_seqlens
    return PackedCausalBatch(
        input_ids=inputs.to(device, non_blocking=pin_memory),
        target_ids=targets.to(device, non_blocking=pin_memory),
        cu_seqlens=offsets.to(device, non_blocking=pin_memory),
        max_seqlen=batch.max_seqlen,
        loss_token_count=batch.loss_token_count,
    )


def _initialize_training_engine(
    model_config: DecoderConfig, contract: Any, *, device: str
) -> TrainingEngine:
    """Map the immutable scientific contract into the executed engine."""

    return TrainingEngine(
        initialize_decoder(model_config, seed=contract.seed),
        EngineConfig.from_contract(contract),
        device=device,
    )


def _plan_groups(plan: DeviceExecutionPlan) -> dict[int, tuple[DeviceSlice, ...]]:
    groups: dict[int, list[DeviceSlice]] = {}
    last = -1
    for item in plan.slices:
        if item.global_step_index < last:
            raise ValueError("device slices must be ordered by global step")
        groups.setdefault(item.global_step_index, []).append(item)
        last = item.global_step_index
    return {key: tuple(value) for key, value in groups.items()}


def _planned_microbatch_ordinals(
    plan: DeviceExecutionPlan, *, start_step: int, total_steps: int
) -> Iterator[tuple[int, ...]]:
    """Yield exact device slices in global-step order for prefetch execution."""

    if type(start_step) is not int or type(total_steps) is not int:
        raise ValueError("planned microbatch step bounds must be integers")
    if not 0 <= start_step <= total_steps:
        raise ValueError("planned microbatch step bounds are invalid")
    groups = _plan_groups(plan)
    for step_index in range(start_step, total_steps):
        slices = groups.get(step_index)
        if not slices:
            raise ValueError(f"device plan has no slices for global step {step_index}")
        for item in slices:
            yield tuple(range(item.sequence_start, item.sequence_stop))


def _latest_request(operator_root: Path, *, state: dict[str, Any]) -> dict[str, Any] | None:
    root = operator_root / "requests"
    if not root.is_dir():
        return None
    accepted: list[dict[str, Any]] = []
    for path in root.glob("*.json"):
        try:
            request = _json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        expected = stable_json_hash({key: value for key, value in request.items() if key != "request_hash"})
        if (
            request.get("request_hash") == expected
            and request.get("run_identity") == state["run_identity"]
            and request.get("attempt_hash") == state["attempt_hash"]
            and request.get("fencing_epoch") == state["fencing_epoch"]
            and not (operator_root / "request-acks" / f"{request.get('request_id')}.json").is_file()
        ):
            accepted.append(request)
    return max(accepted, key=lambda row: float(row["requested_at_epoch"]), default=None)


def _ack_request(operator_root: Path, request: dict[str, Any], **values: Any) -> None:
    root = operator_root / "request-acks"
    root.mkdir(mode=0o700, exist_ok=True)
    payload = {
        "artifact_kind": "twc_pretraining_control_ack_v1",
        "schema_version": 1,
        "request_id": request["request_id"],
        "request_hash": request["request_hash"],
        **values,
    }
    payload["ack_hash"] = stable_json_hash(payload)
    _atomic_json(root / f"{request['request_id']}.json", payload)


def run_worker(
    *,
    spec_path: Path,
    operator_root: Path,
    launch_id: str,
    activation_deadline_monotonic: float,
    engine: TrainingEngine | None = None,
    batch_source: BatchSource | None = None,
    validation_batch_source: BatchSource | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    spec = load_worker_spec(spec_path)
    operator = TrainingOperator(operator_root)
    state = operator.status()
    if state["state"] not in {"STARTING", "RUNNING", "QUIESCING"}:
        raise ValueError("worker requires STARTING, RUNNING, or QUIESCING operator state")
    contract = load_worker_contract(_json(spec.contract_path))
    attempts = _load_attempts(operator_root)
    attempt = attempts[-1]
    if state["run_identity"] != contract.identity or state["attempt_hash"] != attempt.identity:
        raise ValueError("worker operator/contract/attempt binding mismatch")
    if attempt.fencing_epoch != state["fencing_epoch"]:
        raise ValueError("worker attempt does not own the fencing epoch")
    _validate_runtime_bindings(spec, contract, attempt)
    global_plan = _load_global_plan(spec.global_step_plan_path)
    device_plan = _load_device_plan(spec.device_execution_plan_path)
    if global_plan.plan_hash != contract.global_step_plan_hash:
        raise ValueError("worker global-step plan does not bind the run contract")
    if device_plan.plan_hash != attempt.device_execution_plan_hash or device_plan.global_plan_hash != global_plan.plan_hash:
        raise ValueError("worker device plan binding mismatch")
    _validate_plan_contract(global_plan, device_plan, contract)
    if (
        isinstance(contract, ExperimentTrainingContractV2)
        and state.get("checkpoint_generation") is not None
    ):
        raise ValueError("v2 configuration execution cannot resume a checkpoint")
    configuration_physical_evidence = _validate_configuration_runtime_bindings(
        spec=spec,
        contract=contract,
        attempt=attempt,
        global_plan=global_plan,
        device_plan=device_plan,
    )
    performance_window: ConfigurationPerformanceWindowAccumulator | None = None
    performance_policy_hash: str | None = None
    if isinstance(contract, ExperimentTrainingContractV2):
        performance_policy_path = (
            operator_root / "configuration-performance-policy.json"
        )
        if performance_policy_path.is_symlink() or not performance_policy_path.is_file():
            raise ValueError("v2 configuration performance policy is missing")
        configuration_performance_policy = load_configuration_performance_policy(
            _json(performance_policy_path),
            expected_execution_path_hash=(
                contract.execution_binding.execution_path_hash
            ),
        )
        performance_window = ConfigurationPerformanceWindowAccumulator(
            configuration_performance_policy
        )
        performance_policy_hash = str(configuration_performance_policy["policy_hash"])
    data_root, records = _load_records(
        spec.records_path,
        contract,
        pack_root=spec.pack_root,
        placement_hash=attempt.data_placement_hash,
        split="train",
        expected_records_hash=attempt.train_records_hash,
    )
    validation_root, validation_records = _load_records(
        spec.validation_records_path, contract, pack_root=spec.pack_root,
        placement_hash=attempt.data_placement_hash, split="validation",
        expected_records_hash=attempt.validation_records_hash,
    )
    evaluation_groups = _evaluation_groups(validation_records, contract)
    evaluation_component_group_hashes: tuple[str, ...] = ()
    if attempt.validation_records_hash is not None:
        evaluation_component_group_hashes = _evaluation_component_group_hashes(
            validation_records,
            evaluation_groups,
            validation_records_hash=attempt.validation_records_hash,
        )
    if global_plan.sequence_count != len(records):
        raise ValueError("worker global plan and BatchSource record counts differ")
    if any(item.sequence_stop > len(records) for item in device_plan.slices):
        raise ValueError("worker device plan exceeds BatchSource records")
    lengths = tuple(record.token_count - 1 for record in records)
    for index, step in enumerate(global_plan.steps):
        selected = lengths[step.sequence_start : step.sequence_stop]
        if step.loss_tokens != sum(selected) or step.attention_cells != sum(value * value for value in selected):
            raise ValueError(f"worker global step {index} disagrees with BatchSource record lengths")
    for index, item in enumerate(device_plan.slices):
        selected = lengths[item.sequence_start : item.sequence_stop]
        if item.loss_tokens != sum(selected) or item.attention_cells != sum(value * value for value in selected):
            raise ValueError(f"worker device slice {index} disagrees with BatchSource record lengths")

    model_config = _load_model(spec.model_path, contract.model_hash)
    if model_config.context_length != contract.context_length:
        raise ValueError("worker model and run context lengths differ")
    expected_parameter_count = getattr(contract, "expected_parameter_count", None)
    if expected_parameter_count is not None and estimate_parameter_count(model_config) != expected_parameter_count:
        raise ValueError("worker model parameter count differs from the engineering contract")
    owns_source = batch_source is None
    owns_validation_source = validation_batch_source is None
    if batch_source is None:
        batch_source = BatchSource(
            root=data_root,
            records=records,
            read_mode=spec.read_mode,
            removable=spec.removable_data,
        )
    if validation_batch_source is None:
        validation_batch_source = BatchSource(
            root=validation_root, records=validation_records,
            read_mode=spec.read_mode, removable=spec.removable_data,
        )
    if engine is None:
        engine = _initialize_training_engine(
            model_config, contract, device=spec.device
        )
    checkpoints = CheckpointManager(
        spec.checkpoint_root,
        persistent_root=spec.persistent_checkpoint_root,
        topology=spec.checkpoint_topology,
        create=False,
    )
    resumed_from_attempt_hash: str | None = None
    resumed_from_generation: int | None = None
    resumed_from_step: int | None = None
    resumed_from_loss_tokens: int | None = None
    resumed_from_global_ordinal: int | None = None
    if state.get("checkpoint_generation") is not None:
        selected_checkpoint = attempt.selected_checkpoint
        if selected_checkpoint is None:
            raise ValueError("resume attempt lacks an exact selected checkpoint")
        selected_generation = int(selected_checkpoint.generation)
        if selected_generation != int(state["checkpoint_generation"]):
            raise ValueError("resume attempt and operator select different checkpoints")
        restored = checkpoints.validate_reference(
            CheckpointReference(
                selected_generation,
                checkpoints.root / f"generation-{selected_generation:08d}",
                selected_checkpoint.durability,
                selected_checkpoint.manifest_sha256,
                selected_checkpoint.state_sha256,
            )
        )
        if (
            restored.run_identity != contract.identity
            or restored.attempt_hash != attempt.parent_attempt_hash
            or restored.step != selected_checkpoint.step
            or restored.consumed_loss_tokens
            != selected_checkpoint.consumed_loss_tokens
            or restored.next_global_ordinal
            != selected_checkpoint.next_global_ordinal
        ):
            raise ValueError("resume checkpoint is outside the selected attempt lineage")
        engine.restore(restored.state)
        torch.set_rng_state(restored.cpu_rng_state)
        if torch.cuda.is_available() and restored.cuda_rng_states:
            torch.cuda.set_rng_state_all(list(restored.cuda_rng_states))
        if restored.step != engine.step or restored.consumed_loss_tokens != engine.consumed_loss_tokens:
            raise ValueError("resume engine cursor differs from checkpoint metadata")
        expected_ordinal = global_plan.steps[engine.step - 1].sequence_stop if engine.step else 0
        if restored.next_global_ordinal != expected_ordinal:
            raise ValueError("resume checkpoint semantic cursor differs from the global plan")
        resumed_from_attempt_hash = restored.attempt_hash
        resumed_from_generation = restored.generation
        resumed_from_step = restored.step
        resumed_from_loss_tokens = restored.consumed_loss_tokens
        resumed_from_global_ordinal = restored.next_global_ordinal

    groups = _plan_groups(device_plan)
    prefetched_train_batches = iter(
        batch_source.prefetch(
            _planned_microbatch_ordinals(
                device_plan,
                start_step=engine.step,
                total_steps=len(global_plan.steps),
            )
        )
    )
    started = monotonic()
    last_local_tokens = engine.consumed_loss_tokens
    last_durable_tokens = engine.consumed_loss_tokens
    last_eval_tokens = engine.consumed_loss_tokens
    last_local_time = started
    last_durable_time = started
    checkpoint_seconds = 0.0
    durable_seconds = 0.0
    durable_blocked_total = 0.0
    evaluation_batches = 0
    latest_evaluation: dict[str, Any] | None = None
    evaluation_runs = 0
    evaluation_ledger_path = operator_root / f"worker-evaluations-{attempt.identity}.jsonl"
    if evaluation_ledger_path.exists() and evaluation_ledger_path.stat().st_size:
        raise ValueError("active attempt already has an evaluation ledger")
    step_times: list[float] = []
    input_seconds_total = 0.0
    last_reference: CheckpointReference | None = None
    last_checkpoint_step: int | None = None
    stop_reason: str | None = None
    stop_latency_seconds = 0.0
    compile_heartbeat_pending = contract.compile_mode != "disabled"
    utilization = CudaUtilizationSampler(spec.device)
    performance_sample_start = 0
    latest_performance_window: dict[str, Any] | None = None
    cancel_event = cancel_event or Event()
    def phase_heartbeat(phase: str) -> None:
        path = operator_root / "worker-telemetry.json"
        try:
            telemetry = _json(path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            telemetry = {
                "artifact_kind": "twc_pretraining_worker_telemetry_v2",
                "schema_version": 2,
                "run_identity": contract.identity,
                "attempt_hash": attempt.identity,
                "fencing_epoch": attempt.fencing_epoch,
                "step": engine.step,
                "consumed_loss_tokens": engine.consumed_loss_tokens,
            }
        telemetry.pop("telemetry_hash", None)
        telemetry.update(
            phase=phase,
            heartbeat_epoch=time.time(),
            **(
                {
                    "checkpoint_filesystems": _checkpoint_filesystem_observations(
                        spec
                    )
                }
                if state.get("runtime_safety") is not None
                else {}
            ),
        )
        telemetry["telemetry_hash"] = stable_json_hash(telemetry)
        _atomic_json(path, telemetry)

    def record_evaluation(result: Any) -> dict[str, Any]:
        nonlocal evaluation_batches, evaluation_runs, latest_evaluation
        evaluation_runs += 1
        evaluation_batches += int(result.batch_count)
        payload = {
            "artifact_kind": "twc_pretraining_evaluation_result_v2",
            "schema_version": 2,
            "run_identity": contract.identity,
            "attempt_hash": attempt.identity,
            "fencing_epoch": attempt.fencing_epoch,
            "evaluation_index": evaluation_runs,
            "previous_receipt_hash": latest_evaluation["receipt_hash"] if latest_evaluation else None,
            "at_consumed_loss_tokens": engine.consumed_loss_tokens,
            "loss": result.loss,
            "loss_tokens": result.loss_tokens,
            "batch_count": result.batch_count,
            "component_losses": list(result.component_losses),
            "component_loss_tokens": list(result.component_loss_tokens),
        }
        if attempt.validation_records_hash is not None:
            if int(result.batch_count) != len(evaluation_component_group_hashes):
                raise ValueError(
                    "evaluation component identities do not align with batches"
                )
            payload.update(
                artifact_kind="twc_pretraining_evaluation_result_v3",
                schema_version=3,
                component_group_hashes=list(evaluation_component_group_hashes),
                validation_records_hash=attempt.validation_records_hash,
            )
        payload["receipt_hash"] = stable_json_hash(payload)
        with evaluation_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        latest_evaluation = payload
        return payload

    failure_recorded = False

    def record_failure(error: BaseException, *, phase: str, cursor_committed: bool) -> None:
        nonlocal failure_recorded
        if failure_recorded:
            return
        failure = {
            "artifact_kind": "twc_pretraining_worker_failure_v1",
            "schema_version": 1,
            "run_identity": contract.identity,
            "attempt_hash": attempt.identity,
            "fencing_epoch": attempt.fencing_epoch,
            "step": engine.step,
            "consumed_loss_tokens": engine.consumed_loss_tokens,
            "failure_class": "out_of_memory" if isinstance(error, torch.OutOfMemoryError) else "worker_error",
            "failure_phase": phase,
            "cursor_committed": cursor_committed,
        }
        failure["receipt_hash"] = stable_json_hash(failure)
        _atomic_json(operator_root / "worker-failure.json", failure)
        elapsed = max(monotonic() - started, 1e-9)
        try:
            telemetry = _json(operator_root / "worker-telemetry.json")
        except (FileNotFoundError, ValueError, json.JSONDecodeError):
            telemetry = {
                "artifact_kind": "twc_pretraining_worker_telemetry_v2",
                "schema_version": 2,
                "run_identity": contract.identity,
                "attempt_hash": attempt.identity,
                "fencing_epoch": attempt.fencing_epoch,
            }
        telemetry.pop("telemetry_hash", None)
        telemetry.update(
            phase="failed",
            heartbeat_epoch=time.time(),
            step=engine.step,
            consumed_loss_tokens=engine.consumed_loss_tokens,
            throughput_loss_tokens_per_second=(
                engine.consumed_loss_tokens / elapsed
            ),
            recent_step_seconds_p50=(
                statistics.median(step_times[-20:]) if step_times else 0.0
            ),
            input_wait_fraction=(
                input_seconds_total / max(sum(step_times), 1e-9)
            ),
            evaluation_batches=evaluation_batches,
            evaluation_loss=(
                latest_evaluation["loss"] if latest_evaluation else None
            ),
            evaluation_loss_tokens=(
                latest_evaluation["loss_tokens"] if latest_evaluation else 0
            ),
            evaluation_result_batch_count=(
                latest_evaluation["batch_count"] if latest_evaluation else 0
            ),
            evaluation_receipt_hash=(
                latest_evaluation["receipt_hash"] if latest_evaluation else None
            ),
            checkpoint_duration_seconds=checkpoint_seconds,
            persistent_sync_duration_seconds=durable_seconds,
            durable_replication_mode=replication_mode,
            durable_replication_blocked_training_seconds_total=durable_blocked_total,
            durable_replication_fraction_of_elapsed=(
                durable_blocked_total / elapsed
            ),
            storage_free_bytes=(
                os.statvfs(spec.checkpoint_root).f_bavail
                * os.statvfs(spec.checkpoint_root).f_frsize
            ),
            **(
                {"checkpoint_filesystems": _checkpoint_filesystem_observations(spec)}
                if state.get("runtime_safety") is not None
                else {}
            ),
            gpu_peak_memory_bytes=(
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            gpu_peak_reserved_memory_bytes=(
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            ),
            stop_latency_seconds=stop_latency_seconds,
            dataloader_workers=0,
            prefetch_factor=attempt.prefetch_batches,
            pin_memory=attempt.pin_memory,
            nonblocking_h2d=bool(
                attempt.pin_memory and spec.device.startswith("cuda")
            ),
            failure_receipt_hash=failure["receipt_hash"],
            failure_class=failure["failure_class"],
            failure_phase=failure["failure_phase"],
            scientific_result_eligible=False,
            production_unlock_eligible=False,
            f1_unlock_eligible=False,
            raw_text_serialized=False,
            raw_url_serialized=False,
            tokens_serialized=False,
            secrets_serialized=False,
            **(latest_performance_window or {}),
            **utilization.evidence(),
        )
        telemetry["telemetry_hash"] = stable_json_hash(telemetry)
        _atomic_json(operator_root / "worker-telemetry.json", telemetry)
        _atomic_json(
            operator_root / f"worker-telemetry-{attempt.identity}.json",
            telemetry,
        )
        failure_recorded = True

    def evaluate_and_record(*, cursor_committed: bool) -> dict[str, Any]:
        phase_heartbeat("evaluation")
        try:
            result = engine.evaluate(
                validation_batch_source.read_batch(group) for group in evaluation_groups
            )
            return record_evaluation(result)
        except BaseException as error:
            record_failure(error, phase="evaluation", cursor_committed=cursor_committed)
            raise

    terminal_durability = (
        "local_verified" if spec.checkpoint_topology == "local_only" else "persistent_verified"
    )
    replication_mode = (
        "local_only_ephemeral" if spec.checkpoint_topology == "local_only" else "synchronous_verified"
    )

    def publish(durability: str, *, durable_cadence: bool = False) -> CheckpointReference:
        nonlocal checkpoint_seconds, durable_seconds, durable_blocked_total, last_reference, last_checkpoint_step, last_local_tokens, last_durable_tokens
        phase_heartbeat("durable_checkpoint" if durability == "persistent_verified" else "local_checkpoint")
        before = monotonic()
        reference = checkpoints.publish(
            CheckpointPayload(
                run_identity=contract.identity,
                attempt_hash=attempt.identity,
                step=engine.step,
                consumed_loss_tokens=engine.consumed_loss_tokens,
                next_global_ordinal=(global_plan.steps[engine.step - 1].sequence_stop if engine.step else 0),
                state=engine.snapshot(),
                cpu_rng_state=torch.get_rng_state(),
                cuda_rng_states=tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
            ),
            durability=durability,
        )
        elapsed = monotonic() - before
        checkpoint_seconds = max(checkpoint_seconds, elapsed)
        last_local_tokens = engine.consumed_loss_tokens
        if reference.durability == "persistent_verified":
            durable_seconds = max(durable_seconds, elapsed)
            durable_blocked_total += elapsed
        if durable_cadence:
            last_durable_tokens = engine.consumed_loss_tokens
        last_reference = reference
        last_checkpoint_step = engine.step
        return reference

    self_token = process_start_token(os.getpid())
    if self_token is None:
        raise RuntimeError("worker could not establish its process-start identity")
    ready = build_worker_ready(
        launch_id=launch_id,
        run_identity=contract.identity,
        attempt_hash=attempt.identity,
        fencing_epoch=attempt.fencing_epoch,
        release_hash=attempt.release_hash,
        environment_hash=attempt.environment_hash,
        data_placement_hash=attempt.data_placement_hash,
        storage_layout_hash=attempt.storage_layout_hash,
        performance_policy_hash=performance_policy_hash,
        pid=os.getpid(),
        process_start_token=self_token,
        ready_at_epoch=time.time(),
        configuration_physical_evidence=configuration_physical_evidence,
    )
    namespace = open_startup_namespace(operator_root, launch_id)
    try:
        publish_ready_once(namespace, "worker", ready)
    finally:
        namespace.close()
    while True:
        state = operator.status()
        activation = activation_state(
            state, role="worker", launch_id=launch_id, receipt=ready,
            pid=os.getpid(), start_token=self_token,
        )
        if activation != "wait":
            break
        if monotonic() >= activation_deadline_monotonic:
            state = operator.status()
            activation = activation_state(
                state, role="worker", launch_id=launch_id, receipt=ready,
                pid=os.getpid(), start_token=self_token,
            )
            if activation == "wait":
                raise TimeoutError("worker activation deadline expired")
            break
        time.sleep(0.025)
    if activation == "terminal":
        close_prefetch = getattr(prefetched_train_batches, "close", None)
        if callable(close_prefetch):
            close_prefetch()
        if owns_source:
            batch_source.close()
        if owns_validation_source:
            validation_batch_source.close()
        return {"status": "activation_already_terminal"}

    try:
        utilization.start()
        phase_heartbeat("activated")
        if observe_process(state["watchdog_process_identity"]) != "confirmed_alive":
            raise RuntimeError("confirmed watchdog is not alive after activation")
        while engine.step < len(global_plan.steps):
            operator.validate_fence(attempt.fencing_epoch)
            state = operator.status()
            request = _latest_request(operator_root, state=state)
            if state.get("state") == "QUIESCING" and request is None:
                raise RuntimeError("QUIESCING worker lacks its exact control request")
            if request is not None:
                action = request["action"]
                if action == "checkpoint":
                    reference = publish("local_verified")
                    checkpoint_step = engine.step
                    checkpoint_ordinal = (
                        global_plan.steps[engine.step - 1].sequence_stop
                        if engine.step
                        else 0
                    )
                    _ack_request(
                        operator_root,
                        request,
                        outcome="checkpointed",
                        generation=reference.generation,
                        path=str(reference.path),
                        durability=reference.durability,
                        step=checkpoint_step,
                        next_global_ordinal=checkpoint_ordinal,
                    )
                    response_deadline = float(request["requested_at_epoch"]) + int(
                        request["deadline_seconds"]
                    )
                    while True:
                        committed = operator.status()
                        if (
                            committed.get("state") in {"RUNNING", "QUIESCING"}
                            and committed.get("checkpoint_generation")
                            == reference.generation
                            and committed.get("step") == checkpoint_step
                            and committed.get("next_global_ordinal")
                            == checkpoint_ordinal
                        ):
                            break
                        if time.time() >= response_deadline:
                            stop_reason = "checkpoint_response_timeout"
                            break
                        time.sleep(0.01)
                    if stop_reason == "checkpoint_response_timeout":
                        break
                else:
                    durability = "local_verified" if action == "pause" else terminal_durability
                    reference = publish(durability, durable_cadence=True)
                    _ack_request(operator_root, request, outcome="stopped", generation=reference.generation)
                    stop_latency_seconds = max(0.0, time.time() - float(request["requested_at_epoch"]))
                    stop_reason = action
                    break
            if cancel_event.is_set():
                requested_at = getattr(cancel_event, "requested_at_epoch", None)
                if requested_at is not None:
                    stop_latency_seconds = max(0.0, time.time() - float(requested_at))
                stop_reason = "signal_stop"
                break
            if performance_window is not None and performance_window.starts_at_step(
                engine.step
            ):
                if spec.device.startswith("cuda"):
                    torch.cuda.synchronize()
                performance_sample_start = utilization.sample_count()
            slices = groups.get(engine.step)
            if not slices:
                raise ValueError(f"device plan has no slices for global step {engine.step}")
            input_timing = [0.0]

            def prepared_microbatches():
                for _ in slices:
                    input_started = monotonic()
                    try:
                        batch = next(prefetched_train_batches)
                    except StopIteration:
                        input_timing[0] += monotonic() - input_started
                        raise ValueError("prefetched train stream ended before the device plan")
                    prepared = _prepare_cuda_batch(batch, device=spec.device, pin_memory=attempt.pin_memory)
                    input_timing[0] += monotonic() - input_started
                    yield prepared
            compute_started = monotonic()
            previous_loss_tokens = engine.consumed_loss_tokens
            if compile_heartbeat_pending:
                phase_heartbeat("compile")
            try:
                active = operator.status()
                if (
                    active.get("active_launch_id") != launch_id
                    or active.get("watchdog_process_identity") != state["watchdog_process_identity"]
                    or observe_process(state["watchdog_process_identity"]) != "confirmed_alive"
                ):
                    raise RuntimeError("confirmed watchdog is not alive before optimizer step")
                result = engine.train_step(
                    prepared_microbatches(), expected_loss_tokens=global_plan.steps[engine.step].loss_tokens
                )
            except BaseException as error:
                record_failure(error, phase="training", cursor_committed=False)
                raise
            compile_heartbeat_pending = False
            step_seconds = monotonic() - compute_started
            input_seconds = input_timing[0]
            step_times.append(step_seconds)
            input_seconds_total += input_seconds
            if performance_window is not None:
                completed_window = performance_window.observe(
                    completed_step=engine.step,
                    loss_tokens=int(result.loss_tokens),
                    step_seconds=step_seconds,
                    input_seconds=input_seconds,
                    utilization_evidence=(
                        utilization.evidence_since(performance_sample_start)
                        if performance_window.completes_at_step(engine.step)
                        else None
                    ),
                )
                if completed_window is not None:
                    latest_performance_window = completed_window
            now = monotonic()
            due_local = (
                engine.consumed_loss_tokens - last_local_tokens >= attempt.checkpoint_every_loss_tokens
                or now - last_local_time >= attempt.checkpoint_max_interval_seconds
            )
            due_durable = (
                engine.consumed_loss_tokens - last_durable_tokens >= attempt.durable_every_loss_tokens
                or now - last_durable_time >= attempt.durable_max_interval_seconds
            )
            if due_durable:
                publish(terminal_durability, durable_cadence=True)
                last_durable_time = now
                last_local_time = now
            elif due_local:
                publish("local_verified")
                last_local_time = now
            evaluation_due = configuration_evaluation_due(
                previous_loss_tokens=previous_loss_tokens,
                consumed_loss_tokens=engine.consumed_loss_tokens,
                last_evaluation_loss_tokens=last_eval_tokens,
                evaluation_interval_loss_tokens=(
                    contract.evaluation_every_loss_tokens
                    if getattr(contract, "evaluation_every_loss_tokens", None)
                    is not None
                    else attempt.evaluation_every_loss_tokens
                ),
                warmup_loss_tokens=contract.warmup_loss_tokens,
                stable_loss_tokens=contract.stable_loss_tokens,
                target_loss_tokens=contract.target_loss_tokens,
            )
            if evaluation_due:
                cursor_committed = bool(
                    last_reference is not None
                    and last_checkpoint_step == engine.step
                    and last_local_tokens == engine.consumed_loss_tokens
                )
                evaluate_and_record(cursor_committed=cursor_committed)
                last_eval_tokens = engine.consumed_loss_tokens
            elapsed = max(monotonic() - started, 1e-9)
            telemetry = {
                "artifact_kind": "twc_pretraining_worker_telemetry_v2",
                "schema_version": 2,
                "run_identity": contract.identity,
                "attempt_hash": attempt.identity,
                "fencing_epoch": attempt.fencing_epoch,
                "phase": "train",
                "heartbeat_epoch": time.time(),
                "step": engine.step,
                "consumed_loss_tokens": engine.consumed_loss_tokens,
                "throughput_loss_tokens_per_second": engine.consumed_loss_tokens / elapsed,
                "recent_step_seconds_p50": statistics.median(step_times[-20:]),
                "input_wait_fraction": input_seconds_total / max(sum(step_times), 1e-9),
                "evaluation_batches": evaluation_batches,
                "evaluation_loss": latest_evaluation["loss"] if latest_evaluation else None,
                "evaluation_loss_tokens": latest_evaluation["loss_tokens"] if latest_evaluation else 0,
                "evaluation_result_batch_count": latest_evaluation["batch_count"] if latest_evaluation else 0,
                "evaluation_receipt_hash": latest_evaluation["receipt_hash"] if latest_evaluation else None,
                "checkpoint_duration_seconds": checkpoint_seconds,
                "persistent_sync_duration_seconds": durable_seconds,
                "durable_replication_mode": replication_mode,
                "durable_replication_blocked_training_seconds_total": durable_blocked_total,
                "durable_replication_fraction_of_elapsed": durable_blocked_total / elapsed,
                "storage_free_bytes": os.statvfs(spec.checkpoint_root).f_bavail * os.statvfs(spec.checkpoint_root).f_frsize,
                **(
                    {"checkpoint_filesystems": _checkpoint_filesystem_observations(spec)}
                    if state.get("runtime_safety") is not None
                    else {}
                ),
                "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                "gpu_peak_reserved_memory_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                **(latest_performance_window or {}),
                **utilization.evidence(),
                "stop_latency_seconds": stop_latency_seconds,
                "dataloader_workers": 0,
                "prefetch_factor": attempt.prefetch_batches,
                "pin_memory": attempt.pin_memory,
                "nonblocking_h2d": bool(attempt.pin_memory and spec.device.startswith("cuda")),
            }
            telemetry["telemetry_hash"] = stable_json_hash(telemetry)
            _atomic_json(operator_root / "worker-telemetry.json", telemetry)
            _atomic_json(operator_root / f"worker-telemetry-{attempt.identity}.json", telemetry)
            if _diagnostic_planned_stop_due(contract, attempt, engine.step):
                decision_started = time.time()
                if last_reference is None or last_reference.durability != terminal_durability:
                    last_reference = publish(terminal_durability, durable_cadence=True)
                stop_latency_seconds = max(time.time() - decision_started, 1e-9)
                stop_reason = "diagnostic_planned_stop"
                break

        completed = (
            engine.step == len(global_plan.steps)
            and global_plan.total_loss_tokens == contract.target_loss_tokens
            and engine.consumed_loss_tokens == contract.target_loss_tokens
        )
        if (
            last_reference is None
            or last_reference.durability != terminal_durability
            or last_checkpoint_step != engine.step
            or last_local_tokens != engine.consumed_loss_tokens
        ):
            last_reference = publish(terminal_durability, durable_cadence=True)
        terminal_evaluation_missing = bool(
            completed
            and (
                latest_evaluation is None
                or latest_evaluation.get("at_consumed_loss_tokens")
                != engine.consumed_loss_tokens
            )
        )
        if engine.step > 0 and (evaluation_batches == 0 or terminal_evaluation_missing):
            evaluate_and_record(cursor_committed=True)
        utilization.stop()
        final_elapsed = max(monotonic() - started, 1e-9)
        phase_heartbeat("stopped" if not completed else "completed")
        for telemetry_path in (
            operator_root / "worker-telemetry.json",
            operator_root / f"worker-telemetry-{attempt.identity}.json",
        ):
            try:
                telemetry = _json(telemetry_path)
            except (FileNotFoundError, ValueError, json.JSONDecodeError):
                telemetry = {
                    "artifact_kind": "twc_pretraining_worker_telemetry_v2",
                    "schema_version": 2,
                    "run_identity": contract.identity,
                    "attempt_hash": attempt.identity,
                    "fencing_epoch": attempt.fencing_epoch,
                }
            telemetry.pop("telemetry_hash", None)
            telemetry.update(
                phase="completed" if completed else "stopped",
                heartbeat_epoch=time.time(),
                step=engine.step,
                consumed_loss_tokens=engine.consumed_loss_tokens,
                evaluation_batches=evaluation_batches,
                evaluation_loss=latest_evaluation["loss"] if latest_evaluation else None,
                evaluation_loss_tokens=latest_evaluation["loss_tokens"] if latest_evaluation else 0,
                evaluation_result_batch_count=latest_evaluation["batch_count"] if latest_evaluation else 0,
                evaluation_receipt_hash=latest_evaluation["receipt_hash"] if latest_evaluation else None,
                stop_latency_seconds=stop_latency_seconds,
                **(
                    {"checkpoint_filesystems": _checkpoint_filesystem_observations(spec)}
                    if state.get("runtime_safety") is not None
                    else {}
                ),
                checkpoint_duration_seconds=checkpoint_seconds,
                persistent_sync_duration_seconds=durable_seconds,
                durable_replication_mode=replication_mode,
                durable_replication_blocked_training_seconds_total=durable_blocked_total,
                durable_replication_fraction_of_elapsed=durable_blocked_total / final_elapsed,
                gpu_peak_reserved_memory_bytes=(
                    torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
                ),
                pin_memory=attempt.pin_memory,
                nonblocking_h2d=bool(attempt.pin_memory and spec.device.startswith("cuda")),
                **(latest_performance_window or {}),
                **_terminal_utilization_evidence(telemetry, utilization.evidence()),
            )
            telemetry["telemetry_hash"] = stable_json_hash(telemetry)
            _atomic_json(telemetry_path, telemetry)
        if last_reference is None:  # pragma: no cover - terminal paths always checkpoint
            raise RuntimeError("terminal worker result requires a checkpoint")
        terminal_checkpoint = checkpoints.validate_reference(last_reference)
        result_payload = {
            "artifact_kind": "twc_pretraining_worker_result_v2",
            "schema_version": 2,
            "run_identity": contract.identity,
            "attempt_hash": attempt.identity,
            "fencing_epoch": attempt.fencing_epoch,
            "step": engine.step,
            "consumed_loss_tokens": engine.consumed_loss_tokens,
            "next_global_ordinal": (global_plan.steps[engine.step - 1].sequence_stop if engine.step else 0),
            "checkpoint_generation": last_reference.generation,
            "checkpoint_durability": last_reference.durability,
            "checkpoint_manifest_sha256": terminal_checkpoint.manifest_sha256,
            "checkpoint_state_sha256": terminal_checkpoint.state_sha256,
            "completed": completed,
            "stop_reason": stop_reason or ("completed" if completed else "finite_plan_exhausted"),
            "resumed_from_attempt_hash": resumed_from_attempt_hash,
            "resumed_from_checkpoint_generation": resumed_from_generation,
            "resumed_from_step": resumed_from_step,
            "resumed_from_loss_tokens": resumed_from_loss_tokens,
            "resumed_from_global_ordinal": resumed_from_global_ordinal,
            "evaluation_loss": latest_evaluation["loss"] if latest_evaluation else None,
            "evaluation_loss_tokens": latest_evaluation["loss_tokens"] if latest_evaluation else 0,
            "evaluation_result_batch_count": latest_evaluation["batch_count"] if latest_evaluation else 0,
            "evaluation_receipt_hash": latest_evaluation["receipt_hash"] if latest_evaluation else None,
        }
        result_payload["result_hash"] = stable_json_hash(result_payload)
        outcomes = operator_root / "attempt-outcomes"
        outcomes.mkdir(mode=0o700, exist_ok=True)
        outcome_path = outcomes / f"{attempt.identity}.json"
        if outcome_path.exists():
            if _json(outcome_path) != result_payload:
                raise ValueError("attempt outcome is immutable")
        else:
            _atomic_json(outcome_path, result_payload)
        return result_payload
    except BaseException as error:
        if not failure_recorded:
            cursor_committed = bool(
                last_reference is not None
                and last_checkpoint_step == engine.step
                and last_local_tokens == engine.consumed_loss_tokens
            )
            record_failure(
                error,
                phase="runtime",
                cursor_committed=cursor_committed,
            )
        raise
    finally:
        utilization.stop()
        close_prefetch = getattr(prefetched_train_batches, "close", None) if "prefetched_train_batches" in locals() else None
        if callable(close_prefetch):
            close_prefetch()
        if owns_source:
            batch_source.close()
        if owns_validation_source:
            validation_batch_source.close()


def _diagnostic_planned_stop_due(contract: Any, attempt: ExecutionAttempt, step: int) -> bool:
    return (
        getattr(contract, "initial_attempt_stop_step", None) is not None
        and attempt.parent_attempt_hash is None
        and step >= contract.initial_attempt_stop_step
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--operator-root", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--activation-deadline-monotonic", required=True, type=float)
    args = parser.parse_args(argv)
    if not math.isfinite(args.activation_deadline_monotonic):
        parser.error("--activation-deadline-monotonic must be finite")
    cancelled = CancellationSignal()

    def request_stop(_signum: int, _frame: object) -> None:
        cancelled.request()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    run_worker(
        spec_path=Path(args.spec),
        operator_root=Path(args.operator_root),
        launch_id=args.launch_id,
        activation_deadline_monotonic=args.activation_deadline_monotonic,
        cancel_event=cancelled,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
