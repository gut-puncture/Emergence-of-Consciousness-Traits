"""Local, provider-neutral training lifecycle and readiness gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
import uuid
import hashlib
import math
from typing import Any

from twc.repro.hashing import stable_json_hash
from twc.pretraining.training.checkpoints import CheckpointManager, CheckpointReference, LoadedCheckpoint
from twc.pretraining.training.contracts import ExecutionAttempt, RunContract, load_execution_attempt
from twc.pretraining.training.storage import checkpoint_filesystem_requirements
from twc.pretraining.training.evaluation_evidence import validate_evaluation_chain
from twc.pretraining.training.startup import load_watchdog_ready, load_worker_ready
from twc.pretraining.configuration_performance import load_configuration_performance_policy
from twc.runtime.host import host_identity
from twc.runtime.processes import observe_process, process_start_token


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class CanaryReceipt:
    run_identity: str
    data_manifest_hash: str
    environment_hash: str
    device_execution_plan_hash: str
    host_identity_digest: str
    checkpoint_receipt_hash: str
    telemetry_evidence_hash: str
    trained_steps: int
    evaluation_batches: int
    evaluation_loss: float
    evaluation_loss_tokens: int
    evaluation_receipt_hash: str
    stop_resume_verified: bool
    median_gpu_utilization: float
    input_wait_fraction: float
    gpu_peak_memory_bytes: int
    checkpoint_duration_seconds: float
    persistent_sync_duration_seconds: float
    stop_latency_seconds: float
    dataloader_workers: int
    prefetch_factor: int
    event_chain_hash: str
    receipt_hash: str

    @classmethod
    def create(cls, **values: Any) -> "CanaryReceipt":
        payload = {
            "artifact_kind": "twc_pretraining_empirical_canary_v3",
            "schema_version": 3,
            **values,
        }
        return cls(**values, receipt_hash=stable_json_hash(payload))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanaryReceipt":
        fields = set(cls.__dataclass_fields__)
        metadata = {"artifact_kind", "schema_version"}
        unknown = set(payload) - fields - metadata
        missing = fields - set(payload)
        if unknown:
            raise ValueError(f"canary receipt has unknown fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"canary receipt is missing fields: {sorted(missing)}")
        if payload.get("artifact_kind") != "twc_pretraining_empirical_canary_v3" or payload.get("schema_version") != 3:
            raise ValueError("unsupported canary receipt artifact")
        return cls(**{field: payload[field] for field in fields})

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_pretraining_empirical_canary_v3",
            "schema_version": 3,
            **asdict(self),
        }

    def validate(
        self,
        expected_run_identity: str,
        expected_data_manifest_hash: str,
        *,
        minimum_gpu_utilization: float = 90.0,
        maximum_input_wait_fraction: float = 0.10,
    ) -> None:
        unhashed = {key: value for key, value in self.to_dict().items() if key != "receipt_hash"}
        if self.receipt_hash != stable_json_hash(unhashed):
            raise ValueError("canary receipt hash mismatch")
        if self.run_identity != expected_run_identity:
            raise ValueError("empirical canary run identity mismatch")
        if self.data_manifest_hash != expected_data_manifest_hash:
            raise ValueError("empirical canary data manifest mismatch")
        for field in (
            "environment_hash", "device_execution_plan_hash", "host_identity_digest",
            "checkpoint_receipt_hash", "telemetry_evidence_hash", "evaluation_receipt_hash",
        ):
            value = str(getattr(self, field))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"empirical canary {field} must be a SHA-256 hash")
        if len(self.event_chain_hash) != 64:
            raise ValueError("empirical canary event chain hash is invalid")
        if self.gpu_peak_memory_bytes <= 0 or min(
            self.checkpoint_duration_seconds,
            self.persistent_sync_duration_seconds,
            self.stop_latency_seconds,
        ) <= 0:
            raise ValueError("empirical canary runtime telemetry is incomplete")
        if self.dataloader_workers < 0 or self.prefetch_factor <= 0:
            raise ValueError("empirical canary prefetch telemetry is invalid")
        if self.trained_steps <= 0 or self.evaluation_batches <= 0 or not self.stop_resume_verified:
            raise ValueError("empirical canary must train, evaluate, and prove stop/resume")
        if type(self.evaluation_loss_tokens) is not int or self.evaluation_loss_tokens <= 0:
            raise ValueError("empirical canary evaluation loss-token count is invalid")
        if type(self.evaluation_loss) not in (int, float) or not math.isfinite(float(self.evaluation_loss)) or self.evaluation_loss < 0:
            raise ValueError("empirical canary evaluation loss is invalid")
        if not 0 <= minimum_gpu_utilization <= 100 or self.median_gpu_utilization < minimum_gpu_utilization:
            raise ValueError("empirical canary GPU utilization is below its required threshold")
        if not 0 <= maximum_input_wait_fraction <= 1 or not 0 <= self.input_wait_fraction <= maximum_input_wait_fraction:
            raise ValueError("empirical canary input-wait fraction exceeds its required threshold")


@dataclass(frozen=True, slots=True)
class StaticPreflightReceipt:
    run_identity: str
    data_manifest_hash: str
    environment_hash: str
    data_validation_hash: str
    feed_benchmark_hash: str
    max_canary_steps: int
    max_canary_seconds: int
    receipt_hash: str

    @classmethod
    def create(cls, **values: Any) -> "StaticPreflightReceipt":
        payload = {"artifact_kind": "twc_pretraining_static_preflight_v1", "schema_version": 1, **values}
        return cls(**values, receipt_hash=stable_json_hash(payload))

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_kind": "twc_pretraining_static_preflight_v1", "schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StaticPreflightReceipt":
        return _strict_dataclass_artifact(cls, payload, "twc_pretraining_static_preflight_v1")

    def validate(self, run_identity: str, data_manifest_hash: str) -> None:
        unhashed = {key: value for key, value in self.to_dict().items() if key != "receipt_hash"}
        if self.receipt_hash != stable_json_hash(unhashed):
            raise ValueError("static preflight receipt hash mismatch")
        if self.run_identity != run_identity or self.data_manifest_hash != data_manifest_hash:
            raise ValueError("static preflight run/data binding mismatch")
        for field in ("environment_hash", "data_validation_hash", "feed_benchmark_hash"):
            if len(str(getattr(self, field))) != 64:
                raise ValueError(f"static preflight {field} is invalid")
        if not 1 <= self.max_canary_steps <= 100 or not 1 <= self.max_canary_seconds <= 1800:
            raise ValueError("static preflight canary bound exceeds hard limit")


@dataclass(frozen=True, slots=True)
class BackupReceipt:
    run_identity: str
    checkpoint_generation: int
    checkpoint_receipt_hash: str
    source_inventory_hash: str
    destination_inventory_hash: str
    source_content_root: str
    destination_readback_root: str
    restore_probe_hash: str
    source_storage_identity_hash: str
    destination_storage_identity_hash: str
    source_root_binding_hash: str
    destination_root_binding_hash: str
    source_file_count: int
    destination_file_count: int
    source_bytes: int
    destination_bytes: int
    receipt_hash: str

    @classmethod
    def create(cls, **values: Any) -> "BackupReceipt":
        payload = {"artifact_kind": "twc_pretraining_backup_receipt_v1", "schema_version": 1, **values}
        return cls(**values, receipt_hash=stable_json_hash(payload))

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_kind": "twc_pretraining_backup_receipt_v1", "schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BackupReceipt":
        return _strict_dataclass_artifact(cls, payload, "twc_pretraining_backup_receipt_v1")

    def validate(self) -> None:
        unhashed = {key: value for key, value in self.to_dict().items() if key != "receipt_hash"}
        if self.receipt_hash != stable_json_hash(unhashed):
            raise ValueError("backup receipt hash mismatch")
        for field in (
            "run_identity", "checkpoint_receipt_hash", "source_inventory_hash", "destination_inventory_hash",
            "source_content_root", "destination_readback_root", "restore_probe_hash",
            "source_storage_identity_hash", "destination_storage_identity_hash",
            "source_root_binding_hash", "destination_root_binding_hash",
        ):
            value = str(getattr(self, field))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"backup {field} must be a SHA-256 hash")
        if self.source_file_count <= 0 or self.source_file_count != self.destination_file_count:
            raise ValueError("backup source/destination file counts differ")
        if self.source_bytes <= 0 or self.source_bytes != self.destination_bytes:
            raise ValueError("backup source/destination byte counts differ")
        if self.source_content_root != self.destination_readback_root:
            raise ValueError("backup destination readback does not match source content root")
        if self.source_storage_identity_hash == self.destination_storage_identity_hash:
            raise ValueError("backup destination storage identity must differ from source")


@dataclass(frozen=True, slots=True)
class GuardPolicy:
    price_snapshot_hash: str
    gpu_hourly_price_usd: float
    max_spend_usd: float
    stop_reserve_seconds: int
    min_free_bytes: int
    stall_warning_seconds: int
    stall_stop_seconds: int
    evidence_max_age_seconds: int
    policy_hash: str

    @classmethod
    def create(cls, **values: Any) -> "GuardPolicy":
        payload = {"artifact_kind": "twc_pretraining_guard_policy_v1", "schema_version": 1, **values}
        return cls(**values, policy_hash=stable_json_hash(payload))

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_kind": "twc_pretraining_guard_policy_v1", "schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GuardPolicy":
        return _strict_dataclass_artifact(cls, payload, "twc_pretraining_guard_policy_v1")

    def validate(self) -> None:
        unhashed = {key: value for key, value in self.to_dict().items() if key != "policy_hash"}
        if self.policy_hash != stable_json_hash(unhashed):
            raise ValueError("guard policy hash mismatch")
        if len(self.price_snapshot_hash) != 64:
            raise ValueError("guard price snapshot hash is invalid")
        if self.gpu_hourly_price_usd <= 0 or self.max_spend_usd <= 0 or self.stop_reserve_seconds <= 0:
            raise ValueError("guard cost values must be positive")
        if self.min_free_bytes <= 0:
            raise ValueError("guard storage reserve must be positive and derived from the storage layout")
        if self.stall_warning_seconds <= 0 or self.stall_stop_seconds <= self.stall_warning_seconds or self.evidence_max_age_seconds <= 0:
            raise ValueError("guard stall thresholds are invalid")


@dataclass(frozen=True, slots=True)
class GuardEvidence:
    policy_hash: str
    cumulative_billed_seconds: float
    free_bytes: int
    heartbeat_age_seconds: float
    throughput_loss_tokens_per_second: float
    input_wait_fraction: float
    observed_at_epoch: float
    evidence_hash: str

    @classmethod
    def create(cls, **values: Any) -> "GuardEvidence":
        payload = {"artifact_kind": "twc_pretraining_guard_evidence_v1", "schema_version": 1, **values}
        return cls(**values, evidence_hash=stable_json_hash(payload))

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_kind": "twc_pretraining_guard_evidence_v1", "schema_version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GuardEvidence":
        return _strict_dataclass_artifact(cls, payload, "twc_pretraining_guard_evidence_v1")

    def validate(
        self,
        policy: GuardPolicy,
        *,
        now_epoch: float | None = None,
        enforce_accelerator_performance: bool = True,
    ) -> None:
        unhashed = {key: value for key, value in self.to_dict().items() if key != "evidence_hash"}
        if self.evidence_hash != stable_json_hash(unhashed) or self.policy_hash != policy.policy_hash:
            raise ValueError("guard evidence hash or policy binding mismatch")
        spend = self.cumulative_billed_seconds / 3600 * policy.gpu_hourly_price_usd
        reserve = policy.stop_reserve_seconds / 3600 * policy.gpu_hourly_price_usd
        if spend + reserve > policy.max_spend_usd:
            raise ValueError("guard spend plus stop reserve exceeds cap")
        if self.free_bytes < policy.min_free_bytes:
            raise ValueError("guard free storage is below reserve")
        if self.heartbeat_age_seconds > policy.stall_warning_seconds or self.throughput_loss_tokens_per_second <= 0:
            raise ValueError("guard heartbeat or throughput is unhealthy")
        if not 0 <= self.input_wait_fraction <= 1:
            raise ValueError("guard input-wait fraction must be in [0, 1]")
        if enforce_accelerator_performance and self.input_wait_fraction > 0.10:
            raise ValueError("guard input-wait fraction exceeds 10 percent")
        now = time.time() if now_epoch is None else now_epoch
        if self.observed_at_epoch <= 0 or now - self.observed_at_epoch > policy.evidence_max_age_seconds or self.observed_at_epoch > now + 5:
            raise ValueError("guard evidence is stale or future-dated")


def _strict_dataclass_artifact(cls, payload: dict[str, Any], kind: str):
    fields = set(cls.__dataclass_fields__)
    metadata = {"artifact_kind", "schema_version"}
    unknown, missing = set(payload) - fields - metadata, fields - set(payload)
    if unknown or missing:
        raise ValueError(f"{kind} fields invalid; unknown={sorted(unknown)} missing={sorted(missing)}")
    if payload.get("artifact_kind") != kind or payload.get("schema_version") != 1:
        raise ValueError(f"unsupported {kind} artifact")
    return cls(**{field: payload[field] for field in fields})


def _checkpoint_inventory(generation_root: Path) -> tuple[list[dict[str, Any]], int]:
    required = ("state.pt", "manifest.json", "READY.json")
    rows: list[dict[str, Any]] = []
    total = 0
    for name in required:
        path = generation_root / name
        if not path.is_file():
            raise ValueError(f"backup checkpoint is missing {name}")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        rows.append({"name": name, "bytes": size, "sha256": digest.hexdigest()})
        total += size
    return rows, total


def build_backup_receipt(
    *,
    checkpoint_manager: CheckpointManager,
    generation: int,
    destination_root: Path,
) -> BackupReceipt:
    source = checkpoint_manager.root / f"generation-{generation:08d}"
    destination_root = destination_root.resolve()
    source_root = checkpoint_manager.root.resolve()
    if destination_root == source_root or destination_root in source_root.parents or source_root in destination_root.parents:
        raise ValueError("backup destination must be a separate non-nested root")
    source_storage_identity_hash = _storage_identity_hash(source_root)
    destination_storage_identity_hash = _storage_identity_hash(destination_root)
    if source_storage_identity_hash == destination_storage_identity_hash:
        raise ValueError("backup destination must be on a distinct storage device")
    source_loaded = checkpoint_manager.validate_reference(
        CheckpointReference(generation, source, "persistent_verified")
    )
    destination_generation = destination_root / source.name
    destination_manager = CheckpointManager(destination_root)
    destination_loaded = destination_manager.validate_reference(
        CheckpointReference(generation, destination_generation, "local_verified")
    )
    if (
        destination_loaded.run_identity != source_loaded.run_identity
        or destination_loaded.step != source_loaded.step
        or destination_loaded.next_global_ordinal != source_loaded.next_global_ordinal
        or destination_loaded.consumed_loss_tokens != source_loaded.consumed_loss_tokens
    ):
        raise ValueError("backup restore probe differs from source checkpoint")
    source_rows, source_bytes = _checkpoint_inventory(source)
    destination_rows, destination_bytes = _checkpoint_inventory(destination_generation)
    source_root_hash = stable_json_hash(source_rows)
    destination_root_hash = stable_json_hash(destination_rows)
    persistent = json.loads((source / "PERSISTENT.json").read_text(encoding="utf-8"))
    return BackupReceipt.create(
        run_identity=source_loaded.run_identity,
        checkpoint_generation=generation,
        checkpoint_receipt_hash=stable_json_hash(persistent),
        source_inventory_hash=stable_json_hash({"files": source_rows, "bytes": source_bytes}),
        destination_inventory_hash=stable_json_hash({"files": destination_rows, "bytes": destination_bytes}),
        source_content_root=source_root_hash,
        destination_readback_root=destination_root_hash,
        restore_probe_hash=stable_json_hash(
            {
                "run_identity": destination_loaded.run_identity,
                "step": destination_loaded.step,
                "consumed_loss_tokens": destination_loaded.consumed_loss_tokens,
                "next_global_ordinal": destination_loaded.next_global_ordinal,
            }
        ),
        source_storage_identity_hash=source_storage_identity_hash,
        destination_storage_identity_hash=destination_storage_identity_hash,
        source_root_binding_hash=stable_json_hash({"resolved_root": str(source_root)}),
        destination_root_binding_hash=stable_json_hash({"resolved_root": str(destination_root)}),
        source_file_count=len(source_rows),
        destination_file_count=len(destination_rows),
        source_bytes=source_bytes,
        destination_bytes=destination_bytes,
    )


def _validate_live_backup_destination(receipt: BackupReceipt, destination_root: Path) -> None:
    receipt.validate()
    generation_root = destination_root.resolve() / f"generation-{receipt.checkpoint_generation:08d}"
    if stable_json_hash({"resolved_root": str(destination_root.resolve())}) != receipt.destination_root_binding_hash:
        raise ValueError("live backup destination root binding drifted")
    if _storage_identity_hash(destination_root.resolve()) != receipt.destination_storage_identity_hash:
        raise ValueError("live backup destination storage identity drifted")
    rows, total = _checkpoint_inventory(generation_root)
    if stable_json_hash({"files": rows, "bytes": total}) != receipt.destination_inventory_hash:
        raise ValueError("live backup destination inventory drifted")
    if stable_json_hash(rows) != receipt.destination_readback_root:
        raise ValueError("live backup destination content root drifted")
    manager = CheckpointManager(destination_root.resolve())
    loaded = manager.validate_reference(
        CheckpointReference(receipt.checkpoint_generation, generation_root, "local_verified")
    )
    probe = stable_json_hash(
        {
            "run_identity": loaded.run_identity,
            "step": loaded.step,
            "consumed_loss_tokens": loaded.consumed_loss_tokens,
            "next_global_ordinal": loaded.next_global_ordinal,
        }
    )
    if probe != receipt.restore_probe_hash or loaded.run_identity != receipt.run_identity:
        raise ValueError("live backup destination restore probe failed")


def _storage_identity_hash(root: Path) -> str:
    return stable_json_hash({"st_dev": root.stat().st_dev})


_TRANSITIONS = {
    "PLANNED": {"PREFLIGHTED"},
    "PREFLIGHTED": {"STARTING"},
    "STARTING": {"RUNNING", "STOPPED", "FAILED", "ORPHANED"},
    "RUNNING": {"QUIESCING", "STOPPING", "COMPLETED", "FAILED", "ORPHANED"},
    "QUIESCING": {"STOPPING", "STOPPED", "COMPLETED", "FAILED", "ORPHANED"},
    "STOPPING": {"STOPPED", "FAILED"},
    "STOPPED": {"STARTING", "FAILED"},
    "ORPHANED": {"STOPPED", "FAILED"},
    "COMPLETED": set(),
    "FAILED": {"PREFLIGHTED"},
}
_RUN_MODES = frozenset({"run"})
_LAUNCH_FIELDS = {
    "start_command", "start_failure", "active_mode", "active_attempt_started_epoch",
    "process_identity", "watchdog_process_identity", "worker_ready_receipt",
    "watchdog_ready_receipt", "active_launch_id", "startup_orphan_processes",
    "launch_history",
}


def validate_runtime_safety_settings(payload: object) -> dict[str, int]:
    if not isinstance(payload, dict) or set(payload) != {
        "heartbeat_timeout_seconds",
        "cooperative_stop_grace_seconds",
    }:
        raise ValueError("runtime safety settings are not canonical")
    if any(
        type(payload[field]) is not int or not 1 <= payload[field] <= 86_400
        for field in payload
    ):
        raise ValueError("runtime safety timeout or stop grace is invalid")
    return {field: int(payload[field]) for field in sorted(payload)}


def _require_launch_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("launch command ID must be exactly 32 lowercase hexadecimal characters")
    return value


class TrainingOperator:
    """Own one compact state file; never scans logs/checkpoints for status."""

    def __init__(self, root: Path, *, checkpoint_manager: CheckpointManager | None = None) -> None:
        self.root = root
        self.state_path = root / "state.json"
        self.contract_path = root / "run_contract.json"
        self.attempts_path = root / "attempts.jsonl"
        self.lease_path = root / "writer.lease"
        self._owns_lease = False
        self._lease_id: str | None = None
        self.checkpoint_manager = checkpoint_manager

    def plan(self, contract: RunContract, attempt: ExecutionAttempt) -> dict[str, Any]:
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)
        if not self._owns_lease:
            self._acquire_writer()
        if self.state_path.exists():
            self.release_writer()
            raise FileExistsError("operator run is already planned")
        if attempt.run_identity != contract.identity:
            self.release_writer()
            raise ValueError("attempt does not bind the run contract")
        contract_payload = {**contract.to_dict(), "run_identity": contract.identity}
        if self.contract_path.exists():
            if stable_json_hash(
                json.loads(self.contract_path.read_text(encoding="utf-8"))
            ) != stable_json_hash(contract_payload):
                self.release_writer()
                raise ValueError("interrupted plan has a conflicting run contract")
        else:
            self._atomic_json(self.contract_path, contract_payload)
        if self.attempts_path.exists():
            attempts = self._attempt_chain()
            if len(attempts) != 1 or attempts[0] != attempt:
                self.release_writer()
                raise ValueError("interrupted plan has a conflicting execution attempt")
        else:
            self._append_attempt(attempt)
        state = {
            "artifact_kind": "twc_pretraining_operator_state_v2",
            "schema_version": 2,
            "run_identity": contract.identity,
            "attempt_hash": attempt.identity,
            "state": "PLANNED",
            "generation": 0,
            "fencing_epoch": attempt.fencing_epoch,
            "step": 0,
            "next_global_ordinal": 0,
            "checkpoint_generation": None,
            "checkpoint_durability": "none",
            "process_identity": None,
            "watchdog_process_identity": None,
            "worker_ready_receipt": None,
            "watchdog_ready_receipt": None,
            "active_launch_id": None,
            "active_mode": None,
            "active_attempt_started_epoch": None,
            "start_command": None,
            "start_failure": None,
            "startup_orphan_processes": None,
            "launch_history": [],
            "backup_receipt": None,
            "backup_destination_root": None,
        }
        self._write_state(state)
        return dict(state)

    def prepare_runtime(
        self,
        *,
        runtime_safety: dict[str, int],
        storage_layout_receipt: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a generic run startable from real liveness and storage facts."""

        self._require_writer()
        state = self._read_state()
        if state["state"] != "PLANNED":
            raise ValueError("runtime preparation requires PLANNED state")
        checkpoint_filesystem_requirements(storage_layout_receipt)
        attempt = self._attempt_chain()[-1]
        if attempt.identity != state["attempt_hash"]:
            raise ValueError("runtime preparation attempt binding mismatch")
        if attempt.storage_layout_hash != storage_layout_receipt["receipt_hash"]:
            raise ValueError("runtime preparation storage binding mismatch")
        state["runtime_safety"] = validate_runtime_safety_settings(runtime_safety)
        return self._transition(state, "PREFLIGHTED")

    def reserve_start(self, *, command_id: str, mode: str) -> dict[str, Any]:
        """Reserve STARTING durably before a launcher creates the worker."""
        self._require_writer()
        state = self._read_state()
        self._require_launch_state_shape(state)
        _require_launch_id(command_id)
        if mode not in _RUN_MODES:
            raise ValueError("start mode must be run")
        existing = state.get("start_command")
        if existing is not None:
            if existing.get("command_id") == command_id and existing.get("mode") == mode:
                return state
            raise ValueError("another start command is already reserved")
        self._clear_current_launch(state, clear_mode=False)
        state["start_failure"] = None
        state["start_command"] = {
            "command_id": command_id,
            "mode": mode,
            "kind": "start",
            "run_identity": state["run_identity"],
            "attempt_hash": state["attempt_hash"],
            "parent_attempt_hash": None,
            "fencing_epoch": int(state.get("fencing_epoch", 0)),
        }
        state["active_mode"] = mode
        return self._transition(state, "STARTING")

    def reserve_resume(
        self,
        *,
        command_id: str,
        attempt: ExecutionAttempt,
        mode: str,
        guard_evidence: GuardEvidence | None = None,
    ) -> dict[str, Any]:
        """Bind the successor attempt before spawning so READY can be exact."""
        self._require_writer()
        state = self._read_state()
        self._require_launch_state_shape(state)
        _require_launch_id(command_id)
        if attempt.run_identity != state["run_identity"]:
            raise ValueError("resume attempt does not bind operator run")
        if attempt.fencing_epoch != int(state.get("fencing_epoch", 0)):
            raise ValueError("resume attempt fencing epoch does not own the operator")
        if mode not in _RUN_MODES:
            raise ValueError("resume mode must be run")
        if state.get("runtime_safety") is None:
            raise ValueError("legacy preflighted attempts are read-only")
        validate_runtime_safety_settings(state["runtime_safety"])
        if guard_evidence is not None:
            raise ValueError("generic resume does not accept guard evidence")
        guard_payload = None
        reservation = {
            "command_id": command_id,
            "mode": mode,
            "kind": "resume",
            "run_identity": attempt.run_identity,
            "attempt_hash": attempt.identity,
            "parent_attempt_hash": attempt.parent_attempt_hash,
            "fencing_epoch": attempt.fencing_epoch,
        }
        attempts = self._attempt_chain()
        if state["state"] == "STARTING":
            if (
                state.get("start_command") == reservation
                and state.get("attempt_hash") == attempt.identity
                and state.get("active_mode") == mode
                and state.get("guard_evidence") == guard_payload
                and attempts[-1] == attempt
            ):
                return state
            raise ValueError("resume reservation conflicts with active STARTING state")
        failure = state.get("start_failure")
        if state["state"] == "STOPPED" and isinstance(failure, dict) and all(
            failure.get(key) == value
            for key, value in {
                "command_id": command_id,
                "kind": "resume",
                "attempt_hash": attempt.identity,
            }.items()
        ):
            if (
                state.get("attempt_hash") == attempt.identity
                and state.get("active_mode") == mode
                and state.get("guard_evidence") == guard_payload
                and attempts[-1] == attempt
            ):
                return state
            raise ValueError("failed resume replay conflicts with durable reservation evidence")
        if state["state"] != "STOPPED" or state.get("checkpoint_durability") == "none":
            raise ValueError("resume reservation requires a verified stopped checkpoint")
        state_attempt_hash = state.get("attempt_hash")
        if attempts[-1].identity == state_attempt_hash:
            attempt.validate_successor(attempts[-1])
            self._append_attempt(attempt)
        elif (
            attempts[-1] == attempt
            and len(attempts) >= 2
            and attempts[-2].identity == state_attempt_hash
        ):
            attempt.validate_successor(attempts[-2])
        else:
            raise ValueError("execution-attempt ledger and stopped operator state diverged")
        self._clear_current_launch(state, clear_mode=False)
        state.update(
            attempt_hash=attempt.identity,
            active_mode=mode,
            start_command=reservation,
        )
        state["start_failure"] = None
        return self._transition(state, "STARTING")

    def confirm_start(
        self,
        *,
        command_id: str,
        command_kind: str,
        worker_process_identity: dict[str, Any],
        watchdog_process_identity: dict[str, Any],
        worker_ready_receipt: dict[str, Any],
        watchdog_ready_receipt: dict[str, Any],
    ) -> dict[str, Any]:
        self._require_writer()
        _require_launch_id(command_id)
        if command_kind not in {"start", "resume"}:
            raise ValueError("start confirmation command kind is invalid")
        self._validate_process_identity(worker_process_identity)
        self._validate_process_identity(watchdog_process_identity)
        worker_ready = load_worker_ready(worker_ready_receipt)
        watchdog_ready = load_watchdog_ready(watchdog_ready_receipt)
        state = self._read_state()
        self._require_launch_state_shape(state)
        attempts = self._attempt_chain()
        self._validate_launch_history(state, attempts)
        replay = self._confirmed_launch_replay(
            state=state, command_id=command_id, command_kind=command_kind,
            worker_identity=worker_process_identity,
            watchdog_identity=watchdog_process_identity,
            worker_ready=worker_ready, watchdog_ready=watchdog_ready,
        )
        if replay:
            return state
        reserved = state.get("start_command")
        if state.get("state") != "STARTING" or not isinstance(reserved, dict):
            raise ValueError("start confirmation requires STARTING state")
        attempt = attempts[-1]
        expected_reservation = {
            "command_id": command_id, "kind": command_kind,
            "run_identity": state["run_identity"], "attempt_hash": attempt.identity,
            "parent_attempt_hash": attempt.parent_attempt_hash,
            "fencing_epoch": attempt.fencing_epoch,
        }
        if any(reserved.get(key) != value for key, value in expected_reservation.items()):
            raise ValueError("start confirmation reservation mismatch")
        if worker_process_identity["pid"] == watchdog_process_identity["pid"] or (
            worker_process_identity["process_group_id"] == watchdog_process_identity["process_group_id"]
        ):
            raise ValueError("worker and watchdog identities must be distinct")
        expected_identity = {
            "worker": ("r7-pretraining-worker", "trainer", worker_ready),
            "watchdog": ("r7-pretraining-watchdog", "guard", watchdog_ready),
        }
        for role, identity in (("worker", worker_process_identity), ("watchdog", watchdog_process_identity)):
            label, canonical_role, ready = expected_identity[role]
            if (
                identity["process_group_id"] != identity["pid"]
                or identity["attempt_id"] != attempt.identity
                or identity["label"] != label
                or identity["role"] != canonical_role
                or ready["process_pid"] != identity["pid"]
                or ready["process_start_token"] != identity["process_start_token"]
            ):
                raise ValueError(f"{role} process/READY identity mismatch")
        expected_worker = {
            "run_identity": attempt.run_identity, "attempt_hash": attempt.identity,
            "fencing_epoch": attempt.fencing_epoch, "release_hash": attempt.release_hash,
            "environment_hash": attempt.environment_hash,
            "data_placement_hash": attempt.data_placement_hash,
            "storage_layout_hash": attempt.storage_layout_hash,
        }
        mismatches = [key for key, value in expected_worker.items() if worker_ready.get(key) != value]
        if mismatches:
            raise ValueError(f"worker READY attempt binding mismatch: {sorted(mismatches)}")
        for key, value in {
            "launch_id": command_id, "run_identity": attempt.run_identity,
            "attempt_hash": attempt.identity, "fencing_epoch": attempt.fencing_epoch,
            "release_hash": attempt.release_hash,
        }.items():
            if watchdog_ready.get(key) != value or worker_ready.get(key) != value:
                raise ValueError(f"dual READY {key} mismatch")
        if watchdog_ready["ready_at_epoch"] > worker_ready["ready_at_epoch"]:
            raise ValueError("watchdog READY must precede worker READY")
        policy_hash = self._current_performance_policy_hash()
        if worker_ready["performance_policy_hash"] != policy_hash or watchdog_ready["performance_policy_hash"] != policy_hash:
            raise ValueError("dual READY performance policy mismatch")
        if observe_process(worker_process_identity) != "confirmed_alive" or observe_process(watchdog_process_identity) != "confirmed_alive":
            raise ValueError("both exact startup processes must be alive at confirmation")
        activated_at = time.time()
        entry = {
            "schema_version": 1, "launch_id": command_id, "command_kind": command_kind,
            "run_identity": attempt.run_identity, "attempt_hash": attempt.identity,
            "fencing_epoch": attempt.fencing_epoch, "activated_at_epoch": activated_at,
            "worker_process_identity": worker_process_identity,
            "watchdog_process_identity": watchdog_process_identity,
            "worker_ready_receipt": worker_ready, "watchdog_ready_receipt": watchdog_ready,
        }
        entry["activation_hash"] = stable_json_hash(entry)
        state["process_identity"] = worker_process_identity
        state["watchdog_process_identity"] = watchdog_process_identity
        state["worker_ready_receipt"] = worker_ready
        state["watchdog_ready_receipt"] = watchdog_ready
        state["active_launch_id"] = command_id
        state["active_attempt_started_epoch"] = activated_at
        state["launch_history"] = [*state["launch_history"], entry]
        state["start_command"] = None
        state["start_failure"] = None
        return self._transition(state, "RUNNING")

    def fail_start(
        self,
        *,
        command_id: str,
        failure_code: str,
        child_role: str,
        exit_code: int | None = None,
    ) -> dict[str, Any]:
        self._require_writer()
        state = self._read_state()
        self._require_launch_state_shape(state)
        reserved = state.get("start_command")
        if state.get("state") != "STARTING" or not isinstance(reserved, dict):
            raise ValueError("start failure requires STARTING state")
        if reserved.get("command_id") != command_id:
            raise ValueError("start failure command id mismatch")
        kind = reserved.get("kind")
        state["start_failure"] = self._start_failure_payload(
            command_id=command_id, kind=kind, attempt_hash=state.get("attempt_hash"),
            failure_code=failure_code, child_role=child_role, exit_code=exit_code,
        )
        self._clear_current_launch(state)
        return self._transition(state, "STOPPED" if kind == "resume" else "FAILED")

    def request_action(
        self, request_id: str, *, action: str, deadline_seconds: int
    ) -> dict[str, Any]:
        """Publish an immutable, attempt-bound control request and enter QUIESCING."""
        acquired = not self._owns_lease
        if acquired:
            self.claim_writer()
        try:
            return self._request_action_locked(
                request_id, action=action, deadline_seconds=deadline_seconds
            )
        finally:
            if acquired:
                self.release_writer()

    def _request_action_locked(
        self, request_id: str, *, action: str, deadline_seconds: int
    ) -> dict[str, Any]:
        self._require_writer()
        if not request_id or len(request_id) > 128 or any(c in "/\\\x00" for c in request_id):
            raise ValueError("control request id is invalid")
        if action not in {"checkpoint", "pause", "stop", "emergency_stop"}:
            raise ValueError("unsupported control action")
        if type(deadline_seconds) is not int or not 1 <= deadline_seconds <= 86400:
            raise ValueError("control request deadline is invalid")
        state = self._read_state()
        if state.get("state") not in {"RUNNING", "QUIESCING"}:
            raise ValueError("control request requires RUNNING or QUIESCING state")
        requests = self.root / "requests"
        requests.mkdir(mode=0o700, exist_ok=True)
        path = requests / f"{request_id}.json"
        intent = {
            "artifact_kind": "twc_pretraining_control_request_v1",
            "schema_version": 1,
            "request_id": request_id,
            "action": action,
            "deadline_seconds": deadline_seconds,
            "run_identity": state["run_identity"],
            "attempt_hash": state["attempt_hash"],
            "fencing_epoch": int(state.get("fencing_epoch", 0)),
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_intent = {
                key: value for key, value in existing.items() if key not in {"request_hash", "requested_at_epoch"}
            }
            if existing_intent != intent:
                raise ValueError("conflicting control request id reuse")
            return existing
        request = {**intent, "requested_at_epoch": time.time()}
        request["request_hash"] = stable_json_hash(request)
        encoded = (json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n").encode()
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return self.request_action(request_id, action=action, deadline_seconds=deadline_seconds)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_dir(requests)
        if action in {"pause", "stop", "emergency_stop"} and state["state"] == "RUNNING":
            state["active_request_id"] = request_id
            self._transition(state, "QUIESCING")
        return request

    def validate_fence(self, fencing_epoch: int) -> None:
        if type(fencing_epoch) is not int:
            raise ValueError("fencing epoch must be an integer")
        current = int(self._read_state().get("fencing_epoch", 0))
        if fencing_epoch != current:
            raise ValueError("fencing epoch does not own the active run")

    def mark_orphaned(
        self,
        *,
        reason: str,
        command_id: str | None = None,
        failure_code: str | None = None,
        child_role: str | None = None,
        exit_code: int | None = None,
        startup_orphan_processes: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        acquired = not self._owns_lease
        if acquired:
            self.claim_writer()
        try:
            state = self._read_state()
            if state.get("state") not in {"STARTING", "RUNNING", "QUIESCING"}:
                raise ValueError("only an active or starting run can become orphaned")
            if state["state"] == "STARTING":
                self._require_launch_state_shape(state)
                reserved = state.get("start_command")
                if reason != "startup_cleanup_uncertain":
                    raise ValueError("startup cleanup uncertainty requires its bounded reason code")
                if not isinstance(reserved, dict) or reserved.get("command_id") != command_id:
                    raise ValueError("startup orphan command does not match the reservation")
                state["start_failure"] = self._start_failure_payload(
                    command_id=command_id, kind=reserved.get("kind"),
                    attempt_hash=state.get("attempt_hash"), failure_code=failure_code,
                    child_role=child_role, exit_code=exit_code,
                )
                state["startup_orphan_processes"] = self._validate_startup_orphan_processes(
                    startup_orphan_processes, attempt_hash=str(state["attempt_hash"])
                )
                state["orphaned_reason"] = reason
            else:
                if any(
                    value is not None
                    for value in (
                        command_id, failure_code, child_role, exit_code,
                        startup_orphan_processes,
                    )
                ):
                    raise ValueError("runtime orphaning cannot carry startup-only evidence")
                state["orphaned_reason"] = str(reason)[:500]
            return self._transition(state, "ORPHANED")
        finally:
            if acquired:
                self.release_writer()

    def takeover(self, *, command_id: str, former_host_terminated: bool) -> dict[str, Any]:
        acquired = not self._owns_lease
        if acquired:
            self.claim_writer()
        try:
            state = self._read_state()
            if former_host_terminated is not True:
                raise ValueError("takeover requires proof the former host is terminated")
            existing = state.get("takeover_command")
            if existing is not None:
                if existing.get("command_id") != command_id:
                    raise ValueError("conflicting takeover command id")
                return state
            if state.get("state") != "ORPHANED":
                raise ValueError("takeover requires ORPHANED state")
            self._require_launch_state_shape(state)
            reserved = state.get("start_command")
            startup_kind: str | None = None
            if reserved is not None:
                if not isinstance(reserved, dict) or reserved.get("kind") not in {"start", "resume"}:
                    raise ValueError("startup orphan reservation is invalid")
                startup_kind = str(reserved["kind"])
                failure = state.get("start_failure")
                if (
                    not isinstance(failure, dict)
                    or failure.get("command_id") != reserved.get("command_id")
                    or failure.get("kind") != startup_kind
                    or failure.get("attempt_hash") != state.get("attempt_hash")
                ):
                    raise ValueError("startup orphan failure evidence is invalid")
                if startup_kind == "start" and (
                    int(state.get("step", 0)) != 0
                    or int(state.get("next_global_ordinal", 0)) != 0
                    or state.get("checkpoint_generation") is not None
                    or state.get("checkpoint_durability", "none") != "none"
                ):
                    raise ValueError("fresh-start orphan unexpectedly carries checkpoint progress")
                if startup_kind == "resume" and state.get("checkpoint_durability") == "none":
                    raise ValueError("resume orphan lost predecessor checkpoint evidence")
            state["fencing_epoch"] = int(state.get("fencing_epoch", 0)) + 1
            state["takeover_command"] = {
                "command_id": command_id,
                "former_host_terminated": True,
            }
            self._clear_current_launch(state)
            return self._transition(state, "FAILED" if startup_kind == "start" else "STOPPED")
        finally:
            if acquired:
                self.release_writer()

    def pause(self, *, step: int, next_global_ordinal: int, checkpoint: CheckpointReference) -> dict[str, Any]:
        self._require_writer()
        loaded = self._validate_checkpoint(checkpoint)
        self._validate_checkpoint_progress(loaded, step, next_global_ordinal)
        return self._stop_common(
            step=step,
            next_global_ordinal=next_global_ordinal,
            checkpoint_generation=checkpoint.generation,
            durability=checkpoint.durability,
        )

    def stop(
        self,
        *,
        step: int,
        next_global_ordinal: int,
        checkpoint: CheckpointReference,
    ) -> dict[str, Any]:
        self._require_writer()
        loaded = self._validate_checkpoint(checkpoint)
        self._validate_checkpoint_progress(loaded, step, next_global_ordinal)
        if checkpoint.durability != "persistent_verified":
            raise ValueError("stop requires a persistent verified checkpoint")
        return self._stop_common(
            step=step,
            next_global_ordinal=next_global_ordinal,
            checkpoint_generation=checkpoint.generation,
            durability="persistent_verified",
        )

    def checkpoint(
        self, *, step: int, next_global_ordinal: int, checkpoint: CheckpointReference
    ) -> dict[str, Any]:
        self._require_writer()
        loaded = self._validate_checkpoint(checkpoint)
        self._validate_checkpoint_progress(loaded, step, next_global_ordinal)
        state = self._read_state()
        if state["state"] != "RUNNING":
            raise ValueError("checkpoint requires RUNNING state")
        committed_generation = state.get("checkpoint_generation")
        if (
            type(committed_generation) is int
            and committed_generation >= checkpoint.generation
        ):
            if committed_generation == checkpoint.generation and (
                state["step"] != step
                or state["next_global_ordinal"] != next_global_ordinal
                or state["checkpoint_durability"] != checkpoint.durability
            ):
                raise ValueError("checkpoint generation has conflicting committed identity")
            if (
                step > state["step"]
                or next_global_ordinal > state["next_global_ordinal"]
            ):
                raise ValueError("checkpoint generation cannot move backward")
            return state
        self._validate_progress(state, step, next_global_ordinal)
        state.update(
            step=step,
            next_global_ordinal=next_global_ordinal,
            checkpoint_generation=checkpoint.generation,
            checkpoint_durability=checkpoint.durability,
        )
        self._write_state(state)
        return state

    def recover(self) -> dict[str, Any]:
        state = self._read_state()
        already_completed = state["state"] == "COMPLETED"
        if state["state"] in {"RUNNING", "QUIESCING"} or already_completed:
            if not already_completed:
                observation = observe_process(state["process_identity"])
                if observation == "confirmed_alive":
                    return state
                if observation != "exited":
                    raise RuntimeError("training worker ownership is uncertain; recovery fails closed")
            self._require_writer()
            try:
                result = json.loads(
                    (
                        self.root
                        / "attempt-outcomes"
                        / f"{state['attempt_hash']}.json"
                    ).read_text(encoding="utf-8")
                )
            except (FileNotFoundError, json.JSONDecodeError) as error:
                return self._recover_forced_stop(state, error)
            expected = stable_json_hash({key: value for key, value in result.items() if key != "result_hash"})
            if result.get("result_hash") != expected or result.get("run_identity") != state["run_identity"]:
                raise ValueError("training worker result binding mismatch")
            if result.get("attempt_hash") != state["attempt_hash"]:
                return self._recover_forced_stop(state, RuntimeError("stale worker result"))
            attempt = self._latest_attempt()
            if (
                attempt.identity != state["attempt_hash"]
                or attempt.run_identity != state["run_identity"]
                or result.get("fencing_epoch") != attempt.fencing_epoch
                or state.get("fencing_epoch") != attempt.fencing_epoch
            ):
                raise ValueError("terminal outcome fence or attempt binding is invalid")
            selected = attempt.selected_checkpoint
            expected_resume = {
                "resumed_from_attempt_hash": (
                    selected.attempt_hash if selected is not None else None
                ),
                "resumed_from_checkpoint_generation": (
                    selected.generation if selected is not None else None
                ),
                "resumed_from_step": selected.step if selected is not None else None,
                "resumed_from_loss_tokens": (
                    selected.consumed_loss_tokens if selected is not None else None
                ),
                "resumed_from_global_ordinal": (
                    selected.next_global_ordinal if selected is not None else None
                ),
            }
            if any(result.get(field) != value for field, value in expected_resume.items()):
                raise ValueError("terminal outcome resume lineage is invalid")
            generation = result.get("checkpoint_generation")
            if type(generation) is not int:
                raise RuntimeError("exited training worker has no checkpoint")
            if self.checkpoint_manager is None:
                raise RuntimeError("checkpoint manager is required for recovery")
            checkpoint_manifest_sha256 = result.get("checkpoint_manifest_sha256")
            checkpoint_state_sha256 = result.get("checkpoint_state_sha256")
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    checkpoint_manifest_sha256,
                    checkpoint_state_sha256,
                )
            ):
                raise ValueError(
                    "terminal checkpoint identity requires exact lowercase SHA-256 hashes"
                )
            reference = CheckpointReference(
                generation,
                self.checkpoint_manager.root / f"generation-{generation:08d}",
                str(result["checkpoint_durability"]),
                checkpoint_manifest_sha256,
                checkpoint_state_sha256,
            )
            loaded = self._validate_checkpoint(reference)
            self._validate_checkpoint_progress(loaded, int(result["step"]), int(result["next_global_ordinal"]))
            if (
                loaded.run_identity != state["run_identity"]
                or loaded.attempt_hash != attempt.identity
                or result.get("checkpoint_durability") != loaded.durability
                or result.get("step") != loaded.step
                or result.get("consumed_loss_tokens") != loaded.consumed_loss_tokens
                or result.get("next_global_ordinal") != loaded.next_global_ordinal
            ):
                raise ValueError("terminal outcome checkpoint or progress binding is invalid")
            if already_completed and (
                state.get("checkpoint_generation") != loaded.generation
                or state.get("checkpoint_durability") != loaded.durability
                or state.get("step") != loaded.step
                or state.get("next_global_ordinal") != loaded.next_global_ordinal
            ):
                raise ValueError("completed state and terminal checkpoint binding differ")
            if result.get("completed") is True:
                evaluation_path = (
                    self.root / f"worker-evaluations-{attempt.identity}.jsonl"
                )
                try:
                    evaluation_rows = [
                        json.loads(line)
                        for line in evaluation_path.read_text(encoding="utf-8").splitlines()
                        if line
                    ]
                except (FileNotFoundError, json.JSONDecodeError) as error:
                    raise ValueError("terminal outcome evaluation evidence is invalid") from error
                latest_evaluation = validate_evaluation_chain(
                    evaluation_rows,
                    state["run_identity"],
                    attempt.identity,
                    attempt.fencing_epoch,
                )
                if (
                    latest_evaluation.get("at_consumed_loss_tokens")
                    != loaded.consumed_loss_tokens
                    or result.get("evaluation_receipt_hash")
                    != latest_evaluation.get("receipt_hash")
                    or result.get("evaluation_loss") != latest_evaluation.get("loss")
                    or result.get("evaluation_loss_tokens")
                    != latest_evaluation.get("loss_tokens")
                    or result.get("evaluation_result_batch_count")
                    != latest_evaluation.get("batch_count")
                ):
                    raise ValueError("terminal outcome evaluation binding is invalid")
                contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
                if loaded.consumed_loss_tokens != int(contract["target_loss_tokens"]):
                    raise ValueError("completed worker did not consume the exact target")
            if already_completed:
                if result.get("completed") is not True:
                    raise ValueError("completed state lacks a completed terminal outcome")
                return state
            state.update(
                step=int(result["step"]),
                next_global_ordinal=int(result["next_global_ordinal"]),
                checkpoint_generation=generation,
                checkpoint_durability=reference.durability,
            )
            self._clear_current_launch(state)
            if result.get("completed") is True:
                return self._transition(state, "COMPLETED")
            state = self._transition(state, "STOPPING")
            return self._transition(state, "STOPPED")
        return state

    def _recover_forced_stop(self, state: dict[str, Any], cause: BaseException) -> dict[str, Any]:
        try:
            receipt = json.loads((self.root / "forced-stop.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            try:
                return self._recover_worker_failure(state)
            except (FileNotFoundError, json.JSONDecodeError) as failure_error:
                raise RuntimeError("exited training worker has no result, forced-stop, or failure receipt") from cause
        expected = stable_json_hash({key: value for key, value in receipt.items() if key != "receipt_hash"})
        if receipt.get("receipt_hash") != expected or receipt.get("process_identity_hash") != stable_json_hash(state["process_identity"]):
            raise ValueError("forced-stop receipt binding mismatch")
        if self.checkpoint_manager is None:
            raise RuntimeError("checkpoint manager is required for forced-stop recovery")
        loaded = self.checkpoint_manager.load_latest()
        attempt_hashes = {json.loads(line)["attempt_hash"] for line in self.attempts_path.read_text(encoding="utf-8").splitlines()}
        if loaded.run_identity != state["run_identity"] or loaded.attempt_hash not in attempt_hashes:
            raise ValueError("forced-stop fallback checkpoint is outside the run attempt chain")
        state.update(
            step=loaded.step,
            next_global_ordinal=loaded.next_global_ordinal,
            checkpoint_generation=loaded.generation,
            checkpoint_durability=loaded.durability,
            forced_stop_receipt=receipt,
        )
        self._clear_current_launch(state)
        state = self._transition(state, "STOPPING")
        return self._transition(state, "STOPPED")

    def _recover_worker_failure(self, state: dict[str, Any]) -> dict[str, Any]:
        receipt = json.loads((self.root / "worker-failure.json").read_text(encoding="utf-8"))
        expected = stable_json_hash({key: value for key, value in receipt.items() if key != "receipt_hash"})
        if (
            receipt.get("receipt_hash") != expected
            or receipt.get("run_identity") != state["run_identity"]
            or receipt.get("attempt_hash") != state["attempt_hash"]
            or receipt.get("fencing_epoch") != state.get("fencing_epoch")
        ):
            raise ValueError("worker failure receipt binding mismatch")
        state["worker_failure_receipt"] = receipt
        self._clear_current_launch(state)
        if self.checkpoint_manager is None:
            return self._transition(state, "FAILED")
        try:
            loaded = self.checkpoint_manager.load_latest()
        except RuntimeError:
            return self._transition(state, "FAILED")
        attempt_hashes = {
            json.loads(line)["attempt_hash"]
            for line in self.attempts_path.read_text(encoding="utf-8").splitlines()
        }
        if loaded.run_identity != state["run_identity"] or loaded.attempt_hash not in attempt_hashes:
            raise ValueError("worker failure fallback checkpoint is outside the run attempt chain")
        state.update(
            step=loaded.step,
            next_global_ordinal=loaded.next_global_ordinal,
            checkpoint_generation=loaded.generation,
            checkpoint_durability=loaded.durability,
        )
        state = self._transition(state, "STOPPING")
        return self._transition(state, "STOPPED")

    def status(self) -> dict[str, Any]:
        return self._read_state()

    def record_off_resource_backup(self, receipt: BackupReceipt, *, destination_root: Path) -> dict[str, Any]:
        self._require_writer()
        receipt.validate()
        state = self._read_state()
        if receipt.run_identity != state["run_identity"] or receipt.checkpoint_generation != state["checkpoint_generation"]:
            raise ValueError("backup receipt does not bind the active run/checkpoint")
        if self.checkpoint_manager is None:
            raise RuntimeError("checkpoint manager is required for backup receipt validation")
        if _storage_identity_hash(self.checkpoint_manager.root.resolve()) != receipt.source_storage_identity_hash:
            raise ValueError("backup source storage identity does not match checkpoint root")
        if stable_json_hash({"resolved_root": str(self.checkpoint_manager.root.resolve())}) != receipt.source_root_binding_hash:
            raise ValueError("backup source root binding does not match checkpoint root")
        persistent = json.loads(
            (self.checkpoint_manager.root / f"generation-{receipt.checkpoint_generation:08d}" / "PERSISTENT.json").read_text(encoding="utf-8")
        )
        if receipt.checkpoint_receipt_hash != stable_json_hash(persistent):
            raise ValueError("backup checkpoint receipt binding mismatch")
        _validate_live_backup_destination(receipt, destination_root)
        state["backup_receipt"] = receipt.to_dict()
        state["backup_destination_root"] = str(destination_root.resolve())
        self._write_state(state)
        return state

    def teardown_readiness(self, scope: str) -> dict[str, Any]:
        state = self._read_state()
        stopped = state["state"] in {"STOPPED", "COMPLETED"}
        if scope == "pod":
            ready = stopped and state["checkpoint_durability"] == "persistent_verified"
            reason = None if ready else "persistent verified checkpoint required"
        elif scope == "disk":
            ready = stopped and state["checkpoint_durability"] == "persistent_verified" and bool(state["backup_receipt"])
            if ready:
                try:
                    _validate_live_backup_destination(
                        BackupReceipt.from_dict(state["backup_receipt"]), Path(state["backup_destination_root"])
                    )
                except (OSError, ValueError, RuntimeError):
                    ready = False
            reason = None if ready else "off-resource verified backup required"
        else:
            raise ValueError("teardown scope must be pod or disk")
        return {"scope": scope, "ready": ready, "reason": reason}

    def release_writer(self) -> None:
        if self._owns_lease:
            try:
                lease = json.loads(self.lease_path.read_text(encoding="utf-8"))
                if lease.get("lease_id") == self._lease_id:
                    self.lease_path.unlink()
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            self._owns_lease = False
            self._lease_id = None

    def claim_writer(self) -> None:
        if self._owns_lease:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self._acquire_writer()

    def _stop_common(self, *, step: int, next_global_ordinal: int, checkpoint_generation: int, durability: str) -> dict[str, Any]:
        state = self._read_state()
        if state["state"] not in {"RUNNING", "QUIESCING"}:
            raise ValueError("stop/pause requires RUNNING or QUIESCING state")
        self._validate_progress(state, step, next_global_ordinal)
        state.update(
            step=step,
            next_global_ordinal=next_global_ordinal,
            checkpoint_generation=checkpoint_generation,
            checkpoint_durability=durability,
        )
        self._clear_current_launch(state)
        state = self._transition(state, "STOPPING")
        return self._transition(state, "STOPPED")

    def _transition(self, state: dict[str, Any], target: str) -> dict[str, Any]:
        if target not in _TRANSITIONS[str(state["state"])]:
            raise ValueError(f"illegal operator transition: {state['state']} -> {target}")
        state["state"] = target
        state["generation"] = int(state["generation"]) + 1
        self._write_state(state)
        return state

    @staticmethod
    def _validate_progress(state: dict[str, Any], step: int, next_global_ordinal: int) -> None:
        if step < int(state["step"]) or next_global_ordinal < int(state["next_global_ordinal"]):
            raise ValueError("checkpoint progress cannot move backward")

    @staticmethod
    def _validate_process_identity(identity: dict[str, Any]) -> None:
        fields = {
            "pid", "process_start_token", "host_id", "process_group_id", "command_hash",
            "attempt_id", "label", "role",
        }
        if set(identity) != fields or type(identity.get("pid")) is not int or identity["pid"] <= 0:
            raise ValueError("process identity fields or PID are invalid")
        if type(identity.get("process_group_id")) is not int or identity["process_group_id"] <= 0:
            raise ValueError("positive process-group identity is required")
        for field in ("process_start_token", "command_hash", "attempt_id"):
            value = identity.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"process identity {field} is invalid")
        if not isinstance(identity.get("host_id"), str) or not identity["host_id"]:
            raise ValueError("process host identity is required")
        if not isinstance(identity.get("label"), str) or not isinstance(identity.get("role"), str):
            raise ValueError("process label and role are required")

    @staticmethod
    def _start_failure_payload(
        *, command_id: object, kind: object, attempt_hash: object,
        failure_code: object, child_role: object, exit_code: object,
    ) -> dict[str, Any]:
        try:
            command = _require_launch_id(command_id)
        except ValueError as error:
            raise ValueError("start failure command id is invalid") from error
        if kind not in {"start", "resume"}:
            raise ValueError("start failure reservation kind is invalid")
        if (
            not isinstance(attempt_hash, str)
            or len(attempt_hash) != 64
            or any(character not in "0123456789abcdef" for character in attempt_hash)
        ):
            raise ValueError("start failure attempt hash is invalid")
        child_codes = {
            "child_launch_failed", "child_pre_ready_exit", "child_ready_timeout",
            "child_ready_invalid", "child_identity_unavailable",
        }
        launcher_codes = {"launcher_preflight_failed", "confirmation_rejected"}
        if failure_code not in child_codes | launcher_codes:
            raise ValueError("start failure code is invalid")
        if (failure_code in launcher_codes) != (child_role == "launcher") or (
            failure_code in child_codes and child_role not in {"worker", "watchdog"}
        ):
            raise ValueError("start failure child role is invalid")
        if failure_code == "child_pre_ready_exit":
            if type(exit_code) is not int:
                raise ValueError("start failure exit code is required")
        elif failure_code == "child_launch_failed":
            if exit_code is not None and type(exit_code) is not int:
                raise ValueError("start failure exit code is invalid")
        elif exit_code is not None:
            raise ValueError("start failure exit code must be null")
        return {
            "command_id": command, "kind": kind, "attempt_hash": attempt_hash,
            "failure_code": failure_code, "child_role": child_role,
            "exit_code": exit_code,
        }

    @classmethod
    def _validate_startup_orphan_processes(
        cls, processes: object, *, attempt_hash: str,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(processes, dict) or not set(processes) <= {"worker", "watchdog"}:
            raise ValueError("startup orphan processes must contain only worker/watchdog identities")
        expected = {
            "worker": ("r7-pretraining-worker", "trainer"),
            "watchdog": ("r7-pretraining-watchdog", "guard"),
        }
        fields = {
            "pid", "process_start_token", "host_id", "process_group_id", "command_hash",
            "attempt_id", "label", "role",
        }
        validated: dict[str, dict[str, Any]] = {}
        for name, value in processes.items():
            if not isinstance(value, dict) or set(value) != fields:
                raise ValueError(f"startup {name} identity fields are invalid")
            identity = dict(value)
            label, role = expected[name]
            if (
                type(identity.get("pid")) is not int
                or identity["pid"] <= 0
                or not isinstance(identity.get("host_id"), str)
                or not identity["host_id"]
                or identity.get("attempt_id") != attempt_hash
                or identity.get("label") != label
                or identity.get("role") != role
            ):
                raise ValueError(f"startup {name} identity binding is invalid")
            for field in ("command_hash", "attempt_id"):
                field_value = identity.get(field)
                if (
                    not isinstance(field_value, str)
                    or len(field_value) != 64
                    or any(character not in "0123456789abcdef" for character in field_value)
                ):
                    raise ValueError(f"startup {name} identity {field} is invalid")
            token = identity.get("process_start_token")
            group = identity.get("process_group_id")
            if token is not None and (
                not isinstance(token, str)
                or len(token) != 64
                or any(character not in "0123456789abcdef" for character in token)
            ):
                raise ValueError(f"startup {name} process start token is invalid")
            if group is not None and (type(group) is not int or group != identity["pid"]):
                raise ValueError(f"startup {name} process group is invalid")
            if token is not None and group is not None:
                cls._validate_process_identity(identity)
            validated[name] = identity
        return validated

    @staticmethod
    def _require_launch_state_shape(state: dict[str, Any]) -> None:
        missing = _LAUNCH_FIELDS - set(state)
        if missing or not isinstance(state.get("launch_history"), list):
            raise ValueError(f"operator state lacks corrected launch fields: {sorted(missing)}")

    @staticmethod
    def _clear_current_launch(state: dict[str, Any], *, clear_mode: bool = False) -> None:
        for field in (
            "process_identity", "watchdog_process_identity", "worker_ready_receipt",
            "watchdog_ready_receipt", "active_launch_id", "active_attempt_started_epoch",
            "start_command", "startup_orphan_processes",
        ):
            state[field] = None
        if clear_mode:
            state["active_mode"] = None

    def _validate_launch_history(
        self, state: dict[str, Any], attempts: list[ExecutionAttempt] | None = None,
    ) -> None:
        attempts = self._attempt_chain() if attempts is None else attempts
        history = state["launch_history"]
        if len(history) > len(attempts):
            raise ValueError("launch history is longer than the execution-attempt ledger")
        positions = {attempt.identity: index for index, attempt in enumerate(attempts)}
        attempts_by_hash = {attempt.identity: attempt for attempt in attempts}
        seen_launches: set[str] = set()
        seen_attempts: set[str] = set()
        previous_position = -1
        expected_policy_hash = self._current_performance_policy_hash()
        fields = {
            "schema_version", "launch_id", "command_kind", "run_identity",
            "attempt_hash", "fencing_epoch", "activated_at_epoch",
            "worker_process_identity", "watchdog_process_identity",
            "worker_ready_receipt", "watchdog_ready_receipt", "activation_hash",
        }
        for row in history:
            if not isinstance(row, dict) or set(row) != fields or row.get("schema_version") != 1:
                raise ValueError("launch history entry fields or schema are invalid")
            try:
                launch_id = _require_launch_id(row.get("launch_id"))
            except ValueError as error:
                raise ValueError("launch history launch id is invalid") from error
            attempt_hash = row.get("attempt_hash")
            if launch_id in seen_launches or attempt_hash in seen_attempts:
                raise ValueError("launch history contains duplicate launch or attempt identity")
            if attempt_hash not in positions:
                raise ValueError("launch history attempt is outside the execution-attempt ledger")
            position = positions[str(attempt_hash)]
            if position <= previous_position:
                raise ValueError("launch history attempt order is invalid")
            previous_position = position
            seen_launches.add(launch_id)
            seen_attempts.add(str(attempt_hash))
            attempt = attempts_by_hash[str(attempt_hash)]
            command_kind = row.get("command_kind")
            if command_kind not in {"start", "resume"} or (
                (command_kind == "start") != (attempt.parent_attempt_hash is None)
            ):
                raise ValueError("launch history command kind is invalid")
            activated_at = row.get("activated_at_epoch")
            if (
                type(activated_at) not in (int, float)
                or not math.isfinite(float(activated_at))
                or float(activated_at) <= 0
            ):
                raise ValueError("launch history activation timestamp is invalid")
            unsigned = {key: value for key, value in row.items() if key != "activation_hash"}
            if row.get("activation_hash") != stable_json_hash(unsigned):
                raise ValueError("launch history activation hash mismatch")
            worker_identity = row["worker_process_identity"]
            watchdog_identity = row["watchdog_process_identity"]
            if not isinstance(worker_identity, dict) or not isinstance(watchdog_identity, dict):
                raise ValueError("launch history process identities are invalid")
            self._validate_process_identity(worker_identity)
            self._validate_process_identity(watchdog_identity)
            expected_identities = (
                (worker_identity, "r7-pretraining-worker", "trainer"),
                (watchdog_identity, "r7-pretraining-watchdog", "guard"),
            )
            if worker_identity["pid"] == watchdog_identity["pid"] or (
                worker_identity["process_group_id"] == watchdog_identity["process_group_id"]
            ):
                raise ValueError("launch history process identities are not distinct")
            for identity, label, role in expected_identities:
                if (
                    identity["process_group_id"] != identity["pid"]
                    or identity["attempt_id"] != attempt.identity
                    or identity["label"] != label
                    or identity["role"] != role
                ):
                    raise ValueError("launch history process identity binding mismatch")
            worker_ready = load_worker_ready(row["worker_ready_receipt"])
            watchdog_ready = load_watchdog_ready(row["watchdog_ready_receipt"])
            if worker_ready != row["worker_ready_receipt"] or watchdog_ready != row["watchdog_ready_receipt"]:
                raise ValueError("launch history READY receipt is not canonical")
            common = {
                "launch_id": launch_id, "run_identity": state["run_identity"],
                "attempt_hash": attempt.identity, "fencing_epoch": attempt.fencing_epoch,
                "release_hash": attempt.release_hash,
            }
            if row.get("run_identity") != state["run_identity"] or row.get("fencing_epoch") != attempt.fencing_epoch:
                raise ValueError("launch history run or fence binding mismatch")
            if any(
                worker_ready.get(key) != value or watchdog_ready.get(key) != value
                for key, value in common.items()
            ):
                raise ValueError("launch history dual READY binding mismatch")
            worker_bindings = {
                "environment_hash": attempt.environment_hash,
                "data_placement_hash": attempt.data_placement_hash,
                "storage_layout_hash": attempt.storage_layout_hash,
            }
            if any(worker_ready.get(key) != value for key, value in worker_bindings.items()):
                raise ValueError("launch history worker READY attempt binding mismatch")
            if (
                worker_ready["process_pid"] != worker_identity["pid"]
                or worker_ready["process_start_token"] != worker_identity["process_start_token"]
                or watchdog_ready["process_pid"] != watchdog_identity["pid"]
                or watchdog_ready["process_start_token"] != watchdog_identity["process_start_token"]
                or watchdog_ready["ready_at_epoch"] > worker_ready["ready_at_epoch"]
                or worker_ready["performance_policy_hash"] != expected_policy_hash
                or watchdog_ready["performance_policy_hash"] != expected_policy_hash
            ):
                raise ValueError("launch history process/READY evidence mismatch")
        if state.get("state") in {"RUNNING", "QUIESCING"}:
            if not history:
                raise ValueError("active operator state has no launch history")
            active = history[-1]
            expected_live = {
                "attempt_hash": active["attempt_hash"],
                "fencing_epoch": active["fencing_epoch"],
                "active_launch_id": active["launch_id"],
                "active_attempt_started_epoch": active["activated_at_epoch"],
                "process_identity": active["worker_process_identity"],
                "watchdog_process_identity": active["watchdog_process_identity"],
                "worker_ready_receipt": active["worker_ready_receipt"],
                "watchdog_ready_receipt": active["watchdog_ready_receipt"],
                "start_command": None,
            }
            if any(state.get(key) != value for key, value in expected_live.items()):
                raise ValueError("active launch fields conflict with launch history")

    def _current_performance_policy_hash(self) -> str | None:
        contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        target = self.root / "configuration-performance-policy.json"
        if contract.get("artifact_kind") != "twc_pretraining_experiment_training_contract_v2":
            if target.exists() or target.is_symlink():
                raise ValueError("non-configuration launch has an installed performance policy")
            return None
        binding = contract.get("execution_binding")
        if not isinstance(binding, dict) or not isinstance(binding.get("execution_path_hash"), str):
            raise ValueError("configuration contract execution binding is invalid")
        if target.is_symlink() or not target.is_file():
            raise ValueError("configuration performance policy is missing")
        policy = load_configuration_performance_policy(
            json.loads(target.read_text(encoding="utf-8")),
            expected_execution_path_hash=binding["execution_path_hash"],
        )
        return str(policy["policy_hash"])

    @staticmethod
    def _confirmed_launch_replay(
        *, state: dict[str, Any], command_id: str, command_kind: str,
        worker_identity: dict[str, Any], watchdog_identity: dict[str, Any],
        worker_ready: dict[str, Any], watchdog_ready: dict[str, Any],
    ) -> bool:
        rows = [
            row for row in state["launch_history"]
            if isinstance(row, dict) and (
                row.get("launch_id") == command_id
                or row.get("attempt_hash") == worker_ready.get("attempt_hash")
            )
        ]
        if not rows:
            return False
        if len(rows) != 1:
            raise ValueError("launch confirmation history is conflicting")
        row = rows[0]
        unsigned = {key: value for key, value in row.items() if key != "activation_hash"}
        expected = {
            "launch_id": command_id, "command_kind": command_kind,
            "run_identity": worker_ready["run_identity"],
            "attempt_hash": worker_ready["attempt_hash"],
            "fencing_epoch": worker_ready["fencing_epoch"],
            "worker_process_identity": worker_identity,
            "watchdog_process_identity": watchdog_identity,
            "worker_ready_receipt": worker_ready,
            "watchdog_ready_receipt": watchdog_ready,
        }
        if row.get("activation_hash") != stable_json_hash(unsigned) or any(
            row.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("launch confirmation replay conflicts with activation history")
        if state.get("state") in {"RUNNING", "QUIESCING"}:
            if state.get("active_launch_id") != command_id:
                raise ValueError("launch confirmation replay conflicts with the active launch")
            return True
        if state.get("state") in {"STOPPED", "COMPLETED", "FAILED"}:
            uncleared = (
                "process_identity", "watchdog_process_identity", "worker_ready_receipt",
                "watchdog_ready_receipt", "active_launch_id", "active_attempt_started_epoch",
                "start_command", "startup_orphan_processes",
            )
            if state.get("attempt_hash") != row.get("attempt_hash") or any(
                state.get(field) is not None for field in uncleared
            ):
                raise ValueError("terminal launch confirmation replay lacks exact cleared state")
            return True
        raise ValueError("launch confirmation replay is invalid in the current operator state")

    def _acquire_writer(self) -> None:
        lease_id = uuid.uuid4().hex
        identity = {
            "pid": os.getpid(),
            "process_start_token": process_start_token(os.getpid()),
            "host_id": host_identity(),
        }
        payload = {"lease_id": lease_id, "process_identity": identity}
        while True:
            try:
                descriptor = os.open(self.lease_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                break
            except FileExistsError as error:
                try:
                    existing_bytes = self.lease_path.read_bytes()
                    existing = json.loads(existing_bytes)
                    observation = observe_process(existing["process_identity"])
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    raise FileExistsError("writer lease exists but ownership is uncertain") from error
                if observation == "uncertain":
                    raise FileExistsError(
                        "writer lease exists but ownership is uncertain"
                    ) from error
                if observation != "exited":
                    raise FileExistsError("writer lease already exists") from error
                try:
                    if self.lease_path.read_bytes() == existing_bytes:
                        self.lease_path.unlink()
                except FileNotFoundError:
                    pass
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        self._owns_lease = True
        self._lease_id = lease_id

    def _append_attempt(self, attempt: ExecutionAttempt) -> None:
        self._require_writer()
        prefix = self.attempts_path.read_bytes() if self.attempts_path.exists() else b""
        row = (
            json.dumps(
                {**attempt.to_dict(), "attempt_hash": attempt.identity},
                sort_keys=True,
            )
            + "\n"
        ).encode()
        partial = self.root / f"attempts.jsonl.{uuid.uuid4().hex}.partial"
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(prefix)
                handle.write(row)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, self.attempts_path)
            _fsync_dir(self.root)
        except Exception:
            try:
                partial.unlink()
            except FileNotFoundError:
                pass
            raise

    def _attempt_chain(self) -> list[ExecutionAttempt]:
        try:
            lines = self.attempts_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as error:
            raise RuntimeError("execution-attempt ledger is missing") from error
        if not lines:
            raise RuntimeError("execution-attempt ledger is missing")
        attempts: list[ExecutionAttempt] = []
        identities: set[str] = set()
        attempt_ids: set[str] = set()
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("execution-attempt ledger contains invalid JSON") from error
            if not isinstance(payload, dict):
                raise ValueError("execution-attempt ledger row must be an object")
            attempt = load_execution_attempt(payload)
            if attempt.identity in identities or attempt.attempt_id in attempt_ids:
                raise ValueError("execution-attempt ledger contains duplicate identity")
            if attempts:
                attempt.validate_successor(attempts[-1])
            attempts.append(attempt)
            identities.add(attempt.identity)
            attempt_ids.add(attempt.attempt_id)
        return attempts

    def _latest_attempt(self) -> ExecutionAttempt:
        return self._attempt_chain()[-1]

    def _read_state(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise RuntimeError("operator is not planned") from error
        if payload.get("artifact_kind") == "twc_pretraining_operator_state_v1":
            raise ValueError("R6 v1 operator state must be rebuilt for R7")
        if payload.get("artifact_kind") != "twc_pretraining_operator_state_v2" or payload.get("schema_version") != 2:
            raise ValueError("unsupported operator state")
        expected_hash = stable_json_hash({key: value for key, value in payload.items() if key != "state_hash"})
        if payload.get("state_hash") != expected_hash:
            raise ValueError("operator state hash mismatch")
        return payload

    def _write_state(self, state: dict[str, Any]) -> None:
        payload = {key: value for key, value in state.items() if key != "state_hash"}
        payload["state_hash"] = stable_json_hash(payload)
        state.clear()
        state.update(payload)
        self._atomic_json(self.state_path, payload)

    def _require_writer(self) -> None:
        try:
            lease = json.loads(self.lease_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            lease = {}
        if not self._owns_lease or lease.get("lease_id") != self._lease_id:
            raise RuntimeError("writer lease is required for operator mutation")

    def _validate_checkpoint(self, checkpoint: CheckpointReference) -> LoadedCheckpoint:
        if self.checkpoint_manager is None:
            raise RuntimeError("checkpoint manager is required for checkpoint mutation")
        loaded = self.checkpoint_manager.validate_reference(checkpoint)
        state = self._read_state()
        if loaded.run_identity != state["run_identity"] or loaded.attempt_hash != state["attempt_hash"]:
            raise ValueError("checkpoint does not bind the active run and attempt")
        return loaded

    @staticmethod
    def _validate_checkpoint_progress(loaded: LoadedCheckpoint, step: int, next_global_ordinal: int) -> None:
        if loaded.step != step or loaded.next_global_ordinal != next_global_ordinal:
            raise ValueError("operator progress does not match verified checkpoint")

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_dir(path.parent)
