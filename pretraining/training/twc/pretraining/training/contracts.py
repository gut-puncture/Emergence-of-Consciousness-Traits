"""Immutable scientific configuration and execution-attempt contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from twc.repro.hashing import stable_json_hash


def _require_hash(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class RunContract:
    """Scientific identity; physical slicing and host placement are excluded."""

    run_id: str
    data_manifest_hash: str
    view_hash: str
    validation_view_hash: str
    tokenizer_hash: str
    model_hash: str
    code_hash: str
    global_step_plan_hash: str
    target_loss_tokens: int
    context_length: int
    optimizer_step_token_cap: int
    optimizer_step_attention_cell_cap: int
    optimizer_step_sequence_cap: int
    precision: str
    attention_backend: str
    attention_num_splits: int | str
    loss_backend: str
    compile_mode: str
    allow_tf32: bool
    optimizer: str
    optimizer_backend: str
    learning_rate: float
    adam_betas: tuple[float, float]
    adam_eps: float
    weight_decay: float
    warmup_loss_tokens: int
    stable_loss_tokens: int
    decay_loss_tokens: int
    grad_clip_norm: float
    seed: int
    full_validation_at_schedule_boundaries: bool
    evaluation_every_loss_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id is required")
        for field in (
            "data_manifest_hash", "view_hash", "validation_view_hash", "tokenizer_hash", "model_hash", "code_hash",
            "global_step_plan_hash",
        ):
            _require_hash(getattr(self, field), field)
        for field in (
            "target_loss_tokens", "context_length", "optimizer_step_token_cap",
            "optimizer_step_attention_cell_cap", "optimizer_step_sequence_cap",
            "warmup_loss_tokens", "stable_loss_tokens", "decay_loss_tokens",
        ):
            _positive_int(getattr(self, field), field)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.full_validation_at_schedule_boundaries) is not bool:
            raise ValueError("full_validation_at_schedule_boundaries must be a boolean")
        if self.evaluation_every_loss_tokens is not None:
            _positive_int(
                self.evaluation_every_loss_tokens,
                "evaluation_every_loss_tokens",
            )
        if self.warmup_loss_tokens + self.stable_loss_tokens + self.decay_loss_tokens != self.target_loss_tokens:
            raise ValueError("token schedule phases must sum exactly to target_loss_tokens")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.attention_backend not in {"reference", "native_varlen"}:
            raise ValueError("unsupported attention backend")
        if not (self.attention_num_splits == "auto" or type(self.attention_num_splits) is int and self.attention_num_splits > 0):
            raise ValueError("attention_num_splits must be a positive integer or auto")
        if self.loss_backend not in {"dense_reference", "chunked_projection"}:
            raise ValueError("unsupported loss backend")
        if self.compile_mode not in {"disabled", "default", "reduce-overhead", "max-autotune"}:
            raise ValueError("unsupported compile_mode")
        if type(self.allow_tf32) is not bool:
            raise ValueError("allow_tf32 must be a boolean")
        if self.optimizer != "adamw" or self.optimizer_backend not in {"reference", "fused"}:
            raise ValueError("unsupported optimizer backend")
        beta1, beta2 = self.adam_betas
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("adam_betas must be in [0, 1)")
        numeric = (self.learning_rate, self.adam_eps, self.weight_decay, self.grad_clip_norm)
        if any(type(value) not in {float, int} or not math.isfinite(value) for value in numeric):
            raise ValueError("optimizer values must be finite numbers")
        if self.learning_rate <= 0 or self.adam_eps <= 0 or self.grad_clip_norm <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer values are outside their valid range")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if self.evaluation_every_loss_tokens is None:
            values.pop("evaluation_every_loss_tokens")
            return {
                "artifact_kind": "twc_pretraining_run_contract_v2",
                "schema_version": 2,
                **values,
            }
        return {
            "artifact_kind": "twc_pretraining_run_contract_v3",
            "schema_version": 3,
            **values,
        }


def load_run_contract(payload: dict[str, Any]) -> RunContract:
    if payload.get("artifact_kind") == "twc_pretraining_run_contract_v1":
        raise ValueError("R6 v1 run contracts must be rebuilt for R7")
    fields = set(RunContract.__dataclass_fields__)
    cadence_fields = {"evaluation_every_loss_tokens"}
    metadata = {"artifact_kind", "schema_version", "run_identity"}
    unknown = set(payload) - fields - metadata
    if unknown:
        raise ValueError(f"run contract has unknown fields: {sorted(unknown)}")
    version = (payload.get("artifact_kind"), payload.get("schema_version"))
    if version == ("twc_pretraining_run_contract_v2", 2):
        required_fields = fields - cadence_fields
        if set(payload) & cadence_fields:
            raise ValueError("v2 run contract cannot carry evaluation cadence")
    elif version == ("twc_pretraining_run_contract_v3", 3):
        required_fields = fields
        if payload.get("evaluation_every_loss_tokens") is None:
            raise ValueError("v3 run contract requires evaluation cadence")
    else:
        raise ValueError("unsupported run contract artifact")
    missing = required_fields - set(payload)
    if missing:
        raise ValueError(f"run contract is missing fields: {sorted(missing)}")
    values = {field: payload[field] for field in required_fields}
    if version == ("twc_pretraining_run_contract_v2", 2):
        values["evaluation_every_loss_tokens"] = None
    values["adam_betas"] = tuple(values["adam_betas"])
    contract = RunContract(**values)
    if payload.get("run_identity") is not None and payload["run_identity"] != contract.identity:
        raise ValueError("run contract identity mismatch")
    return contract


def load_worker_contract(payload: dict[str, Any]) -> Any:
    """Dispatch only the production or explicitly non-production worker contracts."""

    if payload.get("artifact_kind") == "twc_gpu_engineering_run_contract_v1":
        from twc.pretraining.training.experiments import load_engineering_canary_contract

        return load_engineering_canary_contract(payload)
    if payload.get("artifact_kind") == "twc_pretraining_experiment_training_contract_v1":
        from twc.pretraining.training.experiments import load_experiment_training_contract

        return load_experiment_training_contract(payload)
    if payload.get("artifact_kind") == "twc_pretraining_experiment_training_contract_v2":
        from twc.pretraining.configuration_study_execution import (
            load_experiment_training_contract_v2,
        )

        return load_experiment_training_contract_v2(payload)
    return load_run_contract(payload)


_CONFIG_FIELDS = {
    "artifact_kind",
    "schema_version",
    "run_id",
    "model",
    "data",
    "training",
    "optimizer",
    "schedule",
    "numerics",
    "evaluation",
    "source",
    "configuration_hash",
    "raw_text_serialized",
    "raw_url_serialized",
    "tokens_serialized",
    "secrets_serialized",
}


def build_training_config(
    *,
    contract: RunContract,
    model: Mapping[str, Any],
    evaluation_manifest: Mapping[str, Any],
    evaluation_implementation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the one portable configuration consumed by every pretraining run."""

    payload = {
        "artifact_kind": "twc_pretraining_config_v1",
        "schema_version": 1,
        "run_id": contract.run_id,
        "model": dict(model),
        "data": {
            "data_manifest_hash": contract.data_manifest_hash,
            "train_view_hash": contract.view_hash,
            "validation_view_hash": contract.validation_view_hash,
            "tokenizer_hash": contract.tokenizer_hash,
            "global_step_plan_hash": contract.global_step_plan_hash,
        },
        "training": {
            "target_loss_tokens": contract.target_loss_tokens,
            "context_length": contract.context_length,
            "optimizer_step_token_cap": contract.optimizer_step_token_cap,
            "optimizer_step_attention_cell_cap": contract.optimizer_step_attention_cell_cap,
            "optimizer_step_sequence_cap": contract.optimizer_step_sequence_cap,
            "seed": contract.seed,
        },
        "optimizer": {
            "name": contract.optimizer,
            "backend": contract.optimizer_backend,
            "learning_rate": contract.learning_rate,
            "adam_betas": list(contract.adam_betas),
            "adam_eps": contract.adam_eps,
            "weight_decay": contract.weight_decay,
            "grad_clip_norm": contract.grad_clip_norm,
        },
        "schedule": {
            "warmup_loss_tokens": contract.warmup_loss_tokens,
            "stable_loss_tokens": contract.stable_loss_tokens,
            "decay_loss_tokens": contract.decay_loss_tokens,
            "full_validation_at_schedule_boundaries": contract.full_validation_at_schedule_boundaries,
        },
        "numerics": {
            "precision": contract.precision,
            "attention_backend": contract.attention_backend,
            "attention_num_splits": contract.attention_num_splits,
            "loss_backend": contract.loss_backend,
            "compile_mode": contract.compile_mode,
            "allow_tf32": contract.allow_tf32,
        },
        "evaluation": {
            "manifest": dict(evaluation_manifest),
            "implementation": dict(evaluation_implementation),
            "every_loss_tokens": contract.evaluation_every_loss_tokens,
        },
        "source": {"implementation_hash": contract.code_hash, **dict(provenance)},
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "tokens_serialized": False,
        "secrets_serialized": False,
    }
    payload["configuration_hash"] = stable_json_hash(payload)
    return load_training_config(payload)


def load_training_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate one portable configuration without consulting runtime state."""

    if not isinstance(payload, Mapping):
        raise ValueError("training configuration must be an object")
    if set(payload) != _CONFIG_FIELDS:
        raise ValueError("training configuration fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_pretraining_config_v1"
        or payload.get("schema_version") != 1
        or any(
            payload.get(field) is not False
            for field in (
                "raw_text_serialized",
                "raw_url_serialized",
                "tokens_serialized",
                "secrets_serialized",
            )
        )
    ):
        raise ValueError("training configuration boundary is invalid")
    contract = training_config_contract_unchecked(payload)
    model = payload["model"]
    if not isinstance(model, Mapping) or model.get("model_hash") != contract.model_hash:
        raise ValueError("training configuration model identity mismatch")
    for field in ("data", "training", "optimizer", "schedule", "numerics", "evaluation", "source"):
        if not isinstance(payload[field], Mapping):
            raise ValueError(f"training configuration {field} must be an object")
    if set(payload["evaluation"]) != {
        "manifest", "implementation", "every_loss_tokens"
    } or any(
        not isinstance(payload["evaluation"][field], Mapping)
        for field in ("manifest", "implementation")
    ) or type(payload["evaluation"]["every_loss_tokens"]) is not int:
        raise ValueError("training configuration evaluation fields are invalid")
    unhashed = {key: value for key, value in payload.items() if key != "configuration_hash"}
    if payload["configuration_hash"] != stable_json_hash(unhashed):
        raise ValueError("training configuration identity mismatch")
    return dict(payload)


def training_config_contract(payload: Mapping[str, Any]) -> RunContract:
    """Return the worker-facing scientific contract from a validated configuration."""

    config = load_training_config(payload)
    return training_config_contract_unchecked(config)


def training_config_contract_unchecked(payload: Mapping[str, Any]) -> RunContract:
    data = payload["data"]
    training = payload["training"]
    optimizer = payload["optimizer"]
    schedule = payload["schedule"]
    numerics = payload["numerics"]
    source = payload["source"]
    evaluation = payload["evaluation"]
    expected = {
        "data": {
            "data_manifest_hash",
            "train_view_hash",
            "validation_view_hash",
            "tokenizer_hash",
            "global_step_plan_hash",
        },
        "training": {
            "target_loss_tokens",
            "context_length",
            "optimizer_step_token_cap",
            "optimizer_step_attention_cell_cap",
            "optimizer_step_sequence_cap",
            "seed",
        },
        "optimizer": {
            "name",
            "backend",
            "learning_rate",
            "adam_betas",
            "adam_eps",
            "weight_decay",
            "grad_clip_norm",
        },
        "schedule": {
            "warmup_loss_tokens",
            "stable_loss_tokens",
            "decay_loss_tokens",
            "full_validation_at_schedule_boundaries",
        },
        "numerics": {
            "precision",
            "attention_backend",
            "attention_num_splits",
            "loss_backend",
            "compile_mode",
            "allow_tf32",
        },
    }
    for name, fields in expected.items():
        value = payload[name]
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(f"training configuration {name} fields are invalid")
    if not isinstance(source, Mapping) or not isinstance(source.get("implementation_hash"), str):
        raise ValueError("training configuration source is invalid")
    return RunContract(
        run_id=payload["run_id"],
        data_manifest_hash=data["data_manifest_hash"],
        view_hash=data["train_view_hash"],
        validation_view_hash=data["validation_view_hash"],
        tokenizer_hash=data["tokenizer_hash"],
        model_hash=payload["model"]["model_hash"],
        code_hash=source["implementation_hash"],
        global_step_plan_hash=data["global_step_plan_hash"],
        target_loss_tokens=training["target_loss_tokens"],
        context_length=training["context_length"],
        optimizer_step_token_cap=training["optimizer_step_token_cap"],
        optimizer_step_attention_cell_cap=training["optimizer_step_attention_cell_cap"],
        optimizer_step_sequence_cap=training["optimizer_step_sequence_cap"],
        precision=numerics["precision"],
        attention_backend=numerics["attention_backend"],
        attention_num_splits=numerics["attention_num_splits"],
        loss_backend=numerics["loss_backend"],
        compile_mode=numerics["compile_mode"],
        allow_tf32=numerics["allow_tf32"],
        optimizer=optimizer["name"],
        optimizer_backend=optimizer["backend"],
        learning_rate=optimizer["learning_rate"],
        adam_betas=tuple(optimizer["adam_betas"]),
        adam_eps=optimizer["adam_eps"],
        weight_decay=optimizer["weight_decay"],
        warmup_loss_tokens=schedule["warmup_loss_tokens"],
        stable_loss_tokens=schedule["stable_loss_tokens"],
        decay_loss_tokens=schedule["decay_loss_tokens"],
        grad_clip_norm=optimizer["grad_clip_norm"],
        seed=training["seed"],
        full_validation_at_schedule_boundaries=schedule[
            "full_validation_at_schedule_boundaries"
        ],
        evaluation_every_loss_tokens=evaluation["every_loss_tokens"],
    )


@dataclass(frozen=True, slots=True)
class SelectedCheckpoint:
    """Exact immutable checkpoint selected for one successor attempt."""

    generation: int
    durability: str
    manifest_sha256: str
    state_sha256: str
    run_identity: str
    attempt_hash: str
    step: int
    consumed_loss_tokens: int
    next_global_ordinal: int

    def __post_init__(self) -> None:
        for field in (
            "manifest_sha256",
            "state_sha256",
            "run_identity",
            "attempt_hash",
        ):
            _require_hash(getattr(self, field), field)
        if self.durability not in {"local_verified", "persistent_verified"}:
            raise ValueError("selected checkpoint durability is invalid")
        for field in (
            "generation",
            "step",
            "consumed_loss_tokens",
            "next_global_ordinal",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"selected checkpoint {field} must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectedCheckpoint":
        if not isinstance(payload, Mapping):
            raise ValueError("selected checkpoint must be an object")
        fields = set(cls.__dataclass_fields__)
        if set(payload) != fields:
            unknown, missing = set(payload) - fields, fields - set(payload)
            raise ValueError(
                "selected checkpoint fields are invalid; "
                f"unknown={sorted(unknown)} missing={sorted(missing)}"
            )
        return cls(**{field: payload[field] for field in fields})


@dataclass(frozen=True, slots=True)
class ExecutionAttempt:
    """Operational provenance for one physical execution attempt."""

    run_identity: str
    attempt_id: str
    parent_attempt_hash: str | None
    device_execution_plan_hash: str
    release_hash: str
    environment_hash: str
    data_placement_hash: str
    storage_layout_hash: str
    host_id: str
    fencing_epoch: int
    prefetch_batches: int
    pin_memory: bool
    telemetry_every_steps: int
    checkpoint_every_loss_tokens: int
    checkpoint_max_interval_seconds: int
    durable_every_loss_tokens: int
    durable_max_interval_seconds: int
    evaluation_every_loss_tokens: int | None
    train_records_hash: str | None = None
    validation_records_hash: str | None = None
    pack_root: str | None = None
    checkpoint_root: str | None = None
    persistent_checkpoint_root: str | None = None
    checkpoint_topology: str | None = None
    device: str | None = None
    read_mode: str | None = None
    selected_checkpoint: SelectedCheckpoint | None = None

    def __post_init__(self) -> None:
        for field in (
            "run_identity", "device_execution_plan_hash", "release_hash", "environment_hash",
            "data_placement_hash", "storage_layout_hash",
        ):
            _require_hash(getattr(self, field), field)
        if self.parent_attempt_hash is not None:
            _require_hash(self.parent_attempt_hash, "parent_attempt_hash")
        if (self.train_records_hash is None) != (self.validation_records_hash is None):
            raise ValueError(
                "execution attempt must bind both train and validation records hashes"
            )
        if self.train_records_hash is not None:
            _require_hash(self.train_records_hash, "train_records_hash")
            _require_hash(self.validation_records_hash, "validation_records_hash")
        if not self.attempt_id or not self.host_id:
            raise ValueError("attempt_id and host_id are required")
        _positive_int(self.fencing_epoch, "fencing_epoch")
        if self.prefetch_batches != 1:
            raise ValueError("R7 single-device BatchSource requires prefetch_batches=1")
        if type(self.pin_memory) is not bool:
            raise ValueError("pin_memory must be a boolean")
        for field in (
            "telemetry_every_steps", "checkpoint_every_loss_tokens", "checkpoint_max_interval_seconds",
            "durable_every_loss_tokens", "durable_max_interval_seconds",
        ):
            _positive_int(getattr(self, field), field)
        if self.durable_every_loss_tokens < self.checkpoint_every_loss_tokens:
            raise ValueError("durable cadence cannot be tighter than local checkpoint cadence")
        if self.durable_max_interval_seconds < self.checkpoint_max_interval_seconds:
            raise ValueError("durable interval cannot be shorter than local checkpoint interval")
        exact_fields = (
            self.pack_root,
            self.checkpoint_root,
            self.checkpoint_topology,
            self.device,
            self.read_mode,
        )
        exact_execution = any(value is not None for value in exact_fields) or (
            self.selected_checkpoint is not None
        )
        if exact_execution:
            if self.evaluation_every_loss_tokens is not None:
                raise ValueError(
                    "new execution attempt cannot carry evaluation cadence"
                )
            if any(value is None for value in exact_fields):
                raise ValueError("exact execution attempt is missing execution settings")
            if self.train_records_hash is None:
                raise ValueError("exact execution attempt requires records bindings")
            assert self.pack_root is not None
            assert self.checkpoint_root is not None
            assert self.checkpoint_topology is not None
            assert self.device is not None
            assert self.read_mode is not None
            for value, label in (
                (self.pack_root, "pack_root"),
                (self.checkpoint_root, "checkpoint_root"),
            ):
                path = Path(value)
                if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
                    raise ValueError(f"execution attempt {label} must be an absolute unaliased path")
            if self.persistent_checkpoint_root is not None:
                persistent = Path(self.persistent_checkpoint_root)
                if not persistent.is_absolute() or any(
                    part in {".", ".."} for part in persistent.parts
                ):
                    raise ValueError(
                        "execution attempt persistent_checkpoint_root must be an absolute unaliased path"
                    )
            if self.checkpoint_topology not in {
                "local_only",
                "direct_persistent",
                "local_then_replicate",
            }:
                raise ValueError("execution attempt checkpoint topology is invalid")
            if (
                self.checkpoint_topology == "local_then_replicate"
            ) != (self.persistent_checkpoint_root is not None):
                raise ValueError("execution attempt checkpoint roots contradict its topology")
            if self.device != "cpu" and not (
                self.device.startswith("cuda:")
                and self.device.removeprefix("cuda:").isdigit()
            ):
                raise ValueError("execution attempt device is invalid")
            if self.read_mode not in {"local_mmap", "buffered_pread"}:
                raise ValueError("execution attempt read mode is invalid")
            if self.selected_checkpoint is not None and not isinstance(
                self.selected_checkpoint, SelectedCheckpoint
            ):
                raise ValueError("execution attempt selected checkpoint is invalid")
        elif self.persistent_checkpoint_root is not None:
            raise ValueError("legacy execution attempt cannot carry a persistent checkpoint root")
        else:
            _positive_int(
                self.evaluation_every_loss_tokens,
                "evaluation_every_loss_tokens",
            )

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        execution_fields = {
            "pack_root",
            "checkpoint_root",
            "persistent_checkpoint_root",
            "checkpoint_topology",
            "device",
            "read_mode",
            "selected_checkpoint",
        }
        if self.pack_root is not None:
            values.pop("evaluation_every_loss_tokens")
            return {
                "artifact_kind": "twc_pretraining_execution_attempt_v4",
                "schema_version": 4,
                **values,
            }
        for field in execution_fields:
            values.pop(field)
        if self.train_records_hash is None:
            values.pop("train_records_hash")
            values.pop("validation_records_hash")
            return {
                "artifact_kind": "twc_pretraining_execution_attempt_v2",
                "schema_version": 2,
                **values,
            }
        return {
            "artifact_kind": "twc_pretraining_execution_attempt_v3",
            "schema_version": 3,
            **values,
        }

    def validate_successor(self, previous: "ExecutionAttempt") -> None:
        if self.run_identity != previous.run_identity:
            raise ValueError("successor run identity does not match")
        if self.parent_attempt_hash != previous.identity:
            raise ValueError("successor parent attempt hash does not match")
        if self.attempt_id == previous.attempt_id:
            raise ValueError("successor attempt_id must be unique")
        if self.fencing_epoch < previous.fencing_epoch:
            raise ValueError("successor fencing epoch cannot move backward")
        if (
            self.train_records_hash,
            self.validation_records_hash,
        ) != (
            previous.train_records_hash,
            previous.validation_records_hash,
        ):
            raise ValueError("successor records bindings do not match")
        if self.selected_checkpoint is not None:
            if self.selected_checkpoint.run_identity != self.run_identity:
                raise ValueError("selected checkpoint run identity does not match")
            if self.selected_checkpoint.attempt_hash != previous.identity:
                raise ValueError("selected checkpoint attempt does not match the predecessor")


def load_execution_attempt(payload: dict[str, Any]) -> ExecutionAttempt:
    if payload.get("artifact_kind") == "twc_pretraining_execution_attempt_v1":
        raise ValueError("R6 v1 execution attempts must be rebuilt for R7")
    fields = set(ExecutionAttempt.__dataclass_fields__)
    records_fields = {"train_records_hash", "validation_records_hash"}
    cadence_fields = {"evaluation_every_loss_tokens"}
    execution_fields = {
        "pack_root",
        "checkpoint_root",
        "persistent_checkpoint_root",
        "checkpoint_topology",
        "device",
        "read_mode",
        "selected_checkpoint",
    }
    metadata = {"artifact_kind", "schema_version", "identity", "attempt_hash"}
    unknown = set(payload) - fields - metadata
    if unknown:
        raise ValueError(f"execution attempt has unknown fields: {sorted(unknown)}")
    version = (payload.get("artifact_kind"), payload.get("schema_version"))
    if version == ("twc_pretraining_execution_attempt_v2", 2):
        required_fields = fields - records_fields - execution_fields
        if set(payload) & records_fields:
            raise ValueError("v2 execution attempt cannot carry v3 records bindings")
        if set(payload) & execution_fields:
            raise ValueError("v2 execution attempt cannot carry v4 execution snapshot")
    elif version == ("twc_pretraining_execution_attempt_v3", 3):
        required_fields = fields - execution_fields
        if set(payload) & execution_fields:
            raise ValueError("v3 execution attempt cannot carry v4 execution snapshot")
    elif version == ("twc_pretraining_execution_attempt_v4", 4):
        required_fields = fields - cadence_fields
        if set(payload) & cadence_fields:
            raise ValueError("v4 execution attempt cannot carry evaluation cadence")
    else:
        raise ValueError("unsupported execution attempt artifact")
    missing = required_fields - set(payload)
    if missing:
        raise ValueError(f"execution attempt is missing fields: {sorted(missing)}")
    values = {field: payload[field] for field in required_fields}
    if version == ("twc_pretraining_execution_attempt_v4", 4):
        values["evaluation_every_loss_tokens"] = None
        selected = values["selected_checkpoint"]
        values["selected_checkpoint"] = (
            None
            if selected is None
            else SelectedCheckpoint.from_dict(selected)
        )
    else:
        values.update({field: None for field in execution_fields})
        if version == ("twc_pretraining_execution_attempt_v2", 2):
            values.update(train_records_hash=None, validation_records_hash=None)
    attempt = ExecutionAttempt(**values)
    declared = payload.get("identity", payload.get("attempt_hash"))
    if declared is not None and declared != attempt.identity:
        raise ValueError("execution attempt identity mismatch")
    return attempt


_ALLOWED_RESUME_DRIFT = frozenset(
    {
        "pack_root",
        "checkpoint_root",
        "persistent_checkpoint_root",
        "read_mode",
        "pin_memory",
        "telemetry_every_steps",
        "checkpoint_every_loss_tokens",
        "checkpoint_max_interval_seconds",
        "durable_every_loss_tokens",
        "durable_max_interval_seconds",
        "data_placement_hash",
        "storage_layout_hash",
    }
)

_RESUME_MECHANICS = frozenset(
    {
        "attempt_id",
        "parent_attempt_hash",
        "host_id",
        "fencing_epoch",
        "selected_checkpoint",
    }
)


def validate_resume_compatibility(
    previous: ExecutionAttempt,
    successor: ExecutionAttempt,
    *,
    release_semantics_verified: bool = False,
) -> frozenset[str]:
    """Validate a prepared successor before it is appended to attempt history."""

    if previous.pack_root is None or successor.pack_root is None:
        raise ValueError("resume compatibility requires exact execution attempts")
    if successor.selected_checkpoint is None:
        raise ValueError("resume compatibility requires a selected checkpoint")

    successor.validate_successor(previous)

    changed = {
        field
        for field in ExecutionAttempt.__dataclass_fields__
        if getattr(previous, field) != getattr(successor, field)
    }
    allowed = set(_ALLOWED_RESUME_DRIFT)
    if release_semantics_verified:
        allowed.add("release_hash")
    prohibited = changed - allowed - _RESUME_MECHANICS
    if prohibited:
        raise ValueError(
            "resume is incompatible: immutable execution settings changed: "
            f"{sorted(prohibited)}"
        )

    if (
        "data_placement_hash" in changed
        and "pack_root" not in changed
    ):
        raise ValueError(
            "resume is incompatible: data placement receipt changed without its path"
        )
    if (
        "storage_layout_hash" in changed
        and not changed
        & {"checkpoint_root", "persistent_checkpoint_root"}
    ):
        raise ValueError(
            "resume is incompatible: storage layout receipt changed without its path"
        )

    return frozenset(changed & allowed)
