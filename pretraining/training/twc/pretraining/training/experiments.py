"""Fail-closed scientific contracts for bounded pretraining experiments.

These artifacts are deliberately distinct from :class:`RunContract`.  They can
describe configuration-search runs and a configuration-freeze decision, but
they can never authorize a production run or unlock accepted training data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import yaml

from twc.repro.hashing import stable_json_hash


_EXPERIMENT_PHASES = frozenset({"s_screen", "m_transfer", "l_confirmation"})
_EVALUATION_PHASES = _EXPERIMENT_PHASES | {"post_freeze_target_diagnostics"}
_LEARNING_RATES = frozenset({4e-4, 6e-4, 9e-4})
_CHECKPOINT_MULTIPLES = (2, 4, 8, 16, 24)
_EVIDENCE_CLASSIFICATIONS = frozenset(
    {
        "descriptive_observation",
        "within_scale_controlled",
        "cross_scale_directional",
        "decision_sufficient",
        "uninformative_floor",
        "contradictory",
        "invalid",
    }
)
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _require_hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a non-empty lowercase identifier")
    return value


def _require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    numeric = float(value)
    if nonnegative and numeric < 0:
        raise ValueError(f"{label} must be nonnegative")
    return numeric


def _require_unique(values: tuple[str, ...], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """One reviewed dense decoder shape in the bounded search space."""

    profile_id: str
    scale_class: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    parameter_count: int
    train_phase: str | None
    vocab_size: int = 32_768
    context_length: int = 4_096
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    architecture_family: str = "conventional_dense_decoder"
    norm: str = "rmsnorm"
    norm_epsilon: float = 1e-5
    position_encoding: str = "rope"
    rope_theta: int = 10_000
    mlp: str = "swiglu"
    attention: str = "grouped_query_attention"
    fused_qkv: bool = True
    tied_embeddings: bool = True
    bias: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _require_identifier(self.profile_id, "profile_id")
        if self.scale_class not in {"s", "m", "l", "endpoint"}:
            raise ValueError("unsupported model scale_class")
        if self.train_phase is not None and self.train_phase not in _EXPERIMENT_PHASES:
            raise ValueError("unsupported model train_phase")
        dimensions = (
            self.num_layers,
            self.hidden_size,
            self.intermediate_size,
            self.parameter_count,
            self.vocab_size,
            self.context_length,
            self.num_attention_heads,
            self.num_key_value_heads,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("model profile dimensions must be positive integers")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        if self.parameter_count != self._estimated_parameter_count():
            raise ValueError("model profile parameter_count does not match its architecture")
        if self.architecture_family != "conventional_dense_decoder":
            raise ValueError("experiments must remain conventional dense decoders")
        fixed = (
            self.norm == "rmsnorm",
            self.norm_epsilon == 1e-5,
            self.position_encoding == "rope",
            self.rope_theta == 10_000,
            self.mlp == "swiglu",
            self.attention == "grouped_query_attention",
            self.fused_qkv is True,
            self.tied_embeddings is True,
            self.bias is False,
            self.dropout == 0.0,
        )
        if not all(fixed):
            raise ValueError("experiment profiles cannot alter the conventional decoder architecture")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def _estimated_parameter_count(self) -> int:
        kv_size = self.num_key_value_heads * self.head_dim
        embedding = self.vocab_size * self.hidden_size
        attention = self.hidden_size * (self.hidden_size + 2 * kv_size) + self.hidden_size**2
        mlp = 3 * self.hidden_size * self.intermediate_size
        block_norms = 2 * self.hidden_size
        return embedding + self.num_layers * (attention + mlp + block_norms) + self.hidden_size


def _profile(
    profile_id: str,
    scale: str,
    layers: int,
    width: int,
    intermediate: int,
    count: int,
    phase: str | None,
) -> ModelProfile:
    return ModelProfile(profile_id, scale, layers, width, intermediate, count, phase)


MODEL_PROFILES: Mapping[str, ModelProfile] = MappingProxyType(
    {
        "s_deep": _profile("s_deep", "s", 15, 256, 832, 20_438_784, "s_screen"),
        "s_wide": _profile("s_wide", "s", 8, 320, 1_024, 20_403_520, "s_screen"),
        "m_deep": _profile("m_deep", "m", 16, 320, 1_040, 30_566_720, "m_transfer"),
        "m_wide": _profile("m_wide", "m", 10, 384, 1_248, 30_654_336, "m_transfer"),
        "l_deep": _profile("l_deep", "l", 20, 384, 1_248, 48_725_376, "l_confirmation"),
        "l_wide": _profile("l_wide", "l", 14, 448, 1_456, 49_113_792, "l_confirmation"),
        "r7_control": _profile("r7_control", "endpoint", 26, 512, 1_664, 100_297_216, None),
        "r7_wide_endpoint": _profile(
            "r7_wide_endpoint", "endpoint", 16, 640, 2_048, 100_291_200, None
        ),
    }
)

_PHASE_TOKEN_UNIT = MappingProxyType(
    {
        "s_screen": min(MODEL_PROFILES["s_deep"].parameter_count, MODEL_PROFILES["s_wide"].parameter_count),
        "m_transfer": min(MODEL_PROFILES["m_deep"].parameter_count, MODEL_PROFILES["m_wide"].parameter_count),
        "l_confirmation": min(MODEL_PROFILES["l_deep"].parameter_count, MODEL_PROFILES["l_wide"].parameter_count),
    }
)


@dataclass(frozen=True, slots=True)
class EngineeringCanaryContract:
    """Short diagnostic execution identity; never accepted as production/F1 evidence."""

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
    model_profile_id: str
    expected_parameter_count: int
    initial_attempt_stop_step: int

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, "run_id")
        for field in (
            "data_manifest_hash", "view_hash", "validation_view_hash", "tokenizer_hash",
            "model_hash", "code_hash", "global_step_plan_hash",
        ):
            _require_hash(getattr(self, field), field)
        for field in (
            "target_loss_tokens", "context_length", "optimizer_step_token_cap",
            "optimizer_step_attention_cell_cap", "optimizer_step_sequence_cap", "warmup_loss_tokens",
            "stable_loss_tokens", "decay_loss_tokens", "expected_parameter_count",
            "initial_attempt_stop_step",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.target_loss_tokens > 100_000_000:
            raise ValueError("engineering canary token target exceeds its diagnostic bound")
        if self.context_length != 4096:
            raise ValueError("engineering canary context_length must remain 4096")
        if self.warmup_loss_tokens + self.stable_loss_tokens + self.decay_loss_tokens != self.target_loss_tokens:
            raise ValueError("engineering canary schedule must sum to its target")
        if self.model_profile_id != "r7_control" or self.expected_parameter_count != MODEL_PROFILES["r7_control"].parameter_count:
            raise ValueError("engineering canary must use the exact R7 control model")
        if self.initial_attempt_stop_step != 10:
            raise ValueError("engineering canary initial controlled stop must remain step 10")
        if self.precision != "bf16" or self.attention_backend != "native_varlen":
            raise ValueError("engineering canary must exercise BF16 native variable-length attention")
        if not (self.attention_num_splits == "auto" or type(self.attention_num_splits) is int and self.attention_num_splits > 0):
            raise ValueError("engineering canary attention_num_splits is invalid")
        if self.loss_backend != "chunked_projection" or self.optimizer != "adamw" or self.optimizer_backend != "fused":
            raise ValueError("engineering canary must exercise the production loss/optimizer backends")
        if self.compile_mode not in {"disabled", "default", "reduce-overhead", "max-autotune"}:
            raise ValueError("engineering canary compile mode is invalid")
        if type(self.allow_tf32) is not bool or self.full_validation_at_schedule_boundaries is not True:
            raise ValueError("engineering canary boolean safety fields are invalid")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("engineering canary seed must be nonnegative")
        beta1, beta2 = self.adam_betas
        numeric = (self.learning_rate, beta1, beta2, self.adam_eps, self.weight_decay, self.grad_clip_norm)
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in numeric):
            raise ValueError("engineering canary optimizer values must be finite")
        if self.learning_rate <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or self.adam_eps <= 0 or self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("engineering canary optimizer values are invalid")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_gpu_engineering_run_contract_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            **asdict(self),
        }


def load_engineering_canary_contract(payload: Mapping[str, Any]) -> EngineeringCanaryContract:
    fields = set(EngineeringCanaryContract.__dataclass_fields__)
    metadata = {
        "artifact_kind", "schema_version", "production_unlock_eligible", "f1_unlock_eligible", "run_identity"
    }
    if set(payload) - fields - metadata or fields - set(payload):
        raise ValueError("engineering canary contract fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_gpu_engineering_run_contract_v1"
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
        or payload.get("f1_unlock_eligible") is not False
    ):
        raise ValueError("engineering canary contract boundary is invalid")
    values = {field: payload[field] for field in fields}
    values["adam_betas"] = tuple(values["adam_betas"])
    contract = EngineeringCanaryContract(**values)
    if payload.get("run_identity") not in {None, contract.identity}:
        raise ValueError("engineering canary contract identity mismatch")
    return contract


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    """Scientific identity for one bounded configuration experiment."""

    experiment_id: str
    campaign_hash: str
    data_manifest_hash: str
    validation_view_hash: str
    tokenizer_hash: str
    code_hash: str
    evaluation_manifest_hash: str
    data_order_hash: str
    data_order_policy_hash: str
    schedule_hash: str
    schedule_policy_hash: str
    release_hash: str
    phase: str
    profile_id: str
    learning_rate: float
    token_horizon_parameters: int
    schedule_tail_fraction: float
    seed: int
    paired_order_seed: int
    conditional_branch_evidence_hash: str | None

    def __post_init__(self) -> None:
        _require_identifier(self.experiment_id, "experiment_id")
        for field in (
            "campaign_hash",
            "data_manifest_hash",
            "validation_view_hash",
            "tokenizer_hash",
            "code_hash",
            "evaluation_manifest_hash",
            "data_order_hash",
            "data_order_policy_hash",
            "schedule_hash",
            "schedule_policy_hash",
            "release_hash",
        ):
            _require_hash(getattr(self, field), field)
        if self.phase not in _EXPERIMENT_PHASES:
            raise ValueError("unsupported experiment phase")
        profile = MODEL_PROFILES.get(self.profile_id)
        if profile is None:
            raise ValueError("unknown experiment model profile")
        if profile.train_phase is None:
            raise ValueError("endpoint model profiles are not trainable experiments")
        if profile.train_phase != self.phase:
            raise ValueError("experiment profile is not allowed in phase")
        if type(self.learning_rate) not in {int, float} or self.learning_rate not in _LEARNING_RATES:
            raise ValueError("learning_rate is outside the reviewed search set")
        if type(self.token_horizon_parameters) is not int:
            raise ValueError("token horizon must be an integer parameter multiple")
        allowed_horizons = {
            "s_screen": {8},
            "m_transfer": {8, 16},
            "l_confirmation": {16, 24},
        }[self.phase]
        if self.token_horizon_parameters not in allowed_horizons:
            raise ValueError("token horizon is outside the reviewed phase wave")
        if type(self.schedule_tail_fraction) not in {int, float} or self.schedule_tail_fraction not in {
            0.1,
            0.2,
        }:
            raise ValueError("schedule_tail_fraction must be 0.1 or 0.2")
        _require_nonnegative_int(self.seed, "seed")
        _require_nonnegative_int(self.paired_order_seed, "paired_order_seed")
        if self.schedule_tail_fraction == 0.2:
            if self.phase == "s_screen":
                raise ValueError("conditional schedule branch is not allowed in the S screen")
            _require_hash(self.conditional_branch_evidence_hash, "conditional branch evidence")
        elif self.conditional_branch_evidence_hash is not None:
            raise ValueError("control schedule cannot declare conditional branch evidence")

    @property
    def profile(self) -> ModelProfile:
        return MODEL_PROFILES[self.profile_id]

    @property
    def target_loss_tokens(self) -> int:
        return _PHASE_TOKEN_UNIT[self.phase] * self.token_horizon_parameters

    @property
    def checkpoint_loss_tokens(self) -> tuple[int, ...]:
        return tuple(
            multiple * _PHASE_TOKEN_UNIT[self.phase]
            for multiple in _CHECKPOINT_MULTIPLES
            if multiple <= self.token_horizon_parameters
        )

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_pretraining_experiment_contract_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "accepted_manifest_unlock_eligible": False,
            **asdict(self),
            "parameter_count": self.profile.parameter_count,
            "target_loss_tokens": self.target_loss_tokens,
            "checkpoint_loss_tokens": self.checkpoint_loss_tokens,
        }


def load_experiment_contract(payload: Mapping[str, Any]) -> ExperimentContract:
    fields = set(ExperimentContract.__dataclass_fields__)
    derived = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "accepted_manifest_unlock_eligible",
        "parameter_count",
        "target_loss_tokens",
        "checkpoint_loss_tokens",
        "experiment_identity",
    }
    unknown, missing = set(payload) - fields - derived, fields - set(payload)
    if unknown:
        raise ValueError(f"experiment contract has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"experiment contract is missing fields: {sorted(missing)}")
    if (
        payload.get("artifact_kind") != "twc_pretraining_experiment_contract_v1"
        or payload.get("schema_version") != 1
    ):
        raise ValueError("unsupported experiment contract artifact")
    if payload.get("production_unlock_eligible") is not False:
        raise ValueError("experiment contracts cannot unlock production")
    if payload.get("accepted_manifest_unlock_eligible") is not False:
        raise ValueError("experiment contracts cannot unlock accepted manifests")
    contract = ExperimentContract(**{field: payload[field] for field in fields})
    if payload.get("parameter_count") != contract.profile.parameter_count:
        raise ValueError("experiment parameter_count mismatch")
    if payload.get("target_loss_tokens") != contract.target_loss_tokens:
        raise ValueError("experiment target_loss_tokens mismatch")
    if tuple(payload.get("checkpoint_loss_tokens", ())) != contract.checkpoint_loss_tokens:
        raise ValueError("experiment checkpoint_loss_tokens mismatch")
    if payload.get("experiment_identity") not in {None, contract.identity}:
        raise ValueError("experiment identity mismatch")
    return contract


@dataclass(frozen=True, slots=True)
class ExperimentTrainingContract:
    """Strict experiment compiler output consumed by the shared single-device worker."""

    experiment: ExperimentContract
    view_hash: str
    model_hash: str
    global_step_plan_hash: str
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
    adam_betas: tuple[float, float]
    adam_eps: float
    weight_decay: float
    warmup_loss_tokens: int
    stable_loss_tokens: int
    decay_loss_tokens: int
    grad_clip_norm: float
    full_validation_at_schedule_boundaries: bool

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, ExperimentContract):
            raise ValueError("experiment training contract requires an ExperimentContract")
        for field in ("view_hash", "model_hash", "global_step_plan_hash"):
            _require_hash(getattr(self, field), field)
        for field in (
            "optimizer_step_token_cap", "optimizer_step_attention_cell_cap", "optimizer_step_sequence_cap",
            "warmup_loss_tokens", "stable_loss_tokens", "decay_loss_tokens",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.warmup_loss_tokens + self.stable_loss_tokens + self.decay_loss_tokens != self.target_loss_tokens:
            raise ValueError("experiment training schedule must sum to its exact token horizon")
        expected_decay = int(self.target_loss_tokens * self.experiment.schedule_tail_fraction)
        if self.decay_loss_tokens != expected_decay:
            raise ValueError("experiment decay tokens do not match the predeclared schedule tail")
        if not 0.01 <= self.warmup_loss_tokens / self.target_loss_tokens <= 0.05:
            raise ValueError("experiment warmup must remain within the reviewed 1-5 percent band")
        if self.precision != "bf16" or self.attention_backend != "native_varlen":
            raise ValueError("experiments must exercise BF16 native variable-length attention")
        if not (self.attention_num_splits == "auto" or type(self.attention_num_splits) is int and self.attention_num_splits > 0):
            raise ValueError("experiment attention_num_splits is invalid")
        if self.loss_backend != "chunked_projection" or self.optimizer != "adamw" or self.optimizer_backend != "fused":
            raise ValueError("experiments must use the shared production loss and optimizer family")
        if self.compile_mode not in {"disabled", "default", "reduce-overhead", "max-autotune"}:
            raise ValueError("experiment compile mode is invalid")
        if type(self.allow_tf32) is not bool or self.full_validation_at_schedule_boundaries is not True:
            raise ValueError("experiment boolean safety fields are invalid")
        beta1, beta2 = self.adam_betas
        numeric = (beta1, beta2, self.adam_eps, self.weight_decay, self.grad_clip_norm)
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in numeric):
            raise ValueError("experiment optimizer values must be finite")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or self.adam_eps <= 0 or self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("experiment optimizer values are invalid")

    @property
    def run_id(self) -> str:
        return self.experiment.experiment_id

    @property
    def data_manifest_hash(self) -> str:
        return self.experiment.data_manifest_hash

    @property
    def validation_view_hash(self) -> str:
        return self.experiment.validation_view_hash

    @property
    def tokenizer_hash(self) -> str:
        return self.experiment.tokenizer_hash

    @property
    def code_hash(self) -> str:
        return self.experiment.code_hash

    @property
    def target_loss_tokens(self) -> int:
        return self.experiment.target_loss_tokens

    @property
    def context_length(self) -> int:
        return self.experiment.profile.context_length

    @property
    def learning_rate(self) -> float:
        return self.experiment.learning_rate

    @property
    def seed(self) -> int:
        return self.experiment.seed

    @property
    def expected_parameter_count(self) -> int:
        return self.experiment.profile.parameter_count

    @property
    def model_profile_id(self) -> str:
        return self.experiment.profile_id

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_pretraining_experiment_training_contract_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "experiment": {**self.experiment.to_dict(), "experiment_identity": self.experiment.identity},
            **{field: value for field, value in asdict(self).items() if field != "experiment"},
            "run_id": self.run_id,
            "data_manifest_hash": self.data_manifest_hash,
            "validation_view_hash": self.validation_view_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "code_hash": self.code_hash,
            "target_loss_tokens": self.target_loss_tokens,
            "context_length": self.context_length,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "expected_parameter_count": self.expected_parameter_count,
            "model_profile_id": self.model_profile_id,
        }


def load_experiment_training_contract(payload: Mapping[str, Any]) -> ExperimentTrainingContract:
    direct_fields = set(ExperimentTrainingContract.__dataclass_fields__) - {"experiment"}
    derived = {
        "artifact_kind", "schema_version", "production_unlock_eligible", "f1_unlock_eligible",
        "experiment", "run_identity", "run_id", "data_manifest_hash", "validation_view_hash",
        "tokenizer_hash", "code_hash", "target_loss_tokens", "context_length", "learning_rate",
        "seed", "expected_parameter_count", "model_profile_id",
    }
    if set(payload) - direct_fields - derived or direct_fields - set(payload):
        raise ValueError("experiment training contract fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_pretraining_experiment_training_contract_v1"
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
        or payload.get("f1_unlock_eligible") is not False
        or not isinstance(payload.get("experiment"), Mapping)
    ):
        raise ValueError("experiment training contract boundary is invalid")
    experiment = load_experiment_contract(payload["experiment"])
    values = {field: payload[field] for field in direct_fields}
    values["adam_betas"] = tuple(values["adam_betas"])
    contract = ExperimentTrainingContract(experiment=experiment, **values)
    canonical = contract.to_dict()
    for field in (
        "run_id", "data_manifest_hash", "validation_view_hash", "tokenizer_hash", "code_hash",
        "target_loss_tokens", "context_length", "learning_rate", "seed", "expected_parameter_count",
        "model_profile_id",
    ):
        if payload.get(field) != canonical[field]:
            raise ValueError(f"experiment training derived field mismatch: {field}")
    if payload.get("run_identity") not in {None, contract.identity}:
        raise ValueError("experiment training identity mismatch")
    return contract


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    task_id: str
    selection_role: str
    target_proximity: str
    may_rank_models: bool
    allowed_phase: tuple[str, ...]
    may_invalidate_for_correctness: bool
    dataset_id: str
    dataset_version: str
    item_set_hash: str
    scorer_id: str
    scorer_version: str
    scorer_code_hash: str
    normalization_id: str
    normalization_version: str
    normalization_hash: str
    implementation_status: str
    implementation_receipt_hash: str | None
    provenance_status: str
    contamination_review_status: str

    def __post_init__(self) -> None:
        _require_identifier(self.task_id, "task_id")
        for field in ("dataset_id", "scorer_id", "normalization_id"):
            _require_identifier(getattr(self, field), field)
        for field in ("dataset_version", "scorer_version", "normalization_version"):
            _require_identifier(getattr(self, field), field)
        for field in ("item_set_hash", "scorer_code_hash", "normalization_hash"):
            _require_hash(getattr(self, field), field)
        if self.implementation_status not in {"planned", "implemented_verified"}:
            raise ValueError("unsupported evaluation implementation_status")
        if self.implementation_status == "planned":
            if self.implementation_receipt_hash is not None:
                raise ValueError("planned evaluation task cannot have an implementation receipt")
        else:
            _require_hash(self.implementation_receipt_hash, "implementation_receipt_hash")
        if self.provenance_status != "reviewed":
            raise ValueError("evaluation specification provenance_status must be reviewed")
        if self.contamination_review_status not in {"cleared", "sealed_target"}:
            raise ValueError("unsupported contamination review status")
        if self.selection_role not in {
            "selection_primary",
            "selection_secondary",
            "diagnostic_only",
            "correctness_only",
        }:
            raise ValueError("unsupported evaluation selection_role")
        if self.target_proximity not in {"neutral", "adjacent", "target"}:
            raise ValueError("unsupported evaluation target_proximity")
        if type(self.may_rank_models) is not bool or type(self.may_invalidate_for_correctness) is not bool:
            raise ValueError("evaluation permissions must be booleans")
        if not self.allowed_phase or any(phase not in _EVALUATION_PHASES for phase in self.allowed_phase):
            raise ValueError("evaluation task has an invalid allowed_phase")
        _require_unique(self.allowed_phase, "allowed_phase values")
        if self.target_proximity != "neutral":
            if self.may_rank_models:
                raise ValueError("target-proximate tasks may not rank models")
            if self.selection_role not in {"diagnostic_only", "correctness_only"}:
                raise ValueError("target-proximate tasks must be diagnostic or correctness-only")
            if set(self.allowed_phase) != {"post_freeze_target_diagnostics"}:
                raise ValueError("target-proximate tasks must remain sealed until after configuration freeze")
            if self.contamination_review_status != "sealed_target":
                raise ValueError("target-proximate tasks require sealed_target contamination status")
        elif self.contamination_review_status != "cleared":
            raise ValueError("neutral tasks require cleared contamination review")
        if self.may_rank_models and self.selection_role not in {
            "selection_primary",
            "selection_secondary",
        }:
            raise ValueError("diagnostic or correctness-only tasks may not rank models")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    manifest_id: str
    tasks: tuple[EvaluationTask, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.manifest_id, "manifest_id")
        if not self.tasks or any(not isinstance(task, EvaluationTask) for task in self.tasks):
            raise ValueError("evaluation manifest requires declared tasks")
        _require_unique(tuple(task.task_id for task in self.tasks), "evaluation task IDs")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_pretraining_evaluation_manifest_v2",
            "schema_version": 2,
            "manifest_id": self.manifest_id,
            "tasks": tuple(task.to_dict() for task in self.tasks),
        }

    def validate_use(
        self,
        *,
        phase: str,
        observed_task_ids: Iterable[str],
        ranking_task_ids: Iterable[str],
        correctness_invalidation_task_ids: Iterable[str] = (),
    ) -> None:
        if phase not in _EVALUATION_PHASES:
            raise ValueError("unsupported evaluation phase")
        observed = tuple(observed_task_ids)
        ranking = tuple(ranking_task_ids)
        invalidating = tuple(correctness_invalidation_task_ids)
        _require_unique(observed, "observed task IDs")
        _require_unique(ranking, "ranking task IDs")
        _require_unique(invalidating, "correctness invalidation task IDs")
        tasks = {task.task_id: task for task in self.tasks}
        undeclared = (set(observed) | set(ranking) | set(invalidating)) - set(tasks)
        if undeclared:
            raise ValueError(f"evaluation use contains undeclared tasks: {sorted(undeclared)}")
        if not set(ranking).issubset(observed):
            raise ValueError("ranking tasks must be present in observed tasks")
        if not set(invalidating).issubset(observed):
            raise ValueError("correctness invalidation tasks must be present in observed tasks")
        for task_id in observed:
            if tasks[task_id].implementation_status != "implemented_verified":
                raise ValueError(f"evaluation task {task_id} is not implemented and verified")
            if phase not in tasks[task_id].allowed_phase:
                raise ValueError(f"evaluation task {task_id} is not allowed in phase {phase}")
        for task_id in ranking:
            if not tasks[task_id].may_rank_models:
                raise ValueError(f"evaluation task {task_id} may not rank models")
        for task_id in invalidating:
            if not tasks[task_id].may_invalidate_for_correctness:
                raise ValueError(f"evaluation task {task_id} may not invalidate for correctness")


def load_evaluation_manifest(payload: Mapping[str, Any]) -> EvaluationManifest:
    expected = {"artifact_kind", "schema_version", "manifest_id", "tasks"}
    if set(payload) != expected:
        raise ValueError("evaluation manifest fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_pretraining_evaluation_manifest_v2"
        or payload.get("schema_version") != 2
        or not isinstance(payload.get("tasks"), (list, tuple))
    ):
        raise ValueError("evaluation manifest artifact is invalid")
    fields = set(EvaluationTask.__dataclass_fields__)
    tasks: list[EvaluationTask] = []
    for row in payload["tasks"]:
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError("evaluation task fields are invalid")
        values = dict(row)
        values["allowed_phase"] = tuple(values["allowed_phase"])
        tasks.append(EvaluationTask(**values))
    return EvaluationManifest(manifest_id=str(payload["manifest_id"]), tasks=tuple(tasks))


def compile_evaluation_manifest_yaml(path: Path) -> EvaluationManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "version", "manifest_id", "used_for_training", "evaluator_implementation_status", "tasks"
    }:
        raise ValueError("evaluation YAML fields are invalid")
    if payload["version"] != 2 or payload["used_for_training"] is not False:
        raise ValueError("evaluation YAML version/training boundary is invalid")
    rows = payload["tasks"]
    if not isinstance(rows, list):
        raise ValueError("evaluation YAML tasks must be a list")
    fields = {
        "task_id", "selection_role", "target_proximity", "may_rank_models",
        "allowed_phase", "may_invalidate_for_correctness",
        "dataset_id", "dataset_version", "item_set_hash", "scorer_id", "scorer_version",
        "scorer_code_hash", "normalization_id", "normalization_version", "normalization_hash",
        "provenance_status", "contamination_review_status",
    }
    tasks: list[EvaluationTask] = []
    def yaml_hash(value: object) -> object:
        # YAML parses unquoted all-digit SHA-256 values as integers. Restore their
        # fixed-width lexical form before the normal strict hash validator runs.
        return f"{value:064d}" if type(value) is int and value >= 0 else value

    for row in rows:
        if not isinstance(row, dict) or set(row) != fields or not isinstance(row["allowed_phase"], list):
            raise ValueError("evaluation YAML task fields are invalid")
        tasks.append(
            EvaluationTask(
                task_id=row["task_id"],
                selection_role=row["selection_role"],
                target_proximity=row["target_proximity"],
                may_rank_models=row["may_rank_models"],
                allowed_phase=tuple(row["allowed_phase"]),
                may_invalidate_for_correctness=row["may_invalidate_for_correctness"],
                dataset_id=row["dataset_id"],
                dataset_version=row["dataset_version"],
                item_set_hash=yaml_hash(row["item_set_hash"]),
                scorer_id=row["scorer_id"],
                scorer_version=row["scorer_version"],
                scorer_code_hash=yaml_hash(row["scorer_code_hash"]),
                normalization_id=row["normalization_id"],
                normalization_version=row["normalization_version"],
                normalization_hash=yaml_hash(row["normalization_hash"]),
                implementation_status=payload["evaluator_implementation_status"],
                implementation_receipt_hash=None,
                provenance_status=row["provenance_status"],
                contamination_review_status=row["contamination_review_status"],
            )
        )
    return EvaluationManifest(manifest_id=payload["manifest_id"], tasks=tuple(tasks))


@dataclass(frozen=True, slots=True)
class EvaluationTaskImplementationReceipt:
    """Evidence that turns one inert task specification into an executable task."""

    task_id: str
    planned_task_hash: str
    release_hash: str
    item_set_hash: str
    scorer_code_hash: str
    normalization_hash: str
    provenance_evidence_hash: str
    contamination_review_evidence_hash: str
    implementation_test_evidence_hash: str

    def __post_init__(self) -> None:
        _require_identifier(self.task_id, "task_id")
        for field in (
            "planned_task_hash", "release_hash", "item_set_hash", "scorer_code_hash",
            "normalization_hash", "provenance_evidence_hash",
            "contamination_review_evidence_hash", "implementation_test_evidence_hash",
        ):
            _require_hash(getattr(self, field), field)

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_evaluation_task_implementation_receipt_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            **asdict(self),
        }


def load_evaluation_task_implementation_receipt(
    payload: Mapping[str, Any],
) -> EvaluationTaskImplementationReceipt:
    fields = set(EvaluationTaskImplementationReceipt.__dataclass_fields__)
    metadata = {"artifact_kind", "schema_version", "production_unlock_eligible"}
    if set(payload) != fields | metadata:
        raise ValueError("evaluation implementation receipt fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_evaluation_task_implementation_receipt_v1"
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
    ):
        raise ValueError("evaluation implementation receipt artifact is invalid")
    return EvaluationTaskImplementationReceipt(
        **{field: payload[field] for field in fields}
    )


def bind_evaluation_task(
    manifest: EvaluationManifest,
    receipt: EvaluationTaskImplementationReceipt,
) -> EvaluationManifest:
    if not isinstance(manifest, EvaluationManifest) or not isinstance(
        receipt, EvaluationTaskImplementationReceipt
    ):
        raise ValueError("evaluation binding requires a manifest and implementation receipt")
    tasks = {task.task_id: task for task in manifest.tasks}
    task = tasks.get(receipt.task_id)
    if task is None:
        raise ValueError("implementation receipt task is absent from evaluator manifest")
    if task.implementation_status != "planned":
        raise ValueError("implementation receipt can bind only a planned task")
    if task.identity != receipt.planned_task_hash:
        raise ValueError("implementation receipt planned task identity mismatch")
    bound = replace(
        task,
        item_set_hash=receipt.item_set_hash,
        scorer_code_hash=receipt.scorer_code_hash,
        normalization_hash=receipt.normalization_hash,
        implementation_status="implemented_verified",
        implementation_receipt_hash=receipt.identity,
    )
    return replace(
        manifest,
        tasks=tuple(bound if item.task_id == bound.task_id else item for item in manifest.tasks),
    )


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """A report vocabulary that prevents an observation becoming a causal claim."""

    evidence_hash: str
    classification: str
    observed_effect: str
    plausible_mechanisms: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    contradictory_evidence: tuple[str, ...]
    remaining_uncertainty: str
    scaling_outlook: str

    def __post_init__(self) -> None:
        _require_hash(self.evidence_hash, "evidence_hash")
        if self.classification not in _EVIDENCE_CLASSIFICATIONS:
            raise ValueError("unsupported evidence classification")
        for field in ("observed_effect", "remaining_uncertainty", "scaling_outlook"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty statement")
        for field in ("plausible_mechanisms", "alternative_explanations", "contradictory_evidence"):
            values = getattr(self, field)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{field} entries must be non-empty statements")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_pretraining_evidence_assessment_v1",
            "schema_version": 1,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class TaskAggregateReceipt:
    """Hash-bound per-item scores produced by one immutable evaluator task."""

    task_id: str
    evaluation_manifest_hash: str
    task_identity_hash: str
    item_set_hash: str
    scorer_code_hash: str
    normalization_hash: str
    scores: tuple[float, ...]
    weights: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.task_id, "task_id")
        for field in (
            "evaluation_manifest_hash", "task_identity_hash", "item_set_hash",
            "scorer_code_hash", "normalization_hash",
        ):
            _require_hash(getattr(self, field), field)
        if len(self.scores) < 2:
            raise ValueError("task aggregates require at least two item scores")
        if len(self.weights) != len(self.scores):
            raise ValueError("task aggregate weights must align with scores")
        for index, score in enumerate(self.scores):
            _require_finite(score, f"scores[{index}]")
        for index, weight in enumerate(self.weights):
            if type(weight) is not int or weight <= 0:
                raise ValueError(f"weights[{index}] must be a positive integer")

    @classmethod
    def from_scores(
        cls,
        manifest: EvaluationManifest,
        task_id: str,
        scores: Iterable[float],
        *,
        weights: Iterable[int] | None = None,
    ) -> TaskAggregateReceipt:
        if not isinstance(manifest, EvaluationManifest):
            raise ValueError("task aggregate requires a reviewed evaluator manifest")
        tasks = {task.task_id: task for task in manifest.tasks}
        if task_id not in tasks:
            raise ValueError("task aggregate task is absent from evaluator manifest")
        task = tasks[task_id]
        parsed_scores = tuple(float(score) for score in scores)
        parsed_weights = (
            tuple(weights) if weights is not None else tuple(1 for _ in parsed_scores)
        )
        return cls(
            task_id=task.task_id,
            evaluation_manifest_hash=manifest.identity,
            task_identity_hash=task.identity,
            item_set_hash=task.item_set_hash,
            scorer_code_hash=task.scorer_code_hash,
            normalization_hash=task.normalization_hash,
            scores=parsed_scores,
            weights=parsed_weights,
        )

    @property
    def mean(self) -> float:
        return sum(score * weight for score, weight in zip(self.scores, self.weights, strict=True)) / sum(self.weights)

    @property
    def identity(self) -> str:
        return stable_json_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class ExperimentResultReceipt:
    """Measured outputs bound to the experiment, release, data order, and evaluator."""

    result_id: str
    contract_hash: str
    profile_id: str
    phase: str
    seed: int
    paired_order_seed: int
    tokenizer_hash: str
    data_manifest_hash: str
    data_order_hash: str
    validation_split_hash: str
    token_horizon_parameters: int
    schedule_hash: str
    evaluator_manifest_hash: str
    code_hash: str
    release_hash: str
    evaluation_receipt_hashes: tuple[str, ...]
    evaluation_curve_loss_tokens: tuple[int, ...]
    evaluation_curve_losses: tuple[float, ...]
    task_aggregates: tuple[TaskAggregateReceipt, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.result_id, "result_id")
        _require_identifier(self.profile_id, "profile_id")
        if self.phase not in _EXPERIMENT_PHASES:
            raise ValueError("result receipt phase is invalid")
        _require_nonnegative_int(self.seed, "seed")
        _require_nonnegative_int(self.paired_order_seed, "paired_order_seed")
        if self.token_horizon_parameters not in _CHECKPOINT_MULTIPLES:
            raise ValueError("result token horizon is invalid")
        for field in (
            "contract_hash", "tokenizer_hash", "data_manifest_hash", "data_order_hash",
            "validation_split_hash", "schedule_hash", "evaluator_manifest_hash", "code_hash",
            "release_hash",
        ):
            _require_hash(getattr(self, field), field)
        if not self.evaluation_receipt_hashes:
            raise ValueError("result receipt requires an evaluation chain")
        for index, value in enumerate(self.evaluation_receipt_hashes):
            _require_hash(value, f"evaluation_receipt_hashes[{index}]")
        if not (
            len(self.evaluation_receipt_hashes)
            == len(self.evaluation_curve_loss_tokens)
            == len(self.evaluation_curve_losses)
        ):
            raise ValueError("result evaluation curve fields must align")
        previous_tokens = -1
        for index, (tokens, loss) in enumerate(
            zip(self.evaluation_curve_loss_tokens, self.evaluation_curve_losses, strict=True)
        ):
            _require_nonnegative_int(tokens, f"evaluation_curve_loss_tokens[{index}]")
            if tokens <= previous_tokens:
                raise ValueError("result evaluation curve token positions must increase")
            _require_finite(loss, f"evaluation_curve_losses[{index}]", nonnegative=True)
            previous_tokens = tokens
        if not self.task_aggregates or any(not isinstance(item, TaskAggregateReceipt) for item in self.task_aggregates):
            raise ValueError("result receipt requires task aggregates")
        _require_unique(tuple(item.task_id for item in self.task_aggregates), "result task IDs")
        validation = next(
            (item for item in self.task_aggregates if item.task_id == "validation_nll"),
            None,
        )
        if validation is not None and not math.isclose(
            self.evaluation_curve_losses[-1], validation.mean, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError("result terminal learning-curve loss must equal validation NLL")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_pretraining_experiment_result_v2",
            "schema_version": 2,
            **asdict(self),
        }


def load_experiment_result_receipt(payload: Mapping[str, Any]) -> ExperimentResultReceipt:
    fields = set(ExperimentResultReceipt.__dataclass_fields__)
    metadata = {"artifact_kind", "schema_version"}
    if set(payload) != fields | metadata:
        raise ValueError("experiment result fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_pretraining_experiment_result_v2"
        or payload.get("schema_version") != 2
        or not isinstance(payload.get("task_aggregates"), (list, tuple))
    ):
        raise ValueError("experiment result artifact is invalid")
    task_fields = set(TaskAggregateReceipt.__dataclass_fields__)
    tasks: list[TaskAggregateReceipt] = []
    for row in payload["task_aggregates"]:
        if not isinstance(row, Mapping) or set(row) != task_fields:
            raise ValueError("experiment result task aggregate fields are invalid")
        values = dict(row)
        values["scores"] = tuple(values["scores"])
        values["weights"] = tuple(values["weights"])
        tasks.append(TaskAggregateReceipt(**values))
    values = {field: payload[field] for field in fields}
    values["evaluation_receipt_hashes"] = tuple(values["evaluation_receipt_hashes"])
    values["evaluation_curve_loss_tokens"] = tuple(values["evaluation_curve_loss_tokens"])
    values["evaluation_curve_losses"] = tuple(values["evaluation_curve_losses"])
    values["task_aggregates"] = tuple(tasks)
    return ExperimentResultReceipt(**values)


def build_validation_nll_result_receipt(
    *,
    result_id: str,
    training_contract: ExperimentTrainingContract,
    evaluation_manifest: EvaluationManifest,
    implementation_receipt: EvaluationTaskImplementationReceipt,
    execution_attempt: Any,
    evaluation_rows: Iterable[Mapping[str, Any]],
    worker_result: Mapping[str, Any],
    operator_state: Mapping[str, Any],
) -> ExperimentResultReceipt:
    """Convert a terminal worker evaluation into non-self-asserted experiment evidence."""

    if not isinstance(training_contract, ExperimentTrainingContract):
        raise ValueError("validation result requires an experiment training contract")
    experiment = training_contract.experiment
    from twc.pretraining.training.evaluation_evidence import validate_evaluation_chain

    try:
        attempt_payload = execution_attempt.to_dict()
        attempt_identity = execution_attempt.identity
    except (AttributeError, TypeError) as error:
        raise ValueError("validation result requires a validated execution attempt") from error
    if (
        not isinstance(attempt_payload, dict)
        or attempt_payload.get("artifact_kind") != "twc_pretraining_execution_attempt_v2"
        or attempt_payload.get("schema_version") != 2
        or attempt_identity != stable_json_hash(attempt_payload)
    ):
        raise ValueError("validation result requires a validated execution attempt")
    if (
        execution_attempt.run_identity != training_contract.identity
        or execution_attempt.release_hash != experiment.release_hash
    ):
        raise ValueError("validation execution attempt does not bind the experiment release")
    if evaluation_manifest.identity != experiment.evaluation_manifest_hash:
        raise ValueError("validation evaluator manifest does not match the experiment")
    tasks = {task.task_id: task for task in evaluation_manifest.tasks}
    task = tasks.get("validation_nll")
    if task is None or task.implementation_receipt_hash != implementation_receipt.identity:
        raise ValueError("validation task implementation receipt mismatch")
    if (
        implementation_receipt.task_id != task.task_id
        or implementation_receipt.release_hash != experiment.release_hash
        or implementation_receipt.item_set_hash != task.item_set_hash
        or implementation_receipt.scorer_code_hash != task.scorer_code_hash
        or implementation_receipt.normalization_hash != task.normalization_hash
    ):
        raise ValueError("validation task implementation identity mismatch")
    evaluation_manifest.validate_use(
        phase=experiment.phase,
        observed_task_ids=(task.task_id,),
        ranking_task_ids=(task.task_id,),
    )
    rows = [dict(row) for row in evaluation_rows]
    evaluation_row = validate_evaluation_chain(
        rows,
        training_contract.identity,
        execution_attempt.identity,
        execution_attempt.fencing_epoch,
    )
    if (
        evaluation_row.get("artifact_kind") != "twc_pretraining_evaluation_result_v2"
        or evaluation_row.get("schema_version") != 2
        or evaluation_row.get("at_consumed_loss_tokens") != experiment.target_loss_tokens
        or evaluation_row.get("loss_tokens") != 8_388_608
    ):
        raise ValueError("terminal evaluation row is not bound to the completed experiment")
    if (
        worker_result.get("artifact_kind") != "twc_pretraining_worker_result_v2"
        or worker_result.get("schema_version") != 2
        or worker_result.get("result_hash")
        != stable_json_hash({key: value for key, value in worker_result.items() if key != "result_hash"})
        or worker_result.get("run_identity") != training_contract.identity
        or worker_result.get("attempt_hash") != execution_attempt.identity
        or worker_result.get("fencing_epoch") != execution_attempt.fencing_epoch
        or worker_result.get("completed") is not True
        or worker_result.get("consumed_loss_tokens") != experiment.target_loss_tokens
        or worker_result.get("evaluation_receipt_hash") != evaluation_row.get("receipt_hash")
        or worker_result.get("evaluation_loss_tokens") != 8_388_608
        or worker_result.get("evaluation_result_batch_count") != evaluation_row.get("batch_count")
    ):
        raise ValueError("terminal worker result is invalid or not evaluation-bound")
    if (
        operator_state.get("artifact_kind") != "twc_pretraining_operator_state_v2"
        or operator_state.get("schema_version") != 2
        or operator_state.get("state_hash")
        != stable_json_hash({key: value for key, value in operator_state.items() if key != "state_hash"})
        or operator_state.get("state") != "COMPLETED"
        or operator_state.get("run_identity") != training_contract.identity
        or operator_state.get("attempt_hash") != execution_attempt.identity
        or operator_state.get("fencing_epoch") != execution_attempt.fencing_epoch
        or operator_state.get("step") != worker_result.get("step")
        or operator_state.get("next_global_ordinal") != worker_result.get("next_global_ordinal")
    ):
        raise ValueError("terminal operator state is invalid or not worker-bound")
    scores = evaluation_row.get("component_losses")
    weights = evaluation_row.get("component_loss_tokens")
    if (
        not isinstance(scores, list)
        or not isinstance(weights, list)
        or len(scores) < 2
        or len(scores) != evaluation_row.get("batch_count")
        or len(weights) != len(scores)
        or any(type(value) not in {int, float} or not math.isfinite(value) or value < 0 for value in scores)
        or any(type(value) is not int or value <= 0 for value in weights)
        or sum(weights) != evaluation_row.get("loss_tokens")
    ):
        raise ValueError("terminal evaluation component metrics are invalid")
    weighted_loss = sum(value * weight for value, weight in zip(scores, weights, strict=True)) / sum(weights)
    if not math.isclose(weighted_loss, float(evaluation_row.get("loss")), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("terminal evaluation weighted loss mismatch")
    return ExperimentResultReceipt(
        result_id=result_id,
        contract_hash=experiment.identity,
        profile_id=experiment.profile_id,
        phase=experiment.phase,
        seed=experiment.seed,
        paired_order_seed=experiment.paired_order_seed,
        tokenizer_hash=experiment.tokenizer_hash,
        data_manifest_hash=experiment.data_manifest_hash,
        data_order_hash=experiment.data_order_hash,
        validation_split_hash=experiment.validation_view_hash,
        token_horizon_parameters=experiment.token_horizon_parameters,
        schedule_hash=experiment.schedule_hash,
        evaluator_manifest_hash=evaluation_manifest.identity,
        code_hash=experiment.code_hash,
        release_hash=experiment.release_hash,
        evaluation_receipt_hashes=tuple(str(row["receipt_hash"]) for row in rows),
        evaluation_curve_loss_tokens=tuple(
            int(row["at_consumed_loss_tokens"]) for row in rows
        ),
        evaluation_curve_losses=tuple(float(row["loss"]) for row in rows),
        task_aggregates=(
            TaskAggregateReceipt.from_scores(
                evaluation_manifest,
                task.task_id,
                scores,
                weights=weights,
            ),
        ),
    )


def _validate_result_contract(result: ExperimentResultReceipt, contract: ExperimentContract) -> None:
    expected = {
        "contract_hash": contract.identity,
        "profile_id": contract.profile_id,
        "phase": contract.phase,
        "seed": contract.seed,
        "paired_order_seed": contract.paired_order_seed,
        "tokenizer_hash": contract.tokenizer_hash,
        "data_manifest_hash": contract.data_manifest_hash,
        "data_order_hash": contract.data_order_hash,
        "validation_split_hash": contract.validation_view_hash,
        "token_horizon_parameters": contract.token_horizon_parameters,
        "evaluator_manifest_hash": contract.evaluation_manifest_hash,
        "code_hash": contract.code_hash,
        "schedule_hash": contract.schedule_hash,
        "release_hash": contract.release_hash,
    }
    for field, value in expected.items():
        if getattr(result, field) != value:
            raise ValueError(f"result receipt {field} does not match its experiment contract")
    if result.evaluation_curve_loss_tokens[-1] != contract.target_loss_tokens:
        raise ValueError("result receipt evaluation curve is not terminal")


@dataclass(frozen=True, slots=True)
class NeutralFamilyEffect:
    family_id: str
    matched_score_improvement: float
    paired_standard_error: float

    def __post_init__(self) -> None:
        _require_identifier(self.family_id, "family_id")
        _require_finite(self.matched_score_improvement, "matched_score_improvement")
        _require_finite(self.paired_standard_error, "paired_standard_error", nonnegative=True)

    @property
    def clears_threshold(self) -> bool:
        return self.matched_score_improvement >= max(0.02, 2 * self.paired_standard_error)


@dataclass(frozen=True, slots=True)
class ConfirmationSeedEvidence:
    seed: int
    paired_comparison_hash: str
    candidate_contract_hash: str
    control_contract_hash: str
    relative_nll_improvement: float
    nll_paired_standard_error: float
    neutral_family_effects: tuple[NeutralFamilyEffect, ...]
    learnability_not_worse: bool
    component_regression: bool

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.seed, "seed")
        _require_hash(self.paired_comparison_hash, "paired_comparison_hash")
        _require_hash(self.candidate_contract_hash, "candidate_contract_hash")
        _require_hash(self.control_contract_hash, "control_contract_hash")
        _require_finite(self.relative_nll_improvement, "relative_nll_improvement")
        _require_finite(
            self.nll_paired_standard_error,
            "nll_paired_standard_error",
            nonnegative=True,
        )
        if any(not isinstance(item, NeutralFamilyEffect) for item in self.neutral_family_effects):
            raise ValueError("neutral_family_effects must contain NeutralFamilyEffect values")
        _require_unique(
            tuple(item.family_id for item in self.neutral_family_effects),
            "neutral capability family IDs",
        )
        if type(self.learnability_not_worse) is not bool or type(self.component_regression) is not bool:
            raise ValueError("confirmation guard values must be booleans")

    @property
    def clears_promotion_threshold(self) -> bool:
        if not self.learnability_not_worse or self.component_regression:
            return False
        nll_clears = self.relative_nll_improvement >= max(
            0.005,
            2 * self.nll_paired_standard_error,
        )
        capability_clears = (
            self.relative_nll_improvement >= 0
            and sum(effect.clears_threshold for effect in self.neutral_family_effects) >= 2
        )
        return nll_clears or capability_clears


@dataclass(frozen=True, slots=True)
class PairedComparisonReceipt:
    """A matched candidate/control comparison whose effects are recomputed from scores."""

    candidate_result: ExperimentResultReceipt
    control_result: ExperimentResultReceipt
    candidate_contract: ExperimentContract
    control_contract: ExperimentContract
    evaluation_manifest: EvaluationManifest
    ranking_task_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_manifest, EvaluationManifest):
            raise ValueError("paired comparison requires an evaluation manifest")
        _require_unique(self.ranking_task_ids, "ranking task IDs")
        if not self.ranking_task_ids:
            raise ValueError("paired comparison requires ranking tasks")
        _validate_result_contract(self.candidate_result, self.candidate_contract)
        _validate_result_contract(self.control_result, self.control_contract)
        if self.evaluation_manifest.identity != self.candidate_result.evaluator_manifest_hash:
            raise ValueError("paired comparison evaluator manifest mismatch")
        self.evaluation_manifest.validate_use(
            phase=self.candidate_result.phase,
            observed_task_ids=self.ranking_task_ids,
            ranking_task_ids=self.ranking_task_ids,
        )
        expected_profiles = {
            "s_screen": ("s_wide", "s_deep"),
            "m_transfer": ("m_wide", "m_deep"),
            "l_confirmation": ("l_wide", "l_deep"),
        }
        if self.candidate_contract.phase not in expected_profiles:
            raise ValueError("paired evidence must be S, M, or L scale")
        if (self.candidate_contract.profile_id, self.control_contract.profile_id) != expected_profiles[self.candidate_contract.phase]:
            raise ValueError("paired comparison candidate/control profiles are invalid")
        matched_fields = (
            "phase", "seed", "paired_order_seed", "tokenizer_hash", "data_manifest_hash",
            "data_order_hash", "validation_split_hash", "token_horizon_parameters", "schedule_hash",
            "evaluator_manifest_hash", "code_hash", "release_hash",
        )
        for field in matched_fields:
            if getattr(self.candidate_result, field) != getattr(self.control_result, field):
                raise ValueError(f"paired comparison {field} mismatch")
        for field in ("data_order_policy_hash", "schedule_policy_hash"):
            if getattr(self.candidate_contract, field) != getattr(self.control_contract, field):
                raise ValueError(f"paired comparison {field} mismatch")
        if self.candidate_contract.learning_rate != self.control_contract.learning_rate:
            raise ValueError("paired comparison learning_rate mismatch")
        candidate_tasks = {item.task_id: item for item in self.candidate_result.task_aggregates}
        control_tasks = {item.task_id: item for item in self.control_result.task_aggregates}
        manifest_tasks = {item.task_id: item for item in self.evaluation_manifest.tasks}
        if set(candidate_tasks) != set(control_tasks) or not set(self.ranking_task_ids).issubset(candidate_tasks):
            raise ValueError("paired comparison task sets mismatch")
        for task_id in candidate_tasks:
            candidate = candidate_tasks[task_id]
            control = control_tasks[task_id]
            task = manifest_tasks.get(task_id)
            if task is None:
                raise ValueError("paired comparison task is absent from evaluator manifest")
            if candidate.evaluation_manifest_hash != self.evaluation_manifest.identity:
                raise ValueError("paired comparison task evaluator manifest mismatch")
            if candidate.task_identity_hash != task.identity:
                raise ValueError("paired comparison task identity does not match evaluator manifest")
            for field in ("item_set_hash", "scorer_code_hash", "normalization_hash"):
                if getattr(candidate, field) != getattr(task, field):
                    raise ValueError(
                        f"paired comparison task {field} does not match evaluator manifest"
                    )
            for field in (
                "evaluation_manifest_hash", "task_identity_hash", "item_set_hash",
                "scorer_code_hash", "normalization_hash", "weights",
            ):
                if getattr(candidate, field) != getattr(control, field):
                    raise ValueError(f"paired comparison task {field} mismatch")
            if len(candidate.scores) != len(control.scores):
                raise ValueError("paired comparison item count mismatch")

    @property
    def identity(self) -> str:
        # Derived effects are a pure function of these bound inputs. Excluding
        # the redundant summary avoids a circular evidence-hash definition.
        return stable_json_hash(
            {
                "artifact_kind": "twc_pretraining_paired_comparison_v1",
                "schema_version": 1,
                **self.to_dict_without_evidence(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_pretraining_paired_comparison_v1",
            "schema_version": 1,
            "candidate_result_hash": self.candidate_result.identity,
            "control_result_hash": self.control_result.identity,
            "candidate_contract_hash": self.candidate_contract.identity,
            "control_contract_hash": self.control_contract.identity,
            "evaluation_manifest_hash": self.evaluation_manifest.identity,
            "ranking_task_ids": self.ranking_task_ids,
            "derived_evidence": asdict(self.evidence),
        }

    def _task_pair(self, task_id: str) -> tuple[TaskAggregateReceipt, TaskAggregateReceipt]:
        candidate = {item.task_id: item for item in self.candidate_result.task_aggregates}[task_id]
        control = {item.task_id: item for item in self.control_result.task_aggregates}[task_id]
        return candidate, control

    @staticmethod
    def _weighted_paired_cluster_se(
        values: tuple[float, ...], weights: tuple[int, ...]
    ) -> float:
        """Kish-effective-sample SE over independent evaluation batches/items."""
        if len(values) < 2:
            return math.inf
        if len(values) != len(weights):
            raise ValueError("paired uncertainty weights must align with values")
        total_weight = sum(weights)
        squared_weight_sum = sum(weight * weight for weight in weights)
        effective_count = total_weight * total_weight / squared_weight_sum
        if effective_count <= 1:
            return math.inf
        mean = sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight
        population_variance = sum(
            weight * (value - mean) ** 2
            for value, weight in zip(values, weights, strict=True)
        ) / total_weight
        sample_variance = population_variance * effective_count / (effective_count - 1)
        return math.sqrt(sample_variance / effective_count)

    @property
    def evidence(self) -> ConfirmationSeedEvidence:
        candidate_nll, control_nll = self._task_pair("validation_nll")
        if control_nll.mean <= 0:
            raise ValueError("control validation NLL must be positive")
        relative = (control_nll.mean - candidate_nll.mean) / control_nll.mean
        relative_differences = tuple(
            (control - candidate) / control_nll.mean
            for candidate, control in zip(candidate_nll.scores, control_nll.scores, strict=True)
        )
        nll_se = self._weighted_paired_cluster_se(
            relative_differences, candidate_nll.weights
        )
        families: list[NeutralFamilyEffect] = []
        component_regression = False
        for task_id in self.ranking_task_ids:
            if task_id == "validation_nll":
                continue
            candidate, control = self._task_pair(task_id)
            differences = tuple(c - k for c, k in zip(candidate.scores, control.scores, strict=True))
            weighted_effect = sum(
                value * weight
                for value, weight in zip(differences, candidate.weights, strict=True)
            ) / sum(candidate.weights)
            effect = NeutralFamilyEffect(
                task_id,
                weighted_effect,
                self._weighted_paired_cluster_se(differences, candidate.weights),
            )
            families.append(effect)
            if effect.matched_score_improvement < -max(0.02, 2 * effect.paired_standard_error):
                component_regression = True
        learnability_not_worse = relative >= -2 * nll_se
        provisional = ConfirmationSeedEvidence(
            seed=self.candidate_result.seed,
            paired_comparison_hash="0" * 64,
            candidate_contract_hash=self.candidate_contract.identity,
            control_contract_hash=self.control_contract.identity,
            relative_nll_improvement=relative,
            nll_paired_standard_error=nll_se,
            neutral_family_effects=tuple(families),
            learnability_not_worse=learnability_not_worse,
            component_regression=component_regression,
        )
        return replace(provisional, paired_comparison_hash=self.identity)

    def to_dict_without_evidence(self) -> dict[str, Any]:
        return {
            "candidate_result_hash": self.candidate_result.identity,
            "control_result_hash": self.control_result.identity,
            "candidate_contract_hash": self.candidate_contract.identity,
            "control_contract_hash": self.control_contract.identity,
            "evaluation_manifest_hash": self.evaluation_manifest.identity,
            "ranking_task_ids": self.ranking_task_ids,
        }


@dataclass(frozen=True, slots=True)
class ConfigurationScreenDecisionReceipt:
    """Predeclared C1 reduction to one matched learning-rate pair for M scale."""

    screen_id: str
    campaign_hash: str
    comparisons: tuple[PairedComparisonReceipt, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.screen_id, "screen_id")
        _require_hash(self.campaign_hash, "campaign_hash")
        if len(self.comparisons) != 3 or any(
            not isinstance(item, PairedComparisonReceipt) for item in self.comparisons
        ):
            raise ValueError("C1 screen requires exactly three paired comparisons")
        learning_rates = tuple(
            item.candidate_contract.learning_rate for item in self.comparisons
        )
        if set(learning_rates) != {4e-4, 6e-4, 9e-4}:
            raise ValueError("C1 screen requires the reviewed learning-rate set")
        first = self.comparisons[0]
        common_contract_fields = (
            "campaign_hash", "data_manifest_hash", "validation_view_hash",
            "tokenizer_hash", "code_hash", "evaluation_manifest_hash",
            "data_order_hash", "data_order_policy_hash", "schedule_hash",
            "schedule_policy_hash", "release_hash", "phase", "seed",
            "paired_order_seed", "token_horizon_parameters",
        )
        for comparison in self.comparisons:
            if comparison.candidate_contract.phase != "s_screen":
                raise ValueError("C1 decision accepts only S-screen evidence")
            if comparison.ranking_task_ids != ("validation_nll",):
                raise ValueError("C1 screen ranks only validation NLL")
            if any(
                contract.campaign_hash != self.campaign_hash
                for contract in (
                    comparison.candidate_contract,
                    comparison.control_contract,
                )
            ):
                raise ValueError("C1 screen comparison campaign hash mismatch")
            if any(
                getattr(comparison.candidate_contract, field)
                != getattr(first.candidate_contract, field)
                for field in common_contract_fields
            ):
                raise ValueError("C1 screen comparisons are not scientifically matched")
            for result in (comparison.candidate_result, comparison.control_result):
                if len(result.evaluation_curve_losses) < 3:
                    raise ValueError("C1 screen requires at least three learning-curve observations")

    @property
    def _selected(self) -> PairedComparisonReceipt:
        def worst_final_nll(comparison: PairedComparisonReceipt) -> tuple[float, float]:
            candidate, control = comparison._task_pair("validation_nll")
            return (
                max(candidate.mean, control.mean),
                comparison.candidate_contract.learning_rate,
            )

        return min(self.comparisons, key=worst_final_nll)

    @property
    def selected_learning_rate(self) -> float:
        return self._selected.candidate_contract.learning_rate

    @property
    def selected_profile_ids(self) -> tuple[str, str]:
        return (
            self._selected.candidate_contract.profile_id,
            self._selected.control_contract.profile_id,
        )

    @property
    def m_transfer_authorized(self) -> bool:
        return all(
            result.evaluation_curve_losses[-1] < result.evaluation_curve_losses[0]
            for comparison in self.comparisons
            for result in (comparison.candidate_result, comparison.control_result)
        )

    @property
    def claim_status(self) -> str:
        return (
            "practical_scaling_observation_not_causal"
            if self.m_transfer_authorized
            else "screen_blocked_unstable_or_nonlearning"
        )

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_screen_decision_v1",
            "schema_version": 1,
            "screen_id": self.screen_id,
            "campaign_hash": self.campaign_hash,
            "comparison_hashes": tuple(item.identity for item in self.comparisons),
            "selection_rule": "lowest_worst_member_final_validation_nll_then_lower_lr",
            "selected_learning_rate": self.selected_learning_rate,
            "selected_profile_ids": self.selected_profile_ids,
            "m_transfer_authorized": self.m_transfer_authorized,
            "claim_status": self.claim_status,
            "production_run_authorized": False,
            "f1_authorized": False,
        }


@dataclass(frozen=True, slots=True)
class ConfigurationFreezeDecisionReceipt:
    freeze_id: str
    campaign_hash: str
    decision: str
    selected_profile_id: str
    medium_comparison: PairedComparisonReceipt
    confirmation_comparisons: tuple[PairedComparisonReceipt, ...]
    evaluation_manifest_hash: str
    ranking_task_ids: tuple[str, ...]
    target_diagnostic_task_ids_observed: tuple[str, ...]
    evidence_classification: str

    def __post_init__(self) -> None:
        _require_identifier(self.freeze_id, "freeze_id")
        for field in (
            "campaign_hash",
            "evaluation_manifest_hash",
        ):
            _require_hash(getattr(self, field), field)
        if self.decision not in {"promote_challenger", "retain_r7_control"}:
            raise ValueError("unsupported configuration freeze decision")
        if not isinstance(self.medium_comparison, PairedComparisonReceipt):
            raise ValueError("configuration freeze requires validated medium comparison")
        if len(self.confirmation_comparisons) not in {0, 2}:
            raise ValueError("configuration freeze requires zero or two unique paired L-scale seeds")
        if any(not isinstance(item, PairedComparisonReceipt) for item in self.confirmation_comparisons):
            raise ValueError("confirmation evidence must be paired comparison receipts")
        comparisons = (self.medium_comparison, *self.confirmation_comparisons)
        if any(
            contract.campaign_hash != self.campaign_hash
            for comparison in comparisons
            for contract in (comparison.candidate_contract, comparison.control_contract)
        ):
            raise ValueError("configuration freeze campaign hash mismatch")
        if any(item.evaluation_manifest.identity != self.evaluation_manifest_hash for item in comparisons):
            raise ValueError("configuration freeze evaluator manifest mismatch")
        if any(item.ranking_task_ids != self.ranking_task_ids for item in comparisons):
            raise ValueError("configuration freeze ranking task mismatch")
        baseline = self.medium_comparison.candidate_contract
        for comparison in comparisons:
            for contract in (comparison.candidate_contract, comparison.control_contract):
                for field in (
                    "tokenizer_hash", "data_manifest_hash", "validation_view_hash",
                    "data_order_policy_hash", "schedule_policy_hash", "code_hash", "release_hash",
                ):
                    if getattr(contract, field) != getattr(baseline, field):
                        raise ValueError(f"configuration freeze {field} mismatch")
        if len({item.candidate_result.seed for item in self.confirmation_comparisons}) != len(self.confirmation_comparisons):
            raise ValueError("configuration freeze requires two unique paired L-scale seeds")
        if self.confirmation_comparisons and len(
            {item.candidate_result.validation_split_hash for item in self.confirmation_comparisons}
        ) != 1:
            raise ValueError("configuration confirmation pairs must share one validation split")
        if not self.ranking_task_ids:
            raise ValueError("configuration freeze requires declared ranking tasks")
        _require_unique(self.ranking_task_ids, "ranking task IDs")
        _require_unique(self.target_diagnostic_task_ids_observed, "target diagnostic task IDs")
        if self.target_diagnostic_task_ids_observed:
            raise ValueError("target diagnostics must remain sealed until after configuration freeze")
        if self.evidence_classification not in _EVIDENCE_CLASSIFICATIONS:
            raise ValueError("unsupported evidence classification")
        if self.decision == "promote_challenger":
            if self.selected_profile_id != "r7_wide_endpoint":
                raise ValueError("challenger promotion must select the reviewed wide endpoint")
            if (
                not self.medium_scale_same_direction
                or len(self.confirmation_seeds) != 2
                or not all(evidence.clears_promotion_threshold for evidence in self.confirmation_seeds)
                or self.evidence_classification != "decision_sufficient"
            ):
                raise ValueError("challenger does not clear the promotion threshold")
        elif self.selected_profile_id != "r7_control":
            raise ValueError("non-promotion decisions must retain the R7 control")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    @property
    def medium_scale_same_direction(self) -> bool:
        evidence = self.medium_comparison.evidence
        return evidence.relative_nll_improvement > 0 or sum(
            effect.matched_score_improvement > 0 for effect in evidence.neutral_family_effects
        ) >= 2

    @property
    def medium_scale_evidence_hash(self) -> str:
        return self.medium_comparison.identity

    @property
    def confirmation_split_hash(self) -> str:
        splits = {item.candidate_result.validation_split_hash for item in self.confirmation_comparisons}
        return next(iter(splits)) if len(splits) == 1 else "0" * 64

    @property
    def confirmation_seeds(self) -> tuple[ConfirmationSeedEvidence, ...]:
        return tuple(item.evidence for item in self.confirmation_comparisons)

    @property
    def production_run_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_pretraining_configuration_freeze_v1",
            "schema_version": 1,
            "production_run_authorized": False,
            "target_diagnostics_remained_sealed": True,
            "freeze_id": self.freeze_id,
            "campaign_hash": self.campaign_hash,
            "decision": self.decision,
            "selected_profile_id": self.selected_profile_id,
            "medium_comparison_hash": self.medium_comparison.identity,
            "medium_scale_same_direction": self.medium_scale_same_direction,
            "confirmation_comparison_hashes": tuple(item.identity for item in self.confirmation_comparisons),
            "confirmation_seeds": tuple(asdict(item) for item in self.confirmation_seeds),
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "ranking_task_ids": self.ranking_task_ids,
            "target_diagnostic_task_ids_observed": self.target_diagnostic_task_ids_observed,
            "evidence_classification": self.evidence_classification,
        }


def build_configuration_freeze_receipt(
    *,
    freeze_id: str,
    campaign_hash: str,
    selected_profile_id: str,
    medium_candidate_result: ExperimentResultReceipt,
    medium_control_result: ExperimentResultReceipt,
    medium_candidate_contract: ExperimentContract,
    medium_control_contract: ExperimentContract,
    confirmation_pairs: tuple[
        tuple[ExperimentResultReceipt, ExperimentResultReceipt, ExperimentContract, ExperimentContract], ...
    ],
    evaluation_manifest: EvaluationManifest,
    ranking_task_ids: tuple[str, ...],
    target_diagnostic_task_ids_observed: tuple[str, ...],
    evidence_classification: str,
) -> ConfigurationFreezeDecisionReceipt:
    """Validate selection use and emit a non-production configuration decision."""

    if target_diagnostic_task_ids_observed:
        raise ValueError("target diagnostics must remain sealed until after configuration freeze")
    observed = ranking_task_ids + target_diagnostic_task_ids_observed
    evaluation_manifest.validate_use(
        phase="l_confirmation",
        observed_task_ids=observed,
        ranking_task_ids=ranking_task_ids,
    )
    medium = PairedComparisonReceipt(
        candidate_result=medium_candidate_result,
        control_result=medium_control_result,
        candidate_contract=medium_candidate_contract,
        control_contract=medium_control_contract,
        evaluation_manifest=evaluation_manifest,
        ranking_task_ids=ranking_task_ids,
    )
    confirmations = tuple(
        PairedComparisonReceipt(
            candidate_result=candidate_result,
            control_result=control_result,
            candidate_contract=candidate_contract,
            control_contract=control_contract,
            evaluation_manifest=evaluation_manifest,
            ranking_task_ids=ranking_task_ids,
        )
        for candidate_result, control_result, candidate_contract, control_contract in confirmation_pairs
    )
    decision = "promote_challenger" if selected_profile_id == "r7_wide_endpoint" else "retain_r7_control"
    return ConfigurationFreezeDecisionReceipt(
        freeze_id=freeze_id,
        campaign_hash=campaign_hash,
        decision=decision,
        selected_profile_id=selected_profile_id,
        medium_comparison=medium,
        confirmation_comparisons=confirmations,
        evaluation_manifest_hash=evaluation_manifest.identity,
        ranking_task_ids=ranking_task_ids,
        target_diagnostic_task_ids_observed=target_diagnostic_task_ids_observed,
        evidence_classification=evidence_classification,
    )
