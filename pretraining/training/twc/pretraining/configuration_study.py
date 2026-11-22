"""Scientific identity for the three-family C2/C3 configuration study.

The module deliberately contains no provider, campaign, price, retry, path, or
packaging-release fields.  Those belong to execution provenance.  C1's v1
artifacts remain immutable and continue to load through their original module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

from twc.pretraining.training.experiments import MODEL_PROFILES, ModelProfile
from twc.repro.hashing import stable_json_hash


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_HASH_FIELDS = frozenset("0123456789abcdef")


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


V2_MODEL_PROFILES: Mapping[str, ModelProfile] = MappingProxyType(
    {
        "m_deep": MODEL_PROFILES["m_deep"],
        "m_wide": MODEL_PROFILES["m_wide"],
        "m_prior_deep": _profile(
            "m_prior_deep", "m", 28, 256, 816, 30_537_984, "m_transfer"
        ),
        "l_deep": MODEL_PROFILES["l_deep"],
        "l_wide": MODEL_PROFILES["l_wide"],
        "l_prior_deep": _profile(
            "l_prior_deep", "l", 31, 320, 1_024, 48_916_160, "l_confirmation"
        ),
        "r7_control": MODEL_PROFILES["r7_control"],
        "r7_wide_endpoint": MODEL_PROFILES["r7_wide_endpoint"],
        "r7_prior_deep_endpoint": _profile(
            "r7_prior_deep_endpoint", "endpoint", 35, 448, 1_440, 100_011_072, None
        ),
    }
)

C2_ANCHOR_PARAMETERS = max(
    V2_MODEL_PROFILES[profile_id].parameter_count
    for profile_id in ("m_deep", "m_wide", "m_prior_deep")
)
C3_ANCHOR_PARAMETERS = max(
    V2_MODEL_PROFILES[profile_id].parameter_count
    for profile_id in ("l_deep", "l_wide", "l_prior_deep")
)
C2_TARGET_LOSS_TOKENS = C2_ANCHOR_PARAMETERS * 20
C3_TARGET_LOSS_TOKENS = C3_ANCHOR_PARAMETERS * 20
C2_EVALUATION_INTERVAL_LOSS_TOKENS = C2_ANCHOR_PARAMETERS * 4
C3_EVALUATION_INTERVAL_LOSS_TOKENS = C3_ANCHOR_PARAMETERS * 4
C2_REQUIRED_EXACT_EVALUATION_LOSS_TOKENS = (C2_TARGET_LOSS_TOKENS,)
C3_REQUIRED_EXACT_EVALUATION_LOSS_TOKENS = (C3_TARGET_LOSS_TOKENS,)


def configuration_evaluation_due(
    *,
    previous_loss_tokens: int,
    consumed_loss_tokens: int,
    last_evaluation_loss_tokens: int,
    evaluation_interval_loss_tokens: int,
    warmup_loss_tokens: int,
    stable_loss_tokens: int,
    target_loss_tokens: int,
) -> bool:
    """Decide one C2/C3 nonterminal evaluation from scientific token cursors."""

    values = (
        previous_loss_tokens,
        consumed_loss_tokens,
        last_evaluation_loss_tokens,
        evaluation_interval_loss_tokens,
        warmup_loss_tokens,
        stable_loss_tokens,
        target_loss_tokens,
    )
    if (
        any(type(value) is not int for value in values)
        or not 0
        <= last_evaluation_loss_tokens
        <= previous_loss_tokens
        < consumed_loss_tokens
        <= target_loss_tokens
        or evaluation_interval_loss_tokens <= 0
        or warmup_loss_tokens <= 0
        or stable_loss_tokens <= 0
        or warmup_loss_tokens + stable_loss_tokens >= target_loss_tokens
    ):
        raise ValueError("configuration evaluation cadence cursors are invalid")
    crossed_schedule_boundary = any(
        previous_loss_tokens < boundary <= consumed_loss_tokens
        for boundary in (
            warmup_loss_tokens,
            warmup_loss_tokens + stable_loss_tokens,
        )
    )
    interval_due = (
        consumed_loss_tokens - last_evaluation_loss_tokens
        >= evaluation_interval_loss_tokens
    )
    return (
        consumed_loss_tokens != target_loss_tokens
        and (interval_due or crossed_schedule_boundary)
    )


_PHASE_SPEC = {
    "m_transfer": (
        "c2",
        17,
        "m_wide",
        C2_TARGET_LOSS_TOKENS,
        C2_EVALUATION_INTERVAL_LOSS_TOKENS,
        C2_REQUIRED_EXACT_EVALUATION_LOSS_TOKENS,
    ),
    "l_confirmation": (
        "c3",
        None,
        "l_wide",
        C3_TARGET_LOSS_TOKENS,
        C3_EVALUATION_INTERVAL_LOSS_TOKENS,
        C3_REQUIRED_EXACT_EVALUATION_LOSS_TOKENS,
    ),
}
_FAMILIES = {
    "r7_scaled_control": {
        "m_transfer": "m_deep",
        "l_confirmation": "l_deep",
        "endpoint": "r7_control",
    },
    "wide_challenger": {
        "m_transfer": "m_wide",
        "l_confirmation": "l_wide",
        "endpoint": "r7_wide_endpoint",
    },
    "prior_deep_narrow": {
        "m_transfer": "m_prior_deep",
        "l_confirmation": "l_prior_deep",
        "endpoint": "r7_prior_deep_endpoint",
    },
}


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HASH_FIELDS:
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _strict_keys(payload: Mapping[str, Any], fields: set[str], metadata: set[str], label: str) -> None:
    if set(payload) != fields | metadata:
        raise ValueError(f"{label} fields are invalid")


@dataclass(frozen=True, slots=True)
class StudyCondition:
    """One logical model/seed condition independent of physical execution."""

    condition_id: str
    phase: str
    family_id: str
    profile_id: str
    seed: int
    paired_order_seed: int
    reference_profile_id: str
    nominal_tokens_per_parameter: int
    target_loss_tokens: int
    evaluation_interval_loss_tokens: int
    evaluation_policy: str
    required_exact_evaluation_loss_tokens: tuple[int, ...]
    endpoint_profile_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.condition_id, "condition_id")
        _require_identifier(self.family_id, "family_id")
        if self.phase not in _PHASE_SPEC:
            raise ValueError("condition phase is unsupported")
        family = _FAMILIES.get(self.family_id)
        if family is None:
            raise ValueError("condition family is unsupported")
        profile = V2_MODEL_PROFILES.get(self.profile_id)
        if profile is None or profile.train_phase is None:
            raise ValueError("condition profile is not trainable")
        if self.profile_id != family[self.phase] or profile.train_phase != self.phase:
            raise ValueError("condition profile does not match its family and phase")
        if self.endpoint_profile_id != family["endpoint"]:
            raise ValueError("condition endpoint projection does not match its family")
        prefix, required_seed, reference_profile_id, target, interval, checkpoints = _PHASE_SPEC[self.phase]
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("condition seed is invalid")
        if required_seed is not None and self.seed != required_seed:
            raise ValueError("m_transfer condition seed must be 17")
        if self.phase == "l_confirmation" and self.seed not in {29, 43}:
            raise ValueError("l_confirmation condition seed must be 29 or 43")
        if self.paired_order_seed != 0:
            raise ValueError("configuration conditions require paired_order_seed=0")
        if self.reference_profile_id != reference_profile_id:
            raise ValueError("condition reference profile must be the largest phase arm")
        if self.nominal_tokens_per_parameter != 20:
            raise ValueError("condition nominal token rule must be 20 per reference parameter")
        if (
            self.target_loss_tokens != target
            or self.evaluation_interval_loss_tokens != interval
            or self.required_exact_evaluation_loss_tokens != checkpoints
        ):
            raise ValueError("condition must use the exact frozen phase horizon and evaluation cadence")
        if self.evaluation_policy != "interval_since_last_eval_plus_schedule_boundaries_v1":
            raise ValueError("condition evaluation policy is unsupported")
        expected_id = f"{prefix}-{self.family_id.replace('_', '-')}-seed{self.seed}"
        if self.condition_id != expected_id:
            raise ValueError("condition_id does not match phase, family, and seed")

    @property
    def profile(self) -> ModelProfile:
        return V2_MODEL_PROFILES[self.profile_id]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConfigurationDecisionPolicyV1:
    """Hash-bound, executable C2/C3 decision policy.

    The descriptive rule identifiers make the scientific contract legible while
    the methods below are the sole implementation of its numeric formulas.
    """

    practical_effect_floor: float = 0.005
    uncertainty_multiplier: float = 2.24
    late_window_interval_multiplier: int = 3
    late_window_nominal_parameter_multiple: int = 12
    late_opposite_direction_rule: str = "any_threshold_clearing_late_point_v1"
    material_regression_rule: str = (
        "terminal_vs_best_prior_late_point_paired_threshold_v1"
    )
    joint_two_seed_rule: str = (
        "equal_seed_mean_rss_eval_mean_se_and_between_seed_sem_v1"
    )
    c2_scale_reversal_rule: str = (
        "opposed_requires_no_raw_interpretation_evidence_hash_v1"
    )
    direct_comparison_rule: str = "consistent_threshold_clearing_both_seeds_v1"
    equivalence_rule: str = "joint_direct_effect_within_resolution_v1"
    equivalent_selection_order: tuple[str, ...] = (
        "lower_worst_seed_terminal_nll",
        "lower_equal_seed_mean_terminal_nll",
        "exact_tie_unordered",
    )

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "practical_effect_floor": 0.005,
            "uncertainty_multiplier": 2.24,
            "late_window_interval_multiplier": 3,
            "late_window_nominal_parameter_multiple": 12,
            "late_opposite_direction_rule": "any_threshold_clearing_late_point_v1",
            "material_regression_rule": (
                "terminal_vs_best_prior_late_point_paired_threshold_v1"
            ),
            "joint_two_seed_rule": (
                "equal_seed_mean_rss_eval_mean_se_and_between_seed_sem_v1"
            ),
            "c2_scale_reversal_rule": (
                "opposed_requires_no_raw_interpretation_evidence_hash_v1"
            ),
            "direct_comparison_rule": "consistent_threshold_clearing_both_seeds_v1",
            "equivalence_rule": "joint_direct_effect_within_resolution_v1",
            "equivalent_selection_order": (
                "lower_worst_seed_terminal_nll",
                "lower_equal_seed_mean_terminal_nll",
                "exact_tie_unordered",
            ),
        }
        for field, frozen_value in expected.items():
            actual = getattr(self, field)
            if type(actual) is not type(frozen_value) or actual != frozen_value:
                raise ValueError(f"configuration decision policy field is not frozen: {field}")

    def threshold(self, paired_standard_error: float) -> float:
        if (
            type(paired_standard_error) not in {int, float}
            or not math.isfinite(float(paired_standard_error))
            or paired_standard_error < 0
        ):
            raise ValueError("paired_standard_error must be nonnegative")
        standard_error = float(paired_standard_error)
        return max(
            self.practical_effect_floor,
            self.uncertainty_multiplier * standard_error,
        )

    def late_window_start(self, evaluation_interval_loss_tokens: int) -> int:
        if type(evaluation_interval_loss_tokens) is not int or evaluation_interval_loss_tokens <= 0:
            raise ValueError("evaluation interval must be a positive integer")
        return self.late_window_interval_multiplier * evaluation_interval_loss_tokens

    def joint_two_seed_components(
        self,
        *,
        seed29_effect: float,
        seed29_standard_error: float,
        seed43_effect: float,
        seed43_standard_error: float,
    ) -> tuple[float, float, float, float, float]:
        values = (seed29_effect, seed29_standard_error, seed43_effect, seed43_standard_error)
        if any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in values):
            raise ValueError("joint two-seed inputs must be finite")
        e29 = float(seed29_effect)
        e43 = float(seed43_effect)
        se29 = float(seed29_standard_error)
        se43 = float(seed43_standard_error)
        if se29 < 0 or se43 < 0:
            raise ValueError("seed standard errors must be nonnegative")
        equal_seed_mean = (e29 + e43) / 2
        evaluation_mean_standard_error = math.sqrt(se29 * se29 + se43 * se43) / 2
        between_seed_sem = abs(e29 - e43) / 2
        combined_standard_error = math.sqrt(
            evaluation_mean_standard_error * evaluation_mean_standard_error
            + between_seed_sem * between_seed_sem
        )
        return (
            equal_seed_mean,
            evaluation_mean_standard_error,
            between_seed_sem,
            combined_standard_error,
            self.threshold(combined_standard_error),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConfigurationComparisonSpecV1:
    """The complete predeclared scientific condition matrix and decision policy."""

    conditions: tuple[StudyCondition, ...]
    ranking_task_ids: tuple[str, ...] = ("validation_nll",)
    target_diagnostic_task_ids: tuple[str, ...] = ()
    decision_policy: ConfigurationDecisionPolicyV1 = ConfigurationDecisionPolicyV1()

    def __post_init__(self) -> None:
        if not self.conditions or any(not isinstance(item, StudyCondition) for item in self.conditions):
            raise ValueError("comparison spec requires validated conditions")
        ids = tuple(item.condition_id for item in self.conditions)
        if len(ids) != len(set(ids)):
            raise ValueError("comparison condition IDs must be unique")
        expected = _default_conditions()
        if self.conditions != expected:
            raise ValueError("comparison spec must contain the exact approved C2/C3 condition matrix")
        if self.ranking_task_ids != ("validation_nll",):
            raise ValueError("validation_nll must be the only ranking task")
        if self.target_diagnostic_task_ids:
            raise ValueError("target diagnostics must remain sealed during configuration selection")
        if not isinstance(self.decision_policy, ConfigurationDecisionPolicyV1):
            raise ValueError("comparison spec requires a frozen decision policy")

    @property
    def practical_effect_floor(self) -> float:
        return self.decision_policy.practical_effect_floor

    @property
    def uncertainty_multiplier(self) -> float:
        return self.decision_policy.uncertainty_multiplier

    def condition(self, condition_id: str) -> StudyCondition:
        for condition in self.conditions:
            if condition.condition_id == condition_id:
                return condition
        raise KeyError(condition_id)

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_comparison_spec_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "conditions": tuple(item.to_dict() for item in self.conditions),
            "ranking_task_ids": self.ranking_task_ids,
            "target_diagnostic_task_ids": self.target_diagnostic_task_ids,
            "decision_policy": self.decision_policy.to_dict(),
        }


def _default_conditions() -> tuple[StudyCondition, ...]:
    rows: list[StudyCondition] = []
    for phase, seeds in (("m_transfer", (17,)), ("l_confirmation", (29, 43))):
        prefix, _, reference_profile_id, target, interval, checkpoints = _PHASE_SPEC[phase]
        for seed in seeds:
            for family_id, profiles in _FAMILIES.items():
                rows.append(
                    StudyCondition(
                        condition_id=f"{prefix}-{family_id.replace('_', '-')}-seed{seed}",
                        phase=phase,
                        family_id=family_id,
                        profile_id=profiles[phase],
                        seed=seed,
                        paired_order_seed=0,
                        reference_profile_id=reference_profile_id,
                        nominal_tokens_per_parameter=20,
                        target_loss_tokens=target,
                        evaluation_interval_loss_tokens=interval,
                        evaluation_policy="interval_since_last_eval_plus_schedule_boundaries_v1",
                        required_exact_evaluation_loss_tokens=checkpoints,
                        endpoint_profile_id=profiles["endpoint"],
                    )
                )
    return tuple(rows)


def build_default_c2_c3_comparison_spec() -> ConfigurationComparisonSpecV1:
    return ConfigurationComparisonSpecV1(conditions=_default_conditions())


def load_configuration_comparison_spec(payload: Mapping[str, Any]) -> ConfigurationComparisonSpecV1:
    fields = set(ConfigurationComparisonSpecV1.__dataclass_fields__)
    metadata = {"artifact_kind", "schema_version", "production_unlock_eligible", "f1_unlock_eligible"}
    _strict_keys(payload, fields, metadata, "comparison spec")
    if (
        payload.get("artifact_kind") != "twc_configuration_comparison_spec_v1"
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
        or payload.get("f1_unlock_eligible") is not False
        or not isinstance(payload.get("conditions"), (list, tuple))
        or not isinstance(payload.get("decision_policy"), Mapping)
    ):
        raise ValueError("comparison spec boundary is invalid")
    policy_payload = payload["decision_policy"]
    if set(policy_payload) != set(ConfigurationDecisionPolicyV1.__dataclass_fields__):
        raise ValueError("configuration decision policy fields are invalid")
    condition_fields = set(StudyCondition.__dataclass_fields__)
    conditions: list[StudyCondition] = []
    for row in payload["conditions"]:
        if not isinstance(row, Mapping) or set(row) != condition_fields:
            raise ValueError("comparison condition fields are invalid")
        values = dict(row)
        values["required_exact_evaluation_loss_tokens"] = tuple(
            values["required_exact_evaluation_loss_tokens"]
        )
        conditions.append(StudyCondition(**values))
    return ConfigurationComparisonSpecV1(
        conditions=tuple(conditions),
        ranking_task_ids=tuple(payload["ranking_task_ids"]),
        target_diagnostic_task_ids=tuple(payload["target_diagnostic_task_ids"]),
        decision_policy=ConfigurationDecisionPolicyV1(
            **{
                field: (
                    tuple(policy_payload[field])
                    if field == "equivalent_selection_order"
                    else policy_payload[field]
                )
                for field in ConfigurationDecisionPolicyV1.__dataclass_fields__
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class ConfigurationStudyPlanV2:
    """Scientific study identity after immutable C1/data/tokenizer adoption."""

    study_id: str
    c1_adoption_receipt_hash: str
    comparison_spec: ConfigurationComparisonSpecV1
    tokenizer_adoption_receipt_hash: str
    data_plan_hash: str
    prefix_continuity_receipt_hash: str
    train_view_hashes: tuple[str, str]
    validation_view_hash: str
    evaluation_manifest_hash: str
    data_order_policy_hash: str
    schedule_policy_hash: str
    semantic_code_hash: str
    selected_learning_rate: float
    target_diagnostics_remained_sealed: bool

    def __post_init__(self) -> None:
        _require_identifier(self.study_id, "study_id")
        if not isinstance(self.comparison_spec, ConfigurationComparisonSpecV1):
            raise ValueError("study requires a validated comparison spec")
        for field in (
            "c1_adoption_receipt_hash",
            "tokenizer_adoption_receipt_hash",
            "data_plan_hash",
            "prefix_continuity_receipt_hash",
            "validation_view_hash",
            "evaluation_manifest_hash",
            "data_order_policy_hash",
            "schedule_policy_hash",
            "semantic_code_hash",
        ):
            _require_hash(getattr(self, field), field)
        if len(self.train_view_hashes) != 2:
            raise ValueError("study requires exactly C2 and C3 train view hashes")
        for index, value in enumerate(self.train_view_hashes):
            _require_hash(value, f"train_view_hashes[{index}]")
        if self.selected_learning_rate != 9e-4:
            raise ValueError("study must use the C1-selected learning rate 9e-4")
        if self.target_diagnostics_remained_sealed is not True:
            raise ValueError("target diagnostics must remain sealed")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_study_plan_v2",
            "schema_version": 2,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "study_id": self.study_id,
            "c1_adoption_receipt_hash": self.c1_adoption_receipt_hash,
            "comparison_spec": self.comparison_spec.to_dict(),
            "comparison_spec_hash": self.comparison_spec.identity,
            "tokenizer_adoption_receipt_hash": self.tokenizer_adoption_receipt_hash,
            "data_plan_hash": self.data_plan_hash,
            "prefix_continuity_receipt_hash": self.prefix_continuity_receipt_hash,
            "train_view_hashes": self.train_view_hashes,
            "validation_view_hash": self.validation_view_hash,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "data_order_policy_hash": self.data_order_policy_hash,
            "schedule_policy_hash": self.schedule_policy_hash,
            "semantic_code_hash": self.semantic_code_hash,
            "selected_learning_rate": self.selected_learning_rate,
            "target_diagnostics_remained_sealed": self.target_diagnostics_remained_sealed,
        }


def load_configuration_study_plan(payload: Mapping[str, Any]) -> ConfigurationStudyPlanV2:
    fields = set(ConfigurationStudyPlanV2.__dataclass_fields__) - {"comparison_spec"}
    metadata = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "comparison_spec",
        "comparison_spec_hash",
    }
    _strict_keys(payload, fields, metadata, "configuration study plan")
    if (
        payload.get("artifact_kind") != "twc_configuration_study_plan_v2"
        or payload.get("schema_version") != 2
        or payload.get("production_unlock_eligible") is not False
        or payload.get("f1_unlock_eligible") is not False
        or not isinstance(payload.get("comparison_spec"), Mapping)
    ):
        raise ValueError("configuration study plan boundary is invalid")
    comparison = load_configuration_comparison_spec(payload["comparison_spec"])
    if payload.get("comparison_spec_hash") != comparison.identity:
        raise ValueError("configuration study comparison spec hash mismatch")
    values = {field: payload[field] for field in fields}
    values["train_view_hashes"] = tuple(values["train_view_hashes"])
    return ConfigurationStudyPlanV2(comparison_spec=comparison, **values)


@dataclass(frozen=True, slots=True)
class ExperimentContractV2:
    """One scientific condition under a v2 study; no execution provenance."""

    experiment_id: str
    study_plan_hash: str
    comparison_spec_hash: str
    condition: StudyCondition
    data_manifest_hash: str
    train_view_hash: str
    validation_view_hash: str
    tokenizer_hash: str
    evaluation_manifest_hash: str
    data_order_hash: str
    schedule_hash: str
    semantic_code_hash: str
    learning_rate: float
    schedule_tail_fraction: float

    def __post_init__(self) -> None:
        _require_identifier(self.experiment_id, "experiment_id")
        if not isinstance(self.condition, StudyCondition):
            raise ValueError("experiment requires a validated condition")
        if self.experiment_id != self.condition.condition_id:
            raise ValueError("experiment condition_id does not match experiment_id")
        for field in (
            "study_plan_hash",
            "comparison_spec_hash",
            "data_manifest_hash",
            "train_view_hash",
            "validation_view_hash",
            "tokenizer_hash",
            "evaluation_manifest_hash",
            "data_order_hash",
            "schedule_hash",
            "semantic_code_hash",
        ):
            _require_hash(getattr(self, field), field)
        if self.learning_rate != 9e-4:
            raise ValueError("experiment learning rate must be 9e-4")
        if self.schedule_tail_fraction != 0.1:
            raise ValueError("experiment schedule tail must be 10 percent")

    @property
    def profile(self) -> ModelProfile:
        return self.condition.profile

    @property
    def profile_id(self) -> str:
        return self.condition.profile_id

    @property
    def phase(self) -> str:
        return self.condition.phase

    @property
    def seed(self) -> int:
        return self.condition.seed

    @property
    def paired_order_seed(self) -> int:
        return self.condition.paired_order_seed

    @property
    def target_loss_tokens(self) -> int:
        return self.condition.target_loss_tokens

    @property
    def required_exact_evaluation_loss_tokens(self) -> tuple[int, ...]:
        return self.condition.required_exact_evaluation_loss_tokens

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_pretraining_experiment_contract_v2",
            "schema_version": 2,
            "production_unlock_eligible": False,
            "accepted_manifest_unlock_eligible": False,
            "experiment_id": self.experiment_id,
            "study_plan_hash": self.study_plan_hash,
            "comparison_spec_hash": self.comparison_spec_hash,
            "condition": self.condition.to_dict(),
            "data_manifest_hash": self.data_manifest_hash,
            "train_view_hash": self.train_view_hash,
            "validation_view_hash": self.validation_view_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "evaluation_manifest_hash": self.evaluation_manifest_hash,
            "data_order_hash": self.data_order_hash,
            "schedule_hash": self.schedule_hash,
            "semantic_code_hash": self.semantic_code_hash,
            "learning_rate": self.learning_rate,
            "schedule_tail_fraction": self.schedule_tail_fraction,
            "profile_id": self.profile_id,
            "parameter_count": self.profile.parameter_count,
            "target_loss_tokens": self.target_loss_tokens,
            "required_exact_evaluation_loss_tokens": self.required_exact_evaluation_loss_tokens,
        }


def load_experiment_contract_v2(payload: Mapping[str, Any]) -> ExperimentContractV2:
    fields = set(ExperimentContractV2.__dataclass_fields__) - {"condition"}
    metadata = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "accepted_manifest_unlock_eligible",
        "condition",
        "profile_id",
        "parameter_count",
        "target_loss_tokens",
        "required_exact_evaluation_loss_tokens",
    }
    _strict_keys(payload, fields, metadata, "experiment v2")
    if (
        payload.get("artifact_kind") != "twc_pretraining_experiment_contract_v2"
        or payload.get("schema_version") != 2
        or payload.get("production_unlock_eligible") is not False
        or payload.get("accepted_manifest_unlock_eligible") is not False
        or not isinstance(payload.get("condition"), Mapping)
    ):
        raise ValueError("experiment v2 boundary is invalid")
    condition_values = dict(payload["condition"])
    condition_values["required_exact_evaluation_loss_tokens"] = tuple(
        condition_values["required_exact_evaluation_loss_tokens"]
    )
    condition = StudyCondition(**condition_values)
    contract = ExperimentContractV2(
        condition=condition,
        **{field: payload[field] for field in fields},
    )
    expected = {
        "profile_id": contract.profile_id,
        "parameter_count": contract.profile.parameter_count,
        "target_loss_tokens": contract.target_loss_tokens,
        "required_exact_evaluation_loss_tokens": contract.required_exact_evaluation_loss_tokens,
    }
    for field, value in expected.items():
        actual = (
            tuple(payload[field])
            if field == "required_exact_evaluation_loss_tokens"
            else payload[field]
        )
        if actual != value:
            raise ValueError(f"experiment v2 derived field mismatch: {field}")
    return contract
