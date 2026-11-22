"""Pure scientific assessment artifacts for the C2/C3 configuration study.

This module deliberately has no filesystem, provider, campaign, retry, price,
or operator behavior.  It reduces already-bound validation-NLL evidence to the
predeclared C2/C3 decisions.  Physical execution is represented only by the
hash of the execution binding that produced a result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable, Mapping

from twc.pretraining.configuration_study import (
    ConfigurationDecisionPolicyV1,
    ExperimentContractV2,
    StudyCondition,
    build_default_c2_c3_comparison_spec,
    load_experiment_contract_v2,
)
from twc.repro.hashing import stable_json_hash


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_HASH_CHARACTERS = frozenset("0123456789abcdef")
_CONTROL_FAMILY = "r7_scaled_control"
_CHALLENGER_FAMILIES = ("wide_challenger", "prior_deep_narrow")
_FAMILY_ORDER = (_CONTROL_FAMILY, *_CHALLENGER_FAMILIES)
_STATUSES = frozenset({"supported", "opposed", "unresolved", "blocked"})
_CANONICAL_COMPARISON_SPEC = build_default_c2_c3_comparison_spec()
_POLICY: ConfigurationDecisionPolicyV1 = _CANONICAL_COMPARISON_SPEC.decision_policy


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def _require_hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value) <= _HASH_CHARACTERS
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _require_finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if nonnegative and number < 0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _weighted_mean(values: tuple[float, ...], weights: tuple[int, ...]) -> float:
    return sum(
        value * weight for value, weight in zip(values, weights, strict=True)
    ) / sum(weights)


def _weighted_paired_cluster_se(
    values: tuple[float, ...], weights: tuple[int, ...]
) -> float:
    """Kish-effective-sample SE over paired validation batches."""

    if len(values) != len(weights):
        raise ValueError("paired uncertainty weights must align with values")
    if len(values) < 2:
        return math.inf
    total_weight = sum(weights)
    squared_weight_sum = sum(weight * weight for weight in weights)
    effective_count = total_weight * total_weight / squared_weight_sum
    if effective_count <= 1:
        return math.inf
    mean = _weighted_mean(values, weights)
    population_variance = sum(
        weight * (value - mean) ** 2
        for value, weight in zip(values, weights, strict=True)
    ) / total_weight
    sample_variance = population_variance * effective_count / (effective_count - 1)
    return math.sqrt(sample_variance / effective_count)


def _require_artifact_boundary(
    payload: Mapping[str, Any],
    *,
    fields: set[str],
    artifact_kind: str,
    schema_version: int,
    label: str,
) -> None:
    if set(payload) != fields:
        raise ValueError(f"{label} fields are invalid")
    if (
        payload.get("artifact_kind") != artifact_kind
        or payload.get("schema_version") != schema_version
        or payload.get("production_unlock_eligible") is not False
        or payload.get("f1_unlock_eligible") is not False
    ):
        raise ValueError(f"{label} boundary is invalid")


@dataclass(frozen=True, slots=True)
class ValidationNllEvaluationPoint:
    """No-raw paired evidence retained for every realized evaluation."""

    at_loss_tokens: int
    aggregate_nll: float
    component_losses: tuple[float, ...]
    component_loss_token_weights: tuple[int, ...]
    component_group_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.at_loss_tokens) is not int or self.at_loss_tokens <= 0:
            raise ValueError("evaluation point loss-token position must be positive")
        aggregate = _require_finite(self.aggregate_nll, "aggregate_nll", nonnegative=True)
        if len(self.component_losses) < 2:
            raise ValueError("evaluation point requires at least two paired component losses")
        if (
            len(self.component_losses) != len(self.component_loss_token_weights)
            or len(self.component_losses) != len(self.component_group_hashes)
        ):
            raise ValueError(
                "evaluation point component losses, weights, and identities must align"
            )
        for index, score in enumerate(self.component_losses):
            _require_finite(score, f"component_losses[{index}]", nonnegative=True)
        if any(
            type(weight) is not int or weight <= 0
            for weight in self.component_loss_token_weights
        ):
            raise ValueError("evaluation point weights must be positive integers")
        if len(self.component_group_hashes) != len(set(self.component_group_hashes)):
            raise ValueError("evaluation point component identities must be unique")
        for index, value in enumerate(self.component_group_hashes):
            _require_hash(value, f"component_group_hashes[{index}]")
        derived = _weighted_mean(
            self.component_losses, self.component_loss_token_weights
        )
        if derived <= 0:
            raise ValueError("evaluation point validation NLL must be positive")
        if not math.isclose(aggregate, derived, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("evaluation point aggregate does not match weighted terminal NLL")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentResultV3:
    """Validation-NLL evidence for one scientific contract and physical binding."""

    result_id: str
    scientific_contract: ExperimentContractV2
    execution_binding_hash: str
    task_id: str
    evaluation_points: tuple[ValidationNllEvaluationPoint, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.result_id, "result_id")
        if not isinstance(self.scientific_contract, ExperimentContractV2):
            raise ValueError("result requires an ExperimentContractV2")
        if self.scientific_contract.comparison_spec_hash != _CANONICAL_COMPARISON_SPEC.identity:
            raise ValueError("result does not bind the canonical frozen comparison policy")
        _require_hash(self.execution_binding_hash, "execution_binding_hash")
        if self.task_id != "validation_nll":
            raise ValueError("configuration results may contain only validation_nll")
        if len(self.evaluation_points) < 2 or any(
            not isinstance(point, ValidationNllEvaluationPoint)
            for point in self.evaluation_points
        ):
            raise ValueError("result requires at least two realized evaluation positions")
        previous = 0
        first_weights = self.evaluation_points[0].component_loss_token_weights
        first_group_hashes = self.evaluation_points[0].component_group_hashes
        for point in self.evaluation_points:
            if point.at_loss_tokens <= previous:
                raise ValueError("result realized curve positions must strictly increase")
            if point.component_loss_token_weights != first_weights:
                raise ValueError("result paired score weights must match at every evaluation")
            if point.component_group_hashes != first_group_hashes:
                raise ValueError(
                    "result paired component identities must match at every evaluation"
                )
            previous = point.at_loss_tokens
        target = self.scientific_contract.target_loss_tokens
        if self.evaluation_points[-1].at_loss_tokens != target:
            raise ValueError("result curve does not end at the exact terminal token horizon")
        required = set(self.scientific_contract.required_exact_evaluation_loss_tokens)
        if not required.issubset(self.evaluation_curve_loss_tokens):
            raise ValueError("result is missing a required exact evaluation position")

    @property
    def scientific_contract_hash(self) -> str:
        return self.scientific_contract.identity

    @property
    def family_id(self) -> str:
        return self.scientific_contract.condition.family_id

    @property
    def evaluation_curve_loss_tokens(self) -> tuple[int, ...]:
        return tuple(point.at_loss_tokens for point in self.evaluation_points)

    @property
    def evaluation_curve_losses(self) -> tuple[float, ...]:
        return tuple(point.aggregate_nll for point in self.evaluation_points)

    @property
    def terminal_scores(self) -> tuple[float, ...]:
        return self.evaluation_points[-1].component_losses

    @property
    def terminal_score_weights(self) -> tuple[int, ...]:
        return self.evaluation_points[-1].component_loss_token_weights

    @property
    def terminal_component_group_hashes(self) -> tuple[str, ...]:
        return self.evaluation_points[-1].component_group_hashes

    @property
    def terminal_validation_nll(self) -> float:
        return self.evaluation_points[-1].aggregate_nll

    @property
    def curve_learned(self) -> bool:
        return self.evaluation_curve_losses[-1] < self.evaluation_curve_losses[0]

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_experiment_result_v3",
            "schema_version": 3,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "result_id": self.result_id,
            "scientific_contract": self.scientific_contract.to_dict(),
            "scientific_contract_hash": self.scientific_contract_hash,
            "execution_binding_hash": self.execution_binding_hash,
            "task_id": self.task_id,
            "evaluation_points": tuple(point.to_dict() for point in self.evaluation_points),
            "terminal_validation_nll": self.terminal_validation_nll,
        }


def load_experiment_result_v3(
    payload: Mapping[str, Any],
    *,
    expected_scientific_contract: ExperimentContractV2,
    expected_execution_binding_hash: str,
) -> ExperimentResultV3:
    """Strictly read back a result against its expected scientific/run identity."""

    fields = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "result_id",
        "scientific_contract",
        "scientific_contract_hash",
        "execution_binding_hash",
        "task_id",
        "evaluation_points",
        "terminal_validation_nll",
    }
    _require_artifact_boundary(
        payload,
        fields=fields,
        artifact_kind="twc_configuration_experiment_result_v3",
        schema_version=3,
        label="configuration result v3",
    )
    if not isinstance(expected_scientific_contract, ExperimentContractV2):
        raise ValueError("expected scientific contract must be ExperimentContractV2")
    _require_hash(expected_execution_binding_hash, "expected_execution_binding_hash")
    contract_payload = payload.get("scientific_contract")
    if not isinstance(contract_payload, Mapping):
        raise ValueError("configuration result scientific contract is invalid")
    contract = load_experiment_contract_v2(contract_payload)
    if payload.get("scientific_contract_hash") != contract.identity:
        raise ValueError("configuration result scientific contract hash mismatch")
    if contract.identity != expected_scientific_contract.identity:
        raise ValueError("configuration result does not match the expected scientific contract")
    if payload.get("execution_binding_hash") != expected_execution_binding_hash:
        raise ValueError("configuration result does not match the expected execution binding")
    point_rows = payload.get("evaluation_points")
    if not isinstance(point_rows, (list, tuple)):
        raise ValueError("configuration result evaluation_points is invalid")
    point_fields = set(ValidationNllEvaluationPoint.__dataclass_fields__)
    points: list[ValidationNllEvaluationPoint] = []
    for row in point_rows:
        if not isinstance(row, Mapping) or set(row) != point_fields:
            raise ValueError("configuration result evaluation point fields are invalid")
        values = dict(row)
        if not isinstance(values["component_losses"], (list, tuple)) or not isinstance(
            values["component_loss_token_weights"], (list, tuple)
        ) or not isinstance(values["component_group_hashes"], (list, tuple)):
            raise ValueError("configuration result evaluation point vectors are invalid")
        values["component_losses"] = tuple(values["component_losses"])
        values["component_loss_token_weights"] = tuple(
            values["component_loss_token_weights"]
        )
        values["component_group_hashes"] = tuple(values["component_group_hashes"])
        points.append(ValidationNllEvaluationPoint(**values))
    result = ExperimentResultV3(
        result_id=payload["result_id"],
        scientific_contract=contract,
        execution_binding_hash=payload["execution_binding_hash"],
        task_id=payload["task_id"],
        evaluation_points=tuple(points),
    )
    if stable_json_hash(payload) != stable_json_hash(result.to_dict()):
        raise ValueError("configuration result derived evidence mismatch")
    return result


@dataclass(frozen=True, slots=True)
class PairwiseNllEffect:
    """A paired NLL effect at one exact realized evaluation position."""

    candidate_family_id: str
    reference_family_id: str
    at_loss_tokens: int
    relative_nll_improvement: float
    paired_weighted_cluster_se: float
    threshold: float

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_family_id, "candidate_family_id")
        _require_identifier(self.reference_family_id, "reference_family_id")
        if self.candidate_family_id == self.reference_family_id:
            raise ValueError("pairwise effect requires distinct families")
        if type(self.at_loss_tokens) is not int or self.at_loss_tokens <= 0:
            raise ValueError("pairwise effect position must be positive")
        _require_finite(self.relative_nll_improvement, "relative_nll_improvement")
        _require_finite(
            self.paired_weighted_cluster_se,
            "paired_weighted_cluster_se",
            nonnegative=True,
        )
        expected = _POLICY.threshold(self.paired_weighted_cluster_se)
        if not math.isclose(self.threshold, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("pairwise threshold does not match the frozen policy")


@dataclass(frozen=True, slots=True)
class TerminalRegressionEvidence:
    """Paired terminal-vs-best-prior-late evidence for one arm."""

    family_id: str
    best_prior_at_loss_tokens: int | None
    terminal_at_loss_tokens: int
    relative_nll_regression: float | None
    paired_weighted_cluster_se: float | None
    threshold: float | None
    material: bool

    def __post_init__(self) -> None:
        _require_identifier(self.family_id, "family_id")
        if type(self.terminal_at_loss_tokens) is not int or self.terminal_at_loss_tokens <= 0:
            raise ValueError("terminal regression terminal position is invalid")
        if self.best_prior_at_loss_tokens is None:
            if any(
                value is not None
                for value in (
                    self.relative_nll_regression,
                    self.paired_weighted_cluster_se,
                    self.threshold,
                )
            ) or self.material:
                raise ValueError("terminal regression empty evidence is inconsistent")
            return
        if (
            type(self.best_prior_at_loss_tokens) is not int
            or not 0 < self.best_prior_at_loss_tokens < self.terminal_at_loss_tokens
        ):
            raise ValueError("terminal regression prior position is invalid")
        regression = _require_finite(
            self.relative_nll_regression, "relative_nll_regression"
        )
        standard_error = _require_finite(
            self.paired_weighted_cluster_se,
            "paired_weighted_cluster_se",
            nonnegative=True,
        )
        threshold = _require_finite(self.threshold, "threshold", nonnegative=True)
        if not math.isclose(
            threshold, _POLICY.threshold(standard_error), rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError("terminal regression threshold does not match frozen policy")
        if self.material != (regression >= threshold):
            raise ValueError("terminal regression material flag is inconsistent")


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """One candidate/reference direction with stability guardrails."""

    family_id: str
    reference_family_id: str
    effect: PairwiseNllEffect
    status: str
    late_window_start_loss_tokens: int
    late_effects: tuple[PairwiseNllEffect, ...]
    candidate_curve_learned: bool
    reference_curve_learned: bool
    candidate_terminal_regression: TerminalRegressionEvidence
    reference_terminal_regression: TerminalRegressionEvidence
    late_opposite_direction: bool
    guard_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.family_id, "family_id")
        _require_identifier(self.reference_family_id, "reference_family_id")
        if self.effect.candidate_family_id != self.family_id:
            raise ValueError("candidate assessment effect family mismatch")
        if self.effect.reference_family_id != self.reference_family_id:
            raise ValueError("candidate assessment reference family mismatch")
        if self.status not in _STATUSES:
            raise ValueError("candidate assessment status is invalid")
        if type(self.late_window_start_loss_tokens) is not int or self.late_window_start_loss_tokens <= 0:
            raise ValueError("candidate late-window start is invalid")
        if not self.late_effects or any(
            not isinstance(item, PairwiseNllEffect) for item in self.late_effects
        ):
            raise ValueError("candidate late-point evidence is missing")
        if self.late_effects[0].at_loss_tokens < self.late_window_start_loss_tokens:
            raise ValueError("candidate late-point evidence starts before the frozen window")
        if any(
            type(value) is not bool
            for value in (
                self.candidate_curve_learned,
                self.reference_curve_learned,
                self.late_opposite_direction,
            )
        ):
            raise ValueError("candidate assessment guards must be booleans")
        if self.candidate_terminal_regression.family_id != self.family_id:
            raise ValueError("candidate regression evidence family mismatch")
        if self.reference_terminal_regression.family_id != self.reference_family_id:
            raise ValueError("reference regression evidence family mismatch")
        if any(not isinstance(reason, str) or not reason for reason in self.guard_reasons):
            raise ValueError("candidate guard reasons are invalid")
        if self.status == "blocked":
            raise ValueError("validated finite evidence cannot produce blocked status")


@dataclass(frozen=True, slots=True)
class MultiArmComparisonReceipt:
    """A matched control block with one or two configuration challengers."""

    results: tuple[ExperimentResultV3, ...]

    def __post_init__(self) -> None:
        if not 2 <= len(self.results) <= 3 or any(
            not isinstance(result, ExperimentResultV3) for result in self.results
        ):
            raise ValueError("multi-arm comparison requires control and one or two challengers")
        families = tuple(result.family_id for result in self.results)
        expected_order = tuple(family for family in _FAMILY_ORDER if family in families)
        if families != expected_order or families[0] != _CONTROL_FAMILY:
            raise ValueError("multi-arm comparison families must use canonical control-first order")
        if len(families) != len(set(families)):
            raise ValueError("multi-arm comparison family IDs must be unique")
        first = self.results[0]
        phase = first.scientific_contract.phase
        seed = first.scientific_contract.seed
        if phase == "m_transfer" and (seed != 17 or len(self.results) != 3):
            raise ValueError("C2 comparison requires all three seed-17 families")
        if phase == "l_confirmation" and seed == 29 and len(self.results) != 3:
            raise ValueError("C3a comparison requires all three seed-29 families")
        if phase == "l_confirmation" and seed == 43 and len(self.results) not in {2, 3}:
            raise ValueError("C3b comparison requires control and supported challengers")
        if phase not in {"m_transfer", "l_confirmation"}:
            raise ValueError("multi-arm comparison phase is unsupported")

        contract_fields = (
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
            "learning_rate",
            "schedule_tail_fraction",
            "phase",
            "seed",
            "paired_order_seed",
            "target_loss_tokens",
        )
        for result in self.results[1:]:
            for field in contract_fields:
                if getattr(result.scientific_contract, field) != getattr(
                    first.scientific_contract, field
                ):
                    raise ValueError(f"multi-arm comparison {field} mismatch")
            if result.task_id != first.task_id:
                raise ValueError("multi-arm comparison task mismatch")
            if result.evaluation_curve_loss_tokens != first.evaluation_curve_loss_tokens:
                raise ValueError("multi-arm comparison realized curve positions mismatch")
            if any(
                point.component_loss_token_weights
                != first.evaluation_points[index].component_loss_token_weights
                for index, point in enumerate(result.evaluation_points)
            ):
                raise ValueError(
                    "multi-arm comparison paired score weights at every evaluation mismatch"
                )
            if any(
                point.component_group_hashes
                != first.evaluation_points[index].component_group_hashes
                for index, point in enumerate(result.evaluation_points)
            ):
                raise ValueError(
                    "multi-arm comparison paired component identities mismatch"
                )
        binding_hashes = tuple(result.execution_binding_hash for result in self.results)
        if len(binding_hashes) != len(set(binding_hashes)):
            raise ValueError("multi-arm comparison execution bindings must be unique")

    @property
    def phase(self) -> str:
        return self.results[0].scientific_contract.phase

    @property
    def seed(self) -> int:
        return self.results[0].scientific_contract.seed

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(result.family_id for result in self.results)

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def result(self, family_id: str) -> ExperimentResultV3:
        for result in self.results:
            if result.family_id == family_id:
                return result
        raise KeyError(family_id)

    def pairwise_effect_at(
        self,
        candidate_family_id: str,
        reference_family_id: str,
        point_index: int,
    ) -> PairwiseNllEffect:
        candidate = self.result(candidate_family_id)
        reference = self.result(reference_family_id)
        candidate_point = candidate.evaluation_points[point_index]
        reference_point = reference.evaluation_points[point_index]
        reference_mean = reference_point.aggregate_nll
        relative = (reference_mean - candidate_point.aggregate_nll) / reference_mean
        component_effects = tuple(
            (reference_score - candidate_score) / reference_mean
            for candidate_score, reference_score in zip(
                candidate_point.component_losses,
                reference_point.component_losses,
                strict=True,
            )
        )
        standard_error = _weighted_paired_cluster_se(
            component_effects, candidate_point.component_loss_token_weights
        )
        return PairwiseNllEffect(
            candidate_family_id=candidate_family_id,
            reference_family_id=reference_family_id,
            at_loss_tokens=candidate_point.at_loss_tokens,
            relative_nll_improvement=relative,
            paired_weighted_cluster_se=standard_error,
            threshold=_POLICY.threshold(standard_error),
        )

    def pairwise_effect(
        self, candidate_family_id: str, reference_family_id: str
    ) -> PairwiseNllEffect:
        return self.pairwise_effect_at(candidate_family_id, reference_family_id, -1)

    def _terminal_regression(
        self, family_id: str, late_window_start: int
    ) -> TerminalRegressionEvidence:
        result = self.result(family_id)
        terminal = result.evaluation_points[-1]
        prior_late = tuple(
            point
            for point in result.evaluation_points[:-1]
            if point.at_loss_tokens >= late_window_start
        )
        if not prior_late:
            return TerminalRegressionEvidence(
                family_id=family_id,
                best_prior_at_loss_tokens=None,
                terminal_at_loss_tokens=terminal.at_loss_tokens,
                relative_nll_regression=None,
                paired_weighted_cluster_se=None,
                threshold=None,
                material=False,
            )
        best_prior = min(prior_late, key=lambda point: point.aggregate_nll)
        relative = (
            terminal.aggregate_nll - best_prior.aggregate_nll
        ) / best_prior.aggregate_nll
        component_regressions = tuple(
            (terminal_score - prior_score) / best_prior.aggregate_nll
            for terminal_score, prior_score in zip(
                terminal.component_losses,
                best_prior.component_losses,
                strict=True,
            )
        )
        standard_error = _weighted_paired_cluster_se(
            component_regressions, terminal.component_loss_token_weights
        )
        threshold = _POLICY.threshold(standard_error)
        return TerminalRegressionEvidence(
            family_id=family_id,
            best_prior_at_loss_tokens=best_prior.at_loss_tokens,
            terminal_at_loss_tokens=terminal.at_loss_tokens,
            relative_nll_regression=relative,
            paired_weighted_cluster_se=standard_error,
            threshold=threshold,
            material=relative >= threshold,
        )

    def candidate_assessment(
        self, family_id: str, reference_family_id: str = _CONTROL_FAMILY
    ) -> CandidateAssessment:
        effect = self.pairwise_effect(family_id, reference_family_id)
        candidate = self.result(family_id)
        reference = self.result(reference_family_id)
        interval = candidate.scientific_contract.condition.evaluation_interval_loss_tokens
        late_start = _POLICY.late_window_start(interval)
        late_effects = tuple(
            self.pairwise_effect_at(family_id, reference_family_id, index)
            for index, point in enumerate(candidate.evaluation_points)
            if point.at_loss_tokens >= late_start
        )
        candidate_regression = self._terminal_regression(family_id, late_start)
        reference_regression = self._terminal_regression(reference_family_id, late_start)
        terminal_effect = effect.relative_nll_improvement
        reasons: list[str] = []
        if terminal_effect >= effect.threshold:
            if not candidate.curve_learned:
                reasons.append("candidate_curve_nonlearning")
            if candidate_regression.material:
                reasons.append("candidate_material_terminal_regression")
            late_opposite = any(
                item.relative_nll_improvement <= -item.threshold
                for item in late_effects
            )
            if late_opposite:
                reasons.append("late_control_favoring_effect")
            status = "unresolved" if reasons else "supported"
        elif terminal_effect <= -effect.threshold:
            if not reference.curve_learned:
                reasons.append("reference_curve_nonlearning")
            if reference_regression.material:
                reasons.append("reference_material_terminal_regression")
            late_opposite = any(
                item.relative_nll_improvement >= item.threshold
                for item in late_effects
            )
            if late_opposite:
                reasons.append("late_candidate_favoring_effect")
            status = "unresolved" if reasons else "opposed"
        else:
            late_opposite = False
            status = "unresolved"
        return CandidateAssessment(
            family_id=family_id,
            reference_family_id=reference_family_id,
            effect=effect,
            status=status,
            late_window_start_loss_tokens=late_start,
            late_effects=late_effects,
            candidate_curve_learned=candidate.curve_learned,
            reference_curve_learned=reference.curve_learned,
            candidate_terminal_regression=candidate_regression,
            reference_terminal_regression=reference_regression,
            late_opposite_direction=late_opposite,
            guard_reasons=tuple(reasons),
        )

    @property
    def candidate_assessments(self) -> tuple[CandidateAssessment, ...]:
        return tuple(
            self.candidate_assessment(family_id)
            for family_id in _CHALLENGER_FAMILIES
            if family_id in self.families
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_multi_arm_comparison_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "phase": self.phase,
            "seed": self.seed,
            "result_hashes": tuple(result.identity for result in self.results),
            "candidate_assessments": tuple(
                asdict(assessment) for assessment in self.candidate_assessments
            ),
        }


def load_multi_arm_comparison_receipt(
    payload: Mapping[str, Any],
    *,
    results_by_hash: Mapping[str, ExperimentResultV3],
) -> MultiArmComparisonReceipt:
    """Strictly read back a comparison from independently validated results."""

    fields = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "phase",
        "seed",
        "result_hashes",
        "candidate_assessments",
    }
    _require_artifact_boundary(
        payload,
        fields=fields,
        artifact_kind="twc_configuration_multi_arm_comparison_v1",
        schema_version=1,
        label="multi-arm comparison",
    )
    result_hashes = payload.get("result_hashes")
    if not isinstance(result_hashes, (list, tuple)):
        raise ValueError("multi-arm comparison result hashes are invalid")
    results: list[ExperimentResultV3] = []
    for index, result_hash in enumerate(result_hashes):
        _require_hash(result_hash, f"result_hashes[{index}]")
        result = results_by_hash.get(result_hash)
        if result is None:
            raise ValueError("multi-arm comparison result hash is unavailable")
        if not isinstance(result, ExperimentResultV3) or result.identity != result_hash:
            raise ValueError("multi-arm comparison result map is invalid")
        results.append(result)
    receipt = MultiArmComparisonReceipt(results=tuple(results))
    if stable_json_hash(payload) != stable_json_hash(receipt.to_dict()):
        raise ValueError("multi-arm comparison derived evidence mismatch")
    return receipt


@dataclass(frozen=True, slots=True)
class C3bRequirement:
    """Pure branch receipt: no run, or seed-43 control plus supported arms."""

    source_comparison_hash: str
    branch: str
    required_conditions: tuple[StudyCondition, ...]

    def __post_init__(self) -> None:
        _require_hash(self.source_comparison_hash, "source_comparison_hash")
        if self.branch not in {"not_required", "required"}:
            raise ValueError("C3b requirement branch is invalid")
        if any(not isinstance(item, StudyCondition) for item in self.required_conditions):
            raise ValueError("C3b requirement contains an invalid condition")
        if self.branch == "not_required":
            if self.required_conditions:
                raise ValueError("not-required C3b branch cannot contain conditions")
            return
        if not 2 <= len(self.required_conditions) <= 3:
            raise ValueError("required C3b branch needs control and supported challengers")
        families = tuple(item.family_id for item in self.required_conditions)
        expected = tuple(family for family in _FAMILY_ORDER if family in families)
        if families != expected or families[0] != _CONTROL_FAMILY:
            raise ValueError("C3b conditions must use canonical control-first family order")
        if any(item.phase != "l_confirmation" or item.seed != 43 for item in self.required_conditions):
            raise ValueError("C3b conditions must be L-scale seed-43 conditions")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_c3b_requirement_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "source_comparison_hash": self.source_comparison_hash,
            "branch": self.branch,
            "required_conditions": tuple(
                condition.to_dict() for condition in self.required_conditions
            ),
        }


def derive_c3b_requirement(
    c3a_comparison: MultiArmComparisonReceipt,
    available_conditions: Iterable[StudyCondition],
) -> C3bRequirement:
    """Return no work, or the exact seed-43 block implied by C3a support."""

    if not isinstance(c3a_comparison, MultiArmComparisonReceipt):
        raise ValueError("C3b derivation requires a multi-arm comparison")
    if (
        c3a_comparison.phase != "l_confirmation"
        or c3a_comparison.seed != 29
        or c3a_comparison.families != _FAMILY_ORDER
    ):
        raise ValueError("C3b derivation requires the complete C3a seed-29 block")
    supported = tuple(
        assessment.family_id
        for assessment in c3a_comparison.candidate_assessments
        if assessment.status == "supported"
    )
    if not supported:
        return C3bRequirement(
            source_comparison_hash=c3a_comparison.identity,
            branch="not_required",
            required_conditions=(),
        )
    rows = tuple(available_conditions)
    if any(not isinstance(condition, StudyCondition) for condition in rows):
        raise ValueError("available C3b conditions must be StudyCondition values")
    by_family = {
        condition.family_id: condition
        for condition in rows
        if condition.phase == "l_confirmation" and condition.seed == 43
    }
    required_families = tuple(
        family for family in _FAMILY_ORDER if family == _CONTROL_FAMILY or family in supported
    )
    missing = tuple(family for family in required_families if family not in by_family)
    if missing:
        raise ValueError(f"available conditions are missing C3b families: {missing}")
    return C3bRequirement(
        source_comparison_hash=c3a_comparison.identity,
        branch="required",
        required_conditions=tuple(by_family[family] for family in required_families),
    )


def load_c3b_requirement(
    payload: Mapping[str, Any],
    *,
    source_comparison: MultiArmComparisonReceipt,
) -> C3bRequirement:
    """Strictly read back the branch against the C3a receipt that caused it."""

    fields = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "source_comparison_hash",
        "branch",
        "required_conditions",
    }
    _require_artifact_boundary(
        payload,
        fields=fields,
        artifact_kind="twc_configuration_c3b_requirement_v1",
        schema_version=1,
        label="C3b requirement",
    )
    if not isinstance(source_comparison, MultiArmComparisonReceipt):
        raise ValueError("C3b source comparison must be a validated receipt")
    _require_hash(payload.get("source_comparison_hash"), "source_comparison_hash")
    if payload.get("source_comparison_hash") != source_comparison.identity:
        raise ValueError("C3b source comparison hash mismatch")
    condition_rows = payload.get("required_conditions")
    if not isinstance(condition_rows, (list, tuple)):
        raise ValueError("C3b required conditions are invalid")
    condition_fields = set(StudyCondition.__dataclass_fields__)
    conditions: list[StudyCondition] = []
    for row in condition_rows:
        if not isinstance(row, Mapping) or set(row) != condition_fields:
            raise ValueError("C3b condition fields are invalid")
        values = dict(row)
        checkpoints = values.get("required_exact_evaluation_loss_tokens")
        if not isinstance(checkpoints, (list, tuple)):
            raise ValueError("C3b condition checkpoints are invalid")
        values["required_exact_evaluation_loss_tokens"] = tuple(checkpoints)
        conditions.append(StudyCondition(**values))
    parsed = C3bRequirement(
        source_comparison_hash=payload["source_comparison_hash"],
        branch=payload["branch"],
        required_conditions=tuple(conditions),
    )
    canonical = derive_c3b_requirement(source_comparison, parsed.required_conditions)
    if stable_json_hash(payload) != stable_json_hash(canonical.to_dict()):
        raise ValueError("C3b requirement derived evidence mismatch")
    return canonical


@dataclass(frozen=True, slots=True)
class JointTwoSeedEvidence:
    """Exact equal-seed evidence and the frozen joint uncertainty calculation."""

    candidate_family_id: str
    reference_family_id: str
    seed29_relative_nll_improvement: float
    seed43_relative_nll_improvement: float
    seed29_paired_weighted_cluster_se: float
    seed43_paired_weighted_cluster_se: float
    equal_seed_mean_relative_nll_improvement: float
    evaluation_mean_standard_error: float
    between_seed_sem: float
    combined_standard_error: float
    threshold: float
    clears_joint_threshold: bool

    def __post_init__(self) -> None:
        _require_identifier(self.candidate_family_id, "candidate_family_id")
        _require_identifier(self.reference_family_id, "reference_family_id")
        if self.candidate_family_id == self.reference_family_id:
            raise ValueError("joint evidence requires distinct families")
        expected = _POLICY.joint_two_seed_components(
            seed29_effect=self.seed29_relative_nll_improvement,
            seed29_standard_error=self.seed29_paired_weighted_cluster_se,
            seed43_effect=self.seed43_relative_nll_improvement,
            seed43_standard_error=self.seed43_paired_weighted_cluster_se,
        )
        actual = (
            self.equal_seed_mean_relative_nll_improvement,
            self.evaluation_mean_standard_error,
            self.between_seed_sem,
            self.combined_standard_error,
            self.threshold,
        )
        if any(
            not math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=1e-15)
            for value, expected_value in zip(actual, expected, strict=True)
        ):
            raise ValueError("joint two-seed evidence does not match frozen formula")
        expected_clear = self.equal_seed_mean_relative_nll_improvement >= self.threshold
        if self.clears_joint_threshold != expected_clear:
            raise ValueError("joint two-seed threshold flag is inconsistent")


def _joint_two_seed_evidence(
    seed29_effect: PairwiseNllEffect,
    seed43_effect: PairwiseNllEffect,
) -> JointTwoSeedEvidence:
    if (
        seed29_effect.candidate_family_id != seed43_effect.candidate_family_id
        or seed29_effect.reference_family_id != seed43_effect.reference_family_id
    ):
        raise ValueError("joint two-seed effects do not compare the same families")
    values = _POLICY.joint_two_seed_components(
        seed29_effect=seed29_effect.relative_nll_improvement,
        seed29_standard_error=seed29_effect.paired_weighted_cluster_se,
        seed43_effect=seed43_effect.relative_nll_improvement,
        seed43_standard_error=seed43_effect.paired_weighted_cluster_se,
    )
    mean, evaluation_se, between_seed_sem, combined_se, threshold = values
    return JointTwoSeedEvidence(
        candidate_family_id=seed29_effect.candidate_family_id,
        reference_family_id=seed29_effect.reference_family_id,
        seed29_relative_nll_improvement=seed29_effect.relative_nll_improvement,
        seed43_relative_nll_improvement=seed43_effect.relative_nll_improvement,
        seed29_paired_weighted_cluster_se=seed29_effect.paired_weighted_cluster_se,
        seed43_paired_weighted_cluster_se=seed43_effect.paired_weighted_cluster_se,
        equal_seed_mean_relative_nll_improvement=mean,
        evaluation_mean_standard_error=evaluation_se,
        between_seed_sem=between_seed_sem,
        combined_standard_error=combined_se,
        threshold=threshold,
        clears_joint_threshold=mean >= threshold,
    )


@dataclass(frozen=True, slots=True)
class FinalCandidateAssessment:
    family_id: str
    endpoint_profile_id: str
    c2_status: str
    c3_seed29_status: str
    c3_seed43_status: str | None
    scale_reversal: bool
    scale_reversal_interpretation_evidence_hash: str | None
    joint_evidence: JointTwoSeedEvidence | None
    two_seed_supported: bool
    promotion_qualified: bool
    worst_seed_terminal_nll: float | None
    equal_seed_mean_terminal_nll: float | None

    def __post_init__(self) -> None:
        if self.family_id not in _CHALLENGER_FAMILIES:
            raise ValueError("final candidate family is invalid")
        _require_identifier(self.endpoint_profile_id, "endpoint_profile_id")
        if self.c2_status not in _STATUSES or self.c3_seed29_status not in _STATUSES:
            raise ValueError("final candidate status is invalid")
        if self.c3_seed43_status is not None and self.c3_seed43_status not in _STATUSES:
            raise ValueError("final seed-43 status is invalid")
        if any(
            type(value) is not bool
            for value in (
                self.scale_reversal,
                self.two_seed_supported,
                self.promotion_qualified,
            )
        ):
            raise ValueError("final candidate guards must be booleans")
        if self.scale_reversal_interpretation_evidence_hash is not None:
            _require_hash(
                self.scale_reversal_interpretation_evidence_hash,
                "scale_reversal_interpretation_evidence_hash",
            )
        expected_two_seed_support = (
            self.c3_seed29_status == "supported"
            and self.c3_seed43_status == "supported"
        )
        if self.two_seed_supported != expected_two_seed_support:
            raise ValueError("final candidate two-seed support flag is inconsistent")
        if self.c3_seed43_status is not None and self.c3_seed29_status != "supported":
            raise ValueError("final candidate seed-43 evidence was not opened by C3a")
        expected_scale_reversal = (
            _status_direction(self.c2_status)
            * _status_direction(self.c3_seed29_status)
            == -1
        )
        if self.scale_reversal != expected_scale_reversal:
            raise ValueError("final candidate scale-reversal flag is inconsistent")
        if self.joint_evidence is not None:
            if (
                not isinstance(self.joint_evidence, JointTwoSeedEvidence)
                or self.joint_evidence.candidate_family_id != self.family_id
                or self.joint_evidence.reference_family_id != _CONTROL_FAMILY
            ):
                raise ValueError("final candidate joint evidence is invalid")
        if (self.joint_evidence is None) != (self.c3_seed43_status is None):
            raise ValueError("final candidate joint evidence presence is inconsistent")
        if self.worst_seed_terminal_nll is not None:
            _require_finite(
                self.worst_seed_terminal_nll,
                "worst_seed_terminal_nll",
                nonnegative=True,
            )
        if self.equal_seed_mean_terminal_nll is not None:
            _require_finite(
                self.equal_seed_mean_terminal_nll,
                "equal_seed_mean_terminal_nll",
                nonnegative=True,
            )
        if (self.worst_seed_terminal_nll is None) != (
            self.equal_seed_mean_terminal_nll is None
        ):
            raise ValueError("final candidate seed NLL summaries must be complete")
        if (self.worst_seed_terminal_nll is None) != (self.c3_seed43_status is None):
            raise ValueError("final candidate seed NLL evidence presence is inconsistent")
        if (
            self.worst_seed_terminal_nll is not None
            and self.equal_seed_mean_terminal_nll is not None
            and self.worst_seed_terminal_nll < self.equal_seed_mean_terminal_nll
        ):
            raise ValueError("final candidate worst-seed NLL is inconsistent")
        otherwise_qualified = bool(
            self.two_seed_supported
            and self.joint_evidence is not None
            and self.joint_evidence.clears_joint_threshold
        )
        needs_reversal = otherwise_qualified and self.c2_status == "opposed"
        if self.scale_reversal_interpretation_evidence_hash is not None and not needs_reversal:
            raise ValueError(
                "final candidate has extraneous scale-reversal interpretation evidence"
            )
        expected_promotion = bool(
            otherwise_qualified
            and self.c2_status != "blocked"
            and (
                not needs_reversal
                or self.scale_reversal_interpretation_evidence_hash is not None
            )
        )
        if self.promotion_qualified != expected_promotion:
            raise ValueError("final candidate promotion flag is inconsistent")


@dataclass(frozen=True, slots=True)
class FinalConfigurationAssessment:
    """Final C2/C3 reduction with an explicit conservative endpoint outcome."""

    c2_comparison_hash: str
    c3_seed29_comparison_hash: str
    c3_seed43_comparison_hash: str | None
    candidate_summaries: tuple[FinalCandidateAssessment, ...]
    decision: str
    selected_family_id: str | None
    selected_endpoint_profile_id: str | None
    selection_basis: str
    direct_seed29_effect: PairwiseNllEffect | None
    direct_seed43_effect: PairwiseNllEffect | None
    direct_joint_evidence: JointTwoSeedEvidence | None
    target_diagnostics_remained_sealed: bool = True

    def __post_init__(self) -> None:
        _require_hash(self.c2_comparison_hash, "c2_comparison_hash")
        _require_hash(self.c3_seed29_comparison_hash, "c3_seed29_comparison_hash")
        if self.c3_seed43_comparison_hash is not None:
            _require_hash(self.c3_seed43_comparison_hash, "c3_seed43_comparison_hash")
        if tuple(item.family_id for item in self.candidate_summaries) != _CHALLENGER_FAMILIES:
            raise ValueError("final assessment requires both candidate summaries in canonical order")
        if self.decision not in {
            "select_challenger",
            "retain_control",
            "challengers_unordered",
        }:
            raise ValueError("final configuration decision is invalid")
        if not isinstance(self.selection_basis, str) or not self.selection_basis:
            raise ValueError("final configuration selection basis is required")
        if (self.direct_seed29_effect is None) != (self.direct_seed43_effect is None):
            raise ValueError("final direct seed evidence must be complete")
        if self.direct_joint_evidence is not None and (
            self.direct_seed29_effect is None or self.direct_seed43_effect is None
        ):
            raise ValueError("final direct joint evidence requires both seed effects")
        if self.target_diagnostics_remained_sealed is not True:
            raise ValueError("target diagnostics must remain sealed during selection")
        if self.decision == "select_challenger":
            if self.selected_family_id not in _CHALLENGER_FAMILIES:
                raise ValueError("challenger decision requires a selected challenger")
            _require_identifier(self.selected_endpoint_profile_id, "selected_endpoint_profile_id")
        elif self.decision == "retain_control":
            if self.selected_family_id != _CONTROL_FAMILY:
                raise ValueError("retain-control decision must select the control family")
            _require_identifier(self.selected_endpoint_profile_id, "selected_endpoint_profile_id")
        elif self.selected_family_id is not None or self.selected_endpoint_profile_id is not None:
            raise ValueError("unordered challengers cannot select an endpoint")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_final_assessment_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "c2_comparison_hash": self.c2_comparison_hash,
            "c3_seed29_comparison_hash": self.c3_seed29_comparison_hash,
            "c3_seed43_comparison_hash": self.c3_seed43_comparison_hash,
            "candidate_summaries": tuple(asdict(item) for item in self.candidate_summaries),
            "decision": self.decision,
            "selected_family_id": self.selected_family_id,
            "selected_endpoint_profile_id": self.selected_endpoint_profile_id,
            "selection_basis": self.selection_basis,
            "direct_seed29_effect": (
                asdict(self.direct_seed29_effect)
                if self.direct_seed29_effect is not None
                else None
            ),
            "direct_seed43_effect": (
                asdict(self.direct_seed43_effect)
                if self.direct_seed43_effect is not None
                else None
            ),
            "direct_joint_evidence": (
                asdict(self.direct_joint_evidence)
                if self.direct_joint_evidence is not None
                else None
            ),
            "target_diagnostics_remained_sealed": self.target_diagnostics_remained_sealed,
        }


def _require_study_continuity(
    c2: MultiArmComparisonReceipt,
    c3a: MultiArmComparisonReceipt,
) -> None:
    c2_contract = c2.results[0].scientific_contract
    c3_contract = c3a.results[0].scientific_contract
    for field in (
        "study_plan_hash",
        "comparison_spec_hash",
        "data_manifest_hash",
        "validation_view_hash",
        "tokenizer_hash",
        "evaluation_manifest_hash",
        "semantic_code_hash",
        "learning_rate",
        "schedule_tail_fraction",
        "paired_order_seed",
    ):
        if getattr(c2_contract, field) != getattr(c3_contract, field):
            raise ValueError(f"C2/C3 scientific continuity mismatch: {field}")
    for family_id in _FAMILY_ORDER:
        if (
            c2.result(family_id).scientific_contract.condition.endpoint_profile_id
            != c3a.result(family_id).scientific_contract.condition.endpoint_profile_id
        ):
            raise ValueError("C2/C3 endpoint projection mismatch")


def _status_direction(status: str) -> int:
    return {"supported": 1, "opposed": -1}.get(status, 0)


def assess_final_configuration(
    c2_comparison: MultiArmComparisonReceipt,
    c3_seed29_comparison: MultiArmComparisonReceipt,
    c3_seed43_comparison: MultiArmComparisonReceipt | None,
    *,
    scale_reversal_interpretation_evidence_hashes: Mapping[str, str] | None = None,
) -> FinalConfigurationAssessment:
    """Apply the hash-bound two-seed, reversal, and direct-ordering policy."""

    if not isinstance(c2_comparison, MultiArmComparisonReceipt) or not isinstance(
        c3_seed29_comparison, MultiArmComparisonReceipt
    ):
        raise ValueError("final assessment requires validated comparison receipts")
    if (
        c2_comparison.phase != "m_transfer"
        or c2_comparison.seed != 17
        or c2_comparison.families != _FAMILY_ORDER
    ):
        raise ValueError("final assessment requires the complete C2 seed-17 block")
    if (
        c3_seed29_comparison.phase != "l_confirmation"
        or c3_seed29_comparison.seed != 29
        or c3_seed29_comparison.families != _FAMILY_ORDER
    ):
        raise ValueError("final assessment requires the complete C3a seed-29 block")
    _require_study_continuity(c2_comparison, c3_seed29_comparison)
    reversal_hashes = dict(scale_reversal_interpretation_evidence_hashes or {})
    for family_id, evidence_hash in reversal_hashes.items():
        if family_id not in _CHALLENGER_FAMILIES:
            raise ValueError("extraneous scale-reversal interpretation family")
        _require_hash(evidence_hash, f"scale_reversal_interpretation[{family_id}]")

    c2_statuses = {
        item.family_id: item.status for item in c2_comparison.candidate_assessments
    }
    c3a_statuses = {
        item.family_id: item.status
        for item in c3_seed29_comparison.candidate_assessments
    }
    c3a_supported = tuple(
        family for family in _CHALLENGER_FAMILIES if c3a_statuses[family] == "supported"
    )
    if not c3a_supported:
        if c3_seed43_comparison is not None:
            raise ValueError("C3b comparison is forbidden when C3a supports no challenger")
        if reversal_hashes:
            raise ValueError("extraneous scale-reversal interpretation evidence")
        summaries = tuple(
            FinalCandidateAssessment(
                family_id=family,
                endpoint_profile_id=c3_seed29_comparison.result(family)
                .scientific_contract.condition.endpoint_profile_id,
                c2_status=c2_statuses[family],
                c3_seed29_status=c3a_statuses[family],
                c3_seed43_status=None,
                scale_reversal=(
                    _status_direction(c2_statuses[family])
                    * _status_direction(c3a_statuses[family])
                    == -1
                ),
                scale_reversal_interpretation_evidence_hash=None,
                joint_evidence=None,
                two_seed_supported=False,
                promotion_qualified=False,
                worst_seed_terminal_nll=None,
                equal_seed_mean_terminal_nll=None,
            )
            for family in _CHALLENGER_FAMILIES
        )
        return FinalConfigurationAssessment(
            c2_comparison_hash=c2_comparison.identity,
            c3_seed29_comparison_hash=c3_seed29_comparison.identity,
            c3_seed43_comparison_hash=None,
            candidate_summaries=summaries,
            decision="retain_control",
            selected_family_id=_CONTROL_FAMILY,
            selected_endpoint_profile_id=c3_seed29_comparison.result(_CONTROL_FAMILY)
            .scientific_contract.condition.endpoint_profile_id,
            selection_basis="no_c3a_supported_challenger",
            direct_seed29_effect=None,
            direct_seed43_effect=None,
            direct_joint_evidence=None,
        )

    if c3_seed43_comparison is None:
        raise ValueError("C3b comparison is required for C3a-supported challengers")
    if (
        not isinstance(c3_seed43_comparison, MultiArmComparisonReceipt)
        or c3_seed43_comparison.phase != "l_confirmation"
        or c3_seed43_comparison.seed != 43
    ):
        raise ValueError("C3b comparison must be the validated L-scale seed-43 block")
    required_families = tuple(
        family for family in _FAMILY_ORDER if family == _CONTROL_FAMILY or family in c3a_supported
    )
    if c3_seed43_comparison.families != required_families:
        raise ValueError("C3b comparison does not contain the required C3b families")
    first_a = c3_seed29_comparison.results[0].scientific_contract
    first_b = c3_seed43_comparison.results[0].scientific_contract
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
        "learning_rate",
        "schedule_tail_fraction",
        "paired_order_seed",
        "target_loss_tokens",
    ):
        if getattr(first_a, field) != getattr(first_b, field):
            raise ValueError(f"C3 seed continuity mismatch: {field}")

    c3b_statuses = {
        item.family_id: item.status
        for item in c3_seed43_comparison.candidate_assessments
    }
    summaries: list[FinalCandidateAssessment] = []
    qualifiers: list[str] = []
    allowed_reversal_hash_families: set[str] = set()
    for family in _CHALLENGER_FAMILIES:
        seed43_status = c3b_statuses.get(family)
        two_seed_supported = (
            c3a_statuses[family] == "supported" and seed43_status == "supported"
        )
        worst = None
        equal_seed_mean = None
        joint_evidence = None
        if seed43_status is not None:
            seed29_nll = c3_seed29_comparison.result(family).terminal_validation_nll
            seed43_nll = c3_seed43_comparison.result(family).terminal_validation_nll
            worst = max(seed29_nll, seed43_nll)
            equal_seed_mean = (seed29_nll + seed43_nll) / 2
            joint_evidence = _joint_two_seed_evidence(
                c3_seed29_comparison.pairwise_effect(family, _CONTROL_FAMILY),
                c3_seed43_comparison.pairwise_effect(family, _CONTROL_FAMILY),
            )
        otherwise_c3_qualified = bool(
            two_seed_supported
            and joint_evidence is not None
            and joint_evidence.clears_joint_threshold
        )
        needs_reversal_evidence = otherwise_c3_qualified and c2_statuses[family] == "opposed"
        if needs_reversal_evidence:
            allowed_reversal_hash_families.add(family)
        reversal_hash = reversal_hashes.get(family)
        promotion_qualified = bool(
            otherwise_c3_qualified
            and c2_statuses[family] != "blocked"
            and (not needs_reversal_evidence or reversal_hash is not None)
        )
        if promotion_qualified:
            qualifiers.append(family)
        summaries.append(
            FinalCandidateAssessment(
                family_id=family,
                endpoint_profile_id=c3_seed29_comparison.result(family)
                .scientific_contract.condition.endpoint_profile_id,
                c2_status=c2_statuses[family],
                c3_seed29_status=c3a_statuses[family],
                c3_seed43_status=seed43_status,
                scale_reversal=(
                    _status_direction(c2_statuses[family])
                    * _status_direction(c3a_statuses[family])
                    == -1
                ),
                scale_reversal_interpretation_evidence_hash=reversal_hash,
                joint_evidence=joint_evidence,
                two_seed_supported=two_seed_supported,
                promotion_qualified=promotion_qualified,
                worst_seed_terminal_nll=worst,
                equal_seed_mean_terminal_nll=equal_seed_mean,
            )
        )

    extraneous_reversal_hashes = set(reversal_hashes) - allowed_reversal_hash_families
    if extraneous_reversal_hashes:
        raise ValueError("extraneous scale-reversal interpretation evidence")

    direct_seed29_effect: PairwiseNllEffect | None = None
    direct_seed43_effect: PairwiseNllEffect | None = None
    direct_joint_evidence: JointTwoSeedEvidence | None = None
    if not qualifiers:
        decision = "retain_control"
        selected_family: str | None = _CONTROL_FAMILY
        if any(
            summary.two_seed_supported
            and summary.joint_evidence is not None
            and not summary.joint_evidence.clears_joint_threshold
            for summary in summaries
        ):
            basis = "no_jointly_supported_challenger"
        elif any(
            summary.two_seed_supported
            and summary.joint_evidence is not None
            and summary.joint_evidence.clears_joint_threshold
            and summary.c2_status == "opposed"
            for summary in summaries
        ):
            basis = "missing_scale_reversal_interpretation_evidence"
        else:
            basis = "no_two_seed_supported_challenger"
    elif len(qualifiers) == 1:
        decision = "select_challenger"
        selected_family = qualifiers[0]
        basis = "single_promotion_qualified_challenger"
    else:
        left, right = qualifiers
        direct_seed29_effect = c3_seed29_comparison.pairwise_effect(left, right)
        direct_seed43_effect = c3_seed43_comparison.pairwise_effect(left, right)
        direct_joint_evidence = _joint_two_seed_evidence(
            direct_seed29_effect, direct_seed43_effect
        )
        right_seed29 = c3_seed29_comparison.pairwise_effect(right, left)
        right_seed43 = c3_seed43_comparison.pairwise_effect(right, left)
        left_dominates = all(
            item.relative_nll_improvement >= item.threshold
            for item in (direct_seed29_effect, direct_seed43_effect)
        )
        right_dominates = all(
            item.relative_nll_improvement >= item.threshold
            for item in (right_seed29, right_seed43)
        )
        if left_dominates != right_dominates:
            selected_family = left if left_dominates else right
            decision = "select_challenger"
            basis = "direct_two_seed_dominance"
        elif (
            abs(direct_joint_evidence.equal_seed_mean_relative_nll_improvement)
            < direct_joint_evidence.threshold
        ):
            worst_by_family = {
                summary.family_id: summary.worst_seed_terminal_nll
                for summary in summaries
                if summary.two_seed_supported
            }
            left_worst = worst_by_family[left]
            right_worst = worst_by_family[right]
            assert left_worst is not None and right_worst is not None
            if not math.isclose(left_worst, right_worst, rel_tol=0, abs_tol=1e-12):
                selected_family = left if left_worst < right_worst else right
                decision = "select_challenger"
                basis = "equivalent_within_resolution_worst_seed"
            else:
                mean_by_family = {
                    summary.family_id: summary.equal_seed_mean_terminal_nll
                    for summary in summaries
                    if summary.promotion_qualified
                }
                left_mean = mean_by_family[left]
                right_mean = mean_by_family[right]
                assert left_mean is not None and right_mean is not None
                if not math.isclose(left_mean, right_mean, rel_tol=0, abs_tol=1e-12):
                    selected_family = left if left_mean < right_mean else right
                    decision = "select_challenger"
                    basis = "equivalent_within_resolution_equal_seed_mean"
                else:
                    selected_family = None
                    decision = "challengers_unordered"
                    basis = "equivalent_within_resolution_exact_tie"
        else:
            selected_family = None
            decision = "challengers_unordered"
            basis = "direct_inconsistent_beyond_resolution"

    endpoint = None
    if selected_family is not None:
        endpoint = c3_seed29_comparison.result(selected_family)
        selected_endpoint = endpoint.scientific_contract.condition.endpoint_profile_id
    else:
        selected_endpoint = None
    return FinalConfigurationAssessment(
        c2_comparison_hash=c2_comparison.identity,
        c3_seed29_comparison_hash=c3_seed29_comparison.identity,
        c3_seed43_comparison_hash=c3_seed43_comparison.identity,
        candidate_summaries=tuple(summaries),
        decision=decision,
        selected_family_id=selected_family,
        selected_endpoint_profile_id=selected_endpoint,
        selection_basis=basis,
        direct_seed29_effect=direct_seed29_effect,
        direct_seed43_effect=direct_seed43_effect,
        direct_joint_evidence=direct_joint_evidence,
    )


def load_final_configuration_assessment(
    payload: Mapping[str, Any],
    *,
    c2_comparison: MultiArmComparisonReceipt,
    c3_seed29_comparison: MultiArmComparisonReceipt,
    c3_seed43_comparison: MultiArmComparisonReceipt | None,
    scale_reversal_interpretation_evidence_hashes: Mapping[str, str] | None = None,
) -> FinalConfigurationAssessment:
    """Strictly read back a final decision by recomputing it from its evidence."""

    fields = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "c2_comparison_hash",
        "c3_seed29_comparison_hash",
        "c3_seed43_comparison_hash",
        "candidate_summaries",
        "decision",
        "selected_family_id",
        "selected_endpoint_profile_id",
        "selection_basis",
        "direct_seed29_effect",
        "direct_seed43_effect",
        "direct_joint_evidence",
        "target_diagnostics_remained_sealed",
    }
    _require_artifact_boundary(
        payload,
        fields=fields,
        artifact_kind="twc_configuration_final_assessment_v1",
        schema_version=1,
        label="final configuration assessment",
    )
    canonical = assess_final_configuration(
        c2_comparison,
        c3_seed29_comparison,
        c3_seed43_comparison,
        scale_reversal_interpretation_evidence_hashes=(
            scale_reversal_interpretation_evidence_hashes
        ),
    )
    if stable_json_hash(payload) != stable_json_hash(canonical.to_dict()):
        raise ValueError("final assessment evidence mismatch")
    return canonical
