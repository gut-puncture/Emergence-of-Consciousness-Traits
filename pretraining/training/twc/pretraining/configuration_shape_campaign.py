"""Diagnostic-only campaign and bundle contracts for C2/C3 GPU shape probes.

This module deliberately sits between the scientific study and the ordinary
training compiler.  It derives the same model, schedule, data-order, and engine
settings, but it never constructs or serializes an experiment training
contract.  A shape bundle can therefore exercise the real hot path without
creating scientific-result authority or a circular dependency on the selected
execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import re
from typing import Any, Mapping, Sequence

from twc.pretraining.campaign import AttemptPlan, CampaignPlan, WavePlan
from twc.pretraining.configuration_shape_probe import (
    CONFIGURATION_SHAPE_CAPS,
    SHAPE_MAXIMUM_INPUT_WAIT_FRACTION,
    SHAPE_MINIMUM_MEMORY_HEADROOM_FRACTION,
    SHAPE_PROBE_MEASURED_STEPS,
    SHAPE_PROBE_WARMUP_STEPS,
    SHAPE_THROUGHPUT_KNEE_FRACTION,
)
from twc.pretraining.configuration_study import (
    ConfigurationStudyPlanV2,
    StudyCondition,
    load_configuration_study_plan,
)
from twc.pretraining.configuration_study_compiler import (
    DATA_ORDER_POLICY_V2,
    SCHEDULE_POLICY_V2,
    ConfigurationCompilerInputReceiptV1,
    compile_configuration_model_artifact,
    load_configuration_compiler_input_receipt,
)
from twc.pretraining.configuration_study_execution import (
    ConfigurationShapeProbeExecutionPathReceiptV1,
    load_configuration_shape_probe_execution_path_receipt,
)
from twc.repro.hashing import stable_json_hash


_HASH = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_GLOBAL_CAPS = (262_144, 1_073_741_824, 4_096)
_STAGE_SPEC = {
    "c2": {
        "phase": "m_transfer",
        "seed": 17,
        "profiles": ("m_wide", "m_prior_deep"),
    },
    "c3": {
        "phase": "l_confirmation",
        "seed": 29,
        "profiles": ("l_wide", "l_prior_deep"),
    },
}
_RUNTIME = {
    "precision": "bf16",
    "attention_backend": "native_varlen",
    "attention_num_splits": "auto",
    "loss_backend": "chunked_projection",
    "compile_mode": "default",
    "allow_tf32": True,
    "optimizer": "adamw",
    "optimizer_backend": "fused",
    "adam_betas": (0.9, 0.95),
    "adam_eps": 1e-8,
    "weight_decay": 0.1,
    "grad_clip_norm": 1.0,
}
_BUNDLE_DERIVED_FIELDS = {
    "condition_id",
    "seed",
    "condition_hash",
    "model_hash",
    "study_plan_hash",
    "comparison_spec_hash",
    "compiler_input_hash",
    "execution_matrix_hash",
    "release_hash",
    "data_manifest_hash",
    "train_view_hash",
    "global_step_plan_hash",
    "target_loss_tokens",
    "optimizer_step_token_cap",
    "optimizer_step_attention_cell_cap",
    "optimizer_step_sequence_cap",
    "learning_rate",
    "warmup_loss_tokens",
    "stable_loss_tokens",
    "decay_loss_tokens",
    "precision",
    "attention_backend",
    "attention_num_splits",
    "loss_backend",
    "compile_mode",
    "allow_tf32",
    "optimizer",
    "optimizer_backend",
    "adam_betas",
    "adam_eps",
    "weight_decay",
    "grad_clip_norm",
    "study_plan",
    "compiler_input",
    "execution_matrix",
    "condition",
    "model",
}


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def _jsonable(value: object) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _canonical_json(value: object) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _study(value: ConfigurationStudyPlanV2 | Mapping[str, Any]) -> ConfigurationStudyPlanV2:
    if isinstance(value, ConfigurationStudyPlanV2):
        return load_configuration_study_plan(value.to_dict())
    if isinstance(value, Mapping):
        return load_configuration_study_plan(value)
    raise ValueError("shape probe requires a hash-valid configuration study plan")


def _inputs(
    value: ConfigurationCompilerInputReceiptV1 | Mapping[str, Any],
) -> ConfigurationCompilerInputReceiptV1:
    if isinstance(value, ConfigurationCompilerInputReceiptV1):
        return load_configuration_compiler_input_receipt(value.to_dict())
    if isinstance(value, Mapping):
        return load_configuration_compiler_input_receipt(value)
    raise ValueError("shape probe requires a hash-valid compiler input receipt")


def _stage_spec(stage: str) -> Mapping[str, object]:
    spec = _STAGE_SPEC.get(stage)
    if spec is None:
        raise ValueError("shape probe stage is unsupported")
    return spec


def _condition(
    study: ConfigurationStudyPlanV2, *, stage: str, profile_id: str
) -> StudyCondition:
    spec = _stage_spec(stage)
    if profile_id not in spec["profiles"]:
        raise ValueError("shape probe stage/profile matrix is invalid")
    matches = tuple(
        condition
        for condition in study.comparison_spec.conditions
        if condition.phase == spec["phase"]
        and condition.seed == spec["seed"]
        and condition.profile_id == profile_id
    )
    if len(matches) != 1:
        raise ValueError("shape probe study lacks its exact widest/deepest condition")
    return matches[0]


def _validate_study_inputs(
    study: ConfigurationStudyPlanV2,
    inputs: ConfigurationCompilerInputReceiptV1,
    *,
    stage: str,
) -> None:
    spec = _stage_spec(stage)
    representative = _condition(
        study, stage=stage, profile_id=str(spec["profiles"][0])
    )
    phase_index = 0 if stage == "c2" else 1
    if (
        inputs.tokenizer_adoption_receipt_hash != study.tokenizer_adoption_receipt_hash
        or inputs.data_plan_hash != study.data_plan_hash
        or inputs.prefix_continuity_receipt_hash != study.prefix_continuity_receipt_hash
        or inputs.train_view_hash != study.train_view_hashes[phase_index]
        or inputs.validation_view_hash != study.validation_view_hash
        or inputs.train_view_loss_tokens != representative.target_loss_tokens
        or inputs.global_step_plan_loss_tokens != representative.target_loss_tokens
    ):
        raise ValueError("shape probe compiler input/study stage lineage mismatch")
    if (
        inputs.optimizer_step_token_cap,
        inputs.optimizer_step_attention_cell_cap,
        inputs.optimizer_step_sequence_cap,
    ) != _GLOBAL_CAPS:
        raise ValueError("shape probe global optimizer caps differ from the frozen study")
    if study.data_order_policy_hash != stable_json_hash(DATA_ORDER_POLICY_V2):
        raise ValueError("shape probe study data-order policy hash mismatch")
    if study.schedule_policy_hash != stable_json_hash(SCHEDULE_POLICY_V2):
        raise ValueError("shape probe study schedule policy hash mismatch")


@dataclass(frozen=True, slots=True)
class ConfigurationShapeProbeExecutionMatrixV1:
    """The exact three typed physical points for one profile attempt."""

    stage: str
    profile_id: str
    paths: tuple[ConfigurationShapeProbeExecutionPathReceiptV1, ...]

    def __post_init__(self) -> None:
        spec = _stage_spec(self.stage)
        if self.profile_id not in spec["profiles"]:
            raise ValueError("shape probe execution cap matrix profile is invalid")
        if (
            len(self.paths) != len(CONFIGURATION_SHAPE_CAPS)
            or tuple(path.probe_max_sequences for path in self.paths)
            != CONFIGURATION_SHAPE_CAPS
        ):
            raise ValueError("shape probe execution cap matrix requires exact caps 16, 20, 24")
        if any(
            not isinstance(path, ConfigurationShapeProbeExecutionPathReceiptV1)
            or path.stage != self.stage
            or path.probe_profile_id != self.profile_id
            for path in self.paths
        ):
            raise ValueError("shape probe execution cap matrix stage/profile lineage mismatch")
        baseline = {
            key: value
            for key, value in self.paths[0].to_dict().items()
            if key != "probe_max_sequences"
        }
        for path in self.paths[1:]:
            observed = {
                key: value
                for key, value in path.to_dict().items()
                if key != "probe_max_sequences"
            }
            if observed != baseline:
                raise ValueError("shape probe execution cap matrix shared lineage mismatch")
        if len({path.identity for path in self.paths}) != len(self.paths):
            raise ValueError("shape probe execution cap matrix point identities must be unique")

    @property
    def identity(self) -> str:
        return stable_json_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_shape_probe_execution_matrix_v1",
            "schema_version": 1,
            "execution_purpose": "diagnostic_shape_probe",
            "scientific_campaign_eligible": False,
            "scientific_result_eligible": False,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "tokens_serialized": False,
            "secrets_serialized": False,
            "stage": self.stage,
            "profile_id": self.profile_id,
            "point_receipts": [path.to_dict() for path in self.paths],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["matrix_hash"] = stable_json_hash(payload)
        return payload

    def point(self, max_sequences: int) -> ConfigurationShapeProbeExecutionPathReceiptV1:
        for path in self.paths:
            if path.probe_max_sequences == max_sequences:
                return path
        raise ValueError("shape probe cap is absent from the exact execution matrix")


def build_configuration_shape_probe_execution_matrix(
    *, paths: Sequence[ConfigurationShapeProbeExecutionPathReceiptV1 | Mapping[str, Any]]
) -> ConfigurationShapeProbeExecutionMatrixV1:
    parsed = tuple(
        path
        if isinstance(path, ConfigurationShapeProbeExecutionPathReceiptV1)
        else load_configuration_shape_probe_execution_path_receipt(path)
        for path in paths
    )
    if not parsed:
        raise ValueError("shape probe execution cap matrix requires exact caps 16, 20, 24")
    return ConfigurationShapeProbeExecutionMatrixV1(
        stage=parsed[0].stage,
        profile_id=parsed[0].probe_profile_id,
        paths=parsed,
    )


def load_configuration_shape_probe_execution_matrix(
    payload: Mapping[str, Any],
) -> ConfigurationShapeProbeExecutionMatrixV1:
    expected_fields = {
        "artifact_kind",
        "schema_version",
        "execution_purpose",
        "scientific_campaign_eligible",
        "scientific_result_eligible",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "raw_text_serialized",
        "raw_url_serialized",
        "tokens_serialized",
        "secrets_serialized",
        "stage",
        "profile_id",
        "point_receipts",
        "matrix_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("shape probe execution cap matrix fields are invalid")
    unsigned = {key: value for key, value in payload.items() if key != "matrix_hash"}
    if payload.get("matrix_hash") != stable_json_hash(unsigned):
        raise ValueError("shape probe execution cap matrix hash mismatch")
    if (
        payload.get("artifact_kind")
        != "twc_configuration_shape_probe_execution_matrix_v1"
        or payload.get("schema_version") != 1
        or payload.get("execution_purpose") != "diagnostic_shape_probe"
        or any(
            payload.get(field) is not False
            for field in (
                "scientific_campaign_eligible",
                "scientific_result_eligible",
                "production_unlock_eligible",
                "f1_unlock_eligible",
                "raw_text_serialized",
                "raw_url_serialized",
                "tokens_serialized",
                "secrets_serialized",
            )
        )
        or not isinstance(payload.get("point_receipts"), list)
    ):
        raise ValueError("shape probe execution cap matrix boundary is invalid")
    matrix = build_configuration_shape_probe_execution_matrix(
        paths=tuple(payload["point_receipts"])
    )
    if matrix.stage != payload.get("stage") or matrix.profile_id != payload.get("profile_id"):
        raise ValueError("shape probe execution cap matrix derived lineage mismatch")
    if _jsonable(matrix.to_dict()) != _jsonable(dict(payload)):
        raise ValueError("shape probe execution cap matrix is noncanonical")
    return matrix


def _matrices(
    values: Sequence[ConfigurationShapeProbeExecutionMatrixV1 | Mapping[str, Any]],
) -> tuple[ConfigurationShapeProbeExecutionMatrixV1, ...]:
    return tuple(
        value
        if isinstance(value, ConfigurationShapeProbeExecutionMatrixV1)
        else load_configuration_shape_probe_execution_matrix(value)
        for value in values
    )


def _diagnostic_data_order_hash(
    inputs: ConfigurationCompilerInputReceiptV1,
) -> str:
    return stable_json_hash(
        {
            "artifact_kind": "twc_configuration_shape_probe_data_order_v1",
            "compiler_input_hash": inputs.identity,
            "global_step_plan_hash": inputs.global_step_plan_hash,
            "window_steps": SHAPE_PROBE_WARMUP_STEPS + SHAPE_PROBE_MEASURED_STEPS,
            "paired_order_seed": 0,
        }
    )


def _diagnostic_policy_hash(stage: str) -> str:
    return stable_json_hash(
        {
            "artifact_kind": "twc_configuration_shape_probe_policy_v1",
            "stage": stage,
            "profiles": list(_stage_spec(stage)["profiles"]),
            "sequence_caps": list(CONFIGURATION_SHAPE_CAPS),
            "warmup_steps": SHAPE_PROBE_WARMUP_STEPS,
            "measured_steps": SHAPE_PROBE_MEASURED_STEPS,
            "throughput_knee_fraction": SHAPE_THROUGHPUT_KNEE_FRACTION,
            "minimum_memory_headroom_fraction": SHAPE_MINIMUM_MEMORY_HEADROOM_FRACTION,
            "maximum_input_wait_fraction": SHAPE_MAXIMUM_INPUT_WAIT_FRACTION,
        }
    )


def _validate_matrix_set(
    *,
    stage: str,
    study: ConfigurationStudyPlanV2,
    inputs: ConfigurationCompilerInputReceiptV1,
    release_hash: str,
    matrices: tuple[ConfigurationShapeProbeExecutionMatrixV1, ...],
) -> None:
    profiles = tuple(_stage_spec(stage)["profiles"])
    if tuple(matrix.profile_id for matrix in matrices) != profiles:
        raise ValueError("shape probe campaign requires exact widest/deepest matrices")
    common: tuple[object, ...] | None = None
    for matrix in matrices:
        point = matrix.paths[0]
        expected = (
            stage,
            study.identity,
            study.comparison_spec.identity,
            inputs.data_manifest_hash,
            inputs.train_view_hash,
            inputs.global_step_plan_hash,
            release_hash,
            point.environment_hash,
            point.placement_hash,
            point.device_identity_hash,
        )
        observed = (
            point.stage,
            point.study_plan_hash,
            point.comparison_spec_hash,
            point.data_manifest_hash,
            point.train_view_hash,
            point.global_step_plan_hash,
            point.release_hash,
            point.environment_hash,
            point.placement_hash,
            point.device_identity_hash,
        )
        if observed != expected:
            raise ValueError("shape probe execution matrix/study/input/release lineage mismatch")
        shared = observed[:7] + observed[7:]
        if common is None:
            common = shared
        elif shared != common:
            raise ValueError("shape probe widest/deepest matrices do not share execution lineage")


def build_configuration_shape_probe_campaign(
    *,
    stage: str,
    study_plan: ConfigurationStudyPlanV2 | Mapping[str, Any],
    compiler_input: ConfigurationCompilerInputReceiptV1 | Mapping[str, Any],
    release_hash: str,
    execution_matrices: Sequence[
        ConfigurationShapeProbeExecutionMatrixV1 | Mapping[str, Any]
    ],
    max_cost_per_attempt_usd: Decimal,
    reserve_usd: Decimal,
) -> CampaignPlan:
    """Build one bounded two-attempt diagnostic campaign for a scale."""

    study = _study(study_plan)
    inputs = _inputs(compiler_input)
    _validate_study_inputs(study, inputs, stage=stage)
    release = _require_hash(release_hash, "shape probe release_hash")
    matrices = _matrices(execution_matrices)
    _validate_matrix_set(
        stage=stage,
        study=study,
        inputs=inputs,
        release_hash=release,
        matrices=matrices,
    )
    if not isinstance(max_cost_per_attempt_usd, Decimal) or not isinstance(
        reserve_usd, Decimal
    ):
        raise ValueError("shape probe campaign costs must be exact Decimal values")
    profiles = tuple(_stage_spec(stage)["profiles"])
    attempts = tuple(
        AttemptPlan(
            attempt_id=_condition(study, stage=stage, profile_id=profile).condition_id,
            model_config_hash=str(
                compile_configuration_model_artifact(profile)["model_hash"]
            ),
            data_order_hash=_diagnostic_data_order_hash(inputs),
            execution_path_hash=matrix.identity,
            max_cost_usd=max_cost_per_attempt_usd,
        )
        for profile, matrix in zip(profiles, matrices, strict=True)
    )
    wave_cap = sum(
        (attempt.max_cost_usd for attempt in attempts), Decimal("0")
    )
    wave = WavePlan(
        wave_id=f"{stage}-shape-probe",
        purpose="configuration_shape_probe",
        cap_usd=wave_cap,
        attempts=attempts,
        prerequisite_wave_ids=(),
    )
    return CampaignPlan(
        campaign_id=f"twc-{stage}-configuration-shape-probe-v1",
        budget_cap_usd=wave_cap + reserve_usd,
        reserve_usd=reserve_usd,
        code_hash=release,
        evaluation_policy_hash=_diagnostic_policy_hash(stage),
        waves=(wave,),
    )


def _schedule(condition: StudyCondition) -> tuple[int, int, int]:
    target = condition.target_loss_tokens
    warmup = int(target * float(SCHEDULE_POLICY_V2["warmup_fraction"]))
    decay = int(target * float(SCHEDULE_POLICY_V2["tail_fraction"]))
    return warmup, target - warmup - decay, decay


def _bundle_derived(
    *,
    stage: str,
    profile_id: str,
    study: ConfigurationStudyPlanV2,
    inputs: ConfigurationCompilerInputReceiptV1,
    matrix: ConfigurationShapeProbeExecutionMatrixV1,
) -> dict[str, Any]:
    _validate_study_inputs(study, inputs, stage=stage)
    condition = _condition(study, stage=stage, profile_id=profile_id)
    if matrix.stage != stage or matrix.profile_id != profile_id:
        raise ValueError("shape probe bundle execution matrix stage/profile mismatch")
    model = compile_configuration_model_artifact(profile_id)
    warmup, stable, decay = _schedule(condition)
    return {
        "condition_id": condition.condition_id,
        "seed": condition.seed,
        "condition_hash": stable_json_hash(_jsonable(condition.to_dict())),
        "model_hash": model["model_hash"],
        "study_plan_hash": study.identity,
        "comparison_spec_hash": study.comparison_spec.identity,
        "compiler_input_hash": inputs.identity,
        "execution_matrix_hash": matrix.identity,
        "release_hash": matrix.paths[0].release_hash,
        "data_manifest_hash": inputs.data_manifest_hash,
        "train_view_hash": inputs.train_view_hash,
        "global_step_plan_hash": inputs.global_step_plan_hash,
        "target_loss_tokens": condition.target_loss_tokens,
        "optimizer_step_token_cap": inputs.optimizer_step_token_cap,
        "optimizer_step_attention_cell_cap": inputs.optimizer_step_attention_cell_cap,
        "optimizer_step_sequence_cap": inputs.optimizer_step_sequence_cap,
        "learning_rate": study.selected_learning_rate,
        "warmup_loss_tokens": warmup,
        "stable_loss_tokens": stable,
        "decay_loss_tokens": decay,
        **_RUNTIME,
        "study_plan": _jsonable(study.to_dict()),
        "compiler_input": _jsonable(inputs.to_dict()),
        "execution_matrix": _jsonable(matrix.to_dict()),
        "condition": _jsonable(condition.to_dict()),
        "model": _jsonable(model),
    }


@dataclass(frozen=True, slots=True)
class ConfigurationShapeProbeBundleV1:
    """One profile's diagnostic hot-path contract, with no scientific binding."""

    stage: str
    profile_id: str
    campaign_hash: str
    campaign_wave_id: str
    campaign_attempt_id: str
    _derived_json: str

    def __post_init__(self) -> None:
        _stage_spec(self.stage)
        _require_identifier(self.profile_id, "shape probe profile_id")
        _require_hash(self.campaign_hash, "shape probe campaign_hash")
        _require_identifier(self.campaign_wave_id, "shape probe campaign_wave_id")
        _require_identifier(self.campaign_attempt_id, "shape probe campaign_attempt_id")
        try:
            derived = json.loads(self._derived_json)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("shape probe bundle derived payload is invalid") from error
        if not isinstance(derived, dict) or self._derived_json != _canonical_json(derived):
            raise ValueError("shape probe bundle derived payload is noncanonical")
        study = load_configuration_study_plan(derived["study_plan"])
        inputs = load_configuration_compiler_input_receipt(derived["compiler_input"])
        matrix = load_configuration_shape_probe_execution_matrix(
            derived["execution_matrix"]
        )
        expected = _bundle_derived(
            stage=self.stage,
            profile_id=self.profile_id,
            study=study,
            inputs=inputs,
            matrix=matrix,
        )
        if _jsonable(derived) != _jsonable(expected):
            raise ValueError("shape probe bundle derived model/condition/runtime mismatch")

    def _derived(self) -> dict[str, Any]:
        value = json.loads(self._derived_json)
        assert isinstance(value, dict)
        return value

    def __getattr__(self, name: str) -> Any:
        derived = self._derived()
        if name in derived:
            return derived[name]
        raise AttributeError(name)

    @property
    def execution_matrix(self) -> ConfigurationShapeProbeExecutionMatrixV1:
        return load_configuration_shape_probe_execution_matrix(
            self._derived()["execution_matrix"]
        )

    @property
    def model(self) -> dict[str, Any]:
        return dict(self._derived()["model"])

    @property
    def identity(self) -> str:
        return stable_json_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_shape_probe_bundle_v1",
            "schema_version": 1,
            "execution_purpose": "diagnostic_shape_probe",
            "scientific_campaign_eligible": False,
            "scientific_identity_eligible": False,
            "scientific_result_eligible": False,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "accepted_manifest_unlock_eligible": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "tokens_serialized": False,
            "secrets_serialized": False,
            "stage": self.stage,
            "profile_id": self.profile_id,
            "campaign_hash": self.campaign_hash,
            "campaign_wave_id": self.campaign_wave_id,
            "campaign_attempt_id": self.campaign_attempt_id,
            **self._derived(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["bundle_hash"] = stable_json_hash(payload)
        return payload


def build_configuration_shape_probe_bundle(
    *,
    stage: str,
    profile_id: str,
    study_plan: ConfigurationStudyPlanV2 | Mapping[str, Any],
    compiler_input: ConfigurationCompilerInputReceiptV1 | Mapping[str, Any],
    campaign_plan: CampaignPlan,
    execution_matrix: ConfigurationShapeProbeExecutionMatrixV1 | Mapping[str, Any],
) -> ConfigurationShapeProbeBundleV1:
    if not isinstance(campaign_plan, CampaignPlan):
        raise ValueError("shape probe bundle requires a typed CampaignPlan")
    study = _study(study_plan)
    inputs = _inputs(compiler_input)
    matrix = (
        execution_matrix
        if isinstance(execution_matrix, ConfigurationShapeProbeExecutionMatrixV1)
        else load_configuration_shape_probe_execution_matrix(execution_matrix)
    )
    _validate_study_inputs(study, inputs, stage=stage)
    if matrix.stage != stage or matrix.profile_id != profile_id:
        raise ValueError("shape probe bundle execution matrix stage/profile mismatch")
    profiles = tuple(_stage_spec(stage)["profiles"])
    attempt = campaign_plan.wave(f"{stage}-shape-probe").attempt(
        _condition(study, stage=stage, profile_id=profile_id).condition_id
    )
    if attempt.execution_path_hash != matrix.identity:
        raise ValueError("shape probe bundle campaign attempt/matrix mismatch")
    # Validate the campaign's fixed non-matrix fields directly; the other
    # profile's matrix remains independently bound by its attempt hash.
    if (
        campaign_plan.campaign_id != f"twc-{stage}-configuration-shape-probe-v1"
        or campaign_plan.code_hash != matrix.paths[0].release_hash
        or campaign_plan.evaluation_policy_hash != _diagnostic_policy_hash(stage)
        or len(campaign_plan.waves) != 1
        or campaign_plan.waves[0].purpose != "configuration_shape_probe"
        or campaign_plan.waves[0].wave_id != f"{stage}-shape-probe"
        or tuple(item.attempt_id for item in campaign_plan.waves[0].attempts)
        != tuple(_condition(study, stage=stage, profile_id=value).condition_id for value in profiles)
        or campaign_plan.waves[0].cap_usd
        != sum((item.max_cost_usd for item in campaign_plan.waves[0].attempts), Decimal("0"))
        or campaign_plan.budget_cap_usd
        != campaign_plan.waves[0].cap_usd + campaign_plan.reserve_usd
    ):
        raise ValueError("shape probe bundle campaign is not canonical")
    expected_attempt = AttemptPlan(
        attempt_id=attempt.attempt_id,
        model_config_hash=str(compile_configuration_model_artifact(profile_id)["model_hash"]),
        data_order_hash=_diagnostic_data_order_hash(inputs),
        execution_path_hash=matrix.identity,
        max_cost_usd=attempt.max_cost_usd,
    )
    if attempt != expected_attempt:
        raise ValueError("shape probe bundle campaign attempt lineage mismatch")
    derived = _bundle_derived(
        stage=stage,
        profile_id=profile_id,
        study=study,
        inputs=inputs,
        matrix=matrix,
    )
    return ConfigurationShapeProbeBundleV1(
        stage=stage,
        profile_id=profile_id,
        campaign_hash=campaign_plan.identity,
        campaign_wave_id=campaign_plan.waves[0].wave_id,
        campaign_attempt_id=attempt.attempt_id,
        _derived_json=_canonical_json(derived),
    )


def load_configuration_shape_probe_bundle(
    payload: Mapping[str, Any], *, campaign_plan: CampaignPlan
) -> ConfigurationShapeProbeBundleV1:
    if not isinstance(payload, Mapping):
        raise ValueError("shape probe bundle must be an object")
    if not isinstance(campaign_plan, CampaignPlan):
        raise ValueError("shape probe bundle requires a typed CampaignPlan")
    expected_metadata = {
        "artifact_kind",
        "schema_version",
        "execution_purpose",
        "scientific_campaign_eligible",
        "scientific_identity_eligible",
        "scientific_result_eligible",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "accepted_manifest_unlock_eligible",
        "raw_text_serialized",
        "raw_url_serialized",
        "tokens_serialized",
        "secrets_serialized",
        "stage",
        "profile_id",
        "campaign_hash",
        "campaign_wave_id",
        "campaign_attempt_id",
        "bundle_hash",
    }
    if set(payload) != expected_metadata | _BUNDLE_DERIVED_FIELDS:
        raise ValueError("shape probe bundle fields are invalid")
    unsigned = {key: value for key, value in payload.items() if key != "bundle_hash"}
    if payload.get("bundle_hash") != stable_json_hash(unsigned):
        raise ValueError("shape probe bundle hash mismatch")
    if (
        payload.get("artifact_kind") != "twc_configuration_shape_probe_bundle_v1"
        or payload.get("schema_version") != 1
        or payload.get("execution_purpose") != "diagnostic_shape_probe"
        or any(
            payload.get(field) is not False
            for field in expected_metadata
            - {
                "artifact_kind",
                "schema_version",
                "execution_purpose",
                "stage",
                "profile_id",
                "campaign_hash",
                "campaign_wave_id",
                "campaign_attempt_id",
                "bundle_hash",
            }
        )
    ):
        raise ValueError("shape probe bundle safety boundary is invalid")
    if payload.get("campaign_hash") != campaign_plan.identity:
        raise ValueError("shape probe bundle campaign hash mismatch")
    bundle = build_configuration_shape_probe_bundle(
        stage=str(payload["stage"]),
        profile_id=str(payload["profile_id"]),
        study_plan=payload["study_plan"],
        compiler_input=payload["compiler_input"],
        campaign_plan=campaign_plan,
        execution_matrix=payload["execution_matrix"],
    )
    if _jsonable(bundle.to_dict()) != _jsonable(dict(payload)):
        raise ValueError("shape probe bundle cannot be reconstructed canonically")
    return bundle
