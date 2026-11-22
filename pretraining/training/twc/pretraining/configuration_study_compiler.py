"""Pure C2/C3 compiler for the v2 configuration study.

Scientific identity is compiled first from a sealed study plan and compact
no-raw input receipt.  Campaign and release provenance are then attached in an
execution binding, so retry or placement changes do not rewrite the scientific
contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from twc.pretraining.campaign import AttemptPlan, CampaignPlan, WavePlan
from twc.pretraining.configuration_assessment import (
    C3bRequirement,
    MultiArmComparisonReceipt,
    load_c3b_requirement,
)
from twc.pretraining.configuration_stage_authorization import (
    ConfigurationStageAuthorizationV1,
    load_configuration_stage_authorization,
)
from twc.pretraining.configuration_composite_data import (
    C1PackContinuityReceiptV1,
    C1TokenizerAdoptionReceiptV1,
    C2_TRAIN_LOSS_TOKENS,
    C3_TRAIN_LOSS_TOKENS,
    FROZEN_VALIDATION_LOSS_TOKENS,
    NestedCompositeViewsReceiptV1,
    load_c1_pack_continuity_receipt,
    load_c1_tokenizer_adoption_receipt,
    load_nested_composite_views_receipt,
)
from twc.pretraining.configuration_semantic_code import (
    SEMANTIC_CODE_PATHS_V2,
    build_configuration_semantic_code_receipt,
    validate_configuration_semantic_code_receipt,
)
from twc.pretraining.configuration_study_lineage import (
    C1ScreenAdoptionReceiptV1,
    load_c1_screen_adoption_receipt,
    load_finite_sequence_view,
)
from twc.pretraining.configuration_study import (
    ConfigurationStudyPlanV2,
    ExperimentContractV2,
    StudyCondition,
    V2_MODEL_PROFILES,
    build_default_c2_c3_comparison_spec,
    load_configuration_study_plan,
    load_experiment_contract_v2,
)
from twc.pretraining.configuration_study_execution import (
    ConfigurationExecutionPathReceiptV1,
    ExperimentExecutionBindingV1,
    ExperimentTrainingContractV2,
    load_configuration_execution_path_receipt,
    load_experiment_execution_binding,
    load_experiment_training_contract_v2,
    validate_configuration_execution_path_for_training,
)
from twc.pretraining.training.experiments import (
    EvaluationManifest,
    EvaluationTaskImplementationReceipt,
    bind_evaluation_task,
    load_evaluation_manifest,
    load_evaluation_task_implementation_receipt,
)
from twc.pretraining.data_pipeline.contracts import hash_without
from twc.pretraining.data_pipeline.sequence_pack import (
    validate_global_step_plan,
    validate_isolated_sequence_pack,
)
from twc.pretraining.training.model import DecoderConfig, estimate_parameter_count
from twc.repro.hashing import stable_json_hash
from twc.repro.public_artifacts import validate_public_no_raw_payload


VALIDATION_LOSS_TOKENS = 8_388_608
CONFIGURATION_GLOBAL_STEP_CAPS_V2 = (262_144, 1_073_741_824, 4_096)
DATA_ORDER_POLICY_V2 = {
    "policy": "finite_pack_canonical_order_with_paired_seed_v2",
    "repeat_allowed": False,
    "reshuffle_within_pair": False,
    "paired_order_seed": 0,
}
SCHEDULE_POLICY_V2 = {
    "policy": "token_linear_warmup_stable_cosine_tail_v2",
    "warmup_fraction": 0.02,
    "tail_fraction": 0.10,
    "evaluation_policy": "interval_since_last_eval_plus_schedule_boundaries_v1",
}
VALIDATION_NORMALIZATION_V1 = {
    "normalization": "sum_batch_nll_over_sum_loss_tokens_v1",
    "padding_included": False,
    "sequence_boundaries_preserved": True,
}
def _require_hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_money(value: object, label: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise ValueError(f"{label} must be a positive exact Decimal")
    if not value.is_finite() or value <= 0 or value.as_tuple().exponent < -6:
        raise ValueError(f"{label} must be positive with at most six decimal places")
    return value


@dataclass(frozen=True, slots=True)
class ConfigurationCompilerInputReceiptV1:
    """Hash-only data/view/plan boundary consumed by the paid-run compiler."""

    data_manifest_hash: str
    tokenizer_hash: str
    tokenizer_adoption_receipt_hash: str
    data_plan_hash: str
    prefix_continuity_receipt_hash: str
    train_view_hash: str
    train_view_pack_manifest_hash: str
    train_view_loss_tokens: int
    validation_view_hash: str
    validation_view_pack_manifest_hash: str
    validation_view_loss_tokens: int
    global_step_plan_hash: str
    global_step_plan_view_hash: str
    global_step_plan_pack_manifest_hash: str
    global_step_plan_loss_tokens: int
    optimizer_step_token_cap: int
    optimizer_step_attention_cell_cap: int
    optimizer_step_sequence_cap: int
    finite_views: bool
    repeat_allowed: bool
    raw_text_serialized: bool
    raw_url_serialized: bool
    tokens_serialized: bool

    def __post_init__(self) -> None:
        for field in (
            "data_manifest_hash",
            "tokenizer_hash",
            "tokenizer_adoption_receipt_hash",
            "data_plan_hash",
            "prefix_continuity_receipt_hash",
            "train_view_hash",
            "train_view_pack_manifest_hash",
            "validation_view_hash",
            "validation_view_pack_manifest_hash",
            "global_step_plan_hash",
            "global_step_plan_view_hash",
            "global_step_plan_pack_manifest_hash",
        ):
            _require_hash(getattr(self, field), field)
        for field in (
            "train_view_loss_tokens",
            "validation_view_loss_tokens",
            "global_step_plan_loss_tokens",
            "optimizer_step_token_cap",
            "optimizer_step_attention_cell_cap",
            "optimizer_step_sequence_cap",
        ):
            _positive_int(getattr(self, field), field)
        if self.validation_view_loss_tokens != VALIDATION_LOSS_TOKENS:
            raise ValueError("validation view must contain exactly 8,388,608 loss tokens")
        if (
            self.train_view_pack_manifest_hash != self.data_manifest_hash
            or self.validation_view_pack_manifest_hash != self.data_manifest_hash
            or self.global_step_plan_pack_manifest_hash != self.data_manifest_hash
            or self.global_step_plan_view_hash != self.train_view_hash
            or self.global_step_plan_loss_tokens != self.train_view_loss_tokens
        ):
            raise ValueError("compiler input receipt lineage is inconsistent")
        if (
            self.finite_views is not True
            or self.repeat_allowed is not False
            or self.raw_text_serialized is not False
            or self.raw_url_serialized is not False
            or self.tokens_serialized is not False
        ):
            raise ValueError("compiler inputs must be finite, no-repeat, and no-raw")

    @property
    def identity(self) -> str:
        return stable_json_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_compiler_input_receipt_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "accepted_manifest_unlock_eligible": False,
            **asdict(self),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_hash"] = stable_json_hash(payload)
        return payload


def load_configuration_compiler_input_receipt(
    payload: Mapping[str, Any],
) -> ConfigurationCompilerInputReceiptV1:
    fields = set(ConfigurationCompilerInputReceiptV1.__dataclass_fields__)
    metadata = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "accepted_manifest_unlock_eligible",
        "receipt_hash",
    }
    if set(payload) != fields | metadata:
        raise ValueError("compiler input receipt fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_configuration_compiler_input_receipt_v1"
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
        or payload.get("accepted_manifest_unlock_eligible") is not False
    ):
        raise ValueError("compiler input receipt boundary is invalid")
    expected = stable_json_hash({key: value for key, value in payload.items() if key != "receipt_hash"})
    if payload.get("receipt_hash") != expected:
        raise ValueError("compiler input receipt hash mismatch")
    return ConfigurationCompilerInputReceiptV1(**{field: payload[field] for field in fields})


def _c1_adoption(
    value: C1ScreenAdoptionReceiptV1 | Mapping[str, Any],
) -> C1ScreenAdoptionReceiptV1:
    if isinstance(value, C1ScreenAdoptionReceiptV1):
        return load_c1_screen_adoption_receipt(value.to_dict())
    if isinstance(value, Mapping):
        return load_c1_screen_adoption_receipt(value)
    raise ValueError("study builder requires a hash-valid C1 screen adoption receipt")


def _tokenizer_adoption(
    value: C1TokenizerAdoptionReceiptV1 | Mapping[str, Any],
) -> C1TokenizerAdoptionReceiptV1:
    if isinstance(value, C1TokenizerAdoptionReceiptV1):
        return load_c1_tokenizer_adoption_receipt(value.to_dict())
    if isinstance(value, Mapping):
        return load_c1_tokenizer_adoption_receipt(value)
    raise ValueError("study builder requires a hash-valid tokenizer adoption receipt")


def _continuity(
    value: C1PackContinuityReceiptV1 | Mapping[str, Any],
) -> C1PackContinuityReceiptV1:
    if isinstance(value, C1PackContinuityReceiptV1):
        return load_c1_pack_continuity_receipt(value.to_dict())
    if isinstance(value, Mapping):
        return load_c1_pack_continuity_receipt(value)
    raise ValueError("study builder requires a hash-valid pack-continuity receipt")


def _nested_views(
    value: NestedCompositeViewsReceiptV1 | Mapping[str, Any],
) -> NestedCompositeViewsReceiptV1:
    if isinstance(value, NestedCompositeViewsReceiptV1):
        return load_nested_composite_views_receipt(value.to_dict())
    if isinstance(value, Mapping):
        return load_nested_composite_views_receipt(value)
    raise ValueError("study builder requires a hash-valid nested-view receipt")


def _evaluation_manifest(
    value: EvaluationManifest | Mapping[str, Any],
) -> EvaluationManifest:
    if isinstance(value, EvaluationManifest):
        return load_evaluation_manifest(value.to_dict())
    if isinstance(value, Mapping):
        return load_evaluation_manifest(value)
    raise ValueError("study builder requires a hash-valid evaluation manifest")


def _validate_study_lineage(
    *,
    tokenizer: C1TokenizerAdoptionReceiptV1,
    continuity: C1PackContinuityReceiptV1,
    nested: NestedCompositeViewsReceiptV1,
    c2: Mapping[str, Any],
    c3: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    if (
        continuity.tokenizer_adoption_receipt_hash != tokenizer.receipt_hash
        or continuity.tokenizer_manifest_hash != tokenizer.tokenizer_manifest_hash
    ):
        raise ValueError("pack continuity differs from the tokenizer adoption receipt")
    if (
        nested.continuity_receipt_hash != continuity.receipt_hash
        or nested.composite_pack_manifest_hash
        != continuity.composite_pack_manifest_hash
    ):
        raise ValueError("nested views differ from the pack-continuity receipt")
    expected = (
        (
            c2,
            "train",
            C2_TRAIN_LOSS_TOKENS,
            nested.train_view_hashes[1],
            "nested C2",
        ),
        (
            c3,
            "train",
            C3_TRAIN_LOSS_TOKENS,
            nested.train_view_hashes[2],
            "nested C3",
        ),
        (
            validation,
            "validation",
            FROZEN_VALIDATION_LOSS_TOKENS,
            nested.validation_view_hash,
            "nested validation",
        ),
    )
    for view, split, tokens, view_hash, label in expected:
        if (
            view["pack_manifest_hash"] != continuity.composite_pack_manifest_hash
            or view["split"] != split
            or view["loss_token_count"] != tokens
            or view["view_hash"] != view_hash
        ):
            raise ValueError(f"{label} view lineage is inconsistent")


def build_configuration_study_plan_v2(
    *,
    c1_adoption: C1ScreenAdoptionReceiptV1 | Mapping[str, Any],
    tokenizer_adoption: C1TokenizerAdoptionReceiptV1 | Mapping[str, Any],
    continuity: C1PackContinuityReceiptV1 | Mapping[str, Any],
    nested_views: NestedCompositeViewsReceiptV1 | Mapping[str, Any],
    c2_train_view: Mapping[str, Any],
    c3_train_view: Mapping[str, Any],
    validation_view: Mapping[str, Any],
    evaluation_manifest: EvaluationManifest | Mapping[str, Any],
    semantic_code_receipt: Mapping[str, Any],
) -> ConfigurationStudyPlanV2:
    """Build the study identity only from strict, mutually bound artifacts."""

    adoption = _c1_adoption(c1_adoption)
    tokenizer = _tokenizer_adoption(tokenizer_adoption)
    continuity_receipt = _continuity(continuity)
    nested = _nested_views(nested_views)
    c2 = load_finite_sequence_view(c2_train_view)
    c3 = load_finite_sequence_view(c3_train_view)
    validation = load_finite_sequence_view(validation_view)
    evaluator = _evaluation_manifest(evaluation_manifest)
    semantic = validate_configuration_semantic_code_receipt(semantic_code_receipt)
    _validate_study_lineage(
        tokenizer=tokenizer,
        continuity=continuity_receipt,
        nested=nested,
        c2=c2,
        c3=c3,
        validation=validation,
    )

    tasks = {task.task_id: task for task in evaluator.tasks}
    validation_task = tasks.get("validation_nll")
    if (
        validation_task is None
        or validation_task.implementation_status != "implemented_verified"
        or validation_task.target_proximity != "neutral"
        or validation_task.selection_role != "selection_primary"
        or validation_task.may_rank_models is not True
        or validation_task.item_set_hash != validation["view_hash"]
        or validation_task.scorer_code_hash != semantic["semantic_code_hash"]
    ):
        raise ValueError("evaluation manifest is not bound to validation and semantic code")
    if any(
        task.implementation_status != "planned" or task.may_rank_models is not False
        for task in evaluator.tasks
        if task.target_proximity != "neutral"
    ):
        raise ValueError("target-adjacent diagnostics must remain sealed")

    return ConfigurationStudyPlanV2(
        study_id="twc-c2-c3-configuration-v2",
        c1_adoption_receipt_hash=adoption.receipt_hash,
        comparison_spec=build_default_c2_c3_comparison_spec(),
        tokenizer_adoption_receipt_hash=tokenizer.receipt_hash,
        data_plan_hash=nested.receipt_hash,
        prefix_continuity_receipt_hash=continuity_receipt.receipt_hash,
        train_view_hashes=(str(c2["view_hash"]), str(c3["view_hash"])),
        validation_view_hash=str(validation["view_hash"]),
        evaluation_manifest_hash=evaluator.identity,
        data_order_policy_hash=stable_json_hash(DATA_ORDER_POLICY_V2),
        schedule_policy_hash=stable_json_hash(SCHEDULE_POLICY_V2),
        semantic_code_hash=str(semantic["semantic_code_hash"]),
        selected_learning_rate=adoption.selected_learning_rate,
        target_diagnostics_remained_sealed=True,
    )


_GLOBAL_STEP_PLAN_FIELDS = {
    "artifact_kind",
    "schema_version",
    "view_hash",
    "pack_manifest_hash",
    "max_loss_tokens",
    "max_attention_cells",
    "max_sequences",
    "step_count",
    "loss_token_count",
    "steps",
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "plan_hash",
}


def _load_global_step_plan_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _GLOBAL_STEP_PLAN_FIELDS:
        raise ValueError("global step plan fields are invalid")
    plan = dict(payload)
    if (
        plan.get("artifact_kind") != "twc_global_step_plan_v1"
        or plan.get("schema_version") != 1
        or plan.get("plan_hash") != hash_without(plan, "plan_hash")
        or not isinstance(plan.get("steps"), list)
        or type(plan.get("step_count")) is not int
        or plan["step_count"] != len(plan["steps"])
        or plan.get("raw_text_serialized") is not False
        or plan.get("raw_url_serialized") is not False
        or plan.get("token_serialized") is not False
    ):
        raise ValueError("global step plan boundary is invalid")
    for field in (
        "max_loss_tokens",
        "max_attention_cells",
        "max_sequences",
        "loss_token_count",
    ):
        _positive_int(plan.get(field), f"global step plan {field}")
    _require_hash(plan.get("view_hash"), "global step plan view_hash")
    _require_hash(plan.get("pack_manifest_hash"), "global step plan pack_manifest_hash")
    validate_public_no_raw_payload(plan)
    return plan


def build_configuration_compiler_input_receipt_v1(
    *,
    study_plan: ConfigurationStudyPlanV2 | Mapping[str, Any],
    tokenizer_adoption: C1TokenizerAdoptionReceiptV1 | Mapping[str, Any],
    continuity: C1PackContinuityReceiptV1 | Mapping[str, Any],
    nested_views: NestedCompositeViewsReceiptV1 | Mapping[str, Any],
    pack_manifest: Mapping[str, Any],
    pack_root: Path,
    train_view: Mapping[str, Any],
    validation_view: Mapping[str, Any],
    global_step_plan: Mapping[str, Any],
) -> ConfigurationCompilerInputReceiptV1:
    """Derive a paid-run compiler receipt from validated pack artifacts."""

    study = _study(study_plan)
    tokenizer = _tokenizer_adoption(tokenizer_adoption)
    continuity_receipt = _continuity(continuity)
    nested = _nested_views(nested_views)
    train = load_finite_sequence_view(train_view)
    validation = load_finite_sequence_view(validation_view)
    plan = _load_global_step_plan_summary(global_step_plan)
    if (
        plan["max_loss_tokens"],
        plan["max_attention_cells"],
        plan["max_sequences"],
    ) != CONFIGURATION_GLOBAL_STEP_CAPS_V2:
        raise ValueError("global step plan caps differ from the frozen configuration study")
    validate_isolated_sequence_pack(pack_manifest, pack_root)
    validate_global_step_plan(
        plan,
        view=train,
        pack_manifest=pack_manifest,
        pack_root=pack_root,
    )
    pack_hash = _require_hash(
        pack_manifest.get("manifest_hash"), "pack manifest hash"
    )
    if (
        pack_manifest.get("artifact_kind")
        != "twc_isolated_sequence_pack_manifest_v1"
        or pack_manifest.get("schema_version") != 1
        or pack_manifest.get("tokenizer_manifest_hash")
        != tokenizer.tokenizer_manifest_hash
        or pack_manifest.get("inline_tokens_present") is not False
        or pack_manifest.get("raw_text_serialized") is not False
        or pack_manifest.get("raw_url_serialized") is not False
        or pack_manifest.get("token_serialized") is not False
    ):
        raise ValueError("pack manifest differs from the adopted no-raw tokenizer lineage")
    if (
        continuity_receipt.tokenizer_adoption_receipt_hash != tokenizer.receipt_hash
        or continuity_receipt.tokenizer_manifest_hash
        != tokenizer.tokenizer_manifest_hash
        or nested.continuity_receipt_hash != continuity_receipt.receipt_hash
        or nested.composite_pack_manifest_hash != pack_hash
    ):
        raise ValueError("compiler input adoption and continuity lineage is inconsistent")
    train_slots = {
        nested.train_view_hashes[1]: C2_TRAIN_LOSS_TOKENS,
        nested.train_view_hashes[2]: C3_TRAIN_LOSS_TOKENS,
    }
    if (
        train.get("split") != "train"
        or train.get("pack_manifest_hash") != pack_hash
        or train.get("view_hash") not in train_slots
        or train.get("loss_token_count") != train_slots.get(train.get("view_hash"))
        or validation.get("split") != "validation"
        or validation.get("pack_manifest_hash") != pack_hash
        or validation.get("view_hash") != nested.validation_view_hash
        or validation.get("loss_token_count") != FROZEN_VALIDATION_LOSS_TOKENS
    ):
        raise ValueError("compiler input finite views differ from the nested-view receipt")
    if (
        tokenizer.receipt_hash != study.tokenizer_adoption_receipt_hash
        or continuity_receipt.receipt_hash != study.prefix_continuity_receipt_hash
        or nested.receipt_hash != study.data_plan_hash
        or pack_hash != continuity_receipt.composite_pack_manifest_hash
        or train["view_hash"] not in study.train_view_hashes
        or validation["view_hash"] != study.validation_view_hash
        or plan["view_hash"] != train["view_hash"]
        or plan["pack_manifest_hash"] != pack_hash
        or plan["loss_token_count"] != train["loss_token_count"]
    ):
        raise ValueError("compiler input artifacts do not match the sealed study lineage")

    return ConfigurationCompilerInputReceiptV1(
        data_manifest_hash=pack_hash,
        tokenizer_hash=tokenizer.tokenizer_manifest_hash,
        tokenizer_adoption_receipt_hash=tokenizer.receipt_hash,
        data_plan_hash=nested.receipt_hash,
        prefix_continuity_receipt_hash=continuity_receipt.receipt_hash,
        train_view_hash=str(train["view_hash"]),
        train_view_pack_manifest_hash=str(train["pack_manifest_hash"]),
        train_view_loss_tokens=int(train["loss_token_count"]),
        validation_view_hash=str(validation["view_hash"]),
        validation_view_pack_manifest_hash=str(validation["pack_manifest_hash"]),
        validation_view_loss_tokens=int(validation["loss_token_count"]),
        global_step_plan_hash=str(plan["plan_hash"]),
        global_step_plan_view_hash=str(plan["view_hash"]),
        global_step_plan_pack_manifest_hash=str(plan["pack_manifest_hash"]),
        global_step_plan_loss_tokens=int(plan["loss_token_count"]),
        optimizer_step_token_cap=int(plan["max_loss_tokens"]),
        optimizer_step_attention_cell_cap=int(plan["max_attention_cells"]),
        optimizer_step_sequence_cap=int(plan["max_sequences"]),
        finite_views=True,
        repeat_allowed=False,
        raw_text_serialized=False,
        raw_url_serialized=False,
        tokens_serialized=False,
    )


def _study(value: ConfigurationStudyPlanV2 | Mapping[str, Any]) -> ConfigurationStudyPlanV2:
    if isinstance(value, ConfigurationStudyPlanV2):
        return load_configuration_study_plan(value.to_dict())
    if isinstance(value, Mapping):
        return load_configuration_study_plan(value)
    raise ValueError("configuration compiler requires a hash-valid study plan")


def _inputs(
    value: ConfigurationCompilerInputReceiptV1 | Mapping[str, Any],
) -> ConfigurationCompilerInputReceiptV1:
    if isinstance(value, ConfigurationCompilerInputReceiptV1):
        return load_configuration_compiler_input_receipt(value.to_dict())
    if isinstance(value, Mapping):
        return load_configuration_compiler_input_receipt(value)
    raise ValueError("configuration compiler requires a hash-valid input receipt")


def bind_configuration_validation_nll(
    *,
    planned_evaluation_manifest: EvaluationManifest,
    validation_view_hash: str,
    semantic_code_hash: str,
    provenance_evidence_hash: str,
    contamination_review_evidence_hash: str,
    implementation_test_evidence_hash: str,
) -> tuple[EvaluationManifest, EvaluationTaskImplementationReceipt]:
    """Bind only the neutral validation-NLL scorer to the exact finite view."""

    if not isinstance(planned_evaluation_manifest, EvaluationManifest):
        raise ValueError("planned evaluator manifest is required")
    for value, label in (
        (validation_view_hash, "validation_view_hash"),
        (semantic_code_hash, "semantic_code_hash"),
        (provenance_evidence_hash, "provenance_evidence_hash"),
        (contamination_review_evidence_hash, "contamination_review_evidence_hash"),
        (implementation_test_evidence_hash, "implementation_test_evidence_hash"),
    ):
        _require_hash(value, label)
    tasks = {task.task_id: task for task in planned_evaluation_manifest.tasks}
    task = tasks.get("validation_nll")
    if (
        task is None
        or task.implementation_status != "planned"
        or task.target_proximity != "neutral"
        or task.may_rank_models is not True
    ):
        raise ValueError("planned neutral validation_nll ranking task is required")
    receipt = EvaluationTaskImplementationReceipt(
        task_id="validation_nll",
        planned_task_hash=task.identity,
        release_hash=semantic_code_hash,
        item_set_hash=validation_view_hash,
        scorer_code_hash=semantic_code_hash,
        normalization_hash=stable_json_hash(VALIDATION_NORMALIZATION_V1),
        provenance_evidence_hash=provenance_evidence_hash,
        contamination_review_evidence_hash=contamination_review_evidence_hash,
        implementation_test_evidence_hash=implementation_test_evidence_hash,
    )
    return bind_evaluation_task(planned_evaluation_manifest, receipt), receipt


def _compile_configuration_profile(profile_id: str) -> dict[str, Any]:
    profile = V2_MODEL_PROFILES[profile_id]
    config = DecoderConfig(
        vocab_size=profile.vocab_size,
        context_length=profile.context_length,
        hidden_size=profile.hidden_size,
        num_layers=profile.num_layers,
        num_attention_heads=profile.num_attention_heads,
        num_key_value_heads=profile.num_key_value_heads,
        intermediate_size=profile.intermediate_size,
        dropout=profile.dropout,
    )
    if estimate_parameter_count(config) != profile.parameter_count:
        raise ValueError("v2 profile and executable decoder parameter counts differ")
    payload: dict[str, Any] = {
        "artifact_kind": "twc_base_model_spec_v1",
        "schema_version": 1,
        "expected_parameter_count": profile.parameter_count,
        **asdict(config),
    }
    payload["model_hash"] = stable_json_hash(payload)
    return payload


def compile_configuration_model_artifact(profile_id: str) -> dict[str, Any]:
    """Compile a reviewed trainable C2/C3 shape into the decoder schema."""

    profile = V2_MODEL_PROFILES.get(profile_id)
    if profile is None or profile.train_phase is None:
        raise ValueError("profile_id is not a trainable v2 configuration profile")
    return _compile_configuration_profile(profile_id)


def compile_configuration_endpoint_model_artifact(profile_id: str) -> dict[str, Any]:
    """Compile one reviewed 100M endpoint without making it a C2/C3 arm."""

    profile = V2_MODEL_PROFILES.get(profile_id)
    if (
        profile is None
        or profile.train_phase is not None
        or profile.scale_class != "endpoint"
    ):
        raise ValueError("profile_id is not a frozen v2 endpoint profile")
    return _compile_configuration_profile(profile_id)


def _condition_stage(condition: StudyCondition) -> str:
    if condition.phase == "m_transfer" and condition.seed == 17:
        return "c2"
    if condition.phase == "l_confirmation" and condition.seed == 29:
        return "c3a"
    if condition.phase == "l_confirmation" and condition.seed == 43:
        return "c3b"
    raise ValueError("condition is outside the approved C2/C3 stages")


def _selected_conditions(
    study: ConfigurationStudyPlanV2, condition_ids: tuple[str, ...]
) -> tuple[str, tuple[StudyCondition, ...]]:
    if not condition_ids or len(condition_ids) != len(set(condition_ids)):
        raise ValueError("campaign condition IDs must be non-empty and unique")
    requested = set(condition_ids)
    unknown = requested - {condition.condition_id for condition in study.comparison_spec.conditions}
    if unknown:
        raise ValueError(f"campaign contains unknown condition IDs: {sorted(unknown)}")
    selected = tuple(
        condition
        for condition in study.comparison_spec.conditions
        if condition.condition_id in requested
    )
    stages = {_condition_stage(condition) for condition in selected}
    if len(stages) != 1:
        raise ValueError("one campaign may open only one C2/C3 stage")
    stage = stages.pop()
    all_stage = tuple(
        condition
        for condition in study.comparison_spec.conditions
        if _condition_stage(condition) == stage
    )
    if stage in {"c2", "c3a"} and selected != all_stage:
        raise ValueError("C2 and C3a require the full three-family stage")
    if stage == "c3b":
        families = {condition.family_id for condition in selected}
        if "r7_scaled_control" not in families or len(selected) < 2:
            raise ValueError("C3b requires the control and at least one challenger")
    return stage, selected


def _validate_c3b_campaign_requirement(
    *,
    study: ConfigurationStudyPlanV2,
    stage: str,
    condition_ids: tuple[str, ...],
    c3b_requirement: C3bRequirement | Mapping[str, Any] | None,
    c3a_comparison: MultiArmComparisonReceipt | None,
) -> None:
    """Close the conditional C3a -> C3b branch before campaign creation."""

    if stage != "c3b":
        if c3b_requirement is not None or c3a_comparison is not None:
            raise ValueError("C3b requirement inputs are only valid for C3b")
        return
    if c3b_requirement is None or c3a_comparison is None:
        raise ValueError("C3b requirement is required with its C3a comparison")
    if not isinstance(c3a_comparison, MultiArmComparisonReceipt):
        raise ValueError("C3b requires a validated C3a comparison")
    payload = (
        c3b_requirement.to_dict()
        if isinstance(c3b_requirement, C3bRequirement)
        else c3b_requirement
    )
    if not isinstance(payload, Mapping):
        raise ValueError("C3b requirement must be a validated artifact")
    requirement = load_c3b_requirement(
        payload,
        source_comparison=c3a_comparison,
    )
    if requirement.branch != "required":
        raise ValueError("C3b campaign requires the required branch, not not_required")
    source_contracts = tuple(
        result.scientific_contract for result in c3a_comparison.results
    )
    if any(
        contract.study_plan_hash != study.identity
        or contract.comparison_spec_hash != study.comparison_spec.identity
        for contract in source_contracts
    ):
        raise ValueError("C3b source comparison belongs to the wrong study or comparison")
    required_ids = tuple(
        condition.condition_id for condition in requirement.required_conditions
    )
    if condition_ids != required_ids:
        raise ValueError("C3b campaign must use the exact ordered required condition IDs")
    try:
        study_conditions = tuple(
            study.comparison_spec.condition(condition_id)
            for condition_id in required_ids
        )
    except KeyError as error:  # pragma: no cover - canonical requirement rejects first.
        raise ValueError("C3b requirement condition is absent from the study") from error
    if requirement.required_conditions != study_conditions:
        raise ValueError("C3b requirement conditions do not match the study comparison")


def _validate_study_inputs(
    study: ConfigurationStudyPlanV2,
    condition: StudyCondition,
    inputs: ConfigurationCompilerInputReceiptV1,
) -> None:
    if inputs.train_view_loss_tokens != condition.target_loss_tokens:
        raise ValueError("train view does not match the exact condition target")
    if inputs.global_step_plan_loss_tokens != condition.target_loss_tokens:
        raise ValueError("global step plan does not match the exact condition target")
    phase_index = 0 if condition.phase == "m_transfer" else 1
    expected = {
        "tokenizer_adoption_receipt_hash": study.tokenizer_adoption_receipt_hash,
        "data_plan_hash": study.data_plan_hash,
        "prefix_continuity_receipt_hash": study.prefix_continuity_receipt_hash,
        "train_view_hash": study.train_view_hashes[phase_index],
        "validation_view_hash": study.validation_view_hash,
    }
    for field, value in expected.items():
        if getattr(inputs, field) != value:
            raise ValueError(f"compiler input/study lineage mismatch: {field}")
    if study.data_order_policy_hash != stable_json_hash(DATA_ORDER_POLICY_V2):
        raise ValueError("study data-order policy hash mismatch")
    if study.schedule_policy_hash != stable_json_hash(SCHEDULE_POLICY_V2):
        raise ValueError("study schedule policy hash mismatch")


def _data_order_hash(inputs: ConfigurationCompilerInputReceiptV1) -> str:
    return stable_json_hash(
        {
            "policy_hash": stable_json_hash(DATA_ORDER_POLICY_V2),
            "train_view_hash": inputs.train_view_hash,
            "global_step_plan_hash": inputs.global_step_plan_hash,
            "optimizer_step_token_cap": inputs.optimizer_step_token_cap,
            "optimizer_step_attention_cell_cap": inputs.optimizer_step_attention_cell_cap,
            "optimizer_step_sequence_cap": inputs.optimizer_step_sequence_cap,
            "paired_order_seed": 0,
        }
    )


def _schedule(condition: StudyCondition) -> tuple[int, int, int, str]:
    target = condition.target_loss_tokens
    warmup = int(target * SCHEDULE_POLICY_V2["warmup_fraction"])
    decay = int(target * SCHEDULE_POLICY_V2["tail_fraction"])
    stable = target - warmup - decay
    schedule_hash = stable_json_hash(
        {
            "policy_hash": stable_json_hash(SCHEDULE_POLICY_V2),
            "target_loss_tokens": target,
            "warmup_loss_tokens": warmup,
            "stable_loss_tokens": stable,
            "decay_loss_tokens": decay,
            "evaluation_interval_loss_tokens": condition.evaluation_interval_loss_tokens,
            "required_exact_evaluation_loss_tokens": condition.required_exact_evaluation_loss_tokens,
        }
    )
    return warmup, stable, decay, schedule_hash


def _bound_evaluation(
    *,
    study: ConfigurationStudyPlanV2,
    inputs: ConfigurationCompilerInputReceiptV1,
    planned_evaluation_manifest: EvaluationManifest,
    provenance_evidence_hash: str,
    contamination_review_evidence_hash: str,
    implementation_test_evidence_hash: str,
) -> tuple[EvaluationManifest, EvaluationTaskImplementationReceipt]:
    manifest, receipt = bind_configuration_validation_nll(
        planned_evaluation_manifest=planned_evaluation_manifest,
        validation_view_hash=inputs.validation_view_hash,
        semantic_code_hash=study.semantic_code_hash,
        provenance_evidence_hash=provenance_evidence_hash,
        contamination_review_evidence_hash=contamination_review_evidence_hash,
        implementation_test_evidence_hash=implementation_test_evidence_hash,
    )
    if manifest.identity != study.evaluation_manifest_hash:
        raise ValueError("bound evaluation manifest does not match the study plan")
    return manifest, receipt


def _validated_execution_path(
    *,
    execution_path_receipt: ConfigurationExecutionPathReceiptV1 | Mapping[str, Any],
    stage: str,
    study: ConfigurationStudyPlanV2,
    inputs: ConfigurationCompilerInputReceiptV1,
    release_hash: str,
    conditions: tuple[StudyCondition, ...],
) -> ConfigurationExecutionPathReceiptV1:
    if isinstance(execution_path_receipt, ConfigurationExecutionPathReceiptV1):
        receipt = execution_path_receipt
    elif isinstance(execution_path_receipt, Mapping):
        receipt = load_configuration_execution_path_receipt(execution_path_receipt)
    else:
        raise ValueError(
            "configuration compiler requires a canonical execution path receipt"
        )
    expected_shape_stage = "c2" if stage == "c2" else "c3"
    expected = {
        "stage": expected_shape_stage,
        "study_plan_hash": study.identity,
        "comparison_spec_hash": study.comparison_spec.identity,
        "compiler_input_hash": inputs.identity,
        "data_manifest_hash": inputs.data_manifest_hash,
        "train_view_hash": inputs.train_view_hash,
        "global_step_plan_hash": inputs.global_step_plan_hash,
        "release_hash": release_hash,
        "global_step_max_loss_tokens": inputs.optimizer_step_token_cap,
        "global_step_max_attention_cells": inputs.optimizer_step_attention_cell_cap,
        "global_step_max_sequences": inputs.optimizer_step_sequence_cap,
    }
    for field_name, value in expected.items():
        if getattr(receipt, field_name) != value:
            raise ValueError(
                f"configuration execution path {field_name} mismatch"
            )
    if any(condition.profile.context_length != receipt.context_length for condition in conditions):
        raise ValueError("configuration execution path condition context mismatch")
    return receipt


def build_configuration_study_campaign_plan(
    *,
    study_plan: ConfigurationStudyPlanV2 | Mapping[str, Any],
    condition_ids: tuple[str, ...],
    inputs: ConfigurationCompilerInputReceiptV1 | Mapping[str, Any],
    planned_evaluation_manifest: EvaluationManifest,
    release_hash: str,
    execution_path_receipt: ConfigurationExecutionPathReceiptV1 | Mapping[str, Any],
    per_attempt_cap_usd: Decimal,
    reserve_usd: Decimal,
    provenance_evidence_hash: str,
    contamination_review_evidence_hash: str,
    implementation_test_evidence_hash: str,
    c3b_requirement: C3bRequirement | Mapping[str, Any] | None = None,
    c3a_comparison: MultiArmComparisonReceipt | None = None,
) -> CampaignPlan:
    """Predeclare one exact stage with one fresh retry for every condition."""

    study = _study(study_plan)
    input_receipt = _inputs(inputs)
    _require_hash(release_hash, "release_hash")
    _positive_money(per_attempt_cap_usd, "per_attempt_cap_usd")
    _positive_money(reserve_usd, "reserve_usd")
    stage, conditions = _selected_conditions(study, condition_ids)
    _validate_c3b_campaign_requirement(
        study=study,
        stage=stage,
        condition_ids=condition_ids,
        c3b_requirement=c3b_requirement,
        c3a_comparison=c3a_comparison,
    )
    execution_path = _validated_execution_path(
        execution_path_receipt=execution_path_receipt,
        stage=stage,
        study=study,
        inputs=input_receipt,
        release_hash=release_hash,
        conditions=conditions,
    )
    for condition in conditions:
        _validate_study_inputs(study, condition, input_receipt)
    evaluation, _ = _bound_evaluation(
        study=study,
        inputs=input_receipt,
        planned_evaluation_manifest=planned_evaluation_manifest,
        provenance_evidence_hash=provenance_evidence_hash,
        contamination_review_evidence_hash=contamination_review_evidence_hash,
        implementation_test_evidence_hash=implementation_test_evidence_hash,
    )
    for condition in conditions:
        evaluation.validate_use(
            phase=condition.phase,
            observed_task_ids=("validation_nll",),
            ranking_task_ids=("validation_nll",),
        )
    order_hash = _data_order_hash(input_receipt)

    primary = tuple(
        AttemptPlan(
            attempt_id=condition.condition_id,
            model_config_hash=str(
                compile_configuration_model_artifact(condition.profile_id)["model_hash"]
            ),
            data_order_hash=order_hash,
            execution_path_hash=execution_path.identity,
            max_cost_usd=per_attempt_cap_usd,
        )
        for condition in conditions
    )
    retries = tuple(
        AttemptPlan(
            attempt_id=f"{attempt.attempt_id}-fresh-retry",
            model_config_hash=attempt.model_config_hash,
            data_order_hash=attempt.data_order_hash,
            execution_path_hash=attempt.execution_path_hash,
            max_cost_usd=attempt.max_cost_usd,
            recovery_mode="fresh_restart",
            predecessor_attempt_id=attempt.attempt_id,
        )
        for attempt in primary
    )
    wave_cap = per_attempt_cap_usd * len(primary)
    waves = (
        WavePlan(f"{stage}-primary", "configuration", wave_cap, primary, ()),
        WavePlan(
            f"{stage}-fresh-retry",
            "configuration",
            wave_cap,
            retries,
            (f"{stage}-primary",),
        ),
    )
    return CampaignPlan(
        campaign_id=f"twc-{stage}-configuration-v2",
        budget_cap_usd=reserve_usd + wave_cap * 2,
        reserve_usd=reserve_usd,
        code_hash=release_hash,
        evaluation_policy_hash=evaluation.identity,
        waves=waves,
    )


def _validate_campaign_shape(
    campaign: CampaignPlan,
    stage: str,
    study: ConfigurationStudyPlanV2,
    inputs: ConfigurationCompilerInputReceiptV1,
    execution_path: ConfigurationExecutionPathReceiptV1,
) -> None:
    if campaign.campaign_id != f"twc-{stage}-configuration-v2":
        raise ValueError("configuration campaign id mismatch")
    expected_wave_ids = (f"{stage}-primary", f"{stage}-fresh-retry")
    if tuple(wave.wave_id for wave in campaign.waves) != expected_wave_ids:
        raise ValueError("configuration campaign must contain primary and fresh-retry waves")
    if (
        tuple(wave.purpose for wave in campaign.waves) != ("configuration", "configuration")
        or campaign.waves[0].prerequisite_wave_ids
        or campaign.waves[1].prerequisite_wave_ids != (f"{stage}-primary",)
    ):
        raise ValueError("configuration campaign wave prerequisites are invalid")
    primary_ids = tuple(attempt.attempt_id for attempt in campaign.waves[0].attempts)
    selected_stage, selected = _selected_conditions(study, primary_ids)
    if selected_stage != stage or tuple(item.condition_id for item in selected) != primary_ids:
        raise ValueError("configuration campaign condition order mismatch")
    retries = campaign.waves[1].attempts
    if tuple(attempt.predecessor_attempt_id for attempt in retries) != primary_ids or tuple(
        attempt.attempt_id for attempt in retries
    ) != tuple(f"{attempt_id}-fresh-retry" for attempt_id in primary_ids):
        raise ValueError("configuration campaign fresh-retry bindings are invalid")
    order_hash = _data_order_hash(inputs)
    for condition, primary, retry in zip(selected, campaign.waves[0].attempts, retries):
        expected_model_hash = compile_configuration_model_artifact(condition.profile_id)[
            "model_hash"
        ]
        for attempt in (primary, retry):
            if attempt.model_config_hash != expected_model_hash:
                raise ValueError("configuration campaign attempt model_config_hash mismatch")
            if attempt.data_order_hash != order_hash:
                raise ValueError("configuration campaign attempt data_order_hash mismatch")
            if attempt.execution_path_hash != execution_path.identity:
                raise ValueError("configuration campaign attempt execution_path_hash mismatch")
        if retry.max_cost_usd != primary.max_cost_usd:
            raise ValueError("configuration campaign retry cost cap mismatch")
    for wave in campaign.waves:
        exact_cap = sum((attempt.max_cost_usd for attempt in wave.attempts), Decimal("0"))
        if wave.cap_usd != exact_cap:
            raise ValueError("configuration campaign wave cap arithmetic mismatch")
    if campaign.budget_cap_usd != (
        campaign.reserve_usd + campaign.waves[0].cap_usd + campaign.waves[1].cap_usd
    ):
        raise ValueError("configuration campaign exact budget arithmetic mismatch")


def compile_configuration_study_bundle(
    *,
    study_plan: ConfigurationStudyPlanV2 | Mapping[str, Any],
    condition_id: str,
    inputs: ConfigurationCompilerInputReceiptV1 | Mapping[str, Any],
    campaign_plan: CampaignPlan,
    campaign_wave_id: str,
    campaign_attempt_id: str,
    planned_evaluation_manifest: EvaluationManifest,
    release_hash: str,
    execution_path_receipt: ConfigurationExecutionPathReceiptV1 | Mapping[str, Any],
    stage_authorization: ConfigurationStageAuthorizationV1 | Mapping[str, Any],
    provenance_evidence_hash: str,
    contamination_review_evidence_hash: str,
    implementation_test_evidence_hash: str,
) -> dict[str, Any]:
    """Compile one exact scientific condition and its physical attempt binding."""

    study = _study(study_plan)
    input_receipt = _inputs(inputs)
    if not isinstance(campaign_plan, CampaignPlan):
        raise ValueError("configuration compiler requires a CampaignPlan")
    _require_hash(release_hash, "release_hash")
    try:
        condition = study.comparison_spec.condition(condition_id)
    except KeyError as error:
        raise ValueError("condition is absent from the study plan") from error
    stage = _condition_stage(condition)
    _validate_study_inputs(study, condition, input_receipt)
    execution_path = _validated_execution_path(
        execution_path_receipt=execution_path_receipt,
        stage=stage,
        study=study,
        inputs=input_receipt,
        release_hash=release_hash,
        conditions=(condition,),
    )
    authorization = (
        load_configuration_stage_authorization(stage_authorization.to_dict())
        if isinstance(stage_authorization, ConfigurationStageAuthorizationV1)
        else load_configuration_stage_authorization(stage_authorization)
    )
    _validate_campaign_shape(
        campaign_plan,
        stage,
        study,
        input_receipt,
        execution_path,
    )
    primary_condition_ids = tuple(
        attempt.attempt_id for attempt in campaign_plan.waves[0].attempts
    )
    if (
        authorization.stage != stage
        or authorization.campaign_hash != campaign_plan.identity
        or authorization.study_plan_hash != study.identity
        or authorization.comparison_spec_hash != study.comparison_spec.identity
        or authorization.release_hash != release_hash
        or authorization.execution_path_hash != execution_path.identity
        or authorization.condition_ids != primary_condition_ids
    ):
        raise ValueError("configuration stage authorization does not match the campaign")
    if campaign_plan.code_hash != release_hash:
        raise ValueError("configuration campaign code_hash mismatch")
    evaluation, implementation = _bound_evaluation(
        study=study,
        inputs=input_receipt,
        planned_evaluation_manifest=planned_evaluation_manifest,
        provenance_evidence_hash=provenance_evidence_hash,
        contamination_review_evidence_hash=contamination_review_evidence_hash,
        implementation_test_evidence_hash=implementation_test_evidence_hash,
    )
    evaluation.validate_use(
        phase=condition.phase,
        observed_task_ids=("validation_nll",),
        ranking_task_ids=("validation_nll",),
    )
    if campaign_plan.evaluation_policy_hash != evaluation.identity:
        raise ValueError("configuration campaign evaluation_policy_hash mismatch")
    expected_waves = {f"{stage}-primary", f"{stage}-fresh-retry"}
    if campaign_wave_id not in expected_waves:
        raise ValueError("campaign wave does not match the condition stage")
    planned_attempt = campaign_plan.wave(campaign_wave_id).attempt(campaign_attempt_id)
    logical_condition_id = (
        str(planned_attempt.predecessor_attempt_id)
        if planned_attempt.recovery_mode == "fresh_restart"
        else planned_attempt.attempt_id
    )
    if logical_condition_id != condition.condition_id:
        raise ValueError("campaign attempt does not match the scientific condition")

    model = compile_configuration_model_artifact(condition.profile_id)
    data_order_hash = _data_order_hash(input_receipt)
    expected_attempt = {
        "model_config_hash": model["model_hash"],
        "data_order_hash": data_order_hash,
        "execution_path_hash": execution_path.identity,
    }
    for field, value in expected_attempt.items():
        if getattr(planned_attempt, field) != value:
            raise ValueError(f"configuration campaign attempt {field} mismatch")
    warmup, stable, decay, schedule_hash = _schedule(condition)
    experiment = ExperimentContractV2(
        experiment_id=condition.condition_id,
        study_plan_hash=study.identity,
        comparison_spec_hash=study.comparison_spec.identity,
        condition=condition,
        data_manifest_hash=input_receipt.data_manifest_hash,
        train_view_hash=input_receipt.train_view_hash,
        validation_view_hash=input_receipt.validation_view_hash,
        tokenizer_hash=input_receipt.tokenizer_hash,
        evaluation_manifest_hash=evaluation.identity,
        data_order_hash=data_order_hash,
        schedule_hash=schedule_hash,
        semantic_code_hash=study.semantic_code_hash,
        learning_rate=study.selected_learning_rate,
        schedule_tail_fraction=0.1,
    )
    binding = ExperimentExecutionBindingV1(
        scientific_contract_hash=experiment.identity,
        campaign_hash=campaign_plan.identity,
        wave_id=campaign_wave_id,
        attempt_id=campaign_attempt_id,
        release_hash=release_hash,
        execution_path_hash=execution_path.identity,
        stage_authorization_hash=authorization.identity,
    )
    training = ExperimentTrainingContractV2(
        experiment=experiment,
        execution_binding=binding,
        view_hash=input_receipt.train_view_hash,
        model_hash=str(model["model_hash"]),
        global_step_plan_hash=input_receipt.global_step_plan_hash,
        optimizer_step_token_cap=input_receipt.optimizer_step_token_cap,
        optimizer_step_attention_cell_cap=input_receipt.optimizer_step_attention_cell_cap,
        optimizer_step_sequence_cap=input_receipt.optimizer_step_sequence_cap,
        precision=execution_path.precision,
        attention_backend=execution_path.attention_backend,
        attention_num_splits=execution_path.attention_num_splits,
        loss_backend=execution_path.loss_backend,
        compile_mode=execution_path.compile_mode,
        allow_tf32=execution_path.allow_tf32,
        optimizer="adamw",
        optimizer_backend=execution_path.optimizer_backend,
        adam_betas=(0.9, 0.95),
        adam_eps=1e-8,
        weight_decay=0.1,
        warmup_loss_tokens=warmup,
        stable_loss_tokens=stable,
        decay_loss_tokens=decay,
        grad_clip_norm=1.0,
        full_validation_at_schedule_boundaries=True,
    )
    validate_configuration_execution_path_for_training(
        execution_path=execution_path,
        training_contract=training,
    )
    bundle: dict[str, Any] = {
        "artifact_kind": "twc_configuration_experiment_bundle_v2",
        "schema_version": 2,
        "production_unlock_eligible": False,
        "accepted_manifest_unlock_eligible": False,
        "study_plan_hash": study.identity,
        "input_receipt_hash": input_receipt.identity,
        "campaign_plan_hash": campaign_plan.identity,
        "campaign_wave_id": campaign_wave_id,
        "campaign_attempt_id": campaign_attempt_id,
        "scientific_contract_hash": experiment.identity,
        "experiment_contract": experiment.to_dict(),
        "execution_binding": binding.to_dict(),
        "execution_path_receipt": execution_path.to_dict(),
        "stage_authorization_hash": authorization.identity,
        "stage_authorization": authorization.to_dict(),
        "training_contract": {**training.to_dict(), "run_identity": training.identity},
        "model": model,
        "evaluation_manifest": evaluation.to_dict(),
        "evaluation_implementation_receipt": implementation.to_dict(),
        "data_order_policy": DATA_ORDER_POLICY_V2,
        "schedule_policy": SCHEDULE_POLICY_V2,
        "validation_normalization": VALIDATION_NORMALIZATION_V1,
        "evaluation_interval_loss_tokens": condition.evaluation_interval_loss_tokens,
        "required_exact_evaluation_loss_tokens": condition.required_exact_evaluation_loss_tokens,
    }
    bundle["bundle_hash"] = stable_json_hash(bundle)
    return bundle


def load_configuration_study_bundle(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate a serialized v2 compiler bundle and its hash chain."""

    expected_fields = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "accepted_manifest_unlock_eligible",
        "study_plan_hash",
        "input_receipt_hash",
        "campaign_plan_hash",
        "campaign_wave_id",
        "campaign_attempt_id",
        "scientific_contract_hash",
        "experiment_contract",
        "execution_binding",
        "execution_path_receipt",
        "stage_authorization_hash",
        "stage_authorization",
        "training_contract",
        "model",
        "evaluation_manifest",
        "evaluation_implementation_receipt",
        "data_order_policy",
        "schedule_policy",
        "validation_normalization",
        "evaluation_interval_loss_tokens",
        "required_exact_evaluation_loss_tokens",
        "bundle_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("configuration study bundle fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_configuration_experiment_bundle_v2"
        or payload.get("schema_version") != 2
        or payload.get("production_unlock_eligible") is not False
        or payload.get("accepted_manifest_unlock_eligible") is not False
    ):
        raise ValueError("configuration study bundle boundary is invalid")
    expected_bundle_hash = stable_json_hash(
        {key: value for key, value in payload.items() if key != "bundle_hash"}
    )
    if payload.get("bundle_hash") != expected_bundle_hash:
        raise ValueError("configuration study bundle hash mismatch")
    for field in (
        "study_plan_hash",
        "input_receipt_hash",
        "campaign_plan_hash",
        "scientific_contract_hash",
    ):
        _require_hash(payload[field], field)
    nested_fields = (
        "experiment_contract",
        "execution_binding",
        "execution_path_receipt",
        "stage_authorization",
        "training_contract",
        "model",
        "evaluation_manifest",
        "evaluation_implementation_receipt",
    )
    if any(not isinstance(payload.get(field), Mapping) for field in nested_fields):
        raise ValueError("configuration study bundle nested artifacts are invalid")

    experiment = load_experiment_contract_v2(payload["experiment_contract"])
    binding = load_experiment_execution_binding(payload["execution_binding"])
    execution_path = load_configuration_execution_path_receipt(
        payload["execution_path_receipt"]
    )
    authorization = load_configuration_stage_authorization(
        payload["stage_authorization"]
    )
    training = load_experiment_training_contract_v2(payload["training_contract"])
    evaluation = load_evaluation_manifest(payload["evaluation_manifest"])
    implementation = load_evaluation_task_implementation_receipt(
        payload["evaluation_implementation_receipt"]
    )
    if (
        payload["scientific_contract_hash"] != experiment.identity
        or binding.scientific_contract_hash != experiment.identity
        or training.experiment.identity != experiment.identity
        or training.execution_binding.identity != binding.identity
        or binding.execution_path_hash != execution_path.identity
        or binding.stage_authorization_hash != authorization.identity
        or payload["stage_authorization_hash"] != authorization.identity
    ):
        raise ValueError("configuration study bundle scientific identity mismatch")
    if (
        payload["campaign_plan_hash"] != binding.campaign_hash
        or payload["campaign_wave_id"] != binding.wave_id
        or payload["campaign_attempt_id"] != binding.attempt_id
    ):
        raise ValueError("configuration study bundle execution binding mismatch")
    primary_stage = "c2" if experiment.phase == "m_transfer" else (
        "c3a" if experiment.seed == 29 else "c3b"
    )
    if (
        authorization.stage != primary_stage
        or authorization.campaign_hash != binding.campaign_hash
        or authorization.study_plan_hash != experiment.study_plan_hash
        or authorization.comparison_spec_hash != experiment.comparison_spec_hash
        or authorization.release_hash != binding.release_hash
        or authorization.execution_path_hash != binding.execution_path_hash
        or experiment.experiment_id not in authorization.condition_ids
    ):
        raise ValueError("configuration study bundle stage authorization mismatch")
    validate_configuration_execution_path_for_training(
        execution_path=execution_path,
        training_contract=training,
    )
    expected_path_stage = "c2" if experiment.phase == "m_transfer" else "c3"
    if (
        execution_path.stage != expected_path_stage
        or execution_path.compiler_input_hash != payload["input_receipt_hash"]
    ):
        raise ValueError("configuration study bundle execution path mismatch")

    model = payload["model"]
    model_fields = {
        "artifact_kind",
        "schema_version",
        "expected_parameter_count",
        "model_hash",
        *(field.name for field in fields(DecoderConfig)),
    }
    if set(model) != model_fields or dict(model) != compile_configuration_model_artifact(
        experiment.profile_id
    ):
        raise ValueError("configuration study bundle model artifact mismatch")
    if training.model_hash != model["model_hash"]:
        raise ValueError("configuration study bundle training/model hash mismatch")

    if evaluation.identity != experiment.evaluation_manifest_hash:
        raise ValueError("configuration study bundle evaluation manifest mismatch")
    implemented = tuple(
        task for task in evaluation.tasks if task.implementation_status == "implemented_verified"
    )
    if (
        len(implemented) != 1
        or implemented[0].task_id != "validation_nll"
        or implemented[0].implementation_receipt_hash != implementation.identity
        or implementation.item_set_hash != experiment.validation_view_hash
        or implementation.release_hash != experiment.semantic_code_hash
        or implementation.scorer_code_hash != experiment.semantic_code_hash
        or implementation.normalization_hash
        != stable_json_hash(VALIDATION_NORMALIZATION_V1)
    ):
        raise ValueError("configuration study bundle evaluator binding mismatch")
    evaluation.validate_use(
        phase=experiment.phase,
        observed_task_ids=("validation_nll",),
        ranking_task_ids=("validation_nll",),
    )

    if dict(payload["data_order_policy"]) != DATA_ORDER_POLICY_V2:
        raise ValueError("configuration study bundle data-order policy mismatch")
    if dict(payload["schedule_policy"]) != SCHEDULE_POLICY_V2:
        raise ValueError("configuration study bundle schedule policy mismatch")
    if dict(payload["validation_normalization"]) != VALIDATION_NORMALIZATION_V1:
        raise ValueError("configuration study bundle validation normalization mismatch")
    expected_data_order_hash = stable_json_hash(
        {
            "policy_hash": stable_json_hash(DATA_ORDER_POLICY_V2),
            "train_view_hash": experiment.train_view_hash,
            "global_step_plan_hash": training.global_step_plan_hash,
            "optimizer_step_token_cap": training.optimizer_step_token_cap,
            "optimizer_step_attention_cell_cap": training.optimizer_step_attention_cell_cap,
            "optimizer_step_sequence_cap": training.optimizer_step_sequence_cap,
            "paired_order_seed": 0,
        }
    )
    warmup, stable, decay, schedule_hash = _schedule(experiment.condition)
    if (
        experiment.data_order_hash != expected_data_order_hash
        or experiment.schedule_hash != schedule_hash
        or (training.warmup_loss_tokens, training.stable_loss_tokens, training.decay_loss_tokens)
        != (warmup, stable, decay)
    ):
        raise ValueError("configuration study bundle order/schedule identity mismatch")
    required_exact = payload["required_exact_evaluation_loss_tokens"]
    if not isinstance(required_exact, (list, tuple)) or (
        payload["evaluation_interval_loss_tokens"]
        != experiment.condition.evaluation_interval_loss_tokens
        or tuple(required_exact) != experiment.required_exact_evaluation_loss_tokens
    ):
        raise ValueError("configuration study bundle evaluation cadence mismatch")
    if payload["input_receipt_hash"] != payload["input_receipt_hash"].lower():
        raise ValueError("configuration study bundle input receipt hash is noncanonical")
    return dict(payload)
