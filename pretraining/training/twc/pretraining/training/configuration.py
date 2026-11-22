"""Compile and validate the one portable pretraining configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from twc.pretraining.configuration_study import (
    ConfigurationStudyPlanV2,
    ExperimentContractV2,
    StudyCondition,
    build_default_c2_c3_comparison_spec,
    load_configuration_study_plan,
)
from twc.pretraining.configuration_study_compiler import (
    ConfigurationCompilerInputReceiptV1,
    _data_order_hash,
    _schedule,
    _validate_study_inputs,
    compile_configuration_model_artifact,
    load_configuration_compiler_input_receipt,
)
from twc.pretraining.configuration_study_lineage import (
    C1ScreenAdoptionReceiptV1,
    load_c1_screen_adoption_receipt,
)
from twc.pretraining.training.contracts import (
    RunContract,
    build_training_config,
    load_training_config,
    training_config_contract,
)
from twc.pretraining.training.experiments import (
    EvaluationManifest,
    EvaluationTaskImplementationReceipt,
    load_evaluation_manifest,
    load_evaluation_task_implementation_receipt,
)


_SOURCE_FILES = {
    "study": "study.json",
    "c1": "c1-adoption.json",
    "inputs": "data-lineage.json",
    "evaluation": "evaluation-manifest.json",
    "implementation": "evaluation-implementation.json",
}
_STUDY_SOURCE_FIELDS = {
    "implementation_hash",
    "source_kind",
    "study_plan_hash",
    "comparison_spec_hash",
    "c1_adoption_receipt_hash",
    "compiler_input_hash",
    "tokenizer_adoption_receipt_hash",
    "data_plan_hash",
    "prefix_continuity_receipt_hash",
    "data_order_hash",
    "schedule_hash",
    "selected_learning_rate",
    "condition",
}
_BASE_SOURCE_FIELDS = {
    "implementation_hash",
    "source_kind",
    "semantic_code_receipt_hash",
    "evaluation_manifest_hash",
    "evaluation_implementation_hash",
}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink JSON file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def compile_base_pretraining_root(
    *,
    run_root: Path,
    run_id: str,
    model_config_path: Path,
    training_config_path: Path,
    pack_manifest_path: Path,
    tokenizer_manifest_path: Path,
    train_view_path: Path,
    validation_view_path: Path,
    global_plan_path: Path,
    semantic_code_receipt_path: Path,
    evaluation_manifest_path: Path,
    evaluation_implementation_path: Path,
) -> dict[str, Any]:
    """Compatibility seam for the lazily loaded compile-time integration."""

    from twc.pretraining.base_recipe_compiler import (
        compile_base_pretraining_root as compile_recipe,
    )

    return compile_recipe(
        run_root=run_root,
        run_id=run_id,
        model_config_path=model_config_path,
        training_config_path=training_config_path,
        pack_manifest_path=pack_manifest_path,
        tokenizer_manifest_path=tokenizer_manifest_path,
        train_view_path=train_view_path,
        validation_view_path=validation_view_path,
        global_plan_path=global_plan_path,
        semantic_code_receipt_path=semantic_code_receipt_path,
        evaluation_manifest_path=evaluation_manifest_path,
        evaluation_implementation_path=evaluation_implementation_path,
    )


def _stage_conditions(
    study: ConfigurationStudyPlanV2, phase: str
) -> tuple[StudyCondition, ...]:
    if phase not in {"m_transfer", "l_confirmation"}:
        raise ValueError("study input phase is unsupported")
    conditions = tuple(
        condition
        for condition in study.comparison_spec.conditions
        if condition.phase == phase
    )
    expected_count = 3 if phase == "m_transfer" else 6
    if len(conditions) != expected_count:
        raise ValueError("study does not contain the exact frozen stage conditions")
    return conditions


def _input_phase(
    study: ConfigurationStudyPlanV2,
    inputs: ConfigurationCompilerInputReceiptV1,
) -> str:
    if inputs.train_view_hash == study.train_view_hashes[0]:
        return "m_transfer"
    if inputs.train_view_hash == study.train_view_hashes[1]:
        return "l_confirmation"
    raise ValueError("study input train view is not frozen")


def _validate_evaluation(
    study: ConfigurationStudyPlanV2,
    manifest: EvaluationManifest,
    implementation: EvaluationTaskImplementationReceipt,
) -> None:
    if manifest.identity != study.evaluation_manifest_hash:
        raise ValueError("evaluation manifest does not match the study")
    task = next((row for row in manifest.tasks if row.task_id == "validation_nll"), None)
    if (
        task is None
        or task.implementation_receipt_hash != implementation.identity
        or implementation.task_id != "validation_nll"
        or implementation.release_hash != study.semantic_code_hash
        or implementation.item_set_hash != study.validation_view_hash
        or task.scorer_code_hash != implementation.scorer_code_hash
        or task.item_set_hash != implementation.item_set_hash
        or task.normalization_hash != implementation.normalization_hash
    ):
        raise ValueError("evaluation implementation does not match the study")


def _build_study_config(
    *,
    study: ConfigurationStudyPlanV2,
    adoption: C1ScreenAdoptionReceiptV1,
    inputs: ConfigurationCompilerInputReceiptV1,
    evaluation: EvaluationManifest,
    implementation: EvaluationTaskImplementationReceipt,
    condition: StudyCondition,
) -> dict[str, Any]:
    if adoption.receipt_hash != study.c1_adoption_receipt_hash:
        raise ValueError("C1 adoption does not match the study")
    if condition not in _stage_conditions(study, _input_phase(study, inputs)):
        raise ValueError("only exact frozen study conditions may be compiled")
    _validate_study_inputs(study, condition, inputs)
    _validate_evaluation(study, evaluation, implementation)
    warmup, stable, decay, schedule_hash = _schedule(condition)
    model = compile_configuration_model_artifact(condition.profile_id)
    science = ExperimentContractV2(
        experiment_id=condition.condition_id,
        study_plan_hash=study.identity,
        comparison_spec_hash=study.comparison_spec.identity,
        condition=condition,
        data_manifest_hash=inputs.data_manifest_hash,
        train_view_hash=inputs.train_view_hash,
        validation_view_hash=inputs.validation_view_hash,
        tokenizer_hash=inputs.tokenizer_hash,
        evaluation_manifest_hash=evaluation.identity,
        data_order_hash=_data_order_hash(inputs),
        schedule_hash=schedule_hash,
        semantic_code_hash=study.semantic_code_hash,
        learning_rate=study.selected_learning_rate,
        schedule_tail_fraction=0.1,
    )
    contract = RunContract(
        run_id=condition.condition_id,
        data_manifest_hash=inputs.data_manifest_hash,
        view_hash=inputs.train_view_hash,
        validation_view_hash=inputs.validation_view_hash,
        tokenizer_hash=inputs.tokenizer_hash,
        model_hash=str(model["model_hash"]),
        code_hash=study.semantic_code_hash,
        global_step_plan_hash=inputs.global_step_plan_hash,
        target_loss_tokens=condition.target_loss_tokens,
        context_length=condition.profile.context_length,
        optimizer_step_token_cap=inputs.optimizer_step_token_cap,
        optimizer_step_attention_cell_cap=inputs.optimizer_step_attention_cell_cap,
        optimizer_step_sequence_cap=inputs.optimizer_step_sequence_cap,
        precision="bf16",
        attention_backend="native_varlen",
        attention_num_splits="auto",
        loss_backend="chunked_projection",
        compile_mode="default",
        allow_tf32=True,
        optimizer="adamw",
        optimizer_backend="fused",
        learning_rate=science.learning_rate,
        adam_betas=(0.9, 0.95),
        adam_eps=1e-8,
        weight_decay=0.1,
        warmup_loss_tokens=warmup,
        stable_loss_tokens=stable,
        decay_loss_tokens=decay,
        grad_clip_norm=1.0,
        seed=condition.seed,
        full_validation_at_schedule_boundaries=True,
        evaluation_every_loss_tokens=condition.evaluation_interval_loss_tokens,
    )
    return build_training_config(
        contract=contract,
        model=model,
        evaluation_manifest=evaluation.to_dict(),
        evaluation_implementation=implementation.to_dict(),
        provenance={
            "source_kind": "configuration_study",
            "study_plan_hash": study.identity,
            "comparison_spec_hash": study.comparison_spec.identity,
            "c1_adoption_receipt_hash": adoption.receipt_hash,
            "compiler_input_hash": inputs.identity,
            "tokenizer_adoption_receipt_hash": study.tokenizer_adoption_receipt_hash,
            "data_plan_hash": study.data_plan_hash,
            "prefix_continuity_receipt_hash": study.prefix_continuity_receipt_hash,
            "data_order_hash": science.data_order_hash,
            "schedule_hash": schedule_hash,
            "selected_learning_rate": study.selected_learning_rate,
            "condition": condition.to_dict(),
        },
    )


def _validate_study_source(config: Mapping[str, Any]) -> None:
    source = config["source"]
    if set(source) != _STUDY_SOURCE_FIELDS:
        raise ValueError("configuration-study source fields are invalid")
    condition_payload = source.get("condition")
    if not isinstance(condition_payload, Mapping):
        raise ValueError("configuration-study source is incomplete")
    values = dict(condition_payload)
    values["required_exact_evaluation_loss_tokens"] = tuple(
        values["required_exact_evaluation_loss_tokens"]
    )
    condition = StudyCondition(**values)
    frozen = build_default_c2_c3_comparison_spec()
    try:
        expected_condition = frozen.condition(condition.condition_id)
    except KeyError as error:
        raise ValueError("configuration-study condition is not frozen") from error
    if condition != expected_condition:
        raise ValueError("configuration-study condition differs from the frozen study")
    for field in _STUDY_SOURCE_FIELDS - {
        "source_kind",
        "selected_learning_rate",
        "condition",
    }:
        value = source[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("configuration-study source hashes are invalid")
    contract = training_config_contract(config)
    warmup, stable, decay, schedule_hash = _schedule(condition)
    if (
        config["run_id"] != condition.condition_id
        or config["model"] != compile_configuration_model_artifact(condition.profile_id)
        or contract.target_loss_tokens != condition.target_loss_tokens
        or contract.context_length != condition.profile.context_length
        or contract.seed != condition.seed
        or contract.evaluation_every_loss_tokens
        != condition.evaluation_interval_loss_tokens
        or contract.full_validation_at_schedule_boundaries is not True
        or source["schedule_hash"] != schedule_hash
        or source["selected_learning_rate"] != contract.learning_rate
        or (
            contract.precision,
            contract.attention_backend,
            contract.attention_num_splits,
            contract.loss_backend,
            contract.compile_mode,
            contract.allow_tf32,
            contract.optimizer,
            contract.optimizer_backend,
            list(contract.adam_betas),
            contract.adam_eps,
            contract.weight_decay,
            contract.grad_clip_norm,
        )
        != (
            "bf16",
            "native_varlen",
            "auto",
            "chunked_projection",
            "default",
            True,
            "adamw",
            "fused",
            [0.9, 0.95],
            1e-8,
            0.1,
            1.0,
        )
        or (
            contract.warmup_loss_tokens,
            contract.stable_loss_tokens,
            contract.decay_loss_tokens,
        )
        != (warmup, stable, decay)
    ):
        raise ValueError("training configuration differs from its frozen study source")
    evaluation = load_evaluation_manifest(config["evaluation"]["manifest"])
    implementation = load_evaluation_task_implementation_receipt(
        config["evaluation"]["implementation"]
    )
    task = next((row for row in evaluation.tasks if row.task_id == "validation_nll"), None)
    if (
        task is None
        or task.implementation_receipt_hash != implementation.identity
        or implementation.task_id != task.task_id
        or implementation.release_hash != contract.code_hash
        or implementation.item_set_hash != contract.validation_view_hash
        or task.scorer_code_hash != implementation.scorer_code_hash
        or task.item_set_hash != implementation.item_set_hash
        or task.normalization_hash != implementation.normalization_hash
    ):
        raise ValueError("training configuration evaluation binding is invalid")


def _validate_base_source(config: Mapping[str, Any]) -> None:
    source = config["source"]
    if set(source) != _BASE_SOURCE_FIELDS:
        raise ValueError("base-pretraining source fields are invalid")
    for field in _BASE_SOURCE_FIELDS - {"source_kind"}:
        value = source[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("base-pretraining source hashes are invalid")
    contract = training_config_contract(config)
    evaluation = load_evaluation_manifest(config["evaluation"]["manifest"])
    implementation = load_evaluation_task_implementation_receipt(
        config["evaluation"]["implementation"]
    )
    task = next(
        (row for row in evaluation.tasks if row.task_id == "validation_nll"), None
    )
    if source["implementation_hash"] != contract.code_hash:
        raise ValueError("base-pretraining source implementation binding is invalid")
    if (
        source["evaluation_manifest_hash"] != evaluation.identity
        or source["evaluation_implementation_hash"] != implementation.identity
        or task is None
        or task.implementation_receipt_hash != implementation.identity
        or implementation.task_id != "validation_nll"
        or implementation.release_hash != contract.code_hash
        or implementation.item_set_hash != contract.validation_view_hash
        or task.scorer_code_hash != implementation.scorer_code_hash
        or task.item_set_hash != implementation.item_set_hash
        or task.normalization_hash != implementation.normalization_hash
    ):
        raise ValueError("base-pretraining evaluation binding is invalid")


def load_run_root(root: Path) -> dict[str, Any]:
    requested = root.expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("run root must be a real non-symlink directory")
    config = load_training_config(
        _read_object(requested / "run.json", "training configuration")
    )
    if config["source"].get("source_kind") == "configuration_study":
        _validate_study_source(config)
    elif config["source"].get("source_kind") == "base_pretraining_recipe":
        _validate_base_source(config)
    return config


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if os.write(descriptor, data) != len(data):
            raise OSError("short training-configuration write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def compile_configuration_study_root(
    study_root: Path, output_root: Path
) -> tuple[dict[str, Any], ...]:
    source = study_root.expanduser()
    if source.is_symlink() or not source.is_dir():
        raise ValueError("study root must be a real non-symlink directory")
    study = load_configuration_study_plan(
        _read_object(source / _SOURCE_FILES["study"], "study plan")
    )
    adoption = load_c1_screen_adoption_receipt(
        _read_object(source / _SOURCE_FILES["c1"], "C1 adoption")
    )
    inputs = load_configuration_compiler_input_receipt(
        _read_object(source / _SOURCE_FILES["inputs"], "data lineage")
    )
    evaluation = load_evaluation_manifest(
        _read_object(source / _SOURCE_FILES["evaluation"], "evaluation manifest")
    )
    implementation = load_evaluation_task_implementation_receipt(
        _read_object(
            source / _SOURCE_FILES["implementation"], "evaluation implementation"
        )
    )
    runs = tuple(
        _build_study_config(
            study=study,
            adoption=adoption,
            inputs=inputs,
            evaluation=evaluation,
            implementation=implementation,
            condition=condition,
        )
        for condition in _stage_conditions(study, _input_phase(study, inputs))
    )
    output = output_root.expanduser()
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if os.path.lexists(partial):
        raise FileExistsError(f"output partial already exists: {partial}")
    partial.mkdir(mode=0o700)
    try:
        for run in runs:
            run_root = partial / str(run["run_id"])
            run_root.mkdir(mode=0o700)
            _write_json(run_root / "run.json", run)
        os.replace(partial, output)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return runs


# Compatibility for callers that still name the original C2-only entrypoint.
compile_c2_study_root = compile_configuration_study_root
