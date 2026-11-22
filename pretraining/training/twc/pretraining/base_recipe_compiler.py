"""Compile a checked base-pretraining recipe at the data/training boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from twc.pretraining.configuration_semantic_code import (
    validate_configuration_semantic_code_receipt,
)
from twc.pretraining.configuration_study_lineage import load_finite_sequence_view
from twc.pretraining.data_pipeline.sequence_pack import (
    iter_view_segments,
    validate_global_step_plan,
    validate_isolated_sequence_pack,
)
from twc.pretraining.data_pipeline.tokenizer import validate_tokenizer_artifacts
from twc.pretraining.training.config_loader import (
    compile_model_yaml,
    compile_training_contract_yaml,
)
from twc.pretraining.training.contracts import build_training_config
from twc.pretraining.training.experiments import (
    load_evaluation_manifest,
    load_evaluation_task_implementation_receipt,
)


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = (json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if os.write(descriptor, data) != len(data):
            raise OSError("short training-configuration write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_run_root(output: Path, config: Mapping[str, Any]) -> None:
    output = output.expanduser()
    if os.path.lexists(output):
        raise FileExistsError(f"run root already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if os.path.lexists(partial):
        raise FileExistsError(f"run root partial already exists: {partial}")
    partial.mkdir(mode=0o700)
    try:
        _write_json(partial / "run.json", config)
        os.replace(partial, output)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


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
    """Compile one checked base-pretraining recipe without runtime facts."""

    model = compile_model_yaml(model_config_path)
    pack = _read_object(pack_manifest_path, "pack manifest")
    tokenizer = _read_object(tokenizer_manifest_path, "tokenizer manifest")
    train_view = _read_object(train_view_path, "train view")
    validation_view = _read_object(validation_view_path, "validation view")
    global_plan = _read_object(global_plan_path, "global plan")
    semantic = validate_configuration_semantic_code_receipt(
        _read_object(semantic_code_receipt_path, "semantic code receipt")
    )
    evaluation = load_evaluation_manifest(
        _read_object(evaluation_manifest_path, "evaluation manifest")
    )
    implementation = load_evaluation_task_implementation_receipt(
        _read_object(evaluation_implementation_path, "evaluation implementation")
    )

    pack_root = pack_manifest_path.parent
    tokenizer_root = tokenizer_manifest_path.parent
    validate_isolated_sequence_pack(pack, pack_root)
    if pack.get("input_artifact_kind") not in {
        "twc_accepted_clean_run_manifest_v1",
        "twc_accepted_candidate_core_run_manifest_v1",
    }:
        raise ValueError("base pretraining requires an accepted input pack")
    pack_hash = str(pack["manifest_hash"])
    validate_tokenizer_artifacts(tokenizer, tokenizer_root)
    tokenizer_hash = str(tokenizer["manifest_hash"])
    if (
        pack.get("tokenizer_manifest_hash") != tokenizer_hash
        or pack.get("input_manifest_hash") != tokenizer.get("corpus_manifest_hash")
    ):
        raise ValueError("tokenizer and pack lineage is invalid")
    if (
        pack.get("vocab_size") != model.config.vocab_size
        or tokenizer.get("actual_vocab_size") != model.config.vocab_size
        or pack.get("context_length") != model.config.context_length
    ):
        raise ValueError("model, tokenizer, and pack dimensions differ")

    view_hashes: dict[str, str] = {}
    for split, raw_view in (("train", train_view), ("validation", validation_view)):
        view = load_finite_sequence_view(raw_view)
        if view["split"] != split or view["pack_manifest_hash"] != pack_hash:
            raise ValueError(f"{split} view binding is invalid")
        segment_count = 0
        final_segment_loss_tokens = 0
        for segment in iter_view_segments(pack, pack_root, view=view):
            segment_count += 1
            final_segment_loss_tokens = segment.loss_token_count
        if (
            segment_count != view["full_segment_count"] + 1
            or final_segment_loss_tokens != view["final_segment_loss_tokens"]
        ):
            raise ValueError(f"{split} finite sequence view summary is invalid")
        view_hashes[split] = str(view["view_hash"])

    validate_global_step_plan(
        global_plan, view=train_view, pack_manifest=pack, pack_root=pack_root
    )
    plan_hash = str(global_plan["plan_hash"])

    contract = compile_training_contract_yaml(
        training_config_path,
        run_id=run_id,
        data_manifest_hash=pack_hash,
        view_hash=view_hashes["train"],
        validation_view_hash=view_hashes["validation"],
        tokenizer_hash=tokenizer_hash,
        model=model,
        code_hash=str(semantic["semantic_code_hash"]),
        global_step_plan_hash=plan_hash,
    )
    if train_view.get("loss_token_count") != contract.target_loss_tokens:
        raise ValueError("train view does not match the training target")
    if global_plan.get("loss_token_count") != contract.target_loss_tokens:
        raise ValueError("global plan does not match the training target")
    if (
        global_plan.get("max_loss_tokens"),
        global_plan.get("max_attention_cells"),
        global_plan.get("max_sequences"),
    ) != (
        contract.optimizer_step_token_cap,
        contract.optimizer_step_attention_cell_cap,
        contract.optimizer_step_sequence_cap,
    ):
        raise ValueError("global plan semantic caps differ from the training policy")

    task = next(
        (item for item in evaluation.tasks if item.task_id == "validation_nll"), None
    )
    if (
        task is None
        or task.implementation_receipt_hash != implementation.identity
        or implementation.task_id != "validation_nll"
        or implementation.release_hash != contract.code_hash
        or implementation.item_set_hash != contract.validation_view_hash
        or task.scorer_code_hash != implementation.scorer_code_hash
        or task.item_set_hash != implementation.item_set_hash
        or task.normalization_hash != implementation.normalization_hash
    ):
        raise ValueError("evaluation implementation binding is invalid")

    config = build_training_config(
        contract=contract,
        model=model.payload,
        evaluation_manifest=evaluation.to_dict(),
        evaluation_implementation=implementation.to_dict(),
        provenance={
            "source_kind": "base_pretraining_recipe",
            "semantic_code_receipt_hash": semantic["receipt_hash"],
            "evaluation_manifest_hash": evaluation.identity,
            "evaluation_implementation_hash": implementation.identity,
        },
    )
    _publish_run_root(run_root, config)
    return config
