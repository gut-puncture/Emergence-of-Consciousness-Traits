"""Strict compiler from checked decoder YAML policies to hash-bound contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from twc.pretraining.training.contracts import ExecutionAttempt, RunContract
from twc.pretraining.training.model import DecoderConfig, estimate_parameter_count
from twc.repro.hashing import stable_json_hash


@dataclass(frozen=True, slots=True)
class CompiledModel:
    config: DecoderConfig
    payload: dict[str, Any]
    model_hash: str


@dataclass(frozen=True, slots=True)
class CompiledTraining:
    contract: RunContract
    attempt: ExecutionAttempt


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _exact(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    unknown, missing = set(mapping) - expected, expected - set(mapping)
    if unknown or missing:
        raise ValueError(f"{label} fields invalid; unknown={sorted(unknown)} missing={sorted(missing)}")


def compile_model_yaml(path: Path) -> CompiledModel:
    source = _yaml(path)
    _exact(source, {"version", "model_class", "target_class", "expected_parameter_count", "architecture"}, "model")
    if (
        source["version"] != 3
        or source["model_class"] != "conventional_dense_decoder"
        or source["target_class"] != "100m_base"
    ):
        raise ValueError("unsupported decoder model policy")
    architecture = source["architecture"]
    if not isinstance(architecture, dict):
        raise ValueError("model architecture must be a mapping")
    numeric = {
        "vocab_size", "context_length", "hidden_size", "num_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "dropout",
    }
    fixed = {
        "norm", "norm_epsilon", "position_encoding", "rope_theta", "mlp", "attention",
        "fused_qkv", "tied_embeddings", "bias",
    }
    _exact(architecture, numeric | fixed, "model architecture")
    expected_fixed = {
        "norm": "rmsnorm", "norm_epsilon": 1e-5, "position_encoding": "rope",
        "rope_theta": 10_000, "mlp": "swiglu", "attention": "grouped_query_attention",
        "fused_qkv": True, "tied_embeddings": True, "bias": False,
    }
    if any(architecture[key] != value for key, value in expected_fixed.items()):
        raise ValueError("model architecture requests unsupported behavior")
    config = DecoderConfig(**{key: architecture[key] for key in numeric})
    count = estimate_parameter_count(config)
    if type(source["expected_parameter_count"]) is not int or source["expected_parameter_count"] != count:
        raise ValueError("model expected_parameter_count mismatch")
    payload: dict[str, Any] = {
        "artifact_kind": "twc_base_model_spec_v1",
        "schema_version": 1,
        "expected_parameter_count": count,
        **architecture,
    }
    payload["model_hash"] = stable_json_hash(payload)
    return CompiledModel(config, payload, str(payload["model_hash"]))


def _compile_training_payload(
    source: dict[str, Any],
    *,
    run_id: str,
    data_manifest_hash: str,
    view_hash: str,
    validation_view_hash: str,
    tokenizer_hash: str,
    model: CompiledModel,
    code_hash: str,
    global_step_plan_hash: str,
) -> RunContract:
    _exact(
        source,
        {
            "version", "run_class", "target_loss_tokens", "context_length",
            "finite_non_repeating_view_required", "seed", "precision", "attention", "loss",
            "compile_mode", "allow_tf32", "optimizer", "schedule_loss_tokens", "semantic_batch",
            "attempt", "checkpointing", "evaluation", "guards", "telemetry",
        },
        "training",
    )
    if (
        source["version"] != 3
        or source["run_class"] != "100m_base"
        or source["finite_non_repeating_view_required"] is not True
    ):
        raise ValueError("unsupported R7 training policy")
    if (
        type(source["target_loss_tokens"]) is not int
        or source["target_loss_tokens"] <= 0
    ):
        raise ValueError("target_loss_tokens must be a positive integer")
    if source["context_length"] != model.config.context_length:
        raise ValueError("training/model context mismatch")
    attention, loss, optimizer = source["attention"], source["loss"], source["optimizer"]
    schedule, semantic = source["schedule_loss_tokens"], source["semantic_batch"]
    attempt_policy, checkpointing = source["attempt"], source["checkpointing"]
    evaluation, guards, telemetry = source["evaluation"], source["guards"], source["telemetry"]
    _exact(attention, {"backend", "num_splits"}, "attention")
    _exact(loss, {"backend"}, "loss")
    _exact(optimizer, {"name", "backend", "learning_rate", "betas", "epsilon", "weight_decay", "grad_clip_norm"}, "optimizer")
    _exact(schedule, {"warmup", "stable", "decay"}, "schedule_loss_tokens")
    _exact(semantic, {"token_cap", "attention_cell_cap", "sequence_cap"}, "semantic_batch")
    _exact(attempt_policy, {"prefetch_batches", "pin_memory", "telemetry_every_steps"}, "attempt")
    _exact(checkpointing, {"local_every_loss_tokens", "local_max_interval_seconds", "durable_every_loss_tokens", "durable_max_interval_seconds", "verified_resume_required"}, "checkpointing")
    _exact(evaluation, {"every_loss_tokens", "full_validation_at_schedule_boundaries"}, "evaluation")
    _exact(guards, {"launch_price_snapshot_required", "cost_cap_required", "derived_storage_required", "heartbeat_required", "backend_receipt_required"}, "guards")
    _exact(telemetry, {"jsonl_required", "background_gpu_sampling_required", "input_wait_required", "recent_tokens_per_second_required", "checkpoint_phase_required"}, "telemetry")
    if (
        checkpointing["verified_resume_required"] is not True
        or evaluation["full_validation_at_schedule_boundaries"] is not True
        or not all(value is True for value in guards.values())
        or not all(value is True for value in telemetry.values())
    ):
        raise ValueError("training safety/readiness requirements cannot be disabled")
    contract = RunContract(
        run_id=run_id,
        data_manifest_hash=data_manifest_hash,
        view_hash=view_hash,
        validation_view_hash=validation_view_hash,
        tokenizer_hash=tokenizer_hash,
        model_hash=model.model_hash,
        code_hash=code_hash,
        global_step_plan_hash=global_step_plan_hash,
        target_loss_tokens=int(source["target_loss_tokens"]),
        context_length=int(source["context_length"]),
        optimizer_step_token_cap=int(semantic["token_cap"]),
        optimizer_step_attention_cell_cap=int(semantic["attention_cell_cap"]),
        optimizer_step_sequence_cap=int(semantic["sequence_cap"]),
        precision=str(source["precision"]),
        attention_backend=str(attention["backend"]),
        attention_num_splits=attention["num_splits"],
        loss_backend=str(loss["backend"]),
        compile_mode=str(source["compile_mode"]),
        allow_tf32=source["allow_tf32"],
        optimizer=str(optimizer["name"]),
        optimizer_backend=str(optimizer["backend"]),
        learning_rate=float(optimizer["learning_rate"]),
        adam_betas=tuple(float(value) for value in optimizer["betas"]),
        adam_eps=float(optimizer["epsilon"]),
        weight_decay=float(optimizer["weight_decay"]),
        warmup_loss_tokens=int(schedule["warmup"]),
        stable_loss_tokens=int(schedule["stable"]),
        decay_loss_tokens=int(schedule["decay"]),
        grad_clip_norm=float(optimizer["grad_clip_norm"]),
        seed=int(source["seed"]),
        full_validation_at_schedule_boundaries=evaluation["full_validation_at_schedule_boundaries"],
        evaluation_every_loss_tokens=int(evaluation["every_loss_tokens"]),
    )
    return contract


def compile_training_contract_yaml(
    path: Path,
    *,
    run_id: str,
    data_manifest_hash: str,
    view_hash: str,
    validation_view_hash: str,
    tokenizer_hash: str,
    model: CompiledModel,
    code_hash: str,
    global_step_plan_hash: str,
) -> RunContract:
    """Compile only immutable scientific identity from one YAML snapshot."""

    return _compile_training_payload(
        _yaml(path),
        run_id=run_id,
        data_manifest_hash=data_manifest_hash,
        view_hash=view_hash,
        validation_view_hash=validation_view_hash,
        tokenizer_hash=tokenizer_hash,
        model=model,
        code_hash=code_hash,
        global_step_plan_hash=global_step_plan_hash,
    )


def compile_training_yaml(
    path: Path,
    *,
    run_id: str,
    data_manifest_hash: str,
    view_hash: str,
    validation_view_hash: str,
    tokenizer_hash: str,
    model: CompiledModel,
    code_hash: str,
    global_step_plan_hash: str,
    attempt_id: str,
    host_id: str,
    device_execution_plan_hash: str,
    release_hash: str,
    environment_hash: str,
    data_placement_hash: str,
    storage_layout_hash: str,
) -> CompiledTraining:
    """Compatibility compiler for callers that also possess runtime facts."""

    source = _yaml(path)
    contract = _compile_training_payload(
        source,
        run_id=run_id,
        data_manifest_hash=data_manifest_hash,
        view_hash=view_hash,
        validation_view_hash=validation_view_hash,
        tokenizer_hash=tokenizer_hash,
        model=model,
        code_hash=code_hash,
        global_step_plan_hash=global_step_plan_hash,
    )
    attempt_policy = source["attempt"]
    checkpointing = source["checkpointing"]
    evaluation = source["evaluation"]
    attempt = ExecutionAttempt(
        run_identity=contract.identity,
        attempt_id=attempt_id,
        parent_attempt_hash=None,
        device_execution_plan_hash=device_execution_plan_hash,
        release_hash=release_hash,
        environment_hash=environment_hash,
        data_placement_hash=data_placement_hash,
        storage_layout_hash=storage_layout_hash,
        host_id=host_id,
        fencing_epoch=1,
        prefetch_batches=int(attempt_policy["prefetch_batches"]),
        pin_memory=attempt_policy["pin_memory"],
        telemetry_every_steps=int(attempt_policy["telemetry_every_steps"]),
        checkpoint_every_loss_tokens=int(checkpointing["local_every_loss_tokens"]),
        checkpoint_max_interval_seconds=int(checkpointing["local_max_interval_seconds"]),
        durable_every_loss_tokens=int(checkpointing["durable_every_loss_tokens"]),
        durable_max_interval_seconds=int(checkpointing["durable_max_interval_seconds"]),
        evaluation_every_loss_tokens=int(evaluation["every_loss_tokens"]),
    )
    return CompiledTraining(contract, attempt)
