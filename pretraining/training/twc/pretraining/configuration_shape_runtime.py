"""Process-isolated runtime for one diagnostic C2/C3 GPU shape point."""

from __future__ import annotations

from dataclasses import fields
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Iterable, Mapping

import torch

from twc.pretraining.campaign import CampaignPlan
from twc.pretraining.configuration_shape_campaign import (
    ConfigurationShapeProbeBundleV1,
    load_configuration_shape_probe_bundle,
)
from twc.pretraining.configuration_shape_probe import (
    CONFIGURATION_SHAPE_CAPS,
    ConfigurationShapeProbeResultV1,
    SHAPE_PROBE_MEASURED_STEPS,
    SHAPE_PROBE_WARMUP_STEPS,
)
from twc.pretraining.gpu_campaign_deployment import (
    validate_gpu_capability_receipt,
    verify_thin_runtime_release,
)
from twc.pretraining.training.data import BatchSource, load_sequence_training_artifacts
from twc.pretraining.training.deployment import (
    build_environment_receipt,
    validate_environment_receipt,
)
from twc.pretraining.training.engine import EngineConfig, TrainingEngine
from twc.pretraining.training.model import DecoderConfig, initialize_decoder
from twc.pretraining.training.planning import (
    DeviceExecutionPlan,
    GlobalStepPlan,
    build_device_execution_plan,
    global_plan_for_sequence_artifacts,
)
from twc.pretraining.training.worker import CudaUtilizationSampler, _prepare_cuda_batch
from twc.repro.hashing import stable_json_hash
from twc.runtime.processes import capture_process_identity


class UtilizationUnavailableError(RuntimeError):
    pass


def _slice_groups(
    device_plan: DeviceExecutionPlan,
    *,
    step_count: int,
) -> tuple[tuple[int, ...], ...]:
    by_step: dict[int, list[Any]] = {}
    for item in device_plan.slices:
        by_step.setdefault(item.global_step_index, []).append(item)
    groups: list[tuple[int, ...]] = []
    for step_index in range(step_count):
        slices = by_step.get(step_index)
        if not slices:
            raise ValueError(f"shape device plan lacks global step {step_index}")
        groups.extend(
            tuple(range(item.sequence_start, item.sequence_stop)) for item in slices
        )
    return tuple(groups)


def _execute_probe_window(
    *,
    engine: Any,
    source: Any,
    global_plan: GlobalStepPlan,
    device_plan: DeviceExecutionPlan,
    device: str,
    warmup_steps: int,
    measured_steps: int,
    sampler: Any,
    prepare_batch: Callable[..., Any] = _prepare_cuda_batch,
    synchronize: Callable[[], None] = torch.cuda.synchronize,
    after_warmup: Callable[[], None] = lambda: None,
    monotonic: Callable[[], float] = time.perf_counter,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_memory: Callable[[], tuple[int, int]] = lambda: (
        int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
    ),
) -> dict[str, Any]:
    """Run an exact warmup+measurement window through engine and BatchSource."""

    if warmup_steps != SHAPE_PROBE_WARMUP_STEPS or measured_steps != SHAPE_PROBE_MEASURED_STEPS:
        raise ValueError("shape runtime window differs from the frozen probe contract")
    total_steps = warmup_steps + measured_steps
    if len(global_plan.steps) < total_steps:
        raise ValueError("shape runtime global plan is shorter than the probe window")
    groups = _slice_groups(device_plan, step_count=total_steps)
    prefetched = iter(source.prefetch(groups))
    step_slices: dict[int, tuple[Any, ...]] = {}
    for item in device_plan.slices:
        if item.global_step_index < total_steps:
            step_slices.setdefault(item.global_step_index, tuple())
            step_slices[item.global_step_index] += (item,)

    measured_step_seconds: list[float] = []
    measured_input_seconds = 0.0
    measured_loss_tokens = 0
    window_step_seconds: list[float] = []
    window_input_seconds = 0.0
    window_loss_tokens = 0
    window_sample_start = 0
    finite_losses = True
    sampler_started = False
    try:
        for step_index in range(total_steps):
            if step_index == warmup_steps:
                synchronize()
                after_warmup()
                sampler.start()
                sampler_started = True
                window_sample_start = sampler.sample_count()
                if progress_callback is not None:
                    allocated, reserved = progress_memory()
                    progress_callback(
                        {
                            "artifact_kind": "twc_configuration_shape_probe_progress_v1",
                            "phase": "warmup_complete",
                            "completed_steps": warmup_steps,
                            "total_steps": total_steps,
                            "measured_steps_completed": 0,
                            "measured_steps_total": measured_steps,
                            "peak_allocated_bytes": allocated,
                            "peak_reserved_bytes": reserved,
                            "heartbeat_epoch": time.time(),
                            "raw_text_serialized": False,
                            "tokens_serialized": False,
                        }
                    )
            slices = step_slices[step_index]
            input_seconds = [0.0]

            def microbatches() -> Iterable[Any]:
                for _item in slices:
                    started = monotonic()
                    try:
                        batch = next(prefetched)
                    except StopIteration as error:
                        raise ValueError("shape prefetch ended before the device plan") from error
                    prepared = prepare_batch(batch, device=device, pin_memory=True)
                    input_seconds[0] += monotonic() - started
                    yield prepared

            synchronize()
            started = monotonic()
            result = engine.train_step(
                microbatches(),
                expected_loss_tokens=global_plan.steps[step_index].loss_tokens,
            )
            synchronize()
            elapsed = max(monotonic() - started, 1e-12)
            if not math.isfinite(float(result.loss)):
                finite_losses = False
                raise FloatingPointError("shape probe observed a non-finite training loss")
            if result.loss_tokens != global_plan.steps[step_index].loss_tokens:
                raise ValueError("shape engine and global-plan loss tokens differ")
            if step_index >= warmup_steps:
                measured_step_seconds.append(elapsed)
                measured_input_seconds += input_seconds[0]
                measured_loss_tokens += int(result.loss_tokens)
                window_step_seconds.append(elapsed)
                window_input_seconds += input_seconds[0]
                window_loss_tokens += int(result.loss_tokens)
                if progress_callback is not None and len(window_step_seconds) == 10:
                    window_seconds = sum(window_step_seconds)
                    utilization = sampler.evidence_since(window_sample_start)
                    allocated, reserved = progress_memory()
                    progress_callback(
                        {
                            "artifact_kind": "twc_configuration_shape_probe_progress_v1",
                            "phase": "measuring",
                            "completed_steps": step_index + 1,
                            "total_steps": total_steps,
                            "measured_steps_completed": len(measured_step_seconds),
                            "measured_steps_total": measured_steps,
                            "window_steps": 10,
                            "window_loss_tokens": window_loss_tokens,
                            "window_seconds": window_seconds,
                            "window_throughput_loss_tokens_per_second": window_loss_tokens / window_seconds,
                            "window_input_wait_fraction": window_input_seconds / window_seconds,
                            **utilization,
                            "peak_allocated_bytes": allocated,
                            "peak_reserved_bytes": reserved,
                            "heartbeat_epoch": time.time(),
                            "raw_text_serialized": False,
                            "tokens_serialized": False,
                        }
                    )
                    window_step_seconds.clear()
                    window_input_seconds = 0.0
                    window_loss_tokens = 0
                    window_sample_start = sampler.sample_count()
    finally:
        if sampler_started:
            sampler.stop()
    if len(measured_step_seconds) != measured_steps:
        raise ValueError("shape runtime did not complete the exact measured window")
    utilization = sampler.evidence()
    if (
        utilization.get("gpu_utilization_status") != "measured"
        or utilization.get("median_gpu_utilization") is None
        or int(utilization.get("gpu_utilization_samples", 0)) <= 0
    ):
        raise UtilizationUnavailableError("shape probe GPU utilization evidence is unavailable")
    measured_seconds = sum(measured_step_seconds)
    return {
        "measured_loss_tokens": measured_loss_tokens,
        "measured_seconds": measured_seconds,
        "throughput_loss_tokens_per_second": measured_loss_tokens / measured_seconds,
        "median_step_seconds": statistics.median(measured_step_seconds),
        "input_wait_fraction": measured_input_seconds / measured_seconds,
        "median_gpu_utilization": float(utilization["median_gpu_utilization"]),
        "finite_losses": finite_losses,
    }


def _engine_config_from_shape_bundle(
    bundle: ConfigurationShapeProbeBundleV1,
) -> EngineConfig:
    """Construct the hot-path config without creating a scientific contract."""

    return EngineConfig(
        learning_rate=float(bundle.learning_rate),
        adam_betas=tuple(float(value) for value in bundle.adam_betas),
        adam_eps=float(bundle.adam_eps),
        weight_decay=float(bundle.weight_decay),
        precision=str(bundle.precision),
        attention_backend=str(bundle.attention_backend),
        compile_mode=str(bundle.compile_mode),
        world_size=1,
        grad_clip_norm=float(bundle.grad_clip_norm),
        warmup_tokens=int(bundle.warmup_loss_tokens),
        stable_until_tokens=int(bundle.warmup_loss_tokens)
        + int(bundle.stable_loss_tokens),
        target_tokens=int(bundle.target_loss_tokens),
        attention_num_splits=bundle.attention_num_splits,
        optimizer_backend=str(bundle.optimizer_backend),
        allow_tf32=bool(bundle.allow_tf32),
    )


def _current_process_identity_hash(*, attempt_id: str, max_sequences: int) -> str:
    identity = capture_process_identity(
        os.getpid(),
        command={
            "argv": list(sys.argv),
            "env": {},
            "label": f"configuration-shape-probe-{max_sequences}",
            "role": "configuration_shape_probe",
        },
        attempt_id=attempt_id,
    )
    return stable_json_hash(identity)


def run_configuration_shape_probe_point(
    *,
    bundle_payload: Mapping[str, Any],
    campaign_plan: CampaignPlan,
    pack_root: Path,
    manifest_path: Path,
    train_view_path: Path,
    global_plan_path: Path,
    placement_receipt: Mapping[str, Any],
    release_receipt: Mapping[str, Any],
    environment_receipt: Mapping[str, Any],
    capability_receipt: Mapping[str, Any],
    max_sequences: int,
    device: str = "cuda:0",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute one fresh diagnostic point and return only no-raw evidence."""

    if device != "cuda:0":
        raise ValueError("configuration shape probes require the single-device cuda:0 path")
    if max_sequences not in CONFIGURATION_SHAPE_CAPS:
        raise ValueError("configuration shape probe sequence cap is not predeclared")
    bundle = load_configuration_shape_probe_bundle(
        bundle_payload, campaign_plan=campaign_plan
    )
    stage = bundle.stage
    profile_id = bundle.profile_id
    point_path = bundle.execution_matrix.point(max_sequences)

    verified_release = verify_thin_runtime_release(release_receipt)
    release_hash = str(verified_release["release_hash"])
    if release_hash != bundle.release_hash or release_hash != point_path.release_hash:
        raise ValueError("shape release does not bind the diagnostic bundle")
    release_root = Path(str(verified_release["path"])).resolve()
    if not Path(__file__).resolve().is_relative_to(release_root):
        raise ValueError("shape probe was not launched from the verified thin release")

    environment = validate_environment_receipt(environment_receipt, require_cuda_ready=True)
    expected_environment = {
        "device": device,
        "precision": bundle.precision,
        "attention_backend": bundle.attention_backend,
        "optimizer_backend": bundle.optimizer_backend,
        "compile_mode": bundle.compile_mode,
        "allow_tf32": bundle.allow_tf32,
    }
    if build_environment_receipt(**expected_environment) != environment:
        raise ValueError("shape probe live environment differs from captured evidence")
    if environment.get("environment_hash") != point_path.environment_hash:
        raise ValueError("shape probe environment differs from the execution matrix")
    validate_gpu_capability_receipt(capability_receipt, require_ready=True)
    if (
        capability_receipt.get("release_hash") != release_hash
        or capability_receipt.get("capability_hash") != point_path.device_identity_hash
        or capability_receipt.get("compile_mode") != bundle.compile_mode
        or capability_receipt.get("torch_version") != environment["torch_version"]
        or capability_receipt.get("cuda_runtime_version") != environment["cuda_runtime_version"]
    ):
        raise ValueError("shape capability/release/environment lineage mismatch")

    artifacts = load_sequence_training_artifacts(
        pack_root=pack_root,
        manifest_path=manifest_path,
        view_path=train_view_path,
        global_plan_path=global_plan_path,
        placement_receipt=dict(placement_receipt),
    )
    global_plan = global_plan_for_sequence_artifacts(artifacts)
    if (
        artifacts.manifest_hash != bundle.data_manifest_hash
        or artifacts.manifest_hash != point_path.data_manifest_hash
        or artifacts.view_hash != bundle.train_view_hash
        or artifacts.view_hash != point_path.train_view_hash
        or artifacts.global_plan_hash != bundle.global_step_plan_hash
        or artifacts.global_plan_hash != point_path.global_step_plan_hash
        or artifacts.placement_hash != placement_receipt.get("receipt_hash")
        or artifacts.placement_hash != point_path.placement_hash
    ):
        raise ValueError("shape pack/view/plan/placement lineage mismatch")
    expected_caps = (
        bundle.optimizer_step_token_cap,
        bundle.optimizer_step_attention_cell_cap,
        bundle.optimizer_step_sequence_cap,
    )
    if (
        global_plan.max_loss_tokens,
        global_plan.max_attention_cells,
        global_plan.max_sequences,
    ) != expected_caps:
        raise ValueError("shape global plan differs from scientific optimizer semantics")
    if global_plan.total_loss_tokens != bundle.target_loss_tokens:
        raise ValueError("shape global plan does not cover the diagnostic token horizon")
    device_plan = build_device_execution_plan(
        global_plan,
        artifacts.loss_token_lengths,
        max_loss_tokens=expected_caps[0],
        max_attention_cells=expected_caps[1],
        max_sequences=max_sequences,
    )

    model_payload = bundle.model
    model_config = DecoderConfig(
        **{field.name: model_payload[field.name] for field in fields(DecoderConfig)}
    )
    total_memory = int(capability_receipt["vram_bytes"])
    live_total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    if live_total_memory != total_memory:
        raise ValueError("shape probe live VRAM differs from capability evidence")

    common = {
        "stage": stage,
        "profile_id": profile_id,
        "condition_id": bundle.condition_id,
        "seed": bundle.seed,
        "condition_hash": bundle.condition_hash,
        "model_hash": str(model_payload["model_hash"]),
        "study_plan_hash": bundle.study_plan_hash,
        "comparison_spec_hash": bundle.comparison_spec_hash,
        "compiler_input_hash": bundle.compiler_input_hash,
        "campaign_hash": bundle.campaign_hash,
        "campaign_wave_id": bundle.campaign_wave_id,
        "campaign_attempt_id": bundle.campaign_attempt_id,
        "execution_matrix_hash": bundle.execution_matrix_hash,
        "point_execution_path_hash": point_path.identity,
        "bundle_hash": bundle.identity,
        "data_manifest_hash": artifacts.manifest_hash,
        "train_view_hash": artifacts.view_hash,
        "global_step_plan_hash": artifacts.global_plan_hash,
        "release_hash": release_hash,
        "environment_hash": str(environment["environment_hash"]),
        "placement_hash": artifacts.placement_hash,
        "device_identity_hash": str(capability_receipt["capability_hash"]),
        "process_identity_hash": _current_process_identity_hash(
            attempt_id=bundle.campaign_attempt_id,
            max_sequences=max_sequences,
        ),
        "max_sequences": max_sequences,
        "max_loss_tokens": expected_caps[0],
        "max_attention_cells": expected_caps[1],
        "optimizer_step_sequence_cap": expected_caps[2],
        "warmup_steps": SHAPE_PROBE_WARMUP_STEPS,
        "measured_steps": SHAPE_PROBE_MEASURED_STEPS,
        "total_memory_bytes": total_memory,
    }
    try:
        engine = TrainingEngine(
            initialize_decoder(model_config, seed=bundle.seed),
            _engine_config_from_shape_bundle(bundle),
            device=device,
        )
        sampler = CudaUtilizationSampler(device)
        with BatchSource(root=artifacts.root, records=artifacts.records, read_mode="local_mmap") as source:
            measurements = _execute_probe_window(
                engine=engine,
                source=source,
                global_plan=global_plan,
                device_plan=device_plan,
                device=device,
                warmup_steps=SHAPE_PROBE_WARMUP_STEPS,
                measured_steps=SHAPE_PROBE_MEASURED_STEPS,
                sampler=sampler,
                after_warmup=torch.cuda.reset_peak_memory_stats,
                progress_callback=progress_callback,
            )
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        result = ConfigurationShapeProbeResultV1(
            **common,
            status="success",
            failure_class=None,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            memory_headroom_fraction=1 - peak_reserved / total_memory,
            **measurements,
        )
    except torch.OutOfMemoryError:
        result = ConfigurationShapeProbeResultV1(
            **common,
            status="out_of_memory",
            failure_class="out_of_memory",
            measured_loss_tokens=0,
            measured_seconds=0.0,
            throughput_loss_tokens_per_second=None,
            median_step_seconds=None,
            input_wait_fraction=None,
            median_gpu_utilization=None,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            memory_headroom_fraction=None,
            finite_losses=False,
        )
    except FloatingPointError:
        result = ConfigurationShapeProbeResultV1(
            **common,
            status="failed",
            failure_class="nonfinite_loss",
            measured_loss_tokens=0,
            measured_seconds=0.0,
            throughput_loss_tokens_per_second=None,
            median_step_seconds=None,
            input_wait_fraction=None,
            median_gpu_utilization=None,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            memory_headroom_fraction=None,
            finite_losses=False,
        )
    except UtilizationUnavailableError:
        result = ConfigurationShapeProbeResultV1(
            **common,
            status="failed",
            failure_class="gpu_utilization_unavailable",
            measured_loss_tokens=0,
            measured_seconds=0.0,
            throughput_loss_tokens_per_second=None,
            median_step_seconds=None,
            input_wait_fraction=None,
            median_gpu_utilization=None,
            peak_allocated_bytes=None,
            peak_reserved_bytes=None,
            memory_headroom_fraction=None,
            finite_losses=False,
        )
    finally:
        torch.cuda.empty_cache()
    return result.to_dict()
