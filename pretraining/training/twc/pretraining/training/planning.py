"""Pure planners that separate scientific step membership from device slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from twc.repro.hashing import stable_json_hash


GLOBAL_STEP_PLAN_KIND = "twc_global_step_plan_v1"
DEVICE_EXECUTION_PLAN_KIND = "twc_device_execution_plan_v1"


@dataclass(frozen=True, slots=True)
class PlannedWork:
    sequence_start: int
    sequence_stop: int
    loss_tokens: int
    attention_cells: int

    @property
    def sequence_range(self) -> tuple[int, int]:
        return self.sequence_start, self.sequence_stop


@dataclass(frozen=True, slots=True)
class GlobalStepPlan:
    steps: tuple[PlannedWork, ...]
    sequence_count: int
    total_loss_tokens: int
    max_loss_tokens: int
    max_attention_cells: int
    max_sequences: int
    plan_hash: str
    artifact_kind: str = GLOBAL_STEP_PLAN_KIND
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class DeviceSlice(PlannedWork):
    global_step_index: int


@dataclass(frozen=True, slots=True)
class DeviceExecutionPlan:
    global_plan_hash: str
    slices: tuple[DeviceSlice, ...]
    max_loss_tokens: int
    max_attention_cells: int
    max_sequences: int
    plan_hash: str
    artifact_kind: str = DEVICE_EXECUTION_PLAN_KIND
    schema_version: int = 1


def build_global_step_plan(
    sequence_loss_tokens: Sequence[int],
    *,
    max_loss_tokens: int = 262_144,
    max_attention_cells: int = 1_073_741_824,
    max_sequences: int = 4_096,
) -> GlobalStepPlan:
    lengths = _validate_lengths(sequence_loss_tokens)
    _validate_caps(max_loss_tokens, max_attention_cells, max_sequences)
    steps = tuple(
        _next_fit(
            lengths,
            start=0,
            stop=len(lengths),
            max_loss_tokens=max_loss_tokens,
            max_attention_cells=max_attention_cells,
            max_sequences=max_sequences,
        )
    )
    payload = {
        "artifact_kind": GLOBAL_STEP_PLAN_KIND,
        "schema_version": 1,
        "sequence_count": len(lengths),
        "total_loss_tokens": sum(lengths),
        "max_loss_tokens": max_loss_tokens,
        "max_attention_cells": max_attention_cells,
        "max_sequences": max_sequences,
        "steps": [_work_dict(item) for item in steps],
    }
    return GlobalStepPlan(
        steps=steps,
        sequence_count=len(lengths),
        total_loss_tokens=sum(lengths),
        max_loss_tokens=max_loss_tokens,
        max_attention_cells=max_attention_cells,
        max_sequences=max_sequences,
        plan_hash=stable_json_hash(payload),
    )


def build_device_execution_plan(
    global_plan: GlobalStepPlan,
    sequence_loss_tokens: Sequence[int],
    *,
    max_loss_tokens: int,
    max_attention_cells: int,
    max_sequences: int,
) -> DeviceExecutionPlan:
    lengths = _validate_lengths(sequence_loss_tokens)
    _validate_caps(max_loss_tokens, max_attention_cells, max_sequences)
    if len(lengths) != global_plan.sequence_count or sum(lengths) != global_plan.total_loss_tokens:
        raise ValueError("sequence lengths do not match the global step plan")
    _validate_global_plan(global_plan, lengths)
    slices: list[DeviceSlice] = []
    for step_index, step in enumerate(global_plan.steps):
        for item in _next_fit(
            lengths,
            start=step.sequence_start,
            stop=step.sequence_stop,
            max_loss_tokens=max_loss_tokens,
            max_attention_cells=max_attention_cells,
            max_sequences=max_sequences,
        ):
            slices.append(
                DeviceSlice(
                    sequence_start=item.sequence_start,
                    sequence_stop=item.sequence_stop,
                    loss_tokens=item.loss_tokens,
                    attention_cells=item.attention_cells,
                    global_step_index=step_index,
                )
            )
    payload = {
        "artifact_kind": DEVICE_EXECUTION_PLAN_KIND,
        "schema_version": 1,
        "global_step_plan_hash": global_plan.plan_hash,
        "max_loss_tokens": max_loss_tokens,
        "max_attention_cells": max_attention_cells,
        "max_sequences": max_sequences,
        "slices": [{**_work_dict(item), "global_step_index": item.global_step_index} for item in slices],
    }
    return DeviceExecutionPlan(
        global_plan_hash=global_plan.plan_hash,
        slices=tuple(slices),
        max_loss_tokens=max_loss_tokens,
        max_attention_cells=max_attention_cells,
        max_sequences=max_sequences,
        plan_hash=stable_json_hash(payload),
    )


def device_execution_plan_artifact(plan: DeviceExecutionPlan) -> dict[str, object]:
    payload: dict[str, object] = {
        "artifact_kind": DEVICE_EXECUTION_PLAN_KIND,
        "schema_version": 1,
        "global_step_plan_hash": plan.global_plan_hash,
        "max_loss_tokens": plan.max_loss_tokens,
        "max_attention_cells": plan.max_attention_cells,
        "max_sequences": plan.max_sequences,
        "slices": [
            {**_work_dict(item), "global_step_index": item.global_step_index}
            for item in plan.slices
        ],
    }
    payload["plan_hash"] = stable_json_hash(payload)
    if payload["plan_hash"] != plan.plan_hash:
        raise ValueError("device plan serialization changed its identity")
    return payload


def global_plan_for_sequence_artifacts(artifacts) -> GlobalStepPlan:
    lengths = tuple(artifacts.loss_token_lengths)
    steps = tuple(
        PlannedWork(
            sequence_start=start,
            sequence_stop=stop,
            loss_tokens=sum(lengths[start:stop]),
            attention_cells=sum(value * value for value in lengths[start:stop]),
        )
        for start, stop in artifacts.global_steps
    )
    _validate_caps(
        artifacts.global_plan_max_loss_tokens,
        artifacts.global_plan_max_attention_cells,
        artifacts.global_plan_max_sequences,
    )
    plan = GlobalStepPlan(
        steps=steps,
        sequence_count=len(artifacts.records),
        total_loss_tokens=sum(lengths),
        max_loss_tokens=artifacts.global_plan_max_loss_tokens,
        max_attention_cells=artifacts.global_plan_max_attention_cells,
        max_sequences=artifacts.global_plan_max_sequences,
        plan_hash=artifacts.global_plan_hash,
    )
    _validate_global_plan(plan, lengths)
    return plan


def _validate_lengths(values: Sequence[int]) -> tuple[int, ...]:
    lengths = tuple(values)
    if not lengths:
        raise ValueError("sequence plan requires at least one sequence")
    for index, value in enumerate(lengths):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"sequence {index} loss tokens must be a positive integer")
    return lengths


def _validate_caps(max_loss_tokens: int, max_attention_cells: int, max_sequences: int) -> None:
    for name, value in (
        ("max_loss_tokens", max_loss_tokens),
        ("max_attention_cells", max_attention_cells),
        ("max_sequences", max_sequences),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _next_fit(
    lengths: tuple[int, ...],
    *,
    start: int,
    stop: int,
    max_loss_tokens: int,
    max_attention_cells: int,
    max_sequences: int,
):
    cursor = start
    while cursor < stop:
        first = lengths[cursor]
        if first > max_loss_tokens:
            raise ValueError(f"sequence {cursor} exceeds the loss-token cap")
        if first * first > max_attention_cells:
            raise ValueError(f"sequence {cursor} exceeds the attention-cell cap")
        batch_stop = cursor
        tokens = 0
        cells = 0
        while batch_stop < stop:
            length = lengths[batch_stop]
            next_count = batch_stop - cursor + 1
            if (
                tokens + length > max_loss_tokens
                or cells + length * length > max_attention_cells
                or next_count > max_sequences
            ):
                break
            tokens += length
            cells += length * length
            batch_stop += 1
        yield PlannedWork(cursor, batch_stop, tokens, cells)
        cursor = batch_stop


def _work_dict(item: PlannedWork) -> dict[str, int]:
    return {
        "sequence_start": item.sequence_start,
        "sequence_stop": item.sequence_stop,
        "loss_tokens": item.loss_tokens,
        "attention_cells": item.attention_cells,
    }


def _validate_global_plan(plan: GlobalStepPlan, lengths: tuple[int, ...]) -> None:
    cursor = 0
    for index, step in enumerate(plan.steps):
        if step.sequence_start != cursor or not step.sequence_start < step.sequence_stop <= len(lengths):
            raise ValueError(f"global step {index} does not cover a contiguous sequence range")
        selected = lengths[step.sequence_start : step.sequence_stop]
        if step.loss_tokens != sum(selected) or step.attention_cells != sum(value * value for value in selected):
            raise ValueError(f"global step {index} work totals do not match sequence lengths")
        if (
            step.loss_tokens > plan.max_loss_tokens
            or step.attention_cells > plan.max_attention_cells
            or len(selected) > plan.max_sequences
        ):
            raise ValueError(f"global step {index} exceeds its declared caps")
        cursor = step.sequence_stop
    if cursor != len(lengths):
        raise ValueError("global step plan does not cover every sequence exactly once")
