"""Strict no-authority evidence for the C2/C3 configuration shape probes.

The shape probe measures one physical batching seam.  It is deliberately
diagnostic: its bundle and results cannot become scientific training evidence.
Selection embeds all six signed observations and is rebuilt from them on load,
so a rehashed summary cannot invent a selected cap or derived metric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

from twc.repro.hashing import stable_json_hash


CONFIGURATION_SHAPE_CAPS = (16, 20, 24)
SHAPE_PROBE_WARMUP_STEPS = 5
# The latest worst device slice across the frozen caps is global step 478.
SHAPE_PROBE_MEASURED_STEPS = 480
SHAPE_THROUGHPUT_KNEE_FRACTION = 0.95
SHAPE_MINIMUM_MEMORY_HEADROOM_FRACTION = 0.15
SHAPE_MAXIMUM_INPUT_WAIT_FRACTION = 0.20

_GLOBAL_CAPS = (262_144, 1_073_741_824, 4_096)
_STAGE_PROFILES = {
    "c2": ("m_wide", "m_prior_deep"),
    "c3": ("l_wide", "l_prior_deep"),
}
_STAGE_SEEDS = {"c2": 17, "c3": 29}
_CONDITION_IDS = {
    ("c2", "m_wide"): "c2-wide-challenger-seed17",
    ("c2", "m_prior_deep"): "c2-prior-deep-narrow-seed17",
    ("c3", "l_wide"): "c3-wide-challenger-seed29",
    ("c3", "l_prior_deep"): "c3-prior-deep-narrow-seed29",
}
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
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result) or minimum is not None and result < minimum:
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return result


@dataclass(frozen=True, slots=True)
class ConfigurationShapeProbeResultV1:
    """One fresh-process observation at one profile/cap point."""

    stage: str
    profile_id: str
    condition_id: str
    seed: int
    condition_hash: str
    model_hash: str
    study_plan_hash: str
    comparison_spec_hash: str
    compiler_input_hash: str
    campaign_hash: str
    campaign_wave_id: str
    campaign_attempt_id: str
    execution_matrix_hash: str
    point_execution_path_hash: str
    bundle_hash: str
    data_manifest_hash: str
    train_view_hash: str
    global_step_plan_hash: str
    release_hash: str
    environment_hash: str
    placement_hash: str
    device_identity_hash: str
    process_identity_hash: str
    max_sequences: int
    max_loss_tokens: int
    max_attention_cells: int
    optimizer_step_sequence_cap: int
    warmup_steps: int
    measured_steps: int
    status: str
    failure_class: str | None
    measured_loss_tokens: int
    measured_seconds: float
    throughput_loss_tokens_per_second: float | None
    median_step_seconds: float | None
    input_wait_fraction: float | None
    median_gpu_utilization: float | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    total_memory_bytes: int
    memory_headroom_fraction: float | None
    finite_losses: bool

    def __post_init__(self) -> None:
        profiles = _STAGE_PROFILES.get(self.stage)
        if profiles is None or self.profile_id not in profiles:
            raise ValueError("shape probe stage/profile matrix is invalid")
        expected_seed = _STAGE_SEEDS[self.stage]
        if type(self.seed) is not int or self.seed != expected_seed:
            raise ValueError("shape probe seed differs from the frozen stage seed")
        expected_condition = _CONDITION_IDS[(self.stage, self.profile_id)]
        if self.condition_id != expected_condition:
            raise ValueError("shape probe condition does not match stage/profile/seed")
        if self.campaign_attempt_id != self.condition_id:
            raise ValueError("shape probe campaign attempt does not match condition")
        if self.campaign_wave_id != f"{self.stage}-shape-probe":
            raise ValueError("shape probe campaign wave does not match stage")
        _require_identifier(self.condition_id, "condition_id")
        _require_identifier(self.campaign_wave_id, "campaign_wave_id")
        _require_identifier(self.campaign_attempt_id, "campaign_attempt_id")
        for field in (
            "condition_hash",
            "model_hash",
            "study_plan_hash",
            "comparison_spec_hash",
            "compiler_input_hash",
            "campaign_hash",
            "execution_matrix_hash",
            "point_execution_path_hash",
            "bundle_hash",
            "data_manifest_hash",
            "train_view_hash",
            "global_step_plan_hash",
            "release_hash",
            "environment_hash",
            "placement_hash",
            "device_identity_hash",
            "process_identity_hash",
        ):
            _require_hash(getattr(self, field), field)
        if self.max_sequences not in CONFIGURATION_SHAPE_CAPS:
            raise ValueError("shape probe max_sequences is not predeclared")
        if (
            self.max_loss_tokens,
            self.max_attention_cells,
            self.optimizer_step_sequence_cap,
        ) != _GLOBAL_CAPS:
            raise ValueError("shape probe global optimizer caps are not frozen")
        if (
            self.warmup_steps != SHAPE_PROBE_WARMUP_STEPS
            or self.measured_steps != SHAPE_PROBE_MEASURED_STEPS
        ):
            raise ValueError("shape probe must use the frozen warmup/measurement window")
        if type(self.total_memory_bytes) is not int or self.total_memory_bytes <= 0:
            raise ValueError("total_memory_bytes must be a positive integer")
        if type(self.measured_loss_tokens) is not int or self.measured_loss_tokens < 0:
            raise ValueError("measured_loss_tokens must be a nonnegative integer")
        _finite(self.measured_seconds, "measured_seconds", minimum=0.0)
        if type(self.finite_losses) is not bool:
            raise ValueError("finite_losses must be a boolean")
        if self.status not in {"success", "out_of_memory", "failed"}:
            raise ValueError("shape probe status is invalid")

        optional_metrics = (
            self.throughput_loss_tokens_per_second,
            self.median_step_seconds,
            self.input_wait_fraction,
            self.median_gpu_utilization,
            self.peak_allocated_bytes,
            self.peak_reserved_bytes,
            self.memory_headroom_fraction,
        )
        if self.status == "success":
            if self.failure_class is not None or any(
                value is None for value in optional_metrics
            ):
                raise ValueError("successful shape probe metrics are incomplete")
            if (
                self.measured_loss_tokens <= 0
                or self.measured_seconds <= 0
                or self.finite_losses is not True
            ):
                raise ValueError("successful shape probe has invalid work evidence")
            throughput = _finite(
                self.throughput_loss_tokens_per_second,
                "throughput_loss_tokens_per_second",
                minimum=0.0,
            )
            if throughput <= 0:
                raise ValueError("successful shape probe throughput must be positive")
            expected_throughput = self.measured_loss_tokens / self.measured_seconds
            if not math.isclose(
                throughput, expected_throughput, rel_tol=1e-12, abs_tol=0.0
            ):
                raise ValueError("shape probe throughput does not match its work evidence")
            median_step = _finite(
                self.median_step_seconds, "median_step_seconds", minimum=0.0
            )
            if median_step <= 0 or median_step > (
                2 * self.measured_seconds / self.measured_steps
            ):
                raise ValueError("shape probe median step is inconsistent with measured time")
            input_wait = _finite(
                self.input_wait_fraction, "input_wait_fraction", minimum=0.0
            )
            utilization = _finite(
                self.median_gpu_utilization,
                "median_gpu_utilization",
                minimum=0.0,
            )
            headroom = _finite(
                self.memory_headroom_fraction,
                "memory_headroom_fraction",
                minimum=0.0,
            )
            if input_wait > 1 or utilization > 100 or headroom > 1:
                raise ValueError("successful shape probe fractions are outside their domains")
            if (
                type(self.peak_allocated_bytes) is not int
                or type(self.peak_reserved_bytes) is not int
                or not 0
                <= self.peak_allocated_bytes
                <= self.peak_reserved_bytes
                <= self.total_memory_bytes
            ):
                raise ValueError("successful shape probe memory evidence is invalid")
            expected_headroom = 1 - self.peak_reserved_bytes / self.total_memory_bytes
            if not math.isclose(
                headroom, expected_headroom, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(
                    "shape probe memory headroom does not match reserved memory"
                )
        elif (
            not isinstance(self.failure_class, str)
            or _IDENTIFIER.fullmatch(self.failure_class) is None
            or any(value is not None for value in optional_metrics)
            or self.measured_loss_tokens != 0
            or self.measured_seconds != 0
            or self.finite_losses is not False
        ):
            raise ValueError(
                "failed shape probe must contain only bounded failure evidence"
            )

    @property
    def identity(self) -> str:
        return stable_json_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_shape_probe_result_v1",
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
            **asdict(self),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_hash"] = stable_json_hash(payload)
        return payload


def load_configuration_shape_probe_result(
    payload: Mapping[str, Any],
) -> ConfigurationShapeProbeResultV1:
    fields = set(ConfigurationShapeProbeResultV1.__dataclass_fields__)
    metadata = {
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
        "receipt_hash",
    }
    if set(payload) != fields | metadata:
        raise ValueError("shape probe result fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_configuration_shape_probe_result_v1"
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
    ):
        raise ValueError("shape probe result safety boundary is invalid")
    expected = stable_json_hash(
        {key: value for key, value in payload.items() if key != "receipt_hash"}
    )
    if payload.get("receipt_hash") != expected:
        raise ValueError("shape probe result hash mismatch")
    result = ConfigurationShapeProbeResultV1(
        **{field: payload[field] for field in fields}
    )
    if result.to_dict() != dict(payload):
        raise ValueError("shape probe result is noncanonical")
    return result


def _shared_execution_lineage(
    result: ConfigurationShapeProbeResultV1,
) -> tuple[object, ...]:
    return (
        result.stage,
        result.seed,
        result.study_plan_hash,
        result.comparison_spec_hash,
        result.compiler_input_hash,
        result.campaign_hash,
        result.campaign_wave_id,
        result.data_manifest_hash,
        result.train_view_hash,
        result.global_step_plan_hash,
        result.release_hash,
        result.environment_hash,
        result.placement_hash,
        result.device_identity_hash,
        result.max_loss_tokens,
        result.max_attention_cells,
        result.optimizer_step_sequence_cap,
        result.warmup_steps,
        result.measured_steps,
        result.total_memory_bytes,
    )


def _profile_lineage(result: ConfigurationShapeProbeResultV1) -> tuple[object, ...]:
    return (
        result.stage,
        result.profile_id,
        result.condition_id,
        result.seed,
        result.condition_hash,
        result.model_hash,
        result.campaign_attempt_id,
        result.execution_matrix_hash,
        result.bundle_hash,
    )


def _parse_ordered_results(
    *,
    stage: str,
    results: Sequence[ConfigurationShapeProbeResultV1 | Mapping[str, Any]],
) -> tuple[ConfigurationShapeProbeResultV1, ...]:
    profiles = _STAGE_PROFILES.get(stage)
    if profiles is None:
        raise ValueError("shape selection stage is unsupported")
    parsed = tuple(
        item
        if isinstance(item, ConfigurationShapeProbeResultV1)
        else load_configuration_shape_probe_result(item)
        for item in results
    )
    if any(item.stage != stage for item in parsed):
        raise ValueError("shape selection profile matrix does not match its stage")
    expected_keys = {
        (profile, cap) for profile in profiles for cap in CONFIGURATION_SHAPE_CAPS
    }
    keys = tuple((item.profile_id, item.max_sequences) for item in parsed)
    if (
        len(parsed) != len(expected_keys)
        or len(set(keys)) != len(keys)
        or set(keys) != expected_keys
    ):
        raise ValueError("shape selection requires the exact probe matrix")
    ordered = tuple(
        next(
            item
            for item in parsed
            if (item.profile_id, item.max_sequences) == (profile, cap)
        )
        for profile in profiles
        for cap in CONFIGURATION_SHAPE_CAPS
    )
    shared = _shared_execution_lineage(ordered[0])
    if any(_shared_execution_lineage(item) != shared for item in ordered[1:]):
        raise ValueError("shape probe results do not share shared execution lineage")
    for profile in profiles:
        points = tuple(item for item in ordered if item.profile_id == profile)
        lineage = _profile_lineage(points[0])
        if any(_profile_lineage(item) != lineage for item in points[1:]):
            raise ValueError("shape probe results have a per-profile lineage splice")
    for field in (
        "condition_hash",
        "model_hash",
        "campaign_attempt_id",
        "execution_matrix_hash",
        "bundle_hash",
    ):
        if len({getattr(ordered[index * len(CONFIGURATION_SHAPE_CAPS)], field) for index in range(len(profiles))}) != len(profiles):
            raise ValueError("shape probe widest/deepest per-profile lineage is not distinct")
    if len({item.point_execution_path_hash for item in ordered}) != len(ordered):
        raise ValueError("shape probe point execution paths are not unique")
    if len({item.process_identity_hash for item in ordered}) != len(ordered):
        raise ValueError("shape probe points did not each run in a separate process")
    return ordered


def build_configuration_shape_selection(
    *,
    stage: str,
    results: Sequence[ConfigurationShapeProbeResultV1 | Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the smallest safe common cap within 95% of peak worst-arm throughput."""

    profiles = _STAGE_PROFILES.get(stage)
    if profiles is None:
        raise ValueError("shape selection stage is unsupported")
    ordered = _parse_ordered_results(stage=stage, results=results)
    cap_rows: list[dict[str, Any]] = []
    for cap in CONFIGURATION_SHAPE_CAPS:
        points = tuple(item for item in ordered if item.max_sequences == cap)
        safe = all(
            item.status == "success"
            and item.finite_losses
            and float(item.memory_headroom_fraction)
            >= SHAPE_MINIMUM_MEMORY_HEADROOM_FRACTION
            and float(item.input_wait_fraction)
            <= SHAPE_MAXIMUM_INPUT_WAIT_FRACTION
            for item in points
        )
        cap_rows.append(
            {
                "max_sequences": cap,
                "eligible": safe,
                "worst_profile_throughput": (
                    min(
                        float(item.throughput_loss_tokens_per_second)
                        for item in points
                    )
                    if safe
                    else None
                ),
                "minimum_headroom": (
                    min(float(item.memory_headroom_fraction) for item in points)
                    if safe
                    else None
                ),
                "maximum_input_wait": (
                    max(float(item.input_wait_fraction) for item in points)
                    if safe
                    else None
                ),
            }
        )
    eligible = tuple(row for row in cap_rows if row["eligible"])
    selected: int | None = None
    if eligible:
        peak = max(float(row["worst_profile_throughput"]) for row in eligible)
        knee = peak * SHAPE_THROUGHPUT_KNEE_FRACTION
        selected = min(
            int(row["max_sequences"])
            for row in eligible
            if float(row["worst_profile_throughput"]) >= knee
        )
    first = ordered[0]
    profile_lineage = [
        {
            "profile_id": profile,
            "condition_id": point.condition_id,
            "condition_hash": point.condition_hash,
            "model_hash": point.model_hash,
            "campaign_attempt_id": point.campaign_attempt_id,
            "execution_matrix_hash": point.execution_matrix_hash,
            "bundle_hash": point.bundle_hash,
        }
        for profile in profiles
        for point in (next(row for row in ordered if row.profile_id == profile),)
    ]
    payload: dict[str, Any] = {
        "artifact_kind": "twc_configuration_shape_selection_v1",
        "schema_version": 1,
        "execution_purpose": "diagnostic_shape_probe",
        "stage": stage,
        "seed": first.seed,
        "profile_ids": list(profiles),
        "sequence_caps": list(CONFIGURATION_SHAPE_CAPS),
        "probe_results": [item.to_dict() for item in ordered],
        "result_hashes": [item.identity for item in ordered],
        "study_plan_hash": first.study_plan_hash,
        "comparison_spec_hash": first.comparison_spec_hash,
        "compiler_input_hash": first.compiler_input_hash,
        "campaign_hash": first.campaign_hash,
        "campaign_wave_id": first.campaign_wave_id,
        "profile_lineage": profile_lineage,
        "shared_data_manifest_hash": first.data_manifest_hash,
        "shared_train_view_hash": first.train_view_hash,
        "shared_global_step_plan_hash": first.global_step_plan_hash,
        "release_hash": first.release_hash,
        "environment_hash": first.environment_hash,
        "placement_hash": first.placement_hash,
        "device_identity_hash": first.device_identity_hash,
        "max_loss_tokens": first.max_loss_tokens,
        "max_attention_cells": first.max_attention_cells,
        "optimizer_step_sequence_cap": first.optimizer_step_sequence_cap,
        "warmup_steps": first.warmup_steps,
        "measured_steps": first.measured_steps,
        "total_memory_bytes": first.total_memory_bytes,
        "throughput_knee_fraction": SHAPE_THROUGHPUT_KNEE_FRACTION,
        "minimum_memory_headroom_fraction": SHAPE_MINIMUM_MEMORY_HEADROOM_FRACTION,
        "maximum_input_wait_fraction": SHAPE_MAXIMUM_INPUT_WAIT_FRACTION,
        "cap_evidence": cap_rows,
        "status": "selected" if selected is not None else "blocked",
        "selected_max_sequences": selected,
        "scientific_campaign_eligible": False,
        "scientific_result_eligible": False,
        "production_unlock_eligible": False,
        "f1_unlock_eligible": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "tokens_serialized": False,
        "secrets_serialized": False,
    }
    payload["selection_hash"] = stable_json_hash(payload)
    return payload


def load_configuration_shape_selection(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "artifact_kind",
        "schema_version",
        "execution_purpose",
        "stage",
        "seed",
        "profile_ids",
        "sequence_caps",
        "probe_results",
        "result_hashes",
        "study_plan_hash",
        "comparison_spec_hash",
        "compiler_input_hash",
        "campaign_hash",
        "campaign_wave_id",
        "profile_lineage",
        "shared_data_manifest_hash",
        "shared_train_view_hash",
        "shared_global_step_plan_hash",
        "release_hash",
        "environment_hash",
        "placement_hash",
        "device_identity_hash",
        "max_loss_tokens",
        "max_attention_cells",
        "optimizer_step_sequence_cap",
        "warmup_steps",
        "measured_steps",
        "total_memory_bytes",
        "throughput_knee_fraction",
        "minimum_memory_headroom_fraction",
        "maximum_input_wait_fraction",
        "cap_evidence",
        "status",
        "selected_max_sequences",
        "scientific_campaign_eligible",
        "scientific_result_eligible",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "raw_text_serialized",
        "raw_url_serialized",
        "tokens_serialized",
        "secrets_serialized",
        "selection_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("shape selection fields are invalid")
    expected_hash = stable_json_hash(
        {key: value for key, value in payload.items() if key != "selection_hash"}
    )
    if payload.get("selection_hash") != expected_hash:
        raise ValueError("shape selection hash mismatch")
    if (
        payload.get("artifact_kind") != "twc_configuration_shape_selection_v1"
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
        or not isinstance(payload.get("probe_results"), list)
    ):
        raise ValueError("shape selection safety boundary is invalid")
    expected = build_configuration_shape_selection(
        stage=str(payload.get("stage")), results=tuple(payload["probe_results"])
    )
    if expected != dict(payload):
        raise ValueError("shape selection cannot be canonically rederived")
    return expected
