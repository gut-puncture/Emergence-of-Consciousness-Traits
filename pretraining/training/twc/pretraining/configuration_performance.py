"""Live, replayable performance gates for C2/C3 configuration attempts.

The generic R7 guard remains responsible for spend, storage, and liveness.  This
module adds only the stricter physical-performance contract derived from the
already-selected configuration shape.  It deliberately permits no live tuning:
changing the selected execution path would invalidate the evidence that chose
it, so a violation stops the attempt and requires a fresh reviewed launch.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from twc.pretraining.configuration_study_execution import (
    ConfigurationExecutionPathReceiptV1,
    load_configuration_execution_path_receipt,
)
from twc.repro.hashing import stable_json_hash


_ZERO_HASH = "0" * 64
_HASH_CHARACTERS = frozenset("0123456789abcdef")
_THROUGHPUT_FRACTION = 0.70
_UTILIZATION_FRACTION = 0.70
_UTILIZATION_ABSOLUTE_FLOOR_PERCENT = 50.0
_CHECKPOINT_LATENCY_CAP_SECONDS = 60.0
_STOP_LATENCY_CAP_SECONDS = 60.0
# Long diagnostic coverage must not delay live performance enforcement.
_LIVE_PERFORMANCE_WINDOW_STEPS = 10


def _require_hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not set(value) <= _HASH_CHARACTERS
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} is below its minimum")
    return result


def _load_path(
    value: ConfigurationExecutionPathReceiptV1 | Mapping[str, Any],
) -> ConfigurationExecutionPathReceiptV1:
    if isinstance(value, ConfigurationExecutionPathReceiptV1):
        return value
    return load_configuration_execution_path_receipt(value)


def build_configuration_performance_policy(
    execution_path: ConfigurationExecutionPathReceiptV1 | Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze live thresholds from the exact selected-cap shape evidence."""

    path = _load_path(execution_path)
    selection = path.to_dict()["shape_selection"]
    selected_cap = path.microbatch_max_sequences
    selected_row = next(
        (
            row
            for row in selection["cap_evidence"]
            if row.get("max_sequences") == selected_cap
        ),
        None,
    )
    if not isinstance(selected_row, Mapping) or selected_row.get("eligible") is not True:
        raise ValueError("performance policy requires eligible selected-cap evidence")
    forecast = _finite(
        selected_row.get("worst_profile_throughput"),
        "selected-cap throughput forecast",
        minimum=0.0,
    )
    if forecast <= 0:
        raise ValueError("selected-cap throughput forecast must be positive")
    selected_probe_rows = tuple(
        row
        for row in selection["probe_results"]
        if row.get("max_sequences") == selected_cap and row.get("status") == "success"
    )
    if not selected_probe_rows:
        raise ValueError("performance policy lacks successful selected-cap probes")
    utilization_ceiling = max(
        _finite(
            row.get("median_gpu_utilization"),
            "selected-cap measured utilization",
            minimum=0.0,
        )
        for row in selected_probe_rows
    )
    if utilization_ceiling < _UTILIZATION_ABSOLUTE_FLOOR_PERCENT:
        raise ValueError(
            "selected shape utilization ceiling is below the live-policy floor"
        )
    total_memory = selection.get("total_memory_bytes")
    warmup_steps = selection.get("warmup_steps")
    if type(total_memory) is not int or total_memory <= 0:
        raise ValueError("selected shape total memory is invalid")
    if type(warmup_steps) is not int or warmup_steps <= 0:
        raise ValueError("selected shape warmup window is invalid")
    payload: dict[str, Any] = {
        "artifact_kind": "twc_configuration_performance_policy_v1",
        "schema_version": 1,
        "stage": path.stage,
        "execution_path_hash": path.identity,
        "shape_selection_hash": path.shape_selection_hash,
        "device_identity_hash": path.device_identity_hash,
        "selected_microbatch_max_sequences": selected_cap,
        "selected_cap_worst_profile_throughput": forecast,
        "minimum_sustained_throughput": forecast * _THROUGHPUT_FRACTION,
        "throughput_threshold_fraction": _THROUGHPUT_FRACTION,
        "measurement_warmup_steps": warmup_steps,
        "measurement_window_steps": _LIVE_PERFORMANCE_WINDOW_STEPS,
        "minimum_observed_train_steps": warmup_steps + _LIVE_PERFORMANCE_WINDOW_STEPS,
        "total_device_memory_bytes": total_memory,
        "minimum_memory_headroom_fraction": 0.15,
        "maximum_input_wait_fraction": 0.20,
        "maximum_checkpoint_latency_seconds": _CHECKPOINT_LATENCY_CAP_SECONDS,
        "maximum_stop_latency_seconds": _STOP_LATENCY_CAP_SECONDS,
        "measured_shape_utilization_ceiling_percent": utilization_ceiling,
        "minimum_gpu_utilization_percent": max(
            _UTILIZATION_ABSOLUTE_FLOOR_PERCENT,
            utilization_ceiling * _UTILIZATION_FRACTION,
        ),
        "utilization_threshold_fraction": _UTILIZATION_FRACTION,
        "utilization_absolute_floor_percent": _UTILIZATION_ABSOLUTE_FLOOR_PERCENT,
        "utilization_threshold_rule": (
            "max_50_percent_or_70_percent_of_selected_cap_measured_ceiling_v1"
        ),
        "maximum_execution_only_adjustments": 0,
        "execution_adjustment_policy": "none_selected_path_is_immutable_v1",
        "scientific_result_eligible": False,
        "production_unlock_eligible": False,
        "f1_unlock_eligible": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "tokens_serialized": False,
        "secrets_serialized": False,
    }
    payload["policy_hash"] = stable_json_hash(payload)
    return load_configuration_performance_policy(
        payload, expected_execution_path_hash=path.identity
    )


_POLICY_FIELDS = {
    "artifact_kind",
    "schema_version",
    "stage",
    "execution_path_hash",
    "shape_selection_hash",
    "device_identity_hash",
    "selected_microbatch_max_sequences",
    "selected_cap_worst_profile_throughput",
    "minimum_sustained_throughput",
    "throughput_threshold_fraction",
    "measurement_warmup_steps",
    "measurement_window_steps",
    "minimum_observed_train_steps",
    "total_device_memory_bytes",
    "minimum_memory_headroom_fraction",
    "maximum_input_wait_fraction",
    "maximum_checkpoint_latency_seconds",
    "maximum_stop_latency_seconds",
    "measured_shape_utilization_ceiling_percent",
    "minimum_gpu_utilization_percent",
    "utilization_threshold_fraction",
    "utilization_absolute_floor_percent",
    "utilization_threshold_rule",
    "maximum_execution_only_adjustments",
    "execution_adjustment_policy",
    "scientific_result_eligible",
    "production_unlock_eligible",
    "f1_unlock_eligible",
    "raw_text_serialized",
    "raw_url_serialized",
    "tokens_serialized",
    "secrets_serialized",
    "policy_hash",
}


def load_configuration_performance_policy(
    payload: Mapping[str, Any], *, expected_execution_path_hash: str | None = None
) -> dict[str, Any]:
    if set(payload) != _POLICY_FIELDS:
        raise ValueError("configuration performance policy fields are invalid")
    expected_hash = stable_json_hash(
        {key: value for key, value in payload.items() if key != "policy_hash"}
    )
    false_fields = (
        "scientific_result_eligible",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "raw_text_serialized",
        "raw_url_serialized",
        "tokens_serialized",
        "secrets_serialized",
    )
    if (
        payload.get("artifact_kind") != "twc_configuration_performance_policy_v1"
        or payload.get("schema_version") != 1
        or payload.get("stage") not in {"c2", "c3"}
        or payload.get("policy_hash") != expected_hash
        or any(payload.get(field) is not False for field in false_fields)
        or payload.get("maximum_execution_only_adjustments") != 0
        or payload.get("execution_adjustment_policy")
        != "none_selected_path_is_immutable_v1"
        or payload.get("utilization_threshold_rule")
        != "max_50_percent_or_70_percent_of_selected_cap_measured_ceiling_v1"
    ):
        raise ValueError("configuration performance policy boundary is invalid")
    for field in (
        "execution_path_hash",
        "shape_selection_hash",
        "device_identity_hash",
        "policy_hash",
    ):
        _require_hash(payload.get(field), f"performance policy {field}")
    if expected_execution_path_hash is not None and payload["execution_path_hash"] != _require_hash(
        expected_execution_path_hash, "expected execution path hash"
    ):
        raise ValueError("configuration performance policy execution path mismatch")
    for field in (
        "selected_microbatch_max_sequences",
        "measurement_warmup_steps",
        "measurement_window_steps",
        "minimum_observed_train_steps",
        "total_device_memory_bytes",
    ):
        if type(payload.get(field)) is not int or payload[field] <= 0:
            raise ValueError(f"configuration performance policy {field} is invalid")
    numeric = (
        "selected_cap_worst_profile_throughput",
        "minimum_sustained_throughput",
        "throughput_threshold_fraction",
        "minimum_memory_headroom_fraction",
        "maximum_input_wait_fraction",
        "maximum_checkpoint_latency_seconds",
        "maximum_stop_latency_seconds",
        "measured_shape_utilization_ceiling_percent",
        "minimum_gpu_utilization_percent",
        "utilization_threshold_fraction",
        "utilization_absolute_floor_percent",
    )
    for field in numeric:
        _finite(payload.get(field), f"configuration performance policy {field}")
    if (
        payload["throughput_threshold_fraction"] != _THROUGHPUT_FRACTION
        or payload["minimum_sustained_throughput"]
        != payload["selected_cap_worst_profile_throughput"] * _THROUGHPUT_FRACTION
        or payload["minimum_observed_train_steps"]
        != payload["measurement_warmup_steps"]
        + payload["measurement_window_steps"]
        or payload["measurement_window_steps"] != _LIVE_PERFORMANCE_WINDOW_STEPS
        or payload["minimum_memory_headroom_fraction"] != 0.15
        or payload["maximum_input_wait_fraction"] != 0.20
        or payload["maximum_checkpoint_latency_seconds"]
        != _CHECKPOINT_LATENCY_CAP_SECONDS
        or payload["maximum_stop_latency_seconds"] != _STOP_LATENCY_CAP_SECONDS
        or payload["utilization_threshold_fraction"] != _UTILIZATION_FRACTION
        or payload["utilization_absolute_floor_percent"]
        != _UTILIZATION_ABSOLUTE_FLOOR_PERCENT
        or payload["minimum_gpu_utilization_percent"]
        != max(
            _UTILIZATION_ABSOLUTE_FLOOR_PERCENT,
            payload["measured_shape_utilization_ceiling_percent"]
            * _UTILIZATION_FRACTION,
        )
        or not 0 < payload["minimum_memory_headroom_fraction"] <= 1
        or not 0 <= payload["maximum_input_wait_fraction"] <= 1
        or not 0 < payload["minimum_gpu_utilization_percent"] <= 100
    ):
        raise ValueError("configuration performance policy thresholds are invalid")
    return dict(payload)


class ConfigurationPerformanceWindowAccumulator:
    """Collect fixed, shape-comparable post-warmup training windows.

    Cumulative worker telemetry remains available for operational diagnosis.
    This accumulator owns only the fixed-size windows used by the live gate and
    rotates after every complete window so later sustained regressions remain
    observable without allowing compile warmup to poison the comparison.
    """

    def __init__(self, policy: Mapping[str, Any]) -> None:
        loaded = load_configuration_performance_policy(policy)
        self._warmup_steps = int(loaded["measurement_warmup_steps"])
        self._window_steps = int(loaded["measurement_window_steps"])
        self._window_start_step = self._warmup_steps
        self._step_seconds: list[float] = []
        self._input_seconds = 0.0
        self._loss_tokens = 0

    def starts_at_step(self, next_step_index: int) -> bool:
        if type(next_step_index) is not int or next_step_index < 0:
            raise ValueError("performance window step index is invalid")
        return (
            next_step_index == self._window_start_step
            and not self._step_seconds
        )

    def completes_at_step(self, completed_step: int) -> bool:
        if type(completed_step) is not int or completed_step < 0:
            raise ValueError("performance window completed step is invalid")
        return completed_step == self._window_start_step + self._window_steps

    def observe(
        self,
        *,
        completed_step: int,
        loss_tokens: int,
        step_seconds: float,
        input_seconds: float,
        utilization_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if type(completed_step) is not int or completed_step <= 0:
            raise ValueError("performance window completed step is invalid")
        if type(loss_tokens) is not int or loss_tokens <= 0:
            raise ValueError("performance window loss tokens are invalid")
        elapsed = _finite(step_seconds, "performance window step seconds", minimum=0.0)
        waited = _finite(input_seconds, "performance window input seconds", minimum=0.0)
        if elapsed <= 0 or waited > elapsed:
            raise ValueError("performance window step timing is invalid")
        if completed_step <= self._warmup_steps:
            if utilization_evidence is not None:
                raise ValueError("warmup step cannot complete a performance window")
            return None
        expected_step = self._window_start_step + len(self._step_seconds) + 1
        if completed_step != expected_step:
            raise ValueError("performance window steps are noncontiguous")
        self._step_seconds.append(elapsed)
        self._input_seconds += waited
        self._loss_tokens += loss_tokens
        if len(self._step_seconds) < self._window_steps:
            if utilization_evidence is not None:
                raise ValueError("partial performance window has utilization evidence")
            return None
        if utilization_evidence is None:
            raise ValueError("complete performance window lacks utilization evidence")
        utilization_status = utilization_evidence.get("gpu_utilization_status")
        utilization = utilization_evidence.get("median_gpu_utilization")
        samples = utilization_evidence.get("gpu_utilization_samples")
        if not isinstance(utilization_status, str):
            raise ValueError("performance window utilization status is invalid")
        if utilization is not None:
            _finite(utilization, "performance window utilization", minimum=0.0)
        if type(samples) is not int or samples < 0:
            raise ValueError("performance window utilization samples are invalid")
        measured_seconds = sum(self._step_seconds)
        payload = {
            "performance_window_start_step": self._window_start_step,
            "performance_window_end_step": completed_step,
            "performance_window_train_steps": self._window_steps,
            "performance_window_loss_tokens": self._loss_tokens,
            "performance_window_seconds": measured_seconds,
            "performance_window_throughput_loss_tokens_per_second": (
                self._loss_tokens / measured_seconds
            ),
            "performance_window_input_wait_fraction": (
                self._input_seconds / measured_seconds
            ),
            "performance_window_gpu_utilization_status": utilization_status,
            "performance_window_median_gpu_utilization": utilization,
            "performance_window_gpu_utilization_samples": samples,
        }
        self._window_start_step = completed_step
        self._step_seconds.clear()
        self._input_seconds = 0.0
        self._loss_tokens = 0
        return payload


def _telemetry_hash(telemetry: Mapping[str, Any]) -> str:
    supplied = _require_hash(telemetry.get("telemetry_hash"), "telemetry hash")
    expected = stable_json_hash(
        {key: value for key, value in telemetry.items() if key != "telemetry_hash"}
    )
    if supplied != expected:
        raise ValueError("configuration performance telemetry hash mismatch")
    return supplied


def _metric(
    telemetry: Mapping[str, Any], field: str, violations: list[str]
) -> float | None:
    value = telemetry.get(field)
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        violations.append(f"non_finite_{field}")
        return None
    return float(value)


def build_configuration_performance_observation(
    *,
    policy: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    previous_observation_hash: str,
    sequence: int,
    observed_at_epoch: float,
) -> dict[str, Any]:
    loaded_policy = load_configuration_performance_policy(policy)
    if type(sequence) is not int or sequence <= 0:
        raise ValueError("configuration performance observation sequence is invalid")
    previous = _require_hash(previous_observation_hash, "previous observation hash")
    if sequence == 1 and previous != _ZERO_HASH:
        raise ValueError("first configuration performance observation must start its chain")
    observed = _finite(observed_at_epoch, "performance observed_at_epoch", minimum=0.0)
    telemetry_payload = dict(telemetry)
    telemetry_identity = _telemetry_hash(telemetry_payload)
    run_identity = _require_hash(telemetry_payload.get("run_identity"), "telemetry run identity")
    attempt_hash = _require_hash(telemetry_payload.get("attempt_hash"), "telemetry attempt hash")
    phase = telemetry_payload.get("phase")
    step = telemetry_payload.get("step")
    if not isinstance(phase, str) or type(step) is not int or step < 0:
        raise ValueError("configuration performance telemetry phase/step is invalid")
    heartbeat = _finite(
        telemetry_payload.get("heartbeat_epoch"), "telemetry heartbeat epoch", minimum=0.0
    )
    if heartbeat > observed + 5:
        raise ValueError("configuration performance telemetry is future-dated")

    evaluated = phase in {"train", "stopped", "failed", "completed"} and step >= int(
        loaded_policy["minimum_observed_train_steps"]
    )
    violations: list[str] = []
    memory_headroom: float | None = None
    if evaluated:
        window_start = telemetry_payload.get("performance_window_start_step")
        window_end = telemetry_payload.get("performance_window_end_step")
        window_steps = telemetry_payload.get("performance_window_train_steps")
        window_loss_tokens = telemetry_payload.get("performance_window_loss_tokens")
        window_samples = telemetry_payload.get(
            "performance_window_gpu_utilization_samples"
        )
        required_window_fields = {
            "performance_window_start_step",
            "performance_window_end_step",
            "performance_window_train_steps",
            "performance_window_loss_tokens",
            "performance_window_seconds",
            "performance_window_throughput_loss_tokens_per_second",
            "performance_window_input_wait_fraction",
            "performance_window_gpu_utilization_status",
            "performance_window_median_gpu_utilization",
            "performance_window_gpu_utilization_samples",
        }
        warmup_steps = int(loaded_policy["measurement_warmup_steps"])
        measurement_steps = int(loaded_policy["measurement_window_steps"])
        complete_window = bool(
            required_window_fields <= set(telemetry_payload)
            and type(window_start) is int
            and type(window_end) is int
            and type(window_steps) is int
            and type(window_loss_tokens) is int
            and type(window_samples) is int
            and window_start >= warmup_steps
            and (window_start - warmup_steps) % measurement_steps == 0
            and window_end == window_start + measurement_steps
            and 0 <= step - window_end < measurement_steps
            and window_steps == measurement_steps
            and window_loss_tokens > 0
            and window_samples > 0
        )
        if not complete_window:
            violations.append("performance_window_incomplete")
        throughput = _metric(
            telemetry_payload,
            "performance_window_throughput_loss_tokens_per_second",
            violations,
        )
        input_wait = _metric(
            telemetry_payload, "performance_window_input_wait_fraction", violations
        )
        measured_seconds = _metric(
            telemetry_payload, "performance_window_seconds", violations
        )
        checkpoint_latency = _metric(
            telemetry_payload, "checkpoint_duration_seconds", violations
        )
        peak_reserved = _metric(
            telemetry_payload, "gpu_peak_reserved_memory_bytes", violations
        )
        utilization = _metric(
            telemetry_payload,
            "performance_window_median_gpu_utilization",
            violations,
        )
        stop_latency = _metric(telemetry_payload, "stop_latency_seconds", violations)
        evaluation_loss = telemetry_payload.get("evaluation_loss")
        if evaluation_loss is not None and (
            type(evaluation_loss) not in {int, float}
            or not math.isfinite(float(evaluation_loss))
            or float(evaluation_loss) < 0
        ):
            violations.append("non_finite_evaluation_loss")
        if throughput is not None and throughput < loaded_policy["minimum_sustained_throughput"]:
            violations.append("throughput_below_floor")
        if measured_seconds is not None and measured_seconds <= 0:
            violations.append("performance_window_seconds_nonpositive")
        if input_wait is not None and not 0 <= input_wait <= loaded_policy["maximum_input_wait_fraction"]:
            violations.append("input_wait_above_cap")
        if peak_reserved is not None:
            memory_headroom = 1 - peak_reserved / loaded_policy["total_device_memory_bytes"]
            if not math.isfinite(memory_headroom) or memory_headroom < loaded_policy["minimum_memory_headroom_fraction"]:
                violations.append("memory_headroom_below_floor")
        if (
            telemetry_payload.get("performance_window_gpu_utilization_status")
            != "measured"
        ):
            violations.append("utilization_unavailable")
        if utilization is not None and utilization < loaded_policy["minimum_gpu_utilization_percent"]:
            violations.append("utilization_below_floor")
        if checkpoint_latency is not None and checkpoint_latency > loaded_policy["maximum_checkpoint_latency_seconds"]:
            violations.append("checkpoint_latency_above_cap")
        if phase == "stopped" and stop_latency is not None and stop_latency > loaded_policy["maximum_stop_latency_seconds"]:
            violations.append("stop_latency_above_cap")
    decision = "stop" if violations else ("pass" if evaluated else "not_evaluated")
    payload: dict[str, Any] = {
        "artifact_kind": "twc_configuration_performance_observation_v1",
        "schema_version": 1,
        "sequence": sequence,
        "policy_hash": loaded_policy["policy_hash"],
        "execution_path_hash": loaded_policy["execution_path_hash"],
        "run_identity": run_identity,
        "attempt_hash": attempt_hash,
        "observed_at_epoch": observed,
        "telemetry_hash": telemetry_identity,
        "telemetry": telemetry_payload,
        "evaluated": evaluated,
        "memory_headroom_fraction": memory_headroom,
        "violation_codes": violations,
        "decision": decision,
        "previous_observation_hash": previous,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "tokens_serialized": False,
        "secrets_serialized": False,
    }
    payload["observation_hash"] = stable_json_hash(payload)
    return payload


def _load_observation(
    payload: Mapping[str, Any], *, policy: Mapping[str, Any], sequence: int, previous: str
) -> dict[str, Any]:
    expected_fields = {
        "artifact_kind", "schema_version", "sequence", "policy_hash",
        "execution_path_hash", "run_identity", "attempt_hash", "observed_at_epoch",
        "telemetry_hash", "telemetry", "evaluated", "memory_headroom_fraction",
        "violation_codes", "decision", "previous_observation_hash",
        "raw_text_serialized", "raw_url_serialized", "tokens_serialized",
        "secrets_serialized", "observation_hash",
    }
    if set(payload) != expected_fields or not isinstance(payload.get("telemetry"), Mapping):
        raise ValueError("configuration performance observation fields are invalid")
    if any(
        payload.get(field) is not False
        for field in (
            "raw_text_serialized", "raw_url_serialized", "tokens_serialized",
            "secrets_serialized",
        )
    ):
        raise ValueError("configuration performance observation boundary is invalid")
    rebuilt = build_configuration_performance_observation(
        policy=policy,
        telemetry=payload["telemetry"],
        previous_observation_hash=previous,
        sequence=sequence,
        observed_at_epoch=payload.get("observed_at_epoch"),
    )
    if rebuilt != dict(payload):
        raise ValueError("configuration performance observation is noncanonical")
    return rebuilt


def load_configuration_performance_ledger(
    path: Path, *, policy: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    load_configuration_performance_policy(policy)
    if path.is_symlink() or not path.is_file():
        raise ValueError("configuration performance ledger must be a regular file")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("configuration performance ledger is unreadable") from error
    rows: list[dict[str, Any]] = []
    previous = _ZERO_HASH
    for sequence, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError("configuration performance ledger contains a blank row")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("configuration performance ledger is malformed") from error
        if not isinstance(payload, Mapping):
            raise ValueError("configuration performance ledger rows must be objects")
        row = _load_observation(
            payload, policy=policy, sequence=sequence, previous=previous
        )
        if rows and row["observed_at_epoch"] < rows[-1]["observed_at_epoch"]:
            raise ValueError("configuration performance observation time moved backwards")
        rows.append(row)
        previous = row["observation_hash"]
    if not rows:
        raise ValueError("configuration performance ledger is empty")
    return tuple(rows)


def append_configuration_performance_observation(
    *,
    path: Path,
    policy: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    observed_at_epoch: float,
) -> dict[str, Any]:
    """Append one fsynced observation, deduplicating an unchanged telemetry head."""

    loaded_policy = load_configuration_performance_policy(policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise ValueError("configuration performance ledger must not be a symlink")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        rows: tuple[dict[str, Any], ...]
        if lines:
            rows = load_configuration_performance_ledger(path, policy=loaded_policy)
            telemetry_identity = _telemetry_hash(telemetry)
            matching = tuple(row for row in rows if row["telemetry_hash"] == telemetry_identity)
            if matching:
                return matching[-1]
        else:
            rows = ()
        row = build_configuration_performance_observation(
            policy=loaded_policy,
            telemetry=telemetry,
            previous_observation_hash=(
                rows[-1]["observation_hash"] if rows else _ZERO_HASH
            ),
            sequence=len(rows) + 1,
            observed_at_epoch=observed_at_epoch,
        )
        data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        os.lseek(descriptor, 0, os.SEEK_END)
        if os.write(descriptor, data) != len(data):
            raise OSError("short configuration performance ledger write")
        os.fsync(descriptor)
        return row
    finally:
        os.close(descriptor)


def validate_configuration_performance_closeout(
    *,
    policy: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    expected_run_identity: str,
    expected_attempt_hash: str,
    completed: bool,
) -> dict[str, Any]:
    """Replay a copied ledger and issue a no-unlock local validation receipt."""

    loaded_policy = load_configuration_performance_policy(policy)
    run_identity = _require_hash(expected_run_identity, "expected run identity")
    attempt_hash = _require_hash(expected_attempt_hash, "expected attempt hash")
    if not isinstance(completed, bool) or not observations:
        raise ValueError("configuration performance closeout evidence is invalid")
    previous = _ZERO_HASH
    loaded_rows: list[dict[str, Any]] = []
    for sequence, supplied in enumerate(observations, start=1):
        row = _load_observation(
            supplied, policy=loaded_policy, sequence=sequence, previous=previous
        )
        if row["run_identity"] != run_identity or row["attempt_hash"] != attempt_hash:
            raise ValueError("configuration performance ledger run/attempt mismatch")
        loaded_rows.append(row)
        previous = row["observation_hash"]
    passed_rows = tuple(row for row in loaded_rows if row["decision"] == "pass")
    stopped_rows = tuple(row for row in loaded_rows if row["decision"] == "stop")
    terminal_phase = loaded_rows[-1]["telemetry"].get("phase")
    expected_terminal_phases = {"completed"} if completed else {"stopped", "failed"}
    if terminal_phase not in expected_terminal_phases:
        raise ValueError("configuration performance ledger lacks terminal telemetry")
    if completed and (not passed_rows or stopped_rows):
        raise ValueError("completed attempt violates configuration performance policy")
    payload: dict[str, Any] = {
        "artifact_kind": "twc_configuration_performance_closeout_v1",
        "schema_version": 1,
        "policy_hash": loaded_policy["policy_hash"],
        "execution_path_hash": loaded_policy["execution_path_hash"],
        "run_identity": run_identity,
        "attempt_hash": attempt_hash,
        "observation_count": len(loaded_rows),
        "evaluated_observation_count": sum(
            1 for row in loaded_rows if row["evaluated"]
        ),
        "ledger_head_hash": loaded_rows[-1]["observation_hash"],
        "violation_observation_hashes": [
            row["observation_hash"] for row in stopped_rows
        ],
        "performance_passed": not stopped_rows and bool(passed_rows),
        "completed_attempt_eligible": completed and not stopped_rows and bool(passed_rows),
        "scientific_result_eligible": False,
        "production_unlock_eligible": False,
        "f1_unlock_eligible": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "tokens_serialized": False,
        "secrets_serialized": False,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return load_configuration_performance_closeout(payload)


_CLOSEOUT_FIELDS = {
    "artifact_kind", "schema_version", "policy_hash", "execution_path_hash",
    "run_identity", "attempt_hash", "observation_count",
    "evaluated_observation_count", "ledger_head_hash",
    "violation_observation_hashes", "performance_passed",
    "completed_attempt_eligible", "scientific_result_eligible",
    "production_unlock_eligible", "f1_unlock_eligible", "raw_text_serialized",
    "raw_url_serialized", "tokens_serialized", "secrets_serialized",
    "receipt_hash",
}


def load_configuration_performance_closeout(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if set(payload) != _CLOSEOUT_FIELDS:
        raise ValueError("configuration performance closeout fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_configuration_performance_closeout_v1"
        or payload.get("schema_version") != 1
        or payload.get("receipt_hash")
        != stable_json_hash(
            {key: value for key, value in payload.items() if key != "receipt_hash"}
        )
        or any(
            payload.get(field) is not False
            for field in (
                "scientific_result_eligible", "production_unlock_eligible",
                "f1_unlock_eligible", "raw_text_serialized", "raw_url_serialized",
                "tokens_serialized", "secrets_serialized",
            )
        )
    ):
        raise ValueError("configuration performance closeout boundary is invalid")
    for field in (
        "policy_hash", "execution_path_hash", "run_identity", "attempt_hash",
        "ledger_head_hash", "receipt_hash",
    ):
        _require_hash(payload.get(field), f"performance closeout {field}")
    if type(payload.get("observation_count")) is not int or payload["observation_count"] <= 0:
        raise ValueError("configuration performance closeout counts are invalid")
    if (
        type(payload.get("evaluated_observation_count")) is not int
        or payload["evaluated_observation_count"] < 0
    ):
        raise ValueError("configuration performance closeout counts are invalid")
    if payload["evaluated_observation_count"] > payload["observation_count"]:
        raise ValueError("configuration performance closeout counts are inconsistent")
    violations = payload.get("violation_observation_hashes")
    if not isinstance(violations, list) or len(violations) != len(set(violations)):
        raise ValueError("configuration performance closeout violations are invalid")
    for index, value in enumerate(violations):
        _require_hash(value, f"performance closeout violation[{index}]")
    if (
        type(payload.get("performance_passed")) is not bool
        or type(payload.get("completed_attempt_eligible")) is not bool
        or payload["completed_attempt_eligible"] and not payload["performance_passed"]
        or payload["performance_passed"] and bool(violations)
    ):
        raise ValueError("configuration performance closeout verdict is invalid")
    return dict(payload)
