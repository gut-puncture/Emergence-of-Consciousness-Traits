"""Execution binding and worker contract for configuration-study v2.

Scientific meaning lives in :mod:`twc.pretraining.configuration_study`.
Campaign/release placement lives here and may change without rewriting the
scientific contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import re
from typing import Any, Mapping

from twc.pretraining.campaign import AttemptReceipt, CampaignPlan
from twc.pretraining.configuration_assessment import ExperimentResultV3
from twc.pretraining.configuration_shape_probe import (
    load_configuration_shape_selection,
)
from twc.pretraining.configuration_study import (
    ExperimentContractV2,
    load_experiment_contract_v2,
)
from twc.repro.hashing import stable_json_hash


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_EXECUTION_PATH_GLOBAL_STEP_MAX_LOSS_TOKENS = 262_144
_EXECUTION_PATH_GLOBAL_STEP_MAX_ATTENTION_CELLS = 1_073_741_824
_EXECUTION_PATH_GLOBAL_STEP_MAX_SEQUENCES = 4_096
_EXECUTION_PATH_CONTEXT_LENGTH = 4_096
_SHAPE_PROBE_PROFILES = {
    "c2": ("m_wide", "m_prior_deep"),
    "c3": ("l_wide", "l_prior_deep"),
}
_SHAPE_PROBE_SEQUENCE_CAPS = (16, 20, 24)


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def _require_hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _canonical_shape_selection_json(selection: Mapping[str, Any]) -> str:
    return json.dumps(selection, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _execution_path_values(
    *, stage: str, selection: Mapping[str, Any]
) -> dict[str, Any]:
    if stage not in {"c2", "c3"}:
        raise ValueError("configuration execution path stage is unsupported")
    if selection.get("stage") != stage:
        raise ValueError("configuration execution path stage/scale mismatch")
    if selection.get("status") != "selected":
        raise ValueError("configuration execution path requires a selected shape selection")
    if (
        selection.get("max_loss_tokens"),
        selection.get("max_attention_cells"),
        selection.get("optimizer_step_sequence_cap"),
    ) != (
        _EXECUTION_PATH_GLOBAL_STEP_MAX_LOSS_TOKENS,
        _EXECUTION_PATH_GLOBAL_STEP_MAX_ATTENTION_CELLS,
        _EXECUTION_PATH_GLOBAL_STEP_MAX_SEQUENCES,
    ):
        raise ValueError(
            "configuration execution path shape selection global caps mismatch"
        )
    selected = selection.get("selected_max_sequences")
    if type(selected) is not int or selected <= 0:
        raise ValueError("configuration execution path selected microbatch cap is invalid")
    return {
        "stage": stage,
        "shape_selection_hash": selection["selection_hash"],
        "study_plan_hash": selection["study_plan_hash"],
        "comparison_spec_hash": selection["comparison_spec_hash"],
        "compiler_input_hash": selection["compiler_input_hash"],
        "data_manifest_hash": selection["shared_data_manifest_hash"],
        "train_view_hash": selection["shared_train_view_hash"],
        "global_step_plan_hash": selection["shared_global_step_plan_hash"],
        "release_hash": selection["release_hash"],
        "environment_hash": selection["environment_hash"],
        "placement_hash": selection["placement_hash"],
        "device_identity_hash": selection["device_identity_hash"],
        "device_type": "cuda",
        "is_spot": False,
        "microbatch_max_sequences": selected,
        "gradient_accumulation_policy": (
            "accumulate_all_device_slices_for_exact_global_step_v1"
        ),
        "global_step_max_loss_tokens": _EXECUTION_PATH_GLOBAL_STEP_MAX_LOSS_TOKENS,
        "global_step_max_attention_cells": (
            _EXECUTION_PATH_GLOBAL_STEP_MAX_ATTENTION_CELLS
        ),
        "global_step_max_sequences": _EXECUTION_PATH_GLOBAL_STEP_MAX_SEQUENCES,
        "context_length": _EXECUTION_PATH_CONTEXT_LENGTH,
        "prefetch_batches": 1,
        "pin_memory": True,
        "nonblocking_device_transfer": True,
        "read_mode": "local_mmap",
        "removable_data": False,
        "precision": "bf16",
        "compile_mode": "default",
        "allow_tf32": True,
        "optimizer_backend": "fused",
        "attention_backend": "native_varlen",
        "attention_num_splits": "auto",
        "loss_backend": "chunked_projection",
    }


_EXECUTION_PATH_FIELD_LABELS = {
    "stage": "stage",
    "shape_selection_hash": "shape selection hash",
    "study_plan_hash": "study plan hash",
    "comparison_spec_hash": "comparison spec hash",
    "compiler_input_hash": "compiler input hash",
    "data_manifest_hash": "data manifest hash",
    "train_view_hash": "train view hash",
    "global_step_plan_hash": "global step plan hash",
    "release_hash": "release hash",
    "environment_hash": "environment hash",
    "placement_hash": "placement hash",
    "device_identity_hash": "device identity hash",
    "device_type": "device type",
    "is_spot": "spot policy",
    "microbatch_max_sequences": "microbatch cap",
    "gradient_accumulation_policy": "gradient accumulation policy",
    "global_step_max_loss_tokens": "global step max loss tokens",
    "global_step_max_attention_cells": "global step max attention cells",
    "global_step_max_sequences": "global step max sequences",
    "context_length": "context length",
    "prefetch_batches": "prefetch batches",
    "pin_memory": "pin memory",
    "nonblocking_device_transfer": "nonblocking device transfer",
    "read_mode": "read mode",
    "removable_data": "removable data",
    "precision": "precision",
    "compile_mode": "compile mode",
    "allow_tf32": "TF32 policy",
    "optimizer_backend": "optimizer backend",
    "attention_backend": "attention backend",
    "attention_num_splits": "attention split policy",
    "loss_backend": "loss backend",
}


@dataclass(frozen=True, slots=True)
class ConfigurationExecutionPathReceiptV1:
    """Canonical physical path selected for C2 or C3 GPU execution.

    The nested selection is retained as immutable canonical JSON so typed
    instances cannot be mutated away from the evidence that selected them.
    Serialization includes the no-raw selection so every consumer can strictly
    reconstruct the receipt without trusting an unbound side input.
    """

    stage: str
    shape_selection_hash: str
    study_plan_hash: str
    comparison_spec_hash: str
    compiler_input_hash: str
    data_manifest_hash: str
    train_view_hash: str
    global_step_plan_hash: str
    release_hash: str
    environment_hash: str
    placement_hash: str
    device_identity_hash: str
    device_type: str
    is_spot: bool
    microbatch_max_sequences: int
    gradient_accumulation_policy: str
    global_step_max_loss_tokens: int
    global_step_max_attention_cells: int
    global_step_max_sequences: int
    context_length: int
    prefetch_batches: int
    pin_memory: bool
    nonblocking_device_transfer: bool
    read_mode: str
    removable_data: bool
    precision: str
    compile_mode: str
    allow_tf32: bool
    optimizer_backend: str
    attention_backend: str
    attention_num_splits: int | str
    loss_backend: str
    _shape_selection_json: str = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self._shape_selection_json, str):
            raise ValueError("configuration execution path shape selection is invalid")
        try:
            parsed = json.loads(self._shape_selection_json)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("configuration execution path shape selection is invalid") from error
        if not isinstance(parsed, Mapping):
            raise ValueError("configuration execution path shape selection is invalid")
        selection = load_configuration_shape_selection(parsed)
        if self._shape_selection_json != _canonical_shape_selection_json(selection):
            raise ValueError("configuration execution path shape selection is noncanonical")
        expected = _execution_path_values(stage=self.stage, selection=selection)
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                label = _EXECUTION_PATH_FIELD_LABELS[field_name]
                raise ValueError(f"configuration execution path {label} mismatch")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_execution_path_receipt_v1",
            "schema_version": 1,
            "scientific_identity_eligible": False,
            "scientific_result_eligible": False,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "tokens_serialized": False,
            "secrets_serialized": False,
            "shape_selection": json.loads(self._shape_selection_json),
            **{
                field_name: getattr(self, field_name)
                for field_name in _EXECUTION_PATH_FIELD_LABELS
            },
        }


def build_configuration_execution_path_receipt(
    *,
    stage: str,
    shape_selection: Mapping[str, Any],
) -> ConfigurationExecutionPathReceiptV1:
    """Derive the sole accepted physical path from a strict shape selection."""

    selection = load_configuration_shape_selection(shape_selection)
    values = _execution_path_values(stage=stage, selection=selection)
    return ConfigurationExecutionPathReceiptV1(
        **values,
        _shape_selection_json=_canonical_shape_selection_json(selection),
    )


def load_configuration_execution_path_receipt(
    payload: Mapping[str, Any],
    *,
    shape_selection: Mapping[str, Any] | None = None,
) -> ConfigurationExecutionPathReceiptV1:
    """Strictly rebuild from embedded selection and optionally cross-check its source."""

    fields = set(_EXECUTION_PATH_FIELD_LABELS)
    metadata = {
        "artifact_kind",
        "schema_version",
        "scientific_identity_eligible",
        "scientific_result_eligible",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "raw_text_serialized",
        "raw_url_serialized",
        "tokens_serialized",
        "secrets_serialized",
        "shape_selection",
    }
    if set(payload) != fields | metadata:
        raise ValueError("configuration execution path receipt fields are invalid")
    if (
        payload.get("artifact_kind")
        != "twc_configuration_execution_path_receipt_v1"
        or payload.get("schema_version") != 1
        or any(payload.get(field_name) is not False for field_name in metadata - {
            "artifact_kind", "schema_version", "shape_selection"
        })
    ):
        raise ValueError("configuration execution path receipt boundary is invalid")
    embedded = payload.get("shape_selection")
    if not isinstance(embedded, Mapping):
        raise ValueError("configuration execution path embedded shape selection is invalid")
    loaded_embedded = load_configuration_shape_selection(embedded)
    if shape_selection is not None:
        loaded_external = load_configuration_shape_selection(shape_selection)
        if loaded_external["selection_hash"] != loaded_embedded["selection_hash"]:
            raise ValueError("configuration execution path shape selection mismatch")
    expected = build_configuration_execution_path_receipt(
        stage=payload["stage"],
        shape_selection=loaded_embedded,
    )
    for field_name, label in _EXECUTION_PATH_FIELD_LABELS.items():
        if payload.get(field_name) != getattr(expected, field_name):
            raise ValueError(f"configuration execution path {label} mismatch")
    return expected


@dataclass(frozen=True, slots=True)
class ConfigurationShapeProbeExecutionPathReceiptV1:
    """Physical path for one diagnostic probe point, never scientific work."""

    stage: str
    probe_profile_id: str
    study_plan_hash: str
    comparison_spec_hash: str
    data_manifest_hash: str
    train_view_hash: str
    global_step_plan_hash: str
    release_hash: str
    environment_hash: str
    placement_hash: str
    device_identity_hash: str
    probe_max_sequences: int
    device_type: str = "cuda"
    is_spot: bool = False
    gradient_accumulation_policy: str = (
        "accumulate_all_device_slices_for_exact_global_step_v1"
    )
    global_step_max_loss_tokens: int = _EXECUTION_PATH_GLOBAL_STEP_MAX_LOSS_TOKENS
    global_step_max_attention_cells: int = (
        _EXECUTION_PATH_GLOBAL_STEP_MAX_ATTENTION_CELLS
    )
    global_step_max_sequences: int = _EXECUTION_PATH_GLOBAL_STEP_MAX_SEQUENCES
    context_length: int = _EXECUTION_PATH_CONTEXT_LENGTH
    prefetch_batches: int = 1
    pin_memory: bool = True
    nonblocking_device_transfer: bool = True
    read_mode: str = "local_mmap"
    removable_data: bool = False
    precision: str = "bf16"
    compile_mode: str = "default"
    allow_tf32: bool = True
    optimizer_backend: str = "fused"
    attention_backend: str = "native_varlen"
    attention_num_splits: int | str = "auto"
    loss_backend: str = "chunked_projection"

    def __post_init__(self) -> None:
        profiles = _SHAPE_PROBE_PROFILES.get(self.stage)
        if profiles is None or self.probe_profile_id not in profiles:
            raise ValueError("shape-probe execution path stage/profile mismatch")
        if self.probe_max_sequences not in _SHAPE_PROBE_SEQUENCE_CAPS:
            raise ValueError("shape-probe execution path cap is not predeclared")
        for field_name in (
            "study_plan_hash",
            "comparison_spec_hash",
            "data_manifest_hash",
            "train_view_hash",
            "global_step_plan_hash",
            "release_hash",
            "environment_hash",
            "placement_hash",
            "device_identity_hash",
        ):
            _require_hash(getattr(self, field_name), field_name)
        expected = {
            "device_type": "cuda",
            "is_spot": False,
            "gradient_accumulation_policy": (
                "accumulate_all_device_slices_for_exact_global_step_v1"
            ),
            "global_step_max_loss_tokens": _EXECUTION_PATH_GLOBAL_STEP_MAX_LOSS_TOKENS,
            "global_step_max_attention_cells": _EXECUTION_PATH_GLOBAL_STEP_MAX_ATTENTION_CELLS,
            "global_step_max_sequences": _EXECUTION_PATH_GLOBAL_STEP_MAX_SEQUENCES,
            "context_length": _EXECUTION_PATH_CONTEXT_LENGTH,
            "prefetch_batches": 1,
            "pin_memory": True,
            "nonblocking_device_transfer": True,
            "read_mode": "local_mmap",
            "removable_data": False,
            "precision": "bf16",
            "compile_mode": "default",
            "allow_tf32": True,
            "optimizer_backend": "fused",
            "attention_backend": "native_varlen",
            "attention_num_splits": "auto",
            "loss_backend": "chunked_projection",
        }
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise ValueError(f"shape-probe execution path {field_name} mismatch")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_shape_probe_execution_path_v1",
            "schema_version": 1,
            "execution_purpose": "diagnostic_shape_probe",
            "scientific_campaign_eligible": False,
            "scientific_identity_eligible": False,
            "scientific_result_eligible": False,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "tokens_serialized": False,
            "secrets_serialized": False,
            **asdict(self),
        }


def build_configuration_shape_probe_execution_path_receipt(
    *,
    stage: str,
    probe_profile_id: str,
    study_plan_hash: str,
    comparison_spec_hash: str,
    data_manifest_hash: str,
    train_view_hash: str,
    global_step_plan_hash: str,
    release_hash: str,
    environment_hash: str,
    placement_hash: str,
    device_identity_hash: str,
    probe_max_sequences: int,
) -> ConfigurationShapeProbeExecutionPathReceiptV1:
    return ConfigurationShapeProbeExecutionPathReceiptV1(
        stage=stage,
        probe_profile_id=probe_profile_id,
        study_plan_hash=study_plan_hash,
        comparison_spec_hash=comparison_spec_hash,
        data_manifest_hash=data_manifest_hash,
        train_view_hash=train_view_hash,
        global_step_plan_hash=global_step_plan_hash,
        release_hash=release_hash,
        environment_hash=environment_hash,
        placement_hash=placement_hash,
        device_identity_hash=device_identity_hash,
        probe_max_sequences=probe_max_sequences,
    )


def load_configuration_shape_probe_execution_path_receipt(
    payload: Mapping[str, Any],
) -> ConfigurationShapeProbeExecutionPathReceiptV1:
    fields = set(ConfigurationShapeProbeExecutionPathReceiptV1.__dataclass_fields__)
    metadata = {
        "artifact_kind",
        "schema_version",
        "execution_purpose",
        "scientific_campaign_eligible",
        "scientific_identity_eligible",
        "scientific_result_eligible",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "raw_text_serialized",
        "raw_url_serialized",
        "tokens_serialized",
        "secrets_serialized",
    }
    if set(payload) != fields | metadata:
        raise ValueError("shape-probe execution path fields are invalid")
    if (
        payload.get("artifact_kind")
        != "twc_configuration_shape_probe_execution_path_v1"
        or payload.get("schema_version") != 1
        or payload.get("execution_purpose") != "diagnostic_shape_probe"
        or any(
            payload.get(field_name) is not False
            for field_name in metadata
            - {"artifact_kind", "schema_version", "execution_purpose"}
        )
    ):
        raise ValueError("shape-probe execution path boundary is invalid")
    return ConfigurationShapeProbeExecutionPathReceiptV1(
        **{field_name: payload[field_name] for field_name in fields}
    )


@dataclass(frozen=True, slots=True)
class ExperimentExecutionBindingV1:
    """Bind one scientific condition to one predeclared physical attempt."""

    scientific_contract_hash: str
    campaign_hash: str
    wave_id: str
    attempt_id: str
    release_hash: str
    execution_path_hash: str
    stage_authorization_hash: str

    def __post_init__(self) -> None:
        for field in (
            "scientific_contract_hash",
            "campaign_hash",
            "release_hash",
            "execution_path_hash",
            "stage_authorization_hash",
        ):
            _require_hash(getattr(self, field), field)
        _require_identifier(self.wave_id, "wave_id")
        _require_identifier(self.attempt_id, "attempt_id")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_execution_binding_v1",
            "schema_version": 1,
            **asdict(self),
        }


def load_experiment_execution_binding(payload: Mapping[str, Any]) -> ExperimentExecutionBindingV1:
    fields = set(ExperimentExecutionBindingV1.__dataclass_fields__)
    if set(payload) != fields | {"artifact_kind", "schema_version"}:
        raise ValueError("execution binding fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_configuration_execution_binding_v1"
        or payload.get("schema_version") != 1
    ):
        raise ValueError("execution binding artifact is invalid")
    return ExperimentExecutionBindingV1(**{field: payload[field] for field in fields})


@dataclass(frozen=True, slots=True)
class StopBoundCopybackProofV1:
    """Bind independent local copyback validation to one physical stop.

    The copyback validator owns issuance of this proof.  Reconciliation only
    checks that its already-validated identities close over the exact campaign,
    execution binding, stop receipt, manifest, and optional terminal result.
    """

    campaign_hash: str
    wave_id: str
    attempt_id: str
    execution_binding_hash: str
    attempt_receipt_hash: str
    copyback_manifest_hash: str
    copyback_validation_receipt_hash: str
    result_hash: str | None

    def __post_init__(self) -> None:
        for field in (
            "campaign_hash",
            "execution_binding_hash",
            "attempt_receipt_hash",
            "copyback_manifest_hash",
            "copyback_validation_receipt_hash",
        ):
            _require_hash(getattr(self, field), field)
        _require_identifier(self.wave_id, "wave_id")
        _require_identifier(self.attempt_id, "attempt_id")
        if self.result_hash is not None:
            _require_hash(self.result_hash, "result_hash")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_stop_bound_copyback_proof_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            **asdict(self),
        }


def load_stop_bound_copyback_proof(
    payload: Mapping[str, Any],
) -> StopBoundCopybackProofV1:
    fields = set(StopBoundCopybackProofV1.__dataclass_fields__)
    metadata = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
    }
    if set(payload) != fields | metadata:
        raise ValueError("stop-bound copyback proof fields are invalid")
    if (
        payload.get("artifact_kind")
        != "twc_configuration_stop_bound_copyback_proof_v1"
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
        or payload.get("f1_unlock_eligible") is not False
    ):
        raise ValueError("stop-bound copyback proof boundary is invalid")
    return StopBoundCopybackProofV1(
        **{field: payload[field] for field in fields}
    )


@dataclass(frozen=True, slots=True)
class ReconciledAttemptV1:
    """Validated dependencies for one stopped physical attempt."""

    execution_binding: ExperimentExecutionBindingV1
    attempt_receipt: AttemptReceipt
    copyback_proof: StopBoundCopybackProofV1
    result: ExperimentResultV3 | None

    def __post_init__(self) -> None:
        if not isinstance(self.execution_binding, ExperimentExecutionBindingV1):
            raise ValueError("reconciled attempt requires an execution binding")
        if not isinstance(self.attempt_receipt, AttemptReceipt):
            raise ValueError("reconciled attempt requires an attempt receipt")
        if not isinstance(self.copyback_proof, StopBoundCopybackProofV1):
            raise ValueError("reconciled attempt requires stop-bound copyback proof")
        binding = self.execution_binding
        stop = self.attempt_receipt
        proof = self.copyback_proof
        expected = {
            "campaign_hash": binding.campaign_hash,
            "wave_id": binding.wave_id,
            "attempt_id": binding.attempt_id,
            "execution_binding_hash": binding.identity,
            "attempt_receipt_hash": stop.receipt_hash,
            "copyback_manifest_hash": stop.copyback_manifest_hash,
        }
        for field, value in expected.items():
            if getattr(proof, field) != value:
                raise ValueError(f"stop-bound copyback proof {field} mismatch")
        if (stop.wave_id, stop.attempt_id) != (binding.wave_id, binding.attempt_id):
            raise ValueError("attempt receipt does not match the execution binding")
        if stop.stop_reason == "completed":
            if not isinstance(self.result, ExperimentResultV3):
                raise ValueError("completed attempt requires a validated result")
            if proof.result_hash != self.result.identity:
                raise ValueError("stop-bound copyback proof result hash mismatch")
            if self.result.identity not in stop.artifact_hashes:
                raise ValueError("completed result is absent from the stop-bound copyback")
            if self.result.execution_binding_hash != binding.identity:
                raise ValueError("result execution binding is cross-spliced")
            if self.result.scientific_contract_hash != binding.scientific_contract_hash:
                raise ValueError("result scientific contract is cross-spliced")
        elif self.result is not None or proof.result_hash is not None:
            raise ValueError("noncompleted attempt cannot carry a terminal result")

    @property
    def result_hash(self) -> str | None:
        return self.result.identity if self.result is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "wave_id": self.execution_binding.wave_id,
            "attempt_id": self.execution_binding.attempt_id,
            "scientific_contract_hash": self.execution_binding.scientific_contract_hash,
            "execution_binding_hash": self.execution_binding.identity,
            "attempt_receipt_hash": self.attempt_receipt.receipt_hash,
            "copyback_proof_hash": self.copyback_proof.identity,
            "stop_reason": self.attempt_receipt.stop_reason,
            "result_hash": self.result_hash,
        }


@dataclass(frozen=True, slots=True)
class StageReconciliationReceipt:
    """Exactly one durable result for each ordered logical stage condition."""

    stage: str
    study_plan_hash: str
    comparison_spec_hash: str
    campaign_hash: str
    release_hash: str
    execution_path_hash: str
    condition_ids: tuple[str, ...]
    scientific_contract_hashes: tuple[str, ...]
    successful_attempt_ids: tuple[str, ...]
    result_hashes: tuple[str, ...]
    attempts: tuple[ReconciledAttemptV1, ...]

    def __post_init__(self) -> None:
        if self.stage not in {"c2", "c3a", "c3b"}:
            raise ValueError("configuration reconciliation stage is invalid")
        for field in (
            "study_plan_hash",
            "comparison_spec_hash",
            "campaign_hash",
            "release_hash",
            "execution_path_hash",
        ):
            _require_hash(getattr(self, field), field)
        logical_count = len(self.condition_ids)
        if logical_count == 0 or any(
            len(values) != logical_count
            for values in (
                self.scientific_contract_hashes,
                self.successful_attempt_ids,
                self.result_hashes,
            )
        ):
            raise ValueError("stage reconciliation logical evidence does not align")
        for index, condition_id in enumerate(self.condition_ids):
            _require_identifier(condition_id, f"condition_ids[{index}]")
        for index, attempt_id in enumerate(self.successful_attempt_ids):
            _require_identifier(attempt_id, f"successful_attempt_ids[{index}]")
        for field, values in (
            ("scientific_contract_hashes", self.scientific_contract_hashes),
            ("result_hashes", self.result_hashes),
        ):
            for index, value in enumerate(values):
                _require_hash(value, f"{field}[{index}]")
        for label, values in (
            ("condition", self.condition_ids),
            ("scientific contract", self.scientific_contract_hashes),
            ("successful attempt", self.successful_attempt_ids),
            ("result", self.result_hashes),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"stage reconciliation {label} values must be unique")
        if not self.attempts or any(
            not isinstance(item, ReconciledAttemptV1) for item in self.attempts
        ):
            raise ValueError("stage reconciliation requires validated attempts")
        attempt_ids = tuple(item.execution_binding.attempt_id for item in self.attempts)
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("stage reconciliation attempt IDs must be unique")
        by_attempt = {
            item.execution_binding.attempt_id: item for item in self.attempts
        }
        for index, successful_attempt_id in enumerate(self.successful_attempt_ids):
            attempt = by_attempt.get(successful_attempt_id)
            if attempt is None or attempt.result is None:
                raise ValueError("stage reconciliation successful attempt is unavailable")
            result = attempt.result
            if (
                result.identity != self.result_hashes[index]
                or result.scientific_contract_hash
                != self.scientific_contract_hashes[index]
                or result.scientific_contract.condition.condition_id
                != self.condition_ids[index]
                or result.scientific_contract.study_plan_hash != self.study_plan_hash
                or result.scientific_contract.comparison_spec_hash
                != self.comparison_spec_hash
            ):
                raise ValueError("stage reconciliation logical result bindings are invalid")

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    @property
    def results(self) -> tuple[ExperimentResultV3, ...]:
        by_attempt = {
            item.execution_binding.attempt_id: item.result for item in self.attempts
        }
        results: list[ExperimentResultV3] = []
        for attempt_id in self.successful_attempt_ids:
            result = by_attempt[attempt_id]
            if result is None:  # guarded by __post_init__
                raise ValueError("stage reconciliation successful result is unavailable")
            results.append(result)
        return tuple(results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_stage_reconciliation_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "stage": self.stage,
            "study_plan_hash": self.study_plan_hash,
            "comparison_spec_hash": self.comparison_spec_hash,
            "campaign_hash": self.campaign_hash,
            "release_hash": self.release_hash,
            "execution_path_hash": self.execution_path_hash,
            "condition_ids": self.condition_ids,
            "scientific_contract_hashes": self.scientific_contract_hashes,
            "successful_attempt_ids": self.successful_attempt_ids,
            "result_hashes": self.result_hashes,
            "attempts": tuple(item.to_dict() for item in self.attempts),
        }


def _stage_attempt_pairs(
    campaign_plan: CampaignPlan, stage: str
) -> tuple[tuple[Any, Any], ...]:
    if not isinstance(campaign_plan, CampaignPlan):
        raise ValueError("stage reconciliation requires a CampaignPlan")
    if stage not in {"c2", "c3a", "c3b"}:
        raise ValueError("configuration reconciliation stage is invalid")
    if campaign_plan.campaign_id != f"twc-{stage}-configuration-v2":
        raise ValueError("stage reconciliation campaign mismatch")
    expected_wave_ids = (f"{stage}-primary", f"{stage}-fresh-retry")
    if tuple(wave.wave_id for wave in campaign_plan.waves) != expected_wave_ids:
        raise ValueError("stage reconciliation campaign waves are invalid")
    primary_wave, retry_wave = campaign_plan.waves
    if (
        primary_wave.prerequisite_wave_ids
        or retry_wave.prerequisite_wave_ids != (primary_wave.wave_id,)
        or len(primary_wave.attempts) != len(retry_wave.attempts)
    ):
        raise ValueError("stage reconciliation campaign retry topology is invalid")
    pairs = tuple(zip(primary_wave.attempts, retry_wave.attempts, strict=True))
    for primary, retry in pairs:
        if (
            primary.recovery_mode != "none"
            or retry.recovery_mode != "fresh_restart"
            or retry.predecessor_attempt_id != primary.attempt_id
            or retry.attempt_id != f"{primary.attempt_id}-fresh-retry"
        ):
            raise ValueError("stage reconciliation campaign retry topology is invalid")
    return pairs


def _result_stage(result: ExperimentResultV3) -> str:
    condition = result.scientific_contract.condition
    if condition.phase == "m_transfer" and condition.seed == 17:
        return "c2"
    if condition.phase == "l_confirmation" and condition.seed == 29:
        return "c3a"
    if condition.phase == "l_confirmation" and condition.seed == 43:
        return "c3b"
    raise ValueError("reconciled result is outside the approved C2/C3 stages")


def build_stage_reconciliation_receipt(
    *,
    campaign_plan: CampaignPlan,
    stage: str,
    attempts: tuple[ReconciledAttemptV1, ...],
) -> StageReconciliationReceipt:
    """Reconcile primary/fresh-retry execution into ordered logical results."""

    pairs = _stage_attempt_pairs(campaign_plan, stage)
    if not isinstance(attempts, tuple) or not attempts or any(
        not isinstance(item, ReconciledAttemptV1) for item in attempts
    ):
        raise ValueError("stage reconciliation attempts must be a non-empty tuple")
    observed_ids = tuple(item.execution_binding.attempt_id for item in attempts)
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("stage reconciliation contains duplicate physical attempts")
    declared = tuple(
        attempt.attempt_id
        for wave in campaign_plan.waves
        for attempt in wave.attempts
    )
    undeclared = tuple(attempt_id for attempt_id in observed_ids if attempt_id not in declared)
    if undeclared:
        raise ValueError(f"stage reconciliation contains undeclared attempts: {undeclared}")
    expected_observed_order = tuple(
        attempt_id for attempt_id in declared if attempt_id in set(observed_ids)
    )
    if observed_ids != expected_observed_order:
        raise ValueError("stage reconciliation attempt order is invalid")
    by_id = {item.execution_binding.attempt_id: item for item in attempts}
    for item in attempts:
        binding = item.execution_binding
        planned = campaign_plan.wave(binding.wave_id).attempt(binding.attempt_id)
        if binding.campaign_hash != campaign_plan.identity:
            raise ValueError("stage reconciliation campaign binding mismatch")
        if binding.release_hash != campaign_plan.code_hash:
            raise ValueError("stage reconciliation release binding mismatch")
        if binding.execution_path_hash != planned.execution_path_hash:
            raise ValueError("stage reconciliation execution path binding mismatch")

    condition_ids: list[str] = []
    scientific_hashes: list[str] = []
    successful_attempt_ids: list[str] = []
    result_hashes: list[str] = []
    study_plan_hash: str | None = None
    comparison_spec_hash: str | None = None
    for primary_plan, retry_plan in pairs:
        primary = by_id.get(primary_plan.attempt_id)
        retry = by_id.get(retry_plan.attempt_id)
        if primary is None:
            raise ValueError(
                f"stage reconciliation missing primary attempt: {primary_plan.attempt_id}"
            )
        if primary.attempt_receipt.stop_reason == "completed" and retry is not None:
            if retry.attempt_receipt.stop_reason == "completed":
                raise ValueError(
                    "stage reconciliation duplicate logical success from post-success retry"
                )
            raise ValueError("stage reconciliation contains a post-success retry")
        if retry is not None:
            if primary.attempt_receipt.stop_reason != "failed":
                raise ValueError(
                    "stage reconciliation fresh retry requires a canonically failed "
                    "non-spot primary"
                )
            if (
                retry.execution_binding.scientific_contract_hash
                != primary.execution_binding.scientific_contract_hash
            ):
                raise ValueError("stage reconciliation retry scientific contract is cross-spliced")
        logical_attempts = tuple(item for item in (primary, retry) if item is not None)
        successes = tuple(
            item
            for item in logical_attempts
            if item.attempt_receipt.stop_reason == "completed"
        )
        if len(successes) != 1:
            raise ValueError(
                "stage reconciliation requires exactly one logical success for "
                f"{primary_plan.attempt_id}"
            )
        success = successes[0]
        result = success.result
        if result is None:  # guarded by ReconciledAttemptV1; keeps this seam total.
            raise ValueError("stage reconciliation logical success is missing its result")
        if result.scientific_contract.condition.condition_id != primary_plan.attempt_id:
            raise ValueError("stage reconciliation result condition is cross-spliced")
        if _result_stage(result) != stage:
            raise ValueError("stage reconciliation result belongs to the wrong stage")
        if (
            success.execution_binding.scientific_contract_hash
            != primary.execution_binding.scientific_contract_hash
        ):
            raise ValueError("stage reconciliation logical scientific contract is cross-spliced")
        contract = result.scientific_contract
        if study_plan_hash is None:
            study_plan_hash = contract.study_plan_hash
            comparison_spec_hash = contract.comparison_spec_hash
        elif (
            contract.study_plan_hash != study_plan_hash
            or contract.comparison_spec_hash != comparison_spec_hash
        ):
            raise ValueError("stage reconciliation study or comparison evidence is cross-spliced")
        condition_ids.append(primary_plan.attempt_id)
        scientific_hashes.append(contract.identity)
        successful_attempt_ids.append(success.execution_binding.attempt_id)
        result_hashes.append(result.identity)

    if study_plan_hash is None or comparison_spec_hash is None:  # pragma: no cover
        raise ValueError("stage reconciliation has no logical results")
    execution_paths = {
        attempt.execution_path_hash
        for pair in pairs
        for attempt in pair
    }
    if len(execution_paths) != 1:
        raise ValueError("stage reconciliation campaign has multiple execution paths")
    return StageReconciliationReceipt(
        stage=stage,
        study_plan_hash=study_plan_hash,
        comparison_spec_hash=comparison_spec_hash,
        campaign_hash=campaign_plan.identity,
        release_hash=campaign_plan.code_hash,
        execution_path_hash=execution_paths.pop(),
        condition_ids=tuple(condition_ids),
        scientific_contract_hashes=tuple(scientific_hashes),
        successful_attempt_ids=tuple(successful_attempt_ids),
        result_hashes=tuple(result_hashes),
        attempts=attempts,
    )


def _dependency(
    values: Mapping[str, Any], identity: object, expected_type: type, label: str
) -> Any:
    digest = _require_hash(identity, label)
    value = values.get(digest)
    if not isinstance(value, expected_type):
        raise ValueError(f"stage reconciliation {label} is unavailable")
    actual_identity = (
        value.receipt_hash if isinstance(value, AttemptReceipt) else value.identity
    )
    if actual_identity != digest:
        raise ValueError(f"stage reconciliation {label} map is invalid")
    return value


def load_stage_reconciliation_receipt(
    payload: Mapping[str, Any],
    *,
    campaign_plan: CampaignPlan,
    execution_bindings_by_hash: Mapping[str, ExperimentExecutionBindingV1],
    attempt_receipts_by_hash: Mapping[str, AttemptReceipt],
    copyback_proofs_by_hash: Mapping[str, StopBoundCopybackProofV1],
    results_by_hash: Mapping[str, ExperimentResultV3],
) -> StageReconciliationReceipt:
    """Strictly reconstruct reconciliation from independently loaded inputs."""

    fields = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "stage",
        "study_plan_hash",
        "comparison_spec_hash",
        "campaign_hash",
        "release_hash",
        "execution_path_hash",
        "condition_ids",
        "scientific_contract_hashes",
        "successful_attempt_ids",
        "result_hashes",
        "attempts",
    }
    if set(payload) != fields:
        raise ValueError("stage reconciliation fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_configuration_stage_reconciliation_v1"
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
        or payload.get("f1_unlock_eligible") is not False
    ):
        raise ValueError("stage reconciliation boundary is invalid")
    attempt_rows = payload.get("attempts")
    if not isinstance(attempt_rows, (list, tuple)):
        raise ValueError("stage reconciliation attempts are invalid")
    row_fields = {
        "wave_id",
        "attempt_id",
        "scientific_contract_hash",
        "execution_binding_hash",
        "attempt_receipt_hash",
        "copyback_proof_hash",
        "stop_reason",
        "result_hash",
    }
    attempts: list[ReconciledAttemptV1] = []
    for row in attempt_rows:
        if not isinstance(row, Mapping) or set(row) != row_fields:
            raise ValueError("stage reconciliation attempt fields are invalid")
        binding = _dependency(
            execution_bindings_by_hash,
            row["execution_binding_hash"],
            ExperimentExecutionBindingV1,
            "execution binding",
        )
        stop = _dependency(
            attempt_receipts_by_hash,
            row["attempt_receipt_hash"],
            AttemptReceipt,
            "attempt receipt",
        )
        proof = _dependency(
            copyback_proofs_by_hash,
            row["copyback_proof_hash"],
            StopBoundCopybackProofV1,
            "copyback proof",
        )
        result_hash = row["result_hash"]
        result = None
        if result_hash is not None:
            result = _dependency(
                results_by_hash,
                result_hash,
                ExperimentResultV3,
                "result hash",
            )
        attempt = ReconciledAttemptV1(binding, stop, proof, result)
        if stable_json_hash(row) != stable_json_hash(attempt.to_dict()):
            raise ValueError("stage reconciliation attempt derived evidence mismatch")
        attempts.append(attempt)
    rebuilt = build_stage_reconciliation_receipt(
        campaign_plan=campaign_plan,
        stage=payload["stage"],
        attempts=tuple(attempts),
    )
    if stable_json_hash(payload) != stable_json_hash(rebuilt.to_dict()):
        raise ValueError("stage reconciliation derived evidence mismatch")
    return rebuilt


@dataclass(frozen=True, slots=True)
class ExperimentTrainingContractV2:
    """V2 scientific condition plus execution binding for the shared worker."""

    experiment: ExperimentContractV2
    execution_binding: ExperimentExecutionBindingV1
    view_hash: str
    model_hash: str
    global_step_plan_hash: str
    optimizer_step_token_cap: int
    optimizer_step_attention_cell_cap: int
    optimizer_step_sequence_cap: int
    precision: str
    attention_backend: str
    attention_num_splits: int | str
    loss_backend: str
    compile_mode: str
    allow_tf32: bool
    optimizer: str
    optimizer_backend: str
    adam_betas: tuple[float, float]
    adam_eps: float
    weight_decay: float
    warmup_loss_tokens: int
    stable_loss_tokens: int
    decay_loss_tokens: int
    grad_clip_norm: float
    full_validation_at_schedule_boundaries: bool

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, ExperimentContractV2):
            raise ValueError("v2 training contract requires an ExperimentContractV2")
        if not isinstance(self.execution_binding, ExperimentExecutionBindingV1):
            raise ValueError("v2 training contract requires an execution binding")
        if self.execution_binding.scientific_contract_hash != self.experiment.identity:
            raise ValueError("execution binding scientific contract mismatch")
        if self.view_hash != self.experiment.train_view_hash:
            raise ValueError("v2 training view does not match the experiment")
        for field in ("view_hash", "model_hash", "global_step_plan_hash"):
            _require_hash(getattr(self, field), field)
        for field in (
            "optimizer_step_token_cap",
            "optimizer_step_attention_cell_cap",
            "optimizer_step_sequence_cap",
            "warmup_loss_tokens",
            "stable_loss_tokens",
            "decay_loss_tokens",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.warmup_loss_tokens + self.stable_loss_tokens + self.decay_loss_tokens != self.target_loss_tokens:
            raise ValueError("v2 training schedule must sum to its exact token horizon")
        if self.warmup_loss_tokens != int(self.target_loss_tokens * 0.02):
            raise ValueError("v2 training warmup must be exactly two percent")
        if self.decay_loss_tokens != int(self.target_loss_tokens * self.experiment.schedule_tail_fraction):
            raise ValueError("v2 training decay does not match the schedule tail")
        if self.precision != "bf16" or self.attention_backend != "native_varlen":
            raise ValueError("configuration experiments require BF16 native variable-length attention")
        if not (
            self.attention_num_splits == "auto"
            or type(self.attention_num_splits) is int
            and self.attention_num_splits > 0
        ):
            raise ValueError("v2 attention_num_splits is invalid")
        if self.loss_backend != "chunked_projection":
            raise ValueError("v2 experiments require chunked projection loss")
        if self.compile_mode not in {"disabled", "default", "reduce-overhead", "max-autotune"}:
            raise ValueError("v2 compile mode is invalid")
        if self.optimizer != "adamw" or self.optimizer_backend != "fused":
            raise ValueError("v2 experiments require fused AdamW")
        if type(self.allow_tf32) is not bool or self.full_validation_at_schedule_boundaries is not True:
            raise ValueError("v2 training boolean safety fields are invalid")
        beta1, beta2 = self.adam_betas
        numeric = (beta1, beta2, self.adam_eps, self.weight_decay, self.grad_clip_norm)
        if any(type(value) not in {int, float} or not math.isfinite(value) for value in numeric):
            raise ValueError("v2 optimizer values must be finite")
        if (
            not 0 <= beta1 < 1
            or not 0 <= beta2 < 1
            or self.adam_eps <= 0
            or self.weight_decay < 0
            or self.grad_clip_norm <= 0
        ):
            raise ValueError("v2 optimizer values are invalid")

    @property
    def run_id(self) -> str:
        return self.experiment.experiment_id

    @property
    def data_manifest_hash(self) -> str:
        return self.experiment.data_manifest_hash

    @property
    def validation_view_hash(self) -> str:
        return self.experiment.validation_view_hash

    @property
    def tokenizer_hash(self) -> str:
        return self.experiment.tokenizer_hash

    @property
    def code_hash(self) -> str:
        return self.experiment.semantic_code_hash

    @property
    def target_loss_tokens(self) -> int:
        return self.experiment.target_loss_tokens

    @property
    def context_length(self) -> int:
        return self.experiment.profile.context_length

    @property
    def learning_rate(self) -> float:
        return self.experiment.learning_rate

    @property
    def seed(self) -> int:
        return self.experiment.seed

    @property
    def expected_parameter_count(self) -> int:
        return self.experiment.profile.parameter_count

    @property
    def model_profile_id(self) -> str:
        return self.experiment.profile_id

    @property
    def identity(self) -> str:
        return stable_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_pretraining_experiment_training_contract_v2",
            "schema_version": 2,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "experiment": self.experiment.to_dict(),
            "execution_binding": self.execution_binding.to_dict(),
            **{
                field: value
                for field, value in asdict(self).items()
                if field not in {"experiment", "execution_binding"}
            },
            "run_id": self.run_id,
            "data_manifest_hash": self.data_manifest_hash,
            "validation_view_hash": self.validation_view_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "code_hash": self.code_hash,
            "target_loss_tokens": self.target_loss_tokens,
            "context_length": self.context_length,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "expected_parameter_count": self.expected_parameter_count,
            "model_profile_id": self.model_profile_id,
        }


def load_experiment_training_contract_v2(payload: Mapping[str, Any]) -> ExperimentTrainingContractV2:
    direct_fields = set(ExperimentTrainingContractV2.__dataclass_fields__) - {
        "experiment",
        "execution_binding",
    }
    derived = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "experiment",
        "execution_binding",
        "run_identity",
        "run_id",
        "data_manifest_hash",
        "validation_view_hash",
        "tokenizer_hash",
        "code_hash",
        "target_loss_tokens",
        "context_length",
        "learning_rate",
        "seed",
        "expected_parameter_count",
        "model_profile_id",
    }
    if set(payload) - direct_fields - derived or direct_fields - set(payload):
        raise ValueError("v2 experiment training contract fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_pretraining_experiment_training_contract_v2"
        or payload.get("schema_version") != 2
        or payload.get("production_unlock_eligible") is not False
        or payload.get("f1_unlock_eligible") is not False
        or not isinstance(payload.get("experiment"), Mapping)
        or not isinstance(payload.get("execution_binding"), Mapping)
    ):
        raise ValueError("v2 experiment training contract boundary is invalid")
    values = {field: payload[field] for field in direct_fields}
    values["adam_betas"] = tuple(values["adam_betas"])
    contract = ExperimentTrainingContractV2(
        experiment=load_experiment_contract_v2(payload["experiment"]),
        execution_binding=load_experiment_execution_binding(payload["execution_binding"]),
        **values,
    )
    canonical = contract.to_dict()
    for field in (
        "run_id",
        "data_manifest_hash",
        "validation_view_hash",
        "tokenizer_hash",
        "code_hash",
        "target_loss_tokens",
        "context_length",
        "learning_rate",
        "seed",
        "expected_parameter_count",
        "model_profile_id",
    ):
        if payload.get(field) != canonical[field]:
            raise ValueError(f"v2 experiment training derived field mismatch: {field}")
    if payload.get("run_identity") not in {None, contract.identity}:
        raise ValueError("v2 experiment training identity mismatch")
    return contract


def validate_configuration_execution_path_for_training(
    *,
    execution_path: ConfigurationExecutionPathReceiptV1 | Mapping[str, Any],
    training_contract: ExperimentTrainingContractV2 | Mapping[str, Any],
) -> ConfigurationExecutionPathReceiptV1:
    """Bind the selected physical path to one v2 scientific worker contract.

    This seam deliberately does not accept a device-plan hash.  The execution
    path is the identity of the selected physical policy; the independently
    generated device plan is validated against that policy at worker launch.
    """

    path = (
        execution_path
        if isinstance(execution_path, ConfigurationExecutionPathReceiptV1)
        else load_configuration_execution_path_receipt(execution_path)
    )
    contract = (
        training_contract
        if isinstance(training_contract, ExperimentTrainingContractV2)
        else load_experiment_training_contract_v2(training_contract)
    )
    experiment = contract.experiment
    expected_stage = "c2" if experiment.phase == "m_transfer" else "c3"
    expected = {
        "stage": expected_stage,
        "study_plan_hash": experiment.study_plan_hash,
        "comparison_spec_hash": experiment.comparison_spec_hash,
        "data_manifest_hash": experiment.data_manifest_hash,
        "train_view_hash": experiment.train_view_hash,
        "global_step_plan_hash": contract.global_step_plan_hash,
        "release_hash": contract.execution_binding.release_hash,
        "context_length": contract.context_length,
        "global_step_max_loss_tokens": contract.optimizer_step_token_cap,
        "global_step_max_attention_cells": contract.optimizer_step_attention_cell_cap,
        "global_step_max_sequences": contract.optimizer_step_sequence_cap,
        "precision": contract.precision,
        "attention_backend": contract.attention_backend,
        "attention_num_splits": contract.attention_num_splits,
        "loss_backend": contract.loss_backend,
        "compile_mode": contract.compile_mode,
        "allow_tf32": contract.allow_tf32,
        "optimizer_backend": contract.optimizer_backend,
    }
    for field_name, value in expected.items():
        if getattr(path, field_name) != value:
            raise ValueError(
                f"configuration execution path/training {field_name} mismatch"
            )
    if contract.execution_binding.execution_path_hash != path.identity:
        raise ValueError(
            "configuration execution binding does not identify the exact path receipt"
        )
    return path
