"""Leaf contracts for C2/C3 readiness and stage authorization.

This module owns immutable evidence validation shared by stage orchestration and
the pure study compiler.  It deliberately does not import either of those
higher-level modules, keeping the dependency graph acyclic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Mapping

from twc.pretraining.campaign import CampaignLedger, CampaignPlan
from twc.pretraining.configuration_shape_copyback import (
    ConfigurationShapeCopybackReadbackV1,
    load_configuration_shape_copyback_readback,
)
from twc.pretraining.configuration_study_execution import (
    ConfigurationExecutionPathReceiptV1,
    ExperimentExecutionBindingV1,
)
from twc.pretraining.gpu_provider_evidence import (
    canonical_money,
    validate_configuration_provider_offer,
)
from twc.pretraining.gpu_engineering_verdict import (
    validate_accepted_engineering_canary_verdict,
)
from twc.repro.hashing import stable_json_hash

_HASH = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_AUTHORIZATION_MAX_AGE_SECONDS = 600
_ATTEMPT_START_TO_LAUNCH_MAX_SECONDS = 60


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def _canonical_json(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stage_for_campaign(campaign: CampaignPlan) -> str:
    for stage in ("c2", "c3a", "c3b"):
        if campaign.campaign_id == f"twc-{stage}-configuration-v2":
            return stage
    raise ValueError("configuration stage authorization campaign is unsupported")


def _component_receipt(
    artifact_kind: str, *, schema_version: int = 1, **values: Any
) -> dict[str, Any]:
    payload = {
        "artifact_kind": artifact_kind,
        "schema_version": schema_version,
        "production_unlock_eligible": False,
        "f1_unlock_eligible": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "tokens_serialized": False,
        "secrets_serialized": False,
        **values,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload


def _load_component_receipt(
    payload: Mapping[str, Any],
    *,
    artifact_kind: str,
    value_fields: set[str],
    schema_version: int = 1,
) -> dict[str, Any]:
    metadata = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "raw_text_serialized",
        "raw_url_serialized",
        "tokens_serialized",
        "secrets_serialized",
        "receipt_hash",
    }
    if set(payload) != metadata | value_fields:
        raise ValueError(f"{artifact_kind} fields are invalid")
    if (
        payload.get("artifact_kind") != artifact_kind
        or payload.get("schema_version") != schema_version
        or any(
            payload.get(field_name) is not False
            for field_name in metadata
            - {"artifact_kind", "schema_version", "receipt_hash"}
        )
        or payload.get("receipt_hash")
        != stable_json_hash(
            {key: value for key, value in payload.items() if key != "receipt_hash"}
        )
    ):
        raise ValueError(f"{artifact_kind} boundary or hash is invalid")
    return dict(payload)


def build_configuration_release_validation_receipt(
    *,
    runtime_release_receipt: Mapping[str, Any],
    semantic_code_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind canonical runtime and semantic inventories without serializing code."""

    from twc.pretraining.configuration_semantic_code import (
        validate_configuration_semantic_code_receipt,
    )
    from twc.pretraining.gpu_campaign_deployment import (
        validate_thin_runtime_release_receipt,
    )

    runtime = validate_thin_runtime_release_receipt(runtime_release_receipt)
    semantic = validate_configuration_semantic_code_receipt(semantic_code_receipt)
    release_paths = {str(row["path"]) for row in runtime["inventory"]}
    semantic_paths = {str(row["path"]) for row in semantic["units"]}
    if not semantic_paths.issubset(release_paths):
        raise ValueError("runtime release omits configuration semantic source units")
    return _component_receipt(
        "twc_configuration_release_validation_v1",
        release_hash=str(runtime["release_hash"]),
        runtime_release_receipt_hash=str(runtime["receipt_hash"]),
        semantic_code_hash=str(semantic["semantic_code_hash"]),
        semantic_code_receipt_hash=str(semantic["receipt_hash"]),
        runtime_release=runtime,
        semantic_code=semantic,
    )


def _load_release_validation(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_component_receipt(
        payload,
        artifact_kind="twc_configuration_release_validation_v1",
        value_fields={
            "release_hash",
            "runtime_release_receipt_hash",
            "semantic_code_hash",
            "semantic_code_receipt_hash",
            "runtime_release",
            "semantic_code",
        },
    )
    for name in (
        "release_hash",
        "runtime_release_receipt_hash",
        "semantic_code_hash",
        "semantic_code_receipt_hash",
    ):
        _require_hash(value[name], name)
    rebuilt = build_configuration_release_validation_receipt(
        runtime_release_receipt=value["runtime_release"],
        semantic_code_receipt=value["semantic_code"],
    )
    if rebuilt != value:
        raise ValueError("configuration runtime release validation is noncanonical")
    return rebuilt


def build_configuration_data_readiness_receipt(
    *,
    data_placement_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the complete canonical placement identity/readback receipt."""

    placement = _load_data_placement_receipt(data_placement_receipt)
    return _component_receipt(
        "twc_configuration_data_readiness_v1",
        data_manifest_hash=str(placement["pack_manifest_hash"]),
        placement_hash=str(placement["receipt_hash"]),
        verified_file_count=int(placement["verified_file_count"]),
        data_placement=placement,
    )


def _load_data_placement_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    from pathlib import Path, PurePosixPath

    expected = {
        "artifact_kind",
        "schema_version",
        "pack_manifest_hash",
        "verified_file_count",
        "full_destination_readback",
        "destination_root",
        "destination_device_id",
        "file_identities",
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "receipt_hash",
    }
    unsigned = {key: value for key, value in payload.items() if key != "receipt_hash"}
    identities = payload.get("file_identities")
    if (
        set(payload) != expected
        or payload.get("artifact_kind") != "twc_data_placement_receipt_v1"
        or payload.get("schema_version") != 1
        or payload.get("receipt_hash") != stable_json_hash(unsigned)
        or payload.get("full_destination_readback") is not True
        or payload.get("raw_text_serialized") is not False
        or payload.get("raw_url_serialized") is not False
        or payload.get("token_serialized") is not False
        or not isinstance(identities, list)
        or not identities
        or payload.get("verified_file_count") != len(identities)
    ):
        raise ValueError("configuration data placement receipt is invalid or stale")
    _require_hash(payload.get("pack_manifest_hash"), "placement pack manifest hash")
    destination = payload.get("destination_root")
    destination_device = payload.get("destination_device_id")
    if (
        not isinstance(destination, str)
        or not Path(destination).is_absolute()
        or type(destination_device) is not int
        or destination_device < 0
    ):
        raise ValueError("configuration data placement identity is invalid")
    identity_fields = {
        "relative_path",
        "device_id",
        "inode",
        "byte_count",
        "mtime_ns",
        "sha256",
    }
    for row in identities:
        relative = row.get("relative_path") if isinstance(row, Mapping) else None
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != identity_fields
            or path is None
            or path.is_absolute()
            or path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in path.parts)
            or type(row.get("device_id")) is not int
            or row.get("device_id") != destination_device
            or type(row.get("inode")) is not int
            or int(row["inode"]) < 0
            or type(row.get("byte_count")) is not int
            or int(row["byte_count"]) < 0
            or type(row.get("mtime_ns")) is not int
            or int(row["mtime_ns"]) < 0
        ):
            raise ValueError("configuration data placement file identity is invalid")
        _require_hash(row.get("sha256"), "placement file sha256")
    return dict(payload)


def _load_data_readiness(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_component_receipt(
        payload,
        artifact_kind="twc_configuration_data_readiness_v1",
        value_fields={
            "data_manifest_hash",
            "placement_hash",
            "verified_file_count",
            "data_placement",
        },
    )
    for name in ("data_manifest_hash", "placement_hash"):
        _require_hash(value[name], name)
    rebuilt = build_configuration_data_readiness_receipt(
        data_placement_receipt=value["data_placement"]
    )
    if rebuilt != value:
        raise ValueError("configuration data readiness is noncanonical")
    return rebuilt


def build_configuration_operator_rehearsal_receipt(
    *,
    engineering_verdict: Mapping[str, Any],
    canary_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an accepted current-release G1 verdict to its exact empirical chain."""

    from twc.pretraining.training.operator import CanaryReceipt

    verdict = validate_accepted_engineering_canary_verdict(engineering_verdict)
    canary = CanaryReceipt.from_dict(dict(canary_receipt))
    canary.validate(
        str(verdict["run_identity"]),
        str(verdict["data_manifest_hash"]),
        minimum_gpu_utilization=0.0,
        maximum_input_wait_fraction=1.0,
    )
    expected_canary_lineage = {
        "empirical_receipt_hash": canary.receipt_hash,
        "run_identity": canary.run_identity,
        "data_manifest_hash": canary.data_manifest_hash,
        "device_execution_plan_hash": canary.device_execution_plan_hash,
        "host_identity_digest": canary.host_identity_digest,
        "environment_hash": canary.environment_hash,
    }
    for field_name, value in expected_canary_lineage.items():
        if verdict[field_name] != value:
            raise ValueError(
                f"engineering verdict {field_name} does not match the empirical canary"
            )
    return _component_receipt(
        "twc_configuration_operator_rehearsal_v2",
        schema_version=2,
        run_identity=canary.run_identity,
        data_manifest_hash=canary.data_manifest_hash,
        environment_hash=canary.environment_hash,
        release_hash=str(verdict["release_hash"]),
        capability_hash=str(verdict["capability_hash"]),
        device_execution_plan_hash=canary.device_execution_plan_hash,
        host_identity_digest=canary.host_identity_digest,
        canary_receipt_hash=canary.receipt_hash,
        engineering_verdict_hash=str(verdict["verdict_hash"]),
        event_chain_hash=canary.event_chain_hash,
        stop_resume_verified=canary.stop_resume_verified,
        engineering_verdict=verdict,
        canary=canary.to_dict(),
    )


def _load_operator_rehearsal(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_component_receipt(
        payload,
        artifact_kind="twc_configuration_operator_rehearsal_v2",
        schema_version=2,
        value_fields={
            "run_identity",
            "data_manifest_hash",
            "environment_hash",
            "release_hash",
            "capability_hash",
            "device_execution_plan_hash",
            "host_identity_digest",
            "canary_receipt_hash",
            "engineering_verdict_hash",
            "event_chain_hash",
            "stop_resume_verified",
            "engineering_verdict",
            "canary",
        },
    )
    for name in (
        "run_identity",
        "data_manifest_hash",
        "environment_hash",
        "release_hash",
        "capability_hash",
        "device_execution_plan_hash",
        "host_identity_digest",
        "canary_receipt_hash",
        "engineering_verdict_hash",
        "event_chain_hash",
    ):
        _require_hash(value[name], name)
    rebuilt = build_configuration_operator_rehearsal_receipt(
        engineering_verdict=value["engineering_verdict"],
        canary_receipt=value["canary"],
    )
    if rebuilt != value:
        raise ValueError("configuration operator rehearsal is noncanonical")
    return rebuilt


@dataclass(frozen=True, slots=True)
class ConfigurationStageReadinessV1:
    """Post-shape, pre-scientific-stage gate on the already-rented GPU path."""

    path_stage: str
    study_plan_hash: str
    comparison_spec_hash: str
    execution_path_hash: str
    release_hash: str
    semantic_code_hash: str
    data_manifest_hash: str
    placement_hash: str
    release_validation_hash: str
    data_readiness_hash: str
    operator_rehearsal_hash: str
    shape_profile_ids: tuple[str, ...]
    shape_copyback_readback_receipt_hashes: tuple[str, ...]
    shape_bundle_hashes: tuple[str, ...]
    shape_result_hashes: tuple[str, ...]
    target_diagnostics_remained_sealed: bool
    _execution_path_json: str = field(repr=False, compare=True)
    _release_validation_json: str = field(repr=False, compare=True)
    _data_readiness_json: str = field(repr=False, compare=True)
    _operator_rehearsal_json: str = field(repr=False, compare=True)
    _shape_copyback_readbacks_json: str = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        from twc.pretraining.configuration_study_execution import (
            load_configuration_execution_path_receipt,
        )

        path = load_configuration_execution_path_receipt(json.loads(self._execution_path_json))
        release = _load_release_validation(json.loads(self._release_validation_json))
        data = _load_data_readiness(json.loads(self._data_readiness_json))
        rehearsal = _load_operator_rehearsal(json.loads(self._operator_rehearsal_json))
        shape_rows_raw = json.loads(self._shape_copyback_readbacks_json)
        if not isinstance(shape_rows_raw, list):
            raise ValueError("configuration shape copyback readback list is invalid")
        shape_rows = tuple(
            load_configuration_shape_copyback_readback(row) for row in shape_rows_raw
        )
        selection = path.to_dict()["shape_selection"]
        expected_profiles = tuple(selection["profile_ids"])
        expected_lineage = tuple(selection["profile_lineage"])
        expected_bundle_hashes = tuple(row["bundle_hash"] for row in expected_lineage)
        expected_result_groups = tuple(
            tuple(
                row["receipt_hash"]
                for row in selection["probe_results"]
                if row["profile_id"] == profile_id
            )
            for profile_id in expected_profiles
        )
        if len(shape_rows) != 2:
            raise ValueError("configuration readiness requires both shape-profile copybacks")
        shape_by_bundle = {row.bundle_hash: row for row in shape_rows}
        if len(shape_by_bundle) != 2 or set(shape_by_bundle) != set(expected_bundle_hashes):
            raise ValueError("configuration readiness shape bundle copybacks are incomplete")
        ordered_shape = tuple(shape_by_bundle[value] for value in expected_bundle_hashes)
        if tuple(row.result_hashes for row in ordered_shape) != expected_result_groups:
            raise ValueError("configuration readiness does not reconstruct the exact six shape points")
        for row, lineage in zip(ordered_shape, expected_lineage, strict=True):
            if (
                row.campaign_hash != selection["campaign_hash"]
                or row.campaign_wave_id != selection["campaign_wave_id"]
                or row.campaign_attempt_id != lineage["campaign_attempt_id"]
            ):
                raise ValueError(
                    "configuration readiness shape destination readback campaign lineage mismatch"
                )
        expected = {
            "path_stage": path.stage,
            "study_plan_hash": path.study_plan_hash,
            "comparison_spec_hash": path.comparison_spec_hash,
            "execution_path_hash": path.identity,
            "release_hash": path.release_hash,
            "semantic_code_hash": release["semantic_code_hash"],
            "data_manifest_hash": path.data_manifest_hash,
            "placement_hash": path.placement_hash,
            "release_validation_hash": release["receipt_hash"],
            "data_readiness_hash": data["receipt_hash"],
            "operator_rehearsal_hash": rehearsal["receipt_hash"],
            "shape_profile_ids": expected_profiles,
            "shape_copyback_readback_receipt_hashes": tuple(
                row.identity for row in ordered_shape
            ),
            "shape_bundle_hashes": expected_bundle_hashes,
            "shape_result_hashes": tuple(
                result_hash for group in expected_result_groups for result_hash in group
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"configuration stage readiness {name} mismatch")
        if (
            release["release_hash"] != path.release_hash
            or data["data_manifest_hash"] != path.data_manifest_hash
            or data["placement_hash"] != path.placement_hash
            or rehearsal["release_hash"] != path.release_hash
            or rehearsal["environment_hash"] != path.environment_hash
            or rehearsal["capability_hash"] != path.device_identity_hash
        ):
            raise ValueError("configuration stage readiness component lineage mismatch")
        if self.target_diagnostics_remained_sealed is not True:
            raise ValueError("configuration target diagnostics must remain sealed")

    @property
    def identity(self) -> str:
        return stable_json_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_stage_readiness_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "tokens_serialized": False,
            "secrets_serialized": False,
            **{
                key: value for key, value in asdict(self).items() if not key.startswith("_")
            },
            "execution_path": json.loads(self._execution_path_json),
            "release_validation": json.loads(self._release_validation_json),
            "data_readiness": json.loads(self._data_readiness_json),
            "operator_rehearsal": json.loads(self._operator_rehearsal_json),
            "shape_copyback_readbacks": json.loads(
                self._shape_copyback_readbacks_json
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["readiness_hash"] = stable_json_hash(payload)
        return payload


def build_configuration_stage_readiness(
    *,
    execution_path_receipt: ConfigurationExecutionPathReceiptV1,
    release_validation: Mapping[str, Any],
    data_readiness: Mapping[str, Any],
    operator_rehearsal: Mapping[str, Any],
    shape_copyback_readbacks: tuple[
        ConfigurationShapeCopybackReadbackV1 | Mapping[str, Any], ...
    ],
) -> ConfigurationStageReadinessV1:
    if not isinstance(execution_path_receipt, ConfigurationExecutionPathReceiptV1):
        raise ValueError("configuration readiness requires an execution path receipt")
    release = _load_release_validation(release_validation)
    data = _load_data_readiness(data_readiness)
    rehearsal = _load_operator_rehearsal(operator_rehearsal)
    rows = tuple(
        load_configuration_shape_copyback_readback(
            row.to_dict()
            if isinstance(row, ConfigurationShapeCopybackReadbackV1)
            else row
        )
        for row in shape_copyback_readbacks
    )
    selection = execution_path_receipt.to_dict()["shape_selection"]
    profiles = tuple(selection["profile_ids"])
    lineage = tuple(selection["profile_lineage"])
    bundle_hashes = tuple(row["bundle_hash"] for row in lineage)
    by_bundle = {row.bundle_hash: row for row in rows}
    if len(rows) != 2 or len(by_bundle) != 2 or set(by_bundle) != set(bundle_hashes):
        raise ValueError("configuration readiness requires exact two-profile shape copybacks")
    ordered = tuple(by_bundle[value] for value in bundle_hashes)
    result_groups = tuple(
        tuple(
            row["receipt_hash"]
            for row in selection["probe_results"]
            if row["profile_id"] == profile_id
        )
        for profile_id in profiles
    )
    if tuple(row.result_hashes for row in ordered) != result_groups:
        raise ValueError("configuration readiness requires exact six-point shape copyback")
    for row, lineage_row in zip(ordered, lineage, strict=True):
        if (
            row.campaign_hash != selection["campaign_hash"]
            or row.campaign_wave_id != selection["campaign_wave_id"]
            or row.campaign_attempt_id != lineage_row["campaign_attempt_id"]
        ):
            raise ValueError(
                "configuration readiness requires exact shape campaign destination readbacks"
            )
    return ConfigurationStageReadinessV1(
        path_stage=execution_path_receipt.stage,
        study_plan_hash=execution_path_receipt.study_plan_hash,
        comparison_spec_hash=execution_path_receipt.comparison_spec_hash,
        execution_path_hash=execution_path_receipt.identity,
        release_hash=execution_path_receipt.release_hash,
        semantic_code_hash=str(release["semantic_code_hash"]),
        data_manifest_hash=execution_path_receipt.data_manifest_hash,
        placement_hash=execution_path_receipt.placement_hash,
        release_validation_hash=str(release["receipt_hash"]),
        data_readiness_hash=str(data["receipt_hash"]),
        operator_rehearsal_hash=str(rehearsal["receipt_hash"]),
        shape_profile_ids=profiles,
        shape_copyback_readback_receipt_hashes=tuple(
            row.identity for row in ordered
        ),
        shape_bundle_hashes=bundle_hashes,
        shape_result_hashes=tuple(value for group in result_groups for value in group),
        target_diagnostics_remained_sealed=True,
        _execution_path_json=str(_canonical_json(execution_path_receipt.to_dict())),
        _release_validation_json=str(_canonical_json(release)),
        _data_readiness_json=str(_canonical_json(data)),
        _operator_rehearsal_json=str(_canonical_json(rehearsal)),
        _shape_copyback_readbacks_json=json.dumps(
            [row.to_dict() for row in ordered],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    )


def load_configuration_stage_readiness(
    payload: Mapping[str, Any],
) -> ConfigurationStageReadinessV1:
    public_fields = {
        name
        for name in ConfigurationStageReadinessV1.__dataclass_fields__
        if not name.startswith("_")
    }
    embedded = {
        "execution_path",
        "release_validation",
        "data_readiness",
        "operator_rehearsal",
        "shape_copyback_readbacks",
    }
    metadata = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "raw_text_serialized",
        "raw_url_serialized",
        "tokens_serialized",
        "secrets_serialized",
        "readiness_hash",
    }
    if set(payload) != public_fields | embedded | metadata:
        raise ValueError("configuration stage readiness fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_configuration_stage_readiness_v1"
        or payload.get("schema_version") != 1
        or any(
            payload.get(name) is not False
            for name in metadata
            - {"artifact_kind", "schema_version", "readiness_hash"}
        )
        or payload.get("readiness_hash")
        != stable_json_hash(
            {key: value for key, value in payload.items() if key != "readiness_hash"}
        )
    ):
        raise ValueError("configuration stage readiness boundary or hash is invalid")
    values = {name: payload[name] for name in public_fields}
    for name in (
        "shape_profile_ids",
        "shape_copyback_readback_receipt_hashes",
        "shape_bundle_hashes",
        "shape_result_hashes",
    ):
        if not isinstance(values[name], (list, tuple)):
            raise ValueError(f"configuration stage readiness {name} is invalid")
        values[name] = tuple(values[name])
    return ConfigurationStageReadinessV1(
        **values,
        _execution_path_json=str(_canonical_json(payload["execution_path"])),
        _release_validation_json=str(_canonical_json(payload["release_validation"])),
        _data_readiness_json=str(_canonical_json(payload["data_readiness"])),
        _operator_rehearsal_json=str(_canonical_json(payload["operator_rehearsal"])),
        _shape_copyback_readbacks_json=json.dumps(
            payload["shape_copyback_readbacks"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    )


@dataclass(frozen=True, slots=True)
class ConfigurationStageAuthorizationV1:
    """Immutable companion proving one exact stage campaign may be opened."""

    stage: str
    campaign_hash: str
    study_plan_hash: str
    comparison_spec_hash: str
    release_hash: str
    execution_path_hash: str
    condition_ids: tuple[str, ...]
    provider_offer_hash: str
    provider_offer_observed_at_epoch: int
    authorized_at_epoch: int
    authorization_max_age_seconds: int
    required_budget_usd: str
    wallet_balance_usd: str
    readiness_hash: str
    previous_stage_decision_hash: str | None
    c3b_requirement_hash: str | None
    target_diagnostics_remained_sealed: bool
    _provider_offer_json: str = field(repr=False, compare=True)
    _readiness_json: str = field(repr=False, compare=True)
    _previous_stage_decision_json: str | None = field(repr=False, compare=True)
    _c3b_requirement_json: str | None = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        if self.stage not in {"c2", "c3a", "c3b"}:
            raise ValueError("configuration stage authorization stage is invalid")
        for name in (
            "campaign_hash",
            "study_plan_hash",
            "comparison_spec_hash",
            "release_hash",
            "execution_path_hash",
            "provider_offer_hash",
            "readiness_hash",
        ):
            _require_hash(getattr(self, name), name)
        if not self.condition_ids or len(self.condition_ids) != len(set(self.condition_ids)):
            raise ValueError("configuration stage authorization conditions are invalid")
        for index, condition_id in enumerate(self.condition_ids):
            _require_identifier(condition_id, f"condition_ids[{index}]")
        if type(self.provider_offer_observed_at_epoch) is not int or type(
            self.authorized_at_epoch
        ) is not int:
            raise ValueError("configuration stage authorization timestamps are invalid")
        if self.authorization_max_age_seconds != _AUTHORIZATION_MAX_AGE_SECONDS:
            raise ValueError("configuration stage authorization freshness policy mismatch")
        age = self.authorized_at_epoch - self.provider_offer_observed_at_epoch
        if not 0 <= age <= self.authorization_max_age_seconds:
            raise ValueError("configuration stage authorization provider evidence is not fresh")
        required = canonical_money(self.required_budget_usd, "required stage budget", positive=True)
        wallet = canonical_money(self.wallet_balance_usd, "provider wallet balance")
        if wallet < required:
            raise ValueError("configuration stage authorization wallet is below campaign budget")
        if self.target_diagnostics_remained_sealed is not True:
            raise ValueError("configuration target diagnostics must remain sealed")
        try:
            offer_payload = json.loads(self._provider_offer_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("configuration stage authorization provider offer is invalid") from error
        if not isinstance(offer_payload, Mapping):
            raise ValueError("configuration stage authorization provider offer is invalid")
        offer = validate_configuration_provider_offer(offer_payload)
        if self._provider_offer_json != _canonical_json(offer):
            raise ValueError("configuration stage authorization provider offer is noncanonical")
        if (
            offer["receipt_hash"] != self.provider_offer_hash
            or offer["observed_at_epoch"] != self.provider_offer_observed_at_epoch
            or offer["wallet_balance_usd"] != self.wallet_balance_usd
            or offer["is_spot"] is not False
            or offer["gpu_count"] != 1
            or offer["persistent_disks"] != 0
            or offer["active_pods"] not in {0, 1}
        ):
            raise ValueError("configuration stage authorization provider readiness mismatch")
        try:
            readiness_payload = json.loads(self._readiness_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("configuration stage authorization readiness is invalid") from error
        if not isinstance(readiness_payload, Mapping):
            raise ValueError("configuration stage authorization readiness is invalid")
        readiness = load_configuration_stage_readiness(readiness_payload)
        expected_path_stage = "c2" if self.stage == "c2" else "c3"
        if (
            readiness.identity != self.readiness_hash
            or readiness.path_stage != expected_path_stage
            or readiness.study_plan_hash != self.study_plan_hash
            or readiness.comparison_spec_hash != self.comparison_spec_hash
            or readiness.execution_path_hash != self.execution_path_hash
            or readiness.release_hash != self.release_hash
            or readiness.target_diagnostics_remained_sealed is not True
        ):
            raise ValueError("configuration stage authorization readiness lineage mismatch")
        if self.stage == "c2":
            if any(
                value is not None
                for value in (
                    self.previous_stage_decision_hash,
                    self._previous_stage_decision_json,
                    self.c3b_requirement_hash,
                    self._c3b_requirement_json,
                )
            ):
                raise ValueError("C2 authorization cannot carry predecessor evidence")
        elif self.previous_stage_decision_hash is None or self._previous_stage_decision_json is None:
            raise ValueError("C3 authorization requires previous-stage decision evidence")
        if self.stage != "c3b" and any(
            value is not None
            for value in (self.c3b_requirement_hash, self._c3b_requirement_json)
        ):
            raise ValueError("only C3b authorization may carry a C3b requirement")
        if self.stage == "c3b" and (
            self.c3b_requirement_hash is None or self._c3b_requirement_json is None
        ):
            raise ValueError("C3b authorization requires exact branch evidence")
        for digest, encoded, label in (
            (
                self.previous_stage_decision_hash,
                self._previous_stage_decision_json,
                "previous stage decision",
            ),
            (self.c3b_requirement_hash, self._c3b_requirement_json, "C3b requirement"),
        ):
            if digest is None:
                continue
            _require_hash(digest, f"{label} hash")
            try:
                nested = json.loads(str(encoded))
            except json.JSONDecodeError as error:
                raise ValueError(f"configuration {label} is invalid") from error
            if not isinstance(nested, Mapping) or stable_json_hash(nested) != digest:
                # Receipt artifacts with an explicit receipt hash use that exact identity.
                nested_identity = (
                    nested.get("receipt_hash")
                    or nested.get("decision_evidence_hash")
                    or nested.get("requirement_hash")
                    if isinstance(nested, Mapping)
                    else None
                )
                if nested_identity != digest:
                    raise ValueError(f"configuration {label} hash mismatch")

    @property
    def identity(self) -> str:
        return stable_json_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_stage_authorization_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "tokens_serialized": False,
            "secrets_serialized": False,
            **{
                key: value
                for key, value in asdict(self).items()
                if not key.startswith("_")
            },
            "provider_offer": json.loads(self._provider_offer_json),
            "readiness": json.loads(self._readiness_json),
            "previous_stage_decision": (
                json.loads(self._previous_stage_decision_json)
                if self._previous_stage_decision_json is not None
                else None
            ),
            "c3b_requirement": (
                json.loads(self._c3b_requirement_json)
                if self._c3b_requirement_json is not None
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["authorization_hash"] = stable_json_hash(payload)
        return payload


def load_configuration_stage_authorization(
    payload: Mapping[str, Any],
) -> ConfigurationStageAuthorizationV1:
    public_fields = {
        key
        for key in ConfigurationStageAuthorizationV1.__dataclass_fields__
        if not key.startswith("_")
    }
    metadata = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "raw_text_serialized",
        "raw_url_serialized",
        "tokens_serialized",
        "secrets_serialized",
        "provider_offer",
        "readiness",
        "previous_stage_decision",
        "c3b_requirement",
        "authorization_hash",
    }
    if set(payload) != public_fields | metadata:
        raise ValueError("configuration stage authorization fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_configuration_stage_authorization_v1"
        or payload.get("schema_version") != 1
        or any(
            payload.get(name) is not False
            for name in (
                "production_unlock_eligible",
                "f1_unlock_eligible",
                "raw_text_serialized",
                "raw_url_serialized",
                "tokens_serialized",
                "secrets_serialized",
            )
        )
        or payload.get("authorization_hash")
        != stable_json_hash(
            {key: value for key, value in payload.items() if key != "authorization_hash"}
        )
    ):
        raise ValueError(
            "configuration stage authorization boundary or derived evidence mismatch"
        )
    values = {key: payload[key] for key in public_fields}
    if not isinstance(values["condition_ids"], (list, tuple)):
        raise ValueError("configuration stage authorization condition IDs are invalid")
    values["condition_ids"] = tuple(values["condition_ids"])
    return ConfigurationStageAuthorizationV1(
        **values,
        _provider_offer_json=str(_canonical_json(payload["provider_offer"])),
        _readiness_json=str(_canonical_json(payload["readiness"])),
        _previous_stage_decision_json=_canonical_json(
            payload["previous_stage_decision"]
        ),
        _c3b_requirement_json=_canonical_json(payload["c3b_requirement"]),
    )


_PROVIDER_OFFER_TERM_FIELDS = (
    "provider",
    "location",
    "offer_id",
    "cloud_id",
    "gpu_type",
    "gpu_count",
    "gpu_memory_gb",
    "vcpus",
    "memory_gb",
    "ephemeral_disk_gb_min",
    "price_per_hour_usd",
)


def validate_configuration_attempt_launch(
    *,
    ledger: CampaignLedger,
    stage_authorization: ConfigurationStageAuthorizationV1,
    execution_binding: ExperimentExecutionBindingV1,
    current_provider_offer: Mapping[str, Any],
    launch_at_epoch: int,
) -> dict[str, Any]:
    """Revalidate one stage authorization immediately before paid launch."""

    if not isinstance(ledger, CampaignLedger):
        raise ValueError("configuration launch requires a typed campaign ledger")
    if not isinstance(stage_authorization, ConfigurationStageAuthorizationV1):
        raise ValueError("configuration launch requires typed stage authorization")
    if not isinstance(execution_binding, ExperimentExecutionBindingV1):
        raise ValueError("configuration launch requires a typed execution binding")
    authorization = load_configuration_stage_authorization(
        stage_authorization.to_dict()
    )
    binding = execution_binding
    plan = ledger.plan
    expected_campaign_id = f"twc-{authorization.stage}-configuration-v2"
    expected_wave_ids = {
        f"{authorization.stage}-primary",
        f"{authorization.stage}-fresh-retry",
    }
    if (
        plan.campaign_id != expected_campaign_id
        or plan.identity != authorization.campaign_hash
        or binding.campaign_hash != plan.identity
    ):
        raise ValueError("configuration launch uses a foreign campaign authorization")
    if binding.stage_authorization_hash != authorization.identity:
        raise ValueError("configuration launch authorization hash mismatch")
    if (
        binding.release_hash != authorization.release_hash
        or binding.execution_path_hash != authorization.execution_path_hash
    ):
        raise ValueError("configuration launch authorization lineage mismatch")
    if binding.wave_id not in expected_wave_ids:
        raise ValueError("configuration launch wave is foreign to the authorized stage")
    try:
        wave = plan.wave(binding.wave_id)
        planned = wave.attempt(binding.attempt_id)
    except (KeyError, ValueError) as error:
        raise ValueError(
            "configuration launch wave or attempt is foreign to the campaign"
        ) from error
    primary_ids = tuple(
        attempt.attempt_id
        for attempt in plan.wave(f"{authorization.stage}-primary").attempts
    )
    if primary_ids != authorization.condition_ids:
        raise ValueError("configuration launch authorized condition set is stale")
    authorized_condition_id = (
        str(planned.predecessor_attempt_id)
        if planned.recovery_mode == "fresh_restart"
        else planned.attempt_id
    )
    if authorized_condition_id not in authorization.condition_ids:
        raise ValueError("configuration launch attempt is foreign to the authorization")
    started = ledger.attempt_start(binding.attempt_id, required=False)
    if (
        binding.wave_id not in ledger.open_wave_ids
        or started is None
        or started.wave_id != binding.wave_id
        or started.resume is not None
        or ledger.attempt_stop(binding.attempt_id, required=False) is not None
    ):
        raise ValueError(
            "configuration launch requires its exact active fresh-start attempt"
        )
    if type(launch_at_epoch) is not int or launch_at_epoch <= 0:
        raise ValueError("configuration launch time is invalid")
    start_age = launch_at_epoch - started.started_at_epoch
    if not 0 <= start_age <= _ATTEMPT_START_TO_LAUNCH_MAX_SECONDS:
        raise ValueError("configuration active attempt start is stale at paid launch")
    authorization_age = launch_at_epoch - authorization.authorized_at_epoch
    if not 0 <= authorization_age <= authorization.authorization_max_age_seconds:
        raise ValueError("configuration stage authorization expired before attempt start")

    offer = validate_configuration_provider_offer(current_provider_offer)
    authorized_offer = validate_configuration_provider_offer(
        authorization.to_dict()["provider_offer"]
    )
    if offer["is_spot"] is not False:
        raise ValueError("configuration scientific stages require an on-demand, non-spot offer")
    if any(offer[field] != authorized_offer[field] for field in _PROVIDER_OFFER_TERM_FIELDS):
        raise ValueError("configuration provider offer terms drifted after authorization")
    offer_age = launch_at_epoch - int(offer["observed_at_epoch"])
    if (
        int(offer["observed_at_epoch"])
        < authorization.provider_offer_observed_at_epoch
        or not 0 <= offer_age <= authorization.authorization_max_age_seconds
    ):
        raise ValueError("configuration provider offer expired before attempt start")
    if (
        offer["gpu_count"] != 1
        or offer["active_pods"] != 1
        or offer["persistent_disks"] != 0
    ):
        raise ValueError("configuration launch provider resource state drifted")
    quoted = canonical_money(
        format(started.quoted_hourly_usd, "f"),
        "configuration attempt quoted hourly price",
        positive=True,
    )
    current_price = canonical_money(
        offer["price_per_hour_usd"], "configuration provider hourly price", positive=True
    )
    if quoted != current_price:
        raise ValueError("configuration attempt quoted price drifted from the provider offer")
    wallet = canonical_money(offer["wallet_balance_usd"], "provider wallet balance")
    if wallet < planned.max_cost_usd:
        raise ValueError("configuration provider wallet is below the exact attempt cap")

    start_payload = {
        "request_id": started.request_id,
        "wave_id": started.wave_id,
        "attempt_id": started.attempt_id,
        "quoted_hourly_usd": format(started.quoted_hourly_usd, "f"),
        "started_at_epoch": started.started_at_epoch,
        "resume": None,
    }
    payload: dict[str, Any] = {
        "artifact_kind": "twc_configuration_attempt_launch_gate_v1",
        "schema_version": 1,
        "verified": True,
        "stage": authorization.stage,
        "campaign_hash": plan.identity,
        "wave_id": binding.wave_id,
        "attempt_id": binding.attempt_id,
        "execution_binding_hash": binding.identity,
        "stage_authorization_hash": authorization.identity,
        "attempt_start_hash": stable_json_hash(start_payload),
        "attempt_started_at_epoch": started.started_at_epoch,
        "launch_at_epoch": launch_at_epoch,
        "attempt_start_age_seconds": start_age,
        "authorization_age_seconds": authorization_age,
        "current_provider_offer_hash": offer["receipt_hash"],
        "current_provider_offer_observed_at_epoch": offer["observed_at_epoch"],
        "provider_offer_age_seconds": offer_age,
        "quoted_hourly_usd": format(started.quoted_hourly_usd, "f"),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "tokens_serialized": False,
        "secrets_serialized": False,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload
