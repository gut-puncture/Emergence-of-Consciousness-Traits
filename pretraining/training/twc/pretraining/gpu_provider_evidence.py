"""Provider-bound, no-secret evidence for a GPU campaign.

This module owns the narrow boundary between TWC campaign accounting and a
provider's live offer/resource state.  Training evidence remains provider
agnostic; only these receipts know about offers, pods, and final billing.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from twc.repro.hashing import stable_json_hash


_HEX = frozenset("0123456789abcdef")


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_hash_bound(payload: Mapping[str, Any], hash_field: str) -> None:
    expected = stable_json_hash({key: value for key, value in payload.items() if key != hash_field})
    if payload.get(hash_field) != expected:
        raise ValueError(f"{hash_field} mismatch")


def canonical_money(value: object, field: str, *, positive: bool = False) -> Decimal:
    """Parse a canonical decimal without silently accepting rounded floats."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{field} must be a canonical decimal") from error
    if (
        not parsed.is_finite()
        or parsed < 0
        or (positive and parsed <= 0)
        or parsed.as_tuple().exponent < -6
        or format(parsed, "f") != value
    ):
        raise ValueError(f"{field} must be a canonical nonnegative decimal")
    return parsed


def build_configuration_provider_offer_snapshot(
    *,
    provider: str,
    location: str,
    offer_id: str,
    cloud_id: str,
    gpu_type: str,
    gpu_count: int,
    gpu_memory_gb: int,
    vcpus: int,
    memory_gb: int,
    ephemeral_disk_gb_min: int,
    is_spot: bool,
    price_per_hour_usd: str,
    wallet_balance_usd: str,
    active_pods: int,
    persistent_disks: int,
    observed_at_epoch: int,
) -> dict[str, Any]:
    """Build the immutable snapshot that binds launch guards and billing."""
    for name, value in (
        ("provider", provider), ("location", location), ("offer_id", offer_id),
        ("cloud_id", cloud_id), ("gpu_type", gpu_type),
    ):
        if not isinstance(value, str) or not value or any(char in value for char in "\r\n\0"):
            raise ValueError(f"provider offer {name} is invalid")
    for name, value in (
        ("gpu_count", gpu_count), ("gpu_memory_gb", gpu_memory_gb), ("vcpus", vcpus),
        ("memory_gb", memory_gb), ("ephemeral_disk_gb_min", ephemeral_disk_gb_min),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"provider offer {name} is invalid")
    for name, value in (("active_pods", active_pods), ("persistent_disks", persistent_disks)):
        if type(value) is not int or value < 0:
            raise ValueError(f"provider offer {name} is invalid")
    if type(observed_at_epoch) is not int or observed_at_epoch <= 0:
        raise ValueError("provider offer observed_at_epoch is invalid")
    if type(is_spot) is not bool:
        raise ValueError("provider offer is_spot must be boolean")
    canonical_money(price_per_hour_usd, "provider hourly price", positive=True)
    canonical_money(wallet_balance_usd, "provider wallet balance")
    payload: dict[str, Any] = {
        "artifact_kind": "twc_configuration_provider_offer_snapshot_v1",
        "schema_version": 1,
        "observed_at_epoch": observed_at_epoch,
        "provider": provider,
        "location": location,
        "offer_id": offer_id,
        "cloud_id": cloud_id,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "gpu_memory_gb": gpu_memory_gb,
        "vcpus": vcpus,
        "memory_gb": memory_gb,
        "ephemeral_disk_gb_min": ephemeral_disk_gb_min,
        "is_spot": is_spot,
        "price_per_hour_usd": price_per_hour_usd,
        "wallet_balance_usd": wallet_balance_usd,
        "active_pods": active_pods,
        "persistent_disks": persistent_disks,
        "raw_text_serialized": False,
        "secrets_serialized": False,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload


def validate_configuration_provider_offer(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "artifact_kind", "schema_version", "observed_at_epoch", "provider", "location",
        "offer_id", "cloud_id", "gpu_type", "gpu_count", "gpu_memory_gb", "vcpus",
        "memory_gb", "ephemeral_disk_gb_min", "is_spot", "price_per_hour_usd",
        "wallet_balance_usd", "active_pods", "persistent_disks", "raw_text_serialized",
        "secrets_serialized", "receipt_hash",
    }
    if set(payload) != expected or payload.get("artifact_kind") != "twc_configuration_provider_offer_snapshot_v1" or payload.get("schema_version") != 1:
        raise ValueError("configuration provider offer schema is invalid")
    _validate_hash_bound(payload, "receipt_hash")
    rebuilt = build_configuration_provider_offer_snapshot(
        **{key: payload[key] for key in expected - {"artifact_kind", "schema_version", "raw_text_serialized", "secrets_serialized", "receipt_hash"}}
    )
    if rebuilt != dict(payload):
        raise ValueError("configuration provider offer semantics are invalid")
    return dict(payload)


def build_configuration_stage_billing_baseline(
    *,
    campaign_hash: str,
    provider_offer: Mapping[str, Any],
    pod_id: str,
    observed_at_epoch: int,
    provider_total_usd: str,
    active_pods: int,
    persistent_disks: int,
) -> dict[str, Any]:
    """Bind cumulative same-pod cost immediately before a scientific stage."""

    offer = validate_configuration_provider_offer(provider_offer)
    _digest(campaign_hash, "campaign_hash")
    if not isinstance(pod_id, str) or not pod_id or any(
        char in pod_id for char in "\r\n\0"
    ):
        raise ValueError("configuration stage billing baseline pod identity is invalid")
    if (
        type(observed_at_epoch) is not int
        or observed_at_epoch <= int(offer["observed_at_epoch"])
    ):
        raise ValueError("configuration stage billing baseline time is invalid")
    if active_pods != 1 or persistent_disks != 0:
        raise ValueError(
            "configuration stage billing baseline requires one pod and zero disks"
        )
    canonical_money(provider_total_usd, "provider baseline total")
    payload: dict[str, Any] = {
        "artifact_kind": "twc_configuration_stage_billing_baseline_v1",
        "schema_version": 1,
        "campaign_hash": campaign_hash,
        "provider": offer["provider"],
        "provider_offer_hash": offer["receipt_hash"],
        "pod_identity_digest": hashlib.sha256(pod_id.encode("utf-8")).hexdigest(),
        "observed_at_epoch": observed_at_epoch,
        "provider_total_usd": provider_total_usd,
        "active_pods": active_pods,
        "persistent_disks": persistent_disks,
        "raw_text_serialized": False,
        "secrets_serialized": False,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload


def validate_configuration_stage_billing_baseline(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "artifact_kind", "schema_version", "campaign_hash", "provider",
        "provider_offer_hash", "pod_identity_digest", "observed_at_epoch",
        "provider_total_usd", "active_pods", "persistent_disks",
        "raw_text_serialized", "secrets_serialized", "receipt_hash",
    }
    if (
        set(payload) != expected
        or payload.get("artifact_kind")
        != "twc_configuration_stage_billing_baseline_v1"
        or payload.get("schema_version") != 1
        or payload.get("active_pods") != 1
        or payload.get("persistent_disks") != 0
        or payload.get("raw_text_serialized") is not False
        or payload.get("secrets_serialized") is not False
    ):
        raise ValueError("configuration stage billing baseline schema is invalid")
    _validate_hash_bound(payload, "receipt_hash")
    for field in ("campaign_hash", "provider_offer_hash", "pod_identity_digest"):
        _digest(payload.get(field), field)
    if type(payload.get("observed_at_epoch")) is not int or int(
        payload["observed_at_epoch"]
    ) <= 0:
        raise ValueError("configuration stage billing baseline time is invalid")
    if not isinstance(payload.get("provider"), str) or not payload["provider"]:
        raise ValueError("configuration stage billing baseline provider is invalid")
    canonical_money(payload.get("provider_total_usd"), "provider baseline total")
    return dict(payload)


def build_configuration_provider_billing_observation(
    *,
    stage: str,
    campaign_hash: str,
    provider_offer: Mapping[str, Any],
    pod_id: str,
    observed_at_epoch: int,
    provider_total_usd: str,
    active_pods: int,
    persistent_disks: int,
    previous_observation_hash: str | None,
) -> dict[str, Any]:
    offer = validate_configuration_provider_offer(provider_offer)
    _digest(campaign_hash, "campaign_hash")
    if stage not in {"pre_teardown", "final"}:
        raise ValueError("configuration provider billing stage is invalid")
    if not isinstance(pod_id, str) or not pod_id or any(char in pod_id for char in "\r\n\0"):
        raise ValueError("configuration provider pod identity is invalid")
    if type(observed_at_epoch) is not int or observed_at_epoch <= int(offer["observed_at_epoch"]):
        raise ValueError("configuration provider billing observation time is invalid")
    if type(active_pods) is not int or active_pods < 0 or type(persistent_disks) is not int or persistent_disks < 0:
        raise ValueError("configuration provider resource counts are invalid")
    if stage == "pre_teardown" and (active_pods != 1 or persistent_disks != 0):
        raise ValueError(
            "pre-teardown provider observation requires exactly one active pod and zero disks"
        )
    if stage == "final" and (active_pods != 0 or persistent_disks != 0):
        raise ValueError("final provider observation requires zero active resources")
    canonical_money(provider_total_usd, "provider billed total")
    if previous_observation_hash is not None:
        _digest(previous_observation_hash, "previous_observation_hash")
    if (stage == "pre_teardown") != (previous_observation_hash is None):
        raise ValueError("configuration provider billing observation chain is invalid")
    payload: dict[str, Any] = {
        "artifact_kind": "twc_configuration_provider_billing_observation_v1",
        "schema_version": 1,
        "stage": stage,
        "campaign_hash": campaign_hash,
        "provider": offer["provider"],
        "provider_offer_hash": offer["receipt_hash"],
        "pod_identity_digest": hashlib.sha256(pod_id.encode("utf-8")).hexdigest(),
        "observed_at_epoch": observed_at_epoch,
        "provider_total_usd": provider_total_usd,
        "active_pods": active_pods,
        "persistent_disks": persistent_disks,
        "previous_observation_hash": previous_observation_hash,
        "raw_text_serialized": False,
        "secrets_serialized": False,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload


def validate_configuration_provider_billing_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "artifact_kind", "schema_version", "stage", "campaign_hash", "provider",
        "provider_offer_hash", "pod_identity_digest", "observed_at_epoch",
        "provider_total_usd", "active_pods", "persistent_disks",
        "previous_observation_hash", "raw_text_serialized", "secrets_serialized", "receipt_hash",
    }
    if set(payload) != expected or payload.get("artifact_kind") != "twc_configuration_provider_billing_observation_v1" or payload.get("schema_version") != 1:
        raise ValueError("configuration provider billing observation schema is invalid")
    _validate_hash_bound(payload, "receipt_hash")
    for field in ("campaign_hash", "provider_offer_hash", "pod_identity_digest"):
        _digest(payload.get(field), field)
    canonical_money(payload.get("provider_total_usd"), "provider billed total")
    if type(payload.get("observed_at_epoch")) is not int or int(payload["observed_at_epoch"]) <= 0:
        raise ValueError("configuration provider billing observation time is invalid")
    if type(payload.get("persistent_disks")) is not int or int(payload["persistent_disks"]) < 0:
        raise ValueError("configuration provider resource counts are invalid")
    if payload.get("stage") == "pre_teardown":
        if (
            payload.get("previous_observation_hash") is not None
            or payload.get("active_pods") != 1
            or payload.get("persistent_disks") != 0
        ):
            raise ValueError("pre-teardown provider observation is invalid")
    elif payload.get("stage") == "final":
        _digest(payload.get("previous_observation_hash"), "previous_observation_hash")
        if payload.get("active_pods") != 0 or payload.get("persistent_disks") != 0:
            raise ValueError("final provider observation requires zero active resources")
    else:
        raise ValueError("configuration provider billing stage is invalid")
    if payload.get("raw_text_serialized") is not False or payload.get("secrets_serialized") is not False:
        raise ValueError("configuration provider billing privacy flags are invalid")
    return dict(payload)


def finalize_configuration_provider_billing(
    *,
    closeout: Mapping[str, Any],
    final_observation: Mapping[str, Any],
    campaign_budget_cap_usd: str,
    campaign_reserve_usd: str,
    expected_campaign_hash: str,
) -> dict[str, Any]:
    """Finalize cost only after the provider reports zero campaign resources."""
    accepted_closeouts = {
        "twc_configuration_campaign_pod_teardown_authorization_v2": 2,
        "twc_configuration_stage_provider_pod_teardown_authorization_v1": 1,
        "twc_configuration_shape_abort_provider_pod_teardown_authorization_v1": 1,
    }
    if accepted_closeouts.get(str(closeout.get("artifact_kind"))) != closeout.get(
        "schema_version"
    ):
        raise ValueError("configuration closeout is not billing-finalizable")
    _validate_hash_bound(closeout, "receipt_hash")
    _digest(expected_campaign_hash, "expected_campaign_hash")
    if closeout.get("campaign_hash") != expected_campaign_hash:
        raise ValueError("configuration billing campaign does not match the loaded campaign plan")
    final = validate_configuration_provider_billing_observation(final_observation)
    if final["stage"] != "final" or any(
        final[field] != closeout[field]
        for field in ("campaign_hash", "provider", "provider_offer_hash", "pod_identity_digest")
    ) or final["previous_observation_hash"] != closeout.get("pre_teardown_provider_observation_hash"):
        raise ValueError("final provider observation pod/provider chain mismatch")
    pre_time = closeout.get("pre_teardown_observed_at_epoch")
    if type(pre_time) is not int or final["observed_at_epoch"] <= pre_time:
        raise ValueError("final provider observation must be later than pre-teardown evidence")
    billed = canonical_money(final["provider_total_usd"], "provider final total")
    pre_total = canonical_money(closeout.get("pre_teardown_provider_total_usd"), "pre-teardown provider total")
    if billed < pre_total:
        raise ValueError("final provider total cannot regress below the pre-teardown total")
    budget = canonical_money(campaign_budget_cap_usd, "campaign budget", positive=True)
    reserve = canonical_money(campaign_reserve_usd, "campaign reserve")
    scoped_billed = billed
    if closeout.get("artifact_kind") == (
        "twc_configuration_stage_provider_pod_teardown_authorization_v1"
    ):
        baseline = canonical_money(
            closeout.get("pre_science_provider_total_usd"),
            "pre-science provider total",
        )
        if billed < baseline:
            raise ValueError("final provider total regressed below the scientific baseline")
        scoped_billed = billed - baseline
    if scoped_billed + reserve > budget:
        raise ValueError("final provider cost breaches campaign reserve")
    authorized_cap = canonical_money(
        closeout.get("provider_authorized_total_cap_usd"),
        "provider authorized total cap",
        positive=True,
    )
    if billed > authorized_cap:
        raise ValueError("final provider cost exceeds the authorized campaign scope cap")
    estimate = canonical_money(closeout.get("attempt_estimated_total_usd"), "attempt estimated total")
    payload: dict[str, Any] = {
        "artifact_kind": "twc_configuration_provider_billing_finalization_v1",
        "schema_version": 1,
        "campaign_hash": final["campaign_hash"],
        "closeout_receipt_hash": closeout["receipt_hash"],
        "final_provider_observation_hash": final["receipt_hash"],
        "provider_final_total_usd": format(billed, "f"),
        "provider_authorized_total_cap_usd": format(authorized_cap, "f"),
        "attempt_estimated_total_usd": format(estimate, "f"),
        "provider_minus_attempt_estimate_usd": format(scoped_billed - estimate, "f"),
        "billing_finalized": True,
        "production_unlock": False,
        "f1_unlock": False,
        "raw_text_serialized": False,
        "secrets_serialized": False,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload
