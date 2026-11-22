"""Dependency-light loader for accepted G1 engineering-canary verdicts.

The C2/C3 readiness gate consumes an already-built G1 verdict but must not
import the campaign evidence module: that module depends on the configuration
compiler, which in turn depends on stage authorization.  This leaf owns the
strict accepted-verdict schema so both callers enforce one rule without an
import cycle.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from twc.repro.hashing import stable_json_hash


_HASH = re.compile(r"^[0-9a-f]{64}$")
_HASH_FIELDS = (
    "run_identity",
    "data_manifest_hash",
    "global_step_plan_hash",
    "device_execution_plan_hash",
    "host_identity_digest",
    "release_hash",
    "environment_hash",
    "capability_hash",
    "bootstrap_receipt_hash",
    "parity_receipt_hash",
    "empirical_receipt_hash",
    "verdict_hash",
)
_FIELDS = {
    "artifact_kind",
    "schema_version",
    "status",
    *_HASH_FIELDS,
    "blockers",
    "deferred_issues",
    "production_unlock",
    "f1_unlock",
    "raw_text_serialized",
    "secrets_serialized",
}


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(
            f"engineering verdict {label} must be a lowercase SHA-256 hash"
        )
    return value


def validate_accepted_engineering_canary_verdict(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Load a hash-valid accepted G1 verdict without granting new authority."""

    if not isinstance(payload, Mapping) or set(payload) != _FIELDS:
        raise ValueError("engineering verdict fields are invalid")
    unsigned = {key: value for key, value in payload.items() if key != "verdict_hash"}
    if (
        payload.get("artifact_kind")
        != "twc_gpu_engineering_canary_verdict_v1"
        or payload.get("schema_version") != 1
        or payload.get("verdict_hash") != stable_json_hash(unsigned)
    ):
        raise ValueError("engineering verdict schema or hash is invalid")
    for field in _HASH_FIELDS:
        _require_hash(payload.get(field), field)

    blockers = payload.get("blockers")
    deferred = payload.get("deferred_issues")
    if blockers != []:
        raise ValueError("engineering verdict contains blockers")
    if (
        not isinstance(deferred, list)
        or any(not isinstance(value, str) or not value for value in deferred)
        or len(deferred) != len(set(deferred))
    ):
        raise ValueError("engineering verdict deferred issues are invalid")
    status = payload.get("status")
    if (
        (status == "accepted" and deferred)
        or (status == "accepted_with_deferred_issues" and not deferred)
        or status not in {"accepted", "accepted_with_deferred_issues"}
    ):
        raise ValueError("engineering verdict acceptance status is noncanonical")
    if any(
        payload.get(field) is not False
        for field in (
            "production_unlock",
            "f1_unlock",
            "raw_text_serialized",
            "secrets_serialized",
        )
    ):
        raise ValueError("engineering verdict safety boundary is invalid")
    return dict(payload)
