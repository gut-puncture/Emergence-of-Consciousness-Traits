"""Shared validators for append-only evaluation evidence."""

from __future__ import annotations

import math
from typing import Any

from twc.repro.hashing import stable_json_hash


def validate_evaluation_chain(
    rows: list[dict[str, Any]], run_identity: str, attempt_hash: str, fencing_epoch: int
) -> dict[str, Any]:
    """Validate an attempt-local append-only, token-weighted evaluation ledger."""

    if not rows:
        raise ValueError("evaluation ledger is empty")
    expected_previous: str | None = None
    previous_tokens = -1
    expected_validation_records_hash: str | None = None
    expected_component_group_hashes: tuple[str, ...] | None = None
    v1_fields = {
        "artifact_kind", "schema_version", "run_identity", "attempt_hash", "fencing_epoch",
        "evaluation_index", "previous_receipt_hash", "at_consumed_loss_tokens", "loss",
        "loss_tokens", "batch_count", "receipt_hash",
    }
    for index, row in enumerate(rows, start=1):
        schema_version = row.get("schema_version")
        exact_fields = v1_fields | (
            {"component_losses", "component_loss_tokens"}
            if schema_version in {2, 3}
            else set()
        ) | (
            {"component_group_hashes", "validation_records_hash"}
            if schema_version == 3
            else set()
        )
        expected_kind = f"twc_pretraining_evaluation_result_v{schema_version}"
        if schema_version not in {1, 2, 3} or set(row) != exact_fields:
            raise ValueError("evaluation ledger row fields are invalid")
        if (
            row.get("artifact_kind") != expected_kind
            or row.get("run_identity") != run_identity
            or row.get("attempt_hash") != attempt_hash
            or row.get("fencing_epoch") != fencing_epoch
            or row.get("evaluation_index") != index
            or row.get("previous_receipt_hash") != expected_previous
        ):
            raise ValueError("evaluation ledger identity or chain is invalid")
        expected_hash = stable_json_hash(
            {key: value for key, value in row.items() if key != "receipt_hash"}
        )
        if row.get("receipt_hash") != expected_hash:
            raise ValueError("evaluation ledger receipt hash is invalid")
        at_tokens = row.get("at_consumed_loss_tokens")
        loss_tokens = row.get("loss_tokens")
        batch_count = row.get("batch_count")
        loss = row.get("loss")
        if (
            type(at_tokens) is not int or at_tokens <= previous_tokens
            or type(loss_tokens) is not int or loss_tokens <= 0
            or type(batch_count) is not int or batch_count <= 0
            or type(loss) not in (int, float) or not math.isfinite(float(loss)) or loss < 0
        ):
            raise ValueError("evaluation ledger metrics are invalid")
        if schema_version in {2, 3}:
            component_losses = row.get("component_losses")
            component_tokens = row.get("component_loss_tokens")
            if (
                not isinstance(component_losses, list)
                or not isinstance(component_tokens, list)
                or len(component_losses) != batch_count
                or len(component_tokens) != batch_count
                or any(
                    type(value) not in {int, float}
                    or not math.isfinite(value)
                    or value < 0
                    for value in component_losses
                )
                or any(type(value) is not int or value <= 0 for value in component_tokens)
                or sum(component_tokens) != loss_tokens
                or not math.isclose(
                    sum(
                        value * weight
                        for value, weight in zip(
                            component_losses, component_tokens, strict=True
                        )
                    ) / loss_tokens,
                    float(loss),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("evaluation ledger component metrics are invalid")
        if schema_version == 3:
            component_group_hashes = row.get("component_group_hashes")
            validation_records_hash = row.get("validation_records_hash")
            if (
                not isinstance(component_group_hashes, list)
                or len(component_group_hashes) != batch_count
                or len(component_group_hashes) != len(set(component_group_hashes))
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in component_group_hashes
                )
                or not isinstance(validation_records_hash, str)
                or len(validation_records_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in validation_records_hash
                )
            ):
                raise ValueError("evaluation ledger component identities are invalid")
            group_tuple = tuple(component_group_hashes)
            if expected_validation_records_hash is None:
                expected_validation_records_hash = validation_records_hash
                expected_component_group_hashes = group_tuple
            elif (
                validation_records_hash != expected_validation_records_hash
                or group_tuple != expected_component_group_hashes
            ):
                raise ValueError("evaluation ledger component identities changed")
        expected_previous = expected_hash
        previous_tokens = at_tokens
    return rows[-1]
