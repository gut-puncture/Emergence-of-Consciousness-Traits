"""Canonical, format-neutral R4 -> R6 materialization handoff contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from twc.repro.hashing import stable_json_hash
from twc.repro.public_artifacts import validate_public_no_raw_payload

SELECTED_MATERIALIZATION_HANDOFF_KIND = "twc_selected_materialization_handoff_v1"


@dataclass(frozen=True)
class SelectedMaterializationFact:
    """One selected candidate, enriched with its source-natural-block span."""

    shard_number: int
    candidate_id_hash: str
    global_ordinal: int
    evidence_text_sha256: str
    core_text_sha256: str
    core_char_count: int
    core_byte_count: int
    core_proxy_token_count: int
    first_block_index: int
    last_block_index: int
    core_char_start: int
    core_char_end: int
    source_document_id_hash: str
    segmentation_receipt_hash: str


@dataclass(frozen=True)
class MaterializationInput:
    """Private adapter output bound to R4's selected index and topology roots."""

    selected_index_descriptor: Mapping[str, Any]
    topology_fact_root: str
    export_fact_root: str
    source_content_root: str
    facts: Iterable[SelectedMaterializationFact]


def validate_selected_materialization_handoff(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate R4's compact no-row source-rehydration authorization."""

    normalized = dict(payload)
    kind = normalized.get("artifact_kind")
    if kind != SELECTED_MATERIALIZATION_HANDOFF_KIND:
        if isinstance(kind, str) and "evidence_export" in kind:
            raise ValueError("evidence export is not a materialization handoff")
        raise ValueError("selected materialization handoff kind mismatch")
    validate_public_no_raw_payload(normalized)
    if type(normalized.get("schema_version")) is not int or normalized["schema_version"] != 1:
        raise ValueError("selected materialization handoff schema mismatch")
    if normalized.get("status") != "ready_for_source_rehydration":
        raise ValueError("selected materialization handoff status mismatch")
    for field in (
        "closeout_report_hash",
        "source_manifest_hash",
        "source_content_root",
        "semantic_identity_hash",
        "ledger_fact_root",
        "topology_fact_root",
        "export_fact_root",
        "selected_private_fact_index_root",
    ):
        value = normalized.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"selected materialization handoff {field} is invalid")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"selected materialization handoff {field} is invalid") from exc
    for field in ("selected_candidate_count", "selected_proxy_token_count"):
        value = normalized.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"selected materialization handoff {field} is invalid")
    if {"rows", "candidates", "accepted_rows", "source_paths"}.intersection(normalized):
        raise ValueError("selected materialization handoff must remain compact")
    if normalized.get("natural_block_content_boundaries_present") is not False:
        raise ValueError("selected materialization handoff must declare absent boundaries")
    if normalized.get("source_rehydration_required") is not True:
        raise ValueError("selected materialization handoff must require rehydration")
    for field in (
        "canonical_clean_runs_created",
        "accepted_manifest_created",
        "tokenizer_shards_created",
        "packed_shards_created",
        "training_shards_created",
        "base_pretraining_started",
        "pretraining_started",
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
    ):
        if normalized.get(field) is not False:
            raise ValueError(f"selected materialization handoff must set {field}=false")
    expected_hash = stable_json_hash(
        {key: value for key, value in normalized.items() if key != "handoff_hash"}
    )
    if normalized.get("handoff_hash") != expected_hash:
        raise ValueError("selected materialization handoff hash mismatch")
    return normalized


def _stream_root(rows: Iterable[Any]) -> tuple[str, int]:
    digest = hashlib.sha256()
    digest.update(b"[")
    count = 0
    for row in rows:
        if count:
            digest.update(b",")
        digest.update(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        )
        count += 1
    digest.update(b"]")
    return digest.hexdigest(), count


def selected_fact_stream_root(facts: Iterable[SelectedMaterializationFact]) -> tuple[str, int]:
    """Reproduce the R4 selected_lookup shard row-root algorithm."""

    return _stream_root(
        [
            fact.candidate_id_hash,
            fact.global_ordinal,
            fact.evidence_text_sha256,
            fact.core_text_sha256,
            fact.core_char_count,
            fact.core_byte_count,
            fact.core_proxy_token_count,
            fact.first_block_index,
            fact.last_block_index,
            fact.core_char_start,
            fact.core_char_end,
            fact.source_document_id_hash,
            fact.segmentation_receipt_hash,
        ]
        for fact in facts
    )


def iter_validated_materialization_facts(
    handoff: Mapping[str, Any],
    materialization_input: MaterializationInput,
    *,
    start_shard: int = 0,
) -> Iterator[SelectedMaterializationFact]:
    """Validate private selection proof before yielding any selected facts.

    The adapter must reproduce R4's exact selected-index descriptor and per-
    shard roots.  Aggregate counts alone are intentionally insufficient.
    """

    normalized = validate_selected_materialization_handoff(handoff)
    descriptor = dict(materialization_input.selected_index_descriptor)
    if stable_json_hash(descriptor) != normalized["selected_private_fact_index_root"]:
        raise ValueError("private selected-fact index root mismatch")
    if descriptor.get("artifact_kind") != "twc_filtering_closeout_selected_fact_index_v1":
        raise ValueError("unsupported private selected-fact index descriptor")
    if materialization_input.topology_fact_root != normalized["topology_fact_root"]:
        raise ValueError("materialization topology root mismatch")
    if materialization_input.export_fact_root != normalized["export_fact_root"]:
        raise ValueError("materialization export root mismatch")
    if materialization_input.source_content_root != normalized["source_content_root"]:
        raise ValueError("materialization source-content root mismatch")
    shards = descriptor.get("shards")
    if not isinstance(shards, list):
        raise ValueError("selected-fact index shards must be a list")
    expected_all = {
        int(item["shard_number"]): item for item in shards if isinstance(item, Mapping)
    }
    expected = {
        number: item for number, item in expected_all.items() if number >= start_shard
    }
    if len(expected_all) != len(shards):
        raise ValueError("selected-fact index shard descriptors are invalid")
    current_shard: int | None = None
    current_digest = hashlib.sha256()
    current_digest.update(b"[")
    current_count = 0
    seen_shards: set[int] = set()
    total = 0
    previous_key: tuple[int, str] | None = None

    def validate_shard(shard_number: int) -> None:
        nonlocal current_digest, current_count
        receipt = expected.get(shard_number)
        if receipt is None:
            raise ValueError("materialization fact references an undeclared shard")
        current_digest.update(b"]")
        if receipt.get("selected_fact_root") != current_digest.hexdigest():
            raise ValueError("selected materialization fact root mismatch")
        if receipt.get("selected_count") != current_count:
            raise ValueError("selected materialization fact count mismatch")

    facts = materialization_input.facts
    fact_stream = (
        facts.iter_from_shard(start_shard)
        if hasattr(facts, "iter_from_shard")
        else (fact for fact in facts if fact.shard_number >= start_shard)
    )
    for fact in fact_stream:
        if not isinstance(fact, SelectedMaterializationFact):
            raise ValueError("materialization facts must be SelectedMaterializationFact")
        for field in (
            "candidate_id_hash",
            "evidence_text_sha256",
            "core_text_sha256",
            "source_document_id_hash",
            "segmentation_receipt_hash",
        ):
            value = getattr(fact, field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"materialization fact {field} is invalid")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"materialization fact {field} is invalid") from exc
        for field in (
            "shard_number",
            "global_ordinal",
            "core_proxy_token_count",
            "core_char_count",
            "core_byte_count",
            "first_block_index",
            "last_block_index",
            "core_char_start",
            "core_char_end",
        ):
            value = getattr(fact, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"materialization fact {field} is invalid")
        if fact.last_block_index < fact.first_block_index:
            raise ValueError("materialization fact block span is inverted")
        if fact.core_char_end <= fact.core_char_start:
            raise ValueError("materialization fact character span is invalid")
        if fact.core_char_count != fact.core_char_end - fact.core_char_start:
            raise ValueError("materialization fact character count does not match its span")
        key = (fact.shard_number, fact.candidate_id_hash)
        if previous_key is not None and key <= previous_key:
            raise ValueError("materialization facts must be ordered by shard and candidate")
        previous_key = key
        if current_shard is None:
            current_shard = fact.shard_number
        if fact.shard_number != current_shard:
            validate_shard(current_shard)
            seen_shards.add(current_shard)
            current_shard = fact.shard_number
            current_digest = hashlib.sha256()
            current_digest.update(b"[")
            current_count = 0
        if current_count:
            current_digest.update(b",")
        current_digest.update(
            json.dumps(
                [
                    fact.candidate_id_hash,
                    fact.global_ordinal,
                    fact.evidence_text_sha256,
                    fact.core_text_sha256,
                    fact.core_char_count,
                    fact.core_byte_count,
                    fact.core_proxy_token_count,
                    fact.first_block_index,
                    fact.last_block_index,
                    fact.core_char_start,
                    fact.core_char_end,
                    fact.source_document_id_hash,
                    fact.segmentation_receipt_hash,
                ],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        current_count += 1
        total += 1
        yield fact
    if current_shard is not None:
        validate_shard(current_shard)
        seen_shards.add(current_shard)
    expected_total = sum(int(item["selected_count"]) for item in expected.values())
    if expected_total != total:
        raise ValueError("materialization selected-candidate suffix total mismatch")
    if start_shard == 0 and (
        descriptor.get("selected_count") != total
        or normalized["selected_candidate_count"] != total
    ):
        raise ValueError("materialization selected-candidate total mismatch")
    expected_nonempty = {
        number for number, item in expected.items() if int(item.get("selected_count", -1)) > 0
    }
    if seen_shards != expected_nonempty:
        raise ValueError("materialization fact shard coverage mismatch")
