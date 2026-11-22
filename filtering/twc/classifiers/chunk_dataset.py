"""Chunk-level classifier dataset manifests for the TWC filtering subsystem.

This module is deliberately no-raw and IO-light. It turns already-produced
chunk label evidence into a public classifier manifest that downstream trainers
can bind to private text refetch paths without leaking text, URLs, token IDs, or
source locators.
"""

from __future__ import annotations

import json
import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Sequence

from twc.classifiers.final_labels import plan_dedupe_aware_splits, validate_public_no_raw_payload
from twc.data.chunk_relabeling import build_effective_chunk_label_evidence, load_fresh_chunk_repair_overlay
from twc.data.decision_schema import stable_json_hash


CHUNK_CLASSIFIER_MANIFEST_KIND = "twc_chunk_classifier_training_manifest"
CHUNK_CLASSIFIER_SCHEMA_VERSION = 1
CHUNK_CLASSIFIER_SCHEMA_VERSION_V2 = 2
FINAL_35K_CHUNK_CLASSIFIER_V2_SOURCE_UNIVERSE = "final_35k_chunk_classifier_v2"
AUGMENTED_V2_SOURCE_UNIVERSE = "final_35k_plus_active_learning_v1"
FINAL_35K_SAFE_PARENT_COUNT = 16_938
FINAL_35K_REJECT_PARENT_COUNT = 18_062
FINAL_35K_REPAIR_OVERLAY_LABEL_COUNT = 12
STRONG_EVIDENCE_TIER = "strong_chunk"
FRESH_REPAIR_EVIDENCE_TIER = "fresh_repair_overlay"
WEAK_SAFE_EVIDENCE_TIER = "weak_parent_safe"
INHERITED_PARENT_SAFE_EVIDENCE_TIER = "inherited_parent_safe_v2"
ACTIVE_LEARNING_EVIDENCE_TIER = "active_learning_bof3_chunk_label"
ACTIVE_LEARNING_PRIVATE_CACHE_ROUTE = "active_learning_private_refetch"
CORE_TEXT_CONFLICT_POLICY_HARD_FAIL = "hard_fail"
CORE_TEXT_CONFLICT_POLICY_AUDIT_COMPONENTS = "audit_conflicting_components"
CORE_TEXT_CONFLICT_POLICIES = {
    CORE_TEXT_CONFLICT_POLICY_HARD_FAIL,
    CORE_TEXT_CONFLICT_POLICY_AUDIT_COMPONENTS,
}
CORE_TEXT_CONFLICT_AUDIT_REASON = "core_text_label_conflict_component"
ACTIVE_INHERITED_CONFLICT_POLICY_HARD_FAIL = "hard_fail"
ACTIVE_INHERITED_CONFLICT_POLICY_AUDIT_COMPONENTS = "audit_conflicting_components"
ACTIVE_INHERITED_CONFLICT_POLICIES = {
    ACTIVE_INHERITED_CONFLICT_POLICY_HARD_FAIL,
    ACTIVE_INHERITED_CONFLICT_POLICY_AUDIT_COMPONENTS,
}
ACTIVE_INHERITED_CONFLICT_AUDIT_REASON = "active_reject_inherited_safe_conflict_component"
WEAK_SAFE_TRAINING_WEIGHT = 0.25
ALLOWED_LABELS = {"safe", "reject"}
ALLOWED_SPLITS = {"train", "calibration", "eval", "audit"}
ALLOWED_V2_SOURCE_UNIVERSES = {FINAL_35K_CHUNK_CLASSIFIER_V2_SOURCE_UNIVERSE, AUGMENTED_V2_SOURCE_UNIVERSE}
ACTIVE_LEARNING_PROBE_ID = "production_chunk_active_learning_probe_v1"
ACTIVE_LEARNING_CANONICAL_BUCKETS = (
    "random_production_baseline",
    "near_boundary",
    "model_disagreement",
    "likely_overrejections",
    "risky_safe_likely_missed_reject",
)
LEAKAGE_IDENTITY_FIELDS = (
    "source_document_identity_hash",
    "parent_candidate_id",
    "physical_source_row_key_hash",
    "parent_text_hash_sha256",
    "text_hash_sha256",
    "normalized_text_hash_sha256",
)
PUBLIC_FALSE_FLAGS = (
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "accepted_manifest",
    "labels_are_corpus_approval",
    "training_shards_created",
    "classifier_training_performed",
)
FORBIDDEN_INPUT_KEYS = {
    "raw_text",
    "text",
    "document_text",
    "normalized_text",
    "matched_text",
    "snippet",
    "raw_context",
    "raw_context_snippet",
    "raw_url",
    "source_url",
    "url",
    "tokens",
    "token_ids",
    "input_ids",
}
ACTIVE_VOTE_FORBIDDEN_KEYS = {
    "document_text",
    "matched_text",
    "messages",
    "normalized_text",
    "parquet_url_or_path",
    "prompt",
    "prompt_text",
    "raw_context",
    "raw_context_snippet",
    "raw_response",
    "raw_text",
    "raw_url",
    "request_body",
    "response_body",
    "rationale",
    "rationale_short",
    "risk_spans",
    "source_domain",
    "source_url",
    "snippet",
    "text",
}


def _manifest_source_universe(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("source_universe") or "")


def _allowed_evidence_tiers_for_source_universe(source_universe: str, *, schema_version: int) -> set[str]:
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2:
        tiers = {STRONG_EVIDENCE_TIER, FRESH_REPAIR_EVIDENCE_TIER, INHERITED_PARENT_SAFE_EVIDENCE_TIER}
        if source_universe == AUGMENTED_V2_SOURCE_UNIVERSE:
            tiers.add(ACTIVE_LEARNING_EVIDENCE_TIER)
        return tiers
    return {STRONG_EVIDENCE_TIER, FRESH_REPAIR_EVIDENCE_TIER, WEAK_SAFE_EVIDENCE_TIER}


def _expected_private_cache_route(row: Mapping[str, Any], *, schema_version: int, source_universe: str) -> str | None:
    tier = str(row.get("evidence_tier") or "")
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2:
        if tier == INHERITED_PARENT_SAFE_EVIDENCE_TIER:
            return "safe_parent_source_refetch"
        if tier in {STRONG_EVIDENCE_TIER, FRESH_REPAIR_EVIDENCE_TIER}:
            return "rejected_chunk_stage_private_map"
        if source_universe == AUGMENTED_V2_SOURCE_UNIVERSE and tier == ACTIVE_LEARNING_EVIDENCE_TIER:
            return ACTIVE_LEARNING_PRIVATE_CACHE_ROUTE
    return None


def build_chunk_classifier_training_manifest(
    *,
    chunk_candidates: Iterable[Mapping[str, Any]],
    effective_label_evidence: Mapping[str, Any],
    weak_safe_candidates: Iterable[Mapping[str, Any]] = (),
    inherited_safe_candidates: Iterable[Mapping[str, Any]] = (),
    split_seed: str = "twc-chunk-classifier-v1",
    source_artifact_hashes: Mapping[str, str] | None = None,
    schema_version: int = CHUNK_CLASSIFIER_SCHEMA_VERSION,
    source_universe: str | None = None,
    expected_safe_parent_count: int | None = None,
    expected_reject_parent_count: int | None = None,
    expected_repair_overlay_label_count: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic no-raw chunk classifier training manifest.

    `chunk_candidates` should be no-raw metadata for rejected-parent chunks.
    `effective_label_evidence` is normally produced by
    `build_effective_chunk_label_evidence`, after applying the fresh repair
    overlay. `weak_safe_candidates` are optional no-raw chunk candidates from
    safe-parent documents; they are clearly marked weak and never evaluation
    evidence in legacy schema v1. Schema v2 instead uses
    `inherited_safe_candidates` as first-class inherited-safe chunk evidence for
    the final-35k classifier phase.
    """

    if schema_version not in {CHUNK_CLASSIFIER_SCHEMA_VERSION, CHUNK_CLASSIFIER_SCHEMA_VERSION_V2}:
        raise ValueError("unsupported chunk classifier manifest schema version")
    weak_safe_candidates = list(weak_safe_candidates)
    inherited_safe_candidates = list(inherited_safe_candidates)
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION and inherited_safe_candidates:
        raise ValueError("schema v1 does not support inherited safe candidates")
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2 and weak_safe_candidates:
        raise ValueError("legacy weak parent safe candidates cannot enter v2 manifest")
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2:
        source_universe = source_universe or FINAL_35K_CHUNK_CLASSIFIER_V2_SOURCE_UNIVERSE
        if source_universe != FINAL_35K_CHUNK_CLASSIFIER_V2_SOURCE_UNIVERSE:
            raise ValueError("base v2 builder only supports the final-35k source_universe")

    strong_candidates = [_normalized_candidate(row, index=index) for index, row in enumerate(chunk_candidates)]
    weak_candidates = [
        _normalized_candidate(row, index=len(strong_candidates) + index)
        for index, row in enumerate(weak_safe_candidates)
    ]
    inherited_candidates = [
        _normalized_candidate(row, index=len(strong_candidates) + len(weak_candidates) + index)
        for index, row in enumerate(inherited_safe_candidates)
    ]
    _reject_duplicate_candidates(strong_candidates + weak_candidates + inherited_candidates)
    evidence_by_candidate = _validated_evidence_map(effective_label_evidence)
    strong_ids = {row["candidate_id"] for row in strong_candidates}
    unknown_evidence_ids = sorted(set(evidence_by_candidate) - strong_ids)
    if unknown_evidence_ids:
        raise ValueError("effective label evidence contains candidate_id outside chunk candidates")
    missing_evidence_ids = sorted(strong_ids - set(evidence_by_candidate))
    if missing_evidence_ids:
        raise ValueError("chunk candidates are missing effective label evidence")

    rows: list[dict[str, Any]] = []
    for candidate in strong_candidates:
        evidence = evidence_by_candidate[candidate["candidate_id"]]
        rows.append(_strong_manifest_row(candidate, evidence, schema_version=schema_version))
    for candidate in weak_candidates:
        rows.append(_weak_safe_manifest_row(candidate))
    for candidate in inherited_candidates:
        rows.append(_inherited_safe_manifest_row(candidate))

    split_by_candidate = _plan_leakage_safe_splits(rows, split_seed=split_seed)
    for row in rows:
        row["split"] = split_by_candidate[row["candidate_id"]]
        if row["evidence_tier"] == WEAK_SAFE_EVIDENCE_TIER and row["split"] != "train":
            row["fit_eligible"] = False
            row["eligible_for_eval"] = False
            row["training_weight"] = 0.0
            row["training_role"] = "audit_only_weak_safe_holdout_excluded"
            row["split"] = "audit"
        row["row_hash"] = stable_json_hash({key: value for key, value in row.items() if key != "row_hash"})

    source_artifact_hashes = dict(source_artifact_hashes or {})
    for key, value in source_artifact_hashes.items():
        _required_hash(value, f"source_artifact_hashes.{key}")

    label_counts = Counter(row["classifier_label"] for row in rows)
    evidence_counts = Counter(row["evidence_tier"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    parent_coverage_counts = {
        "reject": len(
            {
                str(row["parent_candidate_id"])
                for row in rows
                if row["evidence_tier"] in {STRONG_EVIDENCE_TIER, FRESH_REPAIR_EVIDENCE_TIER}
            }
        ),
        "safe": len(
            {
                str(row["parent_candidate_id"])
                for row in rows
                if row["evidence_tier"] == INHERITED_PARENT_SAFE_EVIDENCE_TIER
            }
        ),
    }
    if expected_safe_parent_count is not None and parent_coverage_counts["safe"] != int(expected_safe_parent_count):
        raise ValueError("safe parent coverage count mismatch")
    if expected_reject_parent_count is not None and parent_coverage_counts["reject"] != int(expected_reject_parent_count):
        raise ValueError("reject parent coverage count mismatch")
    if (
        expected_repair_overlay_label_count is not None
        and evidence_counts[FRESH_REPAIR_EVIDENCE_TIER] != int(expected_repair_overlay_label_count)
    ):
        raise ValueError("fresh repair overlay label count mismatch")
    split_policy = {
        "policy": "leakage_connected_components_v1",
        "split_seed_hash": stable_json_hash({"split_seed": split_seed}),
        "blocked_identity_fields": [
            "source_document_identity_hash",
            "parent_candidate_id",
            "physical_source_row_key_hash",
            "parent_text_hash_sha256",
            "text_hash_sha256",
            "normalized_text_hash_sha256",
        ],
        "strong_holdout_only": schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION,
    }
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2:
        split_policy["safe_parent_policy"] = "v2_inherited_parent_safe_train_calibration_eval"
    else:
        split_policy["weak_parent_safe_policy"] = "train_only_low_weight_else_audit"
    manifest = {
        "manifest_kind": CHUNK_CLASSIFIER_MANIFEST_KIND,
        "schema_version": schema_version,
        "candidate_count": len(rows),
        "strong_label_count": evidence_counts[STRONG_EVIDENCE_TIER] + evidence_counts[FRESH_REPAIR_EVIDENCE_TIER],
        "weak_label_count": evidence_counts[WEAK_SAFE_EVIDENCE_TIER],
        "repair_overlay_label_count": evidence_counts[FRESH_REPAIR_EVIDENCE_TIER],
        "fit_eligible_count": sum(1 for row in rows if row["fit_eligible"]),
        "eval_eligible_count": sum(1 for row in rows if row["eligible_for_eval"]),
        "label_counts": _sorted_counter(label_counts),
        "evidence_tier_counts": _sorted_counter(evidence_counts),
        "split_counts": _sorted_counter(split_counts),
        "split_policy": split_policy,
        "effective_label_evidence_hash": _optional_hash(
            effective_label_evidence.get("chunk_label_evidence_view_hash")
            or effective_label_evidence.get("effective_chunk_label_evidence_hash"),
            "effective_label_evidence_hash",
        ),
        "source_artifact_hashes": source_artifact_hashes,
        "rows": sorted(rows, key=lambda row: row["candidate_id"]),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2:
        manifest.update(
            {
                "source_universe": source_universe,
                "inherited_safe_label_count": evidence_counts[INHERITED_PARENT_SAFE_EVIDENCE_TIER],
                "parent_coverage_counts": parent_coverage_counts,
                "final_35k_expected_parent_counts": {
                    "safe": None if expected_safe_parent_count is None else int(expected_safe_parent_count),
                    "reject": None if expected_reject_parent_count is None else int(expected_reject_parent_count),
                },
            }
        )
    manifest["training_manifest_hash"] = stable_json_hash(manifest)
    validate_chunk_classifier_training_manifest(manifest)
    return manifest


def build_v2_chunk_classifier_training_manifest(
    *,
    chunk_candidates: Iterable[Mapping[str, Any]],
    effective_label_evidence: Mapping[str, Any],
    inherited_safe_candidates: Iterable[Mapping[str, Any]],
    split_seed: str = "twc-chunk-classifier-v2",
    source_artifact_hashes: Mapping[str, str] | None = None,
    expected_safe_parent_count: int = FINAL_35K_SAFE_PARENT_COUNT,
    expected_reject_parent_count: int = FINAL_35K_REJECT_PARENT_COUNT,
    expected_repair_overlay_label_count: int | None = FINAL_35K_REPAIR_OVERLAY_LABEL_COUNT,
) -> dict[str, Any]:
    """Build the final-35k v2 chunk classifier manifest.

    This is the safe-parent semantics boundary: inherited safe chunks are
    first-class v2 label evidence, not legacy weak-safe rows.
    """

    return build_chunk_classifier_training_manifest(
        chunk_candidates=chunk_candidates,
        effective_label_evidence=effective_label_evidence,
        inherited_safe_candidates=inherited_safe_candidates,
        split_seed=split_seed,
        source_artifact_hashes=source_artifact_hashes,
        schema_version=CHUNK_CLASSIFIER_SCHEMA_VERSION_V2,
        source_universe=FINAL_35K_CHUNK_CLASSIFIER_V2_SOURCE_UNIVERSE,
        expected_safe_parent_count=expected_safe_parent_count,
        expected_reject_parent_count=expected_reject_parent_count,
        expected_repair_overlay_label_count=expected_repair_overlay_label_count,
    )


def build_augmented_v2_chunk_classifier_training_manifest(
    *,
    base_manifest: Mapping[str, Any],
    active_learning_batches: Iterable[Mapping[str, Any]],
    split_seed: str = "twc-chunk-classifier-v2-active-learning-augmented",
    source_artifact_hashes: Mapping[str, str] | None = None,
    minimum_final35k_eval_rows: int = 2,
    minimum_final35k_eval_label_count: int = 0,
    minimum_final35k_calibration_label_count: int = 0,
    maximum_label_conflict_audit_fraction: float = 1.0,
    expected_final35k_candidate_count: int | None = None,
    expected_active_learning_label_count: int | None = 8_000,
    expected_active_learning_batch_count: int | None = 2,
    expected_active_learning_rows_per_batch: int | None = 4_000,
    core_text_conflict_policy: str = CORE_TEXT_CONFLICT_POLICY_HARD_FAIL,
    active_inherited_conflict_policy: str = ACTIVE_INHERITED_CONFLICT_POLICY_HARD_FAIL,
) -> dict[str, Any]:
    """Build the final-35k-plus-active-learning v2 classifier manifest.

    Active-learning rows require a no-raw core-hash proof supplied by the
    caller. The public schedules contain teacher-input hashes but not the
    core-text hash, and this builder deliberately refuses to conflate those
    two contracts.
    """

    active_learning_batches = list(active_learning_batches)
    core_text_conflict_policy = _required_core_text_conflict_policy(core_text_conflict_policy)
    active_inherited_conflict_policy = _required_active_inherited_conflict_policy(active_inherited_conflict_policy)
    validate_chunk_classifier_training_manifest(base_manifest)
    if int(base_manifest.get("schema_version", -1)) != CHUNK_CLASSIFIER_SCHEMA_VERSION_V2:
        raise ValueError("augmented v2 manifest requires a v2 base manifest")
    if _manifest_source_universe(base_manifest) != FINAL_35K_CHUNK_CLASSIFIER_V2_SOURCE_UNIVERSE:
        raise ValueError("augmented v2 manifest requires the final-35k v2 base source_universe")

    rows: list[dict[str, Any]] = []
    original_manifest_hash = _required_hash(base_manifest.get("training_manifest_hash"), "base training_manifest_hash")
    for row in base_manifest.get("rows") or []:
        copied = dict(row)
        copied["original_training_manifest_hash"] = original_manifest_hash
        copied["original_row_hash"] = _required_hash(row.get("row_hash"), "base row_hash")
        copied["source_universe_component"] = "final_35k_chunk_classifier_v2"
        copied.pop("row_hash", None)
        rows.append(copied)

    active_rows: list[dict[str, Any]] = []
    unselected_parse_error_count = 0
    batch_counts: Counter[str] = Counter()
    final_label_counts: Counter[str] = Counter()
    for batch in active_learning_batches:
        batch_rows, parse_error_count = _active_learning_rows_from_batch(batch)
        active_rows.extend(batch_rows)
        unselected_parse_error_count += parse_error_count
        for row in batch_rows:
            batch_counts[str(row["active_learning_batch_id"])] += 1
            final_label_counts[str(row["active_learning_final_label"])] += 1
    if expected_final35k_candidate_count is None:
        expected_final35k_candidate_count = int(base_manifest.get("candidate_count", 0))
    if expected_final35k_candidate_count is not None and len(rows) != int(expected_final35k_candidate_count):
        raise ValueError("augmented manifest final35k candidate count does not match expected source universe")
    if expected_active_learning_label_count is not None and len(active_rows) != int(expected_active_learning_label_count):
        raise ValueError("augmented manifest active-learning label count does not match expected source universe")
    if expected_active_learning_batch_count is not None and len(batch_counts) != int(expected_active_learning_batch_count):
        raise ValueError("augmented manifest active-learning batch count does not match expected source universe")
    if expected_active_learning_rows_per_batch is not None:
        expected_per_batch = int(expected_active_learning_rows_per_batch)
        if any(count != expected_per_batch for count in batch_counts.values()):
            raise ValueError("augmented manifest active-learning per-batch count does not match expected source universe")
    rows.extend(active_rows)

    _reject_duplicate_manifest_rows(rows)
    conflict_text_hashes = _conflicting_text_label_hashes(rows)
    if core_text_conflict_policy == CORE_TEXT_CONFLICT_POLICY_HARD_FAIL and conflict_text_hashes:
        raise ValueError("conflicting classifier labels for identical text hash")
    split_by_candidate = _plan_leakage_safe_splits(rows, split_seed=split_seed)
    for row in rows:
        row["split"] = split_by_candidate[str(row["candidate_id"])]
    _force_active_learning_components_out_of_eval(rows)
    conflict_stats = _apply_core_text_conflict_policy(
        rows,
        conflict_text_hashes=conflict_text_hashes,
        policy=core_text_conflict_policy,
    )
    active_inherited_conflict_stats = _apply_active_inherited_conflict_policy(
        rows,
        policy=active_inherited_conflict_policy,
    )
    _reject_active_reject_inherited_safe_conflicts(rows)
    _require_final35k_eval_viability(
        rows,
        minimum_final35k_eval_rows=minimum_final35k_eval_rows,
        minimum_final35k_eval_label_count=minimum_final35k_eval_label_count,
        minimum_final35k_calibration_label_count=minimum_final35k_calibration_label_count,
    )
    label_conflict_audit_stats = _label_conflict_audit_stats(
        rows,
        maximum_label_conflict_audit_fraction=maximum_label_conflict_audit_fraction,
    )

    for row in rows:
        row["row_hash"] = stable_json_hash({key: value for key, value in row.items() if key != "row_hash"})

    source_artifact_hashes = dict(source_artifact_hashes or {})
    source_artifact_hashes.setdefault("base_final35k_training_manifest_hash", original_manifest_hash)
    for index, batch in enumerate(active_learning_batches):
        schedule = batch.get("schedule") if isinstance(batch, Mapping) else None
        if isinstance(schedule, Mapping):
            schedule_hash = schedule.get("schedule_hash")
            if schedule_hash:
                source_artifact_hashes.setdefault(f"active_learning_batch_{index}_schedule_hash", str(schedule_hash))
    for key, value in source_artifact_hashes.items():
        _required_hash(value, f"source_artifact_hashes.{key}")

    label_counts = Counter(row["classifier_label"] for row in rows)
    evidence_counts = Counter(row["evidence_tier"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    final35k_parent_coverage_counts = {
        "reject": len(
            {
                str(row["parent_candidate_id"])
                for row in rows
                if row.get("source_universe_component") == "final_35k_chunk_classifier_v2"
                and row["evidence_tier"] in {STRONG_EVIDENCE_TIER, FRESH_REPAIR_EVIDENCE_TIER}
            }
        ),
        "safe": len(
            {
                str(row["parent_candidate_id"])
                for row in rows
                if row.get("source_universe_component") == "final_35k_chunk_classifier_v2"
                and row["evidence_tier"] == INHERITED_PARENT_SAFE_EVIDENCE_TIER
            }
        ),
    }
    split_policy = {
        "policy": "leakage_connected_components_v2_active_learning_no_eval",
        "split_seed_hash": stable_json_hash({"split_seed": split_seed}),
        "blocked_identity_fields": [
            "source_document_identity_hash",
            "parent_candidate_id",
            "physical_source_row_key_hash",
            "parent_text_hash_sha256",
            "text_hash_sha256",
            "normalized_text_hash_sha256",
        ],
        "active_learning_eval_policy": "active_or_mixed_components_train_or_calibration_only",
    }
    manifest = {
        "manifest_kind": CHUNK_CLASSIFIER_MANIFEST_KIND,
        "schema_version": CHUNK_CLASSIFIER_SCHEMA_VERSION_V2,
        "source_universe": AUGMENTED_V2_SOURCE_UNIVERSE,
        "candidate_count": len(rows),
        "strong_label_count": evidence_counts[STRONG_EVIDENCE_TIER] + evidence_counts[FRESH_REPAIR_EVIDENCE_TIER],
        "weak_label_count": 0,
        "repair_overlay_label_count": evidence_counts[FRESH_REPAIR_EVIDENCE_TIER],
        "inherited_safe_label_count": evidence_counts[INHERITED_PARENT_SAFE_EVIDENCE_TIER],
        "active_learning_label_count": evidence_counts[ACTIVE_LEARNING_EVIDENCE_TIER],
        "fit_eligible_count": sum(1 for row in rows if row["fit_eligible"]),
        "eval_eligible_count": sum(1 for row in rows if row["eligible_for_eval"]),
        "label_counts": _sorted_counter(label_counts),
        "evidence_tier_counts": _sorted_counter(evidence_counts),
        "split_counts": _sorted_counter(split_counts),
        "split_policy": split_policy,
        **conflict_stats,
        **active_inherited_conflict_stats,
        **label_conflict_audit_stats,
        "minimum_final35k_eval_rows": int(minimum_final35k_eval_rows),
        "minimum_final35k_eval_label_count": int(minimum_final35k_eval_label_count),
        "minimum_final35k_calibration_label_count": int(minimum_final35k_calibration_label_count),
        "maximum_label_conflict_audit_fraction": float(maximum_label_conflict_audit_fraction),
        "effective_label_evidence_hash": base_manifest.get("effective_label_evidence_hash"),
        "source_artifact_hashes": source_artifact_hashes,
        "final35k_parent_coverage_counts": final35k_parent_coverage_counts,
        "final_35k_expected_parent_counts": dict(base_manifest.get("final_35k_expected_parent_counts") or {}),
        "expected_final35k_candidate_count": int(expected_final35k_candidate_count or 0),
        "expected_active_learning_label_count": None if expected_active_learning_label_count is None else int(expected_active_learning_label_count),
        "expected_active_learning_batch_count": None if expected_active_learning_batch_count is None else int(expected_active_learning_batch_count),
        "expected_active_learning_rows_per_batch": None
        if expected_active_learning_rows_per_batch is None
        else int(expected_active_learning_rows_per_batch),
        "active_learning_final_label_counts": _sorted_counter(final_label_counts),
        "active_learning_batch_counts": _sorted_counter(batch_counts),
        "active_learning_unselected_parse_error_vote_count": int(unselected_parse_error_count),
        "rows": sorted(rows, key=lambda row: row["candidate_id"]),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    manifest["training_manifest_hash"] = stable_json_hash(manifest)
    validate_chunk_classifier_training_manifest(manifest)
    return manifest


def _active_learning_rows_from_batch(batch: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    batch_id = str(batch.get("batch_id") or "")
    if not batch_id:
        raise ValueError("active-learning batch requires batch_id")
    schedule = batch.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("active-learning batch requires schedule")
    _validate_active_learning_candidate_schedule_for_manifest(schedule)
    candidates = list(schedule.get("candidates") or [])
    if int(schedule.get("candidate_count", len(candidates))) != len(candidates):
        raise ValueError("active-learning schedule candidate_count mismatch")
    schedule_by_candidate: dict[str, Mapping[str, Any]] = {}
    for row in candidates:
        if not isinstance(row, Mapping):
            raise ValueError("active-learning schedule candidate row must be an object")
        validate_public_no_raw_payload(row)
        _reject_forbidden_input_keys(row)
        candidate_id = _required_hash(row.get("candidate_id_hash"), "candidate_id_hash")
        if candidate_id in schedule_by_candidate:
            raise ValueError("active-learning schedule contains duplicate candidate")
        _required_hash(row.get("candidate_schedule_row_hash"), "candidate_schedule_row_hash")
        schedule_by_candidate[candidate_id] = row

    labels = list(batch.get("labels") or [])
    votes = list(batch.get("votes") or [])
    vote_by_hash: dict[str, Mapping[str, Any]] = {}
    unselected_parse_errors = 0
    for vote in votes:
        if not isinstance(vote, Mapping):
            raise ValueError("active-learning vote row must be an object")
        _validate_active_learning_vote_public_payload(vote)
        if vote.get("parse_status") != "parsed":
            unselected_parse_errors += 1
            continue
        vote_hash = _required_hash(vote.get("label_row_hash"), "label_row_hash")
        if vote_hash in vote_by_hash:
            raise ValueError("active-learning votes contain duplicate selected-vote hash")
        vote_by_hash[vote_hash] = vote

    core_hashes = {str(key): str(value) for key, value in (batch.get("core_text_hashes_by_candidate_id") or {}).items()}
    parent_hashes = {str(key): str(value) for key, value in (batch.get("parent_text_hashes_by_candidate_id") or {}).items()}
    rows = []
    seen_labels: set[str] = set()
    for label in labels:
        if not isinstance(label, Mapping):
            raise ValueError("active-learning label row must be an object")
        validate_public_no_raw_payload(label)
        _reject_forbidden_input_keys(label)
        if label.get("labels_are_active_learning_evidence") is not True:
            raise ValueError("active-learning label row must be marked active-learning evidence")
        if label.get("labels_are_corpus_approval") is not False:
            raise ValueError("active-learning label row must not approve corpus")
        candidate_id = _required_hash(label.get("candidate_id"), "candidate_id")
        if candidate_id in seen_labels:
            raise ValueError("active-learning labels contain duplicate candidate")
        seen_labels.add(candidate_id)
        schedule_row = schedule_by_candidate.get(candidate_id)
        if schedule_row is None:
            raise ValueError("active-learning label missing schedule candidate")
        if label.get("candidate_schedule_row_hash") != schedule_row.get("candidate_schedule_row_hash"):
            raise ValueError("active-learning label/schedule row hash mismatch")
        selected_votes = _selected_active_learning_votes(
            label,
            vote_by_hash=vote_by_hash,
            schedule_hash=str(schedule.get("schedule_hash") or ""),
        )
        _validate_active_learning_final_label_hash(label)
        rows.append(
            _active_learning_manifest_row(
                schedule_row=schedule_row,
                label_row=label,
                selected_votes=selected_votes,
                batch_id=batch_id,
                core_text_hash=_required_hash(core_hashes.get(candidate_id), "active-learning core text hash"),
                parent_text_hash=_optional_hash(parent_hashes.get(candidate_id), "active-learning parent text hash"),
            )
        )
    if set(schedule_by_candidate) - seen_labels:
        raise ValueError("active-learning schedule contains unlabeled candidate")
    return rows, unselected_parse_errors


def _validate_active_learning_candidate_schedule_for_manifest(schedule: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(schedule)
    _reject_forbidden_input_keys(schedule)
    if schedule.get("artifact_kind") != "twc_deepseek_active_learning_candidate_schedule_v1":
        raise ValueError("unsupported active-learning candidate schedule kind")
    if schedule.get("active_learning_probe_id") != ACTIVE_LEARNING_PROBE_ID:
        raise ValueError("active-learning candidate schedule probe id mismatch")
    if schedule.get("deepseek_launched") is not False or schedule.get("paid_api_called") is not False:
        raise ValueError("active-learning candidate schedule must not perform paid calls")
    _require_false_flags(
        schedule,
        (
            "raw_text_serialized",
            "raw_url_serialized",
            "token_serialized",
            "accepted_manifest",
            "training_shards_created",
        ),
        "active-learning candidate schedule",
    )
    candidates = list(schedule.get("candidates") or [])
    if int(schedule.get("candidate_count", -1)) != len(candidates):
        raise ValueError("active-learning candidate schedule candidate_count mismatch")
    if int(schedule.get("target_total", -1)) != len(candidates):
        raise ValueError("active-learning candidate schedule target_total mismatch")
    bucket_targets = {str(key): int(value) for key, value in (schedule.get("bucket_targets") or {}).items()}
    bucket_counts_expected = {str(key): int(value) for key, value in (schedule.get("bucket_counts") or {}).items()}
    if set(bucket_targets) != set(ACTIVE_LEARNING_CANONICAL_BUCKETS):
        raise ValueError("candidate schedule must use canonical active-learning buckets")
    if sum(bucket_targets.values()) != len(candidates):
        raise ValueError("candidate schedule bucket targets do not sum to candidate_count")
    if set(bucket_counts_expected) != set(ACTIVE_LEARNING_CANONICAL_BUCKETS):
        raise ValueError("candidate schedule bucket_counts must use canonical active-learning buckets")
    seen: set[str] = set()
    bucket_counts: Counter[str] = Counter()
    length_threshold = int(schedule.get("length_slice_threshold", 0) or 0)
    for expected_index, row in enumerate(candidates):
        if not isinstance(row, Mapping):
            raise ValueError("candidate schedule rows must be mappings")
        validate_public_no_raw_payload(row)
        _reject_forbidden_input_keys(row)
        if int(row.get("schedule_index", -1)) != expected_index:
            raise ValueError("candidate schedule index mismatch")
        for field in (
            "candidate_id_hash",
            "row_payload_hash",
            "source_document_id_hash",
            "source_file_id_hash",
            "physical_row_hash",
            "source_metadata_hash",
            "source_schedule_hash",
            "schedule_entry_hash",
            "refetch_identity_hash",
            "candidate_policy_hash",
            "config_hash",
            "classifier_freeze_hash",
            "sparse_report_hash",
            "sparse_model_file_sha256",
            "window_mil_report_hash",
            "window_mil_model_file_sha256",
            "candidate_schedule_row_hash",
        ):
            _required_hash(row.get(field), field)
        candidate_id = str(row["candidate_id_hash"])
        if candidate_id in seen:
            raise ValueError("active-learning candidate schedule contains duplicate candidate")
        seen.add(candidate_id)
        if row.get("route") not in {"policy_safe", "policy_reject", "quarantine"}:
            raise ValueError("unsupported candidate schedule route")
        if row.get("sparse_score_finite") is not True or row.get("window_mil_score_finite") is not True:
            raise ValueError("candidate schedule row requires finite sparse and Window-MIL scores")
        primary_bucket = str(row.get("primary_bucket") or "")
        if primary_bucket not in ACTIVE_LEARNING_CANONICAL_BUCKETS:
            raise ValueError("candidate schedule row has non-canonical primary_bucket")
        if str(row.get("intended_bucket") or "") not in ACTIVE_LEARNING_CANONICAL_BUCKETS:
            raise ValueError("candidate schedule row has non-canonical intended_bucket")
        if not isinstance(row.get("secondary_bucket_flags"), Mapping):
            raise ValueError("candidate schedule row missing secondary_bucket_flags")
        if bool(row.get("fallback_used")) and not row.get("fallback_reason_code"):
            raise ValueError("candidate fallback row missing fallback_reason_code")
        if bool(row.get("length_slice_flag")) != (int(row.get("core_token_lower_bound") or 0) >= length_threshold):
            raise ValueError("candidate schedule length_slice_flag mismatch")
        bucket_counts[primary_bucket] += 1
        if row.get("candidate_schedule_row_hash") != stable_json_hash(
            {key: value for key, value in row.items() if key != "candidate_schedule_row_hash"}
        ):
            raise ValueError("candidate schedule row hash mismatch")
    bucket_counts_actual = {bucket: int(bucket_counts[bucket]) for bucket in ACTIVE_LEARNING_CANONICAL_BUCKETS}
    if bucket_counts_actual != bucket_counts_expected:
        raise ValueError("candidate schedule bucket_counts mismatch")
    if bucket_counts_actual != {bucket: int(bucket_targets[bucket]) for bucket in ACTIVE_LEARNING_CANONICAL_BUCKETS}:
        raise ValueError("candidate schedule primary buckets do not match canonical targets")
    if schedule.get("schedule_hash") != stable_json_hash({key: value for key, value in schedule.items() if key != "schedule_hash"}):
        raise ValueError("active-learning candidate schedule hash mismatch")


def _selected_active_learning_votes(
    label: Mapping[str, Any],
    *,
    vote_by_hash: Mapping[str, Mapping[str, Any]],
    schedule_hash: str,
) -> list[Mapping[str, Any]]:
    hashes = [str(value) for value in (label.get("vote_row_hashes") or [])]
    if len(hashes) != 3 or len(set(hashes)) != 3:
        raise ValueError("active-learning label must reference exactly three selected parsed votes")
    selected = []
    for vote_hash in hashes:
        vote = vote_by_hash.get(_required_hash(vote_hash, "selected vote row hash"))
        if vote is None or vote.get("parse_status") != "parsed":
            raise ValueError("active-learning label must reference exactly three selected parsed votes")
        if _required_hash(vote.get("label_row_hash"), "label_row_hash") != stable_json_hash(
            {key: value for key, value in vote.items() if key != "label_row_hash"}
        ):
            raise ValueError("active-learning selected vote row hash mismatch")
        selected.append(vote)
    candidate_ids = {str(vote.get("candidate_id") or "") for vote in selected}
    schedule_hashes = {str(vote.get("candidate_schedule_row_hash") or "") for vote in selected}
    text_hashes = {str(vote.get("text_hash") or "") for vote in selected}
    source_hashes = {str(vote.get("source_candidate_manifest_hash") or "") for vote in selected}
    source_doc_ids = {str(vote.get("source_doc_id") or "") for vote in selected}
    primary_buckets = {str(vote.get("primary_bucket") or "") for vote in selected}
    routes = {str(vote.get("route") or "") for vote in selected}
    bof_indices = {int(vote.get("bof_call_index", vote.get("vote_index", -1))) for vote in selected}
    vote_indices = {int(vote.get("vote_index", vote.get("bof_call_index", -1))) for vote in selected}
    bof_group_keys = {str(vote.get("bof_group_key") or "") for vote in selected}
    probe_group_keys = {str(vote.get("probe_vote_group_key") or vote.get("online_vote_group_key") or "") for vote in selected}
    idempotency_keys = [str(vote.get("idempotency_key") or "") for vote in selected]
    probe_ids = {str(vote.get("active_learning_probe_id") or "") for vote in selected}
    entrypoint_policies = {str(vote.get("entrypoint_policy_version") or "") for vote in selected if vote.get("entrypoint_policy_version")}
    if candidate_ids != {str(label.get("candidate_id") or "")}:
        raise ValueError("active-learning selected votes candidate mismatch")
    if schedule_hashes != {str(label.get("candidate_schedule_row_hash") or "")}:
        raise ValueError("active-learning selected votes schedule-row mismatch")
    if len(text_hashes) != 1 or len(source_hashes) != 1:
        raise ValueError("active-learning selected votes must agree on teacher-input and source hashes")
    if len(source_doc_ids) != 1 or "" in source_doc_ids:
        raise ValueError("active-learning selected votes must agree on source document")
    if primary_buckets != {str(label.get("primary_bucket") or "")}:
        raise ValueError("active-learning selected votes primary bucket mismatch")
    if routes != {str(label.get("route") or "")}:
        raise ValueError("active-learning selected votes route mismatch")
    if bof_indices != {0, 1, 2} or vote_indices != {0, 1, 2}:
        raise ValueError("active-learning selected votes must cover BoF3 slots 0,1,2")
    if len(bof_group_keys) != 1 or "" in bof_group_keys:
        raise ValueError("active-learning selected votes must share a BoF group key")
    if probe_group_keys != {""} and probe_group_keys != bof_group_keys:
        raise ValueError("active-learning selected votes probe group mismatch")
    if len(idempotency_keys) != 3 or len(set(idempotency_keys)) != 3 or any(not key for key in idempotency_keys):
        raise ValueError("active-learning selected votes require three idempotency keys")
    if [str(value) for value in (label.get("vote_idempotency_keys") or [])] != idempotency_keys:
        raise ValueError("active-learning selected votes idempotency mismatch")
    if probe_ids != {str(label.get("active_learning_probe_id") or "")}:
        raise ValueError("active-learning selected votes probe id mismatch")
    if entrypoint_policies and label.get("entrypoint_policy_version") and entrypoint_policies != {str(label.get("entrypoint_policy_version"))}:
        raise ValueError("active-learning selected votes policy mismatch")
    schedule_hash = _required_hash(schedule_hash, "active-learning schedule_hash")
    if label.get("schedule_hash") not in (None, "") and str(label.get("schedule_hash")) != schedule_hash:
        raise ValueError("active-learning final label schedule hash mismatch")
    for vote in selected:
        if vote.get("schedule_hash") not in (None, "") and str(vote.get("schedule_hash")) != schedule_hash:
            raise ValueError("active-learning selected vote schedule hash mismatch")
    for vote in selected:
        _required_hash(vote.get("text_hash"), "active-learning teacher input text hash")
        _required_hash(vote.get("source_candidate_manifest_hash"), "active-learning source candidate manifest hash")
        vote_label = str(vote.get("label") or "")
        if vote_label not in {"safe", "reject"}:
            raise ValueError("active-learning selected vote label is unsupported")
    return selected


def _validate_active_learning_final_label_hash(label: Mapping[str, Any]) -> None:
    expected = stable_json_hash({key: value for key, value in label.items() if key != "final_label_row_hash"})
    if _required_hash(label.get("final_label_row_hash"), "active-learning final label row hash") != expected:
        raise ValueError("active-learning final label row hash mismatch")


def _validate_active_learning_vote_public_payload(vote: Mapping[str, Any]) -> None:
    _require_false_flags(
        vote,
        ("raw_text_serialized", "raw_url_serialized", "token_serialized", "labels_are_corpus_approval"),
        "active-learning vote row",
    )
    _reject_forbidden_active_vote_keys(vote)
    model_output = vote.get("model_output")
    if model_output is not None:
        if not isinstance(model_output, Mapping):
            raise ValueError("active-learning vote model_output must be a compact parsed object")
        if set(model_output) - {"label", "borderline"}:
            raise ValueError("active-learning vote model_output contains unsupported fields")
        if model_output.get("label") not in {"safe", "reject"} or not isinstance(model_output.get("borderline"), bool):
            raise ValueError("active-learning vote model_output is not parsed-label-only")


def _reject_forbidden_active_vote_keys(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            key_lower = key_text.casefold()
            if key_lower == "model_output":
                continue
            if key_lower in ACTIVE_VOTE_FORBIDDEN_KEYS:
                raise ValueError(f"active-learning vote contains raw/private key {key_text} at {path}.{key_text}")
            if "url" in key_lower and key_lower != "raw_url_serialized":
                raise ValueError(f"active-learning vote contains url-like key {key_text} at {path}.{key_text}")
            _reject_forbidden_active_vote_keys(child, path=f"{path}.{key_text}")
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _reject_forbidden_active_vote_keys(child, path=f"{path}[{index}]")


def _active_learning_vote_counts(
    label_row: Mapping[str, Any],
    selected_votes: Sequence[Mapping[str, Any]],
    *,
    final_label: str,
) -> tuple[int, int, int]:
    if _positive_int(label_row.get("required_vote_count"), "active-learning required_vote_count") != 3:
        raise ValueError("active-learning label row must require exactly three votes")
    if _nonnegative_int(label_row.get("parsed_vote_count"), "active-learning parsed_vote_count") != 3:
        raise ValueError("active-learning label row must contain exactly three parsed votes")
    label_counts = Counter(str(vote.get("label") or "") for vote in selected_votes)
    safe_vote_count = int(label_counts.get("safe", 0))
    reject_vote_count = int(label_counts.get("reject", 0))
    borderline_vote_count = sum(1 for vote in selected_votes if bool(vote.get("borderline")))
    reported_counts = {
        str(key): int(value)
        for key, value in (label_row.get("teacher_vote_label_counts") or {}).items()
        if int(value) != 0
    }
    if reported_counts != _sorted_counter(label_counts):
        raise ValueError("active-learning vote label-count summary mismatch")
    if _nonnegative_int(label_row.get("borderline_vote_count", 0), "borderline_vote_count") != borderline_vote_count:
        raise ValueError("active-learning borderline vote summary mismatch")
    strict_safe = safe_vote_count == 3 and borderline_vote_count == 0
    if label_row.get("final_strict_safe") is not None and bool(label_row.get("final_strict_safe")) != strict_safe:
        raise ValueError("active-learning strict-safe flag mismatch")
    if final_label == "safe" and not strict_safe:
        raise ValueError("active-learning strict-safe aggregation mismatch")
    if final_label == "reject_or_borderline" and strict_safe:
        raise ValueError("active-learning reject-or-borderline aggregation mismatch")
    return safe_vote_count, reject_vote_count, borderline_vote_count


def _active_learning_manifest_row(
    *,
    schedule_row: Mapping[str, Any],
    label_row: Mapping[str, Any],
    selected_votes: Sequence[Mapping[str, Any]],
    batch_id: str,
    core_text_hash: str,
    parent_text_hash: str | None,
) -> dict[str, Any]:
    candidate_id = _required_hash(label_row.get("candidate_id"), "candidate_id")
    final_label = str(label_row.get("final_probe_label") or label_row.get("final_label") or "")
    if final_label not in {"safe", "reject_or_borderline"}:
        raise ValueError("active-learning final label must be safe or reject_or_borderline")
    classifier_label = "safe" if final_label == "safe" else "reject"
    teacher_input_text_hash = _required_hash(selected_votes[0].get("text_hash"), "teacher_input_text_hash_sha256")
    source_candidate_manifest_hash = _required_hash(
        selected_votes[0].get("source_candidate_manifest_hash"),
        "source_candidate_manifest_hash",
    )
    if str(selected_votes[0].get("source_doc_id") or "") != str(schedule_row.get("source_document_id_hash") or ""):
        raise ValueError("active-learning selected vote source document mismatch")
    if str(label_row.get("primary_bucket") or "") != str(schedule_row.get("primary_bucket") or ""):
        raise ValueError("active-learning label/schedule primary bucket mismatch")
    if str(label_row.get("route") or "") != str(schedule_row.get("route") or ""):
        raise ValueError("active-learning label/schedule route mismatch")
    source_doc_hash = _required_hash(schedule_row.get("source_document_id_hash"), "source_document_id_hash")
    physical_hash = _required_hash(schedule_row.get("physical_row_hash"), "physical_row_hash")
    schedule_index = _nonnegative_int(schedule_row.get("schedule_index", 0), "active-learning schedule_index")
    safe_vote_count, reject_vote_count, borderline_vote_count = _active_learning_vote_counts(
        label_row,
        selected_votes,
        final_label=final_label,
    )
    candidate_hash = stable_json_hash(
        {
            "candidate_id": candidate_id,
            "candidate_policy_hash": _required_hash(schedule_row.get("candidate_policy_hash"), "candidate_policy_hash"),
            "candidate_schedule_row_hash": _required_hash(
                schedule_row.get("candidate_schedule_row_hash"),
                "candidate_schedule_row_hash",
            ),
            "source_candidate_manifest_hash": source_candidate_manifest_hash,
            "core_text_hash_sha256": core_text_hash,
            "teacher_input_text_hash_sha256": teacher_input_text_hash,
        }
    )
    row = {
        "row_schema_version": CHUNK_CLASSIFIER_SCHEMA_VERSION_V2,
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "parent_candidate_id": source_doc_hash,
        "source_chunk_manifest_hash": source_candidate_manifest_hash,
        "source_document_identity_hash": source_doc_hash,
        "source_file_ref_hash": _required_hash(schedule_row.get("source_file_id_hash"), "source_file_id_hash"),
        "dataset_identity_hash": stable_json_hash(
            {
                "source_metadata_hash": _required_hash(schedule_row.get("source_metadata_hash"), "source_metadata_hash"),
                "source_schedule_hash": _required_hash(schedule_row.get("source_schedule_hash"), "source_schedule_hash"),
            }
        ),
        "physical_source_row_key_hash": physical_hash,
        "parent_text_hash_sha256": parent_text_hash,
        "text_hash_sha256": core_text_hash,
        "normalized_text_hash_sha256": core_text_hash,
        "schedule_entry_hash": _required_hash(schedule_row.get("schedule_entry_hash"), "schedule_entry_hash"),
        "source_schedule_hash": _required_hash(schedule_row.get("source_schedule_hash"), "source_schedule_hash"),
        "source_metadata_hash": _required_hash(schedule_row.get("source_metadata_hash"), "source_metadata_hash"),
        "row_group_index": schedule_index,
        "row_group_local_row_index": schedule_index,
        "global_chunk_index": schedule_index,
        "source_locator_status": "private_refetch_required",
        "public_locator_fields": "active_learning_schedule_index_placeholder",
        "active_learning_schedule_index": schedule_index,
        "label_unit": "window_bundle",
        "leakage_group_hash": stable_json_hash({"candidate_id": candidate_id}),
        "source_bucket_hash": stable_json_hash(
            {
                "batch_id": batch_id,
                "primary_bucket": str(schedule_row.get("primary_bucket") or ""),
                "route": str(schedule_row.get("route") or ""),
                "source_schedule_hash": str(schedule_row.get("source_schedule_hash") or ""),
            }
        ),
        "token_count_bucket": str(schedule_row.get("token_count_bucket") or "unknown"),
        "classifier_label": classifier_label,
        "evidence_tier": ACTIVE_LEARNING_EVIDENCE_TIER,
        "label_source": "active_learning_bof3_chunk_label",
        "parent_label_source": "production_active_learning_chunk_probe",
        "canonical_final_label": classifier_label,
        "aggregation_route_hash": stable_json_hash(
            {
                "active_learning_final_label": final_label,
                "teacher_vote_label_counts": dict(label_row.get("teacher_vote_label_counts") or {}),
                "borderline_vote_count": int(label_row.get("borderline_vote_count") or 0),
            }
        ),
        "active_learning_final_label": final_label,
        "active_learning_batch_id": batch_id,
        "candidate_schedule_row_hash": _required_hash(label_row.get("candidate_schedule_row_hash"), "candidate_schedule_row_hash"),
        "teacher_input_text_hash_sha256": teacher_input_text_hash,
        "selected_vote_row_hashes": [_required_hash(vote.get("label_row_hash"), "label_row_hash") for vote in selected_votes],
        "primary_bucket": str(schedule_row.get("primary_bucket") or ""),
        "intended_bucket": str(schedule_row.get("intended_bucket") or ""),
        "route": str(schedule_row.get("route") or ""),
        "fallback_used": bool(schedule_row.get("fallback_used")),
        "fallback_reason_code": schedule_row.get("fallback_reason_code"),
        "core_token_lower_bound": _nonnegative_int(schedule_row.get("core_token_lower_bound", 0), "core_token_lower_bound"),
        "length_slice_flag": bool(schedule_row.get("length_slice_flag")),
        "sparse_score": float(schedule_row.get("sparse_score", 0.0)),
        "window_mil_score": float(schedule_row.get("window_mil_score", 0.0)),
        "sparse_threshold": float(schedule_row.get("sparse_threshold", 0.0)),
        "window_mil_threshold": float(schedule_row.get("window_mil_threshold", 0.0)),
        "sparse_decision_reject": bool(schedule_row.get("sparse_decision_reject")),
        "window_mil_decision_reject": bool(schedule_row.get("window_mil_decision_reject")),
        "sparse_normalized_margin": float(schedule_row.get("sparse_normalized_margin", 0.0)),
        "window_mil_normalized_margin": float(schedule_row.get("window_mil_normalized_margin", 0.0)),
        "parsed_vote_count": 3,
        "required_vote_count": 3,
        "parse_error_vote_count": 0,
        "transport_failure_count": 0,
        "borderline_vote_count": borderline_vote_count,
        "teacher_safe_vote_count": safe_vote_count,
        "teacher_reject_vote_count": reject_vote_count,
        "fit_eligible": True,
        "eligible_for_eval": False,
        "training_weight": 1.0,
        "training_role": "active_learning_safe_train_or_calibration"
        if final_label == "safe"
        else "active_learning_reject_or_borderline_train_or_calibration",
        "private_cache_route": ACTIVE_LEARNING_PRIVATE_CACHE_ROUTE,
        "source_universe_component": "active_learning_bof3",
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    if row["primary_bucket"] == "":
        raise ValueError("active-learning row requires primary_bucket")
    if row["route"] not in {"policy_safe", "policy_reject", "quarantine"}:
        raise ValueError("active-learning row route is unsupported")
    return row


def _reject_duplicate_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen:
            raise ValueError("duplicate candidate_id in augmented chunk classifier manifest")
        seen.add(candidate_id)


def _reject_conflicting_text_labels(rows: Sequence[Mapping[str, Any]]) -> None:
    if _conflicting_text_label_hashes(rows):
        raise ValueError("conflicting classifier labels for identical text hash")


def _required_core_text_conflict_policy(policy: str) -> str:
    policy = str(policy or "")
    if policy not in CORE_TEXT_CONFLICT_POLICIES:
        raise ValueError("unsupported core text conflict policy")
    return policy


def _required_active_inherited_conflict_policy(policy: str) -> str:
    policy = str(policy or "")
    if policy not in ACTIVE_INHERITED_CONFLICT_POLICIES:
        raise ValueError("unsupported active inherited conflict policy")
    return policy


def _conflicting_text_label_hashes(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    labels_by_text: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        text_hash = str(row.get("text_hash_sha256") or "")
        if not text_hash:
            continue
        labels_by_text[text_hash].add(str(row.get("classifier_label") or ""))
    return {text_hash for text_hash, labels in labels_by_text.items() if len(labels) > 1}


def _apply_core_text_conflict_policy(
    rows: list[dict[str, Any]],
    *,
    conflict_text_hashes: set[str],
    policy: str,
) -> dict[str, Any]:
    policy = _required_core_text_conflict_policy(policy)
    if policy == CORE_TEXT_CONFLICT_POLICY_HARD_FAIL:
        if conflict_text_hashes:
            raise ValueError("conflicting classifier labels for identical text hash")
        return _core_text_conflict_stats(
            rows,
            conflict_text_hashes=conflict_text_hashes,
            audited_groups=set(),
            policy=policy,
        )

    group_conflict_hashes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        text_hash = str(row.get("text_hash_sha256") or "")
        if text_hash in conflict_text_hashes:
            group_conflict_hashes[str(row["leakage_group_hash"])].add(text_hash)
    audited_groups = set(group_conflict_hashes)
    for row in rows:
        leakage_group = str(row["leakage_group_hash"])
        if leakage_group not in audited_groups:
            continue
        row["pre_audit_split"] = row.get("pre_audit_split") or row.get("split")
        row["split"] = "audit"
        row["fit_eligible"] = False
        row["eligible_for_eval"] = False
        row["training_weight"] = 0.0
        row["training_role"] = "audit_only_core_text_label_conflict_component"
        row["audit_policy_version"] = "augmented_v2_label_conflict_audit_v1"
        row["audit_exclusion_reason"] = CORE_TEXT_CONFLICT_AUDIT_REASON
        row["core_text_label_conflict_direct"] = str(row.get("text_hash_sha256") or "") in conflict_text_hashes
        row["core_text_label_conflict_hash_count"] = len(group_conflict_hashes[leakage_group])
        row["core_text_label_conflict_group_hash"] = stable_json_hash(
            {
                "conflicting_text_hashes": sorted(group_conflict_hashes[leakage_group]),
                "leakage_group_hash": leakage_group,
            }
        )
    return _core_text_conflict_stats(
        rows,
        conflict_text_hashes=conflict_text_hashes,
        audited_groups=audited_groups,
        policy=policy,
    )


def _core_text_conflict_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    conflict_text_hashes: set[str],
    audited_groups: set[str],
    policy: str,
) -> dict[str, Any]:
    direct_rows = [row for row in rows if str(row.get("text_hash_sha256") or "") in conflict_text_hashes]
    audited_rows = [row for row in rows if str(row.get("leakage_group_hash") or "") in audited_groups]
    return {
        "core_text_conflict_policy": policy,
        "core_text_conflict_hash_count": len(conflict_text_hashes),
        "core_text_conflict_direct_row_count": len(direct_rows),
        "core_text_conflict_audited_component_count": len(audited_groups),
        "core_text_conflict_audited_row_count": len(audited_rows),
        "core_text_conflict_audited_label_counts": _sorted_counter(Counter(str(row["classifier_label"]) for row in audited_rows)),
        "core_text_conflict_audited_evidence_tier_counts": _sorted_counter(Counter(str(row["evidence_tier"]) for row in audited_rows)),
        "core_text_conflict_audited_source_component_counts": _sorted_counter(
            Counter(str(row.get("source_universe_component") or "") for row in audited_rows)
        ),
    }


def _is_core_text_conflict_audit_row(row: Mapping[str, Any]) -> bool:
    return row.get("audit_exclusion_reason") == CORE_TEXT_CONFLICT_AUDIT_REASON


def _is_active_inherited_conflict_audit_row(row: Mapping[str, Any]) -> bool:
    return row.get("audit_exclusion_reason") == ACTIVE_INHERITED_CONFLICT_AUDIT_REASON


def _is_label_conflict_audit_row(row: Mapping[str, Any]) -> bool:
    return _is_core_text_conflict_audit_row(row) or _is_active_inherited_conflict_audit_row(row)


def _active_inherited_conflict_groups(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["leakage_group_hash"])].append(row)
    conflict_groups: set[str] = set()
    for group_hash, members in by_group.items():
        if members and all(_is_core_text_conflict_audit_row(row) for row in members):
            continue
        has_active_reject = any(
            row.get("evidence_tier") == ACTIVE_LEARNING_EVIDENCE_TIER and row.get("classifier_label") == "reject"
            for row in members
        )
        has_inherited_safe = any(row.get("evidence_tier") == INHERITED_PARENT_SAFE_EVIDENCE_TIER for row in members)
        if has_active_reject and has_inherited_safe:
            conflict_groups.add(group_hash)
    return conflict_groups


def _apply_active_inherited_conflict_policy(
    rows: list[dict[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    policy = _required_active_inherited_conflict_policy(policy)
    conflict_groups = _active_inherited_conflict_groups(rows)
    if policy == ACTIVE_INHERITED_CONFLICT_POLICY_HARD_FAIL:
        if conflict_groups:
            raise ValueError("active-learning reject conflicts with inherited-safe connected component")
        return _active_inherited_conflict_stats(rows, conflict_groups=set(), policy=policy)
    for row in rows:
        group_hash = str(row["leakage_group_hash"])
        if group_hash not in conflict_groups:
            continue
        row["pre_audit_split"] = row.get("pre_audit_split") or row.get("split")
        row["split"] = "audit"
        row["fit_eligible"] = False
        row["eligible_for_eval"] = False
        row["training_weight"] = 0.0
        row["training_role"] = "audit_only_active_reject_inherited_safe_conflict_component"
        row["audit_policy_version"] = "augmented_v2_label_conflict_audit_v1"
        row["audit_exclusion_reason"] = ACTIVE_INHERITED_CONFLICT_AUDIT_REASON
        row["active_inherited_conflict_group_hash"] = stable_json_hash(
            {
                "conflict_type": ACTIVE_INHERITED_CONFLICT_AUDIT_REASON,
                "leakage_group_hash": group_hash,
            }
        )
    return _active_inherited_conflict_stats(rows, conflict_groups=conflict_groups, policy=policy)


def _active_inherited_conflict_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    conflict_groups: set[str],
    policy: str,
) -> dict[str, Any]:
    audited_rows = [row for row in rows if str(row.get("leakage_group_hash") or "") in conflict_groups]
    return {
        "active_inherited_conflict_policy": policy,
        "active_inherited_conflict_component_count": len(conflict_groups),
        "active_inherited_conflict_audited_row_count": len(audited_rows),
        "active_inherited_conflict_active_reject_row_count": sum(
            1
            for row in audited_rows
            if row.get("evidence_tier") == ACTIVE_LEARNING_EVIDENCE_TIER and row.get("classifier_label") == "reject"
        ),
        "active_inherited_conflict_inherited_safe_row_count": sum(
            1 for row in audited_rows if row.get("evidence_tier") == INHERITED_PARENT_SAFE_EVIDENCE_TIER
        ),
        "active_inherited_conflict_audited_label_counts": _sorted_counter(Counter(str(row["classifier_label"]) for row in audited_rows)),
        "active_inherited_conflict_audited_evidence_tier_counts": _sorted_counter(
            Counter(str(row["evidence_tier"]) for row in audited_rows)
        ),
        "active_inherited_conflict_audited_source_component_counts": _sorted_counter(
            Counter(str(row.get("source_universe_component") or "") for row in audited_rows)
        ),
    }


def _label_conflict_audit_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_label_conflict_audit_fraction: float,
) -> dict[str, Any]:
    if maximum_label_conflict_audit_fraction < 0.0 or maximum_label_conflict_audit_fraction > 1.0:
        raise ValueError("maximum_label_conflict_audit_fraction must be between 0 and 1")
    audited_rows = [row for row in rows if _is_label_conflict_audit_row(row)]
    audit_fraction = 0.0 if not rows else len(audited_rows) / len(rows)
    if audit_fraction > maximum_label_conflict_audit_fraction:
        raise ValueError("label conflict audit fraction exceeds configured maximum")
    final35k_eval_rows = [
        row
        for row in rows
        if row.get("source_universe_component") == "final_35k_chunk_classifier_v2" and row.get("split") == "eval"
    ]
    final35k_calibration_rows = [
        row
        for row in rows
        if row.get("source_universe_component") == "final_35k_chunk_classifier_v2" and row.get("split") == "calibration"
    ]
    return {
        "label_conflict_audited_row_count": len(audited_rows),
        "label_conflict_audit_fraction": round(audit_fraction, 8),
        "post_audit_final35k_eval_label_counts": _sorted_counter(Counter(str(row["classifier_label"]) for row in final35k_eval_rows)),
        "post_audit_final35k_eval_evidence_tier_counts": _sorted_counter(Counter(str(row["evidence_tier"]) for row in final35k_eval_rows)),
        "post_audit_final35k_calibration_label_counts": _sorted_counter(
            Counter(str(row["classifier_label"]) for row in final35k_calibration_rows)
        ),
        "post_audit_final35k_calibration_evidence_tier_counts": _sorted_counter(
            Counter(str(row["evidence_tier"]) for row in final35k_calibration_rows)
        ),
    }


def _force_active_learning_components_out_of_eval(rows: list[dict[str, Any]]) -> None:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["leakage_group_hash"])].append(row)
    for group_hash, members in by_group.items():
        if not any(row.get("evidence_tier") == ACTIVE_LEARNING_EVIDENCE_TIER for row in members):
            continue
        target_split = "calibration" if int(stable_json_hash({"active_group": group_hash})[:8], 16) % 5 == 0 else "train"
        for row in members:
            row["split"] = target_split
            row["eligible_for_eval"] = False


def _reject_active_reject_inherited_safe_conflicts(rows: Sequence[Mapping[str, Any]]) -> None:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["leakage_group_hash"])].append(row)
    for members in by_group.values():
        if members and all(_is_label_conflict_audit_row(row) for row in members):
            continue
        has_active_reject = any(
            row.get("evidence_tier") == ACTIVE_LEARNING_EVIDENCE_TIER and row.get("classifier_label") == "reject"
            for row in members
        )
        has_inherited_safe = any(row.get("evidence_tier") == INHERITED_PARENT_SAFE_EVIDENCE_TIER for row in members)
        if has_active_reject and has_inherited_safe:
            raise ValueError("active-learning reject conflicts with inherited-safe connected component")


def _require_final35k_eval_viability(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_final35k_eval_rows: int,
    minimum_final35k_eval_label_count: int,
    minimum_final35k_calibration_label_count: int,
) -> None:
    if minimum_final35k_eval_rows < 0:
        raise ValueError("minimum_final35k_eval_rows must be nonnegative")
    if minimum_final35k_eval_label_count < 0:
        raise ValueError("minimum_final35k_eval_label_count must be nonnegative")
    if minimum_final35k_calibration_label_count < 0:
        raise ValueError("minimum_final35k_calibration_label_count must be nonnegative")
    eval_rows = [
        row
        for row in rows
        if row.get("source_universe_component") == "final_35k_chunk_classifier_v2" and row.get("split") == "eval"
    ]
    if len(eval_rows) < int(minimum_final35k_eval_rows):
        raise ValueError("final35k eval split below augmented-manifest safety threshold")
    if eval_rows and {str(row.get("classifier_label") or "") for row in eval_rows} != {"safe", "reject"}:
        raise ValueError("final35k eval split must retain both safe and reject labels")
    eval_label_counts = Counter(str(row.get("classifier_label") or "") for row in eval_rows)
    for label in ("safe", "reject"):
        if eval_label_counts[label] < int(minimum_final35k_eval_label_count):
            raise ValueError("final35k eval label count below augmented-manifest safety threshold")
    calibration_rows = [
        row
        for row in rows
        if row.get("source_universe_component") == "final_35k_chunk_classifier_v2" and row.get("split") == "calibration"
    ]
    calibration_label_counts = Counter(str(row.get("classifier_label") or "") for row in calibration_rows)
    for label in ("safe", "reject"):
        if calibration_label_counts[label] < int(minimum_final35k_calibration_label_count):
            raise ValueError("final35k calibration label count below augmented-manifest safety threshold")


def _validate_core_text_conflict_policy_for_manifest(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    policy = _required_core_text_conflict_policy(str(manifest.get("core_text_conflict_policy") or CORE_TEXT_CONFLICT_POLICY_HARD_FAIL))
    conflict_text_hashes = _conflicting_text_label_hashes(rows)
    rows_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_group[str(row["leakage_group_hash"])].append(row)
    expected_audited_groups = {
        group_hash
        for group_hash, members in rows_by_group.items()
        if any(str(row.get("text_hash_sha256") or "") in conflict_text_hashes for row in members)
    }
    if policy == CORE_TEXT_CONFLICT_POLICY_HARD_FAIL and conflict_text_hashes:
        raise ValueError("conflicting classifier labels for identical text hash")
    if policy == CORE_TEXT_CONFLICT_POLICY_AUDIT_COMPONENTS:
        for group_hash, members in rows_by_group.items():
            should_audit = group_hash in expected_audited_groups
            if should_audit:
                group_conflict_hashes = sorted(
                    {
                        str(row.get("text_hash_sha256") or "")
                        for row in members
                        if str(row.get("text_hash_sha256") or "") in conflict_text_hashes
                    }
                )
                expected_group_hash = stable_json_hash(
                    {
                        "conflicting_text_hashes": group_conflict_hashes,
                        "leakage_group_hash": group_hash,
                    }
                )
                for row in members:
                    if not _is_core_text_conflict_audit_row(row):
                        raise ValueError("core text conflict component contains fit-eligible row")
                    if row.get("core_text_label_conflict_group_hash") != expected_group_hash:
                        raise ValueError("core text conflict group hash mismatch")
                    if int(row.get("core_text_label_conflict_hash_count", -1)) != len(group_conflict_hashes):
                        raise ValueError("core text conflict hash count mismatch")
                    expected_direct = str(row.get("text_hash_sha256") or "") in conflict_text_hashes
                    if row.get("core_text_label_conflict_direct") is not expected_direct:
                        raise ValueError("core text conflict direct flag mismatch")
            elif any(_is_core_text_conflict_audit_row(row) for row in members):
                raise ValueError("core text conflict audit marker outside conflicting component")
    elif any(_is_core_text_conflict_audit_row(row) for row in rows):
        raise ValueError("core text conflict audit rows require audit policy")

    expected_stats = _core_text_conflict_stats(
        rows,
        conflict_text_hashes=conflict_text_hashes,
        audited_groups=expected_audited_groups if policy == CORE_TEXT_CONFLICT_POLICY_AUDIT_COMPONENTS else set(),
        policy=policy,
    )
    for key, expected in expected_stats.items():
        actual = manifest.get(key)
        if actual != expected:
            raise ValueError(f"chunk classifier {key.replace('_', ' ')} mismatch")


def _validate_active_inherited_conflict_policy_for_manifest(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    policy = _required_active_inherited_conflict_policy(
        str(manifest.get("active_inherited_conflict_policy") or ACTIVE_INHERITED_CONFLICT_POLICY_HARD_FAIL)
    )
    conflict_groups = _active_inherited_conflict_groups(rows)
    rows_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_group[str(row["leakage_group_hash"])].append(row)
    if policy == ACTIVE_INHERITED_CONFLICT_POLICY_HARD_FAIL and conflict_groups:
        raise ValueError("active-learning reject conflicts with inherited-safe connected component")
    if policy == ACTIVE_INHERITED_CONFLICT_POLICY_AUDIT_COMPONENTS:
        for group_hash, members in rows_by_group.items():
            should_audit = group_hash in conflict_groups
            if should_audit:
                expected_group_hash = stable_json_hash(
                    {
                        "conflict_type": ACTIVE_INHERITED_CONFLICT_AUDIT_REASON,
                        "leakage_group_hash": group_hash,
                    }
                )
                for row in members:
                    if not _is_active_inherited_conflict_audit_row(row):
                        raise ValueError("active inherited conflict component contains fit-eligible row")
                    if row.get("active_inherited_conflict_group_hash") != expected_group_hash:
                        raise ValueError("active inherited conflict group hash mismatch")
            elif any(_is_active_inherited_conflict_audit_row(row) for row in members):
                raise ValueError("active inherited conflict audit marker outside conflicting component")
    elif any(_is_active_inherited_conflict_audit_row(row) for row in rows):
        raise ValueError("active inherited conflict audit rows require audit policy")

    expected_stats = _active_inherited_conflict_stats(
        rows,
        conflict_groups=conflict_groups if policy == ACTIVE_INHERITED_CONFLICT_POLICY_AUDIT_COMPONENTS else set(),
        policy=policy,
    )
    for key, expected in expected_stats.items():
        actual = manifest.get(key)
        if actual != expected:
            raise ValueError(f"chunk classifier {key.replace('_', ' ')} mismatch")


def _validate_label_conflict_audit_summary_for_manifest(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    min_eval_rows = _nonnegative_int(manifest.get("minimum_final35k_eval_rows"), "minimum_final35k_eval_rows")
    min_eval_label_count = _nonnegative_int(
        manifest.get("minimum_final35k_eval_label_count"),
        "minimum_final35k_eval_label_count",
    )
    min_calibration_label_count = _nonnegative_int(
        manifest.get("minimum_final35k_calibration_label_count"),
        "minimum_final35k_calibration_label_count",
    )
    max_fraction = float(manifest.get("maximum_label_conflict_audit_fraction"))
    _require_final35k_eval_viability(
        rows,
        minimum_final35k_eval_rows=min_eval_rows,
        minimum_final35k_eval_label_count=min_eval_label_count,
        minimum_final35k_calibration_label_count=min_calibration_label_count,
    )
    expected_stats = _label_conflict_audit_stats(
        rows,
        maximum_label_conflict_audit_fraction=max_fraction,
    )
    for key, expected in expected_stats.items():
        actual = manifest.get(key)
        if actual != expected:
            raise ValueError(f"chunk classifier {key.replace('_', ' ')} mismatch")


def build_chunk_classifier_training_manifest_from_stage_dirs(
    *,
    stage_dirs: Iterable[str | Path],
    repair_overlay_labels_jsonl: str | Path | None = None,
    repair_overlay_provenance_json: str | Path | None = None,
    weak_safe_candidates_jsonl: str | Path | None = None,
    split_seed: str = "twc-chunk-classifier-v1",
    source_artifact_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the classifier manifest directly from chunk relabel stage dirs."""

    candidates, label_rows, stage_hashes = _load_stage_candidates_and_labels(stage_dirs)
    repair_overlay = None
    if repair_overlay_labels_jsonl is not None or repair_overlay_provenance_json is not None:
        if repair_overlay_labels_jsonl is None or repair_overlay_provenance_json is None:
            raise ValueError("repair overlay labels and provenance paths must be supplied together")
        repair_overlay = load_fresh_chunk_repair_overlay(
            repair_overlay_labels_jsonl,
            repair_overlay_provenance_json,
        )
    effective = build_effective_chunk_label_evidence(label_rows, repair_overlay=repair_overlay)
    weak_candidates = _read_jsonl(Path(weak_safe_candidates_jsonl)) if weak_safe_candidates_jsonl is not None else []
    merged_hashes = {
        "stage_manifest_hashes_hash": stable_json_hash(stage_hashes),
        **dict(source_artifact_hashes or {}),
    }
    if repair_overlay is not None:
        merged_hashes["fresh_chunk_repair_overlay_hash"] = str(repair_overlay["fresh_chunk_repair_overlay_hash"])
    return build_chunk_classifier_training_manifest(
        chunk_candidates=candidates,
        effective_label_evidence=effective,
        weak_safe_candidates=weak_candidates,
        split_seed=split_seed,
        source_artifact_hashes=merged_hashes,
    )


def write_chunk_classifier_training_manifest_from_stage_dirs(
    *,
    stage_dirs: Iterable[str | Path],
    output_json: str | Path,
    repair_overlay_labels_jsonl: str | Path | None = None,
    repair_overlay_provenance_json: str | Path | None = None,
    weak_safe_candidates_jsonl: str | Path | None = None,
    split_seed: str = "twc-chunk-classifier-v1",
    source_artifact_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write a byte-stable public classifier manifest from stage directories."""

    manifest = build_chunk_classifier_training_manifest_from_stage_dirs(
        stage_dirs=stage_dirs,
        repair_overlay_labels_jsonl=repair_overlay_labels_jsonl,
        repair_overlay_provenance_json=repair_overlay_provenance_json,
        weak_safe_candidates_jsonl=weak_safe_candidates_jsonl,
        split_seed=split_seed,
        source_artifact_hashes=source_artifact_hashes,
    )
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build/validate no-raw chunk classifier manifests")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-from-stage-dirs")
    build.add_argument("--stage-dir", action="append", default=[])
    build.add_argument("--stage-root")
    build.add_argument("--stage-glob", default="rejected_35k_chunk_relabel_stage*")
    build.add_argument("--repair-overlay-labels-jsonl")
    build.add_argument("--repair-overlay-provenance-json")
    build.add_argument("--weak-safe-candidates-jsonl")
    build.add_argument("--split-seed", default="twc-chunk-classifier-v1")
    build.add_argument("--source-artifact-hash", action="append", default=[])
    build.add_argument("--output-json", required=True)

    build_v2 = subparsers.add_parser("build-v2-from-stage-dirs")
    build_v2.add_argument("--stage-dir", action="append", default=[])
    build_v2.add_argument("--stage-root")
    build_v2.add_argument("--stage-glob", default="rejected_35k_chunk_relabel_stage*")
    build_v2.add_argument("--repair-overlay-labels-jsonl")
    build_v2.add_argument("--repair-overlay-provenance-json")
    build_v2.add_argument("--inherited-safe-candidates-jsonl", required=True)
    build_v2.add_argument("--split-seed", default="twc-chunk-classifier-v2")
    build_v2.add_argument("--source-artifact-hash", action="append", default=[])
    build_v2.add_argument("--expected-safe-parent-count", type=int, default=FINAL_35K_SAFE_PARENT_COUNT)
    build_v2.add_argument("--expected-reject-parent-count", type=int, default=FINAL_35K_REJECT_PARENT_COUNT)
    build_v2.add_argument("--expected-repair-overlay-label-count", type=int, default=FINAL_35K_REPAIR_OVERLAY_LABEL_COUNT)
    build_v2.add_argument("--output-json", required=True)

    build_augmented = subparsers.add_parser("build-augmented-v2-active-learning")
    build_augmented.add_argument("--base-manifest", required=True)
    build_augmented.add_argument(
        "--active-learning-batch",
        action="append",
        required=True,
        help="batch_id:schedule_json:labels_jsonl:votes_jsonl:core_hash_map_json",
    )
    build_augmented.add_argument("--split-seed", default="twc-chunk-classifier-v2-active-learning-augmented")
    build_augmented.add_argument("--source-artifact-hash", action="append", default=[])
    build_augmented.add_argument("--minimum-final35k-eval-rows", type=int, default=2)
    build_augmented.add_argument("--minimum-final35k-eval-label-count", type=int, default=0)
    build_augmented.add_argument("--minimum-final35k-calibration-label-count", type=int, default=0)
    build_augmented.add_argument("--maximum-label-conflict-audit-fraction", type=float, default=1.0)
    build_augmented.add_argument(
        "--core-text-conflict-policy",
        choices=sorted(CORE_TEXT_CONFLICT_POLICIES),
        default=CORE_TEXT_CONFLICT_POLICY_HARD_FAIL,
    )
    build_augmented.add_argument(
        "--active-inherited-conflict-policy",
        choices=sorted(ACTIVE_INHERITED_CONFLICT_POLICIES),
        default=ACTIVE_INHERITED_CONFLICT_POLICY_HARD_FAIL,
    )
    build_augmented.add_argument("--output-json", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--input-json", required=True)
    args = parser.parse_args(argv)

    if args.command == "build-from-stage-dirs":
        source_artifact_hashes = _parse_source_artifact_hashes(args.source_artifact_hash)
        stage_dirs = [Path(path) for path in args.stage_dir]
        if args.stage_root:
            stage_dirs.extend(_complete_stage_dirs_from_root(Path(args.stage_root), args.stage_glob))
        manifest = write_chunk_classifier_training_manifest_from_stage_dirs(
            stage_dirs=stage_dirs,
            output_json=args.output_json,
            repair_overlay_labels_jsonl=args.repair_overlay_labels_jsonl,
            repair_overlay_provenance_json=args.repair_overlay_provenance_json,
            weak_safe_candidates_jsonl=args.weak_safe_candidates_jsonl,
            split_seed=args.split_seed,
            source_artifact_hashes=source_artifact_hashes,
        )
        print(
            json.dumps(
                {
                    "status": "written",
                    "output_json": str(args.output_json),
                    "candidate_count": manifest["candidate_count"],
                    "training_manifest_hash": manifest["training_manifest_hash"],
                    "raw_text_serialized": False,
                    "raw_url_serialized": False,
                    "token_serialized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-v2-from-stage-dirs":
        source_artifact_hashes = _parse_source_artifact_hashes(args.source_artifact_hash)
        stage_dirs = [Path(path) for path in args.stage_dir]
        if args.stage_root:
            stage_dirs.extend(_complete_stage_dirs_from_root(Path(args.stage_root), args.stage_glob))
        candidates, label_rows, stage_hashes = _load_stage_candidates_and_labels(stage_dirs)
        repair_overlay = None
        if args.repair_overlay_labels_jsonl is not None or args.repair_overlay_provenance_json is not None:
            if args.repair_overlay_labels_jsonl is None or args.repair_overlay_provenance_json is None:
                raise ValueError("repair overlay labels and provenance paths must be supplied together")
            repair_overlay = load_fresh_chunk_repair_overlay(
                args.repair_overlay_labels_jsonl,
                args.repair_overlay_provenance_json,
            )
        effective = build_effective_chunk_label_evidence(label_rows, repair_overlay=repair_overlay)
        merged_hashes = {
            "stage_manifest_hashes_hash": stable_json_hash(stage_hashes),
            **dict(source_artifact_hashes or {}),
        }
        if repair_overlay is not None:
            merged_hashes["fresh_chunk_repair_overlay_hash"] = str(repair_overlay["fresh_chunk_repair_overlay_hash"])
        manifest = build_v2_chunk_classifier_training_manifest(
            chunk_candidates=candidates,
            effective_label_evidence=effective,
            inherited_safe_candidates=_read_jsonl(Path(args.inherited_safe_candidates_jsonl)),
            split_seed=args.split_seed,
            source_artifact_hashes=merged_hashes,
            expected_safe_parent_count=args.expected_safe_parent_count,
            expected_reject_parent_count=args.expected_reject_parent_count,
            expected_repair_overlay_label_count=args.expected_repair_overlay_label_count,
        )
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "written",
                    "output_json": str(args.output_json),
                    "schema_version": manifest["schema_version"],
                    "candidate_count": manifest["candidate_count"],
                    "training_manifest_hash": manifest["training_manifest_hash"],
                    "raw_text_serialized": False,
                    "raw_url_serialized": False,
                    "token_serialized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-augmented-v2-active-learning":
        source_artifact_hashes = _parse_source_artifact_hashes(args.source_artifact_hash)
        batches = [_load_active_learning_batch_spec(item) for item in args.active_learning_batch]
        manifest = build_augmented_v2_chunk_classifier_training_manifest(
            base_manifest=json.loads(Path(args.base_manifest).read_text(encoding="utf-8")),
            active_learning_batches=batches,
            split_seed=args.split_seed,
            source_artifact_hashes=source_artifact_hashes,
            minimum_final35k_eval_rows=args.minimum_final35k_eval_rows,
            minimum_final35k_eval_label_count=args.minimum_final35k_eval_label_count,
            minimum_final35k_calibration_label_count=args.minimum_final35k_calibration_label_count,
            maximum_label_conflict_audit_fraction=args.maximum_label_conflict_audit_fraction,
            core_text_conflict_policy=args.core_text_conflict_policy,
            active_inherited_conflict_policy=args.active_inherited_conflict_policy,
        )
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "written",
                    "output_json": str(args.output_json),
                    "schema_version": manifest["schema_version"],
                    "source_universe": manifest["source_universe"],
                    "candidate_count": manifest["candidate_count"],
                    "active_learning_label_count": manifest["active_learning_label_count"],
                    "core_text_conflict_policy": manifest["core_text_conflict_policy"],
                    "core_text_conflict_hash_count": manifest["core_text_conflict_hash_count"],
                    "core_text_conflict_audited_component_count": manifest["core_text_conflict_audited_component_count"],
                    "core_text_conflict_audited_row_count": manifest["core_text_conflict_audited_row_count"],
                    "active_inherited_conflict_policy": manifest["active_inherited_conflict_policy"],
                    "active_inherited_conflict_component_count": manifest["active_inherited_conflict_component_count"],
                    "active_inherited_conflict_audited_row_count": manifest["active_inherited_conflict_audited_row_count"],
                    "label_conflict_audited_row_count": manifest["label_conflict_audited_row_count"],
                    "label_conflict_audit_fraction": manifest["label_conflict_audit_fraction"],
                    "training_manifest_hash": manifest["training_manifest_hash"],
                    "raw_text_serialized": False,
                    "raw_url_serialized": False,
                    "token_serialized": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate":
        manifest = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        validate_chunk_classifier_training_manifest(manifest)
        print(json.dumps({"status": "valid", "training_manifest_hash": manifest["training_manifest_hash"]}, sort_keys=True))
        return 0
    raise AssertionError(args.command)


def validate_chunk_classifier_training_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the public no-raw chunk classifier manifest."""

    validate_public_no_raw_payload(manifest)
    if manifest.get("manifest_kind") != CHUNK_CLASSIFIER_MANIFEST_KIND:
        raise ValueError("unsupported chunk classifier manifest kind")
    schema_version = int(manifest.get("schema_version", -1))
    if schema_version not in {CHUNK_CLASSIFIER_SCHEMA_VERSION, CHUNK_CLASSIFIER_SCHEMA_VERSION_V2}:
        raise ValueError("unsupported chunk classifier manifest schema version")
    source_universe = _manifest_source_universe(manifest)
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2:
        if source_universe not in ALLOWED_V2_SOURCE_UNIVERSES:
            raise ValueError("v2 chunk classifier manifest source_universe mismatch")
    _require_false_flags(manifest, PUBLIC_FALSE_FLAGS, "chunk classifier manifest")
    rows = list(manifest.get("rows") or [])
    if int(manifest.get("candidate_count", -1)) != len(rows):
        raise ValueError("chunk classifier candidate_count mismatch")
    seen_ids: set[str] = set()
    split_by_leakage_group: dict[str, str] = {}
    for index, row in enumerate(rows):
        _validate_manifest_row(row, index=index, schema_version=schema_version, source_universe=source_universe)
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen_ids:
            raise ValueError("chunk classifier manifest contains duplicate candidate_id")
        seen_ids.add(candidate_id)
        leakage_group = str(row["leakage_group_hash"])
        split = str(row["split"])
        previous = split_by_leakage_group.setdefault(leakage_group, split)
        if previous != split:
            raise ValueError("leakage group crosses split boundary")
    _validate_leakage_identity_components(rows)
    expected_counts = {
        "strong_label_count": sum(1 for row in rows if row["evidence_tier"] in {STRONG_EVIDENCE_TIER, FRESH_REPAIR_EVIDENCE_TIER}),
        "weak_label_count": sum(1 for row in rows if row["evidence_tier"] == WEAK_SAFE_EVIDENCE_TIER),
        "repair_overlay_label_count": sum(1 for row in rows if row["evidence_tier"] == FRESH_REPAIR_EVIDENCE_TIER),
        "fit_eligible_count": sum(1 for row in rows if row["fit_eligible"]),
        "eval_eligible_count": sum(1 for row in rows if row["eligible_for_eval"]),
    }
    inherited_safe_count = sum(1 for row in rows if row["evidence_tier"] == INHERITED_PARENT_SAFE_EVIDENCE_TIER)
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2 or "inherited_safe_label_count" in manifest:
        expected_counts["inherited_safe_label_count"] = inherited_safe_count
    elif inherited_safe_count:
        raise ValueError("legacy chunk classifier manifest contains inherited safe rows")
    for key, expected in expected_counts.items():
        if int(manifest.get(key, -1)) != expected:
            raise ValueError(f"chunk classifier manifest {key} mismatch")
    if dict(manifest.get("label_counts") or {}) != _sorted_counter(Counter(row["classifier_label"] for row in rows)):
        raise ValueError("chunk classifier label_counts mismatch")
    if dict(manifest.get("evidence_tier_counts") or {}) != _sorted_counter(Counter(row["evidence_tier"] for row in rows)):
        raise ValueError("chunk classifier evidence_tier_counts mismatch")
    if dict(manifest.get("split_counts") or {}) != _sorted_counter(Counter(row["split"] for row in rows)):
        raise ValueError("chunk classifier split_counts mismatch")
    parent_coverage_counts = {
        "reject": len(
            {
                str(row["parent_candidate_id"])
                for row in rows
                if row["evidence_tier"] in {STRONG_EVIDENCE_TIER, FRESH_REPAIR_EVIDENCE_TIER}
            }
        ),
        "safe": len(
            {
                str(row["parent_candidate_id"])
                for row in rows
                if row["evidence_tier"] == INHERITED_PARENT_SAFE_EVIDENCE_TIER
            }
        ),
    }
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2 and source_universe == FINAL_35K_CHUNK_CLASSIFIER_V2_SOURCE_UNIVERSE:
        if dict(manifest.get("parent_coverage_counts") or {}) != parent_coverage_counts:
            raise ValueError("chunk classifier parent_coverage_counts mismatch")
        expected_parent_counts = manifest.get("final_35k_expected_parent_counts")
        if isinstance(expected_parent_counts, Mapping):
            expected_safe = expected_parent_counts.get("safe")
            expected_reject = expected_parent_counts.get("reject")
            if expected_safe is not None and parent_coverage_counts["safe"] != int(expected_safe):
                raise ValueError("safe parent coverage count mismatch")
            if expected_reject is not None and parent_coverage_counts["reject"] != int(expected_reject):
                raise ValueError("reject parent coverage count mismatch")
        if any(row["evidence_tier"] == WEAK_SAFE_EVIDENCE_TIER for row in rows):
            raise ValueError("legacy weak parent safe rows cannot enter v2 manifest")
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2 and source_universe == AUGMENTED_V2_SOURCE_UNIVERSE:
        active_rows = [row for row in rows if row.get("evidence_tier") == ACTIVE_LEARNING_EVIDENCE_TIER]
        final35k_rows = [row for row in rows if row.get("evidence_tier") != ACTIVE_LEARNING_EVIDENCE_TIER]
        source_artifact_hashes = manifest.get("source_artifact_hashes") or {}
        if not isinstance(source_artifact_hashes, Mapping):
            raise ValueError("chunk classifier source_artifact_hashes must be an object")
        base_manifest_hash = source_artifact_hashes.get("base_final35k_training_manifest_hash")
        if final35k_rows:
            base_manifest_hash = _required_hash(base_manifest_hash, "base_final35k_training_manifest_hash")
        for row in final35k_rows:
            if row.get("source_universe_component") != "final_35k_chunk_classifier_v2":
                raise ValueError("augmented final35k row source_universe_component mismatch")
            if _required_hash(row.get("original_training_manifest_hash"), "original training manifest hash") != base_manifest_hash:
                raise ValueError("augmented final35k original manifest hash mismatch")
            _required_hash(row.get("original_row_hash"), "original row hash")
        for row in active_rows:
            if row.get("source_universe_component") != "active_learning_bof3":
                raise ValueError("active-learning row source_universe_component mismatch")
        if int(manifest.get("active_learning_label_count", -1)) != len(active_rows):
            raise ValueError("chunk classifier active_learning_label_count mismatch")
        final35k_parent_counts = {
            "reject": len(
                {
                    str(row["parent_candidate_id"])
                    for row in rows
                    if row.get("source_universe_component") == "final_35k_chunk_classifier_v2"
                    and row["evidence_tier"] in {STRONG_EVIDENCE_TIER, FRESH_REPAIR_EVIDENCE_TIER}
                }
            ),
            "safe": len(
                {
                    str(row["parent_candidate_id"])
                    for row in rows
                    if row.get("source_universe_component") == "final_35k_chunk_classifier_v2"
                    and row["evidence_tier"] == INHERITED_PARENT_SAFE_EVIDENCE_TIER
                }
            ),
        }
        if dict(manifest.get("final35k_parent_coverage_counts") or {}) != final35k_parent_counts:
            raise ValueError("chunk classifier final35k_parent_coverage_counts mismatch")
        expected_final35k_candidate_count = int(manifest.get("expected_final35k_candidate_count", len(final35k_rows)))
        if len(final35k_rows) != expected_final35k_candidate_count:
            raise ValueError("chunk classifier expected final35k candidate count mismatch")
        expected_parent_counts = manifest.get("final_35k_expected_parent_counts")
        if isinstance(expected_parent_counts, Mapping) and final35k_rows:
            expected_safe = expected_parent_counts.get("safe")
            expected_reject = expected_parent_counts.get("reject")
            if expected_safe is not None and final35k_parent_counts["safe"] != int(expected_safe):
                raise ValueError("augmented safe parent coverage count mismatch")
            if expected_reject is not None and final35k_parent_counts["reject"] != int(expected_reject):
                raise ValueError("augmented reject parent coverage count mismatch")
        expected_active_count = manifest.get("expected_active_learning_label_count")
        if expected_active_count is not None and len(active_rows) != int(expected_active_count):
            raise ValueError("chunk classifier expected active-learning label count mismatch")
        expected_batch_count = manifest.get("expected_active_learning_batch_count")
        batch_counts_actual = _sorted_counter(Counter(str(row["active_learning_batch_id"]) for row in active_rows))
        if expected_batch_count is not None and len(batch_counts_actual) != int(expected_batch_count):
            raise ValueError("chunk classifier expected active-learning batch count mismatch")
        expected_rows_per_batch = manifest.get("expected_active_learning_rows_per_batch")
        if expected_rows_per_batch is not None and any(count != int(expected_rows_per_batch) for count in batch_counts_actual.values()):
            raise ValueError("chunk classifier expected active-learning per-batch count mismatch")
        if any(row.get("eligible_for_eval") is True or row.get("split") == "eval" for row in active_rows):
            raise ValueError("active-learning rows must not enter eval")
        _validate_core_text_conflict_policy_for_manifest(manifest, rows)
        _validate_active_inherited_conflict_policy_for_manifest(manifest, rows)
        _validate_label_conflict_audit_summary_for_manifest(manifest, rows)
        rows_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_group[str(row["leakage_group_hash"])].append(row)
        for members in rows_by_group.values():
            if any(row.get("evidence_tier") == ACTIVE_LEARNING_EVIDENCE_TIER for row in members):
                if any(row.get("eligible_for_eval") is True or row.get("split") == "eval" for row in members):
                    raise ValueError("active-learning connected components must not enter eval")
                if members and all(_is_label_conflict_audit_row(row) for row in members):
                    continue
                if any(
                    row.get("evidence_tier") == ACTIVE_LEARNING_EVIDENCE_TIER and row.get("classifier_label") == "reject"
                    for row in members
                ) and any(row.get("evidence_tier") == INHERITED_PARENT_SAFE_EVIDENCE_TIER for row in members):
                    raise ValueError("active-learning reject conflicts with inherited-safe connected component")
        if dict(manifest.get("active_learning_final_label_counts") or {}) != _sorted_counter(
            Counter(str(row["active_learning_final_label"]) for row in active_rows)
        ):
            raise ValueError("chunk classifier active_learning_final_label_counts mismatch")
        if dict(manifest.get("active_learning_batch_counts") or {}) != batch_counts_actual:
            raise ValueError("chunk classifier active_learning_batch_counts mismatch")
        _nonnegative_int(
            manifest.get("active_learning_unselected_parse_error_vote_count", 0),
            "active_learning_unselected_parse_error_vote_count",
        )
    expected_hash = stable_json_hash({key: value for key, value in manifest.items() if key != "training_manifest_hash"})
    if manifest.get("training_manifest_hash") != expected_hash:
        raise ValueError("chunk classifier training manifest hash mismatch")


def _validate_leakage_identity_components(rows: Sequence[Mapping[str, Any]]) -> None:
    by_identity: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for field in LEAKAGE_IDENTITY_FIELDS:
            value = row.get(field)
            if value:
                by_identity[(field, str(value))].append(row)
    for members in by_identity.values():
        if len(members) < 2:
            continue
        splits = {str(row.get("split") or "") for row in members}
        if len(splits) != 1:
            raise ValueError("identity component crosses split boundary")
        leakage_groups = {str(row.get("leakage_group_hash") or "") for row in members}
        if len(leakage_groups) != 1:
            raise ValueError("identity component has inconsistent leakage group")


def _strong_manifest_row(candidate: dict[str, Any], evidence: Mapping[str, Any], *, schema_version: int) -> dict[str, Any]:
    label = _required_label(evidence.get("final_label") or evidence.get("label"), "final_label")
    compact_validated_evidence = (
        evidence.get("chunk_label_evidence_schema_version") == 1
        and evidence.get("chunk_label_evidence_hash") not in (None, "")
    )
    parsed_vote_count = _nonnegative_int(
        evidence.get("parsed_vote_count", 3 if compact_validated_evidence else None),
        "parsed_vote_count",
    )
    required_vote_count = _positive_int(
        evidence.get("required_vote_count", 3 if compact_validated_evidence else None),
        "required_vote_count",
    )
    parse_error_vote_count = _nonnegative_int(
        evidence.get("parse_error_vote_count", 0 if compact_validated_evidence else None),
        "parse_error_vote_count",
    )
    if parsed_vote_count == 0 and parse_error_vote_count > 0:
        raise ValueError("parse-error-only chunk label evidence cannot train classifier")
    if parsed_vote_count < required_vote_count:
        raise ValueError("chunk label evidence is missing selected parsed votes")
    source = str(evidence.get("label_source") or "canonical_chunk_label")
    if source == FRESH_REPAIR_EVIDENCE_TIER:
        evidence_tier = FRESH_REPAIR_EVIDENCE_TIER
    elif source in {"canonical", "canonical_chunk_label"}:
        evidence_tier = STRONG_EVIDENCE_TIER
    else:
        raise ValueError(f"unsupported chunk label source: {source}")
    _require_false_flags(evidence, ("raw_text_serialized", "raw_url_serialized", "token_serialized"), "chunk label evidence")
    if _required_hash(evidence.get("candidate_hash"), "candidate_hash") != candidate["candidate_hash"]:
        raise ValueError("chunk candidate/evidence candidate_hash mismatch")
    if _required_hash(evidence.get("parent_candidate_id"), "parent_candidate_id") != candidate["parent_candidate_id"]:
        raise ValueError("chunk candidate/evidence parent_candidate_id mismatch")
    if _required_hash(evidence.get("source_chunk_manifest_hash"), "source_chunk_manifest_hash") != candidate["source_chunk_manifest_hash"]:
        raise ValueError("chunk candidate/evidence source_chunk_manifest_hash mismatch")
    return {
        **_common_public_row(candidate, schema_version=schema_version),
        "classifier_label": label,
        "evidence_tier": evidence_tier,
        "label_source": source,
        "parent_label_source": "final_35k_rejected_parent_chunk_relabel" if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2 else "rejected_parent_chunk_relabel",
        "canonical_final_label": _optional_label(evidence.get("canonical_final_label")),
        "aggregation_route_hash": stable_json_hash({"aggregation_route": str(evidence.get("aggregation_route") or "")}),
        "parsed_vote_count": parsed_vote_count,
        "required_vote_count": required_vote_count,
        "parse_error_vote_count": parse_error_vote_count,
        "transport_failure_count": _nonnegative_int(evidence.get("transport_failure_count", 0), "transport_failure_count"),
        "fit_eligible": True,
        "eligible_for_eval": True,
        "training_weight": 1.0,
        "training_role": "strong_chunk_reject" if label == "reject" else "strong_chunk_safe",
        "private_cache_route": "rejected_chunk_stage_private_map",
    }


def _weak_safe_manifest_row(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        **_common_public_row(candidate, schema_version=CHUNK_CLASSIFIER_SCHEMA_VERSION),
        "classifier_label": "safe",
        "evidence_tier": WEAK_SAFE_EVIDENCE_TIER,
        "label_source": "safe_parent_weak_chunk_label",
        "parent_label_source": "legacy_safe_parent_weak_label",
        "canonical_final_label": "safe",
        "aggregation_route_hash": stable_json_hash({"aggregation_route": "weak_safe_parent_inheritance"}),
        "parsed_vote_count": 0,
        "required_vote_count": 0,
        "parse_error_vote_count": 0,
        "transport_failure_count": 0,
        "fit_eligible": True,
        "eligible_for_eval": False,
        "training_weight": WEAK_SAFE_TRAINING_WEIGHT,
        "training_role": "weak_parent_safe_train_only",
        "private_cache_route": "legacy_weak_safe_not_cacheable",
    }


def _inherited_safe_manifest_row(candidate: dict[str, Any]) -> dict[str, Any]:
    if str(candidate.get("teacher_label_use") or "") == "safe_parent_weak_chunk_label":
        raise ValueError("legacy weak parent safe candidates cannot enter v2 manifest")
    parent_label_source = str(candidate.get("parent_label_source") or "final_35k_bof3_parent_label")
    if not parent_label_source:
        raise ValueError("inherited safe candidate requires parent_label_source")
    return {
        **_common_public_row(candidate, schema_version=CHUNK_CLASSIFIER_SCHEMA_VERSION_V2),
        "classifier_label": "safe",
        "evidence_tier": INHERITED_PARENT_SAFE_EVIDENCE_TIER,
        "label_source": "inherited_parent_safe_v2",
        "parent_label_source": parent_label_source,
        "canonical_final_label": "safe",
        "aggregation_route_hash": stable_json_hash({"aggregation_route": "final_35k_parent_safe_inheritance_v2"}),
        "parsed_vote_count": 0,
        "required_vote_count": 0,
        "parse_error_vote_count": 0,
        "transport_failure_count": 0,
        "fit_eligible": True,
        "eligible_for_eval": True,
        "training_weight": 1.0,
        "training_role": "inherited_parent_safe_v2",
        "private_cache_route": "safe_parent_source_refetch",
    }


def _common_public_row(candidate: Mapping[str, Any], *, schema_version: int) -> dict[str, Any]:
    return {
        "row_schema_version": schema_version,
        "candidate_id": candidate["candidate_id"],
        "candidate_hash": candidate["candidate_hash"],
        "parent_candidate_id": candidate["parent_candidate_id"],
        "source_chunk_manifest_hash": candidate["source_chunk_manifest_hash"],
        "source_document_identity_hash": candidate["source_document_identity_hash"],
        "source_file_ref_hash": candidate["source_file_ref_hash"],
        "dataset_identity_hash": candidate["dataset_identity_hash"],
        "physical_source_row_key_hash": candidate["physical_source_row_key_hash"],
        "parent_text_hash_sha256": candidate["parent_text_hash_sha256"],
        "text_hash_sha256": candidate["text_hash_sha256"],
        "normalized_text_hash_sha256": candidate["normalized_text_hash_sha256"],
        "schedule_entry_hash": candidate["schedule_entry_hash"],
        "source_schedule_hash": candidate["source_schedule_hash"],
        "source_metadata_hash": candidate["source_metadata_hash"],
        "row_group_index": candidate["row_group_index"],
        "row_group_local_row_index": candidate["row_group_local_row_index"],
        "global_chunk_index": candidate["global_chunk_index"],
        "label_unit": candidate["label_unit"],
        "leakage_group_hash": candidate["leakage_group_hash"],
        "source_bucket_hash": candidate["source_bucket_hash"],
        "token_count_bucket": candidate["token_count_bucket"],
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }


def _plan_leakage_safe_splits(rows: list[dict[str, Any]], *, split_seed: str) -> dict[str, str]:
    if not rows:
        return {}
    union = _UnionFind(row["candidate_id"] for row in rows)
    by_identity: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        for field in LEAKAGE_IDENTITY_FIELDS:
            value = row.get(field)
            if value:
                by_identity[(field, str(value))].append(row["candidate_id"])
    for members in by_identity.values():
        first = members[0]
        for member in members[1:]:
            union.union(first, member)
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        root = union.find(row["candidate_id"])
        components[root].append(row)
    for members in components.values():
        leakage_group_hash = stable_json_hash(
            {
                "leakage_component_candidate_ids": sorted(row["candidate_id"] for row in members),
                "split_seed": split_seed,
            }
        )
        for row in members:
            row["leakage_group_hash"] = leakage_group_hash

    split_rows = []
    for root, members in sorted(components.items()):
        has_strong = any(row["evidence_tier"] != WEAK_SAFE_EVIDENCE_TIER for row in members)
        labels = sorted({row["classifier_label"] for row in members})
        tiers = sorted({row["evidence_tier"] for row in members})
        tier = "strong" if has_strong else "weak_safe_only"
        split_rows.append(
            {
                "candidate_id": root,
                "dedupe_group_hash": members[0]["leakage_group_hash"],
                "stratum_key": ":".join(labels + [tier] + tiers),
                "fit_eligible": True,
            }
        )
    component_splits = plan_dedupe_aware_splits(split_rows, split_seed=split_seed)
    assignments: dict[str, str] = {}
    for root, members in components.items():
        has_strong = any(row["evidence_tier"] != WEAK_SAFE_EVIDENCE_TIER for row in members)
        split = component_splits[root] if has_strong else "train"
        for row in members:
            assignments[row["candidate_id"]] = split
    return assignments


def _normalized_candidate(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    _reject_forbidden_input_keys(row)
    _require_false_flags(row, ("raw_text_serialized", "raw_url_serialized", "token_serialized"), "chunk candidate")
    source_doc_id = str(row.get("source_doc_id") or row.get("source_document_id") or "")
    source_file_id = str(row.get("source_file_id") or "")
    dataset_payload = {
        "dataset_repo_id": str(row.get("dataset_repo_id") or ""),
        "dataset_config": str(row.get("dataset_config") or ""),
        "dataset_split": str(row.get("dataset_split") or ""),
        "resolved_repo_sha": str(row.get("resolved_repo_sha") or ""),
    }
    source_bucket_hash = row.get("source_bucket_hash") or stable_json_hash(
        {
            "dataset": dataset_payload,
            "source_file_id": source_file_id,
            "source_metadata_hash": str(row.get("source_metadata_hash") or ""),
        }
    )
    return {
        "candidate_id": _required_hash(row.get("candidate_id"), "candidate_id"),
        "candidate_hash": _required_hash(row.get("candidate_hash"), "candidate_hash"),
        "parent_candidate_id": _required_hash(row.get("parent_candidate_id"), "parent_candidate_id"),
        "source_chunk_manifest_hash": _required_hash(row.get("source_chunk_manifest_hash"), "source_chunk_manifest_hash"),
        "physical_source_row_key_hash": _required_hash(row.get("physical_source_row_key_hash"), "physical_source_row_key_hash"),
        "parent_text_hash_sha256": _optional_hash(row.get("parent_text_hash_sha256"), "parent_text_hash_sha256"),
        "text_hash_sha256": _required_hash(row.get("text_hash_sha256"), "text_hash_sha256"),
        "normalized_text_hash_sha256": _required_hash(row.get("normalized_text_hash_sha256"), "normalized_text_hash_sha256"),
        "schedule_entry_hash": _required_hash(row.get("schedule_entry_hash"), "schedule_entry_hash"),
        "source_schedule_hash": _required_hash(row.get("source_schedule_hash"), "source_schedule_hash"),
        "source_metadata_hash": _optional_hash(row.get("source_metadata_hash"), "source_metadata_hash"),
        "row_group_index": _nonnegative_int(row.get("row_group_index"), "row_group_index"),
        "row_group_local_row_index": _nonnegative_int(row.get("row_group_local_row_index"), "row_group_local_row_index"),
        "global_chunk_index": _nonnegative_int(row.get("global_chunk_index", index), "global_chunk_index"),
        "label_unit": str(row.get("label_unit") or "classifier_candidate_chunk"),
        "source_document_identity_hash": stable_json_hash({"source_doc_id": source_doc_id or str(row.get("parent_candidate_id") or "")}),
        "source_file_ref_hash": stable_json_hash({"source_file_id": source_file_id}),
        "dataset_identity_hash": stable_json_hash(dataset_payload),
        "leakage_group_hash": stable_json_hash({"candidate_id": str(row.get("candidate_id") or "")}),
        "source_bucket_hash": _required_hash(source_bucket_hash, "source_bucket_hash"),
        "token_count_bucket": str(row.get("token_count_bucket") or "unknown"),
        "teacher_label_use": str(row.get("teacher_label_use") or ""),
        "parent_label_source": str(row.get("parent_label_source") or ""),
    }


def _validated_evidence_map(effective_label_evidence: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    validate_public_no_raw_payload(effective_label_evidence)
    raw = effective_label_evidence.get("labels_by_candidate_id")
    if not isinstance(raw, Mapping):
        raise ValueError("effective_label_evidence must contain labels_by_candidate_id")
    by_candidate: dict[str, Mapping[str, Any]] = {}
    for key, row in raw.items():
        if not isinstance(row, Mapping):
            raise ValueError("effective label evidence row must be an object")
        candidate_id = _required_hash(row.get("candidate_id"), "candidate_id")
        if candidate_id != str(key):
            raise ValueError("effective label evidence key/candidate_id mismatch")
        if candidate_id in by_candidate:
            raise ValueError("duplicate candidate_id in effective label evidence")
        by_candidate[candidate_id] = row
    return by_candidate


def _validate_manifest_row(row: Mapping[str, Any], *, index: int, schema_version: int, source_universe: str = "") -> None:
    _require_false_flags(row, PUBLIC_FALSE_FLAGS, f"chunk classifier row {index}")
    for field in (
        "candidate_id",
        "candidate_hash",
        "parent_candidate_id",
        "source_chunk_manifest_hash",
        "source_document_identity_hash",
        "source_file_ref_hash",
        "dataset_identity_hash",
        "physical_source_row_key_hash",
        "text_hash_sha256",
        "normalized_text_hash_sha256",
        "schedule_entry_hash",
        "source_schedule_hash",
        "leakage_group_hash",
        "row_hash",
    ):
        _required_hash(row.get(field), field)
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2 or row.get("source_bucket_hash") not in (None, ""):
        _required_hash(row.get("source_bucket_hash"), "source_bucket_hash")
    _optional_hash(row.get("parent_text_hash_sha256"), "parent_text_hash_sha256")
    _optional_hash(row.get("source_metadata_hash"), "source_metadata_hash")
    if row.get("classifier_label") not in ALLOWED_LABELS:
        raise ValueError("chunk classifier row label must be safe or reject")
    if row.get("split") not in ALLOWED_SPLITS:
        raise ValueError("chunk classifier row split is unsupported")
    if row.get("label_unit") not in {"classifier_candidate_chunk", "window_bundle"}:
        raise ValueError("chunk classifier row label_unit must be chunk-level")
    tier = str(row.get("evidence_tier") or "")
    allowed_tiers = _allowed_evidence_tiers_for_source_universe(source_universe, schema_version=schema_version)
    if tier not in allowed_tiers:
        raise ValueError("chunk classifier row evidence_tier is unsupported")
    is_core_conflict_audit = _is_core_text_conflict_audit_row(row)
    is_active_inherited_conflict_audit = _is_active_inherited_conflict_audit_row(row)
    is_label_conflict_audit = is_core_conflict_audit or is_active_inherited_conflict_audit
    if is_label_conflict_audit:
        if source_universe != AUGMENTED_V2_SOURCE_UNIVERSE:
            raise ValueError("label conflict audit rows require augmented source_universe")
        if row.get("split") != "audit" or row.get("fit_eligible") is not False or row.get("eligible_for_eval") is not False:
            raise ValueError("label conflict audit rows must be audit-only")
        if float(row.get("training_weight", 0.0)) != 0.0:
            raise ValueError("label conflict audit rows must use zero manifest weight")
        if row.get("pre_audit_split") not in {"train", "calibration", "eval"}:
            raise ValueError("label conflict audit rows require pre-audit split")
        if row.get("audit_policy_version") != "augmented_v2_label_conflict_audit_v1":
            raise ValueError("label conflict audit policy version mismatch")
    if is_core_conflict_audit:
        if row.get("core_text_label_conflict_direct") not in {True, False}:
            raise ValueError("core text conflict audit rows require direct-conflict boolean")
        _positive_int(row.get("core_text_label_conflict_hash_count"), "core_text_label_conflict_hash_count")
        _required_hash(row.get("core_text_label_conflict_group_hash"), "core_text_label_conflict_group_hash")
    if is_active_inherited_conflict_audit:
        _required_hash(row.get("active_inherited_conflict_group_hash"), "active_inherited_conflict_group_hash")
    if tier == WEAK_SAFE_EVIDENCE_TIER and row.get("eligible_for_eval") is not False:
        raise ValueError("weak parent safe rows must not be eval eligible")
    if schema_version == CHUNK_CLASSIFIER_SCHEMA_VERSION_V2:
        if tier == WEAK_SAFE_EVIDENCE_TIER:
            raise ValueError("legacy weak parent safe rows cannot enter v2 manifest")
        route = str(row.get("private_cache_route") or "")
        expected_route = _expected_private_cache_route(row, schema_version=schema_version, source_universe=source_universe)
        if expected_route is not None and route != expected_route:
            raise ValueError("chunk classifier row private_cache_route mismatch")
        if not str(row.get("parent_label_source") or ""):
            raise ValueError("chunk classifier v2 row requires parent_label_source")
        if not str(row.get("token_count_bucket") or ""):
            raise ValueError("chunk classifier v2 row requires token_count_bucket")
        if tier == INHERITED_PARENT_SAFE_EVIDENCE_TIER:
            if row.get("classifier_label") != "safe":
                raise ValueError("inherited parent safe rows must be safe")
            if not is_label_conflict_audit and row.get("fit_eligible") is not True:
                raise ValueError("inherited parent safe rows must be fit eligible")
            if source_universe == FINAL_35K_CHUNK_CLASSIFIER_V2_SOURCE_UNIVERSE and row.get("eligible_for_eval") is not True:
                raise ValueError("inherited parent safe rows must be fit/eval eligible")
            if not is_label_conflict_audit and float(row.get("training_weight", 0.0)) != 1.0:
                raise ValueError("inherited parent safe rows must use full training weight")
        if tier == ACTIVE_LEARNING_EVIDENCE_TIER:
            if source_universe != AUGMENTED_V2_SOURCE_UNIVERSE:
                raise ValueError("active-learning rows require augmented source_universe")
            if row.get("source_locator_status") != "private_refetch_required":
                raise ValueError("active-learning rows must mark private refetch as source locator owner")
            if row.get("public_locator_fields") != "active_learning_schedule_index_placeholder":
                raise ValueError("active-learning rows must not expose source row-group locators")
            schedule_index = _nonnegative_int(row.get("active_learning_schedule_index"), "active_learning_schedule_index")
            if int(row.get("row_group_index")) != schedule_index or int(row.get("row_group_local_row_index")) != schedule_index:
                raise ValueError("active-learning public locator placeholder mismatch")
            final_label = str(row.get("active_learning_final_label") or "")
            if final_label not in {"safe", "reject_or_borderline"}:
                raise ValueError("active-learning final label is unsupported")
            if row.get("classifier_label") != ("safe" if final_label == "safe" else "reject"):
                raise ValueError("active-learning classifier label mapping mismatch")
            _required_hash(row.get("teacher_input_text_hash_sha256"), "teacher_input_text_hash_sha256")
            _required_hash(row.get("candidate_schedule_row_hash"), "candidate_schedule_row_hash")
            vote_hashes = list(row.get("selected_vote_row_hashes") or [])
            if len(vote_hashes) != 3 or len(set(str(value) for value in vote_hashes)) != 3:
                raise ValueError("active-learning rows require three selected vote hashes")
            for vote_hash in vote_hashes:
                _required_hash(vote_hash, "selected_vote_row_hash")
            if int(row.get("parsed_vote_count", -1)) != 3 or int(row.get("required_vote_count", -1)) != 3:
                raise ValueError("active-learning rows require exactly three parsed votes")
            if row.get("eligible_for_eval") is not False or row.get("split") == "eval":
                raise ValueError("active-learning rows must not be eval eligible")
            if not is_label_conflict_audit and float(row.get("training_weight", 0.0)) != 1.0:
                raise ValueError("active-learning rows must use full manifest weight")
    if row.get("eligible_for_eval") is True and row.get("split") == "audit":
        raise ValueError("audit rows must not be eval eligible")
    expected_hash = stable_json_hash({key: value for key, value in row.items() if key != "row_hash"})
    if row.get("row_hash") != expected_hash:
        raise ValueError("chunk classifier row hash mismatch")


def _reject_duplicate_candidates(candidates: list[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in seen:
            raise ValueError("duplicate candidate_id in chunk classifier candidates")
        seen.add(candidate_id)


def _reject_forbidden_input_keys(row: Mapping[str, Any]) -> None:
    found = sorted(FORBIDDEN_INPUT_KEYS & set(row))
    if found:
        raise ValueError(f"chunk classifier input contains raw/private keys: {found}")


def _require_false_flags(row: Mapping[str, Any], fields: Iterable[str], context: str) -> None:
    for field in fields:
        if field in row and row[field] is not False:
            raise ValueError(f"{context} must keep {field}=false")


def _required_hash(value: Any, field_name: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return candidate


def _optional_hash(value: Any, field_name: str) -> str | None:
    if value in (None, ""):
        return None
    return _required_hash(value, field_name)


def _nonnegative_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return result


def _positive_int(value: Any, field_name: str) -> int:
    result = _nonnegative_int(value, field_name)
    if result < 1:
        raise ValueError(f"{field_name} must be positive")
    return result


def _required_label(value: Any, field_name: str) -> str:
    label = str(value or "")
    if label not in ALLOWED_LABELS:
        raise ValueError(f"{field_name} must be safe or reject")
    return label


def _optional_label(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return _required_label(value, "canonical_final_label")


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _parse_source_artifact_hashes(items: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--source-artifact-hash must be key=sha256")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("--source-artifact-hash key cannot be empty")
        parsed[key] = _required_hash(value, f"source_artifact_hashes.{key}")
    return parsed


def _load_active_learning_batch_spec(item: str) -> dict[str, Any]:
    parts = str(item).split(":", 4)
    if len(parts) != 5:
        raise ValueError("--active-learning-batch must be batch_id:schedule_json:labels_jsonl:votes_jsonl:core_hash_map_json")
    batch_id, schedule_json, labels_jsonl, votes_jsonl, core_hash_map_json = parts
    if not batch_id:
        raise ValueError("active-learning batch_id cannot be empty")
    hash_payload = json.loads(Path(core_hash_map_json).read_text(encoding="utf-8"))
    if not isinstance(hash_payload, Mapping):
        raise ValueError("active-learning core hash map must be a JSON object")
    core_hashes = hash_payload.get("core_text_hashes_by_candidate_id", hash_payload)
    if not isinstance(core_hashes, Mapping):
        raise ValueError("active-learning core hash map missing core_text_hashes_by_candidate_id")
    parent_hashes = hash_payload.get("parent_text_hashes_by_candidate_id", {})
    if parent_hashes is None:
        parent_hashes = {}
    if not isinstance(parent_hashes, Mapping):
        raise ValueError("active-learning parent hash map must be a JSON object")
    return {
        "batch_id": batch_id,
        "schedule": json.loads(Path(schedule_json).read_text(encoding="utf-8")),
        "labels": _read_jsonl(Path(labels_jsonl)),
        "votes": _read_jsonl(Path(votes_jsonl)),
        "core_text_hashes_by_candidate_id": dict(core_hashes),
        "parent_text_hashes_by_candidate_id": dict(parent_hashes),
    }


def _complete_stage_dirs_from_root(stage_root: Path, stage_glob: str) -> list[Path]:
    return [
        path for path in sorted(stage_root.glob(stage_glob))
        if (path / "chunk_relabel_manifest.json").exists()
        and (path / "chunk_labels_live.jsonl").exists()
    ]


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {str(value): str(value) for value in values}

    def find(self, value: str) -> str:
        value = str(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def _load_stage_candidates_and_labels(stage_dirs: Iterable[str | Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    stage_hashes: list[str] = []
    seen_stage_paths: set[str] = set()
    for stage_dir in sorted((Path(path) for path in stage_dirs), key=lambda item: str(item)):
        stage_key = str(stage_dir.resolve(strict=False))
        if stage_key in seen_stage_paths:
            raise ValueError("duplicate chunk relabel stage directory")
        seen_stage_paths.add(stage_key)
        manifest_path = stage_dir / "chunk_relabel_manifest.json"
        labels_path = stage_dir / "chunk_labels_live.jsonl"
        if not manifest_path.exists():
            raise ValueError(f"missing chunk relabel manifest: {manifest_path}")
        if not labels_path.exists():
            raise ValueError(f"missing chunk label rows: {labels_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require_false_flags(
            manifest,
            ("raw_text_serialized", "raw_url_serialized", "token_serialized"),
            "chunk relabel stage manifest",
        )
        manifest_hash = _required_hash(manifest.get("manifest_hash"), "manifest_hash")
        stage_hashes.append(manifest_hash)
        stage_candidates = list(manifest.get("candidates") or [])
        labels = _read_jsonl(labels_path)
        if len(stage_candidates) != len(labels):
            raise ValueError("chunk relabel stage candidate/label count mismatch")
        label_candidate_ids = {str(row.get("candidate_id") or "") for row in labels}
        for candidate in stage_candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            if candidate_id not in label_candidate_ids:
                raise ValueError("chunk relabel stage candidate missing final label")
            candidate_copy = dict(candidate)
            candidate_copy.setdefault("source_chunk_manifest_hash", manifest_hash)
            candidate_copy.setdefault("source_metadata_hash", manifest.get("source_metadata_hash"))
            candidates.append(candidate_copy)
        for row in labels:
            if str(row.get("source_chunk_manifest_hash") or "") != manifest_hash:
                raise ValueError("chunk label row source_chunk_manifest_hash mismatch")
            label_rows.append(dict(row))
    if not candidates:
        raise ValueError("at least one chunk relabel stage directory is required")
    return candidates, label_rows, stage_hashes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
