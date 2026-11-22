"""Classifier filtering contracts that hand off to clean-run and pretraining stages."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.data.decision_schema import stable_json_hash


CHUNK_DECISION_KIND = "twc_chunk_classifier_decision"
CLEAN_RUN_HANDOFF_KIND = "twc_classifier_clean_run_handoff_manifest"
PRIME_CLASSIFIER_RUN_PLAN_KIND = "twc_classifier_prime_run_plan"
FILTERING_HANDOFF_KIND = "twc_classifier_filtering_to_pretraining_handoff"
ALLOWED_CLASSIFIER_DECISIONS = {"safe", "reject", "quarantine", "error"}
PUBLIC_FALSE_FLAGS = (
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "accepted_manifest_created",
    "accepted_manifest",
    "tokenizer_shards_created",
    "training_shards_created",
    "base_pretraining_started",
    "pretraining_started",
)


def build_chunk_classifier_decision(
    *,
    candidate_id: str,
    source_chunk_manifest_hash: str,
    model_scores: Mapping[str, float],
    thresholds: Mapping[str, float],
    uncertainty_margin: float = 0.0,
) -> dict[str, Any]:
    """Build one no-raw chunk decision row from configured model scores."""

    candidate_id = _required_hash(candidate_id, "candidate_id")
    source_chunk_manifest_hash = _required_hash(source_chunk_manifest_hash, "source_chunk_manifest_hash")
    if not thresholds:
        raise ValueError("classifier decision requires at least one model threshold")
    unknown_models = sorted(set(model_scores) - set(thresholds))
    if unknown_models:
        raise ValueError(f"unknown model score keys: {unknown_models}")
    margin = _uncertainty_margin(uncertainty_margin)
    scores: dict[str, float] = {}
    threshold_payload: dict[str, float] = {}
    missing: list[str] = []
    invalid: list[str] = []
    rejecting: list[str] = []
    uncertain: list[str] = []
    for model_id, threshold_value in sorted(thresholds.items()):
        threshold = _finite_score01(threshold_value, f"threshold.{model_id}")
        threshold_payload[model_id] = threshold
        if model_id not in model_scores:
            missing.append(f"missing_model_score:{model_id}")
            continue
        score = _runtime_score01(model_scores[model_id])
        if score is None:
            invalid.append(f"invalid_model_score:{model_id}")
            continue
        scores[model_id] = score
        if score >= threshold:
            rejecting.append(model_id)
        elif margin > 0 and threshold - score <= margin:
            uncertain.append(f"near_threshold:{model_id}")
    if invalid or missing or uncertain:
        decision = "quarantine"
    elif rejecting:
        decision = "reject"
    else:
        decision = "safe"
    payload = {
        "artifact_kind": CHUNK_DECISION_KIND,
        "schema_version": 1,
        "candidate_id": candidate_id,
        "source_chunk_manifest_hash": source_chunk_manifest_hash,
        "classifier_decision": decision,
        "model_scores": scores,
        "thresholds": threshold_payload,
        "rejecting_models": rejecting,
        "quarantine_reasons": missing + invalid + uncertain,
        "classifier_can_approve_corpus": False,
        "pending_required_gates": ["clean_run_gap_validation", "post_filter_downstream_gates"] if decision == "safe" else [],
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "accepted_manifest_created": False,
        "training_shards_created": False,
        "tokenizer_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
    }
    payload["classifier_decision_hash"] = stable_json_hash(payload)
    validate_public_no_raw_payload(payload)
    return payload


def build_clean_run_handoff_manifest(
    *,
    source_document_id_hash: str,
    segmentation_receipt_hash: str,
    blocks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a clean-run handoff that never bridges classifier gaps."""

    source_document_id_hash = _required_hash(source_document_id_hash, "source_document_id_hash")
    segmentation_receipt_hash = _required_hash(segmentation_receipt_hash, "segmentation_receipt_hash")
    normalized_blocks = [_normalized_block(block, source_document_id_hash=source_document_id_hash) for block in blocks]
    normalized_blocks.sort(key=lambda block: block["block_index"])
    _validate_block_sequence(normalized_blocks)

    clean_runs: list[dict[str, Any]] = []
    gap_events: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for block in normalized_blocks:
        if block["decision"] == "safe":
            current.append(block)
            continue
        if current:
            clean_runs.append(_clean_run_payload(source_document_id_hash, len(clean_runs), current, ends_before_gap=True))
            current = []
        gap_events.append(_gap_event_payload(source_document_id_hash, block))
    if current:
        clean_runs.append(_clean_run_payload(source_document_id_hash, len(clean_runs), current, ends_before_gap=False))

    payload = {
        "artifact_kind": CLEAN_RUN_HANDOFF_KIND,
        "schema_version": 1,
        "source_document_id_hash": source_document_id_hash,
        "segmentation_receipt_hash": segmentation_receipt_hash,
        "block_count": len(normalized_blocks),
        "clean_run_count": len(clean_runs),
        "gap_event_count": len(gap_events),
        "decision_counts": _sorted_counter(Counter(block["decision"] for block in normalized_blocks)),
        "clean_runs": clean_runs,
        "gap_events": gap_events,
        "accepted_manifest_created": False,
        "ready_for_tokenizer_and_packer": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "tokenizer_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
    }
    payload["clean_run_manifest_hash"] = stable_json_hash(payload)
    validate_clean_run_handoff_manifest(payload)
    return payload


def validate_clean_run_handoff_manifest(manifest: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(manifest)
    if manifest.get("artifact_kind") != CLEAN_RUN_HANDOFF_KIND:
        raise ValueError("unsupported clean-run handoff kind")
    _require_false_flags(manifest, PUBLIC_FALSE_FLAGS + ("ready_for_tokenizer_and_packer",), "clean-run handoff")
    source_document_id_hash = _required_hash(manifest.get("source_document_id_hash"), "source_document_id_hash")
    block_count = _nonnegative_int(manifest.get("block_count"), "block_count")
    clean_runs = list(manifest.get("clean_runs") or [])
    gap_events = list(manifest.get("gap_events") or [])
    if int(manifest.get("clean_run_count", -1)) != len(clean_runs):
        raise ValueError("clean_run_count mismatch")
    if int(manifest.get("gap_event_count", -1)) != len(gap_events):
        raise ValueError("gap_event_count mismatch")
    gap_indexes: set[int] = set()
    for gap in gap_events:
        gap_index = _validate_gap_event_payload(gap, source_document_id_hash=source_document_id_hash)
        if gap_index in gap_indexes:
            raise ValueError("clean-run handoff contains duplicate gap block_index")
        gap_indexes.add(gap_index)
    run_indexes: set[int] = set()
    run_ids: set[int] = set()
    for run in clean_runs:
        run_id, block_indexes = _validate_clean_run_payload(
            run,
            source_document_id_hash=source_document_id_hash,
            gap_indexes=gap_indexes,
        )
        if run_id in run_ids:
            raise ValueError("clean-run handoff contains duplicate clean_run_index")
        run_ids.add(run_id)
        for index in block_indexes:
            if index in run_indexes:
                raise ValueError("clean-run handoff contains duplicate clean-run block_index")
            run_indexes.add(index)
    if run_indexes & gap_indexes:
        raise ValueError("clean run and gap indexes overlap")
    if run_indexes | gap_indexes != set(range(block_count)):
        raise ValueError("clean-run handoff block coverage mismatch")
    expected_decisions = Counter({"safe": len(run_indexes)})
    expected_decisions.update(str(gap["classifier_decision"]) for gap in gap_events)
    if dict(manifest.get("decision_counts") or {}) != _sorted_counter(expected_decisions):
        raise ValueError("clean-run handoff decision_counts mismatch")
    expected_hash = stable_json_hash({key: value for key, value in manifest.items() if key != "clean_run_manifest_hash"})
    if manifest.get("clean_run_manifest_hash") != expected_hash:
        raise ValueError("clean-run handoff hash mismatch")


def build_classifier_prime_run_plan(
    *,
    run_id: str,
    classifier_dataset_manifest_hash: str,
    classifier_model_manifest_hash: str,
    filtering_config_hash: str,
    bootstrap_plan: Mapping[str, Any],
    local_smoke_report: Mapping[str, Any] | None,
    prime_resource_preflight_hash: str,
    real_entrypoint_command: Sequence[str],
) -> dict[str, Any]:
    """Build the no-paid Prime execution readiness plan for classifier filtering."""

    command = [str(part) for part in real_entrypoint_command]
    if not command:
        raise ValueError("Prime classifier run plan requires a real entrypoint command")
    command_hash = stable_json_hash(command)
    blockers: list[str] = []
    if local_smoke_report is None:
        blockers.append("local_smoke_report_missing")
    else:
        validate_public_no_raw_payload(local_smoke_report)
        if local_smoke_report.get("status") != "passed":
            blockers.append("local_smoke_not_passed")
        if local_smoke_report.get("entrypoint_command_hash") != command_hash:
            blockers.append("local_smoke_entrypoint_command_hash_mismatch")
    validate_public_no_raw_payload(bootstrap_plan)
    payload = {
        "artifact_kind": PRIME_CLASSIFIER_RUN_PLAN_KIND,
        "schema_version": 1,
        "run_id_hash": stable_json_hash({"run_id": str(run_id)}),
        "status": "blocked" if blockers else "ready_for_user_paid_prime_approval",
        "blocking_reasons": sorted(blockers),
        "classifier_dataset_manifest_hash": _required_hash(classifier_dataset_manifest_hash, "classifier_dataset_manifest_hash"),
        "classifier_model_manifest_hash": _required_hash(classifier_model_manifest_hash, "classifier_model_manifest_hash"),
        "filtering_config_hash": _required_hash(filtering_config_hash, "filtering_config_hash"),
        "prime_resource_preflight_hash": _required_hash(prime_resource_preflight_hash, "prime_resource_preflight_hash"),
        "real_entrypoint_command_hash": command_hash,
        "bootstrap_plan_hash": stable_json_hash(bootstrap_plan),
        "local_smoke_report_hash": None if local_smoke_report is None else stable_json_hash(local_smoke_report),
        "local_mac_validation_required": True,
        "gpu_canary_required_before_scale": True,
        "medium_tranche_review_required_before_full_run": True,
        "pod_creation_authorized": False,
        "disk_creation_authorized": False,
        "gpu_rental_performed": False,
        "paid_action_performed": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["prime_run_plan_hash"] = stable_json_hash(payload)
    validate_classifier_prime_run_plan(payload)
    return payload


def validate_classifier_prime_run_plan(plan: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(plan)
    if plan.get("artifact_kind") != PRIME_CLASSIFIER_RUN_PLAN_KIND:
        raise ValueError("unsupported Prime classifier run plan kind")
    if plan.get("status") not in {"blocked", "ready_for_user_paid_prime_approval"}:
        raise ValueError("unsupported Prime classifier run plan status")
    _require_false_flags(
        plan,
        (
            "pod_creation_authorized",
            "disk_creation_authorized",
            "gpu_rental_performed",
            "paid_action_performed",
            "accepted_manifest_created",
            "tokenizer_shards_created",
            "training_shards_created",
            "base_pretraining_started",
            "pretraining_started",
            "raw_text_serialized",
            "raw_url_serialized",
            "token_serialized",
        ),
        "Prime classifier run plan",
    )
    if plan.get("status") == "ready_for_user_paid_prime_approval" and plan.get("blocking_reasons") != []:
        raise ValueError("ready Prime classifier run plan must not have blockers")
    expected_hash = stable_json_hash({key: value for key, value in plan.items() if key != "prime_run_plan_hash"})
    if plan.get("prime_run_plan_hash") != expected_hash:
        raise ValueError("Prime classifier run plan hash mismatch")


def build_classifier_filtering_handoff_manifest(
    *,
    classifier_dataset_manifest_hash: str,
    classifier_model_manifest_hash: str,
    classifier_threshold_manifest_hash: str,
    filtering_config_hash: str,
    clean_run_manifest_hash: str,
    no_raw_validation_report_hash: str,
    accepted_estimated_token_count_lower_bound: int,
    notes: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the minimal classifier/filtering handoff to tokenizer planning."""

    if int(accepted_estimated_token_count_lower_bound) < 1:
        raise ValueError("accepted_estimated_token_count_lower_bound must be positive")
    validate_public_no_raw_payload(notes)
    payload = {
        "artifact_kind": FILTERING_HANDOFF_KIND,
        "schema_version": 1,
        "status": "ready_for_tokenizer_planning",
        "classifier_dataset_manifest_hash": _required_hash(classifier_dataset_manifest_hash, "classifier_dataset_manifest_hash"),
        "classifier_model_manifest_hash": _required_hash(classifier_model_manifest_hash, "classifier_model_manifest_hash"),
        "classifier_threshold_manifest_hash": _required_hash(classifier_threshold_manifest_hash, "classifier_threshold_manifest_hash"),
        "filtering_config_hash": _required_hash(filtering_config_hash, "filtering_config_hash"),
        "clean_run_manifest_hash": _required_hash(clean_run_manifest_hash, "clean_run_manifest_hash"),
        "no_raw_validation_report_hash": _required_hash(no_raw_validation_report_hash, "no_raw_validation_report_hash"),
        "accepted_estimated_token_count_lower_bound": int(accepted_estimated_token_count_lower_bound),
        "notes_hash": stable_json_hash(notes),
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["handoff_manifest_hash"] = stable_json_hash(payload)
    validate_classifier_filtering_handoff_manifest(payload)
    return payload


def validate_classifier_filtering_handoff_manifest(manifest: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(manifest)
    if manifest.get("artifact_kind") != FILTERING_HANDOFF_KIND:
        raise ValueError("unsupported classifier filtering handoff kind")
    if manifest.get("status") != "ready_for_tokenizer_planning":
        raise ValueError("classifier filtering handoff status mismatch")
    _require_false_flags(manifest, PUBLIC_FALSE_FLAGS, "classifier filtering handoff")
    for field in (
        "classifier_dataset_manifest_hash",
        "classifier_model_manifest_hash",
        "classifier_threshold_manifest_hash",
        "filtering_config_hash",
        "clean_run_manifest_hash",
        "no_raw_validation_report_hash",
    ):
        _required_hash(manifest.get(field), field)
    if int(manifest.get("accepted_estimated_token_count_lower_bound", 0)) < 1:
        raise ValueError("classifier filtering handoff token count must be positive")
    expected_hash = stable_json_hash({key: value for key, value in manifest.items() if key != "handoff_manifest_hash"})
    if manifest.get("handoff_manifest_hash") != expected_hash:
        raise ValueError("classifier filtering handoff hash mismatch")


def _normalized_block(block: Mapping[str, Any], *, source_document_id_hash: str) -> dict[str, Any]:
    validate_public_no_raw_payload(block)
    _require_false_flags(block, ("raw_text_serialized", "raw_url_serialized", "token_serialized"), "clean-run handoff block")
    if _required_hash(block.get("source_document_id_hash"), "source_document_id_hash") != source_document_id_hash:
        raise ValueError("clean-run block source_document_id_hash mismatch")
    decision = str(block.get("decision") or "")
    if decision not in ALLOWED_CLASSIFIER_DECISIONS:
        raise ValueError("unsupported classifier block decision")
    return {
        "source_document_id_hash": source_document_id_hash,
        "natural_block_id_hash": _required_hash(block.get("natural_block_id_hash"), "natural_block_id_hash"),
        "block_index": _nonnegative_int(block.get("block_index"), "block_index"),
        "decision": decision,
        "classifier_decision_hash": _required_hash(block.get("classifier_decision_hash"), "classifier_decision_hash"),
        "proxy_token_count": _nonnegative_int(block.get("proxy_token_count", 0), "proxy_token_count"),
    }


def _validate_block_sequence(blocks: Sequence[Mapping[str, Any]]) -> None:
    indexes = [int(block["block_index"]) for block in blocks]
    if len(indexes) != len(set(indexes)):
        raise ValueError("clean-run handoff blocks contain duplicate block_index")
    if indexes != list(range(len(indexes))):
        raise ValueError("clean-run handoff block indexes must be contiguous from zero")


def _clean_run_payload(
    source_document_id_hash: str,
    clean_run_index: int,
    blocks: Sequence[Mapping[str, Any]],
    *,
    ends_before_gap: bool,
) -> dict[str, Any]:
    block_indexes = [int(block["block_index"]) for block in blocks]
    payload = {
        "clean_run_index": clean_run_index,
        "source_document_id_hash": source_document_id_hash,
        "block_indexes": block_indexes,
        "block_id_hashes": [str(block["natural_block_id_hash"]) for block in blocks],
        "proxy_token_count": sum(int(block["proxy_token_count"]) for block in blocks),
        "starts_after_gap": block_indexes[0] > 0,
        "ends_before_gap": ends_before_gap,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["clean_run_hash"] = stable_json_hash(payload)
    return payload


def _gap_event_payload(source_document_id_hash: str, block: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "source_document_id_hash": source_document_id_hash,
        "block_index": int(block["block_index"]),
        "natural_block_id_hash": str(block["natural_block_id_hash"]),
        "classifier_decision": str(block["decision"]),
        "classifier_decision_hash": str(block["classifier_decision_hash"]),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["gap_event_hash"] = stable_json_hash(payload)
    return payload


def _finite_score01(value: Any, field_name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(score):
        raise ValueError(f"{field_name} must be finite")
    if score < 0 or score > 1:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return score


def _runtime_score01(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score < 0 or score > 1:
        return None
    return score


def _uncertainty_margin(value: Any) -> float:
    margin = _finite_score01(value, "uncertainty_margin")
    return margin


def _validate_clean_run_payload(
    run: Mapping[str, Any],
    *,
    source_document_id_hash: str,
    gap_indexes: set[int],
) -> tuple[int, list[int]]:
    _require_false_flags(run, ("raw_text_serialized", "raw_url_serialized", "token_serialized"), "clean run")
    clean_run_index = _nonnegative_int(run.get("clean_run_index"), "clean_run_index")
    if _required_hash(run.get("source_document_id_hash"), "clean run source_document_id_hash") != source_document_id_hash:
        raise ValueError("clean run source_document_id_hash mismatch")
    block_indexes_raw = run.get("block_indexes")
    if not isinstance(block_indexes_raw, list) or not block_indexes_raw:
        raise ValueError("clean run must contain block_indexes")
    block_indexes = [_nonnegative_int(index, "clean run block_index") for index in block_indexes_raw]
    if block_indexes != list(range(block_indexes[0], block_indexes[-1] + 1)):
        raise ValueError("clean run block_indexes must be contiguous")
    for block_id_hash in run.get("block_id_hashes") or []:
        _required_hash(block_id_hash, "clean run block_id_hash")
    if len(run.get("block_id_hashes") or []) != len(block_indexes):
        raise ValueError("clean run block_id_hash count mismatch")
    _nonnegative_int(run.get("proxy_token_count"), "clean run proxy_token_count")
    starts_after_gap = _required_bool(run.get("starts_after_gap"), "clean run starts_after_gap")
    ends_before_gap = _required_bool(run.get("ends_before_gap"), "clean run ends_before_gap")
    if starts_after_gap != (block_indexes[0] > 0 and block_indexes[0] - 1 in gap_indexes):
        raise ValueError("clean run starts_after_gap mismatch")
    if ends_before_gap != (block_indexes[-1] + 1 in gap_indexes):
        raise ValueError("clean run ends_before_gap mismatch")
    _required_hash(run.get("clean_run_hash"), "clean_run_hash")
    expected_hash = stable_json_hash({key: value for key, value in run.items() if key != "clean_run_hash"})
    if run.get("clean_run_hash") != expected_hash:
        raise ValueError("clean_run_hash mismatch")
    return clean_run_index, block_indexes


def _validate_gap_event_payload(gap: Mapping[str, Any], *, source_document_id_hash: str) -> int:
    _require_false_flags(gap, ("raw_text_serialized", "raw_url_serialized", "token_serialized"), "gap event")
    if _required_hash(gap.get("source_document_id_hash"), "gap event source_document_id_hash") != source_document_id_hash:
        raise ValueError("gap event source_document_id_hash mismatch")
    block_index = _nonnegative_int(gap.get("block_index"), "gap event block_index")
    _required_hash(gap.get("natural_block_id_hash"), "gap event natural_block_id_hash")
    decision = str(gap.get("classifier_decision") or "")
    if decision not in {"reject", "quarantine", "error"}:
        raise ValueError("gap event classifier_decision must be reject, quarantine, or error")
    _required_hash(gap.get("classifier_decision_hash"), "gap event classifier_decision_hash")
    _required_hash(gap.get("gap_event_hash"), "gap_event_hash")
    expected_hash = stable_json_hash({key: value for key, value in gap.items() if key != "gap_event_hash"})
    if gap.get("gap_event_hash") != expected_hash:
        raise ValueError("gap_event_hash mismatch")
    return block_index


def _required_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _required_hash(value: Any, field_name: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return candidate


def _nonnegative_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return result


def _require_false_flags(row: Mapping[str, Any], fields: Sequence[str], context: str) -> None:
    for field in fields:
        if field in row and row[field] is not False:
            raise ValueError(f"{context} must keep {field}=false")


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}
