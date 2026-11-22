"""Teacher-candidate distribution, mining plans, and no-raw-text manifests."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import yaml

from twc.data.decision_schema import stable_json_hash


DISTRIBUTION_SCHEMA_VERSION = 1
CANDIDATE_SCHEMA_VERSION = 1
CANDIDATE_MANIFEST_SCHEMA_VERSION = 1
NON_BORDERLINE_READINESS_SCHEMA_VERSION = 1
NON_BORDERLINE_POOL_SCHEMA_VERSION = 1
REPRESENTATIVE_ALGORITHM_VERSION = "representative_metadata_sampler_v1"
TARGETED_MINER_PLAN_VERSION = "targeted_bucket_miner_plan_v1"
DECISION_LOG_TARGETED_CANDIDATE_VERSION = "decision_log_targeted_candidate_bridge_v1"
ACTIVE_LEARNING_PLAN_VERSION = "active_learning_borderline_interface_v1"
ACTIVE_LEARNING_BOF_STRATEGY = "deepseek_v4_pro_bof_disagreement"
MANIFEST_ALGORITHM_VERSION = "teacher_candidate_manifest_v1"
NON_BORDERLINE_READINESS_VERSION = "non_borderline_candidate_pool_readiness_v1"
NON_BORDERLINE_POOL_ARTIFACT_VERSION = "non_borderline_candidate_pool_construction_v1"
RANDOM_BACKFILL_ARTIFACT_VERSION = "non_borderline_random_backfill_diagnostic_v1"
RANDOM_NON_BORDERLINE_BACKFILL_BUCKET = "random_non_borderline_backfill"
RANDOM_BACKFILL_ALLOWED_SCREENING_STATUSES = {
    "passed_no_deterministic_hard_reject",
}

EXPECTED_TOTAL = 35_000
ACTIVE_LEARNING_BORDERLINE_BUCKET = "active_learning_borderline"
EXPECTED_SUBFLOOR_TOTALS = {
    "subtle_implicit_risk": 5_000,
    "safe_stem_control": 3_000,
    "false_friend_hard_negatives": 2_300,
}
REQUIRED_BUCKETS = {
    "representative_coverage",
    "deterministic_high_risk",
    "subtle_implicit_risk",
    "source_wrapper_channel_risk",
    "safe_stem_control",
    "safe_non_stem_boundary_survivors",
    "false_friend_hard_negatives",
    "long_document_window_cases",
    "active_learning_borderline",
    "coverage_repair_reserve",
}
REQUIRED_IDENTITY_FIELDS = (
    "candidate_id",
    "dataset_repo_id",
    "dataset_config",
    "dataset_split",
    "source_metadata_hash",
    "source_schedule_hash",
    "schedule_entry_hash",
    "global_schedule_index",
    "schedule_pass_index",
    "source_file_id",
    "parquet_identity",
    "row_group_index",
    "row_group_local_row_index",
    "deterministic_row_locator",
    "physical_source_row_key_hash",
    "bucket_id",
    "subbucket_id",
    "selection_reason_id",
    "selection_seed",
    "label_unit",
)
RAW_TEXT_KEYS = {
    "text",
    "raw_text",
    "document_text",
    "content",
    "raw_content",
    "normalized_text",
    "snippet",
    "raw_context",
    "raw_context_snippet",
    "matched_text",
    "tokens",
    "token_ids",
}
RAW_URL_KEYS = {
    "url",
    "raw_url",
    "source_url",
    "document_url",
    "query",
    "query_string",
    "raw_query",
    "parquet_url",
    "parquet_url_or_path",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash(payload: Any) -> str:
    return stable_json_hash(payload)


def _canonical_hash(payload: Any, *exclude_keys: str) -> str:
    excluded = set(exclude_keys)
    if isinstance(payload, dict):
        payload = {key: value for key, value in payload.items() if key not in excluded}
    return _hash(payload)


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return parsed


def _looks_like_hash(value: Any, *, length: int | None = None) -> bool:
    if not isinstance(value, str):
        return False
    if length is not None and len(value) != length:
        return False
    return all(char in "0123456789abcdef" for char in value.casefold())


def _reject_public_leaks(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).casefold()
            if lowered in RAW_TEXT_KEYS:
                raise ValueError(f"raw text field is not allowed in candidate artifacts: {path}.{key}")
            if lowered in RAW_URL_KEYS:
                raise ValueError(f"raw URL field is not allowed in candidate artifacts: {path}.{key}")
            _reject_public_leaks(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_public_leaks(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.casefold()
        if lowered.startswith(("http://", "https://")):
            raise ValueError(f"raw URL value is not allowed in candidate artifacts: {path}")
        if re.search(r"sk-[A-Za-z0-9_-]{20,}", value) or re.search(r"hf_[A-Za-z0-9]{20,}", value):
            raise ValueError(f"secret-like value is not allowed in candidate artifacts: {path}")


def load_distribution_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_distribution_config(config)
    return config


def validate_distribution_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("distribution config must be a mapping")
    if config.get("version") != DISTRIBUTION_SCHEMA_VERSION:
        raise ValueError("unsupported teacher candidate distribution config version")
    buckets = config.get("buckets")
    if not isinstance(buckets, dict):
        raise ValueError("distribution config requires buckets")
    missing_buckets = REQUIRED_BUCKETS - set(buckets)
    if missing_buckets:
        raise ValueError(f"distribution config missing buckets: {sorted(missing_buckets)}")
    unknown_buckets = set(buckets) - REQUIRED_BUCKETS
    if unknown_buckets:
        raise ValueError(f"distribution config has unknown buckets: {sorted(unknown_buckets)}")

    quotas = {}
    for bucket_id, bucket in buckets.items():
        quota = _require_int(bucket.get("quota"), f"bucket {bucket_id} quota", minimum=0)
        quotas[bucket_id] = quota
        subfloors = bucket.get("subfloors", {})
        if subfloors is not None and not isinstance(subfloors, dict):
            raise ValueError(f"bucket {bucket_id} subfloors must be a mapping")
        for subbucket_id, subquota in (subfloors or {}).items():
            if not str(subbucket_id):
                raise ValueError(f"bucket {bucket_id} has an empty subbucket id")
            _require_int(subquota, f"bucket {bucket_id} subfloor {subbucket_id}", minimum=0)
    total = sum(quotas.values())
    expected_total = int(config.get("total_candidates", EXPECTED_TOTAL))
    if total != EXPECTED_TOTAL or expected_total != EXPECTED_TOTAL:
        raise ValueError(f"top-level quotas must sum exactly {EXPECTED_TOTAL}")

    subfloor_totals: dict[str, int] = {}
    for bucket_id, expected in EXPECTED_SUBFLOOR_TOTALS.items():
        actual = sum(int(value) for value in buckets[bucket_id].get("subfloors", {}).values())
        subfloor_totals[bucket_id] = actual
        if actual != expected:
            raise ValueError(f"{bucket_id} subfloors must sum exactly {expected}")

    seed_split = config.get("seed_split")
    if not isinstance(seed_split, dict):
        raise ValueError("seed_split is required")
    full_seed = _require_int(seed_split.get("full_seed_set"), "seed_split.full_seed_set", minimum=1)
    provisional = _require_int(seed_split.get("provisional_train_labels"), "seed_split.provisional_train_labels")
    holdout = _require_int(seed_split.get("labeled_holdout"), "seed_split.labeled_holdout")
    if provisional + holdout != full_seed:
        raise ValueError("seed split train labels plus holdout must equal full seed set")
    if seed_split.get("holdout_is_labeled") is not True:
        raise ValueError("seed holdout must be labeled, not an unlabeled pool")
    if seed_split.get("holdout_is_unlabeled_pool") is not False:
        raise ValueError("seed holdout must not be treated as an unlabeled pool")

    label_unit_policy = config.get("label_unit_policy")
    if not isinstance(label_unit_policy, dict):
        raise ValueError("label_unit_policy is required")
    allowed = set(label_unit_policy.get("allowed", []))
    if allowed != {"document", "window_bundle"}:
        raise ValueError("label_unit_policy.allowed must be document and window_bundle")
    if label_unit_policy.get("safe_window_never_proves_document_safe") is not True:
        raise ValueError("safe window labels must never prove a whole document safe")
    if label_unit_policy.get("risky_window_label_rejects_whole_document") is not True:
        raise ValueError("risky window labels must support whole-document rejection")

    active = config.get("active_learning")
    if not isinstance(active, dict):
        raise ValueError("active_learning is required")
    if _require_int(active.get("exploration_buffer_count"), "active_learning.exploration_buffer_count") != 30_000:
        raise ValueError("active learning exploration buffer must be 30000")
    if _require_int(active.get("borderline_selection_count"), "active_learning.borderline_selection_count") != 4_000:
        raise ValueError("active learning borderline selection must be 4000")
    if active.get("strategy") != ACTIVE_LEARNING_BOF_STRATEGY:
        raise ValueError("active learning strategy must be DeepSeek V4 Pro BoF disagreement")
    if active.get("bof_disagreement_required") is not True:
        raise ValueError("active learning requires BoF disagreement evidence")
    if active.get("reviewed_bof_report_required") is not True:
        raise ValueError("active learning requires a reviewed BoF report")
    if active.get("requires_seed_labels") is not False or active.get("requires_provisional_classifiers") is not False:
        raise ValueError("active learning must not require seed labels or provisional classifiers")
    if active.get("classifier_disagreement_required") is not False:
        raise ValueError("classifier disagreement is dormant for active-learning borderline selection")

    _reject_public_leaks(config)
    return {
        "config_hash": _hash(config),
        "total_quota": total,
        "bucket_quotas": quotas,
        "subfloor_totals": subfloor_totals,
        "seed_split": deepcopy(seed_split),
        "label_unit_policy": deepcopy(label_unit_policy),
    }


def validate_candidate_identity(
    candidate: dict[str, Any],
    config: dict[str, Any],
    *,
    extra_bucket_ids: set[str] | None = None,
) -> None:
    validate_distribution_config(config)
    if not isinstance(candidate, dict):
        raise ValueError("candidate must be a mapping")
    _reject_public_leaks(candidate)
    if candidate.get("candidate_schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("unsupported candidate schema version")
    for field in REQUIRED_IDENTITY_FIELDS:
        if field not in candidate or candidate[field] in (None, ""):
            raise ValueError(f"candidate is missing required identity field: {field}")
    dataset = config.get("dataset", {})
    if candidate["dataset_repo_id"] != dataset.get("repo_id"):
        raise ValueError("candidate dataset_repo_id does not match distribution config")
    if candidate["dataset_config"] != dataset.get("config"):
        raise ValueError("candidate dataset_config does not match distribution config")
    if candidate["dataset_split"] != dataset.get("split"):
        raise ValueError("candidate dataset_split does not match distribution config")
    allowed_extra = extra_bucket_ids or set()
    if candidate["bucket_id"] not in config["buckets"] and candidate["bucket_id"] not in allowed_extra:
        raise ValueError(f"unknown candidate bucket_id: {candidate['bucket_id']}")
    if candidate["bucket_id"] == "coverage_repair_reserve":
        _validate_coverage_repair_candidate(candidate, config)
    label_unit = candidate["label_unit"]
    allowed_units = set(config["label_unit_policy"]["allowed"])
    if label_unit not in allowed_units:
        raise ValueError(f"unsupported label_unit: {label_unit}")
    if label_unit == "window_bundle":
        _validate_window_bundle(candidate, config)
    elif "window_bundle" in candidate:
        raise ValueError("document label_unit must not include a window_bundle")
    for key in ("source_metadata_hash", "source_schedule_hash", "schedule_entry_hash"):
        if not _looks_like_hash(candidate[key], length=64):
            raise ValueError(f"{key} must be a 64-character sha256-like hash")
    if "resolved_repo_sha" in candidate and not _looks_like_hash(candidate["resolved_repo_sha"], length=40):
        raise ValueError("resolved_repo_sha must be a 40-character git SHA")
    for key in ("text_hash_sha256", "normalized_text_hash_sha256", "source_domain_hash"):
        if key in candidate and candidate[key] is not None and not _looks_like_hash(candidate[key], length=64):
            raise ValueError(f"{key} must be a 64-character sha256-like hash")
    if not _looks_like_hash(candidate["physical_source_row_key_hash"], length=64):
        raise ValueError("physical_source_row_key_hash must be a 64-character sha256-like hash")
    if candidate["physical_source_row_key_hash"] != physical_source_row_key_hash(candidate):
        raise ValueError("physical_source_row_key_hash drift")
    if not isinstance(candidate.get("parquet_identity"), dict) or not candidate["parquet_identity"]:
        raise ValueError("candidate parquet_identity must be a non-empty mapping")
    locator = candidate.get("deterministic_row_locator")
    if not isinstance(locator, dict):
        raise ValueError("deterministic_row_locator must be a mapping")
    if int(locator.get("row_group_index", -1)) != int(candidate["row_group_index"]):
        raise ValueError("deterministic_row_locator row_group_index mismatch")
    if int(locator.get("row_group_local_row_index", -1)) != int(candidate["row_group_local_row_index"]):
        raise ValueError("deterministic_row_locator row_group_local_row_index mismatch")
    _require_int(candidate["global_schedule_index"], "global_schedule_index")
    _require_int(candidate["schedule_pass_index"], "schedule_pass_index")
    _require_int(candidate["row_group_index"], "row_group_index")
    _require_int(candidate["row_group_local_row_index"], "row_group_local_row_index")


def _validate_coverage_repair_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> None:
    evidence = candidate.get("selection_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("coverage repair candidate requires selection_evidence")
    repair = evidence.get("coverage_repair")
    if not isinstance(repair, dict):
        raise ValueError("coverage repair candidate requires coverage_repair provenance")
    if repair.get("repair_semantics") != "configured_subfloor_repair_v1":
        raise ValueError("coverage repair candidate has unsupported repair semantics")
    if repair.get("repair_purpose_id") != "unmet_configured_subfloor":
        raise ValueError("coverage repair candidate has unsupported repair purpose")
    original_bucket_id = str(repair.get("original_bucket_id") or "")
    original_subbucket_id = str(repair.get("original_subbucket_id") or "")
    if original_bucket_id not in config["buckets"]:
        raise ValueError("coverage repair candidate has unknown original bucket")
    configured_subfloors = config["buckets"][original_bucket_id].get("subfloors") or {}
    if original_subbucket_id not in configured_subfloors:
        raise ValueError("coverage repair candidate original subbucket is not a configured subfloor")
    if candidate.get("subbucket_id") != f"{original_bucket_id}:{original_subbucket_id}":
        raise ValueError("coverage repair candidate subbucket must preserve original bucket and subbucket")
    original_reason = repair.get("original_selection_reason_id")
    if not isinstance(original_reason, str) or not original_reason:
        raise ValueError("coverage repair candidate requires original selection reason")
    if candidate.get("selection_reason_id") != f"coverage_repair_reserve:{original_reason}":
        raise ValueError("coverage repair candidate selection reason must preserve original reason")


def _validate_window_bundle(candidate: dict[str, Any], config: dict[str, Any]) -> None:
    bundle = candidate.get("window_bundle")
    if not isinstance(bundle, dict):
        raise ValueError("window_bundle label_unit requires window_bundle descriptors")
    if bundle.get("safe_window_labels_document_safe") is not False:
        raise ValueError("safe window labels cannot become whole-document safe")
    windows = bundle.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("window_bundle requires at least one sanitized window descriptor")
    for index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError("window descriptor must be a mapping")
        for field in ("window_id", "window_index", "window_hash_sha256", "offset_descriptor", "label_scope"):
            if field not in window or window[field] in (None, ""):
                raise ValueError(f"window descriptor {index} missing {field}")
        if not _looks_like_hash(window["window_hash_sha256"], length=64):
            raise ValueError("window_hash_sha256 must be a 64-character sha256-like hash")
        if not isinstance(window["offset_descriptor"], dict):
            raise ValueError("window offset_descriptor must be sanitized metadata")
        if window["label_scope"] not in {"risk_window_rejects_document", "local_negative_only"}:
            raise ValueError("unsupported window label_scope")
    if config["label_unit_policy"].get("safe_window_never_proves_document_safe") is not True:
        raise ValueError("safe window policy must be enabled")


def _row_identity(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate.get("dataset_repo_id"),
        candidate.get("dataset_config"),
        candidate.get("dataset_split"),
        candidate.get("resolved_repo_sha"),
        _hash(candidate.get("parquet_identity")),
        candidate.get("source_file_id"),
        int(candidate.get("row_group_index")),
        int(candidate.get("row_group_local_row_index")),
    )


def _domain_hash(candidate: dict[str, Any]) -> str:
    value = candidate.get("source_domain_hash")
    if value:
        return str(value)
    source_file_id = candidate.get("source_file_id")
    if source_file_id:
        return f"missing_domain_source_file:{source_file_id}"
    return "missing"


def _non_borderline_buckets(config: dict[str, Any]) -> list[str]:
    return [bucket_id for bucket_id in config["buckets"] if bucket_id != ACTIVE_LEARNING_BORDERLINE_BUCKET]


def _subfloor_diagnostics(
    bucket_id: str,
    selected_candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    selected_by_subbucket = dict(sorted(Counter(candidate["subbucket_id"] for candidate in selected_candidates).items()))
    subfloors = config["buckets"][bucket_id].get("subfloors") or {}
    shortfalls = {
        subbucket_id: max(0, int(quota) - int(selected_by_subbucket.get(subbucket_id, 0)))
        for subbucket_id, quota in sorted(subfloors.items())
    }
    shortfalls = {key: value for key, value in shortfalls.items() if value}
    return {
        "selected_by_subbucket": selected_by_subbucket,
        "subfloor_shortfalls": shortfalls,
        "subfloors_satisfied": not shortfalls,
    }


def _sanitize_parquet_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict) or not identity:
        raise ValueError("parquet_identity must be a non-empty mapping")
    sanitized = {
        key: value
        for key, value in identity.items()
        if str(key).casefold() not in RAW_URL_KEYS
        and not (isinstance(value, str) and value.casefold().startswith(("http://", "https://")))
    }
    if "identity_hash" not in sanitized:
        sanitized["identity_hash"] = _hash(identity)
    if not sanitized:
        raise ValueError("parquet_identity did not contain sanitized identity fields")
    _reject_public_leaks(sanitized)
    return sanitized


def _with_candidate_hash(candidate: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(candidate)
    updated["candidate_hash"] = _canonical_hash(updated, "candidate_hash")
    return updated


def _parquet_identity_hash(value: dict[str, Any]) -> str:
    existing = value.get("parquet_identity_hash")
    if existing:
        return str(existing)
    parquet_identity = value.get("parquet_identity")
    return _hash(parquet_identity) if parquet_identity else ""


def physical_source_row_key(value: dict[str, Any]) -> tuple[Any, ...]:
    if value.get("row_group_index") is None or value.get("row_group_local_row_index") is None:
        raise ValueError("physical source row identity requires row_group_index and row_group_local_row_index")
    source_file_id = str(value.get("source_file_id") or "")
    if source_file_id:
        return (
            "source_file_id",
            source_file_id,
            int(value.get("row_group_index") or 0),
            int(value.get("row_group_local_row_index") or 0),
        )
    parquet_identity_hash = _parquet_identity_hash(value)
    if not parquet_identity_hash:
        raise ValueError("physical source row identity requires source_file_id or parquet_identity_hash")
    return (
        str(value.get("dataset_repo_id") or ""),
        str(value.get("dataset_config") or value.get("config") or ""),
        str(value.get("dataset_split") or value.get("split") or ""),
        str(value.get("resolved_repo_sha") or ""),
        "parquet_identity_hash",
        parquet_identity_hash,
        int(value.get("row_group_index") or 0),
        int(value.get("row_group_local_row_index") or 0),
    )


def physical_source_row_key_hash(value: dict[str, Any]) -> str:
    return _hash(physical_source_row_key(value))


def _row_config(row: dict[str, Any]) -> str | None:
    value = row.get("dataset_config", row.get("config"))
    return str(value) if value is not None else None


def _row_risk_terms(row: dict[str, Any]) -> set[str]:
    terms = set(str(tag).casefold() for tag in row.get("risk_tags", []) if str(tag))
    terms.update(str(reason).casefold() for reason in row.get("rejection_reasons", []) if str(reason))
    for hit in row.get("hits", []) if isinstance(row.get("hits"), list) else []:
        if not isinstance(hit, dict):
            continue
        for field in ("category", "rule_id", "reason", "severity", "scan_scope"):
            value = hit.get(field)
            if value:
                terms.add(str(value).casefold())
    return terms


def _decision_row_bucket_and_subbucket(row: dict[str, Any]) -> tuple[str, str, str] | None:
    status = str(row.get("decision_status", "")).casefold()
    terms = _row_risk_terms(row)
    joined = " ".join(sorted(terms))
    window_summary = row.get("window_summary") if isinstance(row.get("window_summary"), dict) else {}
    failed_windows = int(window_summary.get("failed_window_count") or 0)
    window_count = int(window_summary.get("window_count") or 0)
    if status == "rejected" and failed_windows == 1 and window_count >= 5:
        return ("long_document_window_cases", "one_bad_window", "window_bundle_from_rejected_decision_row")
    if status == "rejected" and (
        "phase2a_source_wrappers" in terms
        or any("wrapper" in term for term in terms)
        or any(term in joined for term in ("faq", "forum", "support", "knowledge_base", "help_center", "review"))
    ):
        return ("source_wrapper_channel_risk", "source_wrapper_signal", "source_wrapper_decision_row")
    if status == "rejected" and any(
        term in joined
        for term in (
            "soft_stance",
            "stance",
            "agentive",
            "seem",
            "appears",
            "may",
            "might",
            "uncertainty",
            "thought",
            "believ",
            "feel",
            "dream",
            "memory",
            "awareness",
            "attention",
            "preference",
            "anthropomorphism",
        )
    ):
        if any(term in joined for term in ("soft_stance", "stance", "seem", "may", "might", "appears")):
            subbucket = "soft_stance_agentive"
        elif any(term in joined for term in ("ai", "assistant", "chatbot", "anthropomorphism")):
            subbucket = "ai_anthropomorphism"
        elif any(term in joined for term in ("awareness", "attention")):
            subbucket = "attention_awareness_terms"
        elif any(term in joined for term in ("preference", "desire", "value")):
            subbucket = "value_preference_language"
        else:
            subbucket = "mental_state_ambiguity"
        return ("subtle_implicit_risk", subbucket, f"decision_log_{subbucket}")
    if status == "rejected":
        return ("deterministic_high_risk", "deterministic_reject", "rejected_decision_row")
    return None


def _window_bundle_from_decision_row(row: dict[str, Any]) -> dict[str, Any]:
    decision_hash = str(row.get("decision_row_hash") or _hash(row))
    windows = []
    seen_indexes: set[int] = set()
    for hit in row.get("hits", []) if isinstance(row.get("hits"), list) else []:
        if not isinstance(hit, dict):
            continue
        index = hit.get("window_index")
        if index is None:
            continue
        try:
            window_index = int(index)
        except (TypeError, ValueError):
            continue
        if window_index in seen_indexes:
            continue
        seen_indexes.add(window_index)
        windows.append(
            {
                "window_id": _hash(["decision_window", decision_hash, window_index]),
                "window_index": window_index,
                "window_hash_sha256": _hash(["decision_window_hash", decision_hash, window_index]),
                "offset_descriptor": {
                    "unit": "normalized_window_metadata",
                    "window_start_normalized": hit.get("window_start_normalized"),
                    "window_end_normalized": hit.get("window_end_normalized"),
                    "window_word_start": hit.get("window_word_start"),
                    "window_word_end": hit.get("window_word_end"),
                },
                "label_scope": "risk_window_rejects_document",
            }
        )
    if not windows:
        windows.append(
            {
                "window_id": _hash(["decision_window", decision_hash, "unknown"]),
                "window_index": 0,
                "window_hash_sha256": _hash(["decision_window_hash", decision_hash, "unknown"]),
                "offset_descriptor": {"unit": "window_index_unavailable"},
                "label_scope": "risk_window_rejects_document",
            }
        )
    summary = row.get("window_summary") if isinstance(row.get("window_summary"), dict) else {}
    return {
        "bundle_id": _hash(["decision_window_bundle", decision_hash]),
        "document_window_count": int(summary.get("window_count") or len(windows)),
        "windows": windows,
        "safe_window_labels_document_safe": False,
    }


def _candidate_from_decision_row(
    row: dict[str, Any],
    *,
    config: dict[str, Any],
    bucket_id: str,
    subbucket_id: str,
    selection_reason_id: str,
) -> dict[str, Any]:
    dataset = config["dataset"]
    if row.get("dataset_repo_id") != dataset["repo_id"]:
        raise ValueError("decision row dataset_repo_id does not match distribution config")
    if _row_config(row) != dataset["config"]:
        raise ValueError("decision row dataset config does not match distribution config")
    if row.get("split") != dataset["split"]:
        raise ValueError("decision row split does not match distribution config")
    if row.get("raw_text_serialized") is not False or row.get("token_serialized") is not False:
        raise ValueError("decision row must be no-raw before candidate conversion")

    parquet_identity = _sanitize_parquet_identity(row.get("parquet_identity"))
    source_domain = row.get("source_domain")
    decision_hash = str(row.get("decision_row_hash") or _hash(row))
    label_unit = "window_bundle" if bucket_id == "long_document_window_cases" else "document"
    candidate = {
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": _hash([DECISION_LOG_TARGETED_CANDIDATE_VERSION, bucket_id, decision_hash]),
        "dataset_repo_id": dataset["repo_id"],
        "dataset_config": dataset["config"],
        "dataset_split": dataset["split"],
        "resolved_repo_sha": row.get("resolved_repo_sha"),
        "source_metadata_hash": row["source_metadata_hash"],
        "source_schedule_hash": row["source_schedule_hash"],
        "schedule_entry_hash": row["schedule_entry_hash"],
        "global_schedule_index": int(row["global_schedule_index"]),
        "schedule_pass_index": int(row["schedule_pass_index"]),
        "source_file_id": row["source_file_id"],
        "parquet_identity": parquet_identity,
        "row_group_index": int(row["row_group_index"]),
        "row_group_local_row_index": int(row["row_group_local_row_index"]),
        "deterministic_row_locator": {
            "mode": "decision_log_row_group_local_offset",
            "row_group_index": int(row["row_group_index"]),
            "row_group_local_row_index": int(row["row_group_local_row_index"]),
            "decision_row_hash": decision_hash,
        },
        "physical_source_row_key_hash": physical_source_row_key_hash(
            {
                "source_file_id": row["source_file_id"],
                "parquet_identity": parquet_identity,
                "row_group_index": int(row["row_group_index"]),
                "row_group_local_row_index": int(row["row_group_local_row_index"]),
            }
        ),
        "bucket_id": bucket_id,
        "subbucket_id": subbucket_id,
        "selection_reason_id": selection_reason_id,
        "selection_seed": int(config.get("selection_seed", 20260603)),
        "label_unit": label_unit,
        "text_hash_sha256": row.get("raw_text_sha256"),
        "normalized_text_hash_sha256": row.get("normalized_text_sha256"),
        "source_domain_hash": _hash(["source_domain", source_domain]) if source_domain else None,
        "selection_evidence": {
            "algorithm_version": DECISION_LOG_TARGETED_CANDIDATE_VERSION,
            "decision_row_hash": decision_hash,
            "decision_status": row.get("decision_status"),
            "hit_count": int(row.get("hit_count") or 0),
            "risk_tag_hashes": sorted(_hash(["risk_tag", tag]) for tag in _row_risk_terms(row)),
            "window_summary": deepcopy(row.get("window_summary") if isinstance(row.get("window_summary"), dict) else {}),
        },
    }
    if label_unit == "window_bundle":
        candidate["window_bundle"] = _window_bundle_from_decision_row(row)
    return _with_candidate_hash(candidate)


def build_targeted_candidates_from_decision_rows(
    rows_or_path: list[dict[str, Any]] | tuple[dict[str, Any], ...] | str | Path,
    *,
    config: dict[str, Any],
    max_rows: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Convert bounded no-raw decision rows into conservative targeted bucket inputs."""

    validate_distribution_config(config)
    if isinstance(rows_or_path, (str, Path)):
        iterator = (json.loads(line) for line in Path(rows_or_path).read_text(encoding="utf-8").splitlines() if line.strip())
    else:
        iterator = iter(rows_or_path)
    pools: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(iterator):
        if max_rows is not None and index >= max_rows:
            break
        _reject_public_leaks(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "hits",
                    "source_domain",
                    "source_doc_id",
                    "source_metadata",
                    "source_metadata_presence",
                    "foundation_decision",
                }
            }
        )
        bucket = _decision_row_bucket_and_subbucket(row)
        if bucket is None:
            continue
        bucket_id, subbucket_id, reason = bucket
        if bucket_id in {"safe_stem_control", "safe_non_stem_boundary_survivors", ACTIVE_LEARNING_BORDERLINE_BUCKET}:
            continue
        candidate = _candidate_from_decision_row(
            row,
            config=config,
            bucket_id=bucket_id,
            subbucket_id=subbucket_id,
            selection_reason_id=reason,
        )
        pools.setdefault(bucket_id, []).append(candidate)
    return {bucket_id: candidates for bucket_id, candidates in sorted(pools.items()) if candidates}


def build_candidate_manifest(
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    source_metadata_hash: str,
    source_tree_hash: str | None = None,
    write_report: bool = False,
    shortfall_reasons: dict[str, list[str]] | None = None,
    caps_reached: dict[str, bool] | None = None,
) -> dict[str, Any]:
    validation = validate_distribution_config(config)
    if write_report:
        raise ValueError("build_candidate_manifest is pure; use write_candidate_manifest for report output")
    normalized = [_with_candidate_hash(candidate) for candidate in candidates]
    bucket_counts = dict(sorted(Counter(candidate["bucket_id"] for candidate in normalized).items()))
    domain_counts = Counter(_domain_hash(candidate) for candidate in normalized)
    duplicate_clusters = Counter(candidate.get("text_hash_sha256") for candidate in normalized if candidate.get("text_hash_sha256"))
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "distribution_version": config["distribution_version"],
        "selection_algorithm_version": MANIFEST_ALGORITHM_VERSION,
        "selection_seed": int(config.get("selection_seed", 20260603)),
        "config_hash": validation["config_hash"],
        "source_metadata_hash": source_metadata_hash,
        "source_tree_hash": source_tree_hash,
        "candidate_count": len(normalized),
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "bucket_counts": bucket_counts,
        "bucket_closure_diagnostics": {
            "requested": validation["bucket_quotas"],
            "selected": bucket_counts,
            "shortfalls": {
                bucket_id: max(0, quota - bucket_counts.get(bucket_id, 0))
                for bucket_id, quota in validation["bucket_quotas"].items()
            },
            "shortfall_reasons": shortfall_reasons or {},
            "caps_reached": caps_reached or {},
        },
        "domain_cap_diagnostics": {
            "max_per_domain": int(config["miner_caps"]["per_domain_max"]),
            "max_observed_per_domain_hash": max(domain_counts.values(), default=0),
            "exceeded": [
                domain_hash
                for domain_hash, count in sorted(domain_counts.items())
                if count > int(config["miner_caps"]["per_domain_max"])
            ],
        },
        "duplicate_cluster_diagnostics": {
            "policy": config["miner_caps"]["duplicate_cluster_policy"],
            "duplicate_text_hash_cluster_count": len([count for count in duplicate_clusters.values() if count > 1]),
        },
        "candidates": normalized,
    }
    manifest["manifest_hash"] = _canonical_hash(manifest, "generated_at_utc", "manifest_hash")
    validate_candidate_manifest(manifest, config)
    return manifest


def validate_candidate_manifest(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    validate_distribution_config(config)
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported candidate manifest schema")
    for flag in ("raw_text_serialized", "token_serialized", "raw_url_serialized", "accepted_manifest", "labels_are_corpus_approval"):
        expected = False
        if flag == "labels_are_corpus_approval":
            expected = False
        if manifest.get(flag) is not expected:
            raise ValueError(f"candidate manifest must set {flag}=false")
    _reject_public_leaks(manifest)
    ids: set[str] = set()
    row_ids: set[tuple[Any, ...]] = set()
    domain_counts: Counter[str] = Counter()
    for candidate in manifest.get("candidates", []):
        validate_candidate_identity(candidate, config)
        candidate_id = candidate["candidate_id"]
        if candidate_id in ids:
            raise ValueError("duplicate candidate id")
        ids.add(candidate_id)
        row_id = _row_identity(candidate)
        if row_id in row_ids:
            raise ValueError("duplicate row identity across candidate buckets")
        row_ids.add(row_id)
        expected_hash = _canonical_hash(candidate, "candidate_hash")
        if candidate.get("candidate_hash") != expected_hash:
            raise ValueError("candidate hash mismatch")
        domain_counts[_domain_hash(candidate)] += 1
    max_per_domain = int(config["miner_caps"]["per_domain_max"])
    if any(count > max_per_domain for count in domain_counts.values()):
        raise ValueError("domain concentration exceeds configured cap")
    stable = _canonical_hash(manifest, "generated_at_utc", "manifest_hash")
    if manifest.get("manifest_hash") != stable:
        raise ValueError("candidate manifest hash mismatch")


def write_candidate_manifest(
    manifest: dict[str, Any],
    path: str | Path,
    *,
    config: dict[str, Any],
    allow_reports: bool = False,
) -> None:
    if not allow_reports:
        raise ValueError("candidate manifest writes require allow_reports=True")
    validate_candidate_manifest(manifest, config)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)


def read_candidate_manifest(path: str | Path, *, config: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_candidate_manifest(manifest, config)
    return manifest


def _file_by_id(metadata_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {file["file_id"]: file for file in metadata_manifest.get("files", [])}


def _row_group_size_band(group: dict[str, Any]) -> str:
    rows = int(group.get("num_rows") or 0)
    total_bytes = int(group.get("total_byte_size") or 0)
    if rows <= 0:
        return "empty"
    bytes_per_row = total_bytes / rows if total_bytes else 0
    if rows < 128:
        row_band = "small_rg"
    elif rows < 1024:
        row_band = "medium_rg"
    else:
        row_band = "large_rg"
    if bytes_per_row < 2048:
        byte_band = "compact"
    elif bytes_per_row < 16384:
        byte_band = "medium_bytes"
    else:
        byte_band = "large_bytes"
    return f"{row_band}:{byte_band}"


def _planned_representative_candidate(
    *,
    metadata_manifest: dict[str, Any],
    file_entry: dict[str, Any],
    group: dict[str, Any],
    row_offset: int,
    index: int,
    seed: int,
    source_schedule_hash: str,
    schedule_pass_index: int,
) -> dict[str, Any]:
    schedule_entry_hash = _hash(
        [
            "representative_entry",
            source_schedule_hash,
            file_entry["file_id"],
            int(group["row_group_index"]),
            int(row_offset),
            index,
        ]
    )
    candidate = {
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": _hash(["representative_candidate", schedule_entry_hash]),
        "dataset_repo_id": metadata_manifest["dataset"]["repo_id"],
        "dataset_config": metadata_manifest["dataset"]["config"],
        "dataset_split": metadata_manifest["dataset"]["split"],
        "resolved_repo_sha": metadata_manifest["dataset"].get("resolved_repo_sha"),
        "source_metadata_hash": metadata_manifest.get("source_metadata_hash") or metadata_manifest["manifest_hash"],
        "source_schedule_hash": source_schedule_hash,
        "schedule_entry_hash": schedule_entry_hash,
        "global_schedule_index": index,
        "schedule_pass_index": schedule_pass_index,
        "source_file_id": file_entry["file_id"],
        "parquet_identity": _sanitize_parquet_identity(file_entry["parquet_identity"]),
        "row_group_index": int(group["row_group_index"]),
        "row_group_local_row_index": int(row_offset),
        "deterministic_row_locator": {
            "mode": "row_group_local_offset",
            "row_group_index": int(group["row_group_index"]),
            "row_group_local_row_index": int(row_offset),
        },
        "physical_source_row_key_hash": physical_source_row_key_hash(
            {
                "source_file_id": file_entry["file_id"],
                "parquet_identity": _sanitize_parquet_identity(file_entry["parquet_identity"]),
                "row_group_index": int(group["row_group_index"]),
                "row_group_local_row_index": int(row_offset),
            }
        ),
        "bucket_id": "representative_coverage",
        "subbucket_id": _row_group_size_band(group),
        "selection_reason_id": "metadata_stratified_representative_row",
        "selection_seed": seed,
        "label_unit": "document",
        "planned_schedule_identity": {
            "algorithm_version": REPRESENTATIVE_ALGORITHM_VERSION,
            "source_metadata_hash": metadata_manifest.get("source_metadata_hash") or metadata_manifest["manifest_hash"],
            "entry_hash": schedule_entry_hash,
        },
    }
    return _with_candidate_hash(candidate)


def build_representative_sample_plan(
    metadata_manifest: dict[str, Any],
    *,
    config: dict[str, Any],
    quota: int | None = None,
    seed: int | None = None,
    replacement_reserve_size: int | None = None,
) -> dict[str, Any]:
    validate_distribution_config(config)
    quota = int(quota if quota is not None else config["representative_sampler"]["quota"])
    reserve_size = int(
        replacement_reserve_size
        if replacement_reserve_size is not None
        else config["representative_sampler"].get("replacement_reserve_size", 500)
    )
    seed = int(seed if seed is not None else config.get("selection_seed", 20260603))
    if quota < 1:
        raise ValueError("representative quota must be positive")
    if reserve_size < 0:
        raise ValueError("replacement reserve size must be non-negative")
    file_map = _file_by_id(metadata_manifest)
    group_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for footer in metadata_manifest.get("parquet_footers", []):
        file_entry = file_map.get(footer.get("file_id"))
        if not file_entry:
            continue
        for group in footer.get("row_groups", []):
            if int(group.get("num_rows") or 0) > 0:
                group_records.append((file_entry, group))
    if not group_records:
        raise ValueError("source metadata has no non-empty row groups")

    total_needed = quota + reserve_size
    source_schedule_hash = _hash(
        [
            REPRESENTATIVE_ALGORITHM_VERSION,
            metadata_manifest.get("source_metadata_hash") or metadata_manifest["manifest_hash"],
            seed,
            quota,
            reserve_size,
        ]
    )
    total_row_positions = sum(int(group["num_rows"]) for _, group in group_records)
    if total_needed > total_row_positions:
        raise ValueError("representative metadata does not contain enough unique row offsets for quota plus reserve")

    selected: list[dict[str, Any]] = []
    used_row_offsets: set[tuple[str, int, int]] = set()
    pass_index = 0
    while len(selected) < total_needed:
        pass_selected_count = len(selected)
        keyed_groups = []
        for file_entry, group in group_records:
            key = _hash(
                [
                    seed,
                    "representative_group_order",
                    pass_index,
                    file_entry["file_id"],
                    int(group["row_group_index"]),
                    _row_group_size_band(group),
                ]
            )
            keyed_groups.append((key, file_entry, group))
        keyed_groups.sort(key=lambda item: item[0])
        for _, file_entry, group in keyed_groups:
            if len(selected) >= total_needed:
                break
            row_count = int(group["num_rows"])
            row_offset = None
            for attempt_index in range(row_count):
                offset_seed = _hash(
                    [
                        seed,
                        "representative_offset",
                        pass_index,
                        attempt_index,
                        file_entry["file_id"],
                        int(group["row_group_index"]),
                        row_count,
                    ]
                )
                candidate_offset = int(offset_seed[:16], 16) % row_count
                row_key = (file_entry["file_id"], int(group["row_group_index"]), int(candidate_offset))
                if row_key not in used_row_offsets:
                    row_offset = candidate_offset
                    used_row_offsets.add(row_key)
                    break
            if row_offset is None:
                continue
            selected.append(
                _planned_representative_candidate(
                    metadata_manifest=metadata_manifest,
                    file_entry=file_entry,
                    group=group,
                    row_offset=row_offset,
                    index=len(selected),
                    seed=seed,
                    source_schedule_hash=source_schedule_hash,
                    schedule_pass_index=pass_index,
                )
            )
        if len(selected) == pass_selected_count:
            raise ValueError("representative sampler could not find more unique row offsets")
        pass_index += 1

    primary = selected[:quota]
    reserve = selected[quota:]
    diagnostics = {
        "source_file_count": len({candidate["source_file_id"] for candidate in primary}),
        "row_group_count": len({(candidate["source_file_id"], candidate["row_group_index"]) for candidate in primary}),
        "row_group_size_bands": sorted({candidate["subbucket_id"] for candidate in primary}),
        "contiguous_prefix_traversal": False,
        "replacement_reserve_size": len(reserve),
        "output_exact_primary_quota": len(primary) == quota,
    }
    plan = {
        "schema_version": 1,
        "plan_version": REPRESENTATIVE_ALGORITHM_VERSION,
        "bucket_id": "representative_coverage",
        "seed": seed,
        "source_metadata_hash": metadata_manifest.get("source_metadata_hash") or metadata_manifest["manifest_hash"],
        "source_schedule_hash": source_schedule_hash,
        "primary_quota": quota,
        "replacement_reserve_size": reserve_size,
        "coverage_diagnostics": diagnostics,
        "primary_candidates": primary,
        "replacement_reserve": reserve,
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
    }
    plan["plan_hash"] = _canonical_hash(plan, "plan_hash")
    return plan


def build_targeted_bucket_plan(
    config: dict[str, Any],
    *,
    available_candidates: dict[str, list[dict[str, Any]]] | None = None,
    requested_buckets: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    validation = validate_distribution_config(config)
    available_candidates = available_candidates or {}
    requested = list(
        requested_buckets
        or [
            bucket
            for bucket in config["buckets"]
            if bucket not in {"representative_coverage", ACTIVE_LEARNING_BORDERLINE_BUCKET}
        ]
    )
    bucket_status: dict[str, dict[str, Any]] = {}
    selected_candidates: dict[str, list[dict[str, Any]]] = {}
    selected_counts: dict[str, int] = {}
    selected_ids: set[str] = set()
    selected_rows: set[tuple[Any, ...]] = set()
    selected_text_hashes: set[str] = set()
    domain_counts: Counter[str] = Counter()
    source_file_counts: Counter[str] = Counter()
    per_domain_max = int(config["miner_caps"]["per_domain_max"])
    per_source_file_max = int(config["miner_caps"]["per_source_file_max"])

    def try_select_candidate(
        candidate: dict[str, Any],
        *,
        bucket_id: str,
        selected_for_bucket: list[dict[str, Any]],
        selected_ids_for_bucket: set[str],
        selected_subbucket_counts: Counter[str],
        silent_existing_ids: set[str],
        rejection_counts: Counter[str],
    ) -> bool:
        _reject_public_leaks(candidate)
        if candidate.get("bucket_id") != bucket_id:
            rejection_counts["wrong_bucket_rejections"] += 1
            return False
        validate_candidate_identity(candidate, config)
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id in silent_existing_ids:
            return False
        if candidate_id in selected_ids:
            rejection_counts["duplicate_id_rejections"] += 1
            return False
        row_id = _row_identity(candidate)
        if row_id in selected_rows:
            rejection_counts["duplicate_row_rejections"] += 1
            return False
        text_hash = candidate.get("text_hash_sha256")
        if text_hash and text_hash in selected_text_hashes:
            rejection_counts["duplicate_cluster_rejections"] += 1
            return False
        domain_hash = _domain_hash(candidate)
        if domain_counts[domain_hash] >= per_domain_max:
            rejection_counts["domain_cap_rejections"] += 1
            return False
        source_file_id = str(candidate["source_file_id"])
        if source_file_counts[source_file_id] >= per_source_file_max:
            rejection_counts["source_file_cap_rejections"] += 1
            return False

        selected = _with_candidate_hash(candidate)
        selected_for_bucket.append(selected)
        selected_ids_for_bucket.add(candidate_id)
        selected_subbucket_counts[str(candidate["subbucket_id"])] += 1
        selected_ids.add(candidate_id)
        selected_rows.add(row_id)
        if text_hash:
            selected_text_hashes.add(str(text_hash))
        domain_counts[domain_hash] += 1
        source_file_counts[source_file_id] += 1
        return True

    for bucket_id in requested:
        if bucket_id not in config["buckets"]:
            raise ValueError(f"unknown requested bucket: {bucket_id}")
        if bucket_id == ACTIVE_LEARNING_BORDERLINE_BUCKET:
            raise ValueError("active-learning borderline bucket requires reviewed BoF disagreement evidence")
        quota = validation["bucket_quotas"][bucket_id]
        pool = available_candidates.get(bucket_id, [])
        selected_for_bucket: list[dict[str, Any]] = []
        selected_ids_for_bucket: set[str] = set()
        selected_subbucket_counts: Counter[str] = Counter()
        rejection_counts: Counter[str] = Counter()
        for subbucket_id, subfloor_quota in (config["buckets"][bucket_id].get("subfloors") or {}).items():
            for candidate in pool:
                if selected_subbucket_counts[subbucket_id] >= int(subfloor_quota):
                    break
                if candidate.get("subbucket_id") != subbucket_id:
                    continue
                try_select_candidate(
                    candidate,
                    bucket_id=bucket_id,
                    selected_for_bucket=selected_for_bucket,
                    selected_ids_for_bucket=selected_ids_for_bucket,
                    selected_subbucket_counts=selected_subbucket_counts,
                    silent_existing_ids=set(),
                    rejection_counts=rejection_counts,
                )
        subfloor_pass_selected_ids = set(selected_ids_for_bucket)
        for candidate in pool:
            if len(selected_for_bucket) >= quota:
                break
            try_select_candidate(
                candidate,
                bucket_id=bucket_id,
                selected_for_bucket=selected_for_bucket,
                selected_ids_for_bucket=selected_ids_for_bucket,
                selected_subbucket_counts=selected_subbucket_counts,
                silent_existing_ids=subfloor_pass_selected_ids,
                rejection_counts=rejection_counts,
            )
        selected_count = len(selected_for_bucket)
        shortfall = max(0, quota - selected_count)
        subfloor_status = _subfloor_diagnostics(bucket_id, selected_for_bucket, config)
        reasons = []
        if shortfall:
            reasons.append("bounded_pool_exhausted")
        if not subfloor_status["subfloors_satisfied"]:
            reasons.append("subfloor_shortfall")
        if rejection_counts["domain_cap_rejections"]:
            reasons.append("per_domain_cap_reached")
        if rejection_counts["source_file_cap_rejections"]:
            reasons.append("per_source_file_cap_reached")
        if rejection_counts["duplicate_row_rejections"] or rejection_counts["duplicate_cluster_rejections"]:
            reasons.append("duplicate_or_near_duplicate_rejected")
        bucket_status[bucket_id] = {
            "quota": quota,
            "available_count": len(pool),
            "selected_count": selected_count,
            "shortfall": shortfall,
            "shortfall_reasons": reasons,
            "selection_mode": "stop_when_quota_filled_bounded_pool",
            "rejected_count": sum(rejection_counts.values()),
            "duplicate_id_rejections": rejection_counts["duplicate_id_rejections"],
            "duplicate_row_rejections": rejection_counts["duplicate_row_rejections"],
            "duplicate_cluster_rejections": rejection_counts["duplicate_cluster_rejections"],
            "wrong_bucket_rejections": rejection_counts["wrong_bucket_rejections"],
            "domain_cap_rejections": rejection_counts["domain_cap_rejections"],
            "source_file_cap_rejections": rejection_counts["source_file_cap_rejections"],
            "selected_by_subbucket": subfloor_status["selected_by_subbucket"],
            "subfloor_shortfalls": subfloor_status["subfloor_shortfalls"],
            "subfloors_satisfied": subfloor_status["subfloors_satisfied"],
        }
        selected_candidates[bucket_id] = selected_for_bucket
        selected_counts[bucket_id] = selected_count
    plan = {
        "schema_version": 1,
        "plan_version": TARGETED_MINER_PLAN_VERSION,
        "requested_buckets": requested,
        "bucket_status": bucket_status,
        "selected_counts": selected_counts,
        "selected_candidates": selected_candidates,
        "caps": deepcopy(config["miner_caps"]),
        "cap_diagnostics": {
            "per_domain_max": per_domain_max,
            "per_source_file_max": per_source_file_max,
            "max_observed_per_domain_hash": max(domain_counts.values(), default=0),
            "max_observed_per_source_file": max(source_file_counts.values(), default=0),
            "selected_domain_hash_count": len(domain_counts),
            "selected_source_file_count": len(source_file_counts),
        },
        "duplicate_cluster_policy": config["miner_caps"]["duplicate_cluster_policy"],
        "fills_from_wrong_bucket": False,
        "miners": {
            bucket_id: {
                "status": "skeleton_planned_no_live_broad_text_mining",
                "bounded": True,
                "stop_when_quota_filled": True,
                "text_verification_required_only_for_targeted_rows": True,
            }
            for bucket_id in requested
        },
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
    }
    plan["plan_hash"] = _canonical_hash(plan, "plan_hash")
    return plan


def _validate_targeted_bucket_plan(plan: dict[str, Any], config: dict[str, Any]) -> None:
    validation = validate_distribution_config(config)
    _reject_public_leaks(plan)
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported targeted bucket plan schema")
    if plan.get("plan_version") != TARGETED_MINER_PLAN_VERSION:
        raise ValueError("unsupported targeted bucket plan version")
    if plan.get("fills_from_wrong_bucket") is not False:
        raise ValueError("targeted bucket plan must not fill from wrong buckets")
    requested = plan.get("requested_buckets")
    if not isinstance(requested, list):
        raise ValueError("targeted bucket plan requested_buckets must be a list")
    if ACTIVE_LEARNING_BORDERLINE_BUCKET in requested:
        raise ValueError("targeted bucket plan must not include active-learning borderline")
    bucket_status = plan.get("bucket_status", {})
    selected_candidates = plan.get("selected_candidates", {})
    selected_counts = plan.get("selected_counts", {})
    if not isinstance(bucket_status, dict) or not isinstance(selected_candidates, dict):
        raise ValueError("targeted bucket plan must include bucket status and selected candidates")
    ids: set[str] = set()
    row_ids: set[tuple[Any, ...]] = set()
    domain_counts: Counter[str] = Counter()
    source_file_counts: Counter[str] = Counter()
    for bucket_id in requested:
        if bucket_id not in validation["bucket_quotas"]:
            raise ValueError(f"unknown targeted bucket: {bucket_id}")
        status = bucket_status.get(bucket_id)
        candidates = selected_candidates.get(bucket_id, [])
        if not isinstance(status, dict) or not isinstance(candidates, list):
            raise ValueError(f"targeted bucket status missing {bucket_id}")
        selected_count = _require_int(status.get("selected_count"), f"{bucket_id} selected_count")
        quota = int(validation["bucket_quotas"][bucket_id])
        if status.get("quota") != quota:
            raise ValueError(f"targeted bucket quota drifted for {bucket_id}")
        if selected_count != len(candidates) or selected_counts.get(bucket_id) != selected_count:
            raise ValueError(f"targeted bucket selected count drifted for {bucket_id}")
        if status.get("shortfall") != max(0, quota - selected_count):
            raise ValueError(f"targeted bucket shortfall drifted for {bucket_id}")
        expected_subfloor_status = _subfloor_diagnostics(bucket_id, candidates, config)
        if status.get("selected_by_subbucket") != expected_subfloor_status["selected_by_subbucket"]:
            raise ValueError(f"targeted bucket subbucket counts drifted for {bucket_id}")
        if status.get("subfloor_shortfalls") != expected_subfloor_status["subfloor_shortfalls"]:
            raise ValueError(f"targeted bucket subfloor shortfalls drifted for {bucket_id}")
        if status.get("subfloors_satisfied") is not expected_subfloor_status["subfloors_satisfied"]:
            raise ValueError(f"targeted bucket subfloor status drifted for {bucket_id}")
        for candidate in candidates:
            validate_candidate_identity(candidate, config)
            if candidate.get("candidate_hash") != _canonical_hash(candidate, "candidate_hash"):
                raise ValueError("targeted candidate hash mismatch")
            if candidate["bucket_id"] != bucket_id:
                raise ValueError("targeted candidate bucket drifted")
            if candidate["candidate_id"] in ids:
                raise ValueError("duplicate candidate id in targeted plan")
            ids.add(candidate["candidate_id"])
            row_id = _row_identity(candidate)
            if row_id in row_ids:
                raise ValueError("duplicate row identity in targeted plan")
            row_ids.add(row_id)
            domain_counts[_domain_hash(candidate)] += 1
            source_file_counts[str(candidate["source_file_id"])] += 1
    if any(count > int(config["miner_caps"]["per_domain_max"]) for count in domain_counts.values()):
        raise ValueError("targeted bucket plan exceeds per-domain cap")
    if any(count > int(config["miner_caps"]["per_source_file_max"]) for count in source_file_counts.values()):
        raise ValueError("targeted bucket plan exceeds per-source-file cap")
    stable = _canonical_hash(plan, "plan_hash")
    if plan.get("plan_hash") != stable:
        raise ValueError("targeted bucket plan hash mismatch")


def build_active_learning_plan(
    config: dict[str, Any],
    *,
    candidate_rows: list[dict[str, Any]] | None = None,
    seed_labels_ready: bool = False,
    provisional_classifiers_ready: bool = False,
    bof_disagreement_report_ready: bool = False,
) -> dict[str, Any]:
    validate_distribution_config(config)
    active = config["active_learning"]
    _ = (seed_labels_ready, provisional_classifiers_ready)
    if candidate_rows and not bof_disagreement_report_ready:
        raise ValueError("active-learning borderline candidates require reviewed BoF disagreement evidence")
    plan = {
        "schema_version": 1,
        "plan_version": ACTIVE_LEARNING_PLAN_VERSION,
        "strategy": ACTIVE_LEARNING_BOF_STRATEGY,
        "exploration_buffer_count": int(active["exploration_buffer_count"]),
        "borderline_selection_count": int(active["borderline_selection_count"]),
        "bof_disagreement_required": bool(active["bof_disagreement_required"]),
        "reviewed_bof_report_required": bool(active["reviewed_bof_report_required"]),
        "label_vote_split_priority": str(active["label_vote_split_priority"]),
        "borderline_true_rate_priority": str(active["borderline_true_rate_priority"]),
        "source_bucket_diversity_priority": str(active["source_bucket_diversity_priority"]),
        "classifier_disagreement_required": bool(active["classifier_disagreement_required"]),
        "requires_seed_labels": bool(active["requires_seed_labels"]),
        "requires_provisional_classifiers": bool(active["requires_provisional_classifiers"]),
        "status": (
            "ready_for_borderline_selection"
            if bof_disagreement_report_ready
            else "blocked_until_reviewed_bof_disagreement_report"
        ),
        "candidate_rows": candidate_rows or [],
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
    }
    _reject_public_leaks(plan)
    plan["plan_hash"] = _canonical_hash(plan, "plan_hash")
    return plan


def _validate_non_borderline_readiness_report(report: dict[str, Any], config: dict[str, Any]) -> None:
    validation = validate_distribution_config(config)
    if not isinstance(report, dict):
        raise ValueError("non-borderline readiness report must be a mapping")
    _reject_public_leaks(report)
    if report.get("schema_version") != NON_BORDERLINE_READINESS_SCHEMA_VERSION:
        raise ValueError("unsupported non-borderline readiness schema")
    if report.get("readiness_version") != NON_BORDERLINE_READINESS_VERSION:
        raise ValueError("unsupported non-borderline readiness version")
    if report.get("readiness_scope") != "non_borderline_final_candidate_pool":
        raise ValueError("unexpected non-borderline readiness scope")
    if report.get("distribution_version") != config["distribution_version"]:
        raise ValueError("readiness report distribution version drifted from config")
    if report.get("config_hash") != validation["config_hash"]:
        raise ValueError("readiness report config hash drifted from config")
    for flag in (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "labels_are_corpus_approval",
        "training_shards_created",
        "paid_calls_made",
        "classifier_training_run",
    ):
        if report.get(flag) is not False:
            raise ValueError(f"non-borderline readiness report must set {flag}=false")
    if report.get("fills_from_wrong_bucket") is not False:
        raise ValueError("non-borderline readiness report must not fill from wrong buckets")

    bucket_quotas = validation["bucket_quotas"]
    expected_active_quota = int(bucket_quotas[ACTIVE_LEARNING_BORDERLINE_BUCKET])
    expected_non_borderline_buckets = _non_borderline_buckets(config)
    expected_non_borderline_quota = sum(int(bucket_quotas[bucket_id]) for bucket_id in expected_non_borderline_buckets)
    reconciliation = report.get("quota_reconciliation", {})
    if not isinstance(reconciliation, dict):
        raise ValueError("readiness report quota reconciliation must be a mapping")
    if reconciliation.get("authoritative_total_candidates") != EXPECTED_TOTAL:
        raise ValueError("readiness report total quota drifted from config contract")
    if reconciliation.get("bucket_quota_sum") != validation["total_quota"]:
        raise ValueError("readiness report bucket quota sum drifted from config contract")
    if reconciliation.get("non_borderline_quota") != expected_non_borderline_quota:
        raise ValueError("readiness report non-borderline quota drifted from config contract")
    if reconciliation.get("active_learning_borderline_quota") != expected_active_quota:
        raise ValueError("readiness report active-learning quota drifted from config contract")
    seed_split = validation["seed_split"]
    expected_seed_fields = {
        "seed_label_target": int(seed_split["full_seed_set"]),
        "seed_provisional_train_labels": int(seed_split["provisional_train_labels"]),
        "seed_labeled_holdout": int(seed_split["labeled_holdout"]),
    }
    for field, expected in expected_seed_fields.items():
        if reconciliation.get(field) != expected:
            raise ValueError(f"readiness report {field} drifted from config contract")
    if reconciliation.get("seed_split_is_separate_from_bucket_quotas") is not True:
        raise ValueError("readiness report must keep seed split separate from bucket quotas")
    if reconciliation.get("ambiguous_27k_or_8k_borderline_claim_status") != "not_authoritative_no_quota_change":
        raise ValueError("readiness report must not treat 27k/8k shorthand as authoritative")
    if reconciliation.get("quota_change_required_before_freeze") is not False:
        raise ValueError("readiness report must not require a quota change")

    if report.get("non_borderline_buckets") != expected_non_borderline_buckets:
        raise ValueError("readiness report non-borderline buckets drifted from config contract")
    if report.get("classifier_dependent_buckets") != []:
        raise ValueError("readiness report classifier-dependent buckets drifted from config contract")
    if report.get("preparable_without_seed_labels_or_classifiers") != expected_non_borderline_buckets:
        raise ValueError("readiness report preparable buckets drifted from config contract")

    active_plan = build_active_learning_plan(config)
    active_dependency = report.get("active_learning_dependency", {})
    if not isinstance(active_dependency, dict):
        raise ValueError("readiness report active-learning dependency must be a mapping")
    if active_dependency.get("bucket_id") != ACTIVE_LEARNING_BORDERLINE_BUCKET:
        raise ValueError("readiness report active-learning dependency bucket drifted")
    if active_dependency.get("quota") != expected_active_quota:
        raise ValueError("readiness report active-learning dependency quota drifted")
    for field in (
        "status",
        "strategy",
        "bof_disagreement_required",
        "reviewed_bof_report_required",
        "requires_seed_labels",
        "requires_provisional_classifiers",
        "borderline_selection_count",
        "label_vote_split_priority",
        "borderline_true_rate_priority",
        "source_bucket_diversity_priority",
    ):
        if active_dependency.get(field) != active_plan[field]:
            raise ValueError(f"readiness report active-learning dependency {field} drifted")

    bucket_readiness = report.get("bucket_readiness", {})
    if not isinstance(bucket_readiness, dict):
        raise ValueError("readiness report bucket readiness must be a mapping")
    if set(bucket_readiness) != set(expected_non_borderline_buckets):
        raise ValueError("readiness report bucket readiness drifted from non-borderline buckets")
    shortfall_buckets: list[str] = []
    planned_shortfall_buckets: list[str] = []
    subfloor_shortfall_buckets: list[str] = []
    not_started_buckets: list[str] = []
    for bucket_id in expected_non_borderline_buckets:
        row = bucket_readiness.get(bucket_id)
        if not isinstance(row, dict):
            raise ValueError(f"readiness report bucket readiness missing {bucket_id}")
        quota = _require_int(row.get("quota"), f"bucket readiness {bucket_id} quota")
        selected_count = _require_int(row.get("selected_count"), f"bucket readiness {bucket_id} selected_count")
        shortfall = _require_int(row.get("shortfall"), f"bucket readiness {bucket_id} shortfall")
        if quota != int(bucket_quotas[bucket_id]):
            raise ValueError(f"readiness report bucket quota drifted for {bucket_id}")
        if selected_count > quota:
            raise ValueError(f"readiness report selected count exceeds quota for {bucket_id}")
        if shortfall != max(0, quota - selected_count):
            raise ValueError(f"readiness report shortfall drifted for {bucket_id}")
        evidence_source = row.get("evidence_source")
        if shortfall:
            if row.get("status") != "shortfall_reported":
                raise ValueError(f"readiness report shortfall status drifted for {bucket_id}")
            shortfall_buckets.append(bucket_id)
            if evidence_source == "not_planned_in_this_report":
                not_started_buckets.append(bucket_id)
            else:
                planned_shortfall_buckets.append(bucket_id)
        elif row.get("subfloors_satisfied") is False:
            if row.get("status") != "subfloor_shortfall_reported":
                raise ValueError(f"readiness report subfloor status drifted for {bucket_id}")
            subfloor_shortfall_buckets.append(bucket_id)
        elif row.get("status") != "quota_satisfied_no_raw_candidates_available":
            raise ValueError(f"readiness report satisfied status drifted for {bucket_id}")

    non_borderline_rows_ready = not shortfall_buckets and not subfloor_shortfall_buckets
    if report.get("non_borderline_candidate_rows_ready") is not non_borderline_rows_ready:
        raise ValueError("readiness report non-borderline readiness flag drifted")
    expected_final_manifest_ready = non_borderline_rows_ready and active_plan["status"] == "ready_for_borderline_selection"
    if report.get("final_distribution_candidate_manifest_ready") is not expected_final_manifest_ready:
        raise ValueError("readiness report final manifest readiness flag drifted")
    blockers = report.get("remaining_blockers", {})
    if not isinstance(blockers, dict):
        raise ValueError("readiness report blockers must be a mapping")
    if blockers.get("non_borderline_shortfall_buckets") != shortfall_buckets:
        raise ValueError("readiness report shortfall bucket list drifted")
    if blockers.get("planned_shortfall_buckets") != planned_shortfall_buckets:
        raise ValueError("readiness report planned shortfall bucket list drifted")
    if blockers.get("not_started_buckets") != not_started_buckets:
        raise ValueError("readiness report not-started bucket list drifted")
    if blockers.get("subfloor_shortfall_buckets") != subfloor_shortfall_buckets:
        raise ValueError("readiness report subfloor shortfall bucket list drifted")
    if blockers.get("active_learning_borderline") != active_plan["status"]:
        raise ValueError("readiness report active-learning blocker drifted")
    stable = _canonical_hash(report, "generated_at_utc", "readiness_hash")
    if report.get("readiness_hash") != stable:
        raise ValueError("non-borderline readiness report hash mismatch")


def build_non_borderline_readiness_report(
    config: dict[str, Any],
    *,
    targeted_plan: dict[str, Any] | None = None,
    representative_plan: dict[str, Any] | None = None,
    seed_manifest_summary: dict[str, Any] | None = None,
    prior_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a no-raw readiness summary for final-pool buckets not using classifiers."""

    if prior_report is not None:
        existing = deepcopy(prior_report)
        _validate_non_borderline_readiness_report(existing, config)
        return existing

    validation = validate_distribution_config(config)
    bucket_quotas = validation["bucket_quotas"]
    if ACTIVE_LEARNING_BORDERLINE_BUCKET not in bucket_quotas:
        raise ValueError("active-learning borderline bucket is required")

    active_plan = build_active_learning_plan(config)
    non_borderline_buckets = _non_borderline_buckets(config)
    active_quota = int(bucket_quotas[ACTIVE_LEARNING_BORDERLINE_BUCKET])
    non_borderline_quota = sum(int(bucket_quotas[bucket_id]) for bucket_id in non_borderline_buckets)

    if targeted_plan is not None:
        _validate_targeted_bucket_plan(targeted_plan, config)
    if representative_plan is not None:
        _reject_public_leaks(representative_plan)
        if representative_plan.get("bucket_id") != "representative_coverage":
            raise ValueError("representative plan must target representative_coverage")
    if seed_manifest_summary is not None:
        _reject_public_leaks(seed_manifest_summary)

    targeted_status = (targeted_plan or {}).get("bucket_status", {})
    bucket_readiness: dict[str, dict[str, Any]] = {}
    shortfall_buckets: list[str] = []
    planned_shortfall_buckets: list[str] = []
    subfloor_shortfall_buckets: list[str] = []
    not_started_buckets: list[str] = []
    for bucket_id in non_borderline_buckets:
        quota = int(bucket_quotas[bucket_id])
        if bucket_id == "representative_coverage" and representative_plan is not None:
            selected_count = len(representative_plan.get("primary_candidates", []))
            shortfall = max(0, quota - selected_count)
            source = "representative_metadata_plan"
            blockers = [] if shortfall == 0 else ["source_metadata_plan_shortfall"]
            subfloor_status = {"selected_by_subbucket": {}, "subfloor_shortfalls": {}, "subfloors_satisfied": True}
        elif bucket_id in targeted_status:
            status = targeted_status[bucket_id]
            selected_count = int(status.get("selected_count", 0))
            shortfall = int(status.get("shortfall", max(0, quota - selected_count)))
            source = "targeted_bucket_plan"
            blockers = list(status.get("shortfall_reasons", []))
            subfloor_status = {
                "selected_by_subbucket": deepcopy(status.get("selected_by_subbucket", {})),
                "subfloor_shortfalls": deepcopy(status.get("subfloor_shortfalls", {})),
                "subfloors_satisfied": bool(status.get("subfloors_satisfied", True)),
            }
        else:
            selected_count = 0
            shortfall = quota
            source = "not_planned_in_this_report"
            blockers = ["bounded_no_raw_candidate_mining_not_run"]
            not_started_buckets.append(bucket_id)
            subfloor_status = _subfloor_diagnostics(bucket_id, [], config)
        if shortfall:
            shortfall_buckets.append(bucket_id)
            if source != "not_planned_in_this_report":
                planned_shortfall_buckets.append(bucket_id)
            state = "shortfall_reported"
        elif not subfloor_status["subfloors_satisfied"]:
            subfloor_shortfall_buckets.append(bucket_id)
            state = "subfloor_shortfall_reported"
        else:
            state = "quota_satisfied_no_raw_candidates_available"
        bucket_readiness[bucket_id] = {
            "quota": quota,
            "selected_count": selected_count,
            "shortfall": shortfall,
            "status": state,
            "evidence_source": source,
            "shortfall_reasons": blockers,
            "selected_by_subbucket": subfloor_status["selected_by_subbucket"],
            "subfloor_shortfalls": subfloor_status["subfloor_shortfalls"],
            "subfloors_satisfied": subfloor_status["subfloors_satisfied"],
        }

    non_borderline_rows_ready = all(
        row["shortfall"] == 0 and row["subfloors_satisfied"] for row in bucket_readiness.values()
    )
    final_manifest_ready = non_borderline_rows_ready and active_plan["status"] == "ready_for_borderline_selection"
    seed_split = validation["seed_split"]
    report = {
        "schema_version": NON_BORDERLINE_READINESS_SCHEMA_VERSION,
        "readiness_version": NON_BORDERLINE_READINESS_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "readiness_scope": "non_borderline_final_candidate_pool",
        "distribution_version": config["distribution_version"],
        "config_hash": validation["config_hash"],
        "selection_seed": int(config.get("selection_seed", 20260603)),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "paid_calls_made": False,
        "classifier_training_run": False,
        "quota_reconciliation": {
            "authoritative_total_candidates": EXPECTED_TOTAL,
            "bucket_quota_sum": validation["total_quota"],
            "non_borderline_quota": non_borderline_quota,
            "active_learning_borderline_quota": active_quota,
            "seed_label_target": int(seed_split["full_seed_set"]),
            "seed_provisional_train_labels": int(seed_split["provisional_train_labels"]),
            "seed_labeled_holdout": int(seed_split["labeled_holdout"]),
            "seed_split_is_separate_from_bucket_quotas": True,
            "ambiguous_27k_or_8k_borderline_claim_status": "not_authoritative_no_quota_change",
            "quota_change_required_before_freeze": False,
        },
        "non_borderline_buckets": non_borderline_buckets,
        "classifier_dependent_buckets": [],
        "preparable_without_seed_labels_or_classifiers": non_borderline_buckets,
        "active_learning_dependency": {
            "bucket_id": ACTIVE_LEARNING_BORDERLINE_BUCKET,
            "quota": active_quota,
            "strategy": active_plan["strategy"],
            "status": active_plan["status"],
            "bof_disagreement_required": active_plan["bof_disagreement_required"],
            "reviewed_bof_report_required": active_plan["reviewed_bof_report_required"],
            "requires_seed_labels": active_plan["requires_seed_labels"],
            "requires_provisional_classifiers": active_plan["requires_provisional_classifiers"],
            "borderline_selection_count": active_plan["borderline_selection_count"],
            "label_vote_split_priority": active_plan["label_vote_split_priority"],
            "borderline_true_rate_priority": active_plan["borderline_true_rate_priority"],
            "source_bucket_diversity_priority": active_plan["source_bucket_diversity_priority"],
        },
        "bucket_readiness": bucket_readiness,
        "fills_from_wrong_bucket": bool((targeted_plan or {}).get("fills_from_wrong_bucket", False)),
        "non_borderline_candidate_rows_ready": non_borderline_rows_ready,
        "final_distribution_candidate_manifest_ready": final_manifest_ready,
        "seed_manifest_summary": seed_manifest_summary or None,
        "remaining_blockers": {
            "non_borderline_shortfall_buckets": shortfall_buckets,
            "planned_shortfall_buckets": planned_shortfall_buckets,
            "not_started_buckets": not_started_buckets,
            "subfloor_shortfall_buckets": subfloor_shortfall_buckets,
            "active_learning_borderline": active_plan["status"],
            "live_paid_calls": "not_run_explicit_unlock_required",
            "classifier_training": "not_run",
            "accepted_corpus_manifest": "not_created",
        },
    }
    report["readiness_hash"] = _canonical_hash(report, "generated_at_utc", "readiness_hash")
    _validate_non_borderline_readiness_report(report, config)
    return report


def _representative_pool_summary(representative_plan: dict[str, Any] | None) -> dict[str, Any]:
    if representative_plan is None:
        return {
            "status": "not_planned_in_this_artifact",
            "plan_hash": None,
            "source_metadata_hash": None,
            "source_schedule_hash": None,
            "primary_count": 0,
            "replacement_reserve_count": 0,
            "contiguous_prefix_traversal": None,
            "primary_candidates": [],
            "replacement_reserve": [],
        }
    diagnostics = representative_plan.get("coverage_diagnostics", {})
    return {
        "status": "planned_from_source_footer_metadata",
        "plan_hash": representative_plan["plan_hash"],
        "source_metadata_hash": representative_plan["source_metadata_hash"],
        "source_schedule_hash": representative_plan["source_schedule_hash"],
        "primary_count": len(representative_plan.get("primary_candidates", [])),
        "replacement_reserve_count": len(representative_plan.get("replacement_reserve", [])),
        "contiguous_prefix_traversal": diagnostics.get("contiguous_prefix_traversal"),
        "coverage_diagnostics": deepcopy(diagnostics),
        "primary_candidates": deepcopy(representative_plan.get("primary_candidates", [])),
        "replacement_reserve": deepcopy(representative_plan.get("replacement_reserve", [])),
    }


def _targeted_pool_summary(targeted_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "bounded_pool_selection_scaffold",
        "plan_hash": targeted_plan["plan_hash"],
        "requested_buckets": deepcopy(targeted_plan["requested_buckets"]),
        "bucket_status": deepcopy(targeted_plan["bucket_status"]),
        "selected_counts": deepcopy(targeted_plan["selected_counts"]),
        "selected_candidates": deepcopy(targeted_plan["selected_candidates"]),
        "caps": deepcopy(targeted_plan["caps"]),
        "cap_diagnostics": deepcopy(targeted_plan["cap_diagnostics"]),
        "duplicate_cluster_policy": targeted_plan["duplicate_cluster_policy"],
        "fills_from_wrong_bucket": targeted_plan["fills_from_wrong_bucket"],
    }


def _targeted_selected_counts_for_non_borderline(config: dict[str, Any], targeted_plan: dict[str, Any]) -> dict[str, int]:
    targeted_counts = targeted_plan.get("selected_counts", {})
    return {
        bucket_id: int(targeted_counts.get(bucket_id, 0))
        for bucket_id in _non_borderline_buckets(config)
        if bucket_id != "representative_coverage"
    }


def _validate_random_backfill_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> None:
    _reject_public_leaks(candidate)
    validate_candidate_identity(candidate, config, extra_bucket_ids={RANDOM_NON_BORDERLINE_BACKFILL_BUCKET})
    if candidate.get("bucket_id") != RANDOM_NON_BORDERLINE_BACKFILL_BUCKET:
        raise ValueError("random backfill candidate must use random_non_borderline_backfill bucket")
    evidence = candidate.get("selection_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("random backfill candidate requires selection_evidence")
    if evidence.get("scarce_bucket_credit") is not False:
        raise ValueError("random backfill candidate must set scarce_bucket_credit=false")
    if evidence.get("provenance_kind") != RANDOM_NON_BORDERLINE_BACKFILL_BUCKET:
        raise ValueError("random backfill candidate must preserve random backfill provenance")
    if evidence.get("public_raw_text_serialized") is not False:
        raise ValueError("random backfill candidate must set public_raw_text_serialized=false")
    if evidence.get("public_raw_url_serialized") is not False:
        raise ValueError("random backfill candidate must set public_raw_url_serialized=false")
    screening_status = evidence.get("deterministic_screening_status")
    if screening_status not in RANDOM_BACKFILL_ALLOWED_SCREENING_STATUSES:
        raise ValueError("random backfill candidate requires passed deterministic screening")


def _candidate_dedupe_keys(candidate: dict[str, Any]) -> tuple[str, tuple[Any, ...], str | None, str | None]:
    return (
        str(candidate.get("candidate_id") or ""),
        _row_identity(candidate),
        str(candidate.get("text_hash_sha256")) if candidate.get("text_hash_sha256") else None,
        str(candidate.get("normalized_text_hash_sha256")) if candidate.get("normalized_text_hash_sha256") else None,
    )


def _add_dedupe_keys(
    candidate: dict[str, Any],
    *,
    candidate_ids: set[str],
    row_ids: set[tuple[Any, ...]],
    text_hashes: set[str],
    normalized_text_hashes: set[str],
) -> None:
    candidate_id, row_id, text_hash, normalized_text_hash = _candidate_dedupe_keys(candidate)
    candidate_ids.add(candidate_id)
    row_ids.add(row_id)
    if text_hash:
        text_hashes.add(text_hash)
    if normalized_text_hash:
        normalized_text_hashes.add(normalized_text_hash)


def _backfill_sort_key(candidate: dict[str, Any], *, random_seed: int, source_schedule_hash: str | None) -> tuple[str, str]:
    return (
        _hash(
            [
                RANDOM_BACKFILL_ARTIFACT_VERSION,
                int(random_seed),
                source_schedule_hash or candidate.get("source_schedule_hash"),
                candidate.get("physical_source_row_key_hash"),
                candidate.get("candidate_id"),
            ]
        ),
        str(candidate.get("candidate_id") or ""),
    )


def _validate_non_borderline_random_backfill_artifact(
    artifact: dict[str, Any],
    *,
    config: dict[str, Any],
) -> None:
    validation = validate_distribution_config(config)
    _reject_public_leaks(artifact)
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported random backfill artifact schema")
    if artifact.get("artifact_version") != RANDOM_BACKFILL_ARTIFACT_VERSION:
        raise ValueError("unsupported random backfill artifact version")
    if artifact.get("artifact_scope") != "31k_non_borderline_random_backfill_diagnostic":
        raise ValueError("unexpected random backfill artifact scope")
    if artifact.get("distribution_version") != config["distribution_version"]:
        raise ValueError("random backfill artifact distribution version drifted")
    if artifact.get("config_hash") != validation["config_hash"]:
        raise ValueError("random backfill artifact config hash drifted")
    if artifact.get("diagnostic_only") is not True:
        raise ValueError("random backfill artifact must be diagnostic_only=true")
    for flag in (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "labels_are_corpus_approval",
        "training_shards_created",
        "paid_calls_made",
        "deepseek_calls_made",
        "classifier_training_run",
    ):
        if artifact.get(flag) is not False:
            raise ValueError(f"random backfill artifact must set {flag}=false")

    expected_non_borderline_buckets = _non_borderline_buckets(config)
    if artifact.get("non_borderline_buckets") != expected_non_borderline_buckets:
        raise ValueError("random backfill artifact non-borderline buckets drifted")
    if artifact.get("random_backfill_bucket_id") != RANDOM_NON_BORDERLINE_BACKFILL_BUCKET:
        raise ValueError("random backfill bucket id drifted")
    target_count = sum(int(validation["bucket_quotas"][bucket_id]) for bucket_id in expected_non_borderline_buckets)
    counts = artifact.get("assembly_counts", {})
    if counts.get("non_borderline_target_count") != target_count:
        raise ValueError("random backfill target count drifted")
    representative_count = _require_int(counts.get("representative_count"), "representative_count")
    targeted_count = _require_int(counts.get("targeted_count"), "targeted_count")
    random_count = _require_int(counts.get("random_backfill_count"), "random_backfill_count")
    assembled_count = _require_int(counts.get("assembled_non_borderline_count"), "assembled_non_borderline_count")
    remaining = _require_int(counts.get("remaining_shortfall"), "remaining_shortfall")
    if representative_count + targeted_count + random_count != assembled_count:
        raise ValueError("random backfill assembled count drifted")
    if assembled_count != target_count or remaining != 0:
        raise ValueError("random backfill artifact must assemble the exact non-borderline target")
    targeted_selected_counts = artifact.get("targeted_selected_counts", {})
    if targeted_selected_counts != {
        bucket_id: int(targeted_selected_counts.get(bucket_id, 0))
        for bucket_id in expected_non_borderline_buckets
        if bucket_id != "representative_coverage"
    }:
        raise ValueError("random backfill targeted selected counts drifted")
    if RANDOM_NON_BORDERLINE_BACKFILL_BUCKET in targeted_selected_counts:
        raise ValueError("random backfill must not appear in scarce targeted counts")
    if sum(int(value) for value in targeted_selected_counts.values()) != targeted_count:
        raise ValueError("random backfill targeted count summary drifted")

    backfill = artifact.get("random_backfill", {})
    if backfill.get("bucket_id") != RANDOM_NON_BORDERLINE_BACKFILL_BUCKET:
        raise ValueError("random backfill summary bucket drifted")
    if backfill.get("scarce_bucket_credit") is not False:
        raise ValueError("random backfill summary must set scarce_bucket_credit=false")
    selected = backfill.get("selected_candidates", [])
    if not isinstance(selected, list):
        raise ValueError("random backfill selected candidates must be a list")
    if backfill.get("selected_count") != len(selected) or len(selected) != random_count:
        raise ValueError("random backfill selected count drifted")
    candidate_ids: set[str] = set()
    row_ids: set[tuple[Any, ...]] = set()
    text_hashes: set[str] = set()
    normalized_text_hashes: set[str] = set()
    for candidate in selected:
        _validate_random_backfill_candidate(candidate, config)
        if candidate.get("candidate_hash") != _canonical_hash(candidate, "candidate_hash"):
            raise ValueError("random backfill candidate hash mismatch")
        candidate_id, row_id, text_hash, normalized_text_hash = _candidate_dedupe_keys(candidate)
        if candidate_id in candidate_ids:
            raise ValueError("duplicate candidate id in random backfill")
        if row_id in row_ids:
            raise ValueError("duplicate row identity in random backfill")
        if text_hash and text_hash in text_hashes:
            raise ValueError("duplicate text hash in random backfill")
        if normalized_text_hash and normalized_text_hash in normalized_text_hashes:
            raise ValueError("duplicate normalized text hash in random backfill")
        _add_dedupe_keys(
            candidate,
            candidate_ids=candidate_ids,
            row_ids=row_ids,
            text_hashes=text_hashes,
            normalized_text_hashes=normalized_text_hashes,
        )
    if backfill.get("selected_candidate_ids_hash") != _hash(sorted(candidate_ids)):
        raise ValueError("random backfill selected id hash drifted")
    if artifact.get("artifact_hash") != _canonical_hash(artifact, "generated_at_utc", "artifact_hash"):
        raise ValueError("random backfill artifact hash mismatch")


def validate_non_borderline_random_backfill_artifact(
    artifact: dict[str, Any],
    *,
    config: dict[str, Any],
) -> None:
    _validate_non_borderline_random_backfill_artifact(artifact, config=config)


def build_non_borderline_random_backfill_artifact(
    config: dict[str, Any],
    *,
    representative_candidates: list[dict[str, Any]],
    targeted_plan: dict[str, Any],
    random_backfill_candidates: list[dict[str, Any]],
    excluded_candidates: list[dict[str, Any]] | None = None,
    random_seed: int,
    source_schedule_hash: str | None = None,
) -> dict[str, Any]:
    """Assemble an exact diagnostic 31k pool by filling shortfall with honest random rows."""

    validation = validate_distribution_config(config)
    _validate_targeted_bucket_plan(targeted_plan, config)
    if source_schedule_hash is not None and not _looks_like_hash(source_schedule_hash, length=64):
        raise ValueError("source_schedule_hash must be a 64-character sha256-like hash")
    expected_non_borderline_buckets = _non_borderline_buckets(config)
    non_borderline_target_count = sum(int(validation["bucket_quotas"][bucket]) for bucket in expected_non_borderline_buckets)

    candidate_ids: set[str] = set()
    row_ids: set[tuple[Any, ...]] = set()
    text_hashes: set[str] = set()
    normalized_text_hashes: set[str] = set()

    representative_selected: list[dict[str, Any]] = []
    for candidate in representative_candidates:
        validate_candidate_identity(candidate, config)
        if candidate.get("bucket_id") != "representative_coverage":
            raise ValueError("representative candidate bucket drifted")
        candidate_id, row_id, text_hash, normalized_text_hash = _candidate_dedupe_keys(candidate)
        if candidate_id in candidate_ids or row_id in row_ids:
            raise ValueError("duplicate representative candidate in random backfill assembly")
        if text_hash and text_hash in text_hashes:
            raise ValueError("duplicate representative text hash in random backfill assembly")
        if normalized_text_hash and normalized_text_hash in normalized_text_hashes:
            raise ValueError("duplicate representative normalized text hash in random backfill assembly")
        _add_dedupe_keys(
            candidate,
            candidate_ids=candidate_ids,
            row_ids=row_ids,
            text_hashes=text_hashes,
            normalized_text_hashes=normalized_text_hashes,
        )
        representative_selected.append(candidate)

    targeted_selected: list[dict[str, Any]] = []
    for bucket_id, candidates in targeted_plan.get("selected_candidates", {}).items():
        if bucket_id == ACTIVE_LEARNING_BORDERLINE_BUCKET:
            raise ValueError("active-learning rows cannot enter random backfill assembly")
        for candidate in candidates:
            validate_candidate_identity(candidate, config)
            candidate_id, row_id, text_hash, normalized_text_hash = _candidate_dedupe_keys(candidate)
            if candidate_id in candidate_ids or row_id in row_ids:
                raise ValueError("duplicate targeted candidate conflicts with existing non-borderline rows")
            if text_hash and text_hash in text_hashes:
                raise ValueError("duplicate targeted text hash conflicts with existing non-borderline rows")
            if normalized_text_hash and normalized_text_hash in normalized_text_hashes:
                raise ValueError("duplicate targeted normalized text hash conflicts with existing non-borderline rows")
            _add_dedupe_keys(
                candidate,
                candidate_ids=candidate_ids,
                row_ids=row_ids,
                text_hashes=text_hashes,
                normalized_text_hashes=normalized_text_hashes,
            )
            targeted_selected.append(candidate)

    excluded_candidate_ids: set[str] = set()
    excluded_row_ids: set[tuple[Any, ...]] = set()
    excluded_text_hashes: set[str] = set()
    excluded_normalized_text_hashes: set[str] = set()
    for candidate in excluded_candidates or []:
        _reject_public_leaks(candidate)
        candidate_id, row_id, text_hash, normalized_text_hash = _candidate_dedupe_keys(candidate)
        excluded_candidate_ids.add(candidate_id)
        excluded_row_ids.add(row_id)
        if text_hash:
            excluded_text_hashes.add(text_hash)
        if normalized_text_hash:
            excluded_normalized_text_hashes.add(normalized_text_hash)

    shortfall = non_borderline_target_count - len(representative_selected) - len(targeted_selected)
    if shortfall < 0:
        raise ValueError("existing representative and targeted rows exceed the non-borderline target")

    eligible: list[dict[str, Any]] = []
    skipped_duplicates = 0
    seen_backfill_ids: set[str] = set()
    seen_backfill_rows: set[tuple[Any, ...]] = set()
    seen_backfill_text_hashes: set[str] = set()
    seen_backfill_normalized_text_hashes: set[str] = set()
    for raw_candidate in random_backfill_candidates:
        candidate = deepcopy(raw_candidate)
        _validate_random_backfill_candidate(candidate, config)
        candidate_id, row_id, text_hash, normalized_text_hash = _candidate_dedupe_keys(candidate)
        duplicate = (
            candidate_id in candidate_ids
            or row_id in row_ids
            or candidate_id in excluded_candidate_ids
            or row_id in excluded_row_ids
            or candidate_id in seen_backfill_ids
            or row_id in seen_backfill_rows
            or (text_hash is not None and (text_hash in text_hashes or text_hash in excluded_text_hashes or text_hash in seen_backfill_text_hashes))
            or (
                normalized_text_hash is not None
                and (
                    normalized_text_hash in normalized_text_hashes
                    or normalized_text_hash in excluded_normalized_text_hashes
                    or normalized_text_hash in seen_backfill_normalized_text_hashes
                )
            )
        )
        if duplicate:
            skipped_duplicates += 1
            continue
        seen_backfill_ids.add(candidate_id)
        seen_backfill_rows.add(row_id)
        if text_hash:
            seen_backfill_text_hashes.add(text_hash)
        if normalized_text_hash:
            seen_backfill_normalized_text_hashes.add(normalized_text_hash)
        eligible.append(candidate)

    eligible = sorted(eligible, key=lambda candidate: _backfill_sort_key(candidate, random_seed=random_seed, source_schedule_hash=source_schedule_hash))
    if len(eligible) < shortfall:
        raise ValueError(
            f"not enough eligible random backfill candidates: required {shortfall}, found {len(eligible)}"
        )
    selected = []
    for rank, candidate in enumerate(eligible[:shortfall]):
        updated = deepcopy(candidate)
        updated["random_backfill_rank"] = rank
        updated["random_backfill_seed"] = int(random_seed)
        updated["random_backfill_selection_key_hash"] = _backfill_sort_key(
            updated,
            random_seed=random_seed,
            source_schedule_hash=source_schedule_hash,
        )[0]
        updated = _with_candidate_hash(updated)
        selected.append(updated)

    targeted_counts = _targeted_selected_counts_for_non_borderline(config, targeted_plan)
    selected_ids_hash = _hash(sorted(str(candidate["candidate_id"]) for candidate in selected))
    artifact = {
        "schema_version": 1,
        "artifact_version": RANDOM_BACKFILL_ARTIFACT_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "artifact_scope": "31k_non_borderline_random_backfill_diagnostic",
        "distribution_version": config["distribution_version"],
        "config_hash": validation["config_hash"],
        "diagnostic_only": True,
        "selection_seed": int(config.get("selection_seed", 20260603)),
        "random_seed": int(random_seed),
        "source_schedule_hash": source_schedule_hash,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "paid_calls_made": False,
        "deepseek_calls_made": False,
        "classifier_training_run": False,
        "non_borderline_buckets": expected_non_borderline_buckets,
        "random_backfill_bucket_id": RANDOM_NON_BORDERLINE_BACKFILL_BUCKET,
        "assembly_counts": {
            "non_borderline_target_count": non_borderline_target_count,
            "representative_count": len(representative_selected),
            "targeted_count": len(targeted_selected),
            "random_backfill_count": len(selected),
            "assembled_non_borderline_count": len(representative_selected) + len(targeted_selected) + len(selected),
            "remaining_shortfall": 0,
        },
        "targeted_selected_counts": targeted_counts,
        "representative_summary": {
            "bucket_id": "representative_coverage",
            "selected_count": len(representative_selected),
            "selected_candidate_ids_hash": _hash(sorted(str(candidate["candidate_id"]) for candidate in representative_selected)),
        },
        "random_backfill": {
            "bucket_id": RANDOM_NON_BORDERLINE_BACKFILL_BUCKET,
            "stratum_id": RANDOM_NON_BORDERLINE_BACKFILL_BUCKET,
            "selection_algorithm": "deterministic_stable_hash_order_v1",
            "deterministic_screening_required": True,
            "allowed_deterministic_screening_statuses": sorted(RANDOM_BACKFILL_ALLOWED_SCREENING_STATUSES),
            "scarce_bucket_credit": False,
            "candidate_pool_count": len(random_backfill_candidates),
            "eligible_count": len(eligible),
            "duplicate_or_excluded_rejections": skipped_duplicates,
            "selected_count": len(selected),
            "selected_candidate_ids_hash": selected_ids_hash,
            "selected_candidates": selected,
        },
        "provenance_honesty": {
            "random_backfill_is_quota_fill_only": True,
            "random_backfill_does_not_credit_scarce_buckets": True,
            "original_targeted_bucket_counts_preserved": True,
            "classifier_output_classes_changed": False,
        },
    }
    artifact["artifact_hash"] = _canonical_hash(artifact, "generated_at_utc", "artifact_hash")
    _validate_non_borderline_random_backfill_artifact(artifact, config=config)
    return artifact


def write_non_borderline_random_backfill_artifact(
    artifact: dict[str, Any],
    path: str | Path,
    *,
    config: dict[str, Any],
    allow_reports: bool = False,
) -> None:
    if not allow_reports:
        raise ValueError("random backfill artifact writes require allow_reports=True")
    _validate_non_borderline_random_backfill_artifact(artifact, config=config)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)


def read_non_borderline_random_backfill_artifact(path: str | Path, *, config: dict[str, Any]) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_non_borderline_random_backfill_artifact(artifact, config=config)
    return artifact


def _validate_representative_plan(plan: dict[str, Any], config: dict[str, Any]) -> None:
    _reject_public_leaks(plan)
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported representative plan schema")
    if plan.get("plan_version") != REPRESENTATIVE_ALGORITHM_VERSION:
        raise ValueError("unsupported representative plan version")
    if plan.get("bucket_id") != "representative_coverage":
        raise ValueError("representative plan must target representative_coverage")
    primary = plan.get("primary_candidates", [])
    reserve = plan.get("replacement_reserve", [])
    if not isinstance(primary, list) or not isinstance(reserve, list):
        raise ValueError("representative plan candidates must be lists")
    row_ids: set[tuple[Any, ...]] = set()
    for candidate in [*primary, *reserve]:
        validate_candidate_identity(candidate, config)
        if candidate.get("candidate_hash") != _canonical_hash(candidate, "candidate_hash"):
            raise ValueError("representative candidate hash mismatch")
        if candidate["bucket_id"] != "representative_coverage":
            raise ValueError("representative candidate bucket drifted")
        row_id = _row_identity(candidate)
        if row_id in row_ids:
            raise ValueError("duplicate row identity in representative plan")
        row_ids.add(row_id)
    stable = _canonical_hash(plan, "plan_hash")
    if plan.get("plan_hash") != stable:
        raise ValueError("representative plan hash mismatch")


def _validate_non_borderline_candidate_pool_artifact(artifact: dict[str, Any], config: dict[str, Any]) -> None:
    validation = validate_distribution_config(config)
    _reject_public_leaks(artifact)
    if artifact.get("schema_version") != NON_BORDERLINE_POOL_SCHEMA_VERSION:
        raise ValueError("unsupported non-borderline pool artifact schema")
    if artifact.get("artifact_version") != NON_BORDERLINE_POOL_ARTIFACT_VERSION:
        raise ValueError("unsupported non-borderline pool artifact version")
    if artifact.get("artifact_scope") != "non_borderline_candidate_pool_construction":
        raise ValueError("unexpected non-borderline pool artifact scope")
    if artifact.get("distribution_version") != config["distribution_version"]:
        raise ValueError("non-borderline pool artifact distribution version drifted")
    if artifact.get("config_hash") != validation["config_hash"]:
        raise ValueError("non-borderline pool artifact config hash drifted")
    for flag in (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "labels_are_corpus_approval",
        "training_shards_created",
        "paid_calls_made",
        "deepseek_calls_made",
        "classifier_training_run",
    ):
        if artifact.get(flag) is not False:
            raise ValueError(f"non-borderline pool artifact must set {flag}=false")

    expected_buckets = _non_borderline_buckets(config)
    if artifact.get("non_borderline_buckets") != expected_buckets:
        raise ValueError("non-borderline pool artifact buckets drifted")
    bucket_readiness = artifact.get("bucket_readiness", {})
    if ACTIVE_LEARNING_BORDERLINE_BUCKET in bucket_readiness:
        raise ValueError("non-borderline pool artifact must not include active-learning bucket readiness")
    if artifact.get("total_non_borderline_quota") != sum(validation["bucket_quotas"][bucket] for bucket in expected_buckets):
        raise ValueError("non-borderline pool artifact quota drifted")

    representative_pool = artifact.get("representative_pool", {})
    targeted_pool = artifact.get("targeted_pool", {})
    if not isinstance(representative_pool, dict) or not isinstance(targeted_pool, dict):
        raise ValueError("pool artifact requires representative_pool and targeted_pool mappings")
    representative_primary = representative_pool.get("primary_candidates", [])
    representative_reserve = representative_pool.get("replacement_reserve", [])
    if not isinstance(representative_primary, list) or not isinstance(representative_reserve, list):
        raise ValueError("representative pool candidates must be lists")
    if representative_pool.get("primary_count") != len(representative_primary):
        raise ValueError("representative primary count drifted")
    if representative_pool.get("replacement_reserve_count") != len(representative_reserve):
        raise ValueError("representative reserve count drifted")

    targeted_status = targeted_pool.get("bucket_status", {})
    targeted_selected = targeted_pool.get("selected_candidates", {})
    if ACTIVE_LEARNING_BORDERLINE_BUCKET in targeted_status or ACTIVE_LEARNING_BORDERLINE_BUCKET in targeted_selected:
        raise ValueError("targeted pool must not include active-learning bucket")
    if not isinstance(targeted_selected, dict):
        raise ValueError("targeted selected candidates must be a mapping")

    row_ids: set[tuple[Any, ...]] = set()
    candidate_ids: set[str] = set()
    selected_counts_by_bucket: Counter[str] = Counter()
    for candidate in [*representative_primary, *representative_reserve]:
        validate_candidate_identity(candidate, config)
        if candidate.get("candidate_hash") != _canonical_hash(candidate, "candidate_hash"):
            raise ValueError("representative pool candidate hash mismatch")
        if candidate["bucket_id"] != "representative_coverage":
            raise ValueError("representative candidate bucket drifted")
        row_id = _row_identity(candidate)
        if row_id in row_ids:
            raise ValueError("duplicate row identity across non-borderline pool")
        row_ids.add(row_id)
        if candidate["candidate_id"] in candidate_ids:
            raise ValueError("duplicate candidate id across non-borderline pool")
        candidate_ids.add(candidate["candidate_id"])
    selected_counts_by_bucket["representative_coverage"] = len(representative_primary)

    for bucket_id, candidates in targeted_selected.items():
        if bucket_id == ACTIVE_LEARNING_BORDERLINE_BUCKET:
            raise ValueError("active-learning bucket cannot appear in non-borderline targeted candidates")
        if bucket_id not in expected_buckets:
            raise ValueError(f"unknown non-borderline targeted bucket: {bucket_id}")
        if not isinstance(candidates, list):
            raise ValueError("targeted selected candidate list must be a list")
        selected_counts_by_bucket[bucket_id] = len(candidates)
        for candidate in candidates:
            validate_candidate_identity(candidate, config)
            if candidate.get("candidate_hash") != _canonical_hash(candidate, "candidate_hash"):
                raise ValueError("targeted pool candidate hash mismatch")
            if candidate["bucket_id"] != bucket_id:
                raise ValueError("targeted selected candidate bucket drifted")
            row_id = _row_identity(candidate)
            if row_id in row_ids:
                raise ValueError("duplicate row identity across non-borderline pool")
            row_ids.add(row_id)
            if candidate["candidate_id"] in candidate_ids:
                raise ValueError("duplicate candidate id across non-borderline pool")
            candidate_ids.add(candidate["candidate_id"])

    if set(bucket_readiness) != set(expected_buckets):
        raise ValueError("non-borderline pool artifact bucket readiness drifted")
    for bucket_id in expected_buckets:
        row = bucket_readiness[bucket_id]
        quota = int(validation["bucket_quotas"][bucket_id])
        selected_count = _require_int(row.get("selected_count"), f"{bucket_id} selected_count")
        if row.get("quota") != quota:
            raise ValueError(f"bucket readiness quota drifted for {bucket_id}")
        if selected_count != selected_counts_by_bucket.get(bucket_id, 0):
            raise ValueError(f"bucket readiness selected count drifted for {bucket_id}")
        if row.get("shortfall") != max(0, quota - selected_count):
            raise ValueError(f"bucket readiness shortfall drifted for {bucket_id}")

    readiness_report = artifact.get("readiness_report")
    if not isinstance(readiness_report, dict):
        raise ValueError("non-borderline pool artifact requires readiness_report")
    _validate_non_borderline_readiness_report(readiness_report, config)
    if readiness_report.get("bucket_readiness") != bucket_readiness:
        raise ValueError("readiness report and pool artifact bucket readiness differ")
    active_plan = build_active_learning_plan(config)
    if artifact.get("active_learning_dependency") != readiness_report.get("active_learning_dependency"):
        raise ValueError("active-learning dependency drifted from readiness report")
    if artifact["active_learning_dependency"]["status"] != active_plan["status"]:
        raise ValueError("active-learning dependency status drifted")

    selected_non_borderline_count = sum(
        int(bucket_readiness[bucket_id]["selected_count"]) for bucket_id in expected_buckets
    )
    pool_state = artifact.get("pool_state", {})
    expected_shortfall = int(artifact["total_non_borderline_quota"]) - selected_non_borderline_count
    if pool_state.get("selected_non_borderline_count") != selected_non_borderline_count:
        raise ValueError("pool state selected count drifted")
    if pool_state.get("shortfall") != expected_shortfall:
        raise ValueError("pool state shortfall drifted")
    if pool_state.get("candidate_rows_ready") is not readiness_report.get("non_borderline_candidate_rows_ready"):
        raise ValueError("pool state readiness flag drifted")
    if artifact.get("remaining_blockers") != readiness_report.get("remaining_blockers"):
        raise ValueError("pool artifact blockers drifted from readiness report")
    stable = _canonical_hash(artifact, "generated_at_utc", "artifact_hash")
    if artifact.get("artifact_hash") != stable:
        raise ValueError("non-borderline pool artifact hash mismatch")


def build_non_borderline_candidate_pool_artifact(
    config: dict[str, Any],
    *,
    metadata_manifest: dict[str, Any] | None = None,
    representative_plan: dict[str, Any] | None = None,
    targeted_plan: dict[str, Any] | None = None,
    seed_manifest_summary: dict[str, Any] | None = None,
    prior_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a no-raw, resumable construction artifact for the 31k non-borderline rows."""

    if prior_artifact is not None:
        existing = deepcopy(prior_artifact)
        _validate_non_borderline_candidate_pool_artifact(existing, config)
        return existing

    validation = validate_distribution_config(config)
    if representative_plan is None and metadata_manifest is not None:
        representative_plan = build_representative_sample_plan(metadata_manifest, config=config)
    if representative_plan is not None:
        _validate_representative_plan(representative_plan, config)
    if targeted_plan is None:
        targeted_plan = build_targeted_bucket_plan(config)
    _validate_targeted_bucket_plan(targeted_plan, config)
    if seed_manifest_summary is not None:
        _reject_public_leaks(seed_manifest_summary)

    readiness_report = build_non_borderline_readiness_report(
        config,
        targeted_plan=targeted_plan,
        representative_plan=representative_plan,
        seed_manifest_summary=seed_manifest_summary,
    )
    expected_buckets = _non_borderline_buckets(config)
    total_non_borderline_quota = sum(int(validation["bucket_quotas"][bucket_id]) for bucket_id in expected_buckets)
    bucket_readiness = deepcopy(readiness_report["bucket_readiness"])
    selected_non_borderline_count = sum(int(row["selected_count"]) for row in bucket_readiness.values())
    artifact = {
        "schema_version": NON_BORDERLINE_POOL_SCHEMA_VERSION,
        "artifact_version": NON_BORDERLINE_POOL_ARTIFACT_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "artifact_scope": "non_borderline_candidate_pool_construction",
        "distribution_version": config["distribution_version"],
        "config_hash": validation["config_hash"],
        "selection_seed": int(config.get("selection_seed", 20260603)),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "paid_calls_made": False,
        "deepseek_calls_made": False,
        "classifier_training_run": False,
        "non_borderline_buckets": expected_buckets,
        "total_non_borderline_quota": total_non_borderline_quota,
        "bucket_readiness": bucket_readiness,
        "pool_state": {
            "selected_non_borderline_count": selected_non_borderline_count,
            "shortfall": total_non_borderline_quota - selected_non_borderline_count,
            "candidate_rows_ready": readiness_report["non_borderline_candidate_rows_ready"],
            "final_35k_manifest_ready": readiness_report["final_distribution_candidate_manifest_ready"],
        },
        "representative_pool": _representative_pool_summary(representative_plan),
        "targeted_pool": _targeted_pool_summary(targeted_plan),
        "active_learning_dependency": deepcopy(readiness_report["active_learning_dependency"]),
        "seed_manifest_summary": seed_manifest_summary or None,
        "readiness_report": readiness_report,
        "remaining_blockers": deepcopy(readiness_report["remaining_blockers"]),
    }
    artifact["artifact_hash"] = _canonical_hash(artifact, "generated_at_utc", "artifact_hash")
    _validate_non_borderline_candidate_pool_artifact(artifact, config)
    return artifact


def write_non_borderline_candidate_pool_artifact(
    artifact: dict[str, Any],
    path: str | Path,
    *,
    config: dict[str, Any],
    allow_reports: bool = False,
) -> None:
    if not allow_reports:
        raise ValueError("non-borderline pool artifact writes require allow_reports=True")
    _validate_non_borderline_candidate_pool_artifact(artifact, config)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)


def read_non_borderline_candidate_pool_artifact(path: str | Path, *, config: dict[str, Any]) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_non_borderline_candidate_pool_artifact(artifact, config)
    return artifact


def summarize_seed_manifest(path_or_manifest: str | Path | dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    validate_distribution_config(config)
    if isinstance(path_or_manifest, (str, Path)):
        manifest = json.loads(Path(path_or_manifest).read_text(encoding="utf-8"))
    else:
        manifest = deepcopy(path_or_manifest)
    _reject_public_leaks(manifest)
    candidates = manifest.get("candidates", [])
    available = {}
    quota_report = manifest.get("quota_report")
    if isinstance(quota_report, dict) and isinstance(quota_report.get("available"), dict):
        available = {str(key): int(value) for key, value in quota_report["available"].items()}
    else:
        counts: Counter[str] = Counter()
        for candidate in candidates:
            for stratum in candidate.get("candidate_strata", []):
                counts[str(stratum)] += 1
        available = dict(counts)

    weak_buckets = []
    if available.get("near_threshold_cases", 0) < 500:
        weak_buckets.append("near_threshold_cases")
    if available.get("long_document_one_bad_window", 0) < 500:
        weak_buckets.append("long_document_one_bad_window")
    if len(candidates) < EXPECTED_TOTAL:
        weak_buckets.append("candidate_count")

    missing_identity = []
    first = candidates[0] if candidates else {}
    for field in ("dataset_repo_id", "dataset_config", "dataset_split", "resolved_repo_sha", "parquet_identity"):
        if field not in first:
            missing_identity.append(field)

    return {
        "schema_version": 1,
        "source_manifest_hash": manifest.get("manifest_hash"),
        "candidate_count": len(candidates),
        "pool_role": "seed_pool_only_not_final_distribution",
        "final_distribution_ready": False,
        "raw_text_serialized": manifest.get("raw_text_serialized"),
        "token_serialized": manifest.get("token_serialized"),
        "accepted_manifest": manifest.get("accepted_manifest"),
        "available_strata": dict(sorted(available.items())),
        "weak_buckets": sorted(set(weak_buckets)),
        "missing_first_class_identity_fields": missing_identity,
        "required_action": "mine_distribution_ready_35k_pool_before_freeze_labeling",
    }
