"""Scientific filtering configuration and selected-policy contracts.

This module owns the immutable configuration identity shared by CPU prefilter,
W1 scoring, and offline validation. It deliberately contains no filesystem,
SQLite, model-loading, controller, or command-line behavior.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from twc.classifiers import chunk_contracts
from twc.classifiers.filtering_handoff import build_chunk_classifier_decision
from twc.classifiers.precision_policy import (
    W1_PRECISION_AMP_BF16,
    W1_PRECISION_FP32,
    build_w1_precision_policy,
    normalize_w1_precision_mode,
    validate_w1_precision_policy,
)
from twc.repro.hashing import stable_json_hash, self_hash_mismatches
from twc.data.risk_ontology import RiskOntology
from twc.pretraining.chunked_clean_run import (
    SegmentationResult,
    _estimate_proxy_tokens_from_normalized,
)
from twc.pretraining.deterministic_filter import (
    evaluate_chunk_candidate_deterministic_filter,
    validate_deterministic_filter_policy_payload,
)
from twc.pretraining.filtering import artifacts as filtering_artifacts


CHUNK_FILTERING_POLICY_VERSION = "chunked_clean_run_filtering_v1"
CHUNK_FILTERING_CONFIG_SCHEMA_VERSION = 2
LOCKED_SELECTED_POLICY_REPORT_HASH = "6b438dd3006d175d02530fa5a43ea5865d3b4e21fc20f9fb1d354adeb049a9bc"
LOCKED_SELECTED_POLICY_HASH = "9619f13cf4ccaf491fc37a2765ee9ea55c645a420b088990f0e6c484bda411d8"
LOCKED_CLASSIFIER_MANIFEST_HASH = "5f77195e7f39ed41fad592f2cce2ac72a5efadbca84b131c258281c38ba207d6"
LOCKED_PRIVATE_EVIDENCE_CACHE_HASH = "212f060d24379a4def722b5908d6406cbaf43536970d264c840e4d829dc9a47b"
LOCKED_SOURCE_METADATA_HASH = "44c6ca9aff74beca8405b2592154b46968af92fac27cf4402cdde4dccd8833b5"
LOCKED_SELECTED_EVAL_DETERMINISTIC_FILTER_POLICY_HASH = (
    "15af9f7f2dc3ed4a99edbf50d4a05bd536e3a902d8608db966202d7a04720c3d"
)
LOCKED_S1_REPORT_HASH = "2d5bdc47ceabf3c734d68fe5cbc21fa07c54507536df100c4ceb8bcf57caf7eb"
LOCKED_S1_MODEL_SHA256 = "90d60762c6a3bcabfe7b2b700338022d5ad361873140417d547968a7c5f1065b"
LOCKED_W1_REPORT_HASH = "e6162eed5c31a4756d39339cc3dba996eceb46924bf0c9175ddb36f346eaf27e"
LOCKED_W1_MODEL_SHA256 = "b7be007d738effbb52a1206702982bcd25e1f815fd2269a26816e4ea4e0befa0"
LOCKED_S1_THRESHOLD = 0.960307401567056
LOCKED_W1_THRESHOLD = 0.9765625
LOCKED_W1_PRECISION = W1_PRECISION_AMP_BF16
LEGACY_CPU_PREFILTER_CONFIG_HASH = "83f3d4f7821c57333920b74636fb7947f791fbc4ce07504a38aeea553f5094c5"
LEGACY_CPU_PREFILTER_SELECTED_POLICY_HASH = "fee1a25ae53f498424aa12a51d5c628ece983c2123665d7c4224975145db299c"
LEGACY_CPU_PREFILTER_SELECTED_POLICY_REPORT_HASH = "ba47f27407ea8f547f3d15a57300f92c98fdc521552c14c19f66da00a1f0b960"
LEGACY_CPU_PREFILTER_W1_THRESHOLD = 0.976143479347229
DEFAULT_TARGET_PROXY_TOKENS = 1024
DEFAULT_CANARY_HIGH_TOKEN_ROWS = 800
DEFAULT_CANARY_RANDOM_ROWS = 200
LEGACY_ACCEPTED_TOKEN_TARGET_LOWER_BOUND = 6_000_000_000
DEFAULT_ACCEPTED_TOKEN_TARGET_LOWER_BOUND = 4_000_000_000
DEFAULT_CANARY_BUDGET_CAP_USD = 3.0
DEFAULT_FULL_RUN_BUDGET_CAP_USD = 20.0
DEFAULT_TOKEN_BYTES_PER_TOKEN_LOWER_BOUND = 8
ALLOWED_DECISIONS = {"safe", "reject", "quarantine", "error"}


@dataclass(frozen=True)
class ChunkFilteringConfig:
    classifier_freeze_hash: str
    sparse_report_hash: str
    sparse_model_file_sha256: str
    window_mil_report_hash: str
    window_mil_model_file_sha256: str
    source_metadata_hash: str
    selected_policy_report_hash: str = LOCKED_SELECTED_POLICY_REPORT_HASH
    selected_policy_hash: str = LOCKED_SELECTED_POLICY_HASH
    classifier_manifest_hash: str = LOCKED_CLASSIFIER_MANIFEST_HASH
    private_evidence_cache_hash: str = LOCKED_PRIVATE_EVIDENCE_CACHE_HASH
    selected_eval_deterministic_filter_policy_hash: str = LOCKED_SELECTED_EVAL_DETERMINISTIC_FILTER_POLICY_HASH
    sparse_threshold: float = LOCKED_S1_THRESHOLD
    window_mil_threshold: float = LOCKED_W1_THRESHOLD
    sparse_model_id: str = chunk_contracts.CHUNK_SPARSE_MODEL_ID
    window_mil_model_id: str = chunk_contracts.CHUNK_WINDOW_MIL_MODEL_ID
    window_mil_input_mode: str = chunk_contracts.WINDOW_INPUT_FULL_EVIDENCE
    window_mil_pooling_mode: str = "topk_mean_logits"
    window_mil_pooling_top_k: int = 3
    window_mil_precision: str = LOCKED_W1_PRECISION
    deterministic_filter_policy_hash: str | None = None
    target_proxy_tokens: int = DEFAULT_TARGET_PROXY_TOKENS
    halo_chars: int = chunk_contracts.DEFAULT_HALO_CHARS
    canary_high_token_rows: int = DEFAULT_CANARY_HIGH_TOKEN_ROWS
    canary_random_rows: int = DEFAULT_CANARY_RANDOM_ROWS
    accepted_token_target_lower_bound: int = DEFAULT_ACCEPTED_TOKEN_TARGET_LOWER_BOUND
    canary_budget_cap_usd: float = DEFAULT_CANARY_BUDGET_CAP_USD
    full_run_budget_cap_usd: float = DEFAULT_FULL_RUN_BUDGET_CAP_USD


def build_chunk_filtering_config_payload(config: ChunkFilteringConfig | Mapping[str, Any]) -> dict[str, Any]:
    config = _coerce_config(config)
    precision_mode = normalize_w1_precision_mode(config.window_mil_precision)
    payload = {
        "artifact_kind": filtering_artifacts.CHUNK_FILTERING_CONFIG_KIND,
        "schema_version": CHUNK_FILTERING_CONFIG_SCHEMA_VERSION,
        "filtering_policy_version": CHUNK_FILTERING_POLICY_VERSION,
        "classifier_freeze_hash": filtering_artifacts._require_sha256(
            config.classifier_freeze_hash,
            "classifier_freeze_hash",
        ),
        "selected_policy_report_hash": filtering_artifacts._require_sha256(
            config.selected_policy_report_hash,
            "selected_policy_report_hash",
        ),
        "selected_policy_hash": filtering_artifacts._require_sha256(
            config.selected_policy_hash,
            "selected_policy_hash",
        ),
        "classifier_manifest_hash": filtering_artifacts._require_sha256(
            config.classifier_manifest_hash,
            "classifier_manifest_hash",
        ),
        "private_evidence_cache_hash": filtering_artifacts._require_sha256(
            config.private_evidence_cache_hash,
            "private_evidence_cache_hash",
        ),
        "sparse_report_hash": filtering_artifacts._require_sha256(
            config.sparse_report_hash,
            "sparse_report_hash",
        ),
        "sparse_model_file_sha256": filtering_artifacts._require_sha256(
            config.sparse_model_file_sha256,
            "sparse_model_file_sha256",
        ),
        "window_mil_report_hash": filtering_artifacts._require_sha256(
            config.window_mil_report_hash,
            "window_mil_report_hash",
        ),
        "window_mil_model_file_sha256": filtering_artifacts._require_sha256(
            config.window_mil_model_file_sha256,
            "window_mil_model_file_sha256",
        ),
        "source_metadata_hash": filtering_artifacts._require_sha256(
            config.source_metadata_hash,
            "source_metadata_hash",
        ),
        "selected_eval_deterministic_filter_policy_hash": filtering_artifacts._require_sha256(
            config.selected_eval_deterministic_filter_policy_hash,
            "selected_eval_deterministic_filter_policy_hash",
        ),
        "sparse_model_id": str(config.sparse_model_id),
        "window_mil_model_id": str(config.window_mil_model_id),
        "window_mil_input_mode": str(config.window_mil_input_mode),
        "window_mil_pooling_mode": str(config.window_mil_pooling_mode),
        "window_mil_pooling_top_k": filtering_artifacts._positive_int(
            config.window_mil_pooling_top_k,
            "window_mil_pooling_top_k",
        ),
        "sparse_threshold": _threshold(config.sparse_threshold, "sparse_threshold"),
        "window_mil_threshold": _threshold(config.window_mil_threshold, "window_mil_threshold"),
        "target_proxy_tokens": filtering_artifacts._positive_int(
            config.target_proxy_tokens,
            "target_proxy_tokens",
        ),
        "halo_chars": filtering_artifacts._nonnegative_int(config.halo_chars, "halo_chars"),
        "canary_high_token_rows": filtering_artifacts._positive_int(
            config.canary_high_token_rows,
            "canary_high_token_rows",
        ),
        "canary_random_rows": filtering_artifacts._positive_int(
            config.canary_random_rows,
            "canary_random_rows",
        ),
        "canary_physical_row_count": int(config.canary_high_token_rows) + int(config.canary_random_rows),
        "accepted_token_target_lower_bound": filtering_artifacts._positive_int(
            config.accepted_token_target_lower_bound,
            "accepted_token_target_lower_bound",
        ),
        "canary_budget_cap_usd": filtering_artifacts._nonnegative_float(
            config.canary_budget_cap_usd,
            "canary_budget_cap_usd",
        ),
        "full_run_budget_cap_usd": filtering_artifacts._nonnegative_float(
            config.full_run_budget_cap_usd,
            "full_run_budget_cap_usd",
        ),
        "thresholds_frozen": True,
        "classifier_can_approve_corpus": False,
        "clean_runs_never_bridge_gaps": True,
        "halos_are_classifier_evidence_only": True,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
    }
    if config.deterministic_filter_policy_hash is not None:
        payload["deterministic_filter_policy_hash"] = filtering_artifacts._require_sha256(
            config.deterministic_filter_policy_hash,
            "deterministic_filter_policy_hash",
        )
    if precision_mode != W1_PRECISION_FP32:
        payload["window_mil_precision"] = precision_mode
        payload["window_mil_precision_policy"] = build_w1_precision_policy(precision_mode)
    payload["config_hash"] = stable_json_hash(payload)
    validate_chunk_filtering_config_payload(payload)
    return payload


def validate_chunk_filtering_config_payload(payload: Mapping[str, Any]) -> None:
    filtering_artifacts.validate_public_no_raw_payload(payload)
    if payload.get("artifact_kind") != filtering_artifacts.CHUNK_FILTERING_CONFIG_KIND:
        raise ValueError("unsupported chunk filtering config kind")
    if payload.get("schema_version") != CHUNK_FILTERING_CONFIG_SCHEMA_VERSION:
        raise ValueError("chunk filtering config schema_version mismatch")
    if payload.get("filtering_policy_version") != CHUNK_FILTERING_POLICY_VERSION:
        raise ValueError("chunk filtering policy version mismatch")
    for field in (
        "classifier_freeze_hash",
        "selected_policy_report_hash",
        "selected_policy_hash",
        "classifier_manifest_hash",
        "private_evidence_cache_hash",
        "sparse_report_hash",
        "sparse_model_file_sha256",
        "window_mil_report_hash",
        "window_mil_model_file_sha256",
        "source_metadata_hash",
        "selected_eval_deterministic_filter_policy_hash",
    ):
        filtering_artifacts._require_sha256(payload.get(field), field)
    if payload.get("deterministic_filter_policy_hash") is not None:
        filtering_artifacts._require_sha256(
            payload.get("deterministic_filter_policy_hash"),
            "deterministic_filter_policy_hash",
        )
    for field in (
        "target_proxy_tokens",
        "canary_high_token_rows",
        "canary_random_rows",
        "accepted_token_target_lower_bound",
    ):
        filtering_artifacts._positive_int(payload.get(field), field)
    filtering_artifacts._nonnegative_int(payload.get("halo_chars"), "halo_chars")
    if payload.get("sparse_model_id") != chunk_contracts.CHUNK_SPARSE_MODEL_ID:
        raise ValueError("chunk filtering config sparse_model_id mismatch")
    if payload.get("window_mil_model_id") != chunk_contracts.CHUNK_WINDOW_MIL_MODEL_ID:
        raise ValueError("chunk filtering config window_mil_model_id mismatch")
    if payload.get("window_mil_input_mode") != chunk_contracts.WINDOW_INPUT_FULL_EVIDENCE:
        raise ValueError("chunk filtering config window_mil_input_mode mismatch")
    if payload.get("window_mil_pooling_mode") != "topk_mean_logits":
        raise ValueError("chunk filtering config window_mil_pooling_mode mismatch")
    if filtering_artifacts._positive_int(
        payload.get("window_mil_pooling_top_k"),
        "window_mil_pooling_top_k",
    ) != 3:
        raise ValueError("chunk filtering config window_mil_pooling_top_k mismatch")
    precision_present = (
        payload.get("window_mil_precision") is not None
        or payload.get("window_mil_precision_policy") is not None
    )
    config_precision = normalize_w1_precision_mode(
        str(payload.get("window_mil_precision") or W1_PRECISION_FP32)
    )
    precision_policy = payload.get("window_mil_precision_policy")
    if precision_present:
        if not isinstance(precision_policy, Mapping):
            raise ValueError("chunk filtering config window_mil_precision_policy missing")
        if config_precision != str(precision_policy.get("precision_mode")):
            raise ValueError("chunk filtering config window_mil_precision policy mismatch")
        validate_w1_precision_policy(precision_policy)
    _threshold(payload.get("sparse_threshold"), "sparse_threshold")
    _threshold(payload.get("window_mil_threshold"), "window_mil_threshold")
    if int(payload.get("canary_physical_row_count", -1)) != int(payload["canary_high_token_rows"]) + int(
        payload["canary_random_rows"]
    ):
        raise ValueError("canary physical row count mismatch")
    if payload.get("thresholds_frozen") is not True:
        raise ValueError("chunk filtering config must freeze thresholds")
    if payload.get("classifier_can_approve_corpus") is not False:
        raise ValueError("classifier must not be able to approve corpus")
    filtering_artifacts._require_false_flags(
        payload,
        filtering_artifacts.PUBLIC_FALSE_FLAGS,
        "chunk filtering config",
    )
    if self_hash_mismatches(payload, "config_hash"):
        raise ValueError("chunk filtering config hash mismatch")


def _validate_prefilter_input_config_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("config_hash") != LEGACY_CPU_PREFILTER_CONFIG_HASH:
        validate_chunk_filtering_config_payload(payload)
        return
    filtering_artifacts.validate_public_no_raw_payload(payload)
    if payload.get("artifact_kind") != filtering_artifacts.CHUNK_FILTERING_CONFIG_KIND:
        raise ValueError("legacy prefilter config kind mismatch")
    if payload.get("schema_version") != CHUNK_FILTERING_CONFIG_SCHEMA_VERSION:
        raise ValueError("legacy prefilter config schema_version mismatch")
    if payload.get("filtering_policy_version") != CHUNK_FILTERING_POLICY_VERSION:
        raise ValueError("legacy prefilter filtering policy version mismatch")
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "deterministic_filter_policy_hash",
        "sparse_report_hash",
        "sparse_model_file_sha256",
        "window_mil_report_hash",
        "window_mil_model_file_sha256",
    ):
        filtering_artifacts._require_sha256(payload.get(field), f"legacy prefilter {field}")
    if payload.get("selected_policy_hash") != LEGACY_CPU_PREFILTER_SELECTED_POLICY_HASH:
        raise ValueError("legacy prefilter selected_policy_hash mismatch")
    if payload.get("selected_policy_report_hash") != LEGACY_CPU_PREFILTER_SELECTED_POLICY_REPORT_HASH:
        raise ValueError("legacy prefilter selected_policy_report_hash mismatch")
    if payload.get("classifier_freeze_hash") != LEGACY_CPU_PREFILTER_SELECTED_POLICY_REPORT_HASH:
        raise ValueError("legacy prefilter classifier_freeze_hash mismatch")
    if float(payload.get("window_mil_threshold")) != float(LEGACY_CPU_PREFILTER_W1_THRESHOLD):
        raise ValueError("legacy prefilter W1 threshold mismatch")
    if int(payload.get("accepted_token_target_lower_bound", 0)) != LEGACY_ACCEPTED_TOKEN_TARGET_LOWER_BOUND:
        raise ValueError("legacy prefilter accepted token target mismatch")
    if payload.get("window_mil_precision") is not None or payload.get("window_mil_precision_policy") is not None:
        raise ValueError("legacy prefilter config must not declare current W1 precision policy")
    filtering_artifacts._require_false_flags(
        payload,
        filtering_artifacts.PUBLIC_FALSE_FLAGS,
        "legacy prefilter config",
    )


def validate_locked_chunk_filtering_config_payload(payload: Mapping[str, Any]) -> None:
    validate_chunk_filtering_config_payload(payload)
    expected = {
        "classifier_freeze_hash": LOCKED_SELECTED_POLICY_REPORT_HASH,
        "selected_policy_report_hash": LOCKED_SELECTED_POLICY_REPORT_HASH,
        "selected_policy_hash": LOCKED_SELECTED_POLICY_HASH,
        "classifier_manifest_hash": LOCKED_CLASSIFIER_MANIFEST_HASH,
        "private_evidence_cache_hash": LOCKED_PRIVATE_EVIDENCE_CACHE_HASH,
        "source_metadata_hash": LOCKED_SOURCE_METADATA_HASH,
        "selected_eval_deterministic_filter_policy_hash": LOCKED_SELECTED_EVAL_DETERMINISTIC_FILTER_POLICY_HASH,
        "deterministic_filter_policy_hash": LOCKED_SELECTED_EVAL_DETERMINISTIC_FILTER_POLICY_HASH,
        "sparse_report_hash": LOCKED_S1_REPORT_HASH,
        "sparse_model_file_sha256": LOCKED_S1_MODEL_SHA256,
        "window_mil_report_hash": LOCKED_W1_REPORT_HASH,
        "window_mil_model_file_sha256": LOCKED_W1_MODEL_SHA256,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise ValueError(f"locked chunk filtering config {field} mismatch")
    if payload.get("deterministic_filter_policy_hash") != payload.get(
        "selected_eval_deterministic_filter_policy_hash"
    ):
        raise ValueError("locked chunk filtering config selected/executed deterministic_filter_policy_hash mismatch")
    if float(payload.get("sparse_threshold")) != LOCKED_S1_THRESHOLD:
        raise ValueError("locked chunk filtering config sparse_threshold mismatch")
    if float(payload.get("window_mil_threshold")) != LOCKED_W1_THRESHOLD:
        raise ValueError("locked chunk filtering config window_mil_threshold mismatch")
    if normalize_w1_precision_mode(
        str(payload.get("window_mil_precision") or W1_PRECISION_FP32)
    ) != LOCKED_W1_PRECISION:
        raise ValueError("locked chunk filtering config window_mil_precision mismatch")


def _retarget_accepted_token_lower_bound(
    config_payload: Mapping[str, Any],
    target: int,
) -> dict[str, Any]:
    payload = dict(config_payload)
    payload.pop("config_hash", None)
    payload["accepted_token_target_lower_bound"] = filtering_artifacts._positive_int(
        target,
        "accepted_token_target_lower_bound",
    )
    payload["config_hash"] = stable_json_hash(payload)
    validate_chunk_filtering_config_payload(payload)
    return payload


def _legacy_cpu_prefilter_config_payload(config_payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(config_payload)
    payload["config_hash"] = LEGACY_CPU_PREFILTER_CONFIG_HASH
    payload["selected_policy_hash"] = LEGACY_CPU_PREFILTER_SELECTED_POLICY_HASH
    payload["selected_policy_report_hash"] = LEGACY_CPU_PREFILTER_SELECTED_POLICY_REPORT_HASH
    payload["classifier_freeze_hash"] = LEGACY_CPU_PREFILTER_SELECTED_POLICY_REPORT_HASH
    payload["window_mil_threshold"] = LEGACY_CPU_PREFILTER_W1_THRESHOLD
    payload["accepted_token_target_lower_bound"] = LEGACY_ACCEPTED_TOKEN_TARGET_LOWER_BOUND
    payload.pop("window_mil_precision", None)
    payload.pop("window_mil_precision_policy", None)
    return payload


def _prefilter_input_config_payload_for_hash(
    config_payload: Mapping[str, Any],
    observed_config_hash: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    observed = filtering_artifacts._require_sha256(observed_config_hash, f"{label} config_hash")
    if observed == config_payload["config_hash"]:
        return config_payload
    if int(config_payload.get("accepted_token_target_lower_bound", 0)) != LEGACY_ACCEPTED_TOKEN_TARGET_LOWER_BOUND:
        legacy_payload = _retarget_accepted_token_lower_bound(
            config_payload,
            LEGACY_ACCEPTED_TOKEN_TARGET_LOWER_BOUND,
        )
        if observed == legacy_payload["config_hash"]:
            return legacy_payload
    if observed == LEGACY_CPU_PREFILTER_CONFIG_HASH:
        return _legacy_cpu_prefilter_config_payload(config_payload)
    raise ValueError(f"{label} config_hash mismatch")


def _freeze_thresholds_by_model_id(freeze_report: Mapping[str, Any]) -> dict[str, float]:
    thresholds = freeze_report.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("freeze report missing thresholds")
    aliases = {
        chunk_contracts.CHUNK_SPARSE_MODEL_ID: (chunk_contracts.CHUNK_SPARSE_MODEL_ID, "sparse"),
        chunk_contracts.CHUNK_WINDOW_MIL_MODEL_ID: (
            chunk_contracts.CHUNK_WINDOW_MIL_MODEL_ID,
            "window_mil",
        ),
    }
    normalized: dict[str, float] = {}
    for model_id, keys in aliases.items():
        for key in keys:
            if key in thresholds:
                normalized[model_id] = _threshold(thresholds[key], f"threshold.{model_id}")
                break
        if model_id not in normalized:
            raise ValueError("freeze report missing model threshold")
    return normalized


def _selected_policy_thresholds_by_model_id(freeze_report: Mapping[str, Any]) -> dict[str, float]:
    selected_policy = freeze_report.get("selected_policy")
    if not isinstance(selected_policy, Mapping):
        raise ValueError("selected policy freeze report missing selected_policy")
    return {
        chunk_contracts.CHUNK_SPARSE_MODEL_ID: _threshold(
            selected_policy.get("sparse_threshold"),
            f"threshold.{chunk_contracts.CHUNK_SPARSE_MODEL_ID}",
        ),
        chunk_contracts.CHUNK_WINDOW_MIL_MODEL_ID: _threshold(
            selected_policy.get("window_mil_threshold"),
            f"threshold.{chunk_contracts.CHUNK_WINDOW_MIL_MODEL_ID}",
        ),
    }


def _validate_selected_policy_freeze_report(
    freeze_report: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> None:
    filtering_artifacts.validate_public_no_raw_payload(freeze_report)
    if freeze_report.get("artifact_kind") != filtering_artifacts.SELECTED_POLICY_FREEZE_REPORT_KIND:
        raise ValueError("unsupported selected policy freeze report kind")
    if freeze_report.get("status") != "ready":
        raise ValueError("selected policy freeze report must be ready")
    filtering_artifacts._validate_report_self_hash_required(freeze_report, "selected policy freeze report")
    filtering_artifacts._require_false_flags(
        freeze_report,
        filtering_artifacts.PUBLIC_FALSE_FLAGS,
        "selected policy freeze report",
    )
    if freeze_report.get("classifier_can_approve_corpus") is not False:
        raise ValueError("selected policy freeze report must not approve corpus")
    if freeze_report.get("labels_are_corpus_approval") is not False:
        raise ValueError("selected policy freeze labels must not approve corpus")
    selected_policy = freeze_report.get("selected_policy")
    if not isinstance(selected_policy, Mapping):
        raise ValueError("selected policy freeze report missing selected_policy")
    if freeze_report.get("selected_policy_hash") != stable_json_hash(selected_policy):
        raise ValueError("selected policy hash mismatch")
    if selected_policy.get("policy_family") != "deterministic_terminal_or_sparse_or_window_mil_veto":
        raise ValueError("selected policy family mismatch")
    if selected_policy.get("selection_split") != "calibration":
        raise ValueError("selected policy must be selected on calibration")
    if selected_policy.get("threshold_direction") != "reject_at_or_above":
        raise ValueError("selected policy threshold direction mismatch")
    if selected_policy.get("deterministic_terminal_core_reject_fixed") is not True:
        raise ValueError("selected policy must fix deterministic terminal rejects")
    if selected_policy.get("halo_hits_terminal") is not False:
        raise ValueError("selected policy must keep halo hits nonterminal")
    if selected_policy.get("audit_only_hits_terminal") is not False:
        raise ValueError("selected policy must keep audit-only hits nonterminal")
    if selected_policy.get("classifier_can_approve_corpus") is not False:
        raise ValueError("selected policy must not approve corpus")
    if selected_policy.get("missing_or_non_finite_score_action") != "block_freeze":
        raise ValueError("selected policy must block missing or non-finite scores")
    thresholds = _selected_policy_thresholds_by_model_id(freeze_report)
    for field in (
        "report_hash",
        "selected_policy_hash",
        "classifier_manifest_hash",
        "evidence_cache_hash",
        "deterministic_filter_policy_hash",
    ):
        filtering_artifacts._require_sha256(freeze_report.get(field), field)
    if config_payload is None:
        return
    validate_chunk_filtering_config_payload(config_payload)
    expected = {
        "report_hash": "selected_policy_report_hash",
        "selected_policy_hash": "selected_policy_hash",
        "classifier_manifest_hash": "classifier_manifest_hash",
        "evidence_cache_hash": "private_evidence_cache_hash",
        "deterministic_filter_policy_hash": "deterministic_filter_policy_hash",
    }
    for report_field, config_field in expected.items():
        if freeze_report.get(report_field) != config_payload.get(config_field):
            raise ValueError(f"selected policy freeze report {report_field} mismatch")
    if freeze_report.get("report_hash") != config_payload.get("classifier_freeze_hash"):
        raise ValueError("selected policy freeze report classifier_freeze_hash mismatch")
    if freeze_report.get("deterministic_filter_policy_hash") != config_payload.get(
        "selected_eval_deterministic_filter_policy_hash"
    ):
        raise ValueError("selected policy freeze deterministic policy hash mismatch")
    config_precision = normalize_w1_precision_mode(
        str(config_payload.get("window_mil_precision") or W1_PRECISION_FP32)
    )
    report_precision_policy = freeze_report.get("window_mil_precision_policy")
    report_precision_hash = freeze_report.get("window_mil_precision_policy_hash")
    if config_precision != W1_PRECISION_FP32 or report_precision_policy is not None or report_precision_hash is not None:
        expected_precision_policy = build_w1_precision_policy(config_precision)
        if not isinstance(report_precision_policy, Mapping):
            raise ValueError("selected policy freeze report window_mil_precision_policy missing")
        validate_w1_precision_policy(report_precision_policy)
        if dict(report_precision_policy) != expected_precision_policy:
            raise ValueError("selected policy freeze report window_mil_precision policy mismatch")
        if report_precision_hash != expected_precision_policy["precision_policy_hash"]:
            raise ValueError("selected policy freeze report window_mil_precision_policy_hash mismatch")
    if thresholds[chunk_contracts.CHUNK_SPARSE_MODEL_ID] != float(config_payload["sparse_threshold"]):
        raise ValueError("selected policy freeze sparse threshold mismatch")
    if thresholds[chunk_contracts.CHUNK_WINDOW_MIL_MODEL_ID] != float(config_payload["window_mil_threshold"]):
        raise ValueError("selected policy freeze Window-MIL threshold mismatch")


def build_filtering_candidates(
    result: SegmentationResult,
    *,
    raw_text: str,
    config_hash: str,
    target_proxy_tokens: int,
    halo_chars: int,
    deterministic_policy_hash: str | None = None,
) -> list[dict[str, Any]]:
    if result.receipt.document_status != "segmented":
        return []
    if raw_text != result.source.raw_text:
        raise ValueError("filtering raw text does not match segmentation result")
    filtering_artifacts._require_sha256(config_hash, "config_hash")
    if deterministic_policy_hash is not None:
        filtering_artifacts._require_sha256(deterministic_policy_hash, "deterministic_policy_hash")
    target = filtering_artifacts._positive_int(target_proxy_tokens, "target_proxy_tokens")
    halo = filtering_artifacts._nonnegative_int(halo_chars, "halo_chars")
    candidates: list[dict[str, Any]] = []
    current: list[Any] = []
    current_proxy = 0
    for block in result.blocks:
        if current and current_proxy >= target:
            candidates.append(
                _candidate_from_blocks(
                    result,
                    raw_text,
                    current,
                    config_hash=config_hash,
                    target_proxy_tokens=target,
                    halo_chars=halo,
                    deterministic_policy_hash=deterministic_policy_hash,
                )
            )
            current = []
            current_proxy = 0
        current.append(block)
        current_proxy += int(block.proxy_token_count)
        if block.block_kind == "oversized_unsegmentable":
            candidates.append(
                _candidate_from_blocks(
                    result,
                    raw_text,
                    current,
                    config_hash=config_hash,
                    target_proxy_tokens=target,
                    halo_chars=halo,
                    deterministic_policy_hash=deterministic_policy_hash,
                )
            )
            current = []
            current_proxy = 0
    if current:
        candidates.append(
            _candidate_from_blocks(
                result,
                raw_text,
                current,
                config_hash=config_hash,
                target_proxy_tokens=target,
                halo_chars=halo,
                deterministic_policy_hash=deterministic_policy_hash,
            )
        )
    _validate_candidate_block_coverage(result, candidates)
    return candidates


def build_chunk_filtering_outputs_for_document(
    result: SegmentationResult,
    *,
    raw_text: str,
    candidates: Sequence[Mapping[str, Any]],
    scorers: Sequence[chunk_contracts.ChunkFilteringScorer],
    thresholds: Mapping[str, float],
    config_hash: str,
    source_chunk_manifest_hash: str,
    deterministic_policy_payload: Mapping[str, Any] | None = None,
    deterministic_ontology: RiskOntology | None = None,
) -> dict[str, Any]:
    filtering_artifacts._require_sha256(config_hash, "config_hash")
    filtering_artifacts._require_sha256(source_chunk_manifest_hash, "source_chunk_manifest_hash")
    if result.receipt.document_status != "segmented":
        return _terminal_filtering_outputs(result, config_hash=config_hash)
    if raw_text != result.source.raw_text:
        raise ValueError("filtering raw text does not match segmentation result")
    deterministic_enabled = deterministic_policy_payload is not None or deterministic_ontology is not None
    deterministic_policy_hash: str | None = None
    if deterministic_enabled:
        if deterministic_policy_payload is None or deterministic_ontology is None:
            raise ValueError("deterministic filter policy and ontology must be provided together")
        validate_deterministic_filter_policy_payload(deterministic_policy_payload)
        deterministic_policy_hash = filtering_artifacts._require_sha256(
            deterministic_policy_payload.get("deterministic_filter_policy_hash"),
            "deterministic_filter_policy_hash",
        )
    candidate_rows = [dict(candidate) for candidate in candidates]
    _validate_candidate_block_coverage(result, candidate_rows)
    for candidate in candidate_rows:
        _validate_candidate_hashes(candidate, raw_text)
        if (
            deterministic_policy_hash is not None
            and candidate.get("deterministic_filter_policy_hash") != deterministic_policy_hash
        ):
            raise ValueError("classifier candidate deterministic filter policy hash mismatch")
    scorer_by_id = {str(scorer.model_id): scorer for scorer in scorers}
    unknown = set(scorer_by_id) - set(thresholds)
    if unknown:
        raise ValueError(f"scorer has no configured threshold: {sorted(unknown)}")
    threshold_payload = {model_id: _threshold(value, f"threshold.{model_id}") for model_id, value in thresholds.items()}
    score_maps: dict[str, dict[str, float]] = {str(candidate["candidate_id_hash"]): {} for candidate in candidate_rows}
    deterministic_results: dict[str, Mapping[str, Any] | None] = {}
    deterministic_error_candidate_ids: set[str] = set()
    scorer_candidate_rows: list[Mapping[str, Any]] = []
    for candidate in candidate_rows:
        candidate_id = str(candidate["candidate_id_hash"])
        deterministic_result: Mapping[str, Any] | None = None
        deterministic_decision = _deterministic_candidate_decision(candidate)
        if deterministic_enabled:
            assert deterministic_ontology is not None and deterministic_policy_payload is not None
            try:
                deterministic_result = evaluate_chunk_candidate_deterministic_filter(
                    candidate,
                    raw_text=raw_text,
                    ontology=deterministic_ontology,
                    policy_payload=deterministic_policy_payload,
                )
            except Exception:
                deterministic_error_candidate_ids.add(candidate_id)
        deterministic_results[candidate_id] = deterministic_result
        if (
            deterministic_decision is None
            and candidate_id not in deterministic_error_candidate_ids
            and not (deterministic_result and deterministic_result.get("terminal_decision") == "reject")
        ):
            scorer_candidate_rows.append(candidate)
    evidence_texts = [
        raw_text[int(candidate["evidence_char_start"]) : int(candidate["evidence_char_end"])]
        for candidate in scorer_candidate_rows
    ]
    metadata = [_candidate_public_metadata(candidate) for candidate in scorer_candidate_rows]
    for model_id, scorer in sorted(scorer_by_id.items()):
        try:
            scores = list(scorer.score_many(list(evidence_texts), list(metadata)))
        except Exception:
            scores = [math.nan] * len(scorer_candidate_rows)
        if len(scores) != len(scorer_candidate_rows):
            scores = [math.nan] * len(scorer_candidate_rows)
        for candidate, score in zip(scorer_candidate_rows, scores, strict=True):
            parsed = chunk_contracts.coerce_optional_risk_score(score)
            if parsed is not None:
                score_maps[str(candidate["candidate_id_hash"])][model_id] = parsed
    decisions: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        candidate_id = str(candidate["candidate_id_hash"])
        deterministic_result = deterministic_results.get(candidate_id)
        deterministic_decision = _deterministic_candidate_decision(candidate)
        if candidate_id in deterministic_error_candidate_ids:
            decision = _manual_chunk_decision(
                candidate_id=candidate_id,
                source_chunk_manifest_hash=source_chunk_manifest_hash,
                status="quarantine",
                thresholds=threshold_payload,
                model_scores={},
                quarantine_reasons=["deterministic_filter_error"],
            )
        elif deterministic_result and deterministic_result.get("terminal_decision") == "reject":
            decision = _manual_chunk_decision(
                candidate_id=candidate_id,
                source_chunk_manifest_hash=source_chunk_manifest_hash,
                status="reject",
                thresholds=threshold_payload,
                model_scores={},
                quarantine_reasons=[],
                deterministic_filter_result=deterministic_result,
            )
        elif deterministic_decision is not None:
            decision = _manual_chunk_decision(
                candidate_id=candidate_id,
                source_chunk_manifest_hash=source_chunk_manifest_hash,
                status=deterministic_decision,
                thresholds=threshold_payload,
                model_scores=score_maps[candidate_id],
                quarantine_reasons=["oversized_unsegmentable"] if deterministic_decision == "quarantine" else [],
                deterministic_filter_result=deterministic_result,
            )
        else:
            decision = build_chunk_classifier_decision(
                candidate_id=candidate_id,
                source_chunk_manifest_hash=source_chunk_manifest_hash,
                model_scores=score_maps[candidate_id],
                thresholds=threshold_payload,
                uncertainty_margin=0.0,
            )
            decision = {
                **decision,
                "candidate_id_hash": candidate_id,
                "decision_id_hash": decision["classifier_decision_hash"],
            }
            if deterministic_result is not None:
                decision = _attach_deterministic_filter_result(decision, deterministic_result)
        decisions.append(decision)
    block_decisions: dict[int, dict[str, Any]] = {}
    for candidate, decision in zip(candidate_rows, decisions, strict=True):
        for block_index in candidate["core_block_indexes"]:
            block_decisions[int(block_index)] = decision
    clean_runs, gap_events, private_rows = _build_clean_runs_and_gaps(result, raw_text, candidate_rows, decisions, block_decisions)
    document_report = {
        "artifact_kind": filtering_artifacts.CHUNK_FILTER_DOCUMENT_REPORT_KIND,
        "schema_version": 1,
        "filtering_policy_version": CHUNK_FILTERING_POLICY_VERSION,
        "source_document_id_hash": result.source.source_document_id_hash,
        "segmentation_receipt_hash": result.receipt.receipt_hash,
        "config_hash": config_hash,
        "candidate_count": len(candidate_rows),
        "decision_counts": filtering_artifacts._sorted_counter(Counter(str(decision["classifier_decision"]) for decision in decisions)),
        "clean_run_count": len(clean_runs),
        "gap_event_count": len(gap_events),
        "accepted_estimated_token_count_lower_bound": sum(int(run["estimated_token_count_lower_bound"]) for run in clean_runs),
        "candidate_core_block_coverage_validated": True,
        "clean_runs_never_bridge_gaps": True,
        "halos_are_classifier_evidence_only": True,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
    }
    if deterministic_policy_hash is not None:
        document_report.update(
            {
                "deterministic_filter_policy_hash": deterministic_policy_hash,
                "deterministic_terminal_reject_count": sum(
                    1
                    for result_item in deterministic_results.values()
                    if result_item and result_item.get("terminal_decision") == "reject"
                ),
                "deterministic_audit_only_candidate_count": sum(
                    1
                    for result_item in deterministic_results.values()
                    if result_item and int(result_item["audit_only_hit_count"]) > 0
                ),
                "deterministic_halo_evidence_candidate_count": sum(
                    1
                    for result_item in deterministic_results.values()
                    if result_item and int(result_item["halo_evidence_hit_count"]) > 0
                ),
                "deterministic_error_quarantine_count": len(deterministic_error_candidate_ids),
            }
        )
    document_report["document_report_hash"] = stable_json_hash(document_report)
    filtering_artifacts.validate_public_no_raw_payload(document_report)
    return {
        "source_document_id_hash": result.source.source_document_id_hash,
        "segmentation_receipt_hash": result.receipt.receipt_hash,
        "candidates": candidate_rows,
        "decisions": decisions,
        "clean_runs": clean_runs,
        "gap_events": gap_events,
        "private_clean_run_rows": private_rows,
        "document_report": document_report,
    }


def _terminal_filtering_outputs(result: SegmentationResult, *, config_hash: str) -> dict[str, Any]:
    source_id = result.source.source_document_id_hash
    decision = result.receipt.document_status
    status = "error" if decision in {"missing_text", "non_string_text"} else "quarantine"
    report = {
        "artifact_kind": filtering_artifacts.CHUNK_FILTER_DOCUMENT_REPORT_KIND,
        "schema_version": 1,
        "filtering_policy_version": CHUNK_FILTERING_POLICY_VERSION,
        "source_document_id_hash": source_id,
        "segmentation_receipt_hash": result.receipt.receipt_hash,
        "config_hash": config_hash,
        "candidate_count": 0,
        "decision_counts": {status: 1},
        "clean_run_count": 0,
        "gap_event_count": 0,
        "accepted_estimated_token_count_lower_bound": 0,
        "candidate_core_block_coverage_validated": True,
        "clean_runs_never_bridge_gaps": True,
        "halos_are_classifier_evidence_only": True,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
    }
    report["document_report_hash"] = stable_json_hash(report)
    return {
        "source_document_id_hash": source_id,
        "segmentation_receipt_hash": result.receipt.receipt_hash,
        "candidates": [],
        "decisions": [],
        "clean_runs": [],
        "gap_events": [],
        "private_clean_run_rows": [],
        "document_report": report,
    }


def _candidate_from_blocks(
    result: SegmentationResult,
    raw_text: str,
    blocks: Sequence[Any],
    *,
    config_hash: str,
    target_proxy_tokens: int,
    halo_chars: int,
    deterministic_policy_hash: str | None,
) -> dict[str, Any]:
    source_id = result.source.source_document_id_hash
    core_start = min(int(block.owned_char_start) for block in blocks)
    core_end = max(int(block.owned_char_end) for block in blocks)
    evidence_start = max(0, core_start - int(halo_chars))
    evidence_end = min(len(raw_text), core_end + int(halo_chars))
    core_text = raw_text[core_start:core_end]
    evidence_text = raw_text[evidence_start:evidence_end]
    block_indexes = [int(block.block_index) for block in blocks]
    core_proxy_token_count = sum(int(block.proxy_token_count) for block in blocks)
    candidate_policy = {
        "filtering_policy_version": CHUNK_FILTERING_POLICY_VERSION,
        "config_hash": config_hash,
        "target_proxy_tokens": int(target_proxy_tokens),
        "halo_chars": int(halo_chars),
    }
    if deterministic_policy_hash is not None:
        candidate_policy["deterministic_filter_policy_hash"] = filtering_artifacts._require_sha256(
            deterministic_policy_hash,
            "deterministic_policy_hash",
        )
    candidate_policy_hash = stable_json_hash(candidate_policy)
    identity = {
        "source_document_id_hash": source_id,
        "segmentation_receipt_hash": result.receipt.receipt_hash,
        "candidate_policy_hash": candidate_policy_hash,
        "core_block_indexes": block_indexes,
        "core_char_start": core_start,
        "core_char_end": core_end,
        "evidence_char_start": evidence_start,
        "evidence_char_end": evidence_end,
        "core_text_sha256": filtering_artifacts._sha256_text(core_text),
        "evidence_text_sha256": filtering_artifacts._sha256_text(evidence_text),
    }
    payload = {
        "candidate_id_hash": stable_json_hash(identity),
        **identity,
        "candidate_policy_hash": candidate_policy_hash,
        "source_document_id_hash": source_id,
        "split": result.source.split,
        "block_id_hashes": [str(block.block_id_hash) for block in blocks],
        "block_kinds": [str(block.block_kind) for block in blocks],
        "core_proxy_token_count": core_proxy_token_count,
        "evidence_proxy_token_count": max(
            core_proxy_token_count,
            _estimate_proxy_tokens_from_normalized(evidence_text),
        ),
        "halo_left_chars": core_start - evidence_start,
        "halo_right_chars": evidence_end - core_end,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    if deterministic_policy_hash is not None:
        payload["deterministic_filter_policy_hash"] = deterministic_policy_hash
    filtering_artifacts.validate_public_no_raw_payload(payload)
    _validate_candidate_hashes(payload, raw_text)
    return payload


def _validate_candidate_hashes(candidate: Mapping[str, Any], raw_text: str) -> None:
    for field in (
        "candidate_id_hash",
        "source_document_id_hash",
        "segmentation_receipt_hash",
        "candidate_policy_hash",
        "core_text_sha256",
        "evidence_text_sha256",
    ):
        filtering_artifacts._require_sha256(candidate.get(field), field)
    core_start = filtering_artifacts._nonnegative_int(candidate.get("core_char_start"), "core_char_start")
    core_end = filtering_artifacts._positive_int(candidate.get("core_char_end"), "core_char_end")
    evidence_start = filtering_artifacts._nonnegative_int(candidate.get("evidence_char_start"), "evidence_char_start")
    evidence_end = filtering_artifacts._positive_int(candidate.get("evidence_char_end"), "evidence_char_end")
    core_proxy_tokens = filtering_artifacts._nonnegative_int(candidate.get("core_proxy_token_count"), "core_proxy_token_count")
    evidence_proxy_tokens = filtering_artifacts._nonnegative_int(candidate.get("evidence_proxy_token_count"), "evidence_proxy_token_count")
    if evidence_proxy_tokens < core_proxy_tokens:
        raise ValueError("candidate evidence_proxy_token_count must cover core_proxy_token_count")
    if not evidence_start <= core_start < core_end <= evidence_end <= len(raw_text):
        raise ValueError("candidate core/evidence spans are invalid")
    if filtering_artifacts._sha256_text(raw_text[core_start:core_end]) != candidate["core_text_sha256"]:
        raise ValueError("candidate core text hash mismatch")
    if filtering_artifacts._sha256_text(raw_text[evidence_start:evidence_end]) != candidate["evidence_text_sha256"]:
        raise ValueError("candidate evidence text hash mismatch")


def _validate_candidate_block_coverage(result: SegmentationResult, candidates: Sequence[Mapping[str, Any]]) -> None:
    expected = list(range(len(result.blocks)))
    covered: list[int] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = filtering_artifacts._require_sha256(candidate.get("candidate_id_hash"), "candidate_id_hash")
        if candidate_id in seen:
            raise ValueError("duplicate classifier candidate id")
        seen.add(candidate_id)
        blocks = list(candidate.get("core_block_indexes") or [])
        if not blocks:
            raise ValueError("classifier candidate must cover blocks")
        if blocks != sorted(int(block) for block in blocks):
            raise ValueError("candidate core block indexes must be sorted")
        covered.extend(int(block) for block in blocks)
    if sorted(covered) != expected or len(covered) != len(set(covered)):
        raise ValueError("candidate core block coverage mismatch")


def _deterministic_candidate_decision(candidate: Mapping[str, Any]) -> str | None:
    if "oversized_unsegmentable" in set(candidate.get("block_kinds") or []):
        return "quarantine"
    return None


def _manual_chunk_decision(
    *,
    candidate_id: str,
    source_chunk_manifest_hash: str,
    status: str,
    thresholds: Mapping[str, float],
    model_scores: Mapping[str, float],
    quarantine_reasons: Sequence[str],
    deterministic_filter_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in ALLOWED_DECISIONS:
        raise ValueError("unsupported manual chunk decision")
    payload = {
        "artifact_kind": "twc_chunk_classifier_decision",
        "schema_version": 1,
        "candidate_id": filtering_artifacts._require_sha256(candidate_id, "candidate_id"),
        "candidate_id_hash": filtering_artifacts._require_sha256(candidate_id, "candidate_id_hash"),
        "source_chunk_manifest_hash": filtering_artifacts._require_sha256(source_chunk_manifest_hash, "source_chunk_manifest_hash"),
        "classifier_decision": status,
        "model_scores": {str(key): chunk_contracts.coerce_risk_score(value) for key, value in sorted(model_scores.items())},
        "thresholds": {str(key): _threshold(value, f"threshold.{key}") for key, value in sorted(thresholds.items())},
        "rejecting_models": [],
        "quarantine_reasons": list(quarantine_reasons),
        "classifier_can_approve_corpus": False,
        "pending_required_gates": ["clean_run_gap_validation", "post_filter_downstream_gates"] if status == "safe" else [],
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
    if deterministic_filter_result is not None:
        payload.update(_deterministic_filter_decision_fields(deterministic_filter_result))
    payload["classifier_decision_hash"] = stable_json_hash(payload)
    payload["decision_id_hash"] = payload["classifier_decision_hash"]
    filtering_artifacts.validate_public_no_raw_payload(payload)
    return payload


def _attach_deterministic_filter_result(
    decision: Mapping[str, Any],
    deterministic_filter_result: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(decision)
    payload.update(_deterministic_filter_decision_fields(deterministic_filter_result))
    payload.pop("classifier_decision_hash", None)
    payload.pop("decision_id_hash", None)
    payload["classifier_decision_hash"] = stable_json_hash(payload)
    payload["decision_id_hash"] = payload["classifier_decision_hash"]
    filtering_artifacts.validate_public_no_raw_payload(payload)
    return payload


def _deterministic_filter_decision_fields(deterministic_filter_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "deterministic_filter_policy_hash": filtering_artifacts._require_sha256(
            deterministic_filter_result.get("deterministic_filter_policy_hash"),
            "deterministic_filter_policy_hash",
        ),
        "deterministic_filter_result_hash": filtering_artifacts._require_sha256(
            deterministic_filter_result.get("deterministic_filter_result_hash"),
            "deterministic_filter_result_hash",
        ),
        "deterministic_terminal_hit_count": int(deterministic_filter_result.get("terminal_hit_count", 0)),
        "deterministic_audit_only_hit_count": int(deterministic_filter_result.get("audit_only_hit_count", 0)),
        "deterministic_halo_evidence_hit_count": int(deterministic_filter_result.get("halo_evidence_hit_count", 0)),
        "deterministic_terminal_rule_ids": sorted(
            str(item) for item in deterministic_filter_result.get("terminal_rule_ids", [])
        ),
        "deterministic_audit_only_rule_ids": sorted(
            str(item) for item in deterministic_filter_result.get("audit_only_rule_ids", [])
        ),
        "deterministic_halo_evidence_rule_ids": sorted(
            str(item) for item in deterministic_filter_result.get("halo_evidence_rule_ids", [])
        ),
    }


def _build_clean_runs_and_gaps(
    result: SegmentationResult,
    raw_text: str,
    candidates: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    block_decisions: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    clean_runs: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    candidate_by_block: dict[int, Mapping[str, Any]] = {}
    for candidate in candidates:
        for block_index in candidate["core_block_indexes"]:
            candidate_by_block[int(block_index)] = candidate
    current: list[Any] = []
    for block in result.blocks:
        decision = block_decisions[int(block.block_index)]
        if decision["classifier_decision"] == "safe":
            current.append(block)
            continue
        if current:
            run, private_row = _clean_run_from_blocks(result, raw_text, current, clean_run_index=len(clean_runs))
            clean_runs.append(run)
            private_rows.append(private_row)
            current = []
        candidate = candidate_by_block[int(block.block_index)]
        gaps.append(_gap_event_from_candidate(result, candidate, decision))
    if current:
        run, private_row = _clean_run_from_blocks(result, raw_text, current, clean_run_index=len(clean_runs))
        clean_runs.append(run)
        private_rows.append(private_row)
    return clean_runs, gaps, private_rows


def _clean_run_from_blocks(
    result: SegmentationResult,
    raw_text: str,
    blocks: Sequence[Any],
    *,
    clean_run_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    block_indexes = [int(block.block_index) for block in blocks]
    start = int(blocks[0].content_char_start)
    end = int(blocks[-1].content_char_end)
    private_text = raw_text[start:end]
    clean_run_id = stable_json_hash(
        {
            "source_document_id_hash": result.source.source_document_id_hash,
            "segmentation_receipt_hash": result.receipt.receipt_hash,
            "clean_run_index": int(clean_run_index),
            "block_indexes": block_indexes,
            "clean_run_text_sha256": filtering_artifacts._sha256_text(private_text),
        }
    )
    public = {
        "clean_run_id_hash": clean_run_id,
        "source_document_id_hash": result.source.source_document_id_hash,
        "segmentation_receipt_hash": result.receipt.receipt_hash,
        "clean_run_index": int(clean_run_index),
        "block_indexes": block_indexes,
        "block_id_hashes": [str(block.block_id_hash) for block in blocks],
        "clean_run_text_sha256": filtering_artifacts._sha256_text(private_text),
        "clean_run_char_count": len(private_text),
        "estimated_token_count_lower_bound": _estimated_token_lower_bound(private_text),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    public["clean_run_hash"] = stable_json_hash(public)
    private = {
        "clean_run_id_hash": clean_run_id,
        "source_document_id_hash": result.source.source_document_id_hash,
        "clean_run_text_sha256": public["clean_run_text_sha256"],
        "private_text": private_text,
    }
    filtering_artifacts.validate_public_no_raw_payload(public)
    return public, private


def _gap_event_from_candidate(
    result: SegmentationResult,
    candidate: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "gap_event_id_hash": stable_json_hash(
            {
                "source_document_id_hash": result.source.source_document_id_hash,
                "candidate_id_hash": candidate["candidate_id_hash"],
                "classifier_decision_hash": decision["classifier_decision_hash"],
                "block_indexes": candidate["core_block_indexes"],
            }
        ),
        "source_document_id_hash": result.source.source_document_id_hash,
        "segmentation_receipt_hash": result.receipt.receipt_hash,
        "candidate_id_hash": candidate["candidate_id_hash"],
        "block_indexes": list(candidate["core_block_indexes"]),
        "classifier_decision": decision["classifier_decision"],
        "classifier_decision_hash": decision["classifier_decision_hash"],
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    filtering_artifacts.validate_public_no_raw_payload(payload)
    return payload


def _candidate_public_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id_hash": candidate["candidate_id_hash"],
        "source_document_id_hash": candidate["source_document_id_hash"],
        "core_block_indexes": list(candidate["core_block_indexes"]),
        "core_char_start": int(candidate["core_char_start"]),
        "core_char_end": int(candidate["core_char_end"]),
        "core_text_sha256": candidate["core_text_sha256"],
        "evidence_text_sha256": candidate["evidence_text_sha256"],
    }


def _estimated_token_lower_bound(text: str) -> int:
    return max(1, len(text.encode("utf-8", errors="surrogatepass")) // DEFAULT_TOKEN_BYTES_PER_TOKEN_LOWER_BOUND)


def _coerce_config(config: ChunkFilteringConfig | Mapping[str, Any]) -> ChunkFilteringConfig:
    if isinstance(config, ChunkFilteringConfig):
        return config
    return ChunkFilteringConfig(
        classifier_freeze_hash=str(config["classifier_freeze_hash"]),
        sparse_report_hash=str(config["sparse_report_hash"]),
        sparse_model_file_sha256=str(config["sparse_model_file_sha256"]),
        window_mil_report_hash=str(config["window_mil_report_hash"]),
        window_mil_model_file_sha256=str(config["window_mil_model_file_sha256"]),
        source_metadata_hash=str(config["source_metadata_hash"]),
        selected_policy_report_hash=str(
            config.get("selected_policy_report_hash", LOCKED_SELECTED_POLICY_REPORT_HASH)
        ),
        selected_policy_hash=str(config.get("selected_policy_hash", LOCKED_SELECTED_POLICY_HASH)),
        classifier_manifest_hash=str(config.get("classifier_manifest_hash", LOCKED_CLASSIFIER_MANIFEST_HASH)),
        private_evidence_cache_hash=str(
            config.get("private_evidence_cache_hash", LOCKED_PRIVATE_EVIDENCE_CACHE_HASH)
        ),
        selected_eval_deterministic_filter_policy_hash=str(
            config.get(
                "selected_eval_deterministic_filter_policy_hash",
                LOCKED_SELECTED_EVAL_DETERMINISTIC_FILTER_POLICY_HASH,
            )
        ),
        sparse_threshold=float(config.get("sparse_threshold", LOCKED_S1_THRESHOLD)),
        window_mil_threshold=float(config.get("window_mil_threshold", LOCKED_W1_THRESHOLD)),
        sparse_model_id=str(config.get("sparse_model_id", chunk_contracts.CHUNK_SPARSE_MODEL_ID)),
        window_mil_model_id=str(config.get("window_mil_model_id", chunk_contracts.CHUNK_WINDOW_MIL_MODEL_ID)),
        window_mil_input_mode=str(
            config.get("window_mil_input_mode", chunk_contracts.WINDOW_INPUT_FULL_EVIDENCE)
        ),
        window_mil_pooling_mode=str(config.get("window_mil_pooling_mode", "topk_mean_logits")),
        window_mil_pooling_top_k=int(config.get("window_mil_pooling_top_k", 3)),
        window_mil_precision=normalize_w1_precision_mode(
            str(config.get("window_mil_precision", LOCKED_W1_PRECISION))
        ),
        deterministic_filter_policy_hash=(
            str(config["deterministic_filter_policy_hash"])
            if config.get("deterministic_filter_policy_hash") is not None
            else None
        ),
        target_proxy_tokens=int(config.get("target_proxy_tokens", DEFAULT_TARGET_PROXY_TOKENS)),
        halo_chars=int(config.get("halo_chars", chunk_contracts.DEFAULT_HALO_CHARS)),
        canary_high_token_rows=int(config.get("canary_high_token_rows", DEFAULT_CANARY_HIGH_TOKEN_ROWS)),
        canary_random_rows=int(config.get("canary_random_rows", DEFAULT_CANARY_RANDOM_ROWS)),
        accepted_token_target_lower_bound=int(
            config.get("accepted_token_target_lower_bound", DEFAULT_ACCEPTED_TOKEN_TARGET_LOWER_BOUND)
        ),
        canary_budget_cap_usd=float(config.get("canary_budget_cap_usd", DEFAULT_CANARY_BUDGET_CAP_USD)),
        full_run_budget_cap_usd=float(config.get("full_run_budget_cap_usd", DEFAULT_FULL_RUN_BUDGET_CAP_USD)),
    )


def _threshold(value: Any, label: str) -> float:
    try:
        score = chunk_contracts.coerce_risk_score(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric and finite") from exc
    if not 0 <= score <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return score
