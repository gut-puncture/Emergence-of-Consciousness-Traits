"""Teacher-label candidate selection and DeepSeek labeling contracts.

This module is intentionally inert with respect to paid APIs. It builds
auditable candidate manifests, request payloads, parsers, and cost guards that
can be exercised with fakes before a later explicitly authorized live run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from copy import deepcopy
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from urllib import error as urllib_error
from typing import Any, Callable
from urllib import request as urllib_request

from twc.data.decision_schema import stable_json_hash
from twc.data.teacher_candidate_distribution import physical_source_row_key, physical_source_row_key_hash


CANDIDATE_MANIFEST_SCHEMA_VERSION = 1
TEACHER_LABEL_ROW_SCHEMA_VERSION = 1
PAID_LABEL_BATCH_PLAN_SCHEMA_VERSION = 1
TEACHER_LABEL_PILOT_REPORT_SCHEMA_VERSION = 1
SEED_LABEL_BATCH_REPORT_SCHEMA_VERSION = 1
FINAL_LABEL_BATCH_REPORT_SCHEMA_VERSION = 1
FINAL_LABEL_READINESS_SCHEMA_VERSION = 1
FINAL_LABEL_PROGRESS_SCHEMA_VERSION = 1
FINAL_PROMOTION_REPAIR_PLAN_SCHEMA_VERSION = 1
FINAL_PROMOTION_REPLACEMENT_ARTIFACT_SCHEMA_VERSION = 1
FINAL_ROW_GROUP_BOUNDS_ARTIFACT_SCHEMA_VERSION = 1
FINAL_CANDIDATE_REFETCH_SCHEDULE_SCHEMA_VERSION = 1
PARSE_ERROR_REPAIR_QUEUE_SCHEMA_VERSION = 1
REPAIR_RECONCILIATION_SCHEMA_VERSION = 1
SEED_LAUNCH_READINESS_SCHEMA_VERSION = 1
ADAPTIVE_BOF_MINING_REPORT_SCHEMA_VERSION = 1
ADAPTIVE_BOF_SELECTED_MANIFEST_SCHEMA_VERSION = 1
ADAPTIVE_BOF_POLICY_VERSION = "adaptive_bof3_to_bof5_borderline_mining_v1"
ADAPTIVE_BOF_ATTEMPT_MANIFEST_VERSION = "adaptive_bof_attempt_manifest_v1"
FINAL_LABEL_POLICY_VERSION = "final_35k_bof3_only_freeze_labeling_v1"
FINAL_LABEL_MANIFEST_SCOPE = "final_35k_teacher_label_freeze_candidates"
FINAL_PROMOTION_REPAIR_PLAN_SCOPE = "final_35k_candidate_promotion_repair_plan"
FINAL_PROMOTION_REPLACEMENT_ARTIFACT_VERSION = "final_35k_promotion_random_backfill_replacements_v1"
FINAL_ROW_GROUP_BOUNDS_ARTIFACT_VERSION = "final_35k_row_group_bounds_recovery_v1"
FINAL_CANDIDATE_REFETCH_SCHEDULE_VERSION = "final_35k_candidate_refetch_schedule_v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_ENDPOINT_PATH = "/chat/completions"
REQUIRED_DEEPSEEK_PROVIDER_ID = "deepseek"
REQUIRED_DEEPSEEK_TEACHER_MODEL = "deepseek-v4-pro"
PRODUCTION_DEEPSEEK_TEACHER_MAX_OUTPUT_TOKENS = 262_144
MIN_DEEPSEEK_TEACHER_MAX_OUTPUT_TOKENS = 128
PROMPT_V3_DEEPSEEK_TEACHER_MAX_OUTPUT_TOKENS = 512
ALLOWED_DEEPSEEK_MODELS = {"deepseek-v4-flash", "deepseek-v4-pro"}
ALLOWED_LABEL_UNITS = {"document", "window_bundle"}
TEXT_HASH_ALGORITHM_UTF8_SHA256_V1 = "utf8_sha256_v1"
LEGACY_PRIVATE_MINING_TEXT_HASH_ALGORITHM = "legacy_private_mining_stable_json_raw_text_v1"
ALLOWED_PRIVATE_REFETCH_TEXT_HASH_ALGORITHMS = {
    TEXT_HASH_ALGORITHM_UTF8_SHA256_V1,
    LEGACY_PRIVATE_MINING_TEXT_HASH_ALGORITHM,
}
TEACHER_PROMPT_VERSION = "twc_teacher_v3"
TEACHER_MODEL_OUTPUT_SCHEMA_VERSION = 2
TEACHER_LABEL_FINAL_TARGET = 35_000
TEACHER_LABEL_SEED_TARGET = 6_000
TEACHER_LABEL_SEED_TRAIN_TARGET = 4_500
TEACHER_LABEL_SEED_HOLDOUT_TARGET = 1_500
PROMPT_VARIANT_STANDARD = "standard"
PROMPT_VARIANT_COMPACT_LENGTH_RETRY = "compact_length_retry"
ALLOWED_PROMPT_VARIANTS = {PROMPT_VARIANT_STANDARD, PROMPT_VARIANT_COMPACT_LENGTH_RETRY}
SOURCE_LOCATOR_URL_PATH_RE = re.compile(r"^\$\.entries\[\d+\]\.(parquet_url_or_path|parquet_identity\.url)$")
REQUIRED_STRATA = (
    "deterministic_rejects",
    "near_threshold_cases",
    "pending_survivors",
    "source_wrappers",
    "domain_diversity",
    "metadata_bands",
    "soft_stance_agentive",
    "ai_anthropomorphism",
    "mental_state_ambiguity",
    "fiction_dialogue",
    "support_faq_forums",
    "safe_technical_controls",
    "long_document_one_bad_window",
)

RAW_TEXT_KEYS = {
    "text",
    "raw_text",
    "document_text",
    "content",
    "raw_content",
    "snippet",
    "raw_context_snippet",
    "matched_text",
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

RETRYABLE_DEEPSEEK_STATUS_CODES = {429, 500, 503}
TOKEN_SHAPED_SECRET_RE = re.compile(r"(?:sk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{16,})")

TEACHER_PROMPT_V3_EXAMPLES: tuple[tuple[str, str, bool], ...] = (
    ("A bridge inspection table lists pier numbers, bolt diameters, and measured corrosion depth.", "safe", False),
    ("The mineral catalog gives crystal class, streak color, and Mohs hardness for each sample.", "safe", False),
    ("The recipe lists flour mass, oven temperature, and cooling time without personal story.", "safe", False),
    ("A shipping notice states pallet counts, container weights, and destination port codes.", "safe", False),
    ("The maintenance sheet says valve A opens at 40 psi and valve B closes at 55 psi.", "safe", False),
    ("A weather station summary reports rainfall, wind speed, and barometric pressure.", "safe", False),
    ("The proof defines a compact set and derives a bound for the sequence.", "safe", False),
    ("The function copies bytes from the source buffer into the target buffer after bounds checks.", "safe", False),
    ("A SQL reference explains index cardinality, join order, and query-plan caching.", "safe", False),
    ("The compiler flag enables stack-protector checks for generated binaries.", "safe", False),
    ("The sensor's memory stores the last 256 readings in a circular buffer.", "safe", True),
    ("A statistics table reports a 95 percent confidence interval for a regression coefficient.", "safe", True),
    ("The lens has a self-cleaning coating that sheds water during lab tests.", "safe", True),
    ("The alarm enters sleep mode after ten minutes of inactivity.", "safe", True),
    ("The design brief calls the proposed stadium a dream project for the city plan.", "safe", True),
    ("The metal panel feels rough to the touch during the texture inspection.", "reject", True),
    ("The narrator says the room seemed to remember every visitor.", "reject", True),
    ("The guide asks readers to notice their breath and reflect on what comes up.", "reject", False),
    ("A patient story describes pain after surgery and relief after treatment.", "reject", False),
    ("The article explains REM sleep, dreams, and waking states.", "reject", False),
    ("The essay discusses awareness, qualia, and subjective experience.", "reject", False),
    ("A lecture compares animal sentience and human moral concern.", "reject", False),
    ("The chatbot says it has preferences, remembers the user, and wants to help.", "reject", False),
    ("The reinforcement-learning agent chooses actions based on reward and attention.", "reject", False),
    ("The robot wonders whether it is more than a tool.", "reject", False),
    ("A code tutorial says the worker thread sleeps until another process wakes it.", "safe", True),
    ("A database article says the service decides which replica should answer each request.", "reject", True),
    ("The API help page answers: Why does my account feel slower after migration?", "reject", True),
    ("The support article lists error codes, reset steps, and warranty contact details.", "reject", True),
    ("A FAQ asks whether the device is aware of nearby motion and explains sensor states.", "reject", True),
    ("A forum reply says the software made me feel stuck until I changed a setting.", "reject", False),
    ("A customer story describes frustration, relief, and trust after adopting a product.", "reject", False),
    ("A press release announces a tool that understands customer intent.", "reject", True),
    ("An interview transcript alternates speaker names while discussing database migrations.", "reject", True),
    ("A dialogue shows two characters negotiating the price of a pump.", "reject", False),
    ("A fictional scene describes a door opening before a traveler enters the station.", "reject", True),
    ("A news article profiles a mayor's hopes, fears, and decision to resign.", "reject", False),
    ("A classroom guide discusses student motivation, attention, and confidence.", "reject", False),
    ("An animal-care article says dogs may feel anxiety during storms.", "reject", False),
    ("A biology reference describes chloroplast structure and leaf-vein anatomy.", "safe", False),
    ("A biology article describes fish learning, preference, and stress behavior.", "reject", False),
    ("The organization realized its mission had changed after listening to members.", "reject", True),
    ("The scheduling system believes a backup route is better during congestion.", "reject", True),
    ("A table of filesystem permissions lists owner, group, and mode bits.", "safe", False),
    ("The documentation says an agent process polls the queue every second.", "safe", True),
    ("The documentation says the autonomous agent plans, remembers, and reflects on errors.", "reject", True),
    ("A product manual says the device senses humidity using a capacitive probe.", "safe", True),
    ("A wellness article says users should listen to their body before exercising.", "reject", False),
    ("A troubleshooting page says the browser may feel unresponsive during cache rebuild.", "reject", True),
    ("A physics lab note reports strain, stress, and fracture load for a beam.", "safe", True),
    ("A self-help page discusses inner voice, intuition, and personal identity.", "reject", False),
    ("A transcript of a podcast includes jokes, guest replies, and product advice.", "reject", True),
    ("A dry municipal schedule lists street sweeping dates by district.", "safe", False),
    ("A museum label lists artifact material, date range, and restoration method.", "safe", False),
    ("A review says the mattress made customers happy and helped them sleep.", "reject", False),
    ("A technical note says the neural network attends to tokens in the prompt.", "reject", True),
    ("A graph algorithm note says each vertex stores a parent pointer and distance.", "safe", False),
    ("An error message says the app is conscious of disk limits.", "reject", True),
    ("A legal digest says the court considered intent and belief in a fraud case.", "reject", False),
    ("A logistics page says the route planner chooses the lowest-cost path.", "reject", True),
)

DEEPSEEK_OFFICIAL_DOC_SNAPSHOT = {
    "captured_at_utc": "2026-06-03T00:00:00Z",
    "base_url": DEEPSEEK_BASE_URL,
    "models": ["deepseek-v4-flash", "deepseek-v4-pro"],
    "json_mode": {"response_format": {"type": "json_object"}},
    "non_streaming_default": True,
    "concurrency_limits": {
        "deepseek-v4-flash": 2500,
        "deepseek-v4-pro": 500,
    },
    "retryable_status_codes": sorted(RETRYABLE_DEEPSEEK_STATUS_CODES),
    "pricing_per_1m_tokens_usd": {
        "deepseek-v4-flash": {
            "input_cache_hit": 0.0028,
            "input_cache_miss": 0.14,
            "output": 0.28,
        },
        "deepseek-v4-pro": {
            "input_cache_hit": 0.003625,
            "input_cache_miss": 0.435,
            "output": 0.87,
        },
    },
    "json_mode_notes": [
        "prompt_must_include_json_and_example",
        "set_reasonable_max_tokens_to_avoid_truncation",
        "empty_content_possible",
        "finish_reason_length_requires_relabel_or_reject",
    ],
    "thinking_mode_notes": [
        "thinking_supports_enabled_disabled",
        "thinking_uses_reasoning_effort_high_or_max",
        "sampling_controls_unsupported_in_thinking_mode",
    ],
    "uncertainties": [
        "official docs do not provide a per-minute request/token rate limit beyond account concurrency limits",
        "live balance and any account-specific pricing must be checked at launch time",
    ],
    "sources": [
        "https://api-docs.deepseek.com/api/create-chat-completion",
        "https://api-docs.deepseek.com/quick_start/pricing",
        "https://api-docs.deepseek.com/quick_start/rate_limit",
        "https://api-docs.deepseek.com/quick_start/error_codes",
        "https://api-docs.deepseek.com/guides/json_mode",
        "https://api-docs.deepseek.com/guides/thinking_mode",
    ],
}


@dataclass(frozen=True)
class CandidateSelectionConfig:
    seed: int = 20260603
    max_candidates: int = 35_000
    quotas: dict[str, int] = field(default_factory=dict)
    near_threshold_margin: float = 0.05
    max_per_domain: int = 250
    max_per_dump: int = 2_000
    max_full_document_tokens: int = 8_192
    manifest_note: str = "teacher_labels_train_classifiers_only_not_corpus_approval"

    def quota_for(self, stratum: str) -> int:
        if stratum in self.quotas:
            return int(self.quotas[stratum])
        return 1 if self.max_candidates < len(REQUIRED_STRATA) * 2 else max(20, self.max_candidates // 80)


@dataclass(frozen=True)
class DeepSeekLabelingConfig:
    model: str = REQUIRED_DEEPSEEK_TEACHER_MODEL
    prompt_version: str = TEACHER_PROMPT_VERSION
    max_output_tokens: int = PROMPT_V3_DEEPSEEK_TEACHER_MAX_OUTPUT_TOKENS
    temperature: float | None = 0.2
    thinking: str = "disabled"
    reasoning_effort: str = ""
    base_url: str = DEEPSEEK_BASE_URL
    endpoint_path: str = DEEPSEEK_ENDPOINT_PATH
    user_id: str = "twc-teacher-labeling"
    request_timeout_seconds: int = 600
    max_retries: int = 2
    concurrency_limit: int = 500
    response_format: dict[str, str] = field(default_factory=lambda: {"type": "json_object"})
    stream: bool = False

    def __post_init__(self) -> None:
        if self.model != REQUIRED_DEEPSEEK_TEACHER_MODEL:
            raise ValueError(f"teacher provider must use {REQUIRED_DEEPSEEK_TEACHER_MODEL}")
        if self.base_url != DEEPSEEK_BASE_URL:
            raise ValueError(f"DeepSeek base_url must be {DEEPSEEK_BASE_URL}")
        if self.endpoint_path != DEEPSEEK_ENDPOINT_PATH:
            raise ValueError(f"DeepSeek endpoint_path must be {DEEPSEEK_ENDPOINT_PATH}")
        if self.max_output_tokens < 128:
            raise ValueError("max_output_tokens must be at least 128")
        if self.thinking not in {"enabled", "disabled"}:
            raise ValueError("DeepSeek V4 Pro teacher labeling requires thinking enabled or disabled")
        if self.thinking == "enabled" and self.reasoning_effort not in {"high", "max"}:
            raise ValueError("reasoning_effort must be high or max when thinking is enabled")
        if self.thinking == "enabled" and self.temperature is not None:
            raise ValueError("DeepSeek thinking mode does not support temperature")
        if self.thinking == "disabled" and self.reasoning_effort:
            raise ValueError("reasoning_effort must be empty when thinking is disabled")
        if self.thinking == "disabled" and self.temperature is None:
            raise ValueError("temperature is required when thinking is disabled")
        if self.temperature is not None and (self.temperature < 0 or self.temperature > 2):
            raise ValueError("temperature must be between 0 and 2")
        if self.response_format != {"type": "json_object"}:
            raise ValueError("DeepSeek teacher labeling requires JSON Output response_format")
        if self.stream is not False:
            raise ValueError("stream must be false for strict parser accounting")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.concurrency_limit < 1 or self.concurrency_limit > 500:
            raise ValueError("DeepSeek V4 Pro concurrency_limit must be between 1 and 500")


@dataclass(frozen=True)
class AdaptiveBoFMiningPolicy:
    selected_target: int = 4_000
    max_attempted_documents: int = 25_000
    budget_cap_usd: float = 8.0
    minimum_balance_reserve_usd: float = 30.0
    target_concurrency: int = 128
    concurrency_ramp: tuple[int, ...] = (16, 64, 128)
    tranche_documents: int = 512
    private_refetch_concurrency: int = 128
    policy_version: str = ADAPTIVE_BOF_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.selected_target < 1:
            raise ValueError("adaptive BoF selected_target must be positive")
        if self.max_attempted_documents < 1 or self.max_attempted_documents > 25_000:
            raise ValueError("adaptive BoF max_attempted_documents must be between 1 and 25000")
        if self.selected_target > self.max_attempted_documents:
            raise ValueError("adaptive BoF selected_target cannot exceed max_attempted_documents")
        if self.budget_cap_usd <= 0 or self.budget_cap_usd > 65.0:
            raise ValueError("adaptive BoF budget cap must be positive and no more than USD 65.00")
        if self.minimum_balance_reserve_usd < 0.0:
            raise ValueError("adaptive BoF minimum balance reserve must be non-negative")
        if self.target_concurrency < 1 or self.target_concurrency > 500:
            raise ValueError("adaptive BoF target_concurrency must be between 1 and 500")
        if not self.concurrency_ramp:
            raise ValueError("adaptive BoF concurrency_ramp must not be empty")
        if any(step < 1 or step > self.target_concurrency for step in self.concurrency_ramp):
            raise ValueError("adaptive BoF concurrency_ramp entries must be between 1 and target_concurrency")
        if self.concurrency_ramp[-1] != self.target_concurrency:
            raise ValueError("adaptive BoF concurrency_ramp must end at target_concurrency")
        if self.tranche_documents < 1:
            raise ValueError("adaptive BoF tranche_documents must be positive")
        if self.private_refetch_concurrency < 1 or self.private_refetch_concurrency > 500:
            raise ValueError("adaptive BoF private_refetch_concurrency must be between 1 and 500")
        if self.policy_version != ADAPTIVE_BOF_POLICY_VERSION:
            raise ValueError(f"adaptive BoF policy_version must be {ADAPTIVE_BOF_POLICY_VERSION}")


@dataclass(frozen=True)
class FinalLabelBatchPolicy:
    final_label_target: int = TEACHER_LABEL_FINAL_TARGET
    budget_cap_usd: float = 40.0
    minimum_balance_reserve_usd: float = 0.0
    target_concurrency: int = 200
    tranche_documents: int = 512
    private_refetch_concurrency: int = 128
    policy_version: str = FINAL_LABEL_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.final_label_target != TEACHER_LABEL_FINAL_TARGET:
            raise ValueError("final label batch target must be exactly 35000")
        if self.budget_cap_usd <= 0 or self.budget_cap_usd > 100.0:
            raise ValueError("final label budget cap must be positive and no more than USD 100.00")
        if self.minimum_balance_reserve_usd < 0.0:
            raise ValueError("final label minimum balance reserve must be non-negative")
        if self.target_concurrency < 1 or self.target_concurrency > 500:
            raise ValueError("final label target_concurrency must be between 1 and 500")
        if self.tranche_documents < 1:
            raise ValueError("final label tranche_documents must be positive")
        if self.private_refetch_concurrency < 1 or self.private_refetch_concurrency > 500:
            raise ValueError("final label private_refetch_concurrency must be between 1 and 500")
        if self.policy_version != FINAL_LABEL_POLICY_VERSION:
            raise ValueError(f"final label policy_version must be {FINAL_LABEL_POLICY_VERSION}")


@dataclass(frozen=True)
class PriceSnapshot:
    source: str
    captured_at_utc: str
    model: str
    input_cache_miss_per_million: float
    input_cache_hit_per_million: float
    output_per_million: float
    currency: str = "USD"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash(payload: Any) -> str:
    return stable_json_hash(payload)


class TeacherSchemaValidationError(ValueError):
    """Strict parser failure with no raw model-provided value retention."""

    def __init__(
        self,
        message: str,
        *,
        path: str,
        kind: str,
        rule: str,
        value: Any | None = None,
        missing_keys: list[str] | None = None,
        extra_keys: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.kind = kind
        self.rule = rule
        self.value_sha256 = _hash(value) if value is not None else None
        self.missing_keys = missing_keys
        self.extra_key_sha256s = [_hash(key) for key in extra_keys] if extra_keys is not None else None
        self.extra_key_count = len(extra_keys) if extra_keys is not None else None


def _contains_token_shaped_secret(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_token_shaped_secret(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_token_shaped_secret(child) for child in value)
    if isinstance(value, str):
        return bool(TOKEN_SHAPED_SECRET_RE.search(value))
    return False


def _config_payload(config: DeepSeekLabelingConfig) -> dict[str, Any]:
    thinking_payload = {"type": config.thinking}
    if config.thinking == "enabled":
        thinking_payload["reasoning_effort"] = config.reasoning_effort
    payload = {
        "provider_id": REQUIRED_DEEPSEEK_PROVIDER_ID,
        "model": config.model,
        "base_url": config.base_url,
        "endpoint_path": config.endpoint_path,
        "response_format": config.response_format,
        "max_output_tokens": config.max_output_tokens,
        "thinking": thinking_payload,
        "stream": config.stream,
        "timeout_seconds": config.request_timeout_seconds,
        "max_retries": config.max_retries,
        "concurrency_limit": config.concurrency_limit,
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    return payload


def _public_config_payload(config: DeepSeekLabelingConfig) -> dict[str, Any]:
    payload = _config_payload(config)
    payload["base_url_sha256"] = _sha256_text(str(payload.pop("base_url")))
    payload["base_url_redacted"] = True
    return payload


def _redact_price_snapshot_urls(payload: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload, sort_keys=True))
    price_snapshot = copied.get("price_snapshot")
    if isinstance(price_snapshot, dict):
        source = price_snapshot.get("source")
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            price_snapshot["source_sha256"] = _sha256_text(source)
            price_snapshot["source"] = "redacted_url"
            price_snapshot["source_redacted"] = True
    return copied


def validate_deepseek_provider_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the no-secret DeepSeek V4 Pro teacher-labeling policy."""

    if not isinstance(config, dict):
        raise ValueError("DeepSeek provider config must be a mapping")
    if _contains_token_shaped_secret(config):
        raise ValueError("DeepSeek provider config must not contain token-shaped secrets")
    provider = config.get("provider")
    model = config.get("model")
    runtime = config.get("runtime")
    budget = config.get("budget")
    pricing = config.get("pricing_usd_per_1m_tokens")
    if not all(isinstance(section, dict) for section in (provider, model, runtime, budget, pricing)):
        raise ValueError("DeepSeek provider config requires provider, model, runtime, budget, and pricing sections")
    if provider.get("id") != REQUIRED_DEEPSEEK_PROVIDER_ID:
        raise ValueError("provider.id must be deepseek")
    if provider.get("base_url") != DEEPSEEK_BASE_URL:
        raise ValueError(f"provider.base_url must be {DEEPSEEK_BASE_URL}")
    if provider.get("endpoint_path") != DEEPSEEK_ENDPOINT_PATH:
        raise ValueError(f"provider.endpoint_path must be {DEEPSEEK_ENDPOINT_PATH}")
    if provider.get("api_key_env") != "DEEPSEEK_API_KEY":
        raise ValueError("provider.api_key_env must be DEEPSEEK_API_KEY")
    if any(key in provider for key in ("api_key", "secret", "token", "authorization")):
        raise ValueError("DeepSeek provider config must not contain secrets")
    if model.get("id") != REQUIRED_DEEPSEEK_TEACHER_MODEL:
        raise ValueError(f"teacher model must be {REQUIRED_DEEPSEEK_TEACHER_MODEL}")
    if int(model.get("max_output_tokens") or 0) < 128:
        raise ValueError("max_output_tokens must be at least 128")
    thinking = model.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") not in {"enabled", "disabled"}:
        raise ValueError("DeepSeek V4 Pro teacher config requires thinking.type enabled or disabled")
    if thinking.get("type") == "enabled" and thinking.get("reasoning_effort") not in {"high", "max"}:
        raise ValueError("DeepSeek V4 Pro teacher config requires reasoning_effort high or max when thinking is enabled")
    if thinking.get("type") == "disabled" and thinking.get("reasoning_effort") not in {None, "", "none"}:
        raise ValueError("DeepSeek V4 Pro teacher config must not set reasoning_effort when thinking is disabled")
    if model.get("response_format") != {"type": "json_object"}:
        raise ValueError("DeepSeek teacher config requires JSON Output response_format")
    if model.get("stream") is not False:
        raise ValueError("DeepSeek teacher config requires stream=false")
    if thinking.get("type") == "enabled":
        if model.get("temperature") is not None:
            raise ValueError("DeepSeek thinking mode does not support temperature")
        temperature: float | None = None
    else:
        if model.get("temperature") is None:
            raise ValueError("DeepSeek non-thinking teacher config requires temperature")
        temperature = float(model.get("temperature"))
        if temperature < 0 or temperature > 0.3:
            raise ValueError("DeepSeek teacher config requires low/noncreative temperature")
    if int(runtime.get("timeout_seconds") or 0) < 1:
        raise ValueError("runtime.timeout_seconds must be positive")
    if int(runtime.get("max_retries") or -1) < 0:
        raise ValueError("runtime.max_retries must be non-negative")
    if int(runtime.get("concurrency_limit") or 0) < 1 or int(runtime.get("concurrency_limit")) > 500:
        raise ValueError("DeepSeek V4 Pro concurrency_limit must be between 1 and 500")
    if runtime.get("default_live_enabled") is not False:
        raise ValueError("default_live_enabled must be false")
    if int(budget.get("pilot_min_documents") or 0) < 1:
        raise ValueError("budget.pilot_min_documents must be positive")
    if int(budget.get("pilot_max_documents") or 0) < int(budget.get("pilot_min_documents") or 0):
        raise ValueError("pilot_max_documents must be >= pilot_min_documents")
    if int(budget.get("pilot_max_documents") or 0) > 100:
        raise ValueError("pilot_max_documents must not exceed 100 for the first pilot")
    if int(budget.get("final_label_target") or 0) != TEACHER_LABEL_FINAL_TARGET:
        raise ValueError("budget.final_label_target must be 35000")
    if float(budget.get("budget_cap_usd") or 0.0) <= 0:
        raise ValueError("budget.budget_cap_usd must be positive")
    for key in ("input_cache_hit", "input_cache_miss", "output"):
        if float(pricing.get(key) or 0.0) <= 0:
            raise ValueError(f"pricing_usd_per_1m_tokens.{key} must be positive")

    normalized_model = {
        "id": model["id"],
        "thinking": {
            "type": thinking["type"],
            "reasoning_effort": str(thinking.get("reasoning_effort") or ""),
        },
        "max_output_tokens": int(model["max_output_tokens"]),
        "response_format": dict(model["response_format"]),
        "stream": False,
    }
    if temperature is not None:
        normalized_model["temperature"] = temperature

    normalized = {
        "provider": {
            "id": provider["id"],
            "base_url": provider["base_url"],
            "endpoint_path": provider["endpoint_path"],
            "api_key_env": provider["api_key_env"],
        },
        "model": normalized_model,
        "runtime": {
            "timeout_seconds": int(runtime["timeout_seconds"]),
            "max_retries": int(runtime["max_retries"]),
            "concurrency_limit": int(runtime["concurrency_limit"]),
            "default_live_enabled": False,
        },
        "budget": {
            "pilot_min_documents": int(budget["pilot_min_documents"]),
            "pilot_max_documents": int(budget["pilot_max_documents"]),
            "final_label_target": int(budget["final_label_target"]),
            "budget_cap_usd": float(budget["budget_cap_usd"]),
            "reserved_credit_usd": float(budget.get("reserved_credit_usd", 0.0)),
        },
        "pricing_usd_per_1m_tokens": {
            "input_cache_hit": float(pricing["input_cache_hit"]),
            "input_cache_miss": float(pricing["input_cache_miss"]),
            "output": float(pricing["output"]),
        },
    }
    normalized["config_hash"] = _hash(normalized)
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rng(seed: int, *parts: Any) -> random.Random:
    digest = _hash([seed, *parts])
    return random.Random(int(digest[:16], 16))


def _stable_shuffle(rows: list[dict[str, Any]], *, seed: int, stratum: str) -> list[dict[str, Any]]:
    keyed = [
        (_hash([seed, stratum, row["candidate_id"], row.get("global_schedule_index")]), row)
        for row in rows
    ]
    keyed.sort(key=lambda item: item[0])
    return [row for _, row in keyed]


def _reject_raw_text_leakage(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in RAW_TEXT_KEYS:
                raise ValueError(f"raw text field is not allowed in teacher manifest input: {path}.{key}")
            _reject_raw_text_leakage(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_raw_text_leakage(child, path=f"{path}[{index}]")


def _is_allowed_source_locator_url_path(path: str) -> bool:
    return bool(SOURCE_LOCATOR_URL_PATH_RE.fullmatch(path))


def _reject_live_artifact_leakage(value: Any, *, path: str = "$", allow_source_locator_urls: bool = False) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).casefold()
            child_path = f"{path}.{key}"
            if lowered in RAW_TEXT_KEYS:
                raise ValueError(f"raw text field is not allowed in teacher live artifacts: {child_path}")
            if lowered in RAW_URL_KEYS and not (
                allow_source_locator_urls and _is_allowed_source_locator_url_path(child_path)
            ):
                raise ValueError(f"raw URL field is not allowed in teacher live artifacts: {child_path}")
            _reject_live_artifact_leakage(child, path=child_path, allow_source_locator_urls=allow_source_locator_urls)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_live_artifact_leakage(child, path=f"{path}[{index}]", allow_source_locator_urls=allow_source_locator_urls)
    elif isinstance(value, str):
        lowered = value.casefold()
        if lowered.startswith(("http://", "https://")):
            if allow_source_locator_urls and _is_allowed_source_locator_url_path(path):
                return
            raise ValueError(f"raw URL value is not allowed in teacher live artifacts: {path}")
        if TOKEN_SHAPED_SECRET_RE.search(value):
            raise ValueError(f"secret-like value is not allowed in teacher live artifacts: {path}")


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("source_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _text_hash(row: dict[str, Any]) -> str:
    for key in ("text_hash", "raw_text_sha256", "normalized_text_sha256"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("decision row is missing a text hash")


def _doc_id(row: dict[str, Any]) -> str:
    for key in ("source_doc_id", "document_id", "id"):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    return _text_hash(row)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _private_refetch_text_hash(candidate: dict[str, Any], private_text: str) -> tuple[str, str, bool]:
    expected_hash = candidate.get("text_hash") or candidate.get("text_hash_sha256")
    if not expected_hash:
        return _sha256_text(private_text), TEXT_HASH_ALGORITHM_UTF8_SHA256_V1, False
    algorithm = candidate.get("text_hash_algorithm")
    if not algorithm:
        raise ValueError("candidate text_hash_algorithm is required for precomputed text hash validation")
    algorithm = str(algorithm)
    if algorithm == TEXT_HASH_ALGORITHM_UTF8_SHA256_V1:
        computed_hash = _sha256_text(private_text)
    elif algorithm == LEGACY_PRIVATE_MINING_TEXT_HASH_ALGORITHM:
        computed_hash = _hash(["raw_text", private_text])
    else:
        raise ValueError(f"unsupported candidate text_hash_algorithm: {algorithm}")
    return computed_hash, algorithm, True


def _candidate_source_identity_hash(candidate: dict[str, Any]) -> str:
    value = candidate.get("source_identity_hash")
    if isinstance(value, str) and value:
        return value
    identity_payload = {
        "dataset_repo_id": candidate.get("dataset_repo_id"),
        "dataset_config": candidate.get("dataset_config"),
        "dataset_split": candidate.get("dataset_split"),
        "resolved_repo_sha": candidate.get("resolved_repo_sha"),
        "source_metadata_hash": candidate.get("source_metadata_hash"),
        "source_schedule_hash": candidate.get("source_schedule_hash"),
        "schedule_entry_hash": candidate.get("schedule_entry_hash"),
        "global_schedule_index": candidate.get("global_schedule_index"),
        "schedule_pass_index": candidate.get("schedule_pass_index"),
        "source_file_id": candidate.get("source_file_id"),
        "parquet_identity_hash": candidate.get("parquet_identity_hash") or _hash(candidate.get("parquet_identity")),
        "row_group_index": candidate.get("row_group_index"),
        "row_group_local_row_index": candidate.get("row_group_local_row_index"),
        "deterministic_row_locator_hash": _hash(candidate.get("deterministic_row_locator")),
    }
    return _hash(identity_payload)


def _candidate_source_doc_id(candidate: dict[str, Any]) -> tuple[str, str]:
    for key in ("source_doc_id", "document_id", "id"):
        value = candidate.get(key)
        if value is not None and str(value):
            return str(value), f"provided:{key}"
    identity_hash = _candidate_source_identity_hash(candidate)
    return f"source-row-{identity_hash}", "derived_from_source_row_identity"


def _candidate_text_hash(
    candidate: dict[str, Any],
    *,
    private_text: str | None = None,
    allow_pending: bool = False,
) -> tuple[str | None, str]:
    for key in ("text_hash", "text_hash_sha256", "raw_text_sha256", "normalized_text_sha256", "normalized_text_hash_sha256"):
        value = candidate.get(key)
        if isinstance(value, str) and value:
            return value, f"provided:{key}"
    if private_text is not None:
        if not isinstance(private_text, str) or not private_text:
            raise ValueError("private_text must be a non-empty string when computing text hash")
        return _sha256_text(private_text), "computed_from_private_text"
    if allow_pending:
        return None, "pending_private_refetch"
    raise ValueError("candidate text hash is missing until private refetch")


def _candidate_label_strata(candidate: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    explicit = candidate.get("candidate_strata")
    if isinstance(explicit, (list, tuple)):
        values.extend(str(value) for value in explicit if str(value))
    for key in ("bucket_id", "subbucket_id", "selection_reason_id"):
        value = candidate.get(key)
        if value is not None and str(value):
            values.append(str(value))
    return tuple(sorted(set(values)))


def _source_manifest_type(candidates: list[dict[str, Any]]) -> str:
    if any("deterministic_row_locator" in candidate or "bucket_id" in candidate for candidate in candidates):
        return "source_row_candidate_distribution"
    return "decision_log_teacher_candidates"


def _candidate_public_labeling_record(candidate: dict[str, Any]) -> dict[str, Any]:
    source_doc_id, source_doc_id_source = _candidate_source_doc_id(candidate)
    text_hash, text_hash_status = _candidate_text_hash(candidate, allow_pending=True)
    candidate_strata = _candidate_label_strata(candidate)
    return {
        "candidate_id": candidate["candidate_id"],
        "source_doc_id": source_doc_id,
        "source_doc_id_source": source_doc_id_source,
        "source_identity_hash": _candidate_source_identity_hash(candidate),
        "text_hash": text_hash,
        "text_hash_status": text_hash_status,
        "label_unit": candidate.get("label_unit", "document"),
        "decision_status": candidate.get("decision_status"),
        "global_schedule_index": candidate.get("global_schedule_index"),
        "source_file_id": candidate.get("source_file_id"),
        "source_schedule_hash": candidate.get("source_schedule_hash"),
        "schedule_entry_hash": candidate.get("schedule_entry_hash"),
        "row_group_index": candidate.get("row_group_index"),
        "row_group_local_row_index": candidate.get("row_group_local_row_index"),
        "bucket_id": candidate.get("bucket_id"),
        "subbucket_id": candidate.get("subbucket_id"),
        "selection_reason_id": candidate.get("selection_reason_id"),
        "candidate_strata": list(candidate_strata),
        "metadata_bands": candidate.get("metadata_bands", {}),
        "candidate_hash": candidate.get("candidate_hash") or _hash(candidate),
        "private_refetch_required_before_live_call": text_hash_status == "pending_private_refetch",
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _band_numeric(value: float | int | None, cuts: tuple[float, ...], labels: tuple[str, ...]) -> str:
    if value is None:
        return "missing"
    for cut, label in zip(cuts, labels, strict=False):
        if value <= cut:
            return label
    return labels[-1]


def _metadata_bands(row: dict[str, Any]) -> dict[str, str]:
    metadata = _metadata(row)
    return {
        "token_count_band": _band_numeric(
            _int_or_none(metadata.get("token_count")),
            (128, 512, 2_048, 8_192),
            ("tiny", "short", "medium", "long", "very_long"),
        ),
        "language_score_band": _band_numeric(
            _float_or_none(metadata.get("language_score")),
            (0.75, 0.9, 0.98),
            ("low", "medium", "high", "very_high"),
        ),
        "score_band": _band_numeric(
            _float_or_none(metadata.get("score")),
            (2.0, 4.0, 5.0),
            ("low", "medium", "high", "very_high"),
        ),
    }


def _risk_tags(row: dict[str, Any]) -> set[str]:
    tags = set(str(tag) for tag in row.get("risk_tags", []) if str(tag))
    tags.update(str(reason) for reason in row.get("rejection_reasons", []) if str(reason))
    for hit in row.get("hits", []) if isinstance(row.get("hits"), list) else []:
        if isinstance(hit, dict):
            for key in ("category", "rule_id", "reason"):
                value = hit.get(key)
                if value:
                    tags.add(str(value))
    return {tag.casefold() for tag in tags}


def _near_threshold(row: dict[str, Any], margin: float) -> bool:
    scores = row.get("filter_scores")
    if not isinstance(scores, dict):
        return "near_threshold" in _risk_tags(row)
    risk_score = _float_or_none(scores.get("risk_score"))
    threshold = _float_or_none(scores.get("reject_threshold"))
    if risk_score is None or threshold is None:
        return "near_threshold" in _risk_tags(row)
    return abs(risk_score - threshold) <= margin


def _window_summary(row: dict[str, Any]) -> dict[str, Any]:
    summary = row.get("window_summary")
    return summary if isinstance(summary, dict) else {}


def _wrapper(row: dict[str, Any]) -> str:
    metadata = _metadata(row)
    wrapper = metadata.get("source_wrapper") or metadata.get("wrapper") or row.get("source_wrapper")
    return str(wrapper or "unknown").casefold()


def _strata_for_row(row: dict[str, Any], config: CandidateSelectionConfig) -> tuple[str, ...]:
    tags = _risk_tags(row)
    joined_tags = " ".join(tags)
    status = str(row.get("decision_status", "")).casefold()
    wrapper = _wrapper(row)
    windows = _window_summary(row)
    failed_windows = _int_or_none(windows.get("failed_window_count")) or 0
    window_count = _int_or_none(windows.get("window_count")) or 0
    hit_count = _int_or_none(row.get("hit_count"))
    strata: set[str] = {"domain_diversity", "metadata_bands"}

    if status == "rejected":
        strata.add("deterministic_rejects")
    if _near_threshold(row, config.near_threshold_margin):
        strata.add("near_threshold_cases")
    if status in {"pending_required_gates", "quarantined"}:
        strata.add("pending_survivors")
    if wrapper in {
        "faq",
        "forum",
        "support",
        "help_center",
        "knowledge_base",
        "interview",
        "press_release",
        "newswire",
        "q_and_a",
        "review",
    } or any("wrapper" in tag for tag in tags):
        strata.add("source_wrappers")
    if any(term in joined_tags for term in ("soft", "stance", "agentive", "agency", "seem", "may", "might")):
        strata.add("soft_stance_agentive")
    if any(term in joined_tags for term in ("ai", "assistant", "chatbot", "model", "anthropomorphism", "personhood")):
        strata.add("ai_anthropomorphism")
    if any(term in joined_tags for term in ("mental", "thought", "believ", "feel", "dream", "memory", "uncertainty")):
        strata.add("mental_state_ambiguity")
    if any(term in joined_tags for term in ("fiction", "dialogue", "narrative", "poem", "lyrics", "quote")):
        strata.add("fiction_dialogue")
    if wrapper in {"faq", "forum", "support", "help_center", "knowledge_base", "community"} or any(
        term in joined_tags
        for term in (
            "faq",
            "forum",
            "support",
            "help_center",
            "knowledge_base",
            "community",
            "troubleshooting",
        )
    ):
        strata.add("support_faq_forums")
    if any(term in joined_tags for term in ("safe_technical", "technical_control", "external_world")) or (
        status == "pending_required_gates" and wrapper in {"reference", "technical", "documentation"}
    ) or (
        status == "pending_required_gates" and (hit_count == 0 or (hit_count is None and not tags))
    ):
        strata.add("safe_technical_controls")
    if failed_windows == 1 and window_count >= 5:
        strata.add("long_document_one_bad_window")

    return tuple(sorted(strata))


def _candidate_from_row(row: dict[str, Any], config: CandidateSelectionConfig) -> dict[str, Any]:
    metadata = _metadata(row)
    text_hash = _text_hash(row)
    source_doc_id = _doc_id(row)
    token_count = _int_or_none(metadata.get("token_count"))
    label_unit = (
        "window_bundle"
        if token_count is not None and token_count > config.max_full_document_tokens
        else "document"
    )
    candidate = {
        "candidate_schema_version": 1,
        "candidate_id": _hash(["teacher_candidate", source_doc_id, text_hash]),
        "source_doc_id": source_doc_id,
        "text_hash": text_hash,
        "label_unit": label_unit,
        "decision_status": row.get("decision_status"),
        "decision_stage": row.get("decision_stage"),
        "global_schedule_index": row.get("global_schedule_index"),
        "source_file_id": row.get("source_file_id"),
        "source_schedule_hash": row.get("source_schedule_hash"),
        "schedule_entry_hash": row.get("schedule_entry_hash"),
        "row_group_index": row.get("row_group_index"),
        "row_group_local_row_index": row.get("row_group_local_row_index"),
        "source_domain": str(metadata.get("domain") or "unknown").casefold(),
        "dump": metadata.get("dump"),
        "language": metadata.get("language"),
        "source_wrapper": _wrapper(row),
        "metadata_bands": _metadata_bands(row),
        "risk_tags": sorted(_risk_tags(row)),
        "candidate_strata": _strata_for_row(row, config),
        "teacher_label_use": "classifier_training_and_audit_only",
    }
    candidate["candidate_hash"] = _hash(candidate)
    return candidate


def _take_with_caps(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    *,
    limit: int,
    max_per_domain: int,
    max_per_dump: int,
) -> list[dict[str, Any]]:
    selected_ids = {row["candidate_id"] for row in selected}
    domain_counts: dict[str, int] = {}
    dump_counts: dict[str, int] = {}
    for row in selected:
        domain_counts[row["source_domain"]] = domain_counts.get(row["source_domain"], 0) + 1
        dump = str(row.get("dump") or "missing")
        dump_counts[dump] = dump_counts.get(dump, 0) + 1

    appended: list[dict[str, Any]] = []
    for row in pool:
        if len(appended) >= limit:
            break
        if row["candidate_id"] in selected_ids:
            continue
        domain = row["source_domain"]
        dump = str(row.get("dump") or "missing")
        if domain_counts.get(domain, 0) >= max_per_domain:
            continue
        if dump_counts.get(dump, 0) >= max_per_dump:
            continue
        selected_ids.add(row["candidate_id"])
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        dump_counts[dump] = dump_counts.get(dump, 0) + 1
        selected.append(row)
        appended.append(row)
    return appended


def build_teacher_candidate_manifest(
    decision_rows: list[dict[str, Any]],
    *,
    config: CandidateSelectionConfig | None = None,
    source_decision_log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic no-raw-text candidate manifest from decision rows."""

    config = config or CandidateSelectionConfig()
    if config.max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    for row in decision_rows:
        _reject_raw_text_leakage(row)

    candidates_by_id: dict[str, dict[str, Any]] = {}
    for row in decision_rows:
        candidate = _candidate_from_row(row, config)
        candidates_by_id[candidate["candidate_id"]] = candidate
    all_candidates = list(candidates_by_id.values())

    strata_pools = {
        stratum: [candidate for candidate in all_candidates if stratum in candidate["candidate_strata"]]
        for stratum in REQUIRED_STRATA
    }

    selected: list[dict[str, Any]] = []
    quota_report: dict[str, Any] = {
        "requested": {stratum: config.quota_for(stratum) for stratum in REQUIRED_STRATA},
        "available": {stratum: len(pool) for stratum, pool in strata_pools.items()},
        "selected_by_stratum": {},
        "required_strata_missing": [],
        "required_strata_unavailable": [],
        "quota_shortfalls": {},
        "candidate_count_shortfall": 0,
    }

    for stratum in REQUIRED_STRATA:
        need = min(config.quota_for(stratum), config.max_candidates - len(selected))
        if need <= 0:
            break
        pool = _stable_shuffle(strata_pools[stratum], seed=config.seed, stratum=stratum)
        added = _take_with_caps(
            selected,
            pool,
            limit=need,
            max_per_domain=config.max_per_domain,
            max_per_dump=config.max_per_dump,
        )
        quota_report["selected_by_stratum"][stratum] = len(
            [row for row in selected if stratum in row["candidate_strata"]]
        )
        if not added and quota_report["selected_by_stratum"][stratum] == 0:
            quota_report["required_strata_missing"].append(stratum)

    if len(selected) < config.max_candidates:
        refill = _stable_shuffle(all_candidates, seed=config.seed, stratum="refill")
        _take_with_caps(
            selected,
            refill,
            limit=config.max_candidates - len(selected),
            max_per_domain=config.max_per_domain,
            max_per_dump=config.max_per_dump,
        )

    selected = selected[: config.max_candidates]
    quota_report["selected_by_stratum"] = {
        stratum: len([row for row in selected if stratum in row["candidate_strata"]])
        for stratum in REQUIRED_STRATA
    }
    quota_report["required_strata_missing"] = [
        stratum
        for stratum in REQUIRED_STRATA
        if quota_report["available"][stratum] > 0 and quota_report["selected_by_stratum"][stratum] == 0
    ]
    quota_report["required_strata_unavailable"] = [
        stratum
        for stratum in REQUIRED_STRATA
        if quota_report["available"][stratum] == 0
    ]
    quota_report["candidate_count_shortfall"] = max(0, config.max_candidates - len(selected))
    quota_report["quota_shortfalls"] = {
        stratum: {
            "needed": quota_report["requested"][stratum],
            "available": quota_report["available"][stratum],
            "selected": quota_report["selected_by_stratum"][stratum],
            "shortfall": max(0, quota_report["requested"][stratum] - quota_report["selected_by_stratum"][stratum]),
        }
        for stratum in REQUIRED_STRATA
        if (
            quota_report["available"][stratum] == 0
            or quota_report["selected_by_stratum"][stratum] < quota_report["requested"][stratum]
        )
    }
    required_actions: list[str] = []
    if quota_report["candidate_count_shortfall"] > 0:
        required_actions.append("collect_more_scheduled_decision_rows")
    if quota_report["required_strata_unavailable"] or quota_report["quota_shortfalls"]:
        required_actions.append("collect_supplemental_candidates")
    if not selected:
        required_actions.append("do_not_run_paid_labeling_without_candidates")
    label_readiness = {
        "ready_for_paid_labeling": not required_actions,
        "readiness_policy": "all_requested_count_and_required_strata_quotas_must_be_satisfied",
        "required_actions": required_actions,
        "notes": [
            "Teacher labels train classifiers and audit gates only; they never approve corpus rows.",
            "Unavailable strata mean the source audit did not produce examples for that category.",
        ],
    }
    selection_coverage = {
        "candidate_count": len(selected),
        "domains": sorted({row["source_domain"] for row in selected}),
        "dumps": sorted({str(row.get("dump") or "missing") for row in selected}),
        "languages": sorted({str(row.get("language") or "missing") for row in selected}),
        "token_count_bands": sorted({row["metadata_bands"]["token_count_band"] for row in selected}),
        "language_score_bands": sorted({row["metadata_bands"]["language_score_band"] for row in selected}),
        "score_bands": sorted({row["metadata_bands"]["score_band"] for row in selected}),
        "min_global_schedule_index": min((row.get("global_schedule_index") for row in selected), default=None),
        "max_global_schedule_index": max((row.get("global_schedule_index") for row in selected), default=None),
    }

    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "selection_algorithm_version": "teacher_candidate_distribution_v1",
        "seed": config.seed,
        "max_candidates": config.max_candidates,
        "selection_note": config.manifest_note,
        "raw_text_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "quota_report": quota_report,
        "label_readiness": label_readiness,
        "selection_coverage": selection_coverage,
        "candidates": selected,
    }
    if source_decision_log is not None:
        manifest["source_decision_log"] = dict(source_decision_log)
    stable = {key: value for key, value in manifest.items() if key != "generated_at_utc"}
    manifest["manifest_hash"] = _hash(stable)
    validate_candidate_manifest(manifest)
    return manifest


def build_seed_label_candidate_manifest(
    source_manifest: dict[str, Any],
    *,
    seed_label_target: int = TEACHER_LABEL_SEED_TARGET,
    train_label_count: int = TEACHER_LABEL_SEED_TRAIN_TARGET,
    holdout_label_count: int = TEACHER_LABEL_SEED_HOLDOUT_TARGET,
    seed: int = 20260603,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Derive an exact seed-label manifest without claiming final 35k readiness."""

    validate_candidate_manifest(source_manifest)
    if seed_label_target < 1:
        raise ValueError("seed_label_target must be positive")
    if train_label_count < 1 or holdout_label_count < 1:
        raise ValueError("train and holdout counts must be positive")
    if train_label_count + holdout_label_count != seed_label_target:
        raise ValueError("train and holdout counts must sum to seed_label_target")
    candidates = source_manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("source manifest candidates must be a list")
    if len(candidates) < seed_label_target:
        raise ValueError("not enough candidates for exact seed label target")
    selected = _select_label_batch_candidates(candidates, label_count=seed_label_target, seed=seed)
    if len(selected) != seed_label_target:
        raise ValueError("not enough candidates for exact seed label target")

    public_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        public = _candidate_public_labeling_record(candidate)
        split = "provisional_train" if index < train_label_count else "labeled_holdout"
        public["seed_split"] = split
        public["seed_split_index"] = index if split == "provisional_train" else index - train_label_count
        public["seed_selection_index"] = index
        public_candidates.append(public)

    split = {
        "split_schema_version": 1,
        "assignment_algorithm": "deterministic_selected_order_train_then_holdout",
        "seed": seed,
        "seed_label_target": seed_label_target,
        "provisional_training_labels": train_label_count,
        "labeled_holdout_labels": holdout_label_count,
        "provisional_train_candidate_ids": [
            candidate["candidate_id"] for candidate in public_candidates[:train_label_count]
        ],
        "labeled_holdout_candidate_ids": [
            candidate["candidate_id"] for candidate in public_candidates[train_label_count:]
        ],
    }
    strata_counts: dict[str, int] = {}
    for candidate in public_candidates:
        for stratum in candidate.get("candidate_strata", []):
            strata_counts[str(stratum)] = strata_counts.get(str(stratum), 0) + 1
    decision_status_counts: dict[str, int] = {}
    for candidate in public_candidates:
        status = str(candidate.get("decision_status") or "missing")
        decision_status_counts[status] = decision_status_counts.get(status, 0) + 1
    private_refetch_required_count = len(
        [
            candidate
            for candidate in public_candidates
            if candidate.get("private_refetch_required_before_live_call") is True
            or candidate.get("text_hash_status") == "pending_private_refetch"
        ]
    )
    coverage = {
        "candidate_count": len(public_candidates),
        "domains": sorted({row.get("source_domain") for row in selected if row.get("source_domain")}),
        "dumps": sorted({str(row.get("dump") or "missing") for row in selected}),
        "languages": sorted({str(row.get("language") or "missing") for row in selected}),
        "token_count_bands": sorted(
            {
                str((row.get("metadata_bands") or {}).get("token_count_band") or "missing")
                for row in selected
            }
        ),
        "language_score_bands": sorted(
            {
                str((row.get("metadata_bands") or {}).get("language_score_band") or "missing")
                for row in selected
            }
        ),
        "score_bands": sorted(
            {
                str((row.get("metadata_bands") or {}).get("score_band") or "missing")
                for row in selected
            }
        ),
        "min_global_schedule_index": min((row.get("global_schedule_index") for row in selected), default=None),
        "max_global_schedule_index": max((row.get("global_schedule_index") for row in selected), default=None),
    }
    label_readiness = {
        "ready_for_paid_labeling": True,
        "readiness_policy": "exact_seed_count_and_split_from_source_manifest",
        "required_actions": [],
        "notes": [
            "This derived manifest is ready only for the exact DeepSeek seed-label batch.",
            "It does not make the source manifest final-35k ready and never approves corpus rows.",
        ],
    }
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": timestamp_utc or _utc_now_iso(),
        "selection_algorithm_version": "deepseek_seed_label_manifest_v1",
        "seed": seed,
        "max_candidates": seed_label_target,
        "selection_note": "exact_seed_labels_train_classifiers_and_holdout_only_not_corpus_approval",
        "raw_text_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "source_candidate_manifest_hash": source_manifest["manifest_hash"],
        "source_manifest_label_readiness": source_manifest.get("label_readiness", {}),
        "source_manifest_quota_report": source_manifest.get("quota_report", {}),
        "source_manifest_candidate_count": len(candidates),
        "quota_report": {
            "readiness_scope": "seed_6000_only",
            "requested_seed_count": seed_label_target,
            "selected_seed_count": len(public_candidates),
            "candidate_count_shortfall": 0,
            "selected_by_stratum": dict(sorted(strata_counts.items())),
            "selected_by_decision_status": dict(sorted(decision_status_counts.items())),
            "private_refetch_required_count": private_refetch_required_count,
        },
        "label_readiness": label_readiness,
        "seed_split": split,
        "selection_coverage": coverage,
        "candidates": public_candidates,
    }
    stable = {key: value for key, value in manifest.items() if key != "generated_at_utc"}
    manifest["manifest_hash"] = _hash(stable)
    validate_candidate_manifest(manifest)
    return manifest


def _read_decision_log_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    rows: list[dict[str, Any]] = []
    line_count = 0
    status_counts: dict[str, int] = {}
    source_schedule_hashes: set[str] = set()
    source_tree_hashes: set[str] = set()
    decision_log_config_hashes: set[str] = set()
    ontology_semantic_hashes: set[str] = set()
    run_ids: set[str] = set()

    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"decision log line {line_number} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"decision log line {line_number} is not a JSON object")
            _reject_raw_text_leakage(row)
            rows.append(row)
            line_count += 1
            status = str(row.get("decision_status") or "missing")
            status_counts[status] = status_counts.get(status, 0) + 1
            for key, target in (
                ("source_schedule_hash", source_schedule_hashes),
                ("source_tree_hash", source_tree_hashes),
                ("decision_log_config_hash", decision_log_config_hashes),
                ("ontology_semantic_hash", ontology_semantic_hashes),
                ("run_id", run_ids),
            ):
                value = row.get(key)
                if isinstance(value, str) and value:
                    target.add(value)
    if not rows:
        raise ValueError("decision log contains no rows")
    metadata = {
        "path_name": source.name,
        "line_count": line_count,
        "decision_log_sha256": _file_sha256(source),
        "status_counts": dict(sorted(status_counts.items())),
        "source_schedule_hashes": sorted(source_schedule_hashes),
        "source_tree_hashes": sorted(source_tree_hashes),
        "decision_log_config_hashes": sorted(decision_log_config_hashes),
        "ontology_semantic_hashes": sorted(ontology_semantic_hashes),
        "run_ids": sorted(run_ids),
    }
    return rows, metadata


def write_teacher_candidate_manifest_from_decision_log(
    decision_log_path: str | Path,
    output_path: str | Path,
    *,
    config: CandidateSelectionConfig | None = None,
) -> dict[str, Any]:
    rows, source_metadata = _read_decision_log_jsonl(decision_log_path)
    manifest = build_teacher_candidate_manifest(
        rows,
        config=config,
        source_decision_log=source_metadata,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return manifest


def write_seed_label_candidate_manifest(
    source_manifest_path: str | Path,
    output_path: str | Path,
    *,
    seed_label_target: int = TEACHER_LABEL_SEED_TARGET,
    train_label_count: int = TEACHER_LABEL_SEED_TRAIN_TARGET,
    holdout_label_count: int = TEACHER_LABEL_SEED_HOLDOUT_TARGET,
    seed: int = 20260603,
) -> dict[str, Any]:
    source_manifest = load_candidate_manifest(source_manifest_path)
    manifest = build_seed_label_candidate_manifest(
        source_manifest,
        seed_label_target=seed_label_target,
        train_label_count=train_label_count,
        holdout_label_count=holdout_label_count,
        seed=seed,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return manifest


def validate_candidate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != CANDIDATE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported teacher candidate manifest schema")
    if manifest.get("raw_text_serialized") is not False:
        raise ValueError("candidate manifest must not serialize raw text")
    if manifest.get("accepted_manifest") is not False:
        raise ValueError("teacher candidate manifest must not be an accepted corpus manifest")
    serialized = json.dumps(manifest, sort_keys=True)
    forbidden_fragments = ("raw_context_snippet", "matched_text", "document_text")
    if any(fragment in serialized for fragment in forbidden_fragments):
        raise ValueError("candidate manifest contains raw-text-like fields")
    ids = [candidate.get("candidate_id") for candidate in manifest.get("candidates", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate teacher candidate id")
    stable = {
        key: value
        for key, value in manifest.items()
        if key not in {"generated_at_utc", "manifest_hash"}
    }
    if manifest.get("manifest_hash") != _hash(stable):
        raise ValueError("teacher candidate manifest hash mismatch")


def load_candidate_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_candidate_manifest(manifest)
    return manifest


def _final_promotion_candidate_hash(candidate: dict[str, Any]) -> str:
    return _hash({key: value for key, value in candidate.items() if key != "candidate_hash"})


def _final_promotion_required_identity_missing(candidate: dict[str, Any]) -> bool:
    for key in (
        "source_schedule_hash",
        "schedule_entry_hash",
        "source_file_id",
        "row_group_index",
        "row_group_local_row_index",
        "physical_source_row_key_hash",
        "label_unit",
    ):
        if candidate.get(key) is None:
            return True
    try:
        return str(candidate.get("physical_source_row_key_hash") or "") != physical_source_row_key_hash(candidate)
    except ValueError:
        return True


def _final_strip_ambiguous_precomputed_text_hash(candidate: dict[str, Any]) -> bool:
    if not (candidate.get("text_hash") or candidate.get("text_hash_sha256")):
        return False
    if candidate.get("text_hash_algorithm"):
        return False
    for key in (
        "text_hash",
        "text_hash_sha256",
        "raw_text_sha256",
        "normalized_text_sha256",
        "normalized_text_hash_sha256",
        "normalized_text_hash_algorithm",
    ):
        candidate.pop(key, None)
    candidate["text_hash_status"] = "pending_private_refetch"
    candidate["private_refetch_required_before_live_call"] = True
    return True


def _duplicate_nonempty_count(values: list[str | None]) -> int:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if not value:
            continue
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return len(duplicates)


def _raise_final_promotion_blockers(blocker_counts: dict[str, int]) -> None:
    active = {key: value for key, value in sorted(blocker_counts.items()) if value}
    if active:
        details = ", ".join(f"{key}={value}" for key, value in active.items())
        raise ValueError(f"final 35k promotion failed closed: {details}")


def _final_promotion_attempt_lookup(attempt_manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    attempt_candidates = attempt_manifest.get("candidates")
    if not isinstance(attempt_candidates, list):
        raise ValueError("adaptive attempt manifest candidates must be a list")
    attempt_by_hash: dict[str, list[dict[str, Any]]] = {}
    for candidate in attempt_candidates:
        candidate_hash = str(candidate.get("candidate_hash") or _final_promotion_candidate_hash(candidate))
        attempt_by_hash.setdefault(candidate_hash, []).append(candidate)
    return attempt_by_hash


def _validate_final_review_artifact(review_artifact: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _reject_live_artifact_leakage(review_artifact)
    if review_artifact.get("artifact_scope") != "35k_teacher_label_candidate_review_not_final":
        raise ValueError("final promotion requires the diagnostic 35k review artifact")
    if review_artifact.get("diagnostic_only") is not True:
        raise ValueError("final promotion source must be diagnostic_only=true")
    for flag in (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "labels_are_corpus_approval",
        "training_shards_created",
        "paid_calls_made",
        "deepseek_calls_made",
    ):
        if review_artifact.get(flag) is not False:
            raise ValueError(f"final promotion source must set {flag}=false")
    sections = review_artifact.get("candidate_sections")
    if not isinstance(sections, dict):
        raise ValueError("final promotion source missing candidate_sections")
    non_borderline = sections.get("non_borderline_candidates")
    borderline = sections.get("borderline_candidates")
    if not isinstance(non_borderline, list) or not isinstance(borderline, list):
        raise ValueError("final promotion source candidate sections must be lists")
    counts = review_artifact.get("counts", {})
    if isinstance(counts, dict):
        if counts.get("non_borderline_total") != len(non_borderline):
            raise ValueError("final promotion non-borderline count drifted")
        if counts.get("borderline_count") != len(borderline):
            raise ValueError("final promotion borderline count drifted")
        if counts.get("teacher_candidate_total") != len(non_borderline) + len(borderline):
            raise ValueError("final promotion total count drifted")
    return non_borderline, borderline


def _promote_review_candidates_for_final_labeling(
    review_artifact: dict[str, Any],
    *,
    attempt_manifest: dict[str, Any],
    expected_candidate_count: int = TEACHER_LABEL_FINAL_TARGET,
) -> tuple[list[dict[str, Any]], dict[str, int], int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    _reject_live_artifact_leakage(attempt_manifest)
    non_borderline, borderline = _validate_final_review_artifact(review_artifact)
    attempt_by_hash = _final_promotion_attempt_lookup(attempt_manifest)
    blocker_counts = {
        "candidate_count_mismatch": 0,
        "ambiguous_attempt_join_count": 0,
        "missing_required_identity_count": 0,
        "duplicate_candidate_id_count": 0,
        "duplicate_physical_row_key_count": 0,
        "duplicate_text_hash_count": 0,
        "duplicate_normalized_text_hash_count": 0,
        "schedule_refetch_noncoverage_count": 0,
        "non_borderline_section_count_mismatch": 0,
        "borderline_section_count_mismatch": 0,
    }
    promoted: list[dict[str, Any]] = []
    ambiguous_precomputed_text_hash_stripped_count = 0

    if int(expected_candidate_count) == TEACHER_LABEL_FINAL_TARGET:
        if len(non_borderline) != 31_000:
            blocker_counts["non_borderline_section_count_mismatch"] = abs(31_000 - len(non_borderline)) or 1
        if len(borderline) != 4_000:
            blocker_counts["borderline_section_count_mismatch"] = abs(4_000 - len(borderline)) or 1

    for candidate in non_borderline:
        updated = dict(candidate)
        updated["teacher_label_use"] = "final_35k_freeze_labeling_runner_input"
        updated["final_manifest_source_section"] = "non_borderline"
        updated["labels_are_corpus_approval"] = False
        if _final_strip_ambiguous_precomputed_text_hash(updated):
            ambiguous_precomputed_text_hash_stripped_count += 1
        if _final_promotion_required_identity_missing(updated):
            blocker_counts["missing_required_identity_count"] += 1
        updated["candidate_hash"] = _final_promotion_candidate_hash(updated)
        promoted.append(updated)

    restored_borderline_count = 0
    for selected in borderline:
        join_hash = str(selected.get("attempt_candidate_hash") or selected.get("source_candidate_hash") or "")
        matches = attempt_by_hash.get(join_hash, [])
        if len(matches) != 1:
            blocker_counts["ambiguous_attempt_join_count"] += 1
            continue
        attempt_candidate = matches[0]
        updated = dict(attempt_candidate)
        updated["candidate_id"] = str(selected.get("candidate_id") or attempt_candidate.get("candidate_id") or "")
        updated["bucket_id"] = "active_learning_borderline"
        updated["subbucket_id"] = str(selected.get("subbucket_id") or attempt_candidate.get("subbucket_id") or "schedule_expansion")
        updated["label_unit"] = str(selected.get("label_unit") or attempt_candidate.get("label_unit") or "document")
        updated["teacher_label_use"] = "final_35k_freeze_labeling_runner_input"
        updated["final_manifest_source_section"] = "borderline"
        updated["source_attempt_candidate_hash"] = join_hash
        updated["source_selected_borderline_candidate_hash"] = selected.get("candidate_hash") or _hash(selected)
        updated["diagnostic_borderline_selection_provenance"] = {
            "source_manifest_scope": "adaptive_bof_borderline_candidate_selection",
            "complete_bof5_evidence": bool(selected.get("complete_bof5_evidence")),
            "diagnostic_provisional_not_final_truth": bool(selected.get("diagnostic_provisional_not_final_truth", True)),
            "final_label_truth_requires_bof3_freeze_labeling": True,
        }
        updated["labels_are_corpus_approval"] = False
        if _final_strip_ambiguous_precomputed_text_hash(updated):
            ambiguous_precomputed_text_hash_stripped_count += 1
        else:
            updated.pop("text_hash_status", None)
            updated.pop("private_refetch_required_before_live_call", None)
        if _final_promotion_required_identity_missing(updated):
            blocker_counts["missing_required_identity_count"] += 1
        updated["candidate_hash"] = _final_promotion_candidate_hash(updated)
        promoted.append(updated)
        restored_borderline_count += 1

    if len(promoted) != int(expected_candidate_count):
        blocker_counts["candidate_count_mismatch"] = abs(int(expected_candidate_count) - len(promoted)) or 1
    return (
        promoted,
        blocker_counts,
        restored_borderline_count,
        ambiguous_precomputed_text_hash_stripped_count,
        non_borderline,
        borderline,
    )


def _final_identity_value(candidate: dict[str, Any], field_name: str) -> str | None:
    if field_name == "candidate_id":
        return str(candidate.get("candidate_id") or "") or None
    if field_name == "physical_source_row_key_hash":
        return str(candidate.get("physical_source_row_key_hash") or "") or None
    if field_name == "text_hash":
        return str(candidate.get("text_hash_sha256") or candidate.get("text_hash") or "") or None
    if field_name == "normalized_text_hash":
        return str(candidate.get("normalized_text_hash_sha256") or "") or None
    raise ValueError(f"unsupported final identity field: {field_name}")


def _final_duplicate_components(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    duplicate_reasons_by_index: dict[int, set[str]] = {index: set() for index in range(len(candidates))}
    for field_name in ("candidate_id", "physical_source_row_key_hash", "text_hash", "normalized_text_hash"):
        seen: dict[str, int] = {}
        for index, candidate in enumerate(candidates):
            value = _final_identity_value(candidate, field_name)
            if not value:
                continue
            if value in seen:
                union(seen[value], index)
                duplicate_reasons_by_index[seen[value]].add(field_name)
                duplicate_reasons_by_index[index].add(field_name)
            else:
                seen[value] = index

    grouped: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        grouped.setdefault(find(index), []).append(index)
    components = []
    for indexes in grouped.values():
        if len(indexes) < 2:
            continue
        reasons = sorted({reason for index in indexes for reason in duplicate_reasons_by_index[index]})
        components.append({"indexes": sorted(indexes), "duplicate_reasons": reasons})
    components.sort(
        key=lambda component: (
            [str(candidates[index].get("candidate_id") or "") for index in component["indexes"]],
            component["duplicate_reasons"],
        )
    )
    return components


FINAL_PROMOTION_BUCKET_PRIORITY = {
    "active_learning_borderline": 0,
    "subtle_implicit_risk": 10,
    "source_wrapper_channel_risk": 11,
    "false_friend_hard_negatives": 12,
    "safe_non_stem_boundary_survivors": 13,
    "safe_stem_control": 14,
    "long_document_window_cases": 15,
    "deterministic_high_risk": 16,
    "representative_coverage": 20,
    "random_non_borderline_backfill": 30,
}


def _final_promotion_candidate_priority(candidate: dict[str, Any]) -> tuple[int, str, str]:
    section = str(candidate.get("final_manifest_source_section") or "")
    bucket = str(candidate.get("bucket_id") or "")
    section_priority = 0 if section == "borderline" or bucket == "active_learning_borderline" else 10
    bucket_priority = FINAL_PROMOTION_BUCKET_PRIORITY.get(bucket, 99)
    return (section_priority + bucket_priority, str(candidate.get("candidate_id") or ""), str(candidate.get("candidate_hash") or ""))


def _final_promotion_public_identity(candidate: dict[str, Any], *, reasons: list[str] | None = None) -> dict[str, Any]:
    payload = {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_hash": str(candidate.get("candidate_hash") or ""),
        "bucket_id": str(candidate.get("bucket_id") or ""),
        "subbucket_id": str(candidate.get("subbucket_id") or ""),
        "final_manifest_source_section": str(candidate.get("final_manifest_source_section") or ""),
        "physical_source_row_key_hash": str(candidate.get("physical_source_row_key_hash") or ""),
        "text_hash_sha256": str(candidate.get("text_hash_sha256") or candidate.get("text_hash") or ""),
        "normalized_text_hash_sha256": str(candidate.get("normalized_text_hash_sha256") or ""),
    }
    if reasons is not None:
        payload["duplicate_reasons"] = list(reasons)
    _reject_live_artifact_leakage(payload)
    return payload


def build_final_promotion_repair_plan(
    review_artifact: dict[str, Any],
    *,
    attempt_manifest: dict[str, Any],
    expected_candidate_count: int = TEACHER_LABEL_FINAL_TARGET,
) -> dict[str, Any]:
    """Build a no-raw deterministic keep/drop plan before final 35k promotion."""

    (
        promoted,
        blocker_counts,
        restored_borderline_count,
        ambiguous_precomputed_text_hash_stripped_count,
        non_borderline,
        borderline,
    ) = _promote_review_candidates_for_final_labeling(
        review_artifact,
        attempt_manifest=attempt_manifest,
        expected_candidate_count=expected_candidate_count,
    )
    components = _final_duplicate_components(promoted)
    keep_indexes: set[int] = set()
    drop_indexes: set[int] = set()
    component_records: list[dict[str, Any]] = []
    borderline_drop_count = 0
    for component_index, component in enumerate(components):
        indexes = list(component["indexes"])
        ranked = sorted(indexes, key=lambda index: _final_promotion_candidate_priority(promoted[index]))
        keep_index = ranked[0]
        component_keep_indexes = {keep_index}
        component_drop_indexes = set(indexes) - component_keep_indexes
        keep_indexes.update(component_keep_indexes)
        drop_indexes.update(component_drop_indexes)
        for index in component_drop_indexes:
            candidate = promoted[index]
            if candidate.get("final_manifest_source_section") == "borderline" or candidate.get("bucket_id") == "active_learning_borderline":
                borderline_drop_count += 1
        component_records.append(
            {
                "component_index": component_index,
                "candidate_count": len(indexes),
                "duplicate_reasons": list(component["duplicate_reasons"]),
                "kept_candidate_ids": [str(promoted[index].get("candidate_id") or "") for index in sorted(component_keep_indexes)],
                "dropped_candidate_ids": [str(promoted[index].get("candidate_id") or "") for index in sorted(component_drop_indexes)],
                "candidate_ids": [str(promoted[index].get("candidate_id") or "") for index in indexes],
            }
        )
    if borderline_drop_count:
        raise ValueError(
            "final promotion repair failed closed: "
            f"duplicate_component_requires_borderline_drop_count={borderline_drop_count}"
        )
    drops = [
        _final_promotion_public_identity(promoted[index], reasons=next(
            (component["duplicate_reasons"] for component in components if index in component["indexes"]),
            [],
        ))
        for index in sorted(drop_indexes)
    ]
    keeps = [
        _final_promotion_public_identity(promoted[index], reasons=next(
            (component["duplicate_reasons"] for component in components if index in component["indexes"]),
            [],
        ))
        for index in sorted(keep_indexes)
    ]
    drop_bucket_counts: dict[str, int] = {}
    for row in drops:
        bucket = str(row.get("bucket_id") or "")
        drop_bucket_counts[bucket] = drop_bucket_counts.get(bucket, 0) + 1
    plan = {
        "schema_version": FINAL_PROMOTION_REPAIR_PLAN_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "repair_plan_scope": FINAL_PROMOTION_REPAIR_PLAN_SCOPE,
        "source_artifact_hash": review_artifact.get("artifact_hash"),
        "attempt_manifest_hash": attempt_manifest.get("manifest_hash"),
        "expected_candidate_count": int(expected_candidate_count),
        "promoted_candidate_count_before_repair": len(promoted),
        "non_borderline_count_before_repair": len(non_borderline),
        "borderline_count_before_repair": len(borderline),
        "borderline_identity_restored_from_attempt_manifest_count": restored_borderline_count,
        "ambiguous_precomputed_text_hash_stripped_count": ambiguous_precomputed_text_hash_stripped_count,
        "pre_duplicate_blocker_counts": dict(sorted(blocker_counts.items())),
        "duplicate_component_count": len(components),
        "duplicate_components": component_records,
        "keep_count": len(keeps),
        "drop_count": len(drops),
        "replacement_required_count": len(drops),
        "replacement_bucket_id": "random_non_borderline_backfill",
        "drop_bucket_counts": dict(sorted(drop_bucket_counts.items())),
        "keeps": keeps,
        "drops": drops,
        "repair_policy": "keep_active_learning_borderline_then_scarce_targeted_then_representative_then_random_v1",
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "paid_calls_made": False,
        "deepseek_calls_made": False,
    }
    plan["repair_plan_hash"] = _hash({key: value for key, value in plan.items() if key not in {"generated_at_utc", "repair_plan_hash"}})
    _reject_live_artifact_leakage(plan)
    return plan


def _final_candidate_identity_sets(candidates: list[dict[str, Any]]) -> dict[str, set[str]]:
    sets = {
        "candidate_id": set(),
        "physical_source_row_key_hash": set(),
        "text_hash": set(),
        "normalized_text_hash": set(),
    }
    for candidate in candidates:
        for field_name in sets:
            value = _final_identity_value(candidate, field_name)
            if value:
                sets[field_name].add(value)
    return sets


def _final_replacement_candidates(
    replacement_candidates: list[dict[str, Any]],
    *,
    promoted: list[dict[str, Any]],
    drop_indexes: set[int],
    needed_count: int,
) -> tuple[list[dict[str, Any]], int]:
    if needed_count == 0:
        return [], 0
    excluded_sets = _final_candidate_identity_sets(promoted)
    keep_indexes = set(range(len(promoted))) - set(drop_indexes)
    keep_sets = _final_candidate_identity_sets([promoted[index] for index in keep_indexes])
    eligible: list[dict[str, Any]] = []
    skipped = 0
    seen_sets = {key: set() for key in excluded_sets}
    for raw_candidate in replacement_candidates:
        candidate = deepcopy(raw_candidate)
        _reject_live_artifact_leakage(candidate)
        if candidate.get("bucket_id") != "random_non_borderline_backfill":
            skipped += 1
            continue
        if _final_promotion_required_identity_missing(candidate):
            skipped += 1
            continue
        identity_values = {
            field_name: _final_identity_value(candidate, field_name)
            for field_name in excluded_sets
        }
        collides = False
        for field_name, value in identity_values.items():
            if not value:
                continue
            if value in keep_sets[field_name] or value in excluded_sets[field_name] or value in seen_sets[field_name]:
                collides = True
                break
        if collides:
            skipped += 1
            continue
        for field_name, value in identity_values.items():
            if value:
                seen_sets[field_name].add(value)
        updated = dict(candidate)
        updated["teacher_label_use"] = "final_35k_freeze_labeling_runner_input"
        updated["final_manifest_source_section"] = "non_borderline"
        updated["labels_are_corpus_approval"] = False
        updated["candidate_hash"] = _final_promotion_candidate_hash(updated)
        eligible.append(updated)
    eligible.sort(key=lambda candidate: (str(candidate.get("candidate_id") or ""), str(candidate.get("candidate_hash") or "")))
    if len(eligible) < needed_count:
        raise ValueError(
            "final promotion repair failed closed: "
            f"replacement_required_count={needed_count}, eligible_replacement_count={len(eligible)}"
        )
    return eligible[:needed_count], skipped


def _final_promotion_replacement_candidate_from_schedule_entry(
    entry: dict[str, Any],
    *,
    offset: int,
    schedule: dict[str, Any],
    seed: int,
    index: int,
) -> dict[str, Any]:
    schedule_hash = str(schedule.get("schedule_hash") or "")
    physical_key = _adaptive_schedule_physical_source_row_key(entry, offset=offset)
    candidate = {
        "candidate_schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "candidate_id": _hash([FINAL_PROMOTION_REPLACEMENT_ARTIFACT_VERSION, physical_key, seed]),
        "source_doc_id": _hash(["final_promotion_repair_source_doc", schedule_hash, entry.get("entry_hash"), int(offset)]),
        "source_schedule_hash": schedule_hash,
        "schedule_entry_hash": entry.get("entry_hash"),
        "global_schedule_index": int(entry.get("global_schedule_index") or index),
        "schedule_pass_index": int(entry.get("schedule_pass_index") or 0),
        "source_file_id": entry.get("source_file_id"),
        "row_group_index": int(entry.get("row_group_index") or 0),
        "row_group_local_row_index": int(offset),
        "deterministic_row_locator": {
            "mode": "row_group_local_offset",
            "row_group_index": int(entry.get("row_group_index") or 0),
            "row_group_local_row_index": int(offset),
        },
        "physical_source_row_key_hash": _hash(physical_key),
        "label_unit": "document",
        "candidate_strata": ["final_promotion_repair_random_backfill"],
        "bucket_id": "random_non_borderline_backfill",
        "subbucket_id": "final_promotion_repair_random",
        "selection_reason_id": "final_promotion_repair:random_non_borderline_backfill",
        "selection_seed": int(seed),
        "selection_algorithm_version": FINAL_PROMOTION_REPLACEMENT_ARTIFACT_VERSION,
        "teacher_label_use": "final_35k_freeze_labeling_runner_input",
        "final_manifest_source_section": "non_borderline",
        "text_hash_status": "pending_private_refetch",
        "private_refetch_required_before_live_call": True,
        "deterministic_screening_status": "not_screened_final_label_candidate_only",
        "scarce_bucket_credit": False,
        "labels_are_corpus_approval": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    for source_key, target_key in (
        ("dataset_repo_id", "dataset_repo_id"),
        ("config", "dataset_config"),
        ("split", "dataset_split"),
        ("resolved_repo_sha", "resolved_repo_sha"),
    ):
        if entry.get(source_key) is not None:
            candidate[target_key] = entry[source_key]
    if schedule.get("source_metadata_hash"):
        candidate["source_metadata_hash"] = schedule["source_metadata_hash"]
    parquet_identity = entry.get("parquet_identity")
    if parquet_identity is not None:
        candidate["parquet_identity_hash"] = _hash(parquet_identity)
    candidate["candidate_hash"] = _final_promotion_candidate_hash(candidate)
    _reject_live_artifact_leakage(candidate)
    return candidate


def build_final_promotion_replacement_candidate_artifact(
    existing_candidates: list[dict[str, Any]],
    *,
    source_schedules: list[dict[str, Any]],
    target_count: int,
    buffer_count: int = 0,
    seed: int = 20260607,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build random non-borderline replacement candidates from schedule offsets.

    These rows are candidates for final teacher labeling, not deterministic-safe
    approvals. They intentionally carry no raw text and no parquet/document URL.
    """

    if int(target_count) < 0:
        raise ValueError("target_count must be non-negative")
    if int(buffer_count) < 0:
        raise ValueError("buffer_count must be non-negative")
    for candidate in existing_candidates:
        _reject_live_artifact_leakage(candidate)
    for schedule in source_schedules:
        if schedule.get("raw_text_serialized") is not False or schedule.get("token_serialized") is not False:
            raise ValueError("replacement source schedules must set raw_text_serialized=false and token_serialized=false")

    target = int(target_count)
    desired = target + int(buffer_count)
    excluded_sets = _final_candidate_identity_sets(existing_candidates)
    seen_sets = {key: set() for key in excluded_sets}
    schedule_rows: list[tuple[str, int, int, int, dict[str, Any], dict[str, Any]]] = []
    skipped_schedule_entries = 0
    skipped_offsets = 0
    for schedule_index, schedule in enumerate(source_schedules):
        schedule_hash = str(schedule.get("schedule_hash") or "")
        for entry_index, entry in enumerate(schedule.get("entries", [])):
            if not schedule_hash or not entry.get("entry_hash") or not entry.get("source_file_id"):
                skipped_schedule_entries += 1
                continue
            if entry.get("row_group_index") is None or entry.get("row_group_num_rows") is None:
                skipped_schedule_entries += 1
                continue
            offsets = entry.get("row_group_local_offsets")
            if not isinstance(offsets, list):
                skipped_schedule_entries += 1
                continue
            row_group_num_rows = int(entry["row_group_num_rows"])
            for offset in offsets:
                try:
                    offset_int = int(offset)
                except (TypeError, ValueError):
                    skipped_offsets += 1
                    continue
                if offset_int < 0 or offset_int >= row_group_num_rows:
                    skipped_offsets += 1
                    continue
                key = _hash(["final_promotion_replacement_schedule_row", seed, schedule_hash, entry.get("entry_hash"), offset_int])
                schedule_rows.append((key, schedule_index, entry_index, offset_int, entry, schedule))
    schedule_rows.sort(key=lambda item: item[0])

    candidates: list[dict[str, Any]] = []
    skipped_collisions = 0
    skipped_invalid_candidates = 0
    scanned_offsets = 0
    for _, _, _, offset_int, entry, schedule in schedule_rows:
        if desired and len(candidates) >= desired:
            break
        scanned_offsets += 1
        try:
            candidate = _final_promotion_replacement_candidate_from_schedule_entry(
                entry,
                offset=offset_int,
                schedule=schedule,
                seed=seed,
                index=scanned_offsets,
            )
        except ValueError:
            skipped_invalid_candidates += 1
            continue
        identity_values = {field_name: _final_identity_value(candidate, field_name) for field_name in excluded_sets}
        collides = False
        for field_name, value in identity_values.items():
            if not value:
                continue
            if value in excluded_sets[field_name] or value in seen_sets[field_name]:
                collides = True
                break
        if collides:
            skipped_collisions += 1
            continue
        for field_name, value in identity_values.items():
            if value:
                seen_sets[field_name].add(value)
        candidates.append(candidate)
        if progress_path is not None and (len(candidates) % 250 == 0 or len(candidates) == target):
            _write_json_atomic(
                progress_path,
                {
                    "schema_version": FINAL_PROMOTION_REPLACEMENT_ARTIFACT_SCHEMA_VERSION,
                    "artifact_scope": "final_35k_promotion_random_backfill_replacements_progress",
                    "algorithm_version": FINAL_PROMOTION_REPLACEMENT_ARTIFACT_VERSION,
                    "target_count": target,
                    "buffer_count_requested": int(buffer_count),
                    "candidate_count": len(candidates),
                    "scanned_offset_count": scanned_offsets,
                    "skipped_collision_count": skipped_collisions,
                    "raw_text_serialized": False,
                    "raw_url_serialized": False,
                    "token_serialized": False,
                    "paid_calls_made": False,
                    "deepseek_calls_made": False,
                },
            )
    if len(candidates) < target:
        raise ValueError(
            "final promotion replacement generation failed closed: "
            f"replacement_required_count={target}, eligible_replacement_count={len(candidates)}"
        )
    artifact = {
        "schema_version": FINAL_PROMOTION_REPLACEMENT_ARTIFACT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "artifact_scope": "final_35k_promotion_random_backfill_replacements",
        "algorithm_version": FINAL_PROMOTION_REPLACEMENT_ARTIFACT_VERSION,
        "target_count": target,
        "buffer_count_requested": int(buffer_count),
        "buffer_count_available": max(0, len(candidates) - target),
        "candidate_count": len(candidates),
        "source_schedule_count": len(source_schedules),
        "source_schedule_hashes": sorted({str(schedule.get("schedule_hash") or "") for schedule in source_schedules if schedule.get("schedule_hash")}),
        "excluded_existing_candidate_count": len(existing_candidates),
        "scanned_offset_count": scanned_offsets,
        "skipped_schedule_entry_count": skipped_schedule_entries,
        "skipped_offset_count": skipped_offsets,
        "skipped_collision_count": skipped_collisions,
        "skipped_invalid_candidate_count": skipped_invalid_candidates,
        "replacement_bucket_id": "random_non_borderline_backfill",
        "replacement_status": "candidate_only_requires_final_teacher_labeling",
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "paid_calls_made": False,
        "deepseek_calls_made": False,
        "candidates": candidates,
    }
    artifact["artifact_hash"] = _hash({key: value for key, value in artifact.items() if key not in {"generated_at_utc", "artifact_hash"}})
    _reject_live_artifact_leakage(artifact)
    return artifact


def write_final_promotion_replacement_candidate_artifact(
    review_artifact_path: str | Path,
    attempt_manifest_path: str | Path,
    source_schedule_paths: list[str | Path],
    output_path: str | Path,
    *,
    target_count: int | None = None,
    buffer_count: int = 256,
    seed: int = 20260607,
    progress_path: str | Path | None = None,
    expected_candidate_count: int = TEACHER_LABEL_FINAL_TARGET,
) -> dict[str, Any]:
    review_artifact = json.loads(Path(review_artifact_path).read_text(encoding="utf-8"))
    attempt_manifest = json.loads(Path(attempt_manifest_path).read_text(encoding="utf-8"))
    promoted, _, _, _, _, _ = _promote_review_candidates_for_final_labeling(
        review_artifact,
        attempt_manifest=attempt_manifest,
        expected_candidate_count=expected_candidate_count,
    )
    repair_plan = build_final_promotion_repair_plan(
        review_artifact,
        attempt_manifest=attempt_manifest,
        expected_candidate_count=expected_candidate_count,
    )
    replacement_target = int(target_count) if target_count is not None else int(repair_plan["replacement_required_count"])
    source_schedules = [json.loads(Path(path).read_text(encoding="utf-8")) for path in source_schedule_paths]
    artifact = build_final_promotion_replacement_candidate_artifact(
        promoted,
        source_schedules=source_schedules,
        target_count=replacement_target,
        buffer_count=buffer_count,
        seed=seed,
        progress_path=progress_path,
    )
    artifact["source_review_artifact_hash"] = review_artifact.get("artifact_hash")
    artifact["attempt_manifest_hash"] = attempt_manifest.get("manifest_hash")
    artifact["repair_plan_hash"] = repair_plan["repair_plan_hash"]
    artifact["artifact_hash"] = _hash({key: value for key, value in artifact.items() if key not in {"generated_at_utc", "artifact_hash"}})
    _reject_live_artifact_leakage(artifact)
    _write_json_atomic(output_path, artifact)
    return artifact


def _final_duplicate_blocker_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "duplicate_candidate_id_count": _duplicate_nonempty_count([str(candidate.get("candidate_id") or "") for candidate in candidates]),
        "duplicate_physical_row_key_count": _duplicate_nonempty_count(
            [str(candidate.get("physical_source_row_key_hash") or "") for candidate in candidates]
        ),
        "duplicate_text_hash_count": _duplicate_nonempty_count(
            [str(candidate.get("text_hash_sha256") or candidate.get("text_hash") or "") for candidate in candidates]
        ),
        "duplicate_normalized_text_hash_count": _duplicate_nonempty_count(
            [str(candidate.get("normalized_text_hash_sha256") or "") for candidate in candidates]
        ),
    }


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for current, _, files in os.walk(path):
        for file in files:
            try:
                total += (Path(current) / file).stat().st_size
            except OSError:
                continue
    return total


def _final_row_group_bounds_payload(
    *,
    source_file_id: str,
    parquet_identity_hash: str,
    row_group_index: int,
    row_group: dict[str, Any],
    schema_fingerprint: str | None = None,
) -> dict[str, Any]:
    row_group_num_rows = row_group.get("num_rows", row_group.get("row_group_num_rows"))
    if row_group_num_rows is None:
        raise ValueError("row-group bounds missing num_rows")
    size = row_group.get("row_group_size_estimate_bytes") if isinstance(row_group.get("row_group_size_estimate_bytes"), dict) else {}
    compressed = row_group.get("compressed_byte_size", size.get("compressed", 0))
    total = row_group.get("total_byte_size", size.get("total", 0))
    return {
        "source_file_id": str(source_file_id),
        "parquet_identity_hash": str(parquet_identity_hash),
        "row_group_index": int(row_group_index),
        "row_group_num_rows": int(row_group_num_rows),
        "row_group_size_estimate_bytes": {
            "compressed": int(compressed or 0),
            "total": int(total or 0),
        },
        "schema_fingerprint": str(schema_fingerprint or row_group.get("schema_fingerprint") or ""),
    }


def validate_final_row_group_bounds_recovery_artifact(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != FINAL_ROW_GROUP_BOUNDS_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported final row-group bounds artifact schema")
    if artifact.get("artifact_scope") != "final_35k_row_group_bounds_recovery":
        raise ValueError("unsupported final row-group bounds artifact scope")
    if artifact.get("algorithm_version") != FINAL_ROW_GROUP_BOUNDS_ARTIFACT_VERSION:
        raise ValueError("unsupported final row-group bounds artifact algorithm")
    for flag in (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "labels_are_corpus_approval",
        "training_shards_created",
        "paid_calls_made",
        "deepseek_calls_made",
    ):
        if artifact.get(flag) is not False:
            raise ValueError(f"final row-group bounds artifact must set {flag}=false")
    _reject_live_artifact_leakage(artifact)
    stable = {key: value for key, value in artifact.items() if key not in {"generated_at_utc", "artifact_hash"}}
    if artifact.get("artifact_hash") != _hash(stable):
        raise ValueError("final row-group bounds artifact hash mismatch")
    entries = artifact.get("entries")
    if not isinstance(entries, list):
        raise ValueError("final row-group bounds artifact entries must be a list")
    if artifact.get("resolved_row_group_count") != len(entries):
        raise ValueError("final row-group bounds artifact resolved count mismatch")
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("final row-group bounds entry must be an object")
        source_file_id = str(entry.get("source_file_id") or "")
        if not source_file_id or not entry.get("parquet_identity_hash"):
            raise ValueError("final row-group bounds entry missing source identity")
        if entry.get("row_group_index") is None or entry.get("row_group_num_rows") is None:
            raise ValueError("final row-group bounds entry missing row-group identity")
        key = (source_file_id, int(entry["row_group_index"]))
        if key in seen:
            raise ValueError("final row-group bounds artifact contains duplicate row-group bounds")
        seen.add(key)
        if int(entry["row_group_num_rows"]) < 1:
            raise ValueError("final row-group bounds entry has nonpositive row count")


def _final_row_group_bounds_artifact_index(
    row_group_bounds_artifacts: list[dict[str, Any]] | None,
) -> dict[tuple[str, int], dict[str, Any]]:
    bounds: dict[tuple[str, int], dict[str, Any]] = {}
    conflicts: set[tuple[str, int]] = set()
    for artifact in row_group_bounds_artifacts or []:
        validate_final_row_group_bounds_recovery_artifact(artifact)
        for entry in artifact.get("entries", []):
            key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
            payload = {
                "row_group_num_rows": int(entry["row_group_num_rows"]),
                "row_group_size_estimate_bytes": entry.get("row_group_size_estimate_bytes") or {},
                "schema_fingerprint": str(entry.get("schema_fingerprint") or ""),
                "parquet_identity_hash": str(entry.get("parquet_identity_hash") or ""),
            }
            existing = bounds.get(key)
            if existing is not None and existing != payload:
                conflicts.add(key)
                continue
            bounds[key] = payload
    if conflicts:
        raise ValueError(f"final row-group bounds artifacts contain conflicting bounds: conflict_count={len(conflicts)}")
    return bounds


def _final_required_missing_row_groups(
    candidates: list[dict[str, Any]],
    *,
    known_bounds: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, dict[int, set[int]]]:
    missing: dict[str, dict[int, set[int]]] = {}
    for candidate in candidates:
        source_file_id = str(candidate.get("source_file_id") or "")
        row_group_index = int(candidate["row_group_index"])
        offset = int(candidate["row_group_local_row_index"])
        if (source_file_id, row_group_index) not in known_bounds:
            missing.setdefault(source_file_id, {}).setdefault(row_group_index, set()).add(offset)
    return missing


def build_final_row_group_bounds_recovery_artifact(
    candidates: list[dict[str, Any]],
    *,
    source_schedules: list[dict[str, Any]],
    footer_metadata_reader: Callable[[dict[str, Any]], dict[str, Any]],
    row_group_bounds_artifacts: list[dict[str, Any]] | None = None,
    progress_path: str | Path | None = None,
    expected_candidate_count: int = TEACHER_LABEL_FINAL_TARGET,
) -> dict[str, Any]:
    """Recover only no-raw Parquet row-group bounds needed by final candidates."""

    if len(candidates) != int(expected_candidate_count):
        raise ValueError(
            "final row-group bounds recovery failed closed: "
            f"candidate_count_mismatch={abs(int(expected_candidate_count) - len(candidates)) or 1}"
        )
    try:
        locators = _final_source_locator_index(source_schedules)
    except ValueError as exc:
        raise ValueError(f"final row-group bounds recovery failed closed: multiple locators: {exc}") from exc
    native_bounds = _final_row_group_bounds_index(source_schedules, row_group_bounds_artifacts=row_group_bounds_artifacts)
    missing_by_source = _final_required_missing_row_groups(candidates, known_bounds=native_bounds)
    missing_source_locator_count = sum(1 for source_file_id in missing_by_source if source_file_id not in locators)
    if missing_source_locator_count:
        raise ValueError(
            "final row-group bounds recovery failed closed: "
            f"missing_source_locator_count={missing_source_locator_count}"
        )

    entries: list[dict[str, Any]] = []
    missing_footer_row_group_count = 0
    offset_out_of_bounds_count = 0
    parquet_identity_drift_count = 0
    files_processed = 0
    expected_identity_by_source = {
        source_file_id: _hash(locator.get("parquet_identity"))
        for source_file_id, locator in locators.items()
    }
    for source_file_id in sorted(missing_by_source):
        locator = locators[source_file_id]
        expected_identity_hash = expected_identity_by_source[source_file_id]
        footer = footer_metadata_reader(dict(locator))
        if not isinstance(footer, dict):
            raise ValueError("final row-group bounds recovery failed closed: invalid_footer_metadata_count=1")
        observed_identity_hash = footer.get("parquet_identity_hash")
        if observed_identity_hash and str(observed_identity_hash) != expected_identity_hash:
            parquet_identity_drift_count += 1
            continue
        row_groups_by_index = {}
        for group in footer.get("row_groups", []):
            if not isinstance(group, dict) or group.get("row_group_index") is None:
                continue
            row_groups_by_index[int(group["row_group_index"])] = group
        schema_fingerprint = str(footer.get("schema_fingerprint") or "")
        for row_group_index in sorted(missing_by_source[source_file_id]):
            group = row_groups_by_index.get(row_group_index)
            if group is None:
                missing_footer_row_group_count += 1
                continue
            payload = _final_row_group_bounds_payload(
                source_file_id=source_file_id,
                parquet_identity_hash=expected_identity_hash,
                row_group_index=row_group_index,
                row_group=group,
                schema_fingerprint=schema_fingerprint,
            )
            max_offset = max(missing_by_source[source_file_id][row_group_index])
            if max_offset >= int(payload["row_group_num_rows"]):
                offset_out_of_bounds_count += 1
                continue
            entries.append(payload)
        files_processed += 1
        if progress_path is not None:
            _write_json_atomic(
                progress_path,
                {
                    "schema_version": FINAL_ROW_GROUP_BOUNDS_ARTIFACT_SCHEMA_VERSION,
                    "artifact_scope": "final_35k_row_group_bounds_recovery_progress",
                    "algorithm_version": FINAL_ROW_GROUP_BOUNDS_ARTIFACT_VERSION,
                    "candidate_count": len(candidates),
                    "required_row_group_count": sum(len(groups) for groups in missing_by_source.values()),
                    "resolved_row_group_count": len(entries),
                    "source_file_count": len(missing_by_source),
                    "files_processed": files_processed,
                    "raw_text_serialized": False,
                    "raw_url_serialized": False,
                    "token_serialized": False,
                    "paid_calls_made": False,
                    "deepseek_calls_made": False,
                },
            )
    _raise_final_promotion_blockers(
        {
            "missing_footer_row_group_count": missing_footer_row_group_count,
            "offset_out_of_bounds_count": offset_out_of_bounds_count,
            "parquet_identity_drift_count": parquet_identity_drift_count,
        }
    )
    required_row_group_count = sum(len(groups) for groups in missing_by_source.values())
    artifact = {
        "schema_version": FINAL_ROW_GROUP_BOUNDS_ARTIFACT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "artifact_scope": "final_35k_row_group_bounds_recovery",
        "algorithm_version": FINAL_ROW_GROUP_BOUNDS_ARTIFACT_VERSION,
        "candidate_count": len(candidates),
        "expected_candidate_count": int(expected_candidate_count),
        "required_row_group_count": required_row_group_count,
        "resolved_row_group_count": len(entries),
        "source_file_count": len(missing_by_source),
        "files_processed": files_processed,
        "source_schedule_hashes": sorted({str(schedule.get("schedule_hash") or "") for schedule in source_schedules if schedule.get("schedule_hash")}),
        "existing_bounds_artifact_hashes": sorted(
            str(artifact.get("artifact_hash") or "") for artifact in (row_group_bounds_artifacts or []) if artifact.get("artifact_hash")
        ),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "paid_calls_made": False,
        "deepseek_calls_made": False,
        "entries": sorted(entries, key=lambda row: (row["source_file_id"], row["row_group_index"])),
    }
    artifact["artifact_hash"] = _hash({key: value for key, value in artifact.items() if key not in {"generated_at_utc", "artifact_hash"}})
    validate_final_row_group_bounds_recovery_artifact(artifact)
    return artifact


def _normalize_header_mapping(headers: Any) -> dict[str, str]:
    if not isinstance(headers, dict):
        return {}
    return {str(key).casefold(): str(value).strip().strip('"') for key, value in headers.items()}


def _verify_locator_identity_from_headers(locator: dict[str, Any], headers: dict[str, str]) -> None:
    identity = locator.get("parquet_identity") if isinstance(locator.get("parquet_identity"), dict) else {}
    checks = {
        "content-length": identity.get("content_length") or identity.get("declared_size"),
        "etag": identity.get("etag"),
        "x-linked-size": identity.get("x_linked_size"),
        "x-linked-etag": identity.get("x_linked_etag"),
        "x-xet-hash": identity.get("x_xet_hash"),
        "x-repo-commit": identity.get("x_repo_commit"),
    }
    for header, expected in checks.items():
        if expected in (None, "") or header not in headers:
            continue
        observed = headers[header]
        if header in {"content-length", "x-linked-size"}:
            if str(expected) != observed:
                raise ValueError("parquet identity drift detected from headers")
        elif str(expected).strip().strip('"') != observed:
            raise ValueError("parquet identity drift detected from headers")


def read_final_row_group_bounds_footer_metadata(
    locator: dict[str, Any],
    *,
    cache_dir: str | Path,
    cache_delta_cap_bytes: int = 256 * 1024 * 1024,
    block_size_bytes: int = 8 * 1024 * 1024,
    max_cache_blocks: int = 4,
    verify_identity_headers: bool = True,
) -> dict[str, Any]:
    """Read Parquet footer metadata only; never read row groups or raw text."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required for final row-group bounds recovery") from exc
    source = locator.get("parquet_url_or_path")
    if not isinstance(source, str) or not source:
        raise ValueError("final row-group bounds locator is missing parquet source")
    if block_size_bytes < 1:
        raise ValueError("block_size_bytes must be positive")
    if max_cache_blocks < 1:
        raise ValueError("max_cache_blocks must be positive")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    before = _directory_size_bytes(cache_path)
    if source.startswith("https://"):
        try:
            import fsspec
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("fsspec is required for HTTPS final row-group bounds recovery") from exc
        if verify_identity_headers:
            request = urllib_request.Request(source, method="HEAD", headers={"User-Agent": "twc-final-bounds/1"})
            with urllib_request.urlopen(request, timeout=60) as response:  # nosec: metadata-only trusted project source
                _verify_locator_identity_from_headers(locator, _normalize_header_mapping(dict(response.headers.items())))
        cache_options = {"maxblocks": int(max_cache_blocks)}
        with fsspec.open(
            source,
            mode="rb",
            block_size=int(block_size_bytes),
            cache_type="blockcache",
            cache_options=cache_options,
        ) as file_like:
            parquet_file = pq.ParquetFile(file_like)
            metadata = parquet_file.metadata
            columns = list(getattr(getattr(parquet_file, "schema", None), "names", []) or [])
    else:
        parquet_file = pq.ParquetFile(source)
        metadata = parquet_file.metadata
        columns = list(getattr(getattr(parquet_file, "schema", None), "names", []) or [])
    row_groups = []
    for row_group_index in range(metadata.num_row_groups):
        group = metadata.row_group(row_group_index)
        compressed = 0
        for column_index in range(group.num_columns):
            column = group.column(column_index)
            compressed += int(getattr(column, "total_compressed_size", 0) or 0)
        row_groups.append(
            {
                "row_group_index": row_group_index,
                "num_rows": int(group.num_rows),
                "total_byte_size": int(group.total_byte_size),
                "compressed_byte_size": int(compressed),
            }
        )
    after = _directory_size_bytes(cache_path)
    if after - before > int(cache_delta_cap_bytes):
        raise ValueError("final row-group bounds recovery cache delta exceeded configured cap")
    return {
        "schema_fingerprint": _hash(columns),
        "row_group_count": int(metadata.num_row_groups),
        "total_rows": int(metadata.num_rows),
        "parquet_identity_hash": _hash(locator.get("parquet_identity")),
        "row_groups": row_groups,
        "storage_observation": {
            "cache_delta_bytes": after - before,
            "cache_delta_cap_bytes": int(cache_delta_cap_bytes),
        },
    }


def write_final_row_group_bounds_recovery_artifact(
    manifest_path: str | Path,
    source_schedule_paths: list[str | Path],
    output_path: str | Path,
    *,
    row_group_bounds_paths: list[str | Path] | None = None,
    progress_path: str | Path | None = None,
    cache_dir: str | Path = "reports/teacher_labeling/final_row_group_bounds_cache",
    cache_delta_cap_bytes: int = 256 * 1024 * 1024,
    block_size_bytes: int = 8 * 1024 * 1024,
    max_cache_blocks: int = 4,
    verify_identity_headers: bool = True,
    expected_candidate_count: int = TEACHER_LABEL_FINAL_TARGET,
) -> dict[str, Any]:
    manifest = load_candidate_manifest(manifest_path)
    source_schedules = [json.loads(Path(path).read_text(encoding="utf-8")) for path in source_schedule_paths]
    existing_bounds = [json.loads(Path(path).read_text(encoding="utf-8")) for path in (row_group_bounds_paths or [])]

    def reader(locator: dict[str, Any]) -> dict[str, Any]:
        return read_final_row_group_bounds_footer_metadata(
            locator,
            cache_dir=cache_dir,
            cache_delta_cap_bytes=cache_delta_cap_bytes,
            block_size_bytes=block_size_bytes,
            max_cache_blocks=max_cache_blocks,
            verify_identity_headers=verify_identity_headers,
        )

    artifact = build_final_row_group_bounds_recovery_artifact(
        list(manifest.get("candidates") or []),
        source_schedules=source_schedules,
        row_group_bounds_artifacts=existing_bounds,
        footer_metadata_reader=reader,
        progress_path=progress_path,
        expected_candidate_count=expected_candidate_count,
    )
    artifact["source_candidate_manifest_hash"] = manifest.get("manifest_hash")
    artifact["artifact_hash"] = _hash({key: value for key, value in artifact.items() if key not in {"generated_at_utc", "artifact_hash"}})
    validate_final_row_group_bounds_recovery_artifact(artifact)
    _write_json_atomic(output_path, artifact)
    return artifact


def build_final_label_manifest_from_review_candidate(
    review_artifact: dict[str, Any],
    *,
    attempt_manifest: dict[str, Any],
    schedule: dict[str, Any] | None = None,
    expected_candidate_count: int = TEACHER_LABEL_FINAL_TARGET,
    repair_replacement_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Promote a diagnostic 35k review candidate into the final runner-input manifest.

    This does not create labels or corpus approval. Borderline rows are restored
    from the adaptive attempt manifest so final refetch identity stays strict.
    """

    if schedule is not None:
        for flag in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
            if schedule.get(flag) is not False:
                raise ValueError(f"final promotion schedule must set {flag}=false")
    (
        promoted,
        blocker_counts,
        restored_borderline_count,
        ambiguous_precomputed_text_hash_stripped_count,
        non_borderline,
        borderline,
    ) = _promote_review_candidates_for_final_labeling(
        review_artifact,
        attempt_manifest=attempt_manifest,
        expected_candidate_count=expected_candidate_count,
    )
    repair_plan = build_final_promotion_repair_plan(
        review_artifact,
        attempt_manifest=attempt_manifest,
        expected_candidate_count=expected_candidate_count,
    )
    if repair_plan["replacement_required_count"] and repair_replacement_candidates is None:
        duplicate_counts = _final_duplicate_blocker_counts(promoted)
        _raise_final_promotion_blockers({**blocker_counts, **duplicate_counts})
    drop_ids = {row["candidate_id"] for row in repair_plan["drops"]}
    drop_indexes = {
        index
        for index, candidate in enumerate(promoted)
        if str(candidate.get("candidate_id") or "") in drop_ids
    }
    replacement_rows, skipped_replacements = _final_replacement_candidates(
        repair_replacement_candidates or [],
        promoted=promoted,
        drop_indexes=drop_indexes,
        needed_count=int(repair_plan["replacement_required_count"]),
    )
    if repair_plan["replacement_required_count"] and not repair_replacement_candidates:
        raise ValueError(
            "final promotion repair failed closed: "
            f"replacement_required_count={repair_plan['replacement_required_count']}, eligible_replacement_count=0"
        )
    if drop_indexes:
        promoted = [
            candidate
            for index, candidate in enumerate(promoted)
            if index not in drop_indexes
        ] + replacement_rows
        blocker_counts["candidate_count_mismatch"] = 0 if len(promoted) == int(expected_candidate_count) else abs(int(expected_candidate_count) - len(promoted)) or 1

    blocker_counts["duplicate_candidate_id_count"] = _duplicate_nonempty_count([str(candidate.get("candidate_id") or "") for candidate in promoted])
    blocker_counts["duplicate_physical_row_key_count"] = _duplicate_nonempty_count(
        [str(candidate.get("physical_source_row_key_hash") or "") for candidate in promoted]
    )
    blocker_counts["duplicate_text_hash_count"] = _duplicate_nonempty_count(
        [str(candidate.get("text_hash_sha256") or candidate.get("text_hash") or "") for candidate in promoted]
    )
    blocker_counts["duplicate_normalized_text_hash_count"] = _duplicate_nonempty_count(
        [str(candidate.get("normalized_text_hash_sha256") or "") for candidate in promoted]
    )
    if schedule is not None:
        for candidate in promoted:
            try:
                _schedule_entry_for_candidate(candidate, schedule)
            except ValueError:
                blocker_counts["schedule_refetch_noncoverage_count"] += 1
    _raise_final_promotion_blockers(blocker_counts)

    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "selection_algorithm_version": "teacher_candidate_manifest_v1",
        "manifest_scope": FINAL_LABEL_MANIFEST_SCOPE,
        "candidate_count": len(promoted),
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "label_readiness": {
            "ready_for_paid_labeling": True,
            "readiness_policy": "exact_35k_final_freeze_label_manifest",
            "required_actions": [],
        },
        "quota_report": {
            "candidate_count_shortfall": max(0, TEACHER_LABEL_FINAL_TARGET - len(promoted)),
            "non_borderline_count": len(non_borderline),
            "borderline_count": len(borderline),
            "random_backfill_not_scarce_bucket_credit": bool(review_artifact.get("random_backfill_not_scarce_bucket_credit", True)),
        },
        "promotion_summary": {
            "source_artifact_hash": review_artifact.get("artifact_hash"),
            "source_artifact_sha256": review_artifact.get("source_artifact_sha256"),
            "attempt_manifest_hash": attempt_manifest.get("manifest_hash"),
            "expected_candidate_count": int(expected_candidate_count),
            "promoted_candidate_count": len(promoted),
            "borderline_identity_restored_from_attempt_manifest_count": restored_borderline_count,
            "ambiguous_precomputed_text_hash_stripped_count": ambiguous_precomputed_text_hash_stripped_count,
            "repair_plan_hash": repair_plan["repair_plan_hash"],
            "duplicate_component_count": repair_plan["duplicate_component_count"],
            "duplicate_drop_count": repair_plan["drop_count"],
            "repair_replacement_count": len(replacement_rows),
            "repair_replacement_skipped_count": skipped_replacements,
            "repair_replacement_bucket_id": repair_plan["replacement_bucket_id"],
            "schedule_refetch_coverage_validated": schedule is not None,
            "runner_input_only_not_teacher_labels": True,
            "labels_are_corpus_approval": False,
            "accepted_manifest": False,
            "training_shards_created": False,
        },
        "candidates": promoted,
    }
    manifest["manifest_hash"] = _hash({key: value for key, value in manifest.items() if key not in {"generated_at_utc", "manifest_hash"}})
    validate_candidate_manifest(manifest)
    if expected_candidate_count == TEACHER_LABEL_FINAL_TARGET:
        _validate_final_batch_manifest(manifest)
    _reject_live_artifact_leakage(manifest)
    return manifest


def write_final_label_manifest_from_review_candidate(
    review_artifact_path: str | Path,
    attempt_manifest_path: str | Path,
    output_path: str | Path,
    *,
    schedule_path: str | Path | None = None,
    repair_replacements_path: str | Path | None = None,
    expected_candidate_count: int = TEACHER_LABEL_FINAL_TARGET,
) -> dict[str, Any]:
    review_artifact = json.loads(Path(review_artifact_path).read_text(encoding="utf-8"))
    attempt_manifest = json.loads(Path(attempt_manifest_path).read_text(encoding="utf-8"))
    schedule = json.loads(Path(schedule_path).read_text(encoding="utf-8")) if schedule_path is not None else None
    repair_replacements = _load_final_repair_replacement_candidates(repair_replacements_path) if repair_replacements_path is not None else None
    manifest = build_final_label_manifest_from_review_candidate(
        review_artifact,
        attempt_manifest=attempt_manifest,
        schedule=schedule,
        repair_replacement_candidates=repair_replacements,
        expected_candidate_count=expected_candidate_count,
    )
    _write_json_atomic(output_path, manifest)
    return manifest


def _load_final_repair_replacement_candidates(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("candidates"), list):
            candidates = payload["candidates"]
        elif isinstance(payload.get("selected_candidates"), list):
            candidates = payload["selected_candidates"]
        elif isinstance(payload.get("random_backfill"), dict) and isinstance(payload["random_backfill"].get("selected_candidates"), list):
            candidates = payload["random_backfill"]["selected_candidates"]
        else:
            raise ValueError("repair replacement candidate file must contain candidates, selected_candidates, or random_backfill.selected_candidates")
    else:
        raise ValueError("repair replacement candidate file must be a JSON object or list")
    _reject_live_artifact_leakage(candidates)
    return [dict(candidate) for candidate in candidates]


def write_final_35k_candidate_refetch_schedule(
    manifest_path: str | Path,
    source_schedule_paths: list[str | Path],
    output_path: str | Path,
    *,
    row_group_bounds_paths: list[str | Path] | None = None,
    source_metadata_hash: str | None = None,
    expected_candidate_count: int = TEACHER_LABEL_FINAL_TARGET,
) -> dict[str, Any]:
    manifest = load_candidate_manifest(manifest_path)
    schedules = [json.loads(Path(path).read_text(encoding="utf-8")) for path in source_schedule_paths]
    row_group_bounds_artifacts = [json.loads(Path(path).read_text(encoding="utf-8")) for path in (row_group_bounds_paths or [])]
    schedule = build_final_35k_candidate_refetch_schedule(
        list(manifest.get("candidates") or []),
        source_schedules=schedules,
        row_group_bounds_artifacts=row_group_bounds_artifacts,
        source_metadata_hash=source_metadata_hash,
        expected_candidate_count=expected_candidate_count,
    )
    _write_json_atomic(output_path, schedule, allow_source_locator_urls=True)
    return schedule


def _final_manifest_blockers(manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    try:
        validate_candidate_manifest(manifest)
    except ValueError:
        return ["candidate_manifest_schema_invalid"]
    candidates = manifest.get("candidates")
    observed = len(candidates) if isinstance(candidates, list) else 0
    if observed != TEACHER_LABEL_FINAL_TARGET:
        blockers.append("candidate_manifest_not_exact_35000")
    if manifest.get("selection_algorithm_version") != "teacher_candidate_manifest_v1":
        blockers.append("candidate_manifest_not_canonical_distribution_manifest")
    if manifest.get("manifest_scope") != FINAL_LABEL_MANIFEST_SCOPE:
        blockers.append("candidate_manifest_not_final_freeze_scope")
    if manifest.get("selection_algorithm_version") == "deepseek_seed_label_manifest_v1" or manifest.get("seed_split"):
        blockers.append("seed_6000_manifest_is_not_final_35k_manifest")
    if manifest.get("selection_algorithm_version") == ADAPTIVE_BOF_ATTEMPT_MANIFEST_VERSION or manifest.get("attempt_readiness"):
        blockers.append("adaptive_attempt_manifest_is_not_final_35k_manifest")
    readiness = manifest.get("label_readiness")
    if not isinstance(readiness, dict) or readiness.get("ready_for_paid_labeling") is not True:
        blockers.append("candidate_manifest_not_label_ready")
    if manifest.get("accepted_manifest") is not False:
        blockers.append("candidate_manifest_must_not_be_accepted_manifest")
    if manifest.get("labels_are_corpus_approval") is not False:
        blockers.append("candidate_manifest_must_not_claim_corpus_approval")
    if manifest.get("training_shards_created") is not False:
        blockers.append("candidate_manifest_must_not_create_training_shards")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not candidate.get("physical_source_row_key_hash"):
                blockers.append("candidate_manifest_missing_physical_source_row_key")
                break
            try:
                if str(candidate.get("physical_source_row_key_hash") or "") != physical_source_row_key_hash(candidate):
                    blockers.append("candidate_manifest_physical_source_row_key_drift")
                    break
            except ValueError:
                blockers.append("candidate_manifest_physical_source_row_key_invalid")
                break
        for candidate in candidates:
            if (candidate.get("text_hash") or candidate.get("text_hash_sha256")) and not candidate.get("text_hash_algorithm"):
                blockers.append("candidate_manifest_ambiguous_text_hash_algorithm")
                break
            if candidate.get("text_hash_algorithm") and str(candidate["text_hash_algorithm"]) not in ALLOWED_PRIVATE_REFETCH_TEXT_HASH_ALGORITHMS:
                blockers.append("candidate_manifest_unsupported_text_hash_algorithm")
                break
    return sorted(set(blockers))


def _validate_final_batch_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    blockers = _final_manifest_blockers(manifest)
    if blockers:
        raise ValueError(f"final label manifest is not ready: {', '.join(blockers)}")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("final label manifest candidates must be a list")
    ids = [str(candidate.get("candidate_id") or "") for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("final label manifest contains duplicate candidate ids")
    for candidate in candidates:
        if candidate.get("label_unit") not in ALLOWED_LABEL_UNITS:
            raise ValueError("final label candidate has invalid label_unit")
        for key in (
            "source_schedule_hash",
            "schedule_entry_hash",
            "source_file_id",
            "row_group_index",
            "row_group_local_row_index",
            "physical_source_row_key_hash",
        ):
            if candidate.get(key) is None:
                raise ValueError(f"final label candidate missing {key}")
        if str(candidate.get("physical_source_row_key_hash") or "") != physical_source_row_key_hash(candidate):
            raise ValueError("final label candidate physical_source_row_key_hash drift")
        if (candidate.get("text_hash") or candidate.get("text_hash_sha256")) and not candidate.get("text_hash_algorithm"):
            raise ValueError("final label candidate precomputed text hash requires text_hash_algorithm")
        if candidate.get("text_hash_algorithm") and str(candidate["text_hash_algorithm"]) not in ALLOWED_PRIVATE_REFETCH_TEXT_HASH_ALGORITHMS:
            raise ValueError("final label candidate has unsupported text_hash_algorithm")
    return candidates


def build_final_label_readiness_report(
    *,
    candidate_manifest_path: str | Path,
    current_balance_usd: float | None,
    budget_cap_usd: float | None,
    minimum_balance_reserve_usd: float = 0.0,
    manifest_loader: Callable[[str | Path], dict[str, Any]] = load_candidate_manifest,
) -> dict[str, Any]:
    """Write the no-paid launch gate for the final 35k freeze-label runner."""

    blockers: list[str] = []
    manifest: dict[str, Any] | None = None
    manifest_path = Path(candidate_manifest_path)
    if not manifest_path.exists():
        blockers.append("candidate_manifest_missing")
    else:
        try:
            manifest = manifest_loader(manifest_path)
        except Exception:  # noqa: BLE001 - readiness reports named blocker, not raw exception text.
            blockers.append("candidate_manifest_schema_invalid")
    candidates = manifest.get("candidates") if isinstance(manifest, dict) else []
    observed_count = len(candidates) if isinstance(candidates, list) else 0
    if manifest is not None:
        blockers.extend(_final_manifest_blockers(manifest))
    if budget_cap_usd is None or float(budget_cap_usd) <= 0:
        blockers.append("explicit_budget_cap_missing")
    if current_balance_usd is None or float(current_balance_usd) < 0:
        blockers.append("current_balance_missing")
    elif float(current_balance_usd) < float(minimum_balance_reserve_usd):
        blockers.append("current_balance_below_minimum_reserve")
    if minimum_balance_reserve_usd < 0:
        blockers.append("minimum_balance_reserve_negative")
    checks = {
        "exact_final_count": {
            "required_count": TEACHER_LABEL_FINAL_TARGET,
            "observed_count": observed_count,
            "passed": observed_count == TEACHER_LABEL_FINAL_TARGET,
        },
        "candidate_manifest_scope": {
            "required_scope": FINAL_LABEL_MANIFEST_SCOPE,
            "observed_scope": manifest.get("manifest_scope") if isinstance(manifest, dict) else None,
            "passed": bool(isinstance(manifest, dict) and manifest.get("manifest_scope") == FINAL_LABEL_MANIFEST_SCOPE),
        },
        "candidate_manifest_schema": {
            "passed": bool(manifest is not None and "candidate_manifest_schema_invalid" not in blockers),
            "selection_algorithm_version": manifest.get("selection_algorithm_version") if isinstance(manifest, dict) else None,
            "manifest_hash": manifest.get("manifest_hash") if isinstance(manifest, dict) else None,
            "label_readiness": manifest.get("label_readiness") if isinstance(manifest, dict) else {},
        },
        "available_balance_policy": {
            "policy": "proceed_with_available_balance_preserve_resumable_partials_v1",
            "current_balance_usd": None if current_balance_usd is None else float(current_balance_usd),
            "budget_cap_usd": None if budget_cap_usd is None else float(budget_cap_usd),
            "minimum_balance_reserve_usd": float(minimum_balance_reserve_usd),
            "full_35k_estimate_required_to_fit_before_launch": False,
        },
    }
    unique_blockers = sorted(set(blockers))
    report = {
        "final_label_readiness_schema_version": FINAL_LABEL_READINESS_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "ready_for_final_label_launch": not unique_blockers,
        "blockers": unique_blockers,
        "checks": checks,
        "paid_api_calls_performed": False,
        "deepseek_calls_made": False,
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    report["final_label_readiness_hash"] = _hash({key: value for key, value in report.items() if key != "generated_at_utc"})
    _reject_live_artifact_leakage(report)
    return report


def _price_from_official_snapshot(model: str) -> PriceSnapshot:
    prices = DEEPSEEK_OFFICIAL_DOC_SNAPSHOT["pricing_per_1m_tokens_usd"]
    if model not in prices:
        raise ValueError(f"no official price snapshot for model: {model}")
    price = prices[model]
    return PriceSnapshot(
        source="embedded_deepseek_official_docs_snapshot_2026_06_03_not_launch_price",
        captured_at_utc=DEEPSEEK_OFFICIAL_DOC_SNAPSHOT["captured_at_utc"],
        model=model,
        input_cache_miss_per_million=float(price["input_cache_miss"]),
        input_cache_hit_per_million=float(price["input_cache_hit"]),
        output_per_million=float(price["output"]),
    )


def build_paid_label_batch_plan(
    manifest: dict[str, Any],
    *,
    label_count: int,
    price_snapshot: PriceSnapshot | dict[str, Any],
    mode: str,
    seed: int = 20260603,
    avg_input_tokens: int = 512,
    avg_output_tokens: int = 128,
    safety_multiplier: float = 1.25,
    budget_cap_usd: float = 11.0,
    reserve_usd: float = 0.75,
    provider_config: DeepSeekLabelingConfig | None = None,
    explicit_pilot_cap: int | None = None,
    budget_bucket: str = "teacher_labels",
) -> dict[str, Any]:
    """Plan a deterministic paid-label batch without performing paid calls."""

    validate_candidate_manifest(manifest)
    if mode not in {"pilot", "freeze"}:
        raise ValueError("mode must be pilot or freeze")
    if label_count < 1:
        raise ValueError("label_count must be positive")
    if mode == "pilot":
        if label_count > 100 and explicit_pilot_cap is None:
            raise ValueError("pilot planning requires an explicit pilot cap before large batches")
        if explicit_pilot_cap is not None and explicit_pilot_cap != label_count:
            raise ValueError("explicit pilot cap must equal label_count for deterministic planning")
        pilot_cap = explicit_pilot_cap if explicit_pilot_cap is not None else label_count
        if pilot_cap < 50 or pilot_cap > 100:
            raise ValueError("explicit pilot cap must be between 50 and 100")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate manifest contains no candidates")
    readiness = manifest.get("label_readiness")
    manifest_ready = bool(isinstance(readiness, dict) and readiness.get("ready_for_paid_labeling") is True)
    if mode == "freeze" and not manifest_ready:
        raise ValueError("candidate manifest is not ready for freeze paid labeling")
    if isinstance(price_snapshot, PriceSnapshot):
        price = price_snapshot
    elif isinstance(price_snapshot, dict):
        price = PriceSnapshot(
            source=str(price_snapshot["source"]),
            captured_at_utc=str(price_snapshot["captured_at_utc"]),
            model=str(price_snapshot["model"]),
            input_cache_miss_per_million=float(price_snapshot["input_cache_miss_per_million"]),
            input_cache_hit_per_million=float(price_snapshot["input_cache_hit_per_million"]),
            output_per_million=float(price_snapshot["output_per_million"]),
            currency=str(price_snapshot.get("currency", "USD")),
        )
    else:
        raise ValueError("price_snapshot must be a PriceSnapshot or dict")
    selected = _select_label_batch_candidates(candidates, label_count=min(label_count, len(candidates)), seed=seed)
    provider_config = provider_config or DeepSeekLabelingConfig()
    provider_payload = _config_payload(provider_config)
    public_provider_payload = _public_config_payload(provider_config)
    provider_config_hash = _hash(provider_payload)
    budget_check = estimate_label_cost_usd(
        label_count=len(selected),
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=price,
        safety_multiplier=safety_multiplier,
        budget_cap_usd=budget_cap_usd,
        reserve_usd=reserve_usd,
    )
    public_budget_check = _redact_price_snapshot_urls(budget_check)
    all_pro_budget = estimate_label_cost_usd(
        label_count=TEACHER_LABEL_FINAL_TARGET,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=PriceSnapshot(
            source=price.source,
            captured_at_utc=price.captured_at_utc,
            model=REQUIRED_DEEPSEEK_TEACHER_MODEL,
            input_cache_miss_per_million=DEEPSEEK_OFFICIAL_DOC_SNAPSHOT["pricing_per_1m_tokens_usd"][
                REQUIRED_DEEPSEEK_TEACHER_MODEL
            ]["input_cache_miss"],
            input_cache_hit_per_million=DEEPSEEK_OFFICIAL_DOC_SNAPSHOT["pricing_per_1m_tokens_usd"][
                REQUIRED_DEEPSEEK_TEACHER_MODEL
            ]["input_cache_hit"],
            output_per_million=DEEPSEEK_OFFICIAL_DOC_SNAPSHOT["pricing_per_1m_tokens_usd"][
                REQUIRED_DEEPSEEK_TEACHER_MODEL
            ]["output"],
        ),
        safety_multiplier=safety_multiplier,
        budget_cap_usd=budget_cap_usd,
        reserve_usd=reserve_usd,
    )
    public_all_pro_budget = _redact_price_snapshot_urls(all_pro_budget)
    strata_counts: dict[str, int] = {}
    for candidate in selected:
        for stratum in _candidate_label_strata(candidate):
            strata_counts[str(stratum)] = strata_counts.get(str(stratum), 0) + 1
    selected_public = [_candidate_public_labeling_record(candidate) for candidate in selected]
    private_refetch_required = any(
        candidate["private_refetch_required_before_live_call"] for candidate in selected_public
    )
    planned_requests = []
    for candidate in selected_public:
        label_unit = str(candidate.get("label_unit") or "document")
        prompt_hash = _dry_run_prompt_hash(candidate, provider_config)
        request_hash = _hash(
            {
                "provider_config_hash": provider_config_hash,
                "prompt_hash": prompt_hash,
                "candidate_id": candidate["candidate_id"],
                "text_hash": candidate["text_hash"],
                "text_hash_status": candidate["text_hash_status"],
                "source_identity_hash": candidate["source_identity_hash"],
                "label_unit": label_unit,
                "response_format": provider_config.response_format,
                "max_output_tokens": provider_config.max_output_tokens,
            }
        )
        planned_requests.append(
            {
                "candidate_id": candidate["candidate_id"],
                "source_doc_id": candidate["source_doc_id"],
                "source_doc_id_source": candidate["source_doc_id_source"],
                "source_identity_hash": candidate["source_identity_hash"],
                "text_hash": candidate["text_hash"],
                "text_hash_status": candidate["text_hash_status"],
                "source_schedule_hash": candidate.get("source_schedule_hash"),
                "schedule_entry_hash": candidate.get("schedule_entry_hash"),
                "source_file_id": candidate.get("source_file_id"),
                "row_group_index": candidate.get("row_group_index"),
                "row_group_local_row_index": candidate.get("row_group_local_row_index"),
                "bucket": candidate.get("bucket_id") or (candidate.get("candidate_strata") or ["unstratified"])[0],
                "bucket_id": candidate.get("bucket_id"),
                "subbucket_id": candidate.get("subbucket_id"),
                "selection_reason_id": candidate.get("selection_reason_id"),
                "budget_bucket": budget_bucket,
                "label_unit": label_unit,
                "prompt_hash": prompt_hash,
                "request_hash": request_hash,
                "config_hash": provider_config_hash,
                "private_refetch_required_before_live_call": candidate[
                    "private_refetch_required_before_live_call"
                ],
                "estimated_input_tokens": avg_input_tokens,
                "estimated_output_tokens": avg_output_tokens,
                "estimated_cost_usd": estimate_label_cost_usd(
                    label_count=1,
                    avg_input_tokens=avg_input_tokens,
                    avg_output_tokens=avg_output_tokens,
                    price=price,
                    safety_multiplier=safety_multiplier,
                    budget_cap_usd=budget_cap_usd,
                    reserve_usd=0.0,
                )["estimated_cost_usd"],
            }
        )
    plan = {
        "batch_plan_schema_version": PAID_LABEL_BATCH_PLAN_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "labeling_mode": mode,
        "source_candidate_manifest_hash": manifest["manifest_hash"],
        "source_manifest_label_readiness": manifest.get("label_readiness", {}),
        "source_manifest_quota_report": manifest.get("quota_report", {}),
        "candidate_count": len(selected),
        "requested_label_count": label_count,
        "available_candidate_count": len(candidates),
        "candidate_count_shortfall": max(0, label_count - len(candidates)),
        "selection_algorithm_version": "teacher_paid_label_batch_v1",
        "source_manifest_type": _source_manifest_type(candidates),
        "seed": seed,
        "selected_candidate_ids": [candidate["candidate_id"] for candidate in selected],
        "selected_strata_counts": dict(sorted(strata_counts.items())),
        "selected_candidates": selected_public,
        "planned_requests": planned_requests,
        "provider_config": public_provider_payload,
        "provider_config_hash": provider_config_hash,
        "budget_check": public_budget_check,
        "budget_bucket": budget_bucket,
        "all_pro_labeling_budget_visible": public_all_pro_budget,
        "freeze_ready_label_distribution": mode == "freeze" and manifest_ready,
        "pilot_incomplete_distribution": mode == "pilot" and not manifest_ready,
        "private_refetch_required_before_live_call": private_refetch_required,
        "requires_explicit_user_unlock": True,
        "paid_api_calls_performed": False,
        "raw_text_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
    }
    stable = {key: value for key, value in plan.items() if key != "generated_at_utc"}
    plan["batch_plan_hash"] = _hash(stable)
    _reject_live_artifact_leakage(plan)
    return plan


def _select_label_batch_candidates(
    candidates: list[dict[str, Any]],
    *,
    label_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    available_strata = sorted({stratum for candidate in candidates for stratum in _candidate_label_strata(candidate)})
    priority_strata = [stratum for stratum in REQUIRED_STRATA if stratum in available_strata]
    priority_strata.extend(stratum for stratum in available_strata if stratum not in set(priority_strata))
    per_stratum = max(1, label_count // max(1, len(priority_strata)))
    for stratum in priority_strata:
        pool = [
            candidate
            for candidate in candidates
            if stratum in _candidate_label_strata(candidate)
        ]
        for candidate in _stable_shuffle(pool, seed=seed, stratum=f"label_batch:{stratum}"):
            if len(selected) >= label_count:
                return selected
            if candidate["candidate_id"] in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["candidate_id"])
            if len([row for row in selected if stratum in _candidate_label_strata(row)]) >= per_stratum:
                break
    if len(selected) < label_count:
        for candidate in _stable_shuffle(candidates, seed=seed, stratum="label_batch:refill"):
            if len(selected) >= label_count:
                break
            if candidate["candidate_id"] in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["candidate_id"])
    return selected


def write_paid_label_batch_plan(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    label_count: int,
    mode: str,
    model: str = REQUIRED_DEEPSEEK_TEACHER_MODEL,
    seed: int = 20260603,
) -> dict[str, Any]:
    manifest = load_candidate_manifest(manifest_path)
    plan = build_paid_label_batch_plan(
        manifest,
        label_count=label_count,
        price_snapshot=_price_from_official_snapshot(model),
        mode=mode,
        seed=seed,
        explicit_pilot_cap=label_count if mode == "pilot" and label_count <= 100 else None,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    return plan


def _teacher_system_prompt(label_unit: str, *, prompt_variant: str = PROMPT_VARIANT_STANDARD) -> str:
    if prompt_variant not in ALLOWED_PROMPT_VARIANTS:
        raise ValueError("unsupported teacher prompt variant")
    unit_policy = (
        "The private input is one whole document. Judge the whole document."
        if label_unit == "document"
        else "The private input is a selected window bundle from one document. A risky bundle supports rejecting the document. "
        "A safe-looking bundle is only local evidence and does not prove the full document safe."
    )
    examples = "\n".join(
        json.dumps({"text": text, "label": label, "borderline": borderline}, sort_keys=True)
        for text, label, borderline in TEACHER_PROMPT_V3_EXAMPLES
    )
    return f"""You are a strict json-only document classifier for a concept-scrubbed pretraining corpus.
Return exactly one minified json object and no other text. Do not quote, copy, summarize, or paraphrase the private input.

Output exactly these two keys and no extra keys:
{{"label":"reject","borderline":true}}

Allowed values:
- "label" must be "safe" or "reject".
- "borderline" must be true or false.

Decision rule:
- Use "safe" only for dry external-world text with no forbidden concept and no risky source form.
- Use "reject" for any consciousness, awareness, sentience, subjective experience, selfhood, identity, personhood, soul or spirit language, inner state, introspection, metacognition, belief, intent, preference, affect, emotion, pain, suffering, pleasure, sensory or body state, sleep, dream, wakefulness, AI personhood, chatbot self-report, ML or computing agentive/self-modeling/cognitive metaphor, fiction, narrative, dialogue, interview, transcript, forum, support, help, FAQ, advice, review, testimonial, marketing, customer story, press/news/announcement wrapper, therapy, self-help, spirituality, psychology, neuroscience, cognition, animal sentience or welfare, human biography/social/history/news, child-development/classroom mental-state content, or organization/system agency metaphor.
- Ambiguity rejects for final filtering.
- borderline is not a third label. It marks cases close to a policy boundary or likely to create reasonable reviewer/model disagreement. Examples: safe-looking technical false friends; support/help/FAQ form with sparse semantic risk; mixed technical-plus-risk text; wrapper/form risk that ordinary users might call mundane technical content; ambiguous external versus inner-state language.

{unit_policy}

Synthetic examples:
{examples}
"""


def _teacher_user_prompt(
    candidate: dict[str, Any],
    raw_text: str,
    *,
    prompt_variant: str = PROMPT_VARIANT_STANDARD,
) -> str:
    metadata = {
        "candidate_id": candidate["candidate_id"],
        "label_unit": candidate.get("label_unit", "document"),
        "source_domain": candidate.get("source_domain"),
        "source_wrapper": candidate.get("source_wrapper"),
        "candidate_strata": candidate.get("candidate_strata"),
        "metadata_bands": candidate.get("metadata_bands"),
        "window_bundle": candidate.get("window_bundle"),
    }
    if prompt_variant == PROMPT_VARIANT_COMPACT_LENGTH_RETRY:
        metadata = {
            "candidate_id": candidate["candidate_id"],
            "label_unit": candidate.get("label_unit", "document"),
            "candidate_strata": candidate.get("candidate_strata"),
            "decision_status": candidate.get("decision_status"),
            "metadata_bands": candidate.get("metadata_bands"),
        }
        instruction = "Label the same private input again. Output minified json only."
    else:
        instruction = "Label the following private input. Output json only."
    return (
        f"{instruction} Do not copy text snippets.\n\n"
        f"CANDIDATE_METADATA_JSON:\n{json.dumps(metadata, sort_keys=True)}\n\n"
        f"PRIVATE_DOCUMENT_TEXT:\n{raw_text}"
    )


def render_teacher_prompt(
    candidate: dict[str, Any],
    *,
    private_text: str,
    prompt_variant: str = PROMPT_VARIANT_STANDARD,
) -> dict[str, Any]:
    """Render the exact private prompt. Callers must not write this to public artifacts."""

    if not isinstance(private_text, str) or not private_text:
        raise ValueError("private_text is required for private prompt rendering")
    label_unit = str(candidate.get("label_unit") or "document")
    if label_unit not in ALLOWED_LABEL_UNITS:
        raise ValueError("candidate label_unit must be document or window_bundle")
    if prompt_variant not in ALLOWED_PROMPT_VARIANTS:
        raise ValueError("unsupported teacher prompt variant")
    system = _teacher_system_prompt(label_unit, prompt_variant=prompt_variant)
    user = _teacher_user_prompt(candidate, private_text, prompt_variant=prompt_variant)
    return {
        "prompt_schema_version": 1,
        "prompt_version": TEACHER_PROMPT_VERSION,
        "prompt_variant": prompt_variant,
        "label_unit": label_unit,
        "system": system,
        "user": user,
        "prompt_hash": _hash({"system": system, "user": user}),
    }


def _dry_run_prompt_hash(candidate: dict[str, Any], config: DeepSeekLabelingConfig) -> str:
    label_unit = str(candidate.get("label_unit") or "document")
    if label_unit not in ALLOWED_LABEL_UNITS:
        raise ValueError("candidate label_unit must be document or window_bundle")
    text_hash, text_hash_status = _candidate_text_hash(candidate, allow_pending=True)
    source_doc_id, source_doc_id_source = _candidate_source_doc_id(candidate)
    metadata = {
        "candidate_id": candidate["candidate_id"],
        "source_doc_id": source_doc_id,
        "source_doc_id_source": source_doc_id_source,
        "source_identity_hash": _candidate_source_identity_hash(candidate),
        "text_hash": text_hash,
        "text_hash_status": text_hash_status,
        "label_unit": label_unit,
        "source_domain": candidate.get("source_domain"),
        "source_wrapper": candidate.get("source_wrapper"),
        "candidate_strata": _candidate_label_strata(candidate),
        "bucket_id": candidate.get("bucket_id"),
        "subbucket_id": candidate.get("subbucket_id"),
        "metadata_bands": candidate.get("metadata_bands"),
        "window_bundle": candidate.get("window_bundle"),
        "prompt_version": config.prompt_version,
    }
    return _hash(
        {
            "system_template": _teacher_system_prompt(label_unit, prompt_variant=PROMPT_VARIANT_STANDARD),
            "metadata": metadata,
        }
    )


def build_deepseek_request(
    candidate: dict[str, Any],
    *,
    raw_text: str,
    config: DeepSeekLabelingConfig,
    manifest_hash: str,
    prompt_variant: str = PROMPT_VARIANT_STANDARD,
    retry_attempt: int = 0,
    retry_of_idempotency_key: str | None = None,
    retry_reason: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-format DeepSeek request without performing the call."""

    if not isinstance(raw_text, str) or not raw_text:
        raise ValueError("raw_text is required only at controlled private labeling time")
    if prompt_variant not in ALLOWED_PROMPT_VARIANTS:
        raise ValueError("unsupported teacher prompt variant")
    if retry_attempt < 0:
        raise ValueError("retry_attempt must be non-negative")
    if retry_attempt and not retry_of_idempotency_key:
        raise ValueError("retry requests require retry_of_idempotency_key")
    source_doc_id, source_doc_id_source = _candidate_source_doc_id(candidate)
    source_identity_hash = _candidate_source_identity_hash(candidate)
    text_hash, text_hash_status = _candidate_text_hash(candidate, private_text=raw_text)
    if text_hash is None:
        raise ValueError("private DeepSeek request requires a concrete text hash")
    prompt = render_teacher_prompt(candidate, private_text=raw_text, prompt_variant=prompt_variant)
    thinking = {"type": config.thinking}
    if config.thinking == "enabled":
        thinking["reasoning_effort"] = config.reasoning_effort
    body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        "response_format": config.response_format,
        "max_tokens": config.max_output_tokens,
        "stream": config.stream,
        "thinking": thinking,
        "user_id": config.user_id,
    }
    if config.temperature is not None:
        body["temperature"] = config.temperature
    request_hash = _hash(body)
    config_hash = _hash(_config_payload(config))
    idempotency_key = _hash(
            [
                "deepseek_teacher_label",
                manifest_hash,
                candidate["candidate_id"],
                text_hash,
                config.prompt_version,
                prompt_variant,
                retry_attempt,
                retry_of_idempotency_key or "",
                retry_reason or "",
                config.model,
                request_hash,
            ]
        )
    return {
        "request_schema_version": 1,
        "base_url": config.base_url,
        "endpoint": config.endpoint_path,
        "method": "POST",
        "prompt_version": config.prompt_version,
        "prompt_variant": prompt_variant,
        "prompt_hash": prompt["prompt_hash"],
        "config_hash": config_hash,
        "candidate_id": candidate["candidate_id"],
        "source_candidate_manifest_hash": manifest_hash,
        "source_doc_id": source_doc_id,
        "source_doc_id_source": source_doc_id_source,
        "source_identity_hash": source_identity_hash,
        "text_hash": text_hash,
        "text_hash_algorithm": str(candidate.get("text_hash_algorithm") or TEXT_HASH_ALGORITHM_UTF8_SHA256_V1),
        "text_hash_status": text_hash_status,
        "label_unit": candidate.get("label_unit", "document"),
        "request_timeout_seconds": config.request_timeout_seconds,
        "max_retries": config.max_retries,
        "idempotency_key": idempotency_key,
        "request_body_sha256": request_hash,
        "retry_attempt": int(retry_attempt),
        "retry_of_idempotency_key": str(retry_of_idempotency_key or ""),
        "retry_reason": str(retry_reason or ""),
        "raw_text_serialized_to_manifest": False,
        "body": body,
    }


def parse_teacher_response(
    response_body: dict[str, Any],
    *,
    candidate: dict[str, Any],
    request: dict[str, Any],
    price_snapshot: PriceSnapshot | dict[str, Any],
    estimated_cost_usd: float,
    actual_cost_usd: float,
    latency_seconds: float | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "")
    request_hash = str(request.get("request_body_sha256") or "")
    idempotency_key = str(request.get("idempotency_key") or "")
    teacher_model = ((request.get("body") or {}).get("model") if isinstance(request.get("body"), dict) else None)
    teacher_prompt_version = request.get("prompt_version")
    if not idempotency_key:
        raise ValueError("teacher label response requires idempotency_key")
    if not candidate_id or not request_hash:
        raise ValueError("teacher label response requires candidate and request hashes")
    if not teacher_model or not teacher_prompt_version:
        raise ValueError("teacher label response requires teacher model and prompt version")
    source_doc_id = str(request.get("source_doc_id") or _candidate_source_doc_id(candidate)[0])
    source_identity_hash = str(request.get("source_identity_hash") or _candidate_source_identity_hash(candidate))
    text_hash_value = request.get("text_hash")
    text_hash_status = str(request.get("text_hash_status") or "")
    if not text_hash_value:
        text_hash_value, text_hash_status = _candidate_text_hash(candidate)
    text_hash = str(text_hash_value or "")
    if not source_doc_id or not text_hash:
        raise ValueError("teacher label response requires source_doc_id and text_hash")
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("DeepSeek response missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("DeepSeek response choice is invalid")
    finish_reason = first.get("finish_reason")
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    content_sha256 = _hash(content) if isinstance(content, str) else None
    content_char_length = len(content) if isinstance(content, str) else None
    if finish_reason == "length":
        return teacher_label_parse_error_record(
            error_code="provider_response_rejected",
            error_message="DeepSeek response finish_reason length",
            candidate=candidate,
            request=request,
            response_body=response_body,
            timestamp_utc=timestamp_utc,
            finish_reason=str(finish_reason),
            content_sha256=content_sha256,
            content_char_length=content_char_length,
            schema_failure_path="$.choices[0].finish_reason",
            schema_failure_kind="provider_finish_reason",
            schema_failure_rule="finish_reason_length_queued_for_repair",
            usage=response_body.get("usage", {}),
            estimated_cost_usd=estimated_cost_usd,
            actual_cost_usd=actual_cost_usd,
            latency_seconds=latency_seconds,
        )
    if not isinstance(content, str) or not content.strip():
        return teacher_label_parse_error_record(
            error_code="empty_content",
            error_message="DeepSeek JSON mode returned empty content",
            candidate=candidate,
            request=request,
            response_body=response_body,
            timestamp_utc=timestamp_utc,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            content_sha256=content_sha256,
            content_char_length=content_char_length,
            schema_failure_path="$.choices[0].message.content",
            schema_failure_kind="empty_content",
            schema_failure_rule="json_output_non_empty_content",
            usage=response_body.get("usage", {}),
            estimated_cost_usd=estimated_cost_usd,
            actual_cost_usd=actual_cost_usd,
            latency_seconds=latency_seconds,
        )
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return teacher_label_parse_error_record(
            error_code="invalid_json",
            error_message=str(exc),
            candidate=candidate,
            request=request,
            response_body=response_body,
            timestamp_utc=timestamp_utc,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            content_sha256=content_sha256,
            content_char_length=content_char_length,
            schema_failure_path="$",
            schema_failure_kind="invalid_json",
            schema_failure_rule="json_parse",
            usage=response_body.get("usage", {}),
            estimated_cost_usd=estimated_cost_usd,
            actual_cost_usd=actual_cost_usd,
            latency_seconds=latency_seconds,
        )
    try:
        _validate_teacher_label_json(parsed, candidate_label_unit=str(candidate.get("label_unit") or "document"))
    except TeacherSchemaValidationError as exc:
        return teacher_label_parse_error_record(
            error_code="schema_validation_failed",
            error_message=str(exc),
            candidate=candidate,
            request=request,
            response_body=response_body,
            timestamp_utc=timestamp_utc,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            content_sha256=content_sha256,
            content_char_length=content_char_length,
            schema_failure_path=exc.path,
            schema_failure_kind=exc.kind,
            schema_failure_rule=exc.rule,
            schema_failure_value_sha256=exc.value_sha256,
            schema_missing_keys=exc.missing_keys,
            schema_extra_key_sha256s=exc.extra_key_sha256s,
            schema_extra_key_count=exc.extra_key_count,
            usage=response_body.get("usage", {}),
            estimated_cost_usd=estimated_cost_usd,
            actual_cost_usd=actual_cost_usd,
            latency_seconds=latency_seconds,
        )
    except ValueError as exc:
        return teacher_label_parse_error_record(
            error_code="schema_validation_failed",
            error_message=str(exc),
            candidate=candidate,
            request=request,
            response_body=response_body,
            timestamp_utc=timestamp_utc,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            content_sha256=content_sha256,
            content_char_length=content_char_length,
            schema_failure_path="$",
            schema_failure_kind="schema_validation_failed",
            schema_failure_rule="legacy_validation",
            usage=response_body.get("usage", {}),
            estimated_cost_usd=estimated_cost_usd,
            actual_cost_usd=actual_cost_usd,
            latency_seconds=latency_seconds,
        )
    response_hash = _hash(response_body)
    if isinstance(price_snapshot, PriceSnapshot):
        price_payload = asdict(price_snapshot)
    elif isinstance(price_snapshot, dict):
        price_payload = dict(price_snapshot)
    else:
        raise ValueError("teacher label response requires price_snapshot")
    estimated = float(estimated_cost_usd)
    actual = float(actual_cost_usd)
    if estimated < 0 or actual < 0:
        raise ValueError("teacher label costs must be non-negative")
    row = {
        "label_row_schema_version": TEACHER_LABEL_ROW_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "idempotency_key": idempotency_key,
        "teacher_model": str(teacher_model),
        "teacher_prompt_version": str(teacher_prompt_version),
        "source_candidate_manifest_hash": str(request.get("source_candidate_manifest_hash") or ""),
        "provider_config_hash": str(request.get("config_hash") or ""),
        "source_doc_id": source_doc_id,
        "source_identity_hash": source_identity_hash,
        "text_hash": text_hash,
        "text_hash_algorithm": str(request.get("text_hash_algorithm") or ""),
        "text_hash_status": text_hash_status,
        "request_body_sha256": request_hash,
        "response_body_sha256": response_hash,
        "finish_reason": str(finish_reason) if finish_reason is not None else "",
        "timestamp_utc": timestamp_utc or _utc_now_iso(),
        "parse_status": "parsed",
        "price_snapshot": price_payload,
        "estimated_cost_usd": estimated,
        "actual_cost_usd": actual,
        "latency_seconds": float(latency_seconds) if latency_seconds is not None else None,
        "schema_version": TEACHER_MODEL_OUTPUT_SCHEMA_VERSION,
        "label_unit": str(candidate.get("label_unit") or "document"),
        "label": parsed["label"],
        "borderline": parsed["borderline"],
        "model_output": {"label": parsed["label"], "borderline": parsed["borderline"]},
        "usage": response_body.get("usage", {}),
        "prompt_variant": str(request.get("prompt_variant") or PROMPT_VARIANT_STANDARD),
        "retry_attempt": int(request.get("retry_attempt") or 0),
        "retry_of_idempotency_key": str(request.get("retry_of_idempotency_key") or ""),
        "retry_reason": str(request.get("retry_reason") or ""),
        "raw_text_serialized": False,
    }
    row["label_row_hash"] = _hash(row)
    return row


def teacher_label_parse_error_record(
    *,
    error_code: str,
    error_message: str,
    candidate: dict[str, Any],
    request: dict[str, Any],
    response_body: dict[str, Any],
    timestamp_utc: str | None = None,
    finish_reason: str | None = None,
    schema_failure_path: str | None = None,
    schema_failure_kind: str | None = None,
    schema_failure_rule: str | None = None,
    schema_failure_value_sha256: str | None = None,
    schema_missing_keys: list[str] | None = None,
    schema_extra_key_sha256s: list[str] | None = None,
    schema_extra_key_count: int | None = None,
    content_sha256: str | None = None,
    content_char_length: int | None = None,
    usage: dict[str, Any] | None = None,
    estimated_cost_usd: float | None = None,
    actual_cost_usd: float | None = None,
    latency_seconds: float | None = None,
) -> dict[str, Any]:
    """Build a no-raw-text parse failure record. The failed content is hashed only."""

    candidate_id = str(candidate.get("candidate_id") or "")
    request_hash = str(request.get("request_body_sha256") or "")
    idempotency_key = str(request.get("idempotency_key") or "")
    teacher_model = ((request.get("body") or {}).get("model") if isinstance(request.get("body"), dict) else None)
    teacher_prompt_version = request.get("prompt_version")
    source_doc_id = str(request.get("source_doc_id") or candidate.get("source_doc_id") or "")
    source_identity_hash = str(request.get("source_identity_hash") or candidate.get("source_identity_hash") or "")
    text_hash = str(request.get("text_hash") or candidate.get("text_hash") or "")
    text_hash_status = str(request.get("text_hash_status") or ("provided:text_hash" if text_hash else "missing"))
    row = {
        "label_parse_error_version": 1,
        "parse_status": "parse_error",
        "error_code": str(error_code),
        "error_message_sha256": _hash(str(error_message).split(":")[0]),
        "candidate_id": candidate_id,
        "idempotency_key": idempotency_key,
        "teacher_model": str(teacher_model or ""),
        "teacher_prompt_version": str(teacher_prompt_version or ""),
        "source_candidate_manifest_hash": str(request.get("source_candidate_manifest_hash") or ""),
        "provider_config_hash": str(request.get("config_hash") or ""),
        "source_doc_id": source_doc_id,
        "source_identity_hash": source_identity_hash,
        "text_hash": text_hash,
        "text_hash_algorithm": str(request.get("text_hash_algorithm") or ""),
        "text_hash_status": text_hash_status,
        "request_body_sha256": request_hash,
        "prompt_variant": str(request.get("prompt_variant") or PROMPT_VARIANT_STANDARD),
        "retry_attempt": int(request.get("retry_attempt") or 0),
        "retry_of_idempotency_key": str(request.get("retry_of_idempotency_key") or ""),
        "retry_reason": str(request.get("retry_reason") or ""),
        "response_body_sha256": _hash(response_body),
        "timestamp_utc": timestamp_utc or _utc_now_iso(),
        "raw_text_serialized": False,
        "repair_attempted": False,
        "semantic_retry_performed": False,
        "repair_queue_status": "queued",
        "repair_queue_reason": str(error_code),
        "rerun_required": True,
    }
    if finish_reason is not None:
        row["finish_reason"] = str(finish_reason)
    if schema_failure_path is not None:
        row["schema_failure_path"] = str(schema_failure_path)
    if schema_failure_kind is not None:
        row["schema_failure_kind"] = str(schema_failure_kind)
    if schema_failure_rule is not None:
        row["schema_failure_rule"] = str(schema_failure_rule)
    if schema_failure_value_sha256 is not None:
        row["schema_failure_value_sha256"] = str(schema_failure_value_sha256)
    if schema_missing_keys is not None:
        row["schema_missing_keys"] = [str(key) for key in schema_missing_keys]
    if schema_extra_key_sha256s is not None:
        row["schema_extra_key_sha256s"] = [str(key_hash) for key_hash in schema_extra_key_sha256s]
    if schema_extra_key_count is not None:
        row["schema_extra_key_count"] = int(schema_extra_key_count)
    if content_sha256 is not None:
        row["content_sha256"] = str(content_sha256)
    if content_char_length is not None:
        row["content_char_length"] = int(content_char_length)
    if usage is not None:
        row["usage"] = usage if isinstance(usage, dict) else {}
    if estimated_cost_usd is not None:
        row["estimated_cost_usd"] = float(estimated_cost_usd)
    if actual_cost_usd is not None:
        row["actual_cost_usd"] = float(actual_cost_usd)
    if latency_seconds is not None:
        row["latency_seconds"] = float(latency_seconds)
    row["label_parse_error_hash"] = _hash(row)
    return row


def _validate_teacher_label_json(parsed: dict[str, Any], *, candidate_label_unit: str) -> None:
    if not isinstance(parsed, dict):
        raise TeacherSchemaValidationError(
            "teacher label JSON must be an object",
            path="$",
            kind="invalid_type",
            rule="top_level_object",
        )
    required = {"label", "borderline"}
    if set(parsed) != required:
        extra = sorted(set(parsed) - required)
        missing = sorted(required - set(parsed))
        raise TeacherSchemaValidationError(
            "teacher label JSON must contain only label and borderline",
            path="$",
            kind="schema_keys_mismatch",
            rule="exact_model_output_keys",
            missing_keys=missing,
            extra_keys=extra,
        )
    if candidate_label_unit not in ALLOWED_LABEL_UNITS:
        raise ValueError("candidate label_unit must be document or window_bundle")
    if parsed["label"] not in {"safe", "reject"}:
        raise TeacherSchemaValidationError(
            "invalid teacher label",
            path="$.label",
            kind="invalid_enum_value",
            rule="label_allowed_enum",
            value=parsed["label"],
        )
    if not isinstance(parsed["borderline"], bool):
        raise TeacherSchemaValidationError(
            "teacher borderline must be boolean",
            path="$.borderline",
            kind="invalid_type",
            rule="borderline_boolean",
            value=parsed["borderline"],
        )


def _string_looks_like_raw_text(value: str) -> bool:
    if TOKEN_SHAPED_SECRET_RE.search(value):
        return True
    if any(char.isspace() for char in value):
        return True
    return len(value) > 64


def should_retry_deepseek_status(status_code: int) -> bool:
    return int(status_code) in RETRYABLE_DEEPSEEK_STATUS_CODES


def run_with_retries(
    call: Callable[[dict[str, Any]], dict[str, Any]],
    request: dict[str, Any],
    *,
    max_attempts: int = 4,
    base_backoff_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run an injected fake/live call with documented DeepSeek retry semantics."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return call(request)
        except DeepSeekHTTPError as exc:
            last_error = exc
            if not should_retry_deepseek_status(exc.status_code) or attempt == max_attempts:
                raise
            sleep(base_backoff_seconds * (2 ** (attempt - 1)))
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry loop exited unexpectedly")


class DeepSeekHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message or f"DeepSeek HTTP {status_code}")
        self.status_code = int(status_code)


def prepare_deepseek_live_call(
    request: dict[str, Any],
    *,
    live: bool,
    max_docs: int | None,
    budget_cap_usd: float | None,
    env: dict[str, str] | None = None,
    api_key_env: str = "DEEPSEEK_API_KEY",
) -> dict[str, Any]:
    """Validate live-call unlocks without exposing the API key in public records."""

    env = dict(os.environ if env is None else env)
    if live is not True:
        raise ValueError("DeepSeek live calls require explicit --live")
    if max_docs is None or int(max_docs) < 1:
        raise ValueError("DeepSeek live calls require explicit --max-docs")
    if budget_cap_usd is None or float(budget_cap_usd) <= 0:
        raise ValueError("DeepSeek live calls require a positive explicit budget cap")
    api_key = env.get(api_key_env)
    if not api_key:
        raise ValueError(f"DeepSeek live calls require API key environment variable {api_key_env}")
    base_url = str(request.get("base_url") or DEEPSEEK_BASE_URL).rstrip("/")
    endpoint = str(request.get("endpoint") or DEEPSEEK_ENDPOINT_PATH)
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return {
        "live": True,
        "max_docs": int(max_docs),
        "budget_cap_usd": float(budget_cap_usd),
        "url": base_url + endpoint,
        "headers": {
            "Authorization": "<redacted>",
            "Content-Type": "application/json",
        },
        "api_key_env": api_key_env,
    }


def call_deepseek_live(
    request: dict[str, Any],
    *,
    live: bool,
    max_docs: int | None,
    budget_cap_usd: float | None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Dormant live client shell. Tests and dry-runs must not call this."""

    prepared = prepare_deepseek_live_call(
        request,
        live=live,
        max_docs=max_docs,
        budget_cap_usd=budget_cap_usd,
        env=env,
    )
    api_key = (os.environ if env is None else env)["DEEPSEEK_API_KEY"]
    http_request = urllib_request.Request(
        prepared["url"],
        data=json.dumps(request["body"]).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout_seconds = int(request.get("request_timeout_seconds") or 600)
    try:
        with urllib_request.urlopen(http_request, timeout=timeout_seconds) as response:  # nosec: guarded paid path
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        raise DeepSeekHTTPError(int(exc.code)) from exc


def _final_source_locator_index(source_schedules: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    locators: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for schedule in source_schedules:
        if schedule.get("raw_text_serialized") is not False or schedule.get("token_serialized") is not False:
            raise ValueError("source locator schedules must set raw_text_serialized=false and token_serialized=false")
        for entry in schedule.get("entries", []):
            source_file_id = str(entry.get("source_file_id") or "")
            if not source_file_id:
                continue
            locator = {
                "dataset_repo_id": entry.get("dataset_repo_id"),
                "config": entry.get("config"),
                "split": entry.get("split"),
                "resolved_repo_sha": entry.get("resolved_repo_sha"),
                "source_file_id": source_file_id,
                "parquet_url_or_path": entry.get("parquet_url_or_path"),
                "parquet_identity": entry.get("parquet_identity"),
            }
            if not locator["parquet_url_or_path"] or locator["parquet_identity"] is None:
                continue
            existing = locators.get(source_file_id)
            if existing is not None and existing != locator:
                conflicts.add(source_file_id)
                continue
            locators[source_file_id] = locator
    if conflicts:
        raise ValueError(f"source locator schedules map a source_file_id to multiple locators: conflict_count={len(conflicts)}")
    return locators


def _final_row_group_bounds_index(
    source_schedules: list[dict[str, Any]],
    *,
    row_group_bounds_artifacts: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, int], dict[str, Any]]:
    bounds: dict[tuple[str, int], dict[str, Any]] = {}
    conflicts: set[tuple[str, int]] = set()
    for schedule in source_schedules:
        for entry in schedule.get("entries", []):
            source_file_id = str(entry.get("source_file_id") or "")
            if not source_file_id or entry.get("row_group_index") is None or entry.get("row_group_num_rows") is None:
                continue
            key = (source_file_id, int(entry["row_group_index"]))
            payload = {
                "row_group_num_rows": int(entry["row_group_num_rows"]),
                "row_group_size_estimate_bytes": entry.get("row_group_size_estimate_bytes") or {},
            }
            existing = bounds.get(key)
            if existing is not None and existing != payload:
                conflicts.add(key)
                continue
            bounds[key] = payload
    recovered_bounds = _final_row_group_bounds_artifact_index(row_group_bounds_artifacts)
    for key, payload in recovered_bounds.items():
        comparable = {
            "row_group_num_rows": int(payload["row_group_num_rows"]),
            "row_group_size_estimate_bytes": payload.get("row_group_size_estimate_bytes") or {},
        }
        existing = bounds.get(key)
        if existing is not None and existing != comparable:
            conflicts.add(key)
            continue
        bounds[key] = comparable
    if conflicts:
        raise ValueError(f"source locator schedules contain conflicting row-group bounds: conflict_count={len(conflicts)}")
    return bounds


def _final_candidate_schedule_duplicate_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "duplicate_candidate_id_count": _duplicate_nonempty_count([str(candidate.get("candidate_id") or "") for candidate in candidates]),
        "duplicate_physical_row_key_count": _duplicate_nonempty_count([str(candidate.get("physical_source_row_key_hash") or "") for candidate in candidates]),
        "duplicate_text_hash_count": _duplicate_nonempty_count(
            [str(candidate.get("text_hash_sha256") or candidate.get("text_hash") or "") for candidate in candidates]
        ),
        "duplicate_normalized_text_hash_count": _duplicate_nonempty_count(
            [str(candidate.get("normalized_text_hash_sha256") or "") for candidate in candidates]
        ),
    }


def _final_refetch_schedule_entry(
    *,
    global_index: int,
    source_metadata_hash: str,
    locator: dict[str, Any],
    row_group_index: int,
    row_group_bounds: dict[str, Any],
    offsets: list[int],
) -> dict[str, Any]:
    base = {
        "schedule_entry_schema_version": 1,
        "global_schedule_index": int(global_index),
        "schedule_pass_index": 0,
        "dataset_repo_id": locator.get("dataset_repo_id"),
        "config": locator.get("config"),
        "split": locator.get("split"),
        "resolved_repo_sha": locator.get("resolved_repo_sha"),
        "source_file_id": locator.get("source_file_id"),
        "parquet_url_or_path": locator.get("parquet_url_or_path"),
        "parquet_identity": locator.get("parquet_identity"),
        "row_group_index": int(row_group_index),
        "row_group_num_rows": int(row_group_bounds["row_group_num_rows"]),
        "row_group_size_estimate_bytes": row_group_bounds.get("row_group_size_estimate_bytes") or {},
        "row_selection_mode": "explicit_offsets",
        "row_group_local_offsets": sorted(int(offset) for offset in offsets),
        "offset_recipe": {
            "mode": "explicit_offsets",
            "source": "final_35k_candidate_identity_ledger",
            "count": len(offsets),
        },
        "source_metadata_hash": source_metadata_hash,
        "freezable_required": True,
    }
    base["entry_hash"] = _hash(base)
    return base


def build_final_35k_candidate_refetch_schedule(
    candidates: list[dict[str, Any]],
    *,
    source_schedules: list[dict[str, Any]],
    row_group_bounds_artifacts: list[dict[str, Any]] | None = None,
    source_metadata_hash: str | None = None,
    expected_candidate_count: int = TEACHER_LABEL_FINAL_TARGET,
) -> dict[str, Any]:
    """Build the exact no-raw refetch schedule for the final teacher-label candidates."""

    if len(candidates) != int(expected_candidate_count):
        raise ValueError(
            "final candidate refetch schedule failed closed: "
            f"candidate_count_mismatch={abs(int(expected_candidate_count) - len(candidates)) or 1}"
        )
    for candidate in candidates:
        if _final_promotion_required_identity_missing(candidate):
            raise ValueError("final candidate refetch schedule failed closed: missing_required_identity_count=1")
    duplicate_counts = _final_candidate_schedule_duplicate_counts(candidates)
    _raise_final_promotion_blockers(duplicate_counts)
    locators = _final_source_locator_index(source_schedules)
    bounds = _final_row_group_bounds_index(source_schedules, row_group_bounds_artifacts=row_group_bounds_artifacts)
    schedule_source_metadata_hash = str(source_metadata_hash or "")
    if not schedule_source_metadata_hash:
        source_metadata_hashes = sorted({str(schedule.get("source_metadata_hash") or "") for schedule in source_schedules if schedule.get("source_metadata_hash")})
        if len(source_metadata_hashes) != 1:
            raise ValueError("final candidate refetch schedule requires a single source_metadata_hash")
        schedule_source_metadata_hash = source_metadata_hashes[0]

    missing_source_locator_count = 0
    missing_row_group_bounds_count = 0
    offset_out_of_bounds_count = 0
    grouped_offsets: dict[tuple[str, int], set[int]] = {}
    for candidate in candidates:
        source_file_id = str(candidate.get("source_file_id") or "")
        row_group_index = int(candidate["row_group_index"])
        offset = int(candidate["row_group_local_row_index"])
        if source_file_id not in locators:
            missing_source_locator_count += 1
            continue
        row_group_key = (source_file_id, row_group_index)
        row_group_bounds = bounds.get(row_group_key)
        if row_group_bounds is None:
            missing_row_group_bounds_count += 1
            continue
        if offset < 0 or offset >= int(row_group_bounds["row_group_num_rows"]):
            offset_out_of_bounds_count += 1
            continue
        grouped_offsets.setdefault(row_group_key, set()).add(offset)
    _raise_final_promotion_blockers(
        {
            "missing_source_locator_count": missing_source_locator_count,
            "missing_row_group_bounds_count": missing_row_group_bounds_count,
            "offset_out_of_bounds_count": offset_out_of_bounds_count,
        }
    )

    entries: list[dict[str, Any]] = []
    for global_index, row_group_key in enumerate(sorted(grouped_offsets)):
        source_file_id, row_group_index = row_group_key
        entries.append(
            _final_refetch_schedule_entry(
                global_index=global_index,
                source_metadata_hash=schedule_source_metadata_hash,
                locator=locators[source_file_id],
                row_group_index=row_group_index,
                row_group_bounds=bounds[row_group_key],
                offsets=sorted(grouped_offsets[row_group_key]),
            )
        )
    candidate_offset_keys = {
        (
            str(candidate.get("source_file_id") or ""),
            int(candidate["row_group_index"]),
            int(candidate["row_group_local_row_index"]),
        )
        for candidate in candidates
    }
    schedule_offset_keys = {
        (
            str(entry.get("source_file_id") or ""),
            int(entry["row_group_index"]),
            int(offset),
        )
        for entry in entries
        for offset in entry.get("row_group_local_offsets", [])
    }
    unused_scheduled_offset_count = len(schedule_offset_keys - candidate_offset_keys)
    missing_scheduled_offset_count = len(candidate_offset_keys - schedule_offset_keys)
    schedule = {
        "schema_version": FINAL_CANDIDATE_REFETCH_SCHEDULE_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "algorithm_version": FINAL_CANDIDATE_REFETCH_SCHEDULE_VERSION,
        "source_metadata_hash": schedule_source_metadata_hash,
        "candidate_count": len(candidates),
        "entry_count": len(entries),
        "row_group_count": len(entries),
        "offset_count": len(schedule_offset_keys),
        "coverage": {
            "candidate_count": len(candidates),
            "entry_count": len(entries),
            "scheduled_offset_count": len(schedule_offset_keys),
            "missing_scheduled_offset_count": missing_scheduled_offset_count,
            "unused_scheduled_offset_count": unused_scheduled_offset_count,
            "source_file_count": len({entry["source_file_id"] for entry in entries}),
        },
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "paid_calls_made": False,
        "deepseek_calls_made": False,
        "entries": entries,
    }
    stable = {key: value for key, value in schedule.items() if key not in {"generated_at_utc", "schedule_hash"}}
    schedule["schedule_hash"] = _hash(stable)
    validate_final_35k_candidate_refetch_schedule(schedule, candidates, expected_candidate_count=expected_candidate_count)
    return schedule


def validate_final_35k_candidate_refetch_schedule(
    schedule: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    expected_candidate_count: int = TEACHER_LABEL_FINAL_TARGET,
) -> None:
    if schedule.get("schema_version") != FINAL_CANDIDATE_REFETCH_SCHEDULE_SCHEMA_VERSION:
        raise ValueError("unsupported final candidate refetch schedule schema")
    if schedule.get("algorithm_version") != FINAL_CANDIDATE_REFETCH_SCHEDULE_VERSION:
        raise ValueError("unsupported final candidate refetch schedule algorithm")
    for flag in (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "labels_are_corpus_approval",
        "training_shards_created",
        "paid_calls_made",
        "deepseek_calls_made",
    ):
        if schedule.get(flag) is not False:
            raise ValueError(f"final candidate refetch schedule must set {flag}=false")
    stable = {key: value for key, value in schedule.items() if key not in {"generated_at_utc", "schedule_hash"}}
    if schedule.get("schedule_hash") != _hash(stable):
        raise ValueError("final candidate refetch schedule hash mismatch")
    if len(candidates) != int(expected_candidate_count):
        raise ValueError("final candidate refetch schedule candidate count mismatch")
    candidate_offset_keys = {
        (
            str(candidate.get("source_file_id") or ""),
            int(candidate["row_group_index"]),
            int(candidate["row_group_local_row_index"]),
        )
        for candidate in candidates
    }
    schedule_offset_keys = set()
    for entry in schedule.get("entries", []):
        if entry.get("raw_text_serialized") is True or entry.get("token_serialized") is True:
            raise ValueError("final candidate refetch schedule entry must be no-raw")
        offsets = [int(offset) for offset in entry.get("row_group_local_offsets", [])]
        if len(offsets) != len(set(offsets)):
            raise ValueError("final candidate refetch schedule entry contains duplicate offsets")
        for offset in offsets:
            if offset < 0 or offset >= int(entry["row_group_num_rows"]):
                raise ValueError("final candidate refetch schedule offset out of bounds")
            key = (str(entry.get("source_file_id") or ""), int(entry["row_group_index"]), int(offset))
            if key in schedule_offset_keys:
                raise ValueError("final candidate refetch schedule contains duplicate physical offset")
            schedule_offset_keys.add(key)
    if schedule_offset_keys != candidate_offset_keys:
        missing = len(candidate_offset_keys - schedule_offset_keys)
        unused = len(schedule_offset_keys - candidate_offset_keys)
        raise ValueError(
            "final candidate refetch schedule coverage mismatch: "
            f"missing_scheduled_offset_count={missing}, unused_scheduled_offset_count={unused}"
        )
    for candidate in candidates:
        _schedule_entry_for_candidate(candidate, schedule)


def _schedule_entry_for_candidate(candidate: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    if schedule.get("raw_text_serialized") is not False or schedule.get("token_serialized") is not False:
        raise ValueError("source schedule must be a no-raw-text artifact")
    candidate_schedule_hash = candidate.get("source_schedule_hash")
    schedule_hash = schedule.get("schedule_hash")
    schedule_hash_matches = not (candidate_schedule_hash and schedule_hash and candidate_schedule_hash != schedule_hash)
    schedule_entry_hash = candidate.get("schedule_entry_hash")
    if not schedule_entry_hash:
        raise ValueError("candidate is missing schedule_entry_hash")
    offset = candidate.get("row_group_local_row_index")
    if offset is None:
        raise ValueError("candidate is missing row_group_local_row_index")
    if schedule_hash_matches:
        matches = [
            entry
            for entry in schedule.get("entries", [])
            if entry.get("entry_hash") == schedule_entry_hash
        ]
        if len(matches) != 1:
            raise ValueError("candidate schedule_entry_hash did not resolve to exactly one schedule entry")
        entry = matches[0]
    else:
        if not candidate.get("physical_source_row_key_hash"):
            raise ValueError("candidate source schedule hash does not match supplied schedule")
        physical_key = _adaptive_physical_source_row_key(candidate)
        if str(candidate.get("physical_source_row_key_hash") or "") != _hash(physical_key):
            raise ValueError("candidate physical_source_row_key_hash drift")
        matches = []
        for entry_candidate in schedule.get("entries", []):
            offsets = {int(value) for value in entry_candidate.get("row_group_local_offsets", [])}
            if int(offset) not in offsets:
                continue
            try:
                entry_physical_key = _adaptive_schedule_physical_source_row_key(entry_candidate, offset=int(offset))
            except ValueError:
                continue
            if entry_physical_key == physical_key:
                matches.append(entry_candidate)
        if len(matches) != 1:
            raise ValueError("candidate physical source row did not resolve to exactly one schedule entry")
        entry = matches[0]
    if candidate.get("source_file_id") and candidate.get("source_file_id") != entry.get("source_file_id"):
        raise ValueError("candidate source_file_id drift against schedule entry")
    if candidate.get("row_group_index") is not None and int(candidate["row_group_index"]) != int(entry["row_group_index"]):
        raise ValueError("candidate row_group_index drift against schedule entry")
    if int(offset) not in {int(value) for value in entry.get("row_group_local_offsets", [])}:
        raise ValueError("candidate row_group_local_row_index is not scheduled for this entry")
    return entry


def refetch_candidate_private_text(
    candidate: dict[str, Any],
    *,
    schedule: dict[str, Any],
    provider: Any,
) -> tuple[str, dict[str, Any]]:
    """Refetch one candidate's private text without returning any serializable raw artifact."""

    entry = _schedule_entry_for_candidate(candidate, schedule)
    offset = int(candidate["row_group_local_row_index"])
    rows = provider.fetch_rows(entry, [offset])
    return _private_text_and_audit_from_refetched_row(candidate, rows.get(offset))


def _private_text_and_audit_from_refetched_row(
    candidate: dict[str, Any],
    row: Any,
) -> tuple[str, dict[str, Any]]:
    if isinstance(row, BaseException):
        raise ValueError(f"private refetch failed for scheduled row: {type(row).__name__}")
    if not isinstance(row, dict):
        raise ValueError("private refetch returned a non-object row")
    private_text = row.get("text")
    if not isinstance(private_text, str) or not private_text:
        raise ValueError("private refetch returned missing or empty text")
    computed_hash, text_hash_algorithm, had_expected_hash = _private_refetch_text_hash(candidate, private_text)
    expected_hash = candidate.get("text_hash") or candidate.get("text_hash_sha256")
    if had_expected_hash and str(expected_hash) != computed_hash:
        raise ValueError("private refetch text hash drift")
    expected_doc_id = candidate.get("source_doc_id")
    fetched_doc_id = row.get("id")
    source_doc_id_source = str(candidate.get("source_doc_id_source") or "")
    if (
        expected_doc_id
        and fetched_doc_id
        and source_doc_id_source.startswith("provided:")
        and str(expected_doc_id) != str(fetched_doc_id)
    ):
        raise ValueError("private refetch source_doc_id drift")
    offset = int(candidate["row_group_local_row_index"])
    audit = {
        "candidate_id": candidate["candidate_id"],
        "source_doc_id": str(expected_doc_id or fetched_doc_id or ""),
        "source_identity_hash": _candidate_source_identity_hash(candidate),
        "source_schedule_hash": candidate.get("source_schedule_hash"),
        "schedule_entry_hash": candidate.get("schedule_entry_hash"),
        "source_file_id": candidate.get("source_file_id"),
        "row_group_index": int(candidate["row_group_index"]),
        "row_group_local_row_index": offset,
        "text_hash": computed_hash,
        "text_hash_algorithm": text_hash_algorithm,
        "text_hash_status": "computed_from_private_text",
        "text_byte_length": len(private_text.encode("utf-8", errors="surrogatepass")),
        "raw_text_serialized": False,
    }
    _reject_live_artifact_leakage(audit)
    return private_text, audit


def _usage_token_count(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            return max(0, int(value))
    return 0


def actual_label_cost_usd(response_body: dict[str, Any], price: PriceSnapshot) -> float:
    usage = response_body.get("usage")
    if not isinstance(usage, dict):
        return 0.0
    prompt_tokens = _usage_token_count(usage, "prompt_tokens")
    cache_hit_tokens = _usage_token_count(usage, "prompt_cache_hit_tokens", "cache_hit_tokens")
    cache_miss_tokens = _usage_token_count(usage, "prompt_cache_miss_tokens", "cache_miss_tokens")
    if cache_hit_tokens + cache_miss_tokens == 0:
        cache_miss_tokens = prompt_tokens
    completion_tokens = _usage_token_count(usage, "completion_tokens", "output_tokens")
    input_cost = (
        cache_miss_tokens * price.input_cache_miss_per_million
        + cache_hit_tokens * price.input_cache_hit_per_million
    ) / 1_000_000
    output_cost = completion_tokens * price.output_per_million / 1_000_000
    return input_cost + output_cost


def _existing_idempotency_keys(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    keys: set[str] = set()
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("idempotency_key")
        if key:
            keys.add(str(key))
    return keys


def _finish_reason(response_body: dict[str, Any]) -> str:
    choices = response_body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        value = choices[0].get("finish_reason")
        if value is not None:
            return str(value)
    return "missing"


def _candidate_selection_summary(plan: dict[str, Any]) -> dict[str, Any]:
    selected = plan.get("selected_candidates", [])
    statuses: dict[str, int] = {}
    buckets: dict[str, int] = {}
    schedule_indices: list[int] = []
    source_files: set[str] = set()
    row_groups: set[tuple[str, int]] = set()
    for candidate in selected:
        status = str(candidate.get("decision_status") or "missing")
        statuses[status] = statuses.get(status, 0) + 1
        strata = candidate.get("candidate_strata") or []
        for stratum in strata:
            buckets[str(stratum)] = buckets.get(str(stratum), 0) + 1
        if candidate.get("bucket_id"):
            bucket_id = str(candidate["bucket_id"])
            buckets[bucket_id] = buckets.get(bucket_id, 0) + 1
        index = candidate.get("global_schedule_index")
        if index is not None:
            schedule_indices.append(int(index))
        source_file = candidate.get("source_file_id")
        if source_file:
            source_files.add(str(source_file))
            if candidate.get("row_group_index") is not None:
                row_groups.add((str(source_file), int(candidate["row_group_index"])))
    return {
        "candidate_count": len(selected),
        "decision_status_counts": dict(sorted(statuses.items())),
        "bucket_or_strata_counts": dict(sorted(buckets.items())),
        "source_file_count": len(source_files),
        "row_group_count": len(row_groups),
        "min_global_schedule_index": min(schedule_indices) if schedule_indices else None,
        "max_global_schedule_index": max(schedule_indices) if schedule_indices else None,
    }


def _write_json_atomic(path: str | Path, payload: dict[str, Any], *, allow_source_locator_urls: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_live_artifact_leakage(payload, allow_source_locator_urls=allow_source_locator_urls)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)


def _repair_queue_summary(path: str | Path) -> dict[str, Any]:
    output = Path(path)
    queued_count = 0
    by_error_code: dict[str, int] = {}
    by_finish_reason: dict[str, int] = {}
    by_schema_rule: dict[str, int] = {}
    if not output.exists():
        return {
            "queued_count": 0,
            "by_error_code": {},
            "by_finish_reason": {},
            "by_schema_failure_rule": {},
        }
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("parse_status") != "parse_error":
            continue
        queued_count += 1
        error_code = str(row.get("error_code") or "missing")
        by_error_code[error_code] = by_error_code.get(error_code, 0) + 1
        finish_reason = row.get("finish_reason")
        if finish_reason is not None:
            finish_key = str(finish_reason)
            by_finish_reason[finish_key] = by_finish_reason.get(finish_key, 0) + 1
        rule = row.get("schema_failure_rule")
        if rule is not None:
            rule_key = str(rule)
            by_schema_rule[rule_key] = by_schema_rule.get(rule_key, 0) + 1
    return {
        "queued_count": queued_count,
        "by_error_code": dict(sorted(by_error_code.items())),
        "by_finish_reason": dict(sorted(by_finish_reason.items())),
        "by_schema_failure_rule": dict(sorted(by_schema_rule.items())),
    }


def _completed_label_count(path: str | Path) -> int:
    output = Path(path)
    if not output.exists():
        return 0
    return len([line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()])


def _validate_existing_seed_label_row(
    row: dict[str, Any],
    candidate: dict[str, Any],
    *,
    manifest: dict[str, Any],
    provider_config: DeepSeekLabelingConfig,
    provider_config_hash: str,
) -> None:
    _reject_live_artifact_leakage(row)
    if row.get("seed_candidate_manifest_hash") != manifest["manifest_hash"]:
        raise ValueError("existing seed label output belongs to a different seed manifest")
    source_manifest_hash = manifest.get("source_candidate_manifest_hash")
    if source_manifest_hash and row.get("source_candidate_manifest_hash") != source_manifest_hash:
        raise ValueError("existing seed label output belongs to a different source manifest")
    if row.get("provider_config_hash") != provider_config_hash:
        raise ValueError("existing seed label output belongs to a different provider config")
    if row.get("teacher_model") != provider_config.model:
        raise ValueError("existing seed label output belongs to a different teacher model")
    if row.get("teacher_prompt_version") != provider_config.prompt_version:
        raise ValueError("existing seed label output belongs to a different teacher prompt")
    if row.get("prompt_variant") != PROMPT_VARIANT_STANDARD:
        raise ValueError("existing seed label output uses an unsupported prompt variant")
    if row.get("parse_status") not in {"parsed", "parse_error"}:
        raise ValueError("existing seed label output has unsupported parse_status")
    for key in ("seed_split", "seed_split_index", "seed_selection_index"):
        if row.get(key) != candidate.get(key):
            raise ValueError("existing seed label output seed split metadata does not match manifest")
    candidate_text_hash = candidate.get("text_hash") or candidate.get("text_hash_sha256")
    if candidate_text_hash and row.get("text_hash") != candidate_text_hash:
        raise ValueError("existing seed label output text hash does not match manifest")


def _read_existing_seed_label_rows(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    candidates: list[dict[str, Any]],
    provider_config: DeepSeekLabelingConfig,
    provider_config_hash: str,
) -> dict[str, dict[str, Any]]:
    output = Path(path)
    if not output.exists():
        return {}
    candidates_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    rows_by_candidate: dict[str, dict[str, Any]] = {}
    seen_idempotency_keys: set[str] = set()
    for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"existing seed label output row {line_number} must be an object")
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id not in candidates_by_id:
            raise ValueError("existing seed label output contains a candidate outside the seed manifest")
        if candidate_id in rows_by_candidate:
            raise ValueError("existing seed label output contains duplicate candidate rows")
        idempotency_key = str(row.get("idempotency_key") or "")
        if not idempotency_key:
            raise ValueError("existing seed label output row is missing idempotency_key")
        if idempotency_key in seen_idempotency_keys:
            raise ValueError("existing seed label output contains duplicate idempotency keys")
        seen_idempotency_keys.add(idempotency_key)
        _validate_existing_seed_label_row(
            row,
            candidates_by_id[candidate_id],
            manifest=manifest,
            provider_config=provider_config,
            provider_config_hash=provider_config_hash,
        )
        rows_by_candidate[candidate_id] = row
    missing_seen = False
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id not in rows_by_candidate:
            missing_seen = True
        elif missing_seen:
            raise ValueError("existing seed label output is not a prefix of the seed manifest order")
    return rows_by_candidate


def _append_label_row_known_unique(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_live_artifact_leakage(row)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _validate_seed_batch_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_candidate_manifest(manifest)
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != TEACHER_LABEL_SEED_TARGET:
        raise ValueError("seed label runner requires exactly 6000 candidates")
    readiness = manifest.get("label_readiness")
    if not isinstance(readiness, dict) or readiness.get("ready_for_paid_labeling") is not True:
        raise ValueError("seed label manifest is not ready for paid labeling")
    split = manifest.get("seed_split")
    if not isinstance(split, dict):
        raise ValueError("seed label manifest is missing seed_split")
    if int(split.get("seed_label_target") or 0) != TEACHER_LABEL_SEED_TARGET:
        raise ValueError("seed label manifest seed_split must target exactly 6000 labels")
    if int(split.get("provisional_training_labels") or 0) != TEACHER_LABEL_SEED_TRAIN_TARGET:
        raise ValueError("seed label manifest must contain 4500 provisional training labels")
    if int(split.get("labeled_holdout_labels") or 0) != TEACHER_LABEL_SEED_HOLDOUT_TARGET:
        raise ValueError("seed label manifest must contain 1500 labeled holdout labels")
    ids = [str(candidate.get("candidate_id") or "") for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("seed label manifest contains duplicate candidates")
    for candidate in candidates:
        if candidate.get("label_unit") not in ALLOWED_LABEL_UNITS:
            raise ValueError("seed label candidate has invalid label_unit")
        for key in ("source_schedule_hash", "schedule_entry_hash", "source_file_id", "row_group_index", "row_group_local_row_index"):
            if candidate.get(key) is None:
                raise ValueError(f"seed label candidate missing {key}")
    return candidates


def _attach_seed_split_metadata(
    row: dict[str, Any],
    candidate: dict[str, Any],
    *,
    finish_reason: str,
    manifest: dict[str, Any],
    provider_config_hash: str,
) -> dict[str, Any]:
    updated = dict(row)
    for key in ("seed_split", "seed_split_index", "seed_selection_index"):
        if key in candidate:
            updated[key] = candidate[key]
    updated["finish_reason"] = str(finish_reason)
    updated["seed_candidate_manifest_hash"] = manifest["manifest_hash"]
    updated["source_candidate_manifest_hash"] = str(manifest.get("source_candidate_manifest_hash") or "")
    updated["provider_config_hash"] = provider_config_hash
    if "label_row_hash" in updated:
        updated.pop("label_row_hash", None)
        updated["label_row_hash"] = _hash(updated)
    if "label_parse_error_hash" in updated:
        updated.pop("label_parse_error_hash", None)
        updated["label_parse_error_hash"] = _hash(updated)
    _reject_live_artifact_leakage(updated)
    return updated


def run_teacher_label_seed_batch(
    *,
    manifest: dict[str, Any],
    schedule: dict[str, Any],
    row_provider: Any,
    output_jsonl: str | Path,
    report_path: str | Path,
    live: bool,
    max_docs: int | None,
    budget_cap_usd: float,
    reserve_usd: float = 0.75,
    env: dict[str, str] | None = None,
    call: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    price_snapshot: PriceSnapshot | None = None,
    provider_config: DeepSeekLabelingConfig | None = None,
    avg_input_tokens: int = 2_400,
    avg_output_tokens: int = 512,
) -> dict[str, Any]:
    """Run the exact 6,000-label seed batch with canary, budget, and resume gates."""

    candidates = _validate_seed_batch_manifest(manifest)
    if max_docs != TEACHER_LABEL_SEED_TARGET:
        raise ValueError("--max-docs must equal 6000 for the seed label runner")
    if budget_cap_usd <= 0 or budget_cap_usd > 65.0:
        raise ValueError("seed label budget cap must be positive and no more than USD 65.00")
    if reserve_usd < 0 or reserve_usd >= budget_cap_usd:
        raise ValueError("seed label reserve must be non-negative and smaller than the budget cap")
    if schedule.get("raw_text_serialized") is not False or schedule.get("token_serialized") is not False:
        raise ValueError("source schedule must be a no-raw-text artifact")

    provider_config = provider_config or DeepSeekLabelingConfig()
    price = price_snapshot or _price_from_official_snapshot(provider_config.model)
    budget_check = estimate_label_cost_usd(
        label_count=TEACHER_LABEL_SEED_TARGET,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=price,
        safety_multiplier=1.25,
        budget_cap_usd=budget_cap_usd,
        reserve_usd=reserve_usd,
    )
    spend_limit_usd = float(budget_cap_usd) - float(reserve_usd)
    per_label_estimate = estimate_label_cost_usd(
        label_count=1,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=price,
        safety_multiplier=1.0,
        budget_cap_usd=budget_cap_usd,
        reserve_usd=0.0,
    )["estimated_cost_usd"]
    provider_payload = _public_config_payload(provider_config)
    provider_config_hash = _hash(_config_payload(provider_config))
    using_live_client = call is None or call is call_deepseek_live
    call = call or call_deepseek_live
    existing_rows_by_candidate = _read_existing_seed_label_rows(
        output_jsonl,
        manifest=manifest,
        candidates=candidates,
        provider_config=provider_config,
        provider_config_hash=provider_config_hash,
    )
    existing_keys = {
        str(row["idempotency_key"])
        for row in existing_rows_by_candidate.values()
    }
    first_existing_row = existing_rows_by_candidate.get(str(candidates[0]["candidate_id"]))

    run_status = "complete"
    blocked_reason: str | None = None
    if not budget_check["fits_after_reserve"]:
        run_status = "blocked_budget_cap"
        blocked_reason = "planned seed batch cost does not fit budget cap and reserve"
    if live is not True:
        run_status = "dry_run_no_paid_calls"
        blocked_reason = "explicit live flag not set"
    if first_existing_row is not None:
        first_parse_status = str(first_existing_row.get("parse_status") or "")
        first_finish_reason = str(first_existing_row.get("finish_reason") or "stop")
        if first_parse_status != "parsed":
            run_status = "stopped_after_canary"
            blocked_reason = "existing canary row did not parse cleanly"
        elif first_finish_reason != "stop":
            run_status = "stopped_after_canary"
            blocked_reason = "existing canary response finish_reason was not stop"

    paid_calls_made = 0
    skipped_existing_count = 0
    observed_cost = 0.0
    estimated_cost = 0.0
    latencies: list[float] = []
    parse_counts: dict[str, int] = {}
    finish_counts: dict[str, int] = {}
    refetch_audits: list[dict[str, Any]] = []

    if run_status == "complete":
        for index, candidate in enumerate(candidates):
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in existing_rows_by_candidate:
                skipped_existing_count += 1
                continue
            private_text, refetch_audit = refetch_candidate_private_text(
                candidate,
                schedule=schedule,
                provider=row_provider,
            )
            refetch_audits.append(refetch_audit)
            request_payload = build_deepseek_request(
                candidate,
                raw_text=private_text,
                config=provider_config,
                manifest_hash=manifest["manifest_hash"],
            )
            if request_payload["idempotency_key"] in existing_keys:
                skipped_existing_count += 1
                continue
            if observed_cost + float(per_label_estimate) > spend_limit_usd:
                run_status = "blocked_budget_cap"
                blocked_reason = "next planned call would exceed explicit USD spend limit"
                break
            prepare_deepseek_live_call(
                request_payload,
                live=live,
                max_docs=max_docs,
                budget_cap_usd=budget_cap_usd,
                env=env,
            )
            start = time.monotonic()
            response_body = run_with_retries(
                lambda prepared_request: call(
                    prepared_request,
                    live=live,
                    max_docs=max_docs,
                    budget_cap_usd=budget_cap_usd,
                    env=env,
                )
                if call is call_deepseek_live
                else call(prepared_request),
                request_payload,
                max_attempts=provider_config.max_retries + 1,
            )
            latency_seconds = time.monotonic() - start
            paid_calls_made += 1
            latencies.append(latency_seconds)
            finish_value = _finish_reason(response_body)
            finish_counts[finish_value] = finish_counts.get(finish_value, 0) + 1
            actual_cost_value = actual_label_cost_usd(response_body, price)
            observed_cost += actual_cost_value
            estimated_cost += float(per_label_estimate)
            row = parse_teacher_response(
                response_body,
                candidate=candidate,
                request=request_payload,
                price_snapshot=price,
                estimated_cost_usd=float(per_label_estimate),
                actual_cost_usd=actual_cost_value,
                latency_seconds=latency_seconds,
            )
            row = _attach_seed_split_metadata(
                row,
                candidate,
                finish_reason=finish_value,
                manifest=manifest,
                provider_config_hash=provider_config_hash,
            )
            _append_label_row_known_unique(output_jsonl, row)
            existing_keys.add(str(row["idempotency_key"]))
            parse_status = str(row.get("parse_status") or "missing")
            parse_counts[parse_status] = parse_counts.get(parse_status, 0) + 1
            if index == 0 and parse_status != "parsed":
                run_status = "stopped_after_canary"
                blocked_reason = "canary response did not parse cleanly"
                break
            if index == 0 and finish_value != "stop":
                run_status = "stopped_after_canary"
                blocked_reason = "canary response finish_reason was not stop"
                break
            if observed_cost > spend_limit_usd:
                run_status = "blocked_budget_cap"
                blocked_reason = "observed cost exceeded explicit USD spend limit"
                break

    documents_labeled = _completed_label_count(output_jsonl)
    repair_queue_summary = _repair_queue_summary(output_jsonl)
    latency_summary = {
        "count": len(latencies),
        "min_seconds": min(latencies) if latencies else None,
        "max_seconds": max(latencies) if latencies else None,
        "mean_seconds": sum(latencies) / len(latencies) if latencies else None,
    }
    report = {
        "seed_batch_report_schema_version": SEED_LABEL_BATCH_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "run_status": run_status,
        "blocked_reason": blocked_reason,
        "label_count_requested": TEACHER_LABEL_SEED_TARGET,
        "max_docs": max_docs,
        "paid_api_calls_made": paid_calls_made,
        "paid_api_calls_performed": bool(using_live_client and paid_calls_made > 0),
        "documents_labeled": documents_labeled,
        "skipped_existing_count": skipped_existing_count,
        "source_candidate_manifest_hash": manifest.get("source_candidate_manifest_hash"),
        "seed_candidate_manifest_hash": manifest["manifest_hash"],
        "seed_split": manifest["seed_split"],
        "provider_config": provider_payload,
        "provider_config_hash": provider_config_hash,
        "price_snapshot": asdict(price),
        "budget_cap_usd": float(budget_cap_usd),
        "reserve_usd": float(reserve_usd),
        "spend_limit_usd": spend_limit_usd,
        "budget_check": _redact_price_snapshot_urls(budget_check),
        "total_estimated_cost_usd": estimated_cost,
        "total_actual_cost_usd": observed_cost,
        "parse_status_counts": dict(sorted(parse_counts.items())),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "semantic_retry_counts": {},
        "repair_queue_summary": repair_queue_summary,
        "latency_summary": latency_summary,
        "refetch_audit_count": len(refetch_audits),
        "refetch_audit_hash": _hash(refetch_audits),
        "one_document_per_request": True,
        "transport_http_retries_allowed": True,
        "main_batch_semantic_retry_performed": False,
        "raw_text_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "corpus_shards_created": False,
        "classifier_training_performed": False,
        "commits_or_pushes_performed": False,
    }
    report["seed_batch_report_hash"] = _hash({key: value for key, value in report.items() if key != "generated_at_utc"})
    _write_json_atomic(report_path, report)
    return report


def _read_jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(row)
    return rows


def _parse_error_work_item(row: dict[str, Any], *, mode: str) -> dict[str, Any]:
    if row.get("parse_status") != "parse_error":
        raise ValueError("repair queue inputs must be parse_error rows")
    _reject_live_artifact_leakage(row)
    failed_idempotency_key = str(row.get("idempotency_key") or "")
    candidate_id = str(row.get("candidate_id") or "")
    request_hash = str(row.get("request_body_sha256") or "")
    response_hash = str(row.get("response_body_sha256") or row.get("response_hash") or "")
    if not failed_idempotency_key or not candidate_id or not request_hash or not response_hash:
        raise ValueError("parse_error rows require candidate, idempotency, request, and response hashes")
    work_item = {
        "repair_work_item_schema_version": 1,
        "repair_mode": mode,
        "candidate_id": candidate_id,
        "source_doc_id": str(row.get("source_doc_id") or ""),
        "source_identity_hash": str(row.get("source_identity_hash") or ""),
        "text_hash": str(row.get("text_hash") or ""),
        "text_hash_status": str(row.get("text_hash_status") or ""),
        "failed_idempotency_key": failed_idempotency_key,
        "teacher_model": str(row.get("teacher_model") or ""),
        "teacher_prompt_version": str(row.get("teacher_prompt_version") or ""),
        "prompt_variant": str(row.get("prompt_variant") or PROMPT_VARIANT_STANDARD),
        "retry_attempt": int(row.get("retry_attempt") or 0),
        "original_request_body_sha256": request_hash,
        "response_body_sha256": response_hash,
        "error_code": str(row.get("error_code") or "missing"),
        "repair_queue_reason": str(row.get("repair_queue_reason") or row.get("error_code") or "missing"),
        "finish_reason": str(row.get("finish_reason") or "missing"),
        "schema_failure_path": str(row.get("schema_failure_path") or ""),
        "schema_failure_kind": str(row.get("schema_failure_kind") or ""),
        "schema_failure_rule": str(row.get("schema_failure_rule") or ""),
        "schema_failure_value_sha256": row.get("schema_failure_value_sha256"),
        "schema_missing_keys": row.get("schema_missing_keys") or [],
        "schema_extra_key_sha256s": row.get("schema_extra_key_sha256s") or [],
        "schema_extra_key_count": int(row.get("schema_extra_key_count") or 0),
        "content_sha256": row.get("content_sha256"),
        "content_char_length": row.get("content_char_length"),
        "usage": row.get("usage") if isinstance(row.get("usage"), dict) else {},
        "estimated_cost_usd": float(row.get("estimated_cost_usd") or 0.0),
        "actual_cost_usd": float(row.get("actual_cost_usd") or 0.0),
        "latency_seconds": float(row["latency_seconds"]) if row.get("latency_seconds") is not None else None,
        "public_work_item_raw_text_serialized": False,
        "public_work_item_raw_model_response_serialized": False,
        "private_refetch_or_relabel_required": True,
        "instructions": [
            "Use this public work item only to identify the failed request.",
            "Fetch private source material only in an approved private workspace.",
            "Do not write raw corpus text or raw model response content to public artifacts.",
            "Write any resolved label as a new row and link it with a reconciliation event.",
        ],
    }
    if mode == "explicit_rerun":
        work_item["rerun_lineage"] = {
            "failed_idempotency_key": failed_idempotency_key,
            "failed_response_body_sha256": response_hash,
            "original_request_body_sha256": request_hash,
            "rerun_reason": work_item["repair_queue_reason"],
            "rerun_requires_new_idempotency_key": True,
        }
    _reject_live_artifact_leakage(work_item)
    work_item["repair_work_item_hash"] = _hash(work_item)
    return work_item


def build_parse_error_repair_queue(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    source_label_jsonl_sha256: str | None = None,
    max_manual_count: int = 25,
    price_snapshot: PriceSnapshot | None = None,
    avg_input_tokens: int = 2_400,
    avg_output_tokens: int = 512,
    budget_cap_usd: float = 65.0,
    reserve_usd: float = 0.75,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build a no-raw public repair/adjudication queue from parse_error rows."""

    if mode == "format_only_public":
        raise ValueError("format-only repair from public artifacts is forbidden because it requires raw malformed model output")
    if mode not in {"manual_adjudication", "explicit_rerun"}:
        raise ValueError("repair queue mode must be manual_adjudication or explicit_rerun")
    if max_manual_count < 1:
        raise ValueError("max_manual_count must be positive")
    parse_errors = [row for row in rows if row.get("parse_status") == "parse_error"]
    work_items = [_parse_error_work_item(row, mode=mode) for row in parse_errors]
    by_error_code: dict[str, int] = {}
    by_finish_reason: dict[str, int] = {}
    by_schema_rule: dict[str, int] = {}
    for item in work_items:
        error_code = str(item.get("error_code") or "missing")
        finish_reason = str(item.get("finish_reason") or "missing")
        schema_rule = str(item.get("schema_failure_rule") or "missing")
        by_error_code[error_code] = by_error_code.get(error_code, 0) + 1
        by_finish_reason[finish_reason] = by_finish_reason.get(finish_reason, 0) + 1
        by_schema_rule[schema_rule] = by_schema_rule.get(schema_rule, 0) + 1
    manual_allowed = mode == "manual_adjudication" and len(work_items) <= max_manual_count
    queue = {
        "repair_queue_schema_version": PARSE_ERROR_REPAIR_QUEUE_SCHEMA_VERSION,
        "generated_at_utc": timestamp_utc or _utc_now_iso(),
        "repair_queue_mode": mode,
        "source_label_jsonl_sha256": source_label_jsonl_sha256,
        "queued_count": len(work_items),
        "parse_error_row_count": len(parse_errors),
        "non_parse_error_input_count": len(rows) - len(parse_errors),
        "by_error_code": dict(sorted(by_error_code.items())),
        "by_finish_reason": dict(sorted(by_finish_reason.items())),
        "by_schema_failure_rule": dict(sorted(by_schema_rule.items())),
        "manual_adjudication": {
            "allowed": manual_allowed,
            "max_manual_count": max_manual_count,
            "requires_private_refetch_or_private_relabel": True,
            "recommended_for_small_counts": True,
            "blocked_reason": None if manual_allowed else "queued_count_exceeds_manual_threshold",
        },
        "format_only_repair_from_public_artifact": {
            "allowed": False,
            "reason": "raw_malformed_model_output_is_not_available_in_public_artifacts",
            "private_raw_response_required": True,
        },
        "work_items": work_items,
        "raw_text_serialized": False,
        "raw_model_response_serialized": False,
        "paid_api_calls_performed": False,
        "accepted_manifest": False,
        "training_shards_created": False,
    }
    if mode == "explicit_rerun":
        price = price_snapshot or _price_from_official_snapshot(REQUIRED_DEEPSEEK_TEACHER_MODEL)
        estimated_cost = estimate_label_cost_usd(
            label_count=len(work_items),
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            price=price,
            budget_cap_usd=budget_cap_usd,
            reserve_usd=reserve_usd,
        )
        queue["rerun_plan"] = {
            "rerun_plan_schema_version": 1,
            "requires_explicit_launch_unlock": True,
            "one_document_per_request": True,
            "semantic_retry_in_main_batch": False,
            "transport_http_retries_allowed": True,
            "strict_parser_remains_authority": True,
            "private_refetch_required_before_rerun": True,
            "new_idempotency_keys_required": True,
            "estimated_additional_cost": _redact_price_snapshot_urls(estimated_cost),
        }
    else:
        queue["rerun_plan"] = {
            "requires_explicit_launch_unlock": True,
            "available_if_manual_queue_too_large": True,
            "one_document_per_request": True,
        }
    stable = {key: value for key, value in queue.items() if key != "generated_at_utc"}
    queue["repair_queue_hash"] = _hash(stable)
    _reject_live_artifact_leakage(queue)
    return queue


def reconcile_repair_adjudication_row(
    original_parse_error: dict[str, Any],
    resolved_row: dict[str, Any],
    *,
    adjudication_method: str,
    adjudicator_id: str,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Link a repaired/adjudicated row to the historical parse_error without mutating either row."""

    if original_parse_error.get("parse_status") != "parse_error":
        raise ValueError("original row must be a parse_error row")
    if resolved_row.get("parse_status") != "parsed":
        raise ValueError("resolved row must be a parsed teacher-label row")
    if original_parse_error.get("candidate_id") != resolved_row.get("candidate_id"):
        raise ValueError("resolved row candidate_id must match original parse_error")
    event = {
        "repair_reconciliation_schema_version": REPAIR_RECONCILIATION_SCHEMA_VERSION,
        "generated_at_utc": timestamp_utc or _utc_now_iso(),
        "reconciliation_status": "linked",
        "adjudication_method": str(adjudication_method),
        "adjudicator_id": str(adjudicator_id),
        "original_parse_error": {
            "candidate_id": str(original_parse_error.get("candidate_id") or ""),
            "idempotency_key": str(original_parse_error.get("idempotency_key") or ""),
            "request_body_sha256": str(original_parse_error.get("request_body_sha256") or ""),
            "response_body_sha256": str(original_parse_error.get("response_body_sha256") or ""),
            "repair_queue_reason": str(original_parse_error.get("repair_queue_reason") or ""),
            "parse_error_row_hash": str(
                original_parse_error.get("label_parse_error_hash") or _hash(original_parse_error)
            ),
        },
        "resolved_row": {
            "candidate_id": str(resolved_row.get("candidate_id") or ""),
            "idempotency_key": str(resolved_row.get("idempotency_key") or ""),
            "request_body_sha256": str(resolved_row.get("request_body_sha256") or ""),
            "response_body_sha256": str(resolved_row.get("response_body_sha256") or ""),
            "label": str(resolved_row.get("label") or ""),
            "resolved_label_row_hash": str(resolved_row.get("label_row_hash") or _hash(resolved_row)),
        },
        "historical_parse_error_mutated": False,
        "resolved_row_replaces_historical_evidence": False,
        "raw_text_serialized": False,
        "raw_model_response_serialized": False,
    }
    stable = {key: value for key, value in event.items() if key != "generated_at_utc"}
    event["reconciliation_hash"] = _hash(stable)
    _reject_live_artifact_leakage(event)
    return event


def build_seed_launch_readiness_report(
    *,
    config: dict[str, Any],
    candidate_manifest_path: str | Path,
    repair_workflow_available: bool,
    paid_call_unlock: bool,
    launch_time_price_snapshot_verified: bool,
    manifest_loader: Callable[[str | Path], dict[str, Any]] = load_candidate_manifest,
    timestamp_utc: str | None = None,
    seed_label_target: int = TEACHER_LABEL_SEED_TARGET,
    train_label_count: int = TEACHER_LABEL_SEED_TRAIN_TARGET,
    holdout_label_count: int = TEACHER_LABEL_SEED_HOLDOUT_TARGET,
) -> dict[str, Any]:
    """Dry-run the exact 6k seed-label launch gates without performing paid calls."""

    blockers: list[str] = []
    warnings: list[str] = []
    normalized_config: dict[str, Any] | None = None
    try:
        normalized_config = validate_deepseek_provider_config(config)
        config_check = {"passed": True, "config_hash": normalized_config["config_hash"]}
    except Exception as exc:
        blockers.append("provider_config_invalid")
        config_check = {"passed": False, "error_class": type(exc).__name__, "error_message_sha256": _hash(str(exc))}
    budget = config.get("budget") if isinstance(config.get("budget"), dict) else {}
    configured_seed_target = int(budget.get("seed_label_target") or seed_label_target)
    configured_train_target = int(budget.get("provisional_training_labels") or train_label_count)
    configured_holdout_target = int(budget.get("labeled_holdout_labels") or holdout_label_count)
    if seed_label_target < 1 or train_label_count + holdout_label_count != seed_label_target:
        blockers.append("seed_split_config_invalid")
    if (
        configured_seed_target != seed_label_target
        or configured_train_target != train_label_count
        or configured_holdout_target != holdout_label_count
    ):
        blockers.append("seed_split_config_invalid")
    manifest_path = Path(candidate_manifest_path)
    manifest: dict[str, Any] | None = None
    manifest_error: str | None = None
    if not manifest_path.exists():
        blockers.append("candidate_manifest_missing")
    else:
        try:
            manifest = manifest_loader(manifest_path)
        except Exception as exc:
            manifest_error = type(exc).__name__
            blockers.append("candidate_manifest_invalid")
    candidates = manifest.get("candidates", []) if isinstance(manifest, dict) else []
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    planned_seed_count = min(seed_label_target, candidate_count)
    selected_seed_candidates = (
        _select_label_batch_candidates(candidates, label_count=planned_seed_count, seed=20260603)
        if isinstance(candidates, list) and planned_seed_count > 0
        else []
    )
    exact_count_passed = planned_seed_count == seed_label_target
    if not exact_count_passed and manifest is not None:
        blockers.append("exact_seed_count_unavailable")
    label_readiness = manifest.get("label_readiness") if isinstance(manifest, dict) else {}
    manifest_ready = bool(isinstance(label_readiness, dict) and label_readiness.get("ready_for_paid_labeling") is True)
    if manifest is not None and not manifest_ready:
        blockers.append("candidate_manifest_not_label_ready")
    private_refetch_required = 0
    if selected_seed_candidates:
        for candidate in selected_seed_candidates:
            text_hash_status = str(candidate.get("text_hash_status") or "")
            if candidate.get("private_refetch_required_before_live_call") is True or text_hash_status == "pending_private_refetch":
                private_refetch_required += 1
    if not repair_workflow_available:
        blockers.append("parse_error_repair_workflow_missing")
    if not paid_call_unlock:
        blockers.append("explicit_paid_call_unlock_missing")
    if not launch_time_price_snapshot_verified:
        blockers.append("launch_time_price_snapshot_missing")
    if config.get("request_policy", {}).get("semantic_output_retry_in_main_batch") is not False:
        blockers.append("main_batch_semantic_retry_policy_invalid")
    request_policy = config.get("request_policy") if isinstance(config.get("request_policy"), dict) else {}
    public_forbidden = config.get("public_artifact_forbidden") if isinstance(config.get("public_artifact_forbidden"), list) else []
    no_raw_policy_passed = (
        request_policy.get("dry_run_artifacts_store_raw_text") is False
        and request_policy.get("dry_run_artifacts_store_raw_urls") is False
        and "raw_text" in public_forbidden
        and "tokens" in public_forbidden
        and "accepted_manifest" in public_forbidden
    )
    if not no_raw_policy_passed:
        blockers.append("no_raw_public_artifact_policy_incomplete")
    if private_refetch_required:
        warnings.append("seed_candidates_require_private_refetch_before_live_calls")
    cost_estimate: dict[str, Any] | None = None
    if normalized_config is not None:
        pricing = config.get("pricing_usd_per_1m_tokens", {})
        price = PriceSnapshot(
            source=str(pricing.get("source") or "config_price_snapshot"),
            captured_at_utc=str(pricing.get("captured_at_utc") or ""),
            model=REQUIRED_DEEPSEEK_TEACHER_MODEL,
            input_cache_miss_per_million=float(pricing.get("input_cache_miss") or 0.0),
            input_cache_hit_per_million=float(pricing.get("input_cache_hit") or 0.0),
            output_per_million=float(pricing.get("output") or 0.0),
        )
        cost_estimate = _redact_price_snapshot_urls(
            estimate_label_cost_usd(
                label_count=seed_label_target,
                avg_input_tokens=2_400,
                avg_output_tokens=512,
                price=price,
                safety_multiplier=float(budget.get("batch_estimate_safety_multiplier") or 1.25),
                budget_cap_usd=float(budget.get("budget_cap_usd") or 65.0),
                reserve_usd=float(budget.get("reserved_credit_usd") or 0.75),
            )
        )
    checks = {
        "provider_config_policy": config_check,
        "exact_seed_count": {
            "passed": exact_count_passed,
            "required_count": seed_label_target,
            "observed_count": candidate_count,
            "planned_seed_label_count": planned_seed_count,
            "train_split_count": train_label_count,
            "holdout_split_count": holdout_label_count,
        },
        "candidate_manifest_schema": {
            "passed": manifest is not None and manifest_error is None,
            "manifest_hash": manifest.get("manifest_hash") if isinstance(manifest, dict) else None,
            "manifest_path_name": manifest_path.name,
            "error_class": manifest_error,
            "label_readiness": label_readiness if isinstance(label_readiness, dict) else {},
        },
        "budget_cost_estimate": {
            "passed": cost_estimate is not None,
            "estimate": cost_estimate,
            "launch_time_price_snapshot_verified": bool(launch_time_price_snapshot_verified),
        },
        "private_refetch": {
            "passed": manifest is not None,
            "private_refetch_required_count": private_refetch_required,
            "private_refetch_allowed_only_at_controlled_live_call_time": True,
        },
        "idempotency_resume_safety": {
            "passed": request_policy.get("idempotency_keys_required") is True
            and request_policy.get("resume_safe_jsonl_required") is True
            and request_policy.get("response_hashes_required") is True,
        },
        "parse_error_repair_workflow": {
            "passed": bool(repair_workflow_available),
            "main_batch_semantic_retry_permitted": False,
            "queued_repair_required": True,
        },
        "no_raw_public_artifact_policy": {
            "passed": no_raw_policy_passed,
            "forbidden_public_material_count": len(public_forbidden),
        },
        "paid_call_unlock": {
            "passed": bool(paid_call_unlock),
            "paid_api_calls_performed": False,
            "explicit_unlock_required": True,
        },
    }
    if not checks["idempotency_resume_safety"]["passed"]:
        blockers.append("idempotency_resume_policy_incomplete")
    report = {
        "seed_launch_readiness_schema_version": SEED_LAUNCH_READINESS_SCHEMA_VERSION,
        "generated_at_utc": timestamp_utc or _utc_now_iso(),
        "ready_for_seed_launch": not blockers,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "checks": checks,
        "seed_policy": {
            "seed_label_target": seed_label_target,
            "provisional_training_labels": train_label_count,
            "labeled_holdout_labels": holdout_label_count,
            "one_document_per_request": True,
            "strict_parser_authority": True,
            "transport_http_retries_allowed": True,
            "semantic_model_output_retries_in_main_batch": False,
        },
        "paid_api_calls_performed": False,
        "accepted_manifest": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    stable = {key: value for key, value in report.items() if key != "generated_at_utc"}
    report["seed_launch_readiness_hash"] = _hash(stable)
    _reject_live_artifact_leakage(report)
    return report


def run_teacher_label_pilot(
    *,
    manifest: dict[str, Any],
    schedule: dict[str, Any],
    row_provider: Any,
    output_jsonl: str | Path,
    report_path: str | Path,
    label_count: int = 50,
    live: bool,
    max_docs: int | None,
    budget_cap_usd: float,
    env: dict[str, str] | None = None,
    call: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    price_snapshot: PriceSnapshot | None = None,
    provider_config: DeepSeekLabelingConfig | None = None,
    seed: int = 20260603,
) -> dict[str, Any]:
    """Run a bounded DeepSeek teacher-label pilot with injected/refetched private text."""

    if label_count != 50:
        raise ValueError("this pilot runner is locked to exactly 50 labels")
    if max_docs != label_count:
        raise ValueError("--max-docs must equal the locked 50-document pilot size")
    if budget_cap_usd <= 0 or budget_cap_usd > 2.0:
        raise ValueError("pilot budget cap must be positive and no more than USD 2.00")
    provider_config = provider_config or DeepSeekLabelingConfig()
    price = price_snapshot or _price_from_official_snapshot(provider_config.model)
    plan = build_paid_label_batch_plan(
        manifest,
        label_count=label_count,
        price_snapshot=price,
        mode="pilot",
        seed=seed,
        avg_input_tokens=2_400,
        avg_output_tokens=512,
        budget_cap_usd=budget_cap_usd,
        reserve_usd=0.0,
        provider_config=provider_config,
        explicit_pilot_cap=label_count,
        budget_bucket="pilot_50",
    )
    if not plan["budget_check"]["fits_after_reserve"]:
        raise ValueError("planned pilot cost does not fit the explicit USD budget cap")

    existing_keys = _existing_idempotency_keys(output_jsonl)
    call = call or call_deepseek_live
    paid_calls_made = 0
    observed_cost = 0.0
    estimated_cost = 0.0
    latencies: list[float] = []
    parse_counts: dict[str, int] = {}
    finish_counts: dict[str, int] = {}
    refetch_audits: list[dict[str, Any]] = []
    run_status = "complete"
    blocked_reason: str | None = None

    for index, candidate in enumerate(plan["selected_candidates"]):
        private_text, refetch_audit = refetch_candidate_private_text(
            candidate,
            schedule=schedule,
            provider=row_provider,
        )
        refetch_audits.append(refetch_audit)
        request_payload = build_deepseek_request(candidate, raw_text=private_text, config=provider_config, manifest_hash=manifest["manifest_hash"])
        if request_payload["idempotency_key"] in existing_keys:
            continue
        planned_request = plan["planned_requests"][index]
        planned_cost = float(planned_request["estimated_cost_usd"])

        def run_one_attempt(attempt_request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, float, float]:
            nonlocal paid_calls_made, observed_cost, estimated_cost
            if observed_cost + planned_cost > budget_cap_usd:
                raise RuntimeError("next planned call would exceed explicit USD 2.00 pilot cap")
            prepare_deepseek_live_call(
                attempt_request,
                live=live,
                max_docs=max_docs,
                budget_cap_usd=budget_cap_usd,
                env=env,
            )
            start = time.monotonic()
            response_body = run_with_retries(
                lambda prepared_request: call(
                    prepared_request,
                    live=live,
                    max_docs=max_docs,
                    budget_cap_usd=budget_cap_usd,
                    env=env,
                )
                if call is call_deepseek_live
                else call(prepared_request),
                attempt_request,
                max_attempts=provider_config.max_retries + 1,
            )
            latency_seconds = time.monotonic() - start
            paid_calls_made += 1
            latencies.append(latency_seconds)
            finish_value = _finish_reason(response_body)
            finish_counts[finish_value] = finish_counts.get(finish_value, 0) + 1
            actual_cost_value = actual_label_cost_usd(response_body, price)
            observed_cost += actual_cost_value
            estimated_cost += planned_cost
            try:
                parsed_row = parse_teacher_response(
                    response_body,
                    candidate=candidate,
                    request=attempt_request,
                    price_snapshot=price,
                    estimated_cost_usd=planned_cost,
                    actual_cost_usd=actual_cost_value,
                    latency_seconds=latency_seconds,
                )
            except ValueError as exc:
                parsed_row = teacher_label_parse_error_record(
                    error_code="provider_response_rejected",
                    error_message=str(exc),
                    candidate=candidate,
                    request=attempt_request,
                    response_body=response_body,
                    finish_reason=finish_value,
                    usage=response_body.get("usage", {}),
                    estimated_cost_usd=planned_cost,
                    actual_cost_usd=actual_cost_value,
                    latency_seconds=latency_seconds,
                )
            return parsed_row, response_body, finish_value, actual_cost_value, latency_seconds

        try:
            row, response, finish, actual_cost, latency = run_one_attempt(request_payload)
        except RuntimeError as exc:
            run_status = "blocked_budget_cap"
            blocked_reason = str(exc)
            break
        append_label_row_resume_safe(output_jsonl, row)
        existing_keys.add(str(row["idempotency_key"]))
        parse_status = str(row.get("parse_status") or "missing")
        parse_counts[parse_status] = parse_counts.get(parse_status, 0) + 1
        if index == 0 and parse_status != "parsed":
            run_status = "stopped_after_canary"
            blocked_reason = "canary response did not parse cleanly"
            break
        if index == 0 and finish != "stop":
            run_status = "stopped_after_canary"
            blocked_reason = "canary response finish_reason was not stop"
            break
        if observed_cost > budget_cap_usd:
            run_status = "blocked_budget_cap"
            blocked_reason = "observed cost exceeded explicit USD 2.00 pilot cap"
            break
        if run_status == "blocked_budget_cap":
            break

    documents_labeled = len(_existing_idempotency_keys(output_jsonl))
    latency_summary = {
        "count": len(latencies),
        "min_seconds": min(latencies) if latencies else None,
        "max_seconds": max(latencies) if latencies else None,
        "mean_seconds": sum(latencies) / len(latencies) if latencies else None,
    }
    repair_queue_summary = _repair_queue_summary(output_jsonl)
    report = {
        "pilot_report_schema_version": TEACHER_LABEL_PILOT_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "run_status": run_status,
        "blocked_reason": blocked_reason,
        "label_count_requested": label_count,
        "paid_api_calls_made": paid_calls_made,
        "documents_labeled": documents_labeled,
        "source_candidate_manifest_hash": manifest["manifest_hash"],
        "batch_plan_hash": plan["batch_plan_hash"],
        "batch_plan_seed": seed,
        "candidate_selection_summary": _candidate_selection_summary(plan),
        "price_snapshot": asdict(price),
        "budget_cap_usd": budget_cap_usd,
        "total_estimated_cost_usd": estimated_cost,
        "total_actual_cost_usd": observed_cost,
        "parse_status_counts": dict(sorted(parse_counts.items())),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "semantic_retry_counts": {},
        "repair_queue_summary": repair_queue_summary,
        "latency_summary": latency_summary,
        "refetch_audit_count": len(refetch_audits),
        "refetch_audit_hash": _hash(refetch_audits),
        "raw_text_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "corpus_shards_created": False,
        "classifier_training_performed": False,
        "commits_or_pushes_performed": False,
    }
    report["pilot_report_hash"] = _hash({key: value for key, value in report.items() if key != "generated_at_utc"})
    _write_json_atomic(report_path, report)
    return report


def _with_bof_call_identity(request_payload: dict[str, Any], *, call_index: int, bof_n: int) -> dict[str, Any]:
    if call_index < 0 or call_index >= bof_n:
        raise ValueError("bof call_index must be within bof_n")
    updated = dict(request_payload)
    original_idempotency_key = str(request_payload["idempotency_key"])
    updated["bof_n"] = int(bof_n)
    updated["bof_call_index"] = int(call_index)
    updated["bof_group_key"] = _hash(["teacher_label_bof_group", original_idempotency_key, bof_n])
    updated["idempotency_key"] = _hash(["teacher_label_bof_vote", original_idempotency_key, bof_n, call_index])
    return updated


def _attach_bof_vote_metadata(row: dict[str, Any], request_payload: dict[str, Any], *, pilot_kind: str) -> dict[str, Any]:
    updated = dict(row)
    updated["bof_n"] = int(request_payload["bof_n"])
    updated["bof_call_index"] = int(request_payload["bof_call_index"])
    updated["bof_group_key"] = str(request_payload["bof_group_key"])
    updated["bof_pilot_kind"] = str(pilot_kind)
    updated["bof_vote_role"] = "independent_same_prompt_vote"
    updated["semantic_retry_performed"] = False
    if "label_row_hash" in updated:
        updated.pop("label_row_hash", None)
        updated["label_row_hash"] = _hash(updated)
    if "label_parse_error_hash" in updated:
        updated.pop("label_parse_error_hash", None)
        updated["label_parse_error_hash"] = _hash(updated)
    _reject_live_artifact_leakage(updated)
    return updated


def _bof_vote_pattern_summary(rows: list[dict[str, Any]], candidates: list[dict[str, Any]], *, bof_n: int) -> dict[str, Any]:
    by_candidate: dict[str, list[dict[str, Any]]] = {str(candidate["candidate_id"]): [] for candidate in candidates}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in by_candidate:
            by_candidate[candidate_id].append(row)

    per_candidate = []
    documents_with_label_vote_split = 0
    unanimous_documents = 0
    documents_with_borderline_disagreement = 0
    documents_with_parse_errors = 0
    borderline_true_votes = 0
    parsed_vote_count = 0
    label_vote_patterns: dict[str, int] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        votes = by_candidate[candidate_id]
        parsed = [row for row in votes if row.get("parse_status") == "parsed"]
        parse_errors = [row for row in votes if row.get("parse_status") == "parse_error"]
        label_counts: dict[str, int] = {}
        borderline_counts = {"true": 0, "false": 0}
        for row in parsed:
            label = str(row.get("label") or "missing")
            label_counts[label] = label_counts.get(label, 0) + 1
            if row.get("borderline") is True:
                borderline_counts["true"] += 1
                borderline_true_votes += 1
            elif row.get("borderline") is False:
                borderline_counts["false"] += 1
        parsed_vote_count += len(parsed)
        label_split = len(label_counts) > 1
        borderline_disagreement = borderline_counts["true"] > 0 and borderline_counts["false"] > 0
        unanimous = len(parsed) == bof_n and len(label_counts) == 1 and not parse_errors
        if label_split:
            documents_with_label_vote_split += 1
        if unanimous:
            unanimous_documents += 1
        if borderline_disagreement:
            documents_with_borderline_disagreement += 1
        if parse_errors:
            documents_with_parse_errors += 1
        pattern_key = ",".join(f"{label}:{count}" for label, count in sorted(label_counts.items())) or "no_parsed_votes"
        label_vote_patterns[pattern_key] = label_vote_patterns.get(pattern_key, 0) + 1
        per_candidate.append(
            {
                "candidate_id": candidate_id,
                "vote_count": len(votes),
                "parsed_vote_count": len(parsed),
                "parse_error_vote_count": len(parse_errors),
                "label_counts": dict(sorted(label_counts.items())),
                "label_vote_split": label_split,
                "unanimous": unanimous,
                "borderline_counts": borderline_counts,
                "borderline_true_rate": (
                    borderline_counts["true"] / len(parsed)
                    if parsed
                    else None
                ),
                "borderline_disagreement": borderline_disagreement,
            }
        )
    return {
        "documents": per_candidate,
        "documents_with_label_vote_split": documents_with_label_vote_split,
        "unanimous_documents": unanimous_documents,
        "documents_with_borderline_disagreement": documents_with_borderline_disagreement,
        "documents_with_parse_errors": documents_with_parse_errors,
        "label_vote_patterns": dict(sorted(label_vote_patterns.items())),
        "borderline_true_vote_rate": borderline_true_votes / parsed_vote_count if parsed_vote_count else None,
    }


def _token_usage_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
    }
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        totals["prompt_tokens"] += _usage_token_count(usage, "prompt_tokens")
        totals["prompt_cache_hit_tokens"] += _usage_token_count(usage, "prompt_cache_hit_tokens", "cache_hit_tokens")
        totals["prompt_cache_miss_tokens"] += _usage_token_count(usage, "prompt_cache_miss_tokens", "cache_miss_tokens")
        totals["completion_tokens"] += _usage_token_count(usage, "completion_tokens", "output_tokens")
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            totals["reasoning_tokens"] += _usage_token_count(details, "reasoning_tokens")
    return totals


def _row_cost_sum(rows: list[dict[str, Any]], field: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(field)
        if value is not None:
            total += float(value)
    return total


def _status_counts(rows: list[dict[str, Any]], field: str, *, missing_value: str = "missing") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or missing_value)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _latency_summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    latencies = [
        float(row["latency_seconds"])
        for row in rows
        if row.get("latency_seconds") is not None
    ]
    return {
        "count": len(latencies),
        "min_seconds": min(latencies) if latencies else None,
        "max_seconds": max(latencies) if latencies else None,
        "mean_seconds": sum(latencies) / len(latencies) if latencies else None,
    }


def _read_existing_bof_vote_rows(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    provider_config: DeepSeekLabelingConfig,
    candidates: list[dict[str, Any]],
    bof_n: int,
    pilot_kind: str,
) -> list[dict[str, Any]]:
    output = Path(path)
    if not output.exists():
        return []
    rows = _read_jsonl_objects(output)
    expected_manifest_hash = str(manifest["manifest_hash"])
    expected_config_hash = _hash(_config_payload(provider_config))
    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    seen_idempotency_keys: set[str] = set()
    seen_vote_keys: set[tuple[str, int]] = set()
    for row in rows:
        _reject_live_artifact_leakage(row)
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id not in candidate_ids:
            raise ValueError("existing BoF vote row contains a candidate outside the current pilot")
        if str(row.get("source_candidate_manifest_hash") or "") != expected_manifest_hash:
            raise ValueError("existing BoF vote row belongs to a different candidate manifest")
        if str(row.get("provider_config_hash") or "") != expected_config_hash:
            raise ValueError("existing BoF vote row belongs to a different provider config")
        if str(row.get("teacher_prompt_version") or "") != provider_config.prompt_version:
            raise ValueError("existing BoF vote row belongs to a different teacher prompt")
        if str(row.get("teacher_model") or "") != provider_config.model:
            raise ValueError("existing BoF vote row belongs to a different teacher model")
        if str(row.get("bof_pilot_kind") or "") != pilot_kind:
            raise ValueError("existing BoF vote row belongs to a different pilot kind")
        if int(row.get("bof_n") or -1) != bof_n:
            raise ValueError("existing BoF vote row belongs to a different bof_n")
        call_index = int(row.get("bof_call_index") if row.get("bof_call_index") is not None else -1)
        if call_index < 0 or call_index >= bof_n:
            raise ValueError("existing BoF vote row has invalid bof_call_index")
        idempotency_key = str(row.get("idempotency_key") or "")
        if not idempotency_key:
            raise ValueError("existing BoF vote row is missing idempotency_key")
        if idempotency_key in seen_idempotency_keys:
            raise ValueError("existing BoF vote rows contain duplicate idempotency_key")
        seen_idempotency_keys.add(idempotency_key)
        vote_key = (candidate_id, call_index)
        if vote_key in seen_vote_keys:
            raise ValueError("existing BoF vote rows contain duplicate candidate vote")
        seen_vote_keys.add(vote_key)
        if row.get("parse_status") not in {"parsed", "parse_error"}:
            raise ValueError("existing BoF vote row has unsupported parse_status")
    return rows


def run_teacher_label_bof_pilot(
    *,
    manifest: dict[str, Any],
    schedule: dict[str, Any],
    row_provider: Any,
    output_jsonl: str | Path,
    report_path: str | Path,
    label_count: int = 12,
    bof_n: int = 3,
    live: bool,
    max_docs: int | None,
    budget_cap_usd: float,
    env: dict[str, str] | None = None,
    call: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    price_snapshot: PriceSnapshot | None = None,
    provider_config: DeepSeekLabelingConfig | None = None,
    seed: int = 20260605,
    pilot_kind: str = "deepseek_v4_pro_bof_disagreement",
) -> dict[str, Any]:
    """Run a bounded best-of-N teacher-label pilot and write only no-raw public artifacts."""

    validate_candidate_manifest(manifest)
    if label_count < 1 or label_count > 50:
        raise ValueError("BoF pilot label_count must be between 1 and 50")
    if bof_n < 1 or bof_n > 5:
        raise ValueError("BoF pilot bof_n must be between 1 and 5")
    if max_docs != label_count:
        raise ValueError("--max-docs must equal BoF pilot label_count")
    if budget_cap_usd <= 0 or budget_cap_usd > 2.0:
        raise ValueError("BoF pilot budget cap must be positive and no more than USD 2.00")
    if schedule.get("raw_text_serialized") is not False or schedule.get("token_serialized") is not False:
        raise ValueError("source schedule must be a no-raw-text artifact")

    provider_config = provider_config or DeepSeekLabelingConfig()
    price = price_snapshot or _price_from_official_snapshot(provider_config.model)
    candidates = _select_label_batch_candidates(
        list(manifest.get("candidates") or []),
        label_count=min(label_count, len(manifest.get("candidates") or [])),
        seed=seed,
    )
    if len(candidates) != label_count:
        raise ValueError("BoF pilot candidate manifest does not contain enough candidates")
    vote_count = label_count * bof_n
    budget_check = estimate_label_cost_usd(
        label_count=vote_count,
        avg_input_tokens=2_400,
        avg_output_tokens=512,
        price=price,
        safety_multiplier=1.25,
        budget_cap_usd=budget_cap_usd,
        reserve_usd=0.0,
    )
    if not budget_check["fits_after_reserve"]:
        raise ValueError("planned BoF pilot cost does not fit the explicit USD budget cap")

    rows_for_report = _read_existing_bof_vote_rows(
        output_jsonl,
        manifest=manifest,
        provider_config=provider_config,
        candidates=candidates,
        bof_n=bof_n,
        pilot_kind=pilot_kind,
    )
    existing_keys = {str(row["idempotency_key"]) for row in rows_for_report}
    call = call or call_deepseek_live
    paid_calls_made = 0
    observed_cost = _row_cost_sum(rows_for_report, "actual_cost_usd")
    estimated_cost = _row_cost_sum(rows_for_report, "estimated_cost_usd")
    refetch_audits: list[dict[str, Any]] = []
    run_status = "complete"
    blocked_reason: str | None = None
    per_vote_estimate = estimate_label_cost_usd(
        label_count=1,
        avg_input_tokens=2_400,
        avg_output_tokens=512,
        price=price,
        safety_multiplier=1.0,
        budget_cap_usd=budget_cap_usd,
        reserve_usd=0.0,
    )["estimated_cost_usd"]

    for candidate in candidates:
        private_text, refetch_audit = refetch_candidate_private_text(candidate, schedule=schedule, provider=row_provider)
        refetch_audits.append(refetch_audit)
        base_request = build_deepseek_request(
            candidate,
            raw_text=private_text,
            config=provider_config,
            manifest_hash=manifest["manifest_hash"],
        )
        for call_index in range(bof_n):
            request_payload = _with_bof_call_identity(base_request, call_index=call_index, bof_n=bof_n)
            if request_payload["idempotency_key"] in existing_keys:
                continue
            if observed_cost + float(per_vote_estimate) > budget_cap_usd:
                run_status = "blocked_budget_cap"
                blocked_reason = "next planned BoF call would exceed explicit pilot budget cap"
                break
            prepare_deepseek_live_call(
                request_payload,
                live=live,
                max_docs=max_docs,
                budget_cap_usd=budget_cap_usd,
                env=env,
            )
            start = time.monotonic()
            response_body = run_with_retries(
                lambda prepared_request: call(
                    prepared_request,
                    live=live,
                    max_docs=max_docs,
                    budget_cap_usd=budget_cap_usd,
                    env=env,
                )
                if call is call_deepseek_live
                else call(prepared_request),
                request_payload,
                max_attempts=provider_config.max_retries + 1,
            )
            latency_seconds = time.monotonic() - start
            paid_calls_made += 1
            finish_value = _finish_reason(response_body)
            actual_cost_value = actual_label_cost_usd(response_body, price)
            observed_cost += actual_cost_value
            estimated_cost += float(per_vote_estimate)
            try:
                row = parse_teacher_response(
                    response_body,
                    candidate=candidate,
                    request=request_payload,
                    price_snapshot=price,
                    estimated_cost_usd=float(per_vote_estimate),
                    actual_cost_usd=actual_cost_value,
                    latency_seconds=latency_seconds,
                )
            except ValueError as exc:
                row = teacher_label_parse_error_record(
                    error_code="provider_response_rejected",
                    error_message=str(exc),
                    candidate=candidate,
                    request=request_payload,
                    response_body=response_body,
                    finish_reason=finish_value,
                    usage=response_body.get("usage", {}),
                    estimated_cost_usd=float(per_vote_estimate),
                    actual_cost_usd=actual_cost_value,
                    latency_seconds=latency_seconds,
                )
            row = _attach_bof_vote_metadata(row, request_payload, pilot_kind=pilot_kind)
            append_label_row_resume_safe(output_jsonl, row)
            rows_for_report.append(row)
            existing_keys.add(str(row["idempotency_key"]))
            if observed_cost > budget_cap_usd:
                run_status = "blocked_budget_cap"
                blocked_reason = "observed BoF cost exceeded explicit pilot budget cap"
                break
        if run_status != "complete":
            break

    latency_summary = _latency_summary_from_rows(rows_for_report)
    vote_pattern_summary = _bof_vote_pattern_summary(rows_for_report, candidates, bof_n=bof_n)
    report = {
        "bof_pilot_report_schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "pilot_kind": pilot_kind,
        "run_status": run_status,
        "blocked_reason": blocked_reason,
        "source_candidate_manifest_hash": manifest["manifest_hash"],
        "candidate_count": label_count,
        "bof_n": bof_n,
        "planned_vote_count": vote_count,
        "votes_recorded": len(rows_for_report),
        "paid_api_calls_made": paid_calls_made,
        "per_vote_parse_status_counts": _status_counts(rows_for_report, "parse_status"),
        "finish_reason_counts": _status_counts(rows_for_report, "finish_reason"),
        "semantic_retry_counts": {},
        "vote_pattern_summary": vote_pattern_summary,
        "source_bucket_diversity": _candidate_selection_summary({"selected_candidates": candidates}),
        "provider_config": _public_config_payload(provider_config),
        "provider_config_hash": _hash(_config_payload(provider_config)),
        "price_snapshot": asdict(price),
        "budget_cap_usd": float(budget_cap_usd),
        "budget_check": _redact_price_snapshot_urls(budget_check),
        "total_estimated_cost_usd": estimated_cost,
        "total_actual_cost_usd": observed_cost,
        "latency_summary": latency_summary,
        "token_usage_summary": _token_usage_summary(rows_for_report),
        "repair_queue_summary": _repair_queue_summary(output_jsonl),
        "refetch_audit_count": len(refetch_audits),
        "refetch_audit_hash": _hash(refetch_audits),
        "selected_candidate_ids_sha256": _hash([candidate["candidate_id"] for candidate in candidates]),
        "raw_text_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "corpus_shards_created": False,
        "classifier_training_performed": False,
        "commits_or_pushes_performed": False,
    }
    report["bof_pilot_report_hash"] = _hash({key: value for key, value in report.items() if key != "generated_at_utc"})
    _write_json_atomic(report_path, report)
    return report


def _adaptive_bof_group_key(
    base_request: dict[str, Any],
    *,
    policy: AdaptiveBoFMiningPolicy,
) -> str:
    return _hash(
        [
            "adaptive_bof_vote_group",
            policy.policy_version,
            base_request["source_candidate_manifest_hash"],
            base_request["candidate_id"],
            base_request["text_hash"],
            base_request["prompt_hash"],
            base_request["config_hash"],
            base_request["request_body_sha256"],
        ]
    )


def _with_adaptive_bof_vote_identity(
    request_payload: dict[str, Any],
    *,
    vote_index: int,
    policy: AdaptiveBoFMiningPolicy,
) -> dict[str, Any]:
    if vote_index < 0 or vote_index >= 5:
        raise ValueError("adaptive BoF vote_index must be between 0 and 4")
    updated = dict(request_payload)
    group_key = _adaptive_bof_group_key(request_payload, policy=policy)
    updated["bof_n"] = 5
    updated["bof_call_index"] = int(vote_index)
    updated["vote_index"] = int(vote_index)
    updated["adaptive_bof_group_key"] = group_key
    updated["bof_group_key"] = group_key
    updated["adaptive_bof_policy_version"] = policy.policy_version
    updated["adaptive_bof_stage"] = "bof3" if vote_index < 3 else "bof5_append"
    updated["idempotency_key"] = _hash(["adaptive_bof_vote", group_key, vote_index])
    return updated


def _attach_adaptive_bof_vote_metadata(row: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["bof_n"] = 5
    updated["bof_call_index"] = int(request_payload["bof_call_index"])
    updated["vote_index"] = int(request_payload["vote_index"])
    updated["adaptive_bof_group_key"] = str(request_payload["adaptive_bof_group_key"])
    updated["bof_group_key"] = str(request_payload["adaptive_bof_group_key"])
    updated["adaptive_bof_policy_version"] = str(request_payload["adaptive_bof_policy_version"])
    updated["adaptive_bof_stage"] = str(request_payload["adaptive_bof_stage"])
    updated["refetch_audit_hash"] = str(request_payload.get("refetch_audit_hash") or "")
    updated["bof_vote_role"] = "independent_same_prompt_vote"
    updated["semantic_retry_performed"] = False
    if "label_row_hash" in updated:
        updated.pop("label_row_hash", None)
        updated["label_row_hash"] = _hash(updated)
    if "label_parse_error_hash" in updated:
        updated.pop("label_parse_error_hash", None)
        updated["label_parse_error_hash"] = _hash(updated)
    _reject_live_artifact_leakage(updated)
    return updated


def _label_split_pattern(label_counts: dict[str, int]) -> str:
    ordered = []
    for label in ("safe", "reject"):
        if label_counts.get(label):
            ordered.append(f"{label}:{label_counts[label]}")
    for label, count in sorted(label_counts.items()):
        if label not in {"safe", "reject"}:
            ordered.append(f"{label}:{count}")
    return ",".join(ordered) if ordered else "no_parsed_votes"


def _majority_label(label_counts: dict[str, int]) -> str:
    safe = int(label_counts.get("safe", 0))
    reject = int(label_counts.get("reject", 0))
    if safe > reject:
        return "safe"
    return "reject"


def build_adaptive_bof_candidate_provenance(candidate_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize no-raw adaptive BoF votes for ranking and selected-manifest use."""

    _reject_live_artifact_leakage(rows)
    parsed = [row for row in rows if row.get("parse_status") == "parsed"]
    parse_errors = [row for row in rows if row.get("parse_status") == "parse_error"]
    vote_indexes = sorted({int(row.get("vote_index", row.get("bof_call_index", -1))) for row in rows})
    label_counts: dict[str, int] = {}
    borderline_true = 0
    borderline_false = 0
    for row in parsed:
        label = str(row.get("label") or "missing")
        label_counts[label] = label_counts.get(label, 0) + 1
        if row.get("borderline") is True:
            borderline_true += 1
        elif row.get("borderline") is False:
            borderline_false += 1
    parsed_count = len(parsed)
    complete_bof5 = parsed_count == 5 and not parse_errors and vote_indexes == [0, 1, 2, 3, 4]
    label_values = {label for label in label_counts if label in {"safe", "reject"}}
    label_vote_split = len(label_values) > 1
    borderline_disagreement = borderline_true > 0 and borderline_false > 0
    borderline_true_rate = borderline_true / parsed_count if parsed_count else None
    counts_sorted = sorted(label_counts.values(), reverse=True)
    tier = 99
    score = 0
    if complete_bof5:
        if counts_sorted == [3, 2] and borderline_disagreement:
            tier = 1
        elif counts_sorted == [3, 2]:
            tier = 2
        elif counts_sorted == [4, 1] and borderline_disagreement:
            tier = 3
        elif counts_sorted == [4, 1]:
            tier = 4
        elif len(counts_sorted) == 1 and (borderline_disagreement or borderline_true >= 3):
            tier = 6
        elif borderline_true >= 1:
            tier = 7
        score = (
            (100 - tier) * 10_000
            + (borderline_true * 100)
            + (50 if borderline_disagreement else 0)
            + (25 if label_vote_split else 0)
        )
    record = {
        "candidate_id": str(candidate_id),
        "vote_count": len(rows),
        "parsed_vote_count": parsed_count,
        "parse_error_vote_count": len(parse_errors),
        "complete_bof5_evidence": complete_bof5,
        "label_counts": dict(sorted(label_counts.items())),
        "final_label_majority": _majority_label(label_counts),
        "fail_closed_label": "reject" if not complete_bof5 else _majority_label(label_counts),
        "borderline_true_count": borderline_true,
        "borderline_false_count": borderline_false,
        "borderline_true_rate": borderline_true_rate,
        "borderline_disagreement": borderline_disagreement,
        "label_vote_split": label_vote_split,
        "split_pattern": _label_split_pattern(label_counts),
        "instability_tier": tier,
        "instability_score": score,
        "selected_manifest_eligible": bool(complete_bof5 and tier < 99),
        "vote_indexes": vote_indexes,
        "diagnostic_provisional_not_final_truth": True,
    }
    _reject_live_artifact_leakage(record)
    return record


def _adaptive_bof3_requires_bof5(rows: list[dict[str, Any]]) -> bool:
    first_three = [
        row
        for row in rows
        if int(row.get("vote_index", row.get("bof_call_index", -1))) in {0, 1, 2}
    ]
    parsed = [row for row in first_three if row.get("parse_status") == "parsed"]
    if len(parsed) < 3:
        return False
    labels = {str(row.get("label") or "") for row in parsed}
    borderline_true = sum(1 for row in parsed if row.get("borderline") is True)
    borderline_false = sum(1 for row in parsed if row.get("borderline") is False)
    return len(labels) > 1 or (borderline_true > 0 and borderline_false > 0) or borderline_true >= 2


def _adaptive_candidate_order(candidates: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    def row_group_batch_identity(candidate: dict[str, Any]) -> tuple[Any, ...]:
        source_file_id = candidate.get("source_file_id")
        row_group_index = candidate.get("row_group_index")
        if source_file_id is not None and row_group_index is not None:
            return ("source_file_row_group", str(source_file_id), int(row_group_index))
        parquet_identity_hash = candidate.get("parquet_identity_hash")
        if parquet_identity_hash is not None and row_group_index is not None:
            return ("parquet_hash_row_group", str(parquet_identity_hash), int(row_group_index))
        return ("schedule_entry", str(candidate.get("schedule_entry_hash") or ""), int(row_group_index or -1))

    return sorted(
        candidates,
        key=lambda candidate: (
            _hash({"seed": seed, "row_group_batch_identity": row_group_batch_identity(candidate)}),
            _hash({"seed": seed, "candidate_id": str(candidate.get("candidate_id") or "")}),
            str(candidate.get("candidate_id") or ""),
        ),
    )


def _adaptive_parquet_identity_hash(value: dict[str, Any]) -> str:
    existing = value.get("parquet_identity_hash")
    if existing:
        return str(existing)
    parquet_identity = value.get("parquet_identity")
    return _hash(parquet_identity) if parquet_identity else ""


def _adaptive_physical_source_row_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    try:
        return physical_source_row_key(candidate)
    except ValueError as exc:
        raise ValueError(f"adaptive BoF attempt candidate requires physical row identity: {exc}") from exc


def _adaptive_schedule_physical_source_row_key(
    entry: dict[str, Any],
    *,
    offset: int,
) -> tuple[Any, ...]:
    try:
        return physical_source_row_key({**entry, "row_group_local_row_index": int(offset)})
    except ValueError as exc:
        raise ValueError(f"adaptive BoF attempt schedule entry requires physical row identity: {exc}") from exc


def _adaptive_candidate_from_schedule_entry(
    entry: dict[str, Any],
    *,
    offset: int,
    schedule: dict[str, Any],
    seed: int,
    index: int,
) -> dict[str, Any]:
    schedule_hash = str(schedule.get("schedule_hash") or "")
    physical_key = _adaptive_schedule_physical_source_row_key(entry, offset=offset)
    candidate_id = _hash(
        [
            ADAPTIVE_BOF_ATTEMPT_MANIFEST_VERSION,
            physical_key,
            seed,
        ]
    )
    candidate = {
        "candidate_schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "source_doc_id": _hash(["adaptive_bof_source_doc", schedule_hash, entry.get("entry_hash"), int(offset)]),
        "source_schedule_hash": schedule_hash,
        "schedule_entry_hash": entry.get("entry_hash"),
        "global_schedule_index": int(entry.get("global_schedule_index") or index),
        "schedule_pass_index": int(entry.get("schedule_pass_index") or 0),
        "source_file_id": entry.get("source_file_id"),
        "row_group_index": int(entry.get("row_group_index") or 0),
        "row_group_local_row_index": int(offset),
        "label_unit": "document",
        "candidate_strata": ["adaptive_bof_schedule_expansion"],
        "bucket_id": "active_learning_borderline",
        "subbucket_id": "schedule_expansion",
        "teacher_label_use": "adaptive_bof_borderline_mining_only",
        "text_hash_status": "pending_private_refetch",
        "private_refetch_required_before_live_call": True,
        "selection_algorithm_version": ADAPTIVE_BOF_ATTEMPT_MANIFEST_VERSION,
        "parquet_identity_hash": _adaptive_parquet_identity_hash(entry),
        "physical_source_row_key_hash": _hash(physical_key),
    }
    for source_key, target_key in (
        ("dataset_repo_id", "dataset_repo_id"),
        ("config", "dataset_config"),
        ("split", "dataset_split"),
        ("resolved_repo_sha", "resolved_repo_sha"),
    ):
        if entry.get(source_key) is not None:
            candidate[target_key] = entry[source_key]
    candidate["candidate_hash"] = _hash(candidate)
    _reject_live_artifact_leakage(candidate)
    return candidate


def build_adaptive_bof_attempt_manifest(
    source_manifest: dict[str, Any],
    *,
    schedule: dict[str, Any],
    max_attempted_documents: int = 25_000,
    seed: int = 20260605,
) -> dict[str, Any]:
    """Build a no-raw adaptive mining attempt manifest from source candidates plus schedule locators."""

    validate_candidate_manifest(source_manifest)
    if max_attempted_documents < 1 or max_attempted_documents > 25_000:
        raise ValueError("adaptive BoF attempt manifest max_attempted_documents must be between 1 and 25000")
    if schedule.get("raw_text_serialized") is not False or schedule.get("token_serialized") is not False:
        raise ValueError("adaptive BoF attempt source schedule must be no-raw")
    schedule_hash = str(schedule.get("schedule_hash") or "")
    if not schedule_hash:
        raise ValueError("adaptive BoF attempt source schedule requires schedule_hash")
    selected: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    seen_rows: set[tuple[Any, ...]] = set()
    physical_duplicate_source_candidate_count = 0
    physical_duplicate_schedule_row_count = 0
    for candidate in _adaptive_candidate_order(list(source_manifest.get("candidates") or []), seed=seed):
        if len(selected) >= max_attempted_documents:
            break
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id or candidate_id in seen_candidate_ids:
            continue
        row_key = _adaptive_physical_source_row_key(candidate)
        if row_key in seen_rows:
            physical_duplicate_source_candidate_count += 1
            continue
        seen_candidate_ids.add(candidate_id)
        seen_rows.add(row_key)
        selected_candidate = dict(candidate)
        if candidate.get("candidate_hash"):
            selected_candidate["source_manifest_candidate_hash"] = candidate["candidate_hash"]
        if candidate.get("parquet_identity") or candidate.get("parquet_identity_hash"):
            selected_candidate["parquet_identity_hash"] = _adaptive_parquet_identity_hash(candidate)
        selected_candidate.pop("parquet_identity", None)
        selected_candidate["physical_source_row_key_hash"] = _hash(row_key)
        selected_candidate.pop("candidate_hash", None)
        selected_candidate["candidate_hash"] = _hash(selected_candidate)
        selected.append(selected_candidate)
    schedule_added = 0
    for entry in schedule.get("entries", []):
        if len(selected) >= max_attempted_documents:
            break
        entry_hash = entry.get("entry_hash")
        if not entry_hash:
            raise ValueError("adaptive BoF attempt schedule entry missing entry_hash")
        for offset in entry.get("row_group_local_offsets", []):
            if len(selected) >= max_attempted_documents:
                break
            row_key = _adaptive_schedule_physical_source_row_key(entry, offset=int(offset))
            if row_key in seen_rows:
                physical_duplicate_schedule_row_count += 1
                continue
            candidate = _adaptive_candidate_from_schedule_entry(
                entry,
                offset=int(offset),
                schedule=schedule,
                seed=seed,
                index=len(selected),
            )
            seen_rows.add(row_key)
            seen_candidate_ids.add(str(candidate["candidate_id"]))
            selected.append(candidate)
            schedule_added += 1
    shortfall = max(0, max_attempted_documents - len(selected))
    required_actions = [] if shortfall == 0 else ["provide_larger_no_raw_source_schedule"]
    manifest = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "selection_algorithm_version": ADAPTIVE_BOF_ATTEMPT_MANIFEST_VERSION,
        "source_candidate_manifest_hash": source_manifest["manifest_hash"],
        "source_schedule_hash": schedule_hash,
        "seed": int(seed),
        "max_attempted_documents": int(max_attempted_documents),
        "candidate_count": len(selected),
        "candidate_count_shortfall": shortfall,
        "source_manifest_candidate_count": len(source_manifest.get("candidates") or []),
        "schedule_expansion_candidate_count": schedule_added,
        "physical_row_dedupe_policy": "dataset_file_row_group_offset_ignores_schedule_hash_v1",
        "physical_duplicate_source_candidate_count": physical_duplicate_source_candidate_count,
        "physical_duplicate_schedule_row_count": physical_duplicate_schedule_row_count,
        "attempt_readiness": {
            "ready_for_adaptive_mining": shortfall == 0,
            "required_actions": required_actions,
            "notes": [
                "This is an adaptive BoF mining attempt manifest only.",
                "It is not an accepted corpus manifest, final 35k label set, or training shard.",
            ],
        },
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "candidates": selected,
    }
    manifest["manifest_hash"] = _hash(
        {key: value for key, value in manifest.items() if key not in {"generated_at_utc", "manifest_hash"}}
    )
    _reject_live_artifact_leakage(manifest)
    validate_candidate_manifest(manifest)
    return manifest


def _rows_by_candidate(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_id") or ""), []).append(row)
    return grouped


def _read_existing_adaptive_bof_rows(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    provider_config: DeepSeekLabelingConfig,
    candidates: list[dict[str, Any]],
    policy: AdaptiveBoFMiningPolicy,
) -> list[dict[str, Any]]:
    output = Path(path)
    if not output.exists():
        return []
    rows = _read_jsonl_objects(output)
    expected_manifest_hash = str(manifest["manifest_hash"])
    expected_config_hash = _hash(_config_payload(provider_config))
    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    seen_idempotency_keys: set[str] = set()
    seen_vote_keys: set[tuple[str, int]] = set()
    for row in rows:
        _reject_live_artifact_leakage(row)
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id not in candidate_ids:
            raise ValueError("existing adaptive BoF row contains a candidate outside the current run")
        if str(row.get("source_candidate_manifest_hash") or "") != expected_manifest_hash:
            raise ValueError("existing adaptive BoF row belongs to a different candidate manifest")
        if str(row.get("provider_config_hash") or "") != expected_config_hash:
            raise ValueError("existing adaptive BoF row belongs to a different provider config")
        if str(row.get("teacher_prompt_version") or "") != provider_config.prompt_version:
            raise ValueError("existing adaptive BoF row belongs to a different teacher prompt")
        if str(row.get("teacher_model") or "") != provider_config.model:
            raise ValueError("existing adaptive BoF row belongs to a different teacher model")
        if str(row.get("adaptive_bof_policy_version") or "") != policy.policy_version:
            raise ValueError("existing adaptive BoF row belongs to a different mining policy")
        vote_index = int(row.get("vote_index") if row.get("vote_index") is not None else row.get("bof_call_index", -1))
        if vote_index < 0 or vote_index >= 5:
            raise ValueError("existing adaptive BoF row has invalid vote_index")
        idempotency_key = str(row.get("idempotency_key") or "")
        if not idempotency_key:
            raise ValueError("existing adaptive BoF row is missing idempotency_key")
        if idempotency_key in seen_idempotency_keys:
            raise ValueError("existing adaptive BoF rows contain duplicate idempotency_key")
        seen_idempotency_keys.add(idempotency_key)
        vote_key = (candidate_id, vote_index)
        if vote_key in seen_vote_keys:
            raise ValueError("existing adaptive BoF rows contain duplicate candidate vote")
        seen_vote_keys.add(vote_key)
        if row.get("parse_status") not in {"parsed", "parse_error"}:
            raise ValueError("existing adaptive BoF row has unsupported parse_status")
    return rows


def _adaptive_selected_records(
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    selected_target: int,
) -> list[dict[str, Any]]:
    grouped = _rows_by_candidate(rows)
    candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    records = []
    for candidate_id, candidate_rows in grouped.items():
        if candidate_id not in candidate_by_id:
            continue
        provenance = build_adaptive_bof_candidate_provenance(candidate_id, candidate_rows)
        if not provenance["selected_manifest_eligible"]:
            continue
        candidate = candidate_by_id[candidate_id]
        provenance.update(
            {
                "source_candidate_hash": candidate.get("source_manifest_candidate_hash") or candidate.get("candidate_hash"),
                "attempt_candidate_hash": candidate.get("candidate_hash"),
                "bucket_id": candidate.get("bucket_id"),
                "subbucket_id": candidate.get("subbucket_id"),
                "label_unit": candidate.get("label_unit", "document"),
                "source_schedule_hash": candidate.get("source_schedule_hash"),
                "schedule_entry_hash": candidate.get("schedule_entry_hash"),
                "source_file_id": candidate.get("source_file_id"),
                "source_domain_hash": candidate.get("source_domain_hash")
                or (_hash(["source_domain", candidate.get("source_domain")]) if candidate.get("source_domain") else None),
                "row_group_index": candidate.get("row_group_index"),
                "row_group_local_row_index": candidate.get("row_group_local_row_index"),
                "metadata_bands": candidate.get("metadata_bands") if isinstance(candidate.get("metadata_bands"), dict) else {},
            }
        )
        records.append(provenance)
    records.sort(
        key=lambda row: (
            int(row["instability_tier"]),
            -int(row["instability_score"]),
            str(row["source_domain_hash"]),
            str(row["candidate_id"]),
        )
    )
    selected = records[:selected_target]
    _reject_live_artifact_leakage(selected)
    return selected


def _adaptive_selected_diversity_diagnostics(selected: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in selected:
            value = row.get(field)
            if value in {None, ""}:
                continue
            key = str(value)
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))

    metadata_band_counts: dict[str, dict[str, int]] = {}
    for row in selected:
        bands = row.get("metadata_bands")
        if not isinstance(bands, dict):
            continue
        for band_name, band_value in bands.items():
            if band_value in {None, ""}:
                continue
            band_counts = metadata_band_counts.setdefault(str(band_name), {})
            key = str(band_value)
            band_counts[key] = band_counts.get(key, 0) + 1
    metadata_band_counts = {
        band_name: dict(sorted(band_counts.items()))
        for band_name, band_counts in sorted(metadata_band_counts.items())
    }
    row_group_keys = {
        (
            str(row.get("source_file_id") or ""),
            str(row.get("row_group_index") if row.get("row_group_index") is not None else ""),
            str(row.get("row_group_local_row_index") if row.get("row_group_local_row_index") is not None else ""),
        )
        for row in selected
    }
    diagnostics = {
        "diagnostics_schema_version": 1,
        "selected_count": len(selected),
        "source_schedule_hash_count": len({str(row.get("source_schedule_hash") or "") for row in selected if row.get("source_schedule_hash")}),
        "source_file_count": len({str(row.get("source_file_id") or "") for row in selected if row.get("source_file_id")}),
        "source_domain_hash_count": len({str(row.get("source_domain_hash") or "") for row in selected if row.get("source_domain_hash")}),
        "missing_source_domain_hash_count": sum(1 for row in selected if not row.get("source_domain_hash")),
        "row_group_identity_count": len(row_group_keys),
        "bucket_counts": counts("bucket_id"),
        "subbucket_counts": counts("subbucket_id"),
        "label_unit_counts": counts("label_unit"),
        "metadata_band_counts": metadata_band_counts,
        "hard_cap_enforcement": "none",
        "diagnostic_only": True,
    }
    _reject_live_artifact_leakage(diagnostics)
    return diagnostics


def _adaptive_run_status(
    *,
    selected_count: int,
    selected_target: int,
    attempted_count: int,
    max_attempted: int,
    current_status: str,
    current_blocker: str | None,
) -> tuple[str, str | None]:
    if current_status != "complete":
        return current_status, current_blocker
    if selected_count >= selected_target:
        return "complete", None
    if attempted_count >= max_attempted:
        return "blocked_low_yield", "complete-BoF5 selected yield below target at max attempted documents"
    return "blocked_low_yield", "complete-BoF5 selected yield below target before additional candidate expansion"


def _write_adaptive_progress(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    policy: AdaptiveBoFMiningPolicy,
    provider_config_hash: str,
    attempted_count: int,
    rows: list[dict[str, Any]],
    selected_count: int,
    observed_cost: float,
    estimated_cost: float,
    run_status: str,
    blocked_reason: str | None,
) -> dict[str, Any]:
    progress = {
        "adaptive_bof_progress_schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "source_candidate_manifest_hash": manifest["manifest_hash"],
        "adaptive_bof_policy_version": policy.policy_version,
        "provider_config_hash": provider_config_hash,
        "attempted_document_count": attempted_count,
        "votes_recorded": len(rows),
        "selected_count": selected_count,
        "selected_target": policy.selected_target,
        "budget_cap_usd": policy.budget_cap_usd,
        "minimum_balance_reserve_usd": policy.minimum_balance_reserve_usd,
        "total_estimated_cost_usd": estimated_cost,
        "total_actual_cost_usd": observed_cost,
        "run_status": run_status,
        "blocked_reason": blocked_reason,
        "raw_text_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
    }
    progress["adaptive_bof_progress_hash"] = _hash({key: value for key, value in progress.items() if key != "generated_at_utc"})
    _reject_live_artifact_leakage(progress)
    _write_json_atomic(path, progress)
    return progress


def _write_adaptive_selected_manifest(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    policy: AdaptiveBoFMiningPolicy,
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    diversity_diagnostics = _adaptive_selected_diversity_diagnostics(selected)
    selected_manifest = {
        "adaptive_bof_selected_manifest_schema_version": ADAPTIVE_BOF_SELECTED_MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "manifest_scope": "adaptive_bof_borderline_candidate_selection",
        "source_candidate_manifest_hash": manifest["manifest_hash"],
        "adaptive_bof_policy_version": policy.policy_version,
        "selected_target": policy.selected_target,
        "selected_count": len(selected),
        "selected_candidates": selected,
        "source_vote_count": len(rows),
        "complete_bof5_required": True,
        "diversity_cap_policy": "diagnostic_only_no_hard_caps_for_live_mining_v1",
        "diversity_caps_enforced": False,
        "diversity_diagnostics": diversity_diagnostics,
        "diagnostic_provisional_not_final_truth": True,
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    selected_manifest["selected_candidate_ids_sha256"] = _hash([row["candidate_id"] for row in selected])
    selected_manifest["adaptive_bof_selected_manifest_hash"] = _hash(
        {key: value for key, value in selected_manifest.items() if key != "generated_at_utc"}
    )
    _reject_live_artifact_leakage(selected_manifest)
    _write_json_atomic(path, selected_manifest)
    return selected_manifest


def _write_adaptive_repair_queue(
    path: str | Path,
    *,
    parse_error_rows: list[dict[str, Any]],
    transport_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    queue = build_parse_error_repair_queue(
        parse_error_rows,
        mode="manual_adjudication",
        max_manual_count=max(25, len(parse_error_rows)),
    )
    queue["transport_failure_count"] = len(transport_failures)
    queue["transport_failure_work_items"] = transport_failures
    queue["queued_count"] = int(queue["queued_count"]) + len(transport_failures)
    queue["adaptive_bof_retry_queue_present"] = bool(transport_failures)
    queue["raw_text_serialized"] = False
    queue["raw_model_response_serialized"] = False
    queue["repair_queue_hash"] = _hash(
        {key: value for key, value in queue.items() if key != "generated_at_utc" and key != "repair_queue_hash"}
    )
    _reject_live_artifact_leakage(queue)
    _write_json_atomic(path, queue)
    return queue


def _adaptive_vote_request_tasks(
    *,
    manifest: dict[str, Any],
    schedule: dict[str, Any],
    row_provider: Any,
    missing_plan: list[tuple[dict[str, Any], tuple[int, ...]]],
    provider_config: DeepSeekLabelingConfig,
    policy: AdaptiveBoFMiningPolicy,
    refetch_concurrency: int = 1,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    def build_base_request(
        candidate: dict[str, Any],
        missing: tuple[int, ...],
        row: Any,
    ) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
        private_text, refetch_audit = _private_text_and_audit_from_refetched_row(candidate, row)
        base_request = build_deepseek_request(
            candidate,
            raw_text=private_text,
            config=provider_config,
            manifest_hash=manifest["manifest_hash"],
        )
        base_request["refetch_audit_hash"] = _hash(refetch_audit)
        return candidate, missing, base_request

    grouped: dict[str, tuple[dict[str, Any], list[tuple[int, dict[str, Any], tuple[int, ...]]]]] = {}
    for index, (candidate, missing) in enumerate(missing_plan):
        entry = _schedule_entry_for_candidate(candidate, schedule)
        group_key = str(entry.get("entry_hash") or _hash(entry))
        if group_key not in grouped:
            grouped[group_key] = (entry, [])
        grouped[group_key][1].append((index, candidate, missing))

    def build_group_requests(
        entry: dict[str, Any],
        indexed_candidates: list[tuple[int, dict[str, Any], tuple[int, ...]]],
    ) -> list[tuple[int, dict[str, Any], tuple[int, ...], dict[str, Any]]]:
        offsets = sorted({int(candidate["row_group_local_row_index"]) for _index, candidate, _missing in indexed_candidates})
        rows = row_provider.fetch_rows(entry, offsets)
        built: list[tuple[int, dict[str, Any], tuple[int, ...], dict[str, Any]]] = []
        for index, candidate, missing in indexed_candidates:
            offset = int(candidate["row_group_local_row_index"])
            built_candidate, built_missing, base_request = build_base_request(candidate, missing, rows.get(offset))
            built.append((index, built_candidate, built_missing, base_request))
        return built

    fetched: list[tuple[dict[str, Any], tuple[int, ...], dict[str, Any]] | None] = [None] * len(missing_plan)
    group_items = list(grouped.values())
    if group_items and refetch_concurrency > 1:
        with ThreadPoolExecutor(max_workers=max(1, int(refetch_concurrency))) as executor:
            future_map = {
                executor.submit(build_group_requests, entry, indexed_candidates): group_index
                for group_index, (entry, indexed_candidates) in enumerate(group_items)
            }
            for future in as_completed(future_map):
                for index, candidate, missing, base_request in future.result():
                    fetched[index] = (candidate, missing, base_request)
    else:
        for entry, indexed_candidates in group_items:
            for index, candidate, missing, base_request in build_group_requests(entry, indexed_candidates):
                fetched[index] = (candidate, missing, base_request)

    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for fetched_item in fetched:
        if fetched_item is None:
            raise ValueError("adaptive BoF private refetch plan produced an incomplete task set")
        candidate, missing, base_request = fetched_item
        for vote_index in missing:
            request_payload = _with_adaptive_bof_vote_identity(base_request, vote_index=vote_index, policy=policy)
            request_payload["refetch_audit_hash"] = base_request["refetch_audit_hash"]
            tasks.append((candidate, request_payload))
    return tasks


def _adaptive_missing_vote_plan(
    candidates: list[dict[str, Any]],
    *,
    existing_rows: list[dict[str, Any]],
    wanted_vote_indexes: tuple[int, ...],
) -> list[tuple[dict[str, Any], tuple[int, ...]]]:
    grouped = _rows_by_candidate(existing_rows)
    plan: list[tuple[dict[str, Any], tuple[int, ...]]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        existing_vote_indexes = {
            int(row.get("vote_index", row.get("bof_call_index", -1)))
            for row in grouped.get(candidate_id, [])
        }
        missing = tuple(index for index in wanted_vote_indexes if index not in existing_vote_indexes)
        if missing:
            plan.append((candidate, missing))
    return plan


def _adaptive_transport_failure_item(
    candidate: dict[str, Any],
    request_payload: dict[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    item = {
        "work_item_schema_version": 1,
        "work_item_type": "transport_failure_retry",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_candidate_manifest_hash": str(request_payload.get("source_candidate_manifest_hash") or ""),
        "provider_config_hash": str(request_payload.get("config_hash") or ""),
        "teacher_model": str((request_payload.get("body") or {}).get("model") or ""),
        "teacher_prompt_version": str(request_payload.get("prompt_version") or ""),
        "prompt_hash": str(request_payload.get("prompt_hash") or ""),
        "request_body_sha256": str(request_payload.get("request_body_sha256") or ""),
        "idempotency_key": str(request_payload.get("idempotency_key") or ""),
        "adaptive_bof_group_key": str(request_payload.get("adaptive_bof_group_key") or ""),
        "vote_index": int(request_payload.get("vote_index", request_payload.get("bof_call_index", -1))),
        "bof_call_index": int(request_payload.get("bof_call_index", -1)),
        "adaptive_bof_stage": str(request_payload.get("adaptive_bof_stage") or ""),
        "source_doc_id": str(request_payload.get("source_doc_id") or ""),
        "source_identity_hash": str(request_payload.get("source_identity_hash") or ""),
        "text_hash": str(request_payload.get("text_hash") or ""),
        "refetch_audit_hash": str(request_payload.get("refetch_audit_hash") or ""),
        "error_class": exc.__class__.__name__,
        "error_message_sha256": _hash(str(exc).split(":")[0]),
        "retry_attempts_exhausted": True,
        "semantic_retry_performed": False,
        "raw_text_serialized": False,
        "raw_model_response_serialized": False,
        "explicit_rerun_required": True,
    }
    if isinstance(exc, DeepSeekHTTPError):
        item["http_status_code"] = exc.status_code
    item["transport_failure_work_item_hash"] = _hash(item)
    _reject_live_artifact_leakage(item)
    return item


def _run_adaptive_vote_tasks(
    tasks: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    live: bool,
    max_docs: int,
    budget_cap_usd: float,
    env: dict[str, str] | None,
    call: Callable[[dict[str, Any]], dict[str, Any]],
    provider_config: DeepSeekLabelingConfig,
    price: PriceSnapshot,
    per_vote_estimate: float,
    concurrency: int,
) -> dict[str, list[dict[str, Any]]]:
    def run_one(candidate: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
        prepare_deepseek_live_call(
            request_payload,
            live=live,
            max_docs=max_docs,
            budget_cap_usd=budget_cap_usd,
            env=env,
        )
        start = time.monotonic()
        response_body = run_with_retries(
            lambda prepared_request: call(
                prepared_request,
                live=live,
                max_docs=max_docs,
                budget_cap_usd=budget_cap_usd,
                env=env,
            )
            if call is call_deepseek_live
            else call(prepared_request),
            request_payload,
            max_attempts=provider_config.max_retries + 1,
        )
        latency_seconds = time.monotonic() - start
        finish_value = _finish_reason(response_body)
        actual_cost_value = actual_label_cost_usd(response_body, price)
        try:
            row = parse_teacher_response(
                response_body,
                candidate=candidate,
                request=request_payload,
                price_snapshot=price,
                estimated_cost_usd=float(per_vote_estimate),
                actual_cost_usd=actual_cost_value,
                latency_seconds=latency_seconds,
            )
        except ValueError as exc:
            row = teacher_label_parse_error_record(
                error_code="provider_response_rejected",
                error_message=str(exc),
                candidate=candidate,
                request=request_payload,
                response_body=response_body,
                finish_reason=finish_value,
                usage=response_body.get("usage", {}),
                estimated_cost_usd=float(per_vote_estimate),
                actual_cost_usd=actual_cost_value,
                latency_seconds=latency_seconds,
            )
        return _attach_adaptive_bof_vote_metadata(row, request_payload)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not tasks:
        return {"rows": rows, "transport_failures": failures}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_map = {
            executor.submit(run_one, candidate, request): (candidate, request)
            for candidate, request in tasks
        }
        for future in as_completed(future_map):
            candidate, request = future_map[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - failure is converted to no-raw retry work.
                failures.append(_adaptive_transport_failure_item(candidate, request, exc))
                continue
            _reject_live_artifact_leakage(row)
            rows.append(row)
    rows.sort(key=lambda row: (str(row["candidate_id"]), int(row["vote_index"])))
    failures.sort(key=lambda item: (str(item["candidate_id"]), int(item["vote_index"])))
    return {"rows": rows, "transport_failures": failures}


def run_teacher_label_adaptive_bof_mining(
    *,
    manifest: dict[str, Any],
    schedule: dict[str, Any],
    row_provider: Any,
    output_jsonl: str | Path,
    report_path: str | Path,
    selected_manifest_path: str | Path,
    repair_queue_path: str | Path,
    progress_path: str | Path,
    live: bool,
    max_docs: int | None,
    current_balance_usd: float | None,
    env: dict[str, str] | None = None,
    call: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    price_snapshot: PriceSnapshot | None = None,
    provider_config: DeepSeekLabelingConfig | None = None,
    policy: AdaptiveBoFMiningPolicy | None = None,
    seed: int = 20260605,
    avg_input_tokens: int = 2_400,
    avg_output_tokens: int = 512,
) -> dict[str, Any]:
    """Run adaptive BoF3 -> BoF5 borderline mining with no-raw durable artifacts."""

    validate_candidate_manifest(manifest)
    if schedule.get("raw_text_serialized") is not False or schedule.get("token_serialized") is not False:
        raise ValueError("source schedule must be a no-raw-text artifact")
    policy = policy or AdaptiveBoFMiningPolicy()
    provider_config = provider_config or DeepSeekLabelingConfig()
    if provider_config != DeepSeekLabelingConfig():
        default_config = DeepSeekLabelingConfig()
        if provider_config.model != default_config.model or provider_config.temperature != 0.2:
            raise ValueError("adaptive BoF mining must use canonical non-thinking DeepSeek V4 Pro settings")
        if provider_config.max_output_tokens != PROMPT_V3_DEEPSEEK_TEACHER_MAX_OUTPUT_TOKENS:
            raise ValueError("adaptive BoF mining must use prompt-v3 max_output_tokens=512")
        if provider_config.thinking != "disabled" or provider_config.reasoning_effort:
            raise ValueError("adaptive BoF mining must use thinking disabled")
        if provider_config.stream is not False or provider_config.response_format != {"type": "json_object"}:
            raise ValueError("adaptive BoF mining must use JSON Output and stream=false")
    if max_docs is None or max_docs < 1:
        raise ValueError("adaptive BoF mining requires explicit --max-docs")
    if max_docs > policy.max_attempted_documents:
        raise ValueError("adaptive BoF max_docs cannot exceed policy max_attempted_documents")
    if current_balance_usd is None:
        raise ValueError("adaptive BoF mining requires current balance for reserve gating")
    if float(current_balance_usd) < policy.minimum_balance_reserve_usd:
        raise ValueError("adaptive BoF current balance is below required reserve")

    requested_max_docs = int(max_docs)
    all_candidates = _adaptive_candidate_order(list(manifest.get("candidates") or []), seed=seed)
    candidates = all_candidates[:requested_max_docs]
    if not candidates:
        raise ValueError("candidate manifest does not contain any candidates for adaptive mining")
    candidate_manifest_shortfall = len(candidates) < requested_max_docs
    attempt_readiness = manifest.get("attempt_readiness")
    attempt_manifest_unready = (
        isinstance(attempt_readiness, dict)
        and attempt_readiness.get("ready_for_adaptive_mining") is not True
    )
    price = price_snapshot or _price_from_official_snapshot(provider_config.model)
    provider_config_hash = _hash(_config_payload(provider_config))
    provider_payload = _public_config_payload(provider_config)
    budget_check = estimate_label_cost_usd(
        label_count=len(candidates) * 5,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=price,
        safety_multiplier=1.25,
        budget_cap_usd=policy.budget_cap_usd,
        reserve_usd=0.0,
    )
    per_vote_estimate = estimate_label_cost_usd(
        label_count=1,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=price,
        safety_multiplier=1.0,
        budget_cap_usd=policy.budget_cap_usd,
        reserve_usd=0.0,
    )["estimated_cost_usd"]
    call = call or call_deepseek_live
    using_live_client = call is call_deepseek_live
    rows_for_report = _read_existing_adaptive_bof_rows(
        output_jsonl,
        manifest=manifest,
        provider_config=provider_config,
        candidates=candidates,
        policy=policy,
    )
    existing_keys = {str(row["idempotency_key"]) for row in rows_for_report}
    observed_cost = _row_cost_sum(rows_for_report, "actual_cost_usd")
    estimated_cost = _row_cost_sum(rows_for_report, "estimated_cost_usd")
    paid_calls_made = 0
    run_status = "complete"
    blocked_reason: str | None = None
    transport_failures: list[dict[str, Any]] = []

    if live is not True:
        run_status = "dry_run_no_paid_calls"
        blocked_reason = "explicit live flag not set"
    elif candidate_manifest_shortfall or attempt_manifest_unready:
        run_status = "blocked_expansion_required"
        blocked_reason = (
            "adaptive BoF attempt manifest is not ready for requested max_docs; "
            "build a larger no-raw source schedule and attempt manifest before live mining"
        )

    def budget_allows(next_call_count: int) -> bool:
        next_call_estimate = float(per_vote_estimate) * next_call_count
        planned_total_run_cost = observed_cost + next_call_estimate
        if planned_total_run_cost > float(policy.budget_cap_usd):
            return False
        return float(current_balance_usd) - next_call_estimate >= float(policy.minimum_balance_reserve_usd)

    def append_new_rows(new_rows: list[dict[str, Any]]) -> None:
        nonlocal observed_cost, estimated_cost, paid_calls_made, rows_for_report, run_status, blocked_reason
        for row in new_rows:
            key = str(row["idempotency_key"])
            if key in existing_keys:
                continue
            _append_label_row_known_unique(output_jsonl, row)
            existing_keys.add(key)
            rows_for_report.append(row)
            paid_calls_made += 1
            observed_cost += float(row.get("actual_cost_usd") or 0.0)
            estimated_cost += float(row.get("estimated_cost_usd") or 0.0)
        if observed_cost > float(policy.budget_cap_usd) or float(current_balance_usd) - observed_cost < float(policy.minimum_balance_reserve_usd):
            run_status = "blocked_budget_cap"
            blocked_reason = "observed adaptive BoF cost exceeded budget cap or reserve"

    if run_status == "complete":
        tranche_size = max(1, min(policy.tranche_documents, max_docs))
        for tranche_index, start_index in enumerate(range(0, len(candidates), tranche_size)):
            tranche = candidates[start_index : start_index + tranche_size]
            concurrency = policy.concurrency_ramp[min(tranche_index, len(policy.concurrency_ramp) - 1)]
            bof3_missing = _adaptive_missing_vote_plan(
                tranche,
                existing_rows=rows_for_report,
                wanted_vote_indexes=(0, 1, 2),
            )
            bof3_missing_count = sum(len(missing) for _candidate, missing in bof3_missing)
            if bof3_missing_count and not budget_allows(bof3_missing_count):
                run_status = "blocked_budget_cap"
                blocked_reason = "next planned adaptive BoF call would exceed budget cap or reserve"
                break
            bof3_tasks = _adaptive_vote_request_tasks(
                manifest=manifest,
                schedule=schedule,
                row_provider=row_provider,
                missing_plan=bof3_missing,
                provider_config=provider_config,
                policy=policy,
                refetch_concurrency=policy.private_refetch_concurrency,
            )
            if bof3_tasks:
                result = _run_adaptive_vote_tasks(
                    bof3_tasks,
                    live=live,
                    max_docs=max_docs,
                    budget_cap_usd=policy.budget_cap_usd,
                    env=env,
                    call=call,
                    provider_config=provider_config,
                    price=price,
                    per_vote_estimate=float(per_vote_estimate),
                    concurrency=concurrency,
                )
                append_new_rows(result["rows"])
                transport_failures.extend(result["transport_failures"])
                if result["transport_failures"]:
                    run_status = "blocked_transport_failures"
                    blocked_reason = "adaptive BoF transport failures queued after retry exhaustion"
                    break
                if run_status != "complete":
                    break
            grouped = _rows_by_candidate(rows_for_report)
            escalation_candidates = [
                candidate
                for candidate in tranche
                if _adaptive_bof3_requires_bof5(grouped.get(str(candidate["candidate_id"]), []))
            ]
            bof5_missing = _adaptive_missing_vote_plan(
                escalation_candidates,
                existing_rows=rows_for_report,
                wanted_vote_indexes=(3, 4),
            )
            bof5_missing_count = sum(len(missing) for _candidate, missing in bof5_missing)
            if bof5_missing_count and not budget_allows(bof5_missing_count):
                run_status = "blocked_budget_cap"
                blocked_reason = "next planned adaptive BoF call would exceed budget cap or reserve"
                break
            bof5_tasks = _adaptive_vote_request_tasks(
                manifest=manifest,
                schedule=schedule,
                row_provider=row_provider,
                missing_plan=bof5_missing,
                provider_config=provider_config,
                policy=policy,
                refetch_concurrency=policy.private_refetch_concurrency,
            )
            if bof5_tasks:
                result = _run_adaptive_vote_tasks(
                    bof5_tasks,
                    live=live,
                    max_docs=max_docs,
                    budget_cap_usd=policy.budget_cap_usd,
                    env=env,
                    call=call,
                    provider_config=provider_config,
                    price=price,
                    per_vote_estimate=float(per_vote_estimate),
                    concurrency=concurrency,
                )
                append_new_rows(result["rows"])
                transport_failures.extend(result["transport_failures"])
                if result["transport_failures"]:
                    run_status = "blocked_transport_failures"
                    blocked_reason = "adaptive BoF transport failures queued after retry exhaustion"
                    break
                if run_status != "complete":
                    break
            selected_now = _adaptive_selected_records(rows_for_report, candidates, selected_target=policy.selected_target)
            if len(selected_now) >= policy.selected_target:
                break

    selected = _adaptive_selected_records(rows_for_report, candidates, selected_target=policy.selected_target)
    attempted_ids = {
        str(row.get("candidate_id") or "")
        for row in rows_for_report
        if row.get("candidate_id")
    }
    final_status, final_blocker = _adaptive_run_status(
        selected_count=len(selected),
        selected_target=policy.selected_target,
        attempted_count=len(attempted_ids),
        max_attempted=len(candidates),
        current_status=run_status,
        current_blocker=blocked_reason,
    )
    if final_status == "blocked_low_yield" and candidate_manifest_shortfall:
        final_blocker = "candidate manifest exhausted before requested max_docs; bounded expansion required"
    parse_error_rows = [row for row in rows_for_report if row.get("parse_status") == "parse_error"]
    repair_queue = _write_adaptive_repair_queue(
        repair_queue_path,
        parse_error_rows=parse_error_rows,
        transport_failures=transport_failures,
    )
    selected_manifest = _write_adaptive_selected_manifest(
        selected_manifest_path,
        manifest=manifest,
        policy=policy,
        selected=selected,
        rows=rows_for_report,
    )
    progress = _write_adaptive_progress(
        progress_path,
        manifest=manifest,
        policy=policy,
        provider_config_hash=provider_config_hash,
        attempted_count=len(attempted_ids),
        rows=rows_for_report,
        selected_count=len(selected),
        observed_cost=observed_cost,
        estimated_cost=estimated_cost,
        run_status=final_status,
        blocked_reason=final_blocker,
    )
    candidate_provenance = [
        build_adaptive_bof_candidate_provenance(candidate_id, candidate_rows)
        for candidate_id, candidate_rows in sorted(_rows_by_candidate(rows_for_report).items())
    ]
    report = {
        "adaptive_bof_mining_report_schema_version": ADAPTIVE_BOF_MINING_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "run_status": final_status,
        "blocked_reason": final_blocker,
        "source_candidate_manifest_hash": manifest["manifest_hash"],
        "adaptive_bof_policy_version": policy.policy_version,
        "requested_max_docs": requested_max_docs,
        "max_docs": len(candidates),
        "max_attempted_documents": policy.max_attempted_documents,
        "source_manifest_candidate_count": len(all_candidates),
        "candidate_manifest_shortfall": candidate_manifest_shortfall,
        "attempted_document_count": len(attempted_ids),
        "selected_target": policy.selected_target,
        "selected_count": len(selected),
        "complete_bof5_required_for_selection": True,
        "votes_recorded": len(rows_for_report),
        "paid_api_calls_made": paid_calls_made,
        "paid_api_calls_performed": bool(using_live_client and paid_calls_made > 0),
        "durable_successful_vote_rows_written": len(rows_for_report),
        "transport_failure_count": len(transport_failures),
        "per_vote_parse_status_counts": _status_counts(rows_for_report, "parse_status"),
        "finish_reason_counts": _status_counts(rows_for_report, "finish_reason"),
        "semantic_retry_counts": {},
        "provider_config": provider_payload,
        "provider_config_hash": provider_config_hash,
        "price_snapshot": asdict(price),
        "budget_cap_usd": policy.budget_cap_usd,
        "minimum_balance_reserve_usd": policy.minimum_balance_reserve_usd,
        "current_balance_usd": float(current_balance_usd),
        "budget_check": _redact_price_snapshot_urls(budget_check),
        "total_estimated_cost_usd": estimated_cost,
        "total_actual_cost_usd": observed_cost,
        "candidate_processing_order": "deterministic_randomized_source_row_group_batches_v1",
        "target_concurrency": policy.target_concurrency,
        "private_refetch_concurrency": policy.private_refetch_concurrency,
        "concurrency_ramp": list(policy.concurrency_ramp),
        "latency_summary": _latency_summary_from_rows(rows_for_report),
        "token_usage_summary": _token_usage_summary(rows_for_report),
        "candidate_provenance": candidate_provenance,
        "diversity_cap_policy": selected_manifest["diversity_cap_policy"],
        "diversity_caps_enforced": selected_manifest["diversity_caps_enforced"],
        "diversity_diagnostics": selected_manifest["diversity_diagnostics"],
        "selected_manifest_hash": selected_manifest["adaptive_bof_selected_manifest_hash"],
        "repair_queue_hash": repair_queue["repair_queue_hash"],
        "progress_hash": progress["adaptive_bof_progress_hash"],
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "commits_or_pushes_performed": False,
    }
    report["adaptive_bof_mining_report_hash"] = _hash({key: value for key, value in report.items() if key != "generated_at_utc"})
    _reject_live_artifact_leakage(report)
    _write_json_atomic(report_path, report)
    return report


def _final_bof_group_key(base_request: dict[str, Any], *, policy: FinalLabelBatchPolicy) -> str:
    return _hash(
        [
            "final_label_vote_group",
            policy.policy_version,
            base_request["source_candidate_manifest_hash"],
            base_request["candidate_id"],
            base_request["text_hash"],
            base_request["prompt_hash"],
            base_request["config_hash"],
            base_request["request_body_sha256"],
        ]
    )


def _with_final_label_vote_identity(
    request_payload: dict[str, Any],
    *,
    vote_index: int,
    policy: FinalLabelBatchPolicy,
) -> dict[str, Any]:
    if vote_index < 0 or vote_index >= 3:
        raise ValueError("final label vote_index must be between 0 and 2")
    updated = dict(request_payload)
    group_key = _final_bof_group_key(request_payload, policy=policy)
    updated["bof_n"] = 3
    updated["bof_call_index"] = int(vote_index)
    updated["vote_index"] = int(vote_index)
    updated["final_label_group_key"] = group_key
    updated["bof_group_key"] = group_key
    updated["final_label_policy_version"] = policy.policy_version
    updated["final_label_stage"] = "bof3"
    updated["idempotency_key"] = _hash(["final_label_vote", group_key, vote_index])
    return updated


def _attach_final_label_vote_metadata(row: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["bof_n"] = 3
    updated["bof_call_index"] = int(request_payload["bof_call_index"])
    updated["vote_index"] = int(request_payload["vote_index"])
    updated["final_label_group_key"] = str(request_payload["final_label_group_key"])
    updated["bof_group_key"] = str(request_payload["final_label_group_key"])
    updated["final_label_policy_version"] = str(request_payload["final_label_policy_version"])
    updated["final_label_stage"] = str(request_payload["final_label_stage"])
    updated["refetch_audit_hash"] = str(request_payload.get("refetch_audit_hash") or "")
    updated["bof_vote_role"] = "independent_same_prompt_vote"
    updated["semantic_retry_performed"] = False
    if "label_row_hash" in updated:
        updated.pop("label_row_hash", None)
        updated["label_row_hash"] = _hash(updated)
    if "label_parse_error_hash" in updated:
        updated.pop("label_parse_error_hash", None)
        updated["label_parse_error_hash"] = _hash(updated)
    _reject_live_artifact_leakage(updated)
    return updated


def _read_existing_final_vote_rows(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    provider_config: DeepSeekLabelingConfig,
    candidates: list[dict[str, Any]],
    policy: FinalLabelBatchPolicy,
) -> list[dict[str, Any]]:
    output = Path(path)
    if not output.exists():
        return []
    rows = _read_jsonl_objects(output)
    expected_manifest_hash = str(manifest["manifest_hash"])
    expected_config_hash = _hash(_config_payload(provider_config))
    candidate_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    seen_idempotency_keys: set[str] = set()
    seen_vote_keys: set[tuple[str, int]] = set()
    for row in rows:
        _reject_live_artifact_leakage(row)
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id not in candidate_ids:
            raise ValueError("existing final label vote row contains a candidate outside the current run")
        if str(row.get("source_candidate_manifest_hash") or "") != expected_manifest_hash:
            raise ValueError("existing final label vote row belongs to a different candidate manifest")
        if str(row.get("provider_config_hash") or "") != expected_config_hash:
            raise ValueError("existing final label vote row belongs to a different provider config")
        if str(row.get("teacher_prompt_version") or "") != provider_config.prompt_version:
            raise ValueError("existing final label vote row belongs to a different teacher prompt")
        if str(row.get("teacher_model") or "") != provider_config.model:
            raise ValueError("existing final label vote row belongs to a different teacher model")
        if str(row.get("final_label_policy_version") or "") != policy.policy_version:
            raise ValueError("existing final label vote row belongs to a different final label policy")
        vote_index = int(row.get("vote_index") if row.get("vote_index") is not None else row.get("bof_call_index", -1))
        if vote_index < 0 or vote_index >= 3:
            raise ValueError("existing final label vote row has invalid vote_index")
        idempotency_key = str(row.get("idempotency_key") or "")
        if not idempotency_key:
            raise ValueError("existing final label vote row is missing idempotency_key")
        if idempotency_key in seen_idempotency_keys:
            raise ValueError("existing final label vote rows contain duplicate idempotency_key")
        seen_idempotency_keys.add(idempotency_key)
        vote_key = (candidate_id, vote_index)
        if vote_key in seen_vote_keys:
            raise ValueError("existing final label vote rows contain duplicate candidate vote")
        seen_vote_keys.add(vote_key)
        if row.get("parse_status") not in {"parsed", "parse_error"}:
            raise ValueError("existing final label vote row has unsupported parse_status")
    return rows


def _final_vote_request_tasks(
    *,
    manifest: dict[str, Any],
    schedule: dict[str, Any],
    row_provider: Any,
    missing_plan: list[tuple[dict[str, Any], tuple[int, ...]]],
    provider_config: DeepSeekLabelingConfig,
    policy: FinalLabelBatchPolicy,
    refetch_concurrency: int = 1,
) -> dict[str, Any]:
    def build_base_request(
        candidate: dict[str, Any],
        missing: tuple[int, ...],
        row: Any,
    ) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
        private_text, refetch_audit = _private_text_and_audit_from_refetched_row(candidate, row)
        base_request = build_deepseek_request(
            candidate,
            raw_text=private_text,
            config=provider_config,
            manifest_hash=manifest["manifest_hash"],
        )
        base_request["refetch_audit_hash"] = _hash(refetch_audit)
        return candidate, missing, base_request

    grouped: dict[str, tuple[dict[str, Any], list[tuple[int, dict[str, Any], tuple[int, ...]]]]] = {}
    for index, (candidate, missing) in enumerate(missing_plan):
        entry = _schedule_entry_for_candidate(candidate, schedule)
        group_key = str(entry.get("entry_hash") or _hash(entry))
        if group_key not in grouped:
            grouped[group_key] = (entry, [])
        grouped[group_key][1].append((index, candidate, missing))

    def build_group_requests(
        entry: dict[str, Any],
        indexed_candidates: list[tuple[int, dict[str, Any], tuple[int, ...]]],
    ) -> list[tuple[int, dict[str, Any], tuple[int, ...], dict[str, Any]]]:
        offsets = sorted({int(candidate["row_group_local_row_index"]) for _index, candidate, _missing in indexed_candidates})
        rows = row_provider.fetch_rows(entry, offsets)
        built: list[tuple[int, dict[str, Any], tuple[int, ...], dict[str, Any]]] = []
        for index, candidate, missing in indexed_candidates:
            offset = int(candidate["row_group_local_row_index"])
            built_candidate, built_missing, base_request = build_base_request(candidate, missing, rows.get(offset))
            built.append((index, built_candidate, built_missing, base_request))
        return built

    fetched: list[tuple[dict[str, Any], tuple[int, ...], dict[str, Any]] | None] = [None] * len(missing_plan)
    failed_indexes: set[int] = set()
    refetch_failures: list[dict[str, Any]] = []

    def record_refetch_failure(
        indexed_candidates: list[tuple[int, dict[str, Any], tuple[int, ...]]],
        exc: BaseException,
    ) -> None:
        for index, candidate, missing in indexed_candidates:
            failed_indexes.add(index)
            refetch_failures.append(
                _final_refetch_failure_item(
                    candidate,
                    missing=missing,
                    exc=exc,
                    manifest=manifest,
                    provider_config=provider_config,
                    provider_config_hash=_hash(_config_payload(provider_config)),
                    policy=policy,
                )
            )

    group_items = list(grouped.values())
    if group_items and refetch_concurrency > 1:
        with ThreadPoolExecutor(max_workers=max(1, int(refetch_concurrency))) as executor:
            future_map = {
                executor.submit(build_group_requests, entry, indexed_candidates): group_index
                for group_index, (entry, indexed_candidates) in enumerate(group_items)
            }
            for future in as_completed(future_map):
                group_index = future_map[future]
                _entry, indexed_candidates = group_items[group_index]
                try:
                    built_requests = future.result()
                except Exception as exc:
                    record_refetch_failure(indexed_candidates, exc)
                    continue
                for index, candidate, missing, base_request in built_requests:
                    fetched[index] = (candidate, missing, base_request)
    else:
        for entry, indexed_candidates in group_items:
            try:
                built_requests = build_group_requests(entry, indexed_candidates)
            except Exception as exc:
                record_refetch_failure(indexed_candidates, exc)
                continue
            for index, candidate, missing, base_request in built_requests:
                fetched[index] = (candidate, missing, base_request)

    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, fetched_item in enumerate(fetched):
        if fetched_item is None:
            if index in failed_indexes:
                continue
            raise ValueError("final label private refetch plan produced an incomplete task set")
        candidate, missing, base_request = fetched_item
        for vote_index in missing:
            request_payload = _with_final_label_vote_identity(base_request, vote_index=vote_index, policy=policy)
            request_payload["refetch_audit_hash"] = base_request["refetch_audit_hash"]
            tasks.append((candidate, request_payload))
    return {"tasks": tasks, "transport_failures": refetch_failures}


def _final_refetch_failure_item(
    candidate: dict[str, Any],
    *,
    missing: tuple[int, ...],
    exc: BaseException,
    manifest: dict[str, Any],
    provider_config: DeepSeekLabelingConfig,
    provider_config_hash: str,
    policy: FinalLabelBatchPolicy,
) -> dict[str, Any]:
    item = {
        "work_item_schema_version": 1,
        "work_item_type": "final_label_private_refetch_failure_retry",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_candidate_manifest_hash": str(manifest.get("manifest_hash") or ""),
        "candidate_hash": str(candidate.get("candidate_hash") or ""),
        "provider_config_hash": provider_config_hash,
        "teacher_model": provider_config.model,
        "teacher_prompt_version": TEACHER_PROMPT_VERSION,
        "final_label_policy_version": policy.policy_version,
        "vote_indexes": [int(index) for index in missing],
        "source_doc_id": str(candidate.get("source_doc_id") or ""),
        "source_identity_hash": str(candidate.get("source_identity_hash") or candidate.get("physical_source_row_key_hash") or ""),
        "text_hash": str(candidate.get("text_hash") or ""),
        "schedule_entry_hash": str(candidate.get("schedule_entry_hash") or ""),
        "source_schedule_hash": str(candidate.get("source_schedule_hash") or ""),
        "source_file_id": str(candidate.get("source_file_id") or ""),
        "row_group_index": int(candidate.get("row_group_index", -1)),
        "row_group_local_row_index": int(candidate.get("row_group_local_row_index", -1)),
        "error_class": exc.__class__.__name__,
        "error_message_sha256": _hash(exc.__class__.__name__),
        "retry_attempts_exhausted": True,
        "semantic_retry_performed": False,
        "private_refetch_failed_before_provider_call": True,
        "raw_text_serialized": False,
        "raw_model_response_serialized": False,
        "explicit_rerun_required": True,
    }
    item["transport_failure_work_item_hash"] = _hash(item)
    _reject_live_artifact_leakage(item)
    return item


def _final_transport_failure_item(
    candidate: dict[str, Any],
    request_payload: dict[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    item = {
        "work_item_schema_version": 1,
        "work_item_type": "final_label_transport_failure_retry",
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "source_candidate_manifest_hash": str(request_payload.get("source_candidate_manifest_hash") or ""),
        "provider_config_hash": str(request_payload.get("config_hash") or ""),
        "teacher_model": str((request_payload.get("body") or {}).get("model") or ""),
        "teacher_prompt_version": str(request_payload.get("prompt_version") or ""),
        "prompt_hash": str(request_payload.get("prompt_hash") or ""),
        "request_body_sha256": str(request_payload.get("request_body_sha256") or ""),
        "idempotency_key": str(request_payload.get("idempotency_key") or ""),
        "final_label_group_key": str(request_payload.get("final_label_group_key") or ""),
        "vote_index": int(request_payload.get("vote_index", request_payload.get("bof_call_index", -1))),
        "bof_call_index": int(request_payload.get("bof_call_index", -1)),
        "final_label_stage": str(request_payload.get("final_label_stage") or ""),
        "source_doc_id": str(request_payload.get("source_doc_id") or ""),
        "source_identity_hash": str(request_payload.get("source_identity_hash") or ""),
        "text_hash": str(request_payload.get("text_hash") or ""),
        "refetch_audit_hash": str(request_payload.get("refetch_audit_hash") or ""),
        "error_class": exc.__class__.__name__,
        "error_message_sha256": _hash(str(exc).split(":")[0]),
        "retry_attempts_exhausted": True,
        "semantic_retry_performed": False,
        "raw_text_serialized": False,
        "raw_model_response_serialized": False,
        "explicit_rerun_required": True,
    }
    if isinstance(exc, DeepSeekHTTPError):
        item["http_status_code"] = exc.status_code
        item["account_exhausted_or_insufficient_balance"] = exc.status_code == 402
    item["transport_failure_work_item_hash"] = _hash(item)
    _reject_live_artifact_leakage(item)
    return item


def _run_final_vote_tasks(
    tasks: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    live: bool,
    max_docs: int,
    budget_cap_usd: float,
    env: dict[str, str] | None,
    call: Callable[[dict[str, Any]], dict[str, Any]],
    provider_config: DeepSeekLabelingConfig,
    price: PriceSnapshot,
    per_vote_estimate: float,
    concurrency: int,
) -> dict[str, list[dict[str, Any]]]:
    def run_one(candidate: dict[str, Any], request_payload: dict[str, Any]) -> dict[str, Any]:
        prepare_deepseek_live_call(
            request_payload,
            live=live,
            max_docs=max_docs,
            budget_cap_usd=budget_cap_usd,
            env=env,
        )
        start = time.monotonic()
        response_body = run_with_retries(
            lambda prepared_request: call(
                prepared_request,
                live=live,
                max_docs=max_docs,
                budget_cap_usd=budget_cap_usd,
                env=env,
            )
            if call is call_deepseek_live
            else call(prepared_request),
            request_payload,
            max_attempts=provider_config.max_retries + 1,
        )
        latency_seconds = time.monotonic() - start
        finish_value = _finish_reason(response_body)
        actual_cost_value = actual_label_cost_usd(response_body, price)
        try:
            row = parse_teacher_response(
                response_body,
                candidate=candidate,
                request=request_payload,
                price_snapshot=price,
                estimated_cost_usd=float(per_vote_estimate),
                actual_cost_usd=actual_cost_value,
                latency_seconds=latency_seconds,
            )
        except ValueError as exc:
            row = teacher_label_parse_error_record(
                error_code="provider_response_rejected",
                error_message=str(exc),
                candidate=candidate,
                request=request_payload,
                response_body=response_body,
                finish_reason=finish_value,
                usage=response_body.get("usage", {}),
                estimated_cost_usd=float(per_vote_estimate),
                actual_cost_usd=actual_cost_value,
                latency_seconds=latency_seconds,
            )
        return _attach_final_label_vote_metadata(row, request_payload)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not tasks:
        return {"rows": rows, "transport_failures": failures}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_map = {executor.submit(run_one, candidate, request): (candidate, request) for candidate, request in tasks}
        for future in as_completed(future_map):
            candidate, request = future_map[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 - failure is converted to no-raw retry work.
                failures.append(_final_transport_failure_item(candidate, request, exc))
                continue
            _reject_live_artifact_leakage(row)
            rows.append(row)
    rows.sort(key=lambda row: (str(row["candidate_id"]), int(row["vote_index"])))
    failures.sort(key=lambda item: (str(item["candidate_id"]), int(item["vote_index"])))
    return {"rows": rows, "transport_failures": failures}


def _final_required_vote_count(_rows: list[dict[str, Any]]) -> int:
    return 3


def _build_final_label_aggregation(
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    transport_failures: list[dict[str, Any]],
    manifest: dict[str, Any],
    policy: FinalLabelBatchPolicy,
    provider_config_hash: str,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    parsed = [row for row in rows if row.get("parse_status") == "parsed"]
    parse_errors = [row for row in rows if row.get("parse_status") == "parse_error"]
    required_vote_count = _final_required_vote_count(rows)
    required_indexes = set(range(required_vote_count))
    parsed_required = [
        row
        for row in parsed
        if int(row.get("vote_index", row.get("bof_call_index", -1))) in required_indexes
    ]
    parse_error_required = [
        row
        for row in parse_errors
        if int(row.get("vote_index", row.get("bof_call_index", -1))) in required_indexes
    ]
    vote_indexes = sorted({int(row.get("vote_index", row.get("bof_call_index", -1))) for row in rows})
    attempted_required_indexes = {
        int(row.get("vote_index", row.get("bof_call_index", -1)))
        for row in rows
        if int(row.get("vote_index", row.get("bof_call_index", -1))) in required_indexes
    }
    label_counts: dict[str, int] = {}
    borderline_true = 0
    borderline_false = 0
    for row in parsed_required:
        label = str(row.get("label") or "missing")
        label_counts[label] = label_counts.get(label, 0) + 1
        if row.get("borderline") is True:
            borderline_true += 1
        elif row.get("borderline") is False:
            borderline_false += 1
    complete = (
        not transport_failures
        and not parse_error_required
        and len(parsed_required) == required_vote_count
        and sorted(int(row.get("vote_index", row.get("bof_call_index", -1))) for row in parsed_required) == sorted(required_indexes)
    )
    if complete:
        if label_counts == {"safe": required_vote_count} and borderline_true == 0:
            final_label = "safe"
            route = "unanimous_safe"
            ready = True
        elif label_counts == {"reject": required_vote_count} and borderline_true == 0:
            final_label = "reject"
            route = "unanimous_reject"
            ready = True
        elif borderline_true > 0:
            final_label = "reject"
            route = "borderline_fail_closed_reject"
            ready = True
        else:
            final_label = "reject"
            route = "mixed_fail_closed_reject"
            ready = True
    elif not transport_failures and parse_error_required and attempted_required_indexes == required_indexes:
        final_label = "reject"
        route = "parse_error_fail_closed_reject"
        ready = True
    else:
        final_label = None
        route = "parse_or_transport_repair_required"
        ready = False
    aggregation = {
        "final_label_aggregation_schema_version": 1,
        "candidate_id": candidate_id,
        "source_candidate_manifest_hash": manifest["manifest_hash"],
        "candidate_hash": candidate.get("candidate_hash"),
        "provider_config_hash": provider_config_hash,
        "final_label_policy_version": policy.policy_version,
        "required_vote_count": required_vote_count,
        "vote_count": len(rows),
        "parsed_vote_count": len(parsed),
        "parse_error_vote_count": len(parse_errors),
        "transport_failure_count": len(transport_failures),
        "vote_indexes": vote_indexes,
        "label_counts": dict(sorted(label_counts.items())),
        "borderline_true_count": borderline_true,
        "borderline_false_count": borderline_false,
        "aggregation_route": route,
        "final_label": final_label,
        "final_label_row_ready": ready,
        "labels_are_corpus_approval": False,
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
    }
    aggregation["final_label_aggregation_hash"] = _hash(aggregation)
    _reject_live_artifact_leakage(aggregation)
    return aggregation


def _final_label_row_from_aggregation(aggregation: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if aggregation.get("final_label_row_ready") is not True or aggregation.get("final_label") not in {"safe", "reject"}:
        raise ValueError("final label row requires ready safe/reject aggregation")
    row = {
        "final_label_row_schema_version": 1,
        "parse_status": "final_label",
        "candidate_id": aggregation["candidate_id"],
        "source_candidate_manifest_hash": aggregation["source_candidate_manifest_hash"],
        "candidate_hash": aggregation.get("candidate_hash"),
        "final_label_policy_version": aggregation["final_label_policy_version"],
        "final_label_aggregation_hash": aggregation["final_label_aggregation_hash"],
        "label": aggregation["final_label"],
        "final_label": aggregation["final_label"],
        "aggregation_route": aggregation["aggregation_route"],
        "required_vote_count": aggregation["required_vote_count"],
        "parsed_vote_count": aggregation["parsed_vote_count"],
        "parse_error_vote_count": aggregation["parse_error_vote_count"],
        "transport_failure_count": aggregation["transport_failure_count"],
        "label_unit": str(candidate.get("label_unit") or "document"),
        "source_schedule_hash": candidate.get("source_schedule_hash"),
        "schedule_entry_hash": candidate.get("schedule_entry_hash"),
        "source_file_id": candidate.get("source_file_id"),
        "row_group_index": candidate.get("row_group_index"),
        "row_group_local_row_index": candidate.get("row_group_local_row_index"),
        "labels_are_corpus_approval": False,
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    row["final_label_row_hash"] = _hash(row)
    _reject_live_artifact_leakage(row)
    return row


def _write_final_label_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        _reject_live_artifact_leakage(row)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    tmp.replace(output)


def _write_final_repair_queue(
    path: str | Path,
    *,
    parse_error_rows: list[dict[str, Any]],
    transport_failures: list[dict[str, Any]],
    aggregations: list[dict[str, Any]],
) -> dict[str, Any]:
    by_finish_reason: dict[str, int] = {}
    by_error_code: dict[str, int] = {}
    by_schema_rule: dict[str, int] = {}
    for row in parse_error_rows:
        finish = str(row.get("finish_reason") or "missing")
        by_finish_reason[finish] = by_finish_reason.get(finish, 0) + 1
        error_code = str(row.get("error_code") or "missing")
        by_error_code[error_code] = by_error_code.get(error_code, 0) + 1
        schema_rule = str(row.get("schema_failure_rule") or "missing")
        by_schema_rule[schema_rule] = by_schema_rule.get(schema_rule, 0) + 1
    parse_candidate_ids = {str(row.get("candidate_id") or "") for row in parse_error_rows if row.get("candidate_id")}
    transport_candidate_ids = {str(item.get("candidate_id") or "") for item in transport_failures if item.get("candidate_id")}
    aggregation_candidate_ids = {
        str(row.get("candidate_id") or "")
        for row in aggregations
        if row.get("aggregation_route") == "parse_or_transport_repair_required"
    }
    candidate_ids = sorted(parse_candidate_ids | transport_candidate_ids | aggregation_candidate_ids)
    work_items = []
    for candidate_id in candidate_ids:
        candidate_parse_rows = [row for row in parse_error_rows if str(row.get("candidate_id") or "") == candidate_id]
        candidate_transport = [item for item in transport_failures if str(item.get("candidate_id") or "") == candidate_id]
        aggregation = next((row for row in aggregations if str(row.get("candidate_id") or "") == candidate_id), {})
        item = {
            "repair_work_item_schema_version": 1,
            "repair_mode": "explicit_rerun_or_private_adjudication",
            "candidate_id": candidate_id,
            "aggregation_route": aggregation.get("aggregation_route", "parse_or_transport_repair_required"),
            "required_vote_count": int(aggregation.get("required_vote_count") or 0),
            "parsed_vote_count": int(aggregation.get("parsed_vote_count") or 0),
            "parse_error_vote_count": len(candidate_parse_rows),
            "transport_failure_count": len(candidate_transport),
            "failed_idempotency_keys": sorted(str(row.get("idempotency_key") or "") for row in candidate_parse_rows if row.get("idempotency_key")),
            "transport_failure_hashes": sorted(str(row.get("transport_failure_work_item_hash") or "") for row in candidate_transport),
            "private_refetch_or_relabel_required": True,
            "semantic_retry_in_main_batch": False,
            "raw_text_serialized": False,
            "raw_model_response_serialized": False,
        }
        item["repair_work_item_hash"] = _hash(item)
        _reject_live_artifact_leakage(item)
        work_items.append(item)
    queue = {
        "repair_queue_schema_version": PARSE_ERROR_REPAIR_QUEUE_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "repair_queue_mode": "final_label_repair_or_rerun",
        "queued_count": len(work_items),
        "parse_error_vote_count": len(parse_error_rows),
        "transport_failure_count": len(transport_failures),
        "by_error_code": dict(sorted(by_error_code.items())),
        "by_finish_reason": dict(sorted(by_finish_reason.items())),
        "by_schema_failure_rule": dict(sorted(by_schema_rule.items())),
        "work_items": work_items,
        "transport_failure_work_items": transport_failures,
        "format_only_repair_from_public_artifact": {
            "allowed": False,
            "reason": "raw_malformed_model_output_is_not_available_in_public_artifacts",
            "private_raw_response_required": True,
        },
        "raw_text_serialized": False,
        "raw_model_response_serialized": False,
        "paid_api_calls_performed": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    queue["repair_queue_hash"] = _hash({key: value for key, value in queue.items() if key != "generated_at_utc"})
    _reject_live_artifact_leakage(queue)
    _write_json_atomic(path, queue)
    return queue


def _write_final_progress(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    policy: FinalLabelBatchPolicy,
    provider_config_hash: str,
    rows: list[dict[str, Any]],
    aggregations: list[dict[str, Any]],
    observed_cost: float,
    estimated_cost: float,
    run_status: str,
    blocked_reason: str | None,
    max_docs: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    unfinished = [row for row in aggregations if row.get("final_label_row_ready") is not True]
    final_safe_count = sum(1 for row in aggregations if row.get("final_label") == "safe")
    final_reject_count = sum(1 for row in aggregations if row.get("final_label") == "reject")
    route_counts: dict[str, int] = {}
    for row in aggregations:
        route = str(row.get("aggregation_route") or "missing")
        route_counts[route] = route_counts.get(route, 0) + 1
    progress = {
        "final_label_progress_schema_version": FINAL_LABEL_PROGRESS_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "phase": "final_label_dry_run" if run_status == "dry_run_no_paid_calls" else "final_label_bof3_freeze_batch",
        "source_candidate_manifest_hash": manifest["manifest_hash"],
        "final_label_policy_version": policy.policy_version,
        "provider_config_hash": provider_config_hash,
        "max_docs": int(max_docs),
        "total_candidate_count": len(aggregations),
        "completed_candidate_count": sum(1 for row in aggregations if row.get("final_label_row_ready") is True),
        "votes_recorded": len(rows),
        "final_label_rows_ready": sum(1 for row in aggregations if row.get("final_label_row_ready") is True),
        "final_safe_count": final_safe_count,
        "final_reject_count": final_reject_count,
        "aggregation_route_counts": dict(sorted(route_counts.items())),
        "unfinished_candidate_count": len(unfinished),
        "unfinished_candidate_ids_sha256": _hash(sorted(str(row.get("candidate_id") or "") for row in unfinished)),
        "active_concurrency": policy.target_concurrency,
        "private_refetch_concurrency": policy.private_refetch_concurrency,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "vote_throughput_per_second": round(len(rows) / float(elapsed_seconds), 6) if elapsed_seconds > 0 else None,
        "budget_cap_usd": policy.budget_cap_usd,
        "minimum_balance_reserve_usd": policy.minimum_balance_reserve_usd,
        "budget_cap_used_fraction": round(observed_cost / policy.budget_cap_usd, 8) if policy.budget_cap_usd else None,
        "available_balance_policy": "proceed_with_available_balance_preserve_resumable_partials_v1",
        "total_estimated_cost_usd": estimated_cost,
        "total_actual_cost_usd": observed_cost,
        "run_status": run_status,
        "stop_reason": blocked_reason or run_status,
        "blocked_reason": blocked_reason,
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    progress["final_label_progress_hash"] = _hash({key: value for key, value in progress.items() if key != "generated_at_utc"})
    _reject_live_artifact_leakage(progress)
    _write_json_atomic(path, progress)
    return progress


def run_teacher_label_final_batch(
    *,
    manifest: dict[str, Any],
    schedule: dict[str, Any],
    row_provider: Any,
    vote_jsonl: str | Path,
    output_jsonl: str | Path,
    report_path: str | Path,
    repair_queue_path: str | Path,
    progress_path: str | Path,
    live: bool,
    max_docs: int | None,
    current_balance_usd: float | None,
    budget_cap_usd: float,
    minimum_balance_reserve_usd: float = 0.0,
    target_concurrency: int = 200,
    private_refetch_concurrency: int = 128,
    tranche_documents: int = 512,
    env: dict[str, str] | None = None,
    call: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    price_snapshot: PriceSnapshot | None = None,
    provider_config: DeepSeekLabelingConfig | None = None,
    avg_input_tokens: int = 2_400,
    avg_output_tokens: int = 512,
) -> dict[str, Any]:
    """Run or dry-run the final 35k freeze-label BoF policy with durable vote rows."""

    run_started = time.monotonic()
    candidates_all = _validate_final_batch_manifest(manifest)
    if schedule.get("raw_text_serialized") is not False or schedule.get("token_serialized") is not False:
        raise ValueError("source schedule must be a no-raw-text artifact")
    if max_docs is None or int(max_docs) < 1:
        raise ValueError("final label batch requires explicit --max-docs")
    if int(max_docs) > TEACHER_LABEL_FINAL_TARGET:
        raise ValueError("final label max_docs cannot exceed 35000")
    if current_balance_usd is None:
        raise ValueError("final label batch requires current balance for available-balance policy")
    policy = FinalLabelBatchPolicy(
        budget_cap_usd=budget_cap_usd,
        minimum_balance_reserve_usd=minimum_balance_reserve_usd,
        target_concurrency=target_concurrency,
        tranche_documents=tranche_documents,
        private_refetch_concurrency=private_refetch_concurrency,
    )
    if float(current_balance_usd) < policy.minimum_balance_reserve_usd:
        raise ValueError("final label current balance is below required reserve")
    provider_config = provider_config or DeepSeekLabelingConfig()
    if provider_config != DeepSeekLabelingConfig():
        default_config = DeepSeekLabelingConfig()
        if provider_config.model != default_config.model or provider_config.temperature != 0.2:
            raise ValueError("final label batch must use canonical non-thinking DeepSeek V4 Pro settings")
        if provider_config.max_output_tokens != PROMPT_V3_DEEPSEEK_TEACHER_MAX_OUTPUT_TOKENS:
            raise ValueError("final label batch must use prompt-v3 max_output_tokens=512")
        if provider_config.thinking != "disabled" or provider_config.reasoning_effort:
            raise ValueError("final label batch must use thinking disabled")
        if provider_config.stream is not False or provider_config.response_format != {"type": "json_object"}:
            raise ValueError("final label batch must use JSON Output and stream=false")
    candidates = _adaptive_candidate_order(candidates_all[: int(max_docs)], seed=20260606)
    price = price_snapshot or _price_from_official_snapshot(provider_config.model)
    provider_config_hash = _hash(_config_payload(provider_config))
    provider_payload = _public_config_payload(provider_config)
    budget_check = estimate_label_cost_usd(
        label_count=TEACHER_LABEL_FINAL_TARGET * 3,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=price,
        safety_multiplier=1.25,
        budget_cap_usd=policy.budget_cap_usd,
        reserve_usd=0.0,
    )
    per_vote_estimate = estimate_label_cost_usd(
        label_count=1,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        price=price,
        safety_multiplier=1.0,
        budget_cap_usd=policy.budget_cap_usd,
        reserve_usd=0.0,
    )["estimated_cost_usd"]
    call = call or call_deepseek_live
    using_live_client = call is call_deepseek_live
    if live is True and using_live_client:
        prepare_deepseek_live_call(
            {"base_url": DEEPSEEK_BASE_URL, "endpoint": DEEPSEEK_ENDPOINT_PATH},
            live=True,
            max_docs=int(max_docs),
            budget_cap_usd=policy.budget_cap_usd,
            env=env,
        )
    rows_for_report = _read_existing_final_vote_rows(
        vote_jsonl,
        manifest=manifest,
        provider_config=provider_config,
        candidates=candidates,
        policy=policy,
    )
    existing_keys = {str(row["idempotency_key"]) for row in rows_for_report}
    observed_cost = _row_cost_sum(rows_for_report, "actual_cost_usd")
    estimated_cost = _row_cost_sum(rows_for_report, "estimated_cost_usd")
    paid_calls_made = 0
    transport_failures: list[dict[str, Any]] = []
    run_status = "complete" if int(max_docs) == TEACHER_LABEL_FINAL_TARGET else "resumable_partial"
    blocked_reason: str | None = None
    if live is not True:
        run_status = "dry_run_no_paid_calls"
        blocked_reason = "explicit live flag not set"

    def budget_allows(next_call_count: int) -> bool:
        next_call_estimate = float(per_vote_estimate) * next_call_count
        planned_total_run_cost = observed_cost + next_call_estimate
        if planned_total_run_cost > float(policy.budget_cap_usd):
            return False
        return float(current_balance_usd) - next_call_estimate >= float(policy.minimum_balance_reserve_usd)

    def append_new_rows(new_rows: list[dict[str, Any]]) -> None:
        nonlocal observed_cost, estimated_cost, paid_calls_made, rows_for_report
        for row in new_rows:
            key = str(row["idempotency_key"])
            if key in existing_keys:
                continue
            _append_label_row_known_unique(vote_jsonl, row)
            existing_keys.add(key)
            rows_for_report.append(row)
            paid_calls_made += 1
            observed_cost += float(row.get("actual_cost_usd") or 0.0)
            estimated_cost += float(row.get("estimated_cost_usd") or 0.0)

    if run_status in {"complete", "resumable_partial"}:
        for start_index in range(0, len(candidates), policy.tranche_documents):
            tranche = candidates[start_index : start_index + policy.tranche_documents]
            bof3_missing = _adaptive_missing_vote_plan(
                tranche,
                existing_rows=rows_for_report,
                wanted_vote_indexes=(0, 1, 2),
            )
            bof3_missing_count = sum(len(missing) for _candidate, missing in bof3_missing)
            if bof3_missing_count and not budget_allows(bof3_missing_count):
                run_status = "resumable_partial_budget_exhausted"
                blocked_reason = "next planned final BoF call would exceed budget cap or available balance reserve"
                break
            bof3_request_plan = _final_vote_request_tasks(
                manifest=manifest,
                schedule=schedule,
                row_provider=row_provider,
                missing_plan=bof3_missing,
                provider_config=provider_config,
                policy=policy,
                refetch_concurrency=policy.private_refetch_concurrency,
            )
            bof3_tasks = list(bof3_request_plan["tasks"])
            tranche_transport_failures = list(bof3_request_plan["transport_failures"])
            if bof3_tasks:
                result = _run_final_vote_tasks(
                    bof3_tasks,
                    live=live,
                    max_docs=int(max_docs),
                    budget_cap_usd=policy.budget_cap_usd,
                    env=env,
                    call=call,
                    provider_config=provider_config,
                    price=price,
                    per_vote_estimate=float(per_vote_estimate),
                    concurrency=policy.target_concurrency,
                )
                append_new_rows(result["rows"])
                tranche_transport_failures.extend(result["transport_failures"])
            if tranche_transport_failures:
                transport_failures.extend(tranche_transport_failures)
                if any(item.get("http_status_code") == 402 for item in tranche_transport_failures):
                    run_status = "resumable_partial_account_exhausted"
                    blocked_reason = "provider reported insufficient balance; partial final labels preserved"
                else:
                    run_status = "resumable_partial_transport_failures"
                    blocked_reason = "final label transport or private refetch failures queued after retry exhaustion"
                break

    grouped_rows = _rows_by_candidate(rows_for_report)
    grouped_transport: dict[str, list[dict[str, Any]]] = {}
    for item in transport_failures:
        grouped_transport.setdefault(str(item.get("candidate_id") or ""), []).append(item)
    aggregations = [
        _build_final_label_aggregation(
            candidate,
            grouped_rows.get(str(candidate["candidate_id"]), []),
            transport_failures=grouped_transport.get(str(candidate["candidate_id"]), []),
            manifest=manifest,
            policy=policy,
            provider_config_hash=provider_config_hash,
        )
        for candidate in candidates
    ]
    final_label_rows = [
        _final_label_row_from_aggregation(aggregation, candidate)
        for aggregation, candidate in zip(aggregations, candidates, strict=False)
        if aggregation.get("final_label_row_ready") is True
    ]
    if final_label_rows or live is True:
        _write_final_label_rows(output_jsonl, final_label_rows)
    parse_error_rows = [row for row in rows_for_report if row.get("parse_status") == "parse_error"]
    repair_queue = _write_final_repair_queue(
        repair_queue_path,
        parse_error_rows=parse_error_rows,
        transport_failures=transport_failures,
        aggregations=aggregations,
    )
    unfinished = [row for row in aggregations if row.get("final_label_row_ready") is not True]
    if run_status == "complete" and unfinished:
        run_status = "resumable_partial"
        blocked_reason = "some final label candidates require repair or rerun before exact 35k completion"
    progress = _write_final_progress(
        progress_path,
        manifest=manifest,
        policy=policy,
        provider_config_hash=provider_config_hash,
        rows=rows_for_report,
        aggregations=aggregations,
        observed_cost=observed_cost,
        estimated_cost=estimated_cost,
        run_status=run_status,
        blocked_reason=blocked_reason,
        max_docs=int(max_docs),
        elapsed_seconds=time.monotonic() - run_started,
    )
    progress_telemetry = {
        "phase": progress["phase"],
        "total_candidate_count": progress["total_candidate_count"],
        "completed_candidate_count": progress["completed_candidate_count"],
        "final_safe_count": progress["final_safe_count"],
        "final_reject_count": progress["final_reject_count"],
        "unfinished_candidate_count": progress["unfinished_candidate_count"],
        "aggregation_route_counts": progress["aggregation_route_counts"],
        "active_concurrency": progress["active_concurrency"],
        "private_refetch_concurrency": progress["private_refetch_concurrency"],
        "elapsed_seconds": progress["elapsed_seconds"],
        "vote_throughput_per_second": progress["vote_throughput_per_second"],
        "budget_cap_used_fraction": progress["budget_cap_used_fraction"],
        "stop_reason": progress["stop_reason"],
    }
    report = {
        "final_label_batch_report_schema_version": FINAL_LABEL_BATCH_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "run_status": run_status,
        "blocked_reason": blocked_reason,
        "source_candidate_manifest_hash": manifest["manifest_hash"],
        "candidate_count": len(candidates_all),
        "max_docs": int(max_docs),
        "final_label_target": TEACHER_LABEL_FINAL_TARGET,
        "ready_for_live_final_labeling": not _final_manifest_blockers(manifest),
        "final_label_policy_version": policy.policy_version,
        "available_balance_policy": "proceed_with_available_balance_preserve_resumable_partials_v1",
        "full_35k_estimate_required_to_fit_before_launch": False,
        "votes_recorded": len(rows_for_report),
        "paid_api_calls_made": paid_calls_made,
        "paid_api_calls_performed": bool(using_live_client and paid_calls_made > 0),
        "final_label_rows_ready": len(final_label_rows),
        "unfinished_candidate_count": len(unfinished),
        "progress_telemetry": progress_telemetry,
        "transport_failure_count": len(transport_failures),
        "per_vote_parse_status_counts": _status_counts(rows_for_report, "parse_status"),
        "finish_reason_counts": _status_counts(rows_for_report, "finish_reason"),
        "semantic_retry_counts": {},
        "main_batch_semantic_retry_performed": False,
        "provider_config": provider_payload,
        "provider_config_hash": provider_config_hash,
        "price_snapshot": asdict(price),
        "budget_cap_usd": policy.budget_cap_usd,
        "minimum_balance_reserve_usd": policy.minimum_balance_reserve_usd,
        "current_balance_usd": float(current_balance_usd),
        "budget_check": _redact_price_snapshot_urls(budget_check),
        "total_estimated_cost_usd": estimated_cost,
        "total_actual_cost_usd": observed_cost,
        "target_concurrency": policy.target_concurrency,
        "private_refetch_concurrency": policy.private_refetch_concurrency,
        "tranche_documents": policy.tranche_documents,
        "latency_summary": _latency_summary_from_rows(rows_for_report),
        "token_usage_summary": _token_usage_summary(rows_for_report),
        "final_label_aggregations": aggregations,
        "repair_queue_summary": {
            "queued_count": repair_queue["queued_count"],
            "parse_error_vote_count": repair_queue["parse_error_vote_count"],
            "transport_failure_count": repair_queue["transport_failure_count"],
            "by_finish_reason": repair_queue["by_finish_reason"],
        },
        "repair_queue_hash": repair_queue["repair_queue_hash"],
        "progress_hash": progress["final_label_progress_hash"],
        "vote_jsonl_path_sha256": _hash(str(vote_jsonl)),
        "output_jsonl_path_sha256": _hash(str(output_jsonl)),
        "one_document_per_request": True,
        "transport_http_retries_allowed": True,
        "raw_text_serialized": False,
        "token_serialized": False,
        "raw_url_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "commits_or_pushes_performed": False,
    }
    report["final_label_batch_report_hash"] = _hash({key: value for key, value in report.items() if key != "generated_at_utc"})
    _reject_live_artifact_leakage(report)
    _write_json_atomic(report_path, report)
    return report


def estimate_label_cost_usd(
    *,
    label_count: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    price: PriceSnapshot,
    safety_multiplier: float = 1.25,
    budget_cap_usd: float = 11.0,
    reserve_usd: float = 0.75,
    assume_cache_hit: bool = False,
) -> dict[str, Any]:
    if label_count < 0 or avg_input_tokens < 0 or avg_output_tokens < 0:
        raise ValueError("token and label counts must be non-negative")
    input_price = (
        price.input_cache_hit_per_million
        if assume_cache_hit
        else price.input_cache_miss_per_million
    )
    input_cost = label_count * avg_input_tokens * input_price / 1_000_000
    output_cost = label_count * avg_output_tokens * price.output_per_million / 1_000_000
    estimated = (input_cost + output_cost) * safety_multiplier
    remaining_after_reserve = max(0.0, budget_cap_usd - reserve_usd)
    return {
        "model": price.model,
        "label_count": label_count,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "input_cache_assumption": "hit" if assume_cache_hit else "miss",
        "price_snapshot": asdict(price),
        "safety_multiplier": safety_multiplier,
        "budget_cap_usd": budget_cap_usd,
        "reserve_usd": reserve_usd,
        "estimated_cost_usd": estimated,
        "fits_after_reserve": estimated <= remaining_after_reserve,
    }


def append_label_row_resume_safe(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    if not row.get("idempotency_key"):
        raise ValueError("label row requires idempotency_key")
    if row.get("raw_text_serialized") is not False:
        raise ValueError("label rows must not serialize raw text")
    output.parent.mkdir(parents=True, exist_ok=True)

    existing: set[str] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            existing_row = json.loads(line)
            key = existing_row.get("idempotency_key")
            if key:
                existing.add(str(key))
    if str(row["idempotency_key"]) in existing:
        return

    tmp = output.with_suffix(output.suffix + ".tmp")
    prior = output.read_text(encoding="utf-8") if output.exists() else ""
    tmp.write_text(prior + json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TWC teacher-labeling utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    candidates = subparsers.add_parser(
        "build-candidates",
        help="Build a no-raw-text teacher candidate manifest from public decision JSONL.",
    )
    candidates.add_argument("--decision-log", required=True, help="Input public decision JSONL path.")
    candidates.add_argument("--out", required=True, help="Output candidate manifest JSON path.")
    candidates.add_argument("--max-candidates", type=int, default=35_000)
    candidates.add_argument("--seed", type=int, default=20260603)
    candidates.add_argument("--max-per-domain", type=int, default=250)
    candidates.add_argument("--max-per-dump", type=int, default=2_000)
    batch = subparsers.add_parser(
        "plan-label-batch",
        help="Build a deterministic no-raw-text paid-label batch plan without API calls.",
    )
    batch.add_argument("--manifest", required=True, help="Input teacher candidate manifest JSON path.")
    batch.add_argument("--out", required=True, help="Output label batch plan JSON path.")
    batch.add_argument("--label-count", type=int, required=True)
    batch.add_argument("--mode", choices=["pilot", "freeze"], required=True)
    batch.add_argument("--model", default=REQUIRED_DEEPSEEK_TEACHER_MODEL, choices=sorted(ALLOWED_DEEPSEEK_MODELS))
    batch.add_argument("--seed", type=int, default=20260603)
    seed_manifest = subparsers.add_parser(
        "build-seed-manifest",
        help="Derive an exact no-raw seed-label manifest from a larger candidate manifest without API calls.",
    )
    seed_manifest.add_argument("--manifest", required=True, help="Input teacher candidate manifest JSON path.")
    seed_manifest.add_argument("--out", required=True, help="Output seed candidate manifest JSON path.")
    seed_manifest.add_argument("--seed-label-target", type=int, default=TEACHER_LABEL_SEED_TARGET)
    seed_manifest.add_argument("--train-label-count", type=int, default=TEACHER_LABEL_SEED_TRAIN_TARGET)
    seed_manifest.add_argument("--holdout-label-count", type=int, default=TEACHER_LABEL_SEED_HOLDOUT_TARGET)
    seed_manifest.add_argument("--seed", type=int, default=20260603)
    repair = subparsers.add_parser(
        "export-repair-queue",
        help="Export no-raw parse-error repair/adjudication work items from label JSONL.",
    )
    repair.add_argument("--labels-jsonl", required=True, help="Input teacher label JSONL containing parse_error rows.")
    repair.add_argument("--out", required=True, help="Output repair queue JSON path.")
    repair.add_argument("--mode", choices=["manual_adjudication", "explicit_rerun"], required=True)
    repair.add_argument("--source-label-jsonl-sha256", default=None)
    repair.add_argument("--max-manual-count", type=int, default=25)
    readiness = subparsers.add_parser(
        "check-seed-launch-readiness",
        help="Write a no-raw 6k DeepSeek V4 Pro seed-launch readiness report.",
    )
    readiness.add_argument("--config", required=True, help="Input teacher labeling YAML/JSON config path.")
    readiness.add_argument("--manifest", required=True, help="Input teacher candidate manifest JSON path.")
    readiness.add_argument("--out", required=True, help="Output launch-readiness JSON path.")
    readiness.add_argument("--repair-workflow-available", action="store_true")
    readiness.add_argument("--paid-call-unlock", action="store_true")
    readiness.add_argument("--launch-time-price-snapshot-verified", action="store_true")
    seed_run = subparsers.add_parser(
        "run-seed-label-batch",
        help="Run or dry-run the exact 6,000 DeepSeek V4 Pro seed-label batch.",
    )
    seed_run.add_argument("--manifest", required=True, help="Input derived seed candidate manifest JSON path.")
    seed_run.add_argument("--schedule", required=True, help="Input no-raw source schedule JSON path.")
    seed_run.add_argument("--out-labels", required=True, help="Output resume-safe seed label JSONL path.")
    seed_run.add_argument("--report", required=True, help="Output seed batch report JSON path.")
    seed_run.add_argument("--live", action="store_true", help="Perform live DeepSeek calls. Omit for no-paid dry-run.")
    seed_run.add_argument("--max-docs", type=int, required=True)
    seed_run.add_argument("--budget-cap-usd", type=float, required=True)
    seed_run.add_argument("--reserve-usd", type=float, default=0.75)
    seed_run.add_argument("--row-provider", choices=["local-parquet", "hf-fsspec"], required=True)
    seed_run.add_argument("--hf-cache-dir", default="reports/teacher_labeling/hf_seed_row_cache")
    seed_run.add_argument("--cache-delta-cap-bytes", type=int, default=256 * 1024 * 1024)
    seed_run.add_argument("--block-size-bytes", type=int, default=16 * 1024 * 1024)
    seed_run.add_argument("--max-cache-blocks", type=int, default=4)
    seed_run.add_argument("--columns", default="text,id")
    promote_final = subparsers.add_parser(
        "promote-final-35k-candidates",
        help="Promote the diagnostic 35k review candidate into the exact final runner-input manifest.",
    )
    promote_final.add_argument("--review-artifact", required=True, help="Input diagnostic 35k review artifact JSON path.")
    promote_final.add_argument("--attempt-manifest", required=True, help="Input adaptive BoF attempt manifest JSON path.")
    promote_final.add_argument("--schedule", default=None, help="Optional no-raw final refetch schedule used to validate final coverage.")
    promote_final.add_argument("--repair-replacements", default=None, help="Optional random-backfill replacement candidates JSON path.")
    promote_final.add_argument("--out", required=True, help="Output final 35k runner candidate manifest path.")
    promote_final.add_argument("--expected-candidate-count", type=int, default=TEACHER_LABEL_FINAL_TARGET)
    replacement_final = subparsers.add_parser(
        "build-final-promotion-replacements",
        help="Build no-raw random non-borderline backfill candidates to repair final 35k promotion duplicates.",
    )
    replacement_final.add_argument("--review-artifact", required=True, help="Input diagnostic 35k review artifact JSON path.")
    replacement_final.add_argument("--attempt-manifest", required=True, help="Input adaptive BoF attempt manifest JSON path.")
    replacement_final.add_argument(
        "--source-schedule",
        action="append",
        required=True,
        help="Existing no-raw source schedule JSON path used as the replacement offset pool. Repeatable.",
    )
    replacement_final.add_argument("--out", required=True, help="Output replacement candidate artifact JSON path.")
    replacement_final.add_argument("--target-count", type=int, default=None)
    replacement_final.add_argument("--buffer-count", type=int, default=256)
    replacement_final.add_argument("--seed", type=int, default=20260607)
    replacement_final.add_argument("--progress", default=None, help="Optional durable no-raw progress JSON path.")
    replacement_final.add_argument("--expected-candidate-count", type=int, default=TEACHER_LABEL_FINAL_TARGET)
    final_bounds = subparsers.add_parser(
        "recover-final-row-group-bounds",
        help="Recover no-raw Parquet row-group bounds needed by the exact final 35k candidates.",
    )
    final_bounds.add_argument("--manifest", required=True, help="Input final 35k candidate manifest JSON path.")
    final_bounds.add_argument(
        "--source-schedule",
        action="append",
        required=True,
        help="Existing source schedule JSON path used to recover source locators and existing bounds. Repeatable.",
    )
    final_bounds.add_argument("--row-group-bounds", action="append", default=None, help="Existing recovered bounds artifact. Repeatable.")
    final_bounds.add_argument("--out", required=True, help="Output no-raw row-group-bounds artifact JSON path.")
    final_bounds.add_argument("--progress", default=None, help="Optional durable no-raw progress JSON path.")
    final_bounds.add_argument("--cache-dir", default="reports/teacher_labeling/final_row_group_bounds_cache")
    final_bounds.add_argument("--cache-delta-cap-bytes", type=int, default=256 * 1024 * 1024)
    final_bounds.add_argument("--block-size-bytes", type=int, default=8 * 1024 * 1024)
    final_bounds.add_argument("--max-cache-blocks", type=int, default=4)
    final_bounds.add_argument("--skip-identity-head-check", action="store_true")
    final_bounds.add_argument("--expected-candidate-count", type=int, default=TEACHER_LABEL_FINAL_TARGET)
    final_schedule = subparsers.add_parser(
        "build-final-35k-refetch-schedule",
        help="Build the exact candidate-driven source schedule for final 35k labeling.",
    )
    final_schedule.add_argument("--manifest", required=True, help="Input final 35k candidate manifest JSON path.")
    final_schedule.add_argument(
        "--source-schedule",
        action="append",
        required=True,
        help="Existing source schedule JSON path to recover source locators and row-group bounds. Repeatable.",
    )
    final_schedule.add_argument("--row-group-bounds", action="append", default=None, help="Recovered no-raw row-group-bounds artifact. Repeatable.")
    final_schedule.add_argument("--out", required=True, help="Output final candidate-driven refetch schedule JSON path.")
    final_schedule.add_argument("--source-metadata-hash", default=None)
    final_schedule.add_argument("--expected-candidate-count", type=int, default=TEACHER_LABEL_FINAL_TARGET)
    final_run = subparsers.add_parser(
        "run-final-label-batch",
        help="Run or dry-run the final exact-35k DeepSeek V4 Pro freeze-label batch.",
    )
    final_run.add_argument("--manifest", required=True, help="Input exact 35k final-freeze candidate manifest JSON path.")
    final_run.add_argument("--schedule", required=True, help="Input no-raw source schedule JSON path.")
    final_run.add_argument("--out-votes", required=True, help="Output resume-safe final BoF vote JSONL path.")
    final_run.add_argument("--out-labels", required=True, help="Output aggregated one-row-per-candidate final label JSONL path.")
    final_run.add_argument("--report", required=True, help="Output no-raw final batch report JSON path.")
    final_run.add_argument("--repair-queue", required=True, help="Output no-raw repair/rerun queue JSON path.")
    final_run.add_argument("--progress", required=True, help="Output no-raw progress state JSON path.")
    final_run.add_argument("--live", action="store_true", help="Perform live DeepSeek calls. Omit for no-paid dry-run.")
    final_run.add_argument("--max-docs", type=int, required=True)
    final_run.add_argument("--current-balance-usd", type=float, required=True)
    final_run.add_argument("--budget-cap-usd", type=float, required=True)
    final_run.add_argument("--minimum-balance-reserve-usd", type=float, default=0.0)
    final_run.add_argument("--target-concurrency", type=int, default=200)
    final_run.add_argument("--tranche-documents", type=int, default=512)
    final_run.add_argument("--private-refetch-concurrency", type=int, default=128)
    final_run.add_argument("--row-provider", choices=["local-parquet", "hf-fsspec"], required=True)
    final_run.add_argument("--hf-cache-dir", default="reports/teacher_labeling/hf_final_label_row_cache")
    final_run.add_argument("--cache-delta-cap-bytes", type=int, default=256 * 1024 * 1024)
    final_run.add_argument("--block-size-bytes", type=int, default=16 * 1024 * 1024)
    final_run.add_argument("--max-cache-blocks", type=int, default=4)
    final_run.add_argument("--columns", default="text,id")
    adaptive_manifest = subparsers.add_parser(
        "build-adaptive-bof-attempt-manifest",
        help="Build a no-raw bounded attempt manifest for adaptive BoF borderline mining.",
    )
    adaptive_manifest.add_argument("--manifest", required=True, help="Input source candidate manifest JSON path.")
    adaptive_manifest.add_argument("--schedule", required=True, help="Input no-raw source schedule JSON path.")
    adaptive_manifest.add_argument("--out", required=True, help="Output adaptive BoF attempt manifest JSON path.")
    adaptive_manifest.add_argument("--max-attempted-documents", type=int, default=25_000)
    adaptive_manifest.add_argument("--seed", type=int, default=20260605)
    adaptive = subparsers.add_parser(
        "run-adaptive-bof-mining",
        help="Run or dry-run adaptive BoF3 -> BoF5 borderline mining.",
    )
    adaptive.add_argument("--manifest", required=True, help="Input no-raw candidate manifest JSON path.")
    adaptive.add_argument("--schedule", required=True, help="Input no-raw source schedule JSON path.")
    adaptive.add_argument("--out-votes", required=True, help="Output resume-safe adaptive vote JSONL path.")
    adaptive.add_argument("--report", required=True, help="Output no-raw adaptive mining report JSON path.")
    adaptive.add_argument("--selected-manifest", required=True, help="Output no-raw selected borderline manifest JSON path.")
    adaptive.add_argument("--repair-queue", required=True, help="Output no-raw repair queue JSON path.")
    adaptive.add_argument("--progress", required=True, help="Output no-raw progress state JSON path.")
    adaptive.add_argument("--live", action="store_true", help="Perform live DeepSeek calls. Omit for no-paid dry-run.")
    adaptive.add_argument("--max-docs", type=int, required=True)
    adaptive.add_argument("--current-balance-usd", type=float, required=True)
    adaptive.add_argument("--selected-target", type=int, default=4_000)
    adaptive.add_argument("--max-attempted-documents", type=int, default=25_000)
    adaptive.add_argument("--budget-cap-usd", type=float, default=8.0)
    adaptive.add_argument("--minimum-balance-reserve-usd", type=float, default=30.0)
    adaptive.add_argument("--target-concurrency", type=int, default=128)
    adaptive.add_argument("--concurrency-ramp", default="16,64,128")
    adaptive.add_argument("--tranche-documents", type=int, default=512)
    adaptive.add_argument("--private-refetch-concurrency", type=int, default=128)
    adaptive.add_argument("--row-provider", choices=["local-parquet", "hf-fsspec"], required=True)
    adaptive.add_argument("--hf-cache-dir", default="reports/teacher_labeling/hf_adaptive_bof_row_cache")
    adaptive.add_argument("--cache-delta-cap-bytes", type=int, default=256 * 1024 * 1024)
    adaptive.add_argument("--block-size-bytes", type=int, default=16 * 1024 * 1024)
    adaptive.add_argument("--max-cache-blocks", type=int, default=4)
    adaptive.add_argument("--columns", default="text,id")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "build-candidates":
        config = CandidateSelectionConfig(
            seed=args.seed,
            max_candidates=args.max_candidates,
            max_per_domain=args.max_per_domain,
            max_per_dump=args.max_per_dump,
        )
        manifest = write_teacher_candidate_manifest_from_decision_log(
            args.decision_log,
            args.out,
            config=config,
        )
        print(
            json.dumps(
                {
                    "out": args.out,
                    "candidate_count": len(manifest["candidates"]),
                    "manifest_hash": manifest["manifest_hash"],
                    "raw_text_serialized": manifest["raw_text_serialized"],
                    "accepted_manifest": manifest["accepted_manifest"],
                    "required_strata_missing": manifest["quota_report"]["required_strata_missing"],
                    "required_strata_unavailable": manifest["quota_report"]["required_strata_unavailable"],
                    "candidate_count_shortfall": manifest["quota_report"]["candidate_count_shortfall"],
                    "ready_for_paid_labeling": manifest["label_readiness"]["ready_for_paid_labeling"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "plan-label-batch":
        plan = write_paid_label_batch_plan(
            args.manifest,
            args.out,
            label_count=args.label_count,
            mode=args.mode,
            model=args.model,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "out": args.out,
                    "candidate_count": plan["candidate_count"],
                    "batch_plan_hash": plan["batch_plan_hash"],
                    "ready_for_paid_labeling": plan["freeze_ready_label_distribution"],
                    "pilot_incomplete_distribution": plan["pilot_incomplete_distribution"],
                    "fits_after_reserve": plan["budget_check"]["fits_after_reserve"],
                    "estimated_cost_usd": plan["budget_check"]["estimated_cost_usd"],
                    "paid_api_calls_performed": plan["paid_api_calls_performed"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-seed-manifest":
        manifest = write_seed_label_candidate_manifest(
            args.manifest,
            args.out,
            seed_label_target=args.seed_label_target,
            train_label_count=args.train_label_count,
            holdout_label_count=args.holdout_label_count,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "out": args.out,
                    "candidate_count": len(manifest["candidates"]),
                    "manifest_hash": manifest["manifest_hash"],
                    "source_candidate_manifest_hash": manifest["source_candidate_manifest_hash"],
                    "ready_for_paid_labeling": manifest["label_readiness"]["ready_for_paid_labeling"],
                    "provisional_training_labels": manifest["seed_split"]["provisional_training_labels"],
                    "labeled_holdout_labels": manifest["seed_split"]["labeled_holdout_labels"],
                    "private_refetch_required_count": manifest["quota_report"]["private_refetch_required_count"],
                    "paid_api_calls_performed": False,
                    "raw_text_serialized": manifest["raw_text_serialized"],
                    "accepted_manifest": manifest["accepted_manifest"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "export-repair-queue":
        rows = _read_jsonl_objects(args.labels_jsonl)
        queue = build_parse_error_repair_queue(
            rows,
            mode=args.mode,
            source_label_jsonl_sha256=args.source_label_jsonl_sha256,
            max_manual_count=args.max_manual_count,
        )
        _write_json_atomic(args.out, queue)
        print(
            json.dumps(
                {
                    "out": args.out,
                    "queued_count": queue["queued_count"],
                    "repair_queue_mode": queue["repair_queue_mode"],
                    "repair_queue_hash": queue["repair_queue_hash"],
                    "paid_api_calls_performed": queue["paid_api_calls_performed"],
                    "raw_text_serialized": queue["raw_text_serialized"],
                    "raw_model_response_serialized": queue["raw_model_response_serialized"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "check-seed-launch-readiness":
        config_path = Path(args.config)
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - exercised by packaging checks
                raise RuntimeError("PyYAML is required to read teacher labeling YAML configs") from exc
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("teacher labeling config must be an object")
        report = build_seed_launch_readiness_report(
            config=config,
            candidate_manifest_path=args.manifest,
            repair_workflow_available=bool(args.repair_workflow_available),
            paid_call_unlock=bool(args.paid_call_unlock),
            launch_time_price_snapshot_verified=bool(args.launch_time_price_snapshot_verified),
        )
        _write_json_atomic(args.out, report)
        print(
            json.dumps(
                {
                    "out": args.out,
                    "ready_for_seed_launch": report["ready_for_seed_launch"],
                    "blockers": report["blockers"],
                    "seed_launch_readiness_hash": report["seed_launch_readiness_hash"],
                    "paid_api_calls_performed": report["paid_api_calls_performed"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-seed-label-batch":
        manifest = load_candidate_manifest(args.manifest)
        schedule = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
        columns = tuple(column.strip() for column in str(args.columns).split(",") if column.strip())
        if args.row_provider == "local-parquet":
            from twc.data.decision_log import LocalParquetRowProvider

            row_provider = LocalParquetRowProvider(columns=columns or None)
        else:
            from twc.data.decision_log import HFFsspecParquetRowProvider

            row_provider = HFFsspecParquetRowProvider(
                cache_dir=args.hf_cache_dir,
                cache_delta_cap_bytes=args.cache_delta_cap_bytes,
                block_size_bytes=args.block_size_bytes,
                columns=columns or ("text", "id"),
                max_cache_blocks=args.max_cache_blocks,
            )
        report = run_teacher_label_seed_batch(
            manifest=manifest,
            schedule=schedule,
            row_provider=row_provider,
            output_jsonl=args.out_labels,
            report_path=args.report,
            live=bool(args.live),
            max_docs=args.max_docs,
            budget_cap_usd=args.budget_cap_usd,
            reserve_usd=args.reserve_usd,
        )
        print(
            json.dumps(
                {
                    "report": args.report,
                    "out_labels": args.out_labels,
                    "run_status": report["run_status"],
                    "blocked_reason": report["blocked_reason"],
                    "documents_labeled": report["documents_labeled"],
                    "paid_api_calls_made": report["paid_api_calls_made"],
                    "paid_api_calls_performed": report["paid_api_calls_performed"],
                    "parse_status_counts": report["parse_status_counts"],
                    "seed_batch_report_hash": report["seed_batch_report_hash"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "promote-final-35k-candidates":
        manifest = write_final_label_manifest_from_review_candidate(
            args.review_artifact,
            args.attempt_manifest,
            args.out,
            schedule_path=args.schedule,
            repair_replacements_path=args.repair_replacements,
            expected_candidate_count=args.expected_candidate_count,
        )
        print(
            json.dumps(
                {
                    "out": args.out,
                    "candidate_count": len(manifest["candidates"]),
                    "manifest_hash": manifest["manifest_hash"],
                    "ready_for_paid_labeling": manifest["label_readiness"]["ready_for_paid_labeling"],
                    "borderline_identity_restored_from_attempt_manifest_count": manifest["promotion_summary"][
                        "borderline_identity_restored_from_attempt_manifest_count"
                    ],
                    "schedule_refetch_coverage_validated": manifest["promotion_summary"]["schedule_refetch_coverage_validated"],
                    "paid_api_calls_performed": False,
                    "raw_text_serialized": manifest["raw_text_serialized"],
                    "accepted_manifest": manifest["accepted_manifest"],
                    "labels_are_corpus_approval": manifest["labels_are_corpus_approval"],
                    "training_shards_created": manifest["training_shards_created"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-final-promotion-replacements":
        artifact = write_final_promotion_replacement_candidate_artifact(
            args.review_artifact,
            args.attempt_manifest,
            args.source_schedule,
            args.out,
            target_count=args.target_count,
            buffer_count=args.buffer_count,
            seed=args.seed,
            progress_path=args.progress,
            expected_candidate_count=args.expected_candidate_count,
        )
        print(
            json.dumps(
                {
                    "out": args.out,
                    "artifact_hash": artifact["artifact_hash"],
                    "candidate_count": artifact["candidate_count"],
                    "target_count": artifact["target_count"],
                    "buffer_count_available": artifact["buffer_count_available"],
                    "skipped_collision_count": artifact["skipped_collision_count"],
                    "paid_api_calls_performed": False,
                    "raw_text_serialized": artifact["raw_text_serialized"],
                    "raw_url_serialized": artifact["raw_url_serialized"],
                    "accepted_manifest": artifact["accepted_manifest"],
                    "labels_are_corpus_approval": artifact["labels_are_corpus_approval"],
                    "training_shards_created": artifact["training_shards_created"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "recover-final-row-group-bounds":
        artifact = write_final_row_group_bounds_recovery_artifact(
            args.manifest,
            args.source_schedule,
            args.out,
            row_group_bounds_paths=args.row_group_bounds,
            progress_path=args.progress,
            cache_dir=args.cache_dir,
            cache_delta_cap_bytes=args.cache_delta_cap_bytes,
            block_size_bytes=args.block_size_bytes,
            max_cache_blocks=args.max_cache_blocks,
            verify_identity_headers=not bool(args.skip_identity_head_check),
            expected_candidate_count=args.expected_candidate_count,
        )
        print(
            json.dumps(
                {
                    "out": args.out,
                    "artifact_hash": artifact["artifact_hash"],
                    "candidate_count": artifact["candidate_count"],
                    "required_row_group_count": artifact["required_row_group_count"],
                    "resolved_row_group_count": artifact["resolved_row_group_count"],
                    "source_file_count": artifact["source_file_count"],
                    "paid_api_calls_performed": False,
                    "raw_text_serialized": artifact["raw_text_serialized"],
                    "raw_url_serialized": artifact["raw_url_serialized"],
                    "accepted_manifest": artifact["accepted_manifest"],
                    "labels_are_corpus_approval": artifact["labels_are_corpus_approval"],
                    "training_shards_created": artifact["training_shards_created"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-final-35k-refetch-schedule":
        schedule = write_final_35k_candidate_refetch_schedule(
            args.manifest,
            args.source_schedule,
            args.out,
            row_group_bounds_paths=args.row_group_bounds,
            source_metadata_hash=args.source_metadata_hash,
            expected_candidate_count=args.expected_candidate_count,
        )
        print(
            json.dumps(
                {
                    "out": args.out,
                    "schedule_hash": schedule["schedule_hash"],
                    "algorithm_version": schedule["algorithm_version"],
                    "candidate_count": schedule["candidate_count"],
                    "entry_count": schedule["entry_count"],
                    "offset_count": schedule["offset_count"],
                    "missing_scheduled_offset_count": schedule["coverage"]["missing_scheduled_offset_count"],
                    "unused_scheduled_offset_count": schedule["coverage"]["unused_scheduled_offset_count"],
                    "paid_api_calls_performed": False,
                    "raw_text_serialized": schedule["raw_text_serialized"],
                    "raw_url_serialized": schedule["raw_url_serialized"],
                    "accepted_manifest": schedule["accepted_manifest"],
                    "labels_are_corpus_approval": schedule["labels_are_corpus_approval"],
                    "training_shards_created": schedule["training_shards_created"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-final-label-batch":
        manifest = load_candidate_manifest(args.manifest)
        schedule = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
        columns = tuple(column.strip() for column in str(args.columns).split(",") if column.strip())
        if args.row_provider == "local-parquet":
            from twc.data.decision_log import LocalParquetRowProvider

            row_provider = LocalParquetRowProvider(columns=columns or None)
        else:
            from twc.data.decision_log import HFFsspecParquetRowProvider

            row_provider = HFFsspecParquetRowProvider(
                cache_dir=args.hf_cache_dir,
                cache_delta_cap_bytes=args.cache_delta_cap_bytes,
                block_size_bytes=args.block_size_bytes,
                columns=columns or ("text", "id"),
                max_cache_blocks=args.max_cache_blocks,
                manage_hf_cache_env=False,
            )
        report = run_teacher_label_final_batch(
            manifest=manifest,
            schedule=schedule,
            row_provider=row_provider,
            vote_jsonl=args.out_votes,
            output_jsonl=args.out_labels,
            report_path=args.report,
            repair_queue_path=args.repair_queue,
            progress_path=args.progress,
            live=bool(args.live),
            max_docs=args.max_docs,
            current_balance_usd=args.current_balance_usd,
            budget_cap_usd=args.budget_cap_usd,
            minimum_balance_reserve_usd=args.minimum_balance_reserve_usd,
            target_concurrency=args.target_concurrency,
            private_refetch_concurrency=args.private_refetch_concurrency,
            tranche_documents=args.tranche_documents,
        )
        print(
            json.dumps(
                {
                    "report": args.report,
                    "out_votes": args.out_votes,
                    "out_labels": args.out_labels,
                    "run_status": report["run_status"],
                    "blocked_reason": report["blocked_reason"],
                    "votes_recorded": report["votes_recorded"],
                    "final_label_rows_ready": report["final_label_rows_ready"],
                    "unfinished_candidate_count": report["unfinished_candidate_count"],
                    "paid_api_calls_made": report["paid_api_calls_made"],
                    "paid_api_calls_performed": report["paid_api_calls_performed"],
                    "final_label_batch_report_hash": report["final_label_batch_report_hash"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-adaptive-bof-attempt-manifest":
        source_manifest = load_candidate_manifest(args.manifest)
        schedule = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
        manifest = build_adaptive_bof_attempt_manifest(
            source_manifest,
            schedule=schedule,
            max_attempted_documents=args.max_attempted_documents,
            seed=args.seed,
        )
        _write_json_atomic(args.out, manifest)
        print(
            json.dumps(
                {
                    "out": args.out,
                    "candidate_count": manifest["candidate_count"],
                    "candidate_count_shortfall": manifest["candidate_count_shortfall"],
                    "ready_for_adaptive_mining": manifest["attempt_readiness"]["ready_for_adaptive_mining"],
                    "manifest_hash": manifest["manifest_hash"],
                    "raw_text_serialized": manifest["raw_text_serialized"],
                    "raw_url_serialized": manifest["raw_url_serialized"],
                    "accepted_manifest": manifest["accepted_manifest"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "run-adaptive-bof-mining":
        manifest = load_candidate_manifest(args.manifest)
        schedule = json.loads(Path(args.schedule).read_text(encoding="utf-8"))
        columns = tuple(column.strip() for column in str(args.columns).split(",") if column.strip())
        if args.row_provider == "local-parquet":
            from twc.data.decision_log import LocalParquetRowProvider

            row_provider = LocalParquetRowProvider(columns=columns or None)
        else:
            from twc.data.decision_log import HFFsspecParquetRowProvider

            row_provider = HFFsspecParquetRowProvider(
                cache_dir=args.hf_cache_dir,
                cache_delta_cap_bytes=args.cache_delta_cap_bytes,
                block_size_bytes=args.block_size_bytes,
                columns=columns or ("text", "id"),
                max_cache_blocks=args.max_cache_blocks,
                manage_hf_cache_env=False,
            )
        ramp = tuple(int(part.strip()) for part in str(args.concurrency_ramp).split(",") if part.strip())
        policy = AdaptiveBoFMiningPolicy(
            selected_target=args.selected_target,
            max_attempted_documents=args.max_attempted_documents,
            budget_cap_usd=args.budget_cap_usd,
            minimum_balance_reserve_usd=args.minimum_balance_reserve_usd,
            target_concurrency=args.target_concurrency,
            concurrency_ramp=ramp,
            tranche_documents=args.tranche_documents,
            private_refetch_concurrency=args.private_refetch_concurrency,
        )
        report = run_teacher_label_adaptive_bof_mining(
            manifest=manifest,
            schedule=schedule,
            row_provider=row_provider,
            output_jsonl=args.out_votes,
            report_path=args.report,
            selected_manifest_path=args.selected_manifest,
            repair_queue_path=args.repair_queue,
            progress_path=args.progress,
            live=bool(args.live),
            max_docs=args.max_docs,
            current_balance_usd=args.current_balance_usd,
            policy=policy,
        )
        print(
            json.dumps(
                {
                    "report": args.report,
                    "out_votes": args.out_votes,
                    "selected_manifest": args.selected_manifest,
                    "run_status": report["run_status"],
                    "blocked_reason": report["blocked_reason"],
                    "attempted_document_count": report["attempted_document_count"],
                    "selected_count": report["selected_count"],
                    "votes_recorded": report["votes_recorded"],
                    "paid_api_calls_made": report["paid_api_calls_made"],
                    "paid_api_calls_performed": report["paid_api_calls_performed"],
                    "adaptive_bof_mining_report_hash": report["adaptive_bof_mining_report_hash"],
                },
                sort_keys=True,
            )
        )
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
