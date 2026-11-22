"""Chunk-native deterministic filter for consciousness-risk concepts.

The older deterministic foundation evaluates whole documents and intentionally
records rich private match detail. This module is the public filtering surface:
it only emits redacted rule IDs/counts and only terminal-rejects core text.
Halo matches are evidence for downstream classifiers, never accepted text gaps.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.data.decision_schema import stable_json_hash
from twc.data.normalize import (
    CANONICAL_STEPS,
    NORMALIZER_VERSION,
    CanonicalNormalization,
    NormalizedDocument,
    normalize_canonical_text,
    normalize_text,
)
from twc.data.risk_ontology import MATCH_TIMEOUT_MS, RiskOntology, RiskRule


DETERMINISTIC_FILTER_POLICY_KIND = "twc_deterministic_filter_v1_policy"
DETERMINISTIC_FILTER_RESULT_KIND = "twc_deterministic_filter_candidate_result"
DETERMINISTIC_FILTER_POLICY_VERSION = "deterministic_filter_v1"
DETERMINISTIC_FILTER_SCHEMA_VERSION = 1
DETERMINISTIC_FILTER_REDACTED_HIT_SCHEMA_VERSION = 1
TERMINAL_SEVERITIES = frozenset(
    {
        "hard_reject",
        "phrase_reject",
        "density_reject",
        "source_reject",
    }
)
TEXT_RULE_SKIPPED_VIEWS = frozenset({"source_metadata", "url"})
ACTION_MAP = {
    "hard_reject": "reject",
    "phrase_reject": "reject",
    "density_reject": "reject",
    "source_reject": "reject",
    "audit_only": "audit_only_pass",
}


@dataclass(frozen=True)
class _RuntimeRule:
    rule: RiskRule
    required_needles: tuple[str, ...]
    density_threshold: int


@dataclass(frozen=True)
class DeterministicFilterRuntime:
    """Validated, no-raw runtime view of the deterministic filter policy."""

    policy_payload: Mapping[str, Any]
    policy_hash: str
    rules: tuple[_RuntimeRule, ...]
    canonical_only: bool
    rules_by_first_required_needle: tuple[tuple[str, tuple[_RuntimeRule, ...]], ...]
    rules_without_required_needles: tuple[_RuntimeRule, ...]


def build_deterministic_filter_policy_payload(ontology: RiskOntology) -> dict[str, Any]:
    """Return the hash-bound public policy payload for deterministic filtering."""

    payload: dict[str, Any] = {
        "artifact_kind": DETERMINISTIC_FILTER_POLICY_KIND,
        "schema_version": DETERMINISTIC_FILTER_SCHEMA_VERSION,
        "policy_version": DETERMINISTIC_FILTER_POLICY_VERSION,
        "ontology_semantic_hash": ontology.semantic_hash,
        "ontology_source_tree_hash": ontology.source_tree_hash,
        "compiled_rule_manifest_hash": stable_json_hash(list(ontology.compiled_rule_manifest)),
        "regex_engine": ontology.regex_engine,
        "regex_engine_version": ontology.regex_engine_version,
        "yaml_parser_version": ontology.yaml_parser_version,
        "python_unicode_version": ontology.python_unicode_version,
        "normalizer_version": NORMALIZER_VERSION,
        "normalization_steps": list(CANONICAL_STEPS),
        "match_timeout_ms": int(ontology.match_timeout_ms),
        "action_map": dict(ACTION_MAP),
        "text_terminal_scan_scope": "candidate_core_only",
        "halo_scan_scope": "evidence_only_never_terminal",
        "source_metadata_scan_scope": "not_serialized_in_chunk_record_v1",
        "redacted_hit_schema_version": DETERMINISTIC_FILTER_REDACTED_HIT_SCHEMA_VERSION,
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
    payload["deterministic_filter_policy_hash"] = _policy_hash(payload)
    validate_deterministic_filter_policy_payload(payload)
    return payload


def build_deterministic_filter_runtime(
    ontology: RiskOntology,
    policy_payload: Mapping[str, Any],
) -> DeterministicFilterRuntime:
    """Build the reusable candidate-evaluation runtime for one worker."""

    validate_deterministic_filter_policy_payload(policy_payload)
    if policy_payload.get("ontology_semantic_hash") != ontology.semantic_hash:
        raise ValueError("deterministic filter runtime ontology semantic hash mismatch")
    if policy_payload.get("ontology_source_tree_hash") != ontology.source_tree_hash:
        raise ValueError("deterministic filter runtime ontology source hash mismatch")
    if policy_payload.get("compiled_rule_manifest_hash") != stable_json_hash(list(ontology.compiled_rule_manifest)):
        raise ValueError("deterministic filter runtime compiled rule manifest mismatch")
    rules = _build_runtime_rules(ontology)
    rules_by_first_needle, rules_without_needles = _index_runtime_rules_by_first_required_needle(rules)
    return DeterministicFilterRuntime(
        policy_payload=policy_payload,
        policy_hash=str(policy_payload["deterministic_filter_policy_hash"]),
        rules=rules,
        canonical_only=all(runtime_rule.rule.view == "canonical" for runtime_rule in rules),
        rules_by_first_required_needle=rules_by_first_needle,
        rules_without_required_needles=rules_without_needles,
    )


def _build_runtime_rules(ontology: RiskOntology) -> tuple[_RuntimeRule, ...]:
    runtime_rules: list[_RuntimeRule] = []
    for rule in ontology.rules:
        if rule.view in TEXT_RULE_SKIPPED_VIEWS or rule.type == "url_regex":
            continue
        runtime_rules.append(
            _RuntimeRule(
                rule=rule,
                required_needles=_required_needles(rule),
                density_threshold=int(rule.threshold),
            )
        )
    return tuple(runtime_rules)


def _index_runtime_rules_by_first_required_needle(
    rules: tuple[_RuntimeRule, ...],
) -> tuple[tuple[tuple[str, tuple[_RuntimeRule, ...]], ...], tuple[_RuntimeRule, ...]]:
    grouped: dict[str, list[_RuntimeRule]] = {}
    no_needle_rules: list[_RuntimeRule] = []
    for runtime_rule in rules:
        if not runtime_rule.required_needles:
            no_needle_rules.append(runtime_rule)
            continue
        first_needle = runtime_rule.required_needles[0]
        grouped.setdefault(first_needle, []).append(runtime_rule)
    return tuple((needle, tuple(group_rules)) for needle, group_rules in grouped.items()), tuple(no_needle_rules)


def validate_deterministic_filter_policy_payload(payload: Mapping[str, Any]) -> None:
    """Validate the deterministic filter public policy payload."""

    validate_public_no_raw_payload(payload)
    if payload.get("artifact_kind") != DETERMINISTIC_FILTER_POLICY_KIND:
        raise ValueError("deterministic filter policy artifact kind mismatch")
    if payload.get("schema_version") != DETERMINISTIC_FILTER_SCHEMA_VERSION:
        raise ValueError("deterministic filter policy schema version mismatch")
    if payload.get("policy_version") != DETERMINISTIC_FILTER_POLICY_VERSION:
        raise ValueError("deterministic filter policy version mismatch")
    expected_hash = _policy_hash(payload)
    if payload.get("deterministic_filter_policy_hash") != expected_hash:
        raise ValueError("deterministic filter policy hash mismatch")
    if payload.get("action_map") != ACTION_MAP:
        raise ValueError("deterministic filter policy action map mismatch")
    if payload.get("text_terminal_scan_scope") != "candidate_core_only":
        raise ValueError("deterministic filter terminal scan scope mismatch")
    if payload.get("halo_scan_scope") != "evidence_only_never_terminal":
        raise ValueError("deterministic filter halo scan scope mismatch")
    if payload.get("match_timeout_ms") != MATCH_TIMEOUT_MS:
        raise ValueError("deterministic filter match timeout mismatch")


def evaluate_chunk_candidate_deterministic_filter(
    candidate: Mapping[str, Any],
    *,
    raw_text: str,
    ontology: RiskOntology,
    policy_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one chunk candidate and return a no-raw redacted result."""

    runtime = build_deterministic_filter_runtime(ontology, policy_payload)
    return evaluate_chunk_candidate_deterministic_filter_with_runtime(
        candidate,
        raw_text=raw_text,
        runtime=runtime,
    )


def evaluate_chunk_candidate_deterministic_filter_with_runtime(
    candidate: Mapping[str, Any],
    *,
    raw_text: str,
    runtime: DeterministicFilterRuntime,
    normalization_cache: MutableMapping[tuple[Any, ...], NormalizedDocument] | None = None,
) -> dict[str, Any]:
    """Evaluate one chunk candidate using a prevalidated runtime policy."""

    candidate_id = _require_hash(candidate.get("candidate_id_hash"), "candidate_id_hash")
    core_start, core_end, evidence_start, evidence_end = _candidate_spans(candidate, raw_text)
    core_text = raw_text[core_start:core_end]
    left_halo_text = raw_text[evidence_start:core_start]
    right_halo_text = raw_text[core_end:evidence_end]

    raw_identity = (id(raw_text), len(raw_text), NORMALIZER_VERSION)
    if runtime.canonical_only:
        core_hits = _match_canonical_rules_runtime(
            _normalize_canonical_cached(
                core_text,
                normalization_cache,
                (*raw_identity, "core", core_start, core_end),
            ).canonical_text,
            runtime,
        )
        halo_hits = _match_canonical_rules_runtime(
            _normalize_canonical_cached(
                f"{left_halo_text}\n{right_halo_text}",
                normalization_cache,
                (*raw_identity, "halo", evidence_start, core_start, core_end, evidence_end),
            ).canonical_text,
            runtime,
        )
    else:
        core_hits = _match_text_rules_runtime(
            _normalize_cached(core_text, normalization_cache, (*raw_identity, "core", core_start, core_end)),
            runtime,
        )
        halo_hits = _match_text_rules_runtime(
            _normalize_cached(
                f"{left_halo_text}\n{right_halo_text}",
                normalization_cache,
                (*raw_identity, "halo", evidence_start, core_start, core_end, evidence_end),
            ),
            runtime,
        )
    terminal_hits = _sorted_hits(hit for hit in core_hits if hit["severity"] in TERMINAL_SEVERITIES)
    audit_hits = _sorted_hits(hit for hit in core_hits if hit["severity"] == "audit_only")
    halo_evidence_hits = _sorted_hits(halo_hits)

    payload: dict[str, Any] = {
        "artifact_kind": DETERMINISTIC_FILTER_RESULT_KIND,
        "schema_version": DETERMINISTIC_FILTER_SCHEMA_VERSION,
        "policy_version": DETERMINISTIC_FILTER_POLICY_VERSION,
        "deterministic_filter_policy_hash": runtime.policy_hash,
        "candidate_id_hash": candidate_id,
        "terminal_decision": "reject" if terminal_hits else None,
        "terminal_hit_count": len(terminal_hits),
        "audit_only_hit_count": len(audit_hits),
        "halo_evidence_hit_count": len(halo_evidence_hits),
        "terminal_rule_ids": _unique_rule_ids(terminal_hits),
        "audit_only_rule_ids": _unique_rule_ids(audit_hits),
        "halo_evidence_rule_ids": _unique_rule_ids(halo_evidence_hits),
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
    payload["deterministic_filter_result_hash"] = stable_json_hash(payload)
    validate_public_no_raw_payload(payload)
    return payload


def _policy_hash(payload: Mapping[str, Any]) -> str:
    return stable_json_hash({key: value for key, value in payload.items() if key != "deterministic_filter_policy_hash"})


def _candidate_spans(candidate: Mapping[str, Any], raw_text: str) -> tuple[int, int, int, int]:
    core_start = _nonnegative_int(candidate.get("core_char_start"), "core_char_start")
    core_end = _positive_int(candidate.get("core_char_end"), "core_char_end")
    evidence_start = _nonnegative_int(candidate.get("evidence_char_start"), "evidence_char_start")
    evidence_end = _positive_int(candidate.get("evidence_char_end"), "evidence_char_end")
    if not evidence_start <= core_start < core_end <= evidence_end <= len(raw_text):
        raise ValueError("candidate deterministic filter spans are invalid")
    return core_start, core_end, evidence_start, evidence_end


def _match_text_rules(normalized: NormalizedDocument, ontology: RiskOntology) -> list[dict[str, str]]:
    return _match_runtime_rules(normalized, _build_runtime_rules(ontology))


def _match_text_rules_runtime(normalized: NormalizedDocument, runtime: DeterministicFilterRuntime) -> list[dict[str, str]]:
    return _match_runtime_rules(normalized, runtime.rules)


def _match_runtime_rules(normalized: NormalizedDocument, rules: tuple[_RuntimeRule, ...]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    view_cache: dict[str, str] = {}
    for runtime_rule in rules:
        rule = runtime_rule.rule
        view_text = view_cache.get(rule.view)
        if view_text is None:
            view_text = normalized.view(rule.view)
            view_cache[rule.view] = view_text
        if runtime_rule.required_needles and any(needle not in view_text for needle in runtime_rule.required_needles):
            continue
        match_count = _match_count(rule, view_text)
        if rule.severity == "density_reject" and match_count < runtime_rule.density_threshold:
            continue
        if match_count > 0:
            hits.append(
                {
                    "rule_id": rule.id,
                    "severity": rule.severity,
                    "category": rule.category,
                }
            )
    return hits


def _match_canonical_rules_runtime(canonical_text: str, runtime: DeterministicFilterRuntime) -> list[dict[str, str]]:
    return _match_runtime_rules_for_view(canonical_text, runtime)


def _match_runtime_rules_for_view(view_text: str, runtime: DeterministicFilterRuntime) -> list[dict[str, str]]:
    if not runtime.rules_by_first_required_needle and not runtime.rules_without_required_needles:
        return _match_runtime_rules_for_view_slow(view_text, runtime.rules)
    hits: list[dict[str, str]] = []
    for runtime_rule in runtime.rules_without_required_needles:
        _append_runtime_rule_hit(hits, runtime_rule, view_text)
    for first_needle, rules in runtime.rules_by_first_required_needle:
        if first_needle not in view_text:
            continue
        for runtime_rule in rules:
            remaining_needles = runtime_rule.required_needles[1:]
            if remaining_needles and any(needle not in view_text for needle in remaining_needles):
                continue
            _append_runtime_rule_hit(hits, runtime_rule, view_text)
    return hits


def _match_runtime_rules_for_view_slow(view_text: str, rules: tuple[_RuntimeRule, ...]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for runtime_rule in rules:
        if runtime_rule.required_needles and any(needle not in view_text for needle in runtime_rule.required_needles):
            continue
        _append_runtime_rule_hit(hits, runtime_rule, view_text)
    return hits


def _append_runtime_rule_hit(hits: list[dict[str, str]], runtime_rule: _RuntimeRule, view_text: str) -> None:
    rule = runtime_rule.rule
    match_count = _match_count(rule, view_text)
    if rule.severity == "density_reject" and match_count < runtime_rule.density_threshold:
        return
    if match_count > 0:
        hits.append(
            {
                "rule_id": rule.id,
                "severity": rule.severity,
                "category": rule.category,
            }
        )


def _normalize_cached(
    text: str,
    cache: MutableMapping[tuple[Any, ...], Any] | None,
    key: tuple[Any, ...],
) -> NormalizedDocument:
    if cache is None:
        return normalize_text(text)
    cached = cache.get(key)
    if cached is None:
        cached = normalize_text(text)
        cache[key] = cached
    return cached


def _normalize_canonical_cached(
    text: str,
    cache: MutableMapping[tuple[Any, ...], Any] | None,
    key: tuple[Any, ...],
) -> CanonicalNormalization:
    key = (*key, "canonical_only")
    if cache is None:
        return normalize_canonical_text(text)
    cached = cache.get(key)
    if cached is None:
        cached = normalize_canonical_text(text)
        cache[key] = cached
    if not isinstance(cached, CanonicalNormalization):
        raise TypeError("deterministic normalization cache entry type mismatch")
    return cached


def _literal_needle(rule: RiskRule) -> str:
    return normalize_text(rule.value or "").view(rule.view)


def _required_needles(rule: RiskRule) -> tuple[str, ...]:
    if rule.type == "literal":
        return (_literal_needle(rule),)
    if rule.type != "phrase" or rule.view != "canonical":
        return ()
    normalized_value = normalize_canonical_text(rule.value or "").canonical_text
    return tuple(dict.fromkeys(part for part in normalized_value.split() if part))


def _match_count(rule: RiskRule, view_text: str) -> int:
    count = 0
    iterator = rule.compiled.finditer(view_text, timeout=MATCH_TIMEOUT_MS / 1000)
    for _match in iterator:
        count += 1
        if count >= max(1, int(rule.threshold)):
            break
    return count


def _unique_rule_ids(hits: list[dict[str, str]]) -> list[str]:
    return sorted({str(hit["rule_id"]) for hit in hits})


def _sorted_hits(hits: Any) -> list[dict[str, str]]:
    return sorted(
        list(hits),
        key=lambda hit: (str(hit["rule_id"]), str(hit["severity"]), str(hit["category"])),
    )


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a sha256 hex string")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)
