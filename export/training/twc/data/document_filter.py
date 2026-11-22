"""Reject-only deterministic filtering foundation for Phase 2A."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

import regex
import yaml

from twc.data.decision_schema import (
    FoundationDecision,
    FilterHit,
    SPAN_PROJECTION_VERSION,
)
from twc.repro.hashing import stable_json_hash
from twc.data.normalize import NormalizedDocument, normalize_text
from twc.data.risk_ontology import MATCH_TIMEOUT_MS, RiskOntology, RiskRule


WINDOW_CONFIG = {
    "window_words": 250,
    "window_overlap_words": 75,
    "max_windows_per_document": None,
    "very_long_doc_policy": "scan_all_windows",
}
FOUNDATION_CONFIG = {
    "phase_id": "phase2a_reject_only_foundation",
    "decision_capability": "reject_only_foundation",
    "required_gates": [
        "source_channel_gate",
        "deterministic_regex_gate",
        "density_gate",
        "safe_topic_gate",
        "local_classifier_ensemble",
        "teacher_audit_failure_gate",
        "post_tokenization_rescan",
    ],
    "emits_acceptance": False,
    "document_windowing": WINDOW_CONFIG,
}
FOUNDATION_CONFIG_HASH = stable_json_hash(FOUNDATION_CONFIG)

REJECTING_SEVERITIES = {
    "hard_reject",
    "density_reject",
    "phrase_reject",
    "source_reject",
}


@dataclass(frozen=True)
class TextWindow:
    index: int
    start_char: int
    end_char: int
    start_word: int
    end_word: int
    text: str


def evaluate_deterministic_foundation(
    raw_text: str,
    *,
    metadata: dict | None,
    ontology: RiskOntology,
) -> FoundationDecision:
    """Evaluate deterministic reject gates without any final acceptance path."""

    try:
        normalized = normalize_text(raw_text)
    except Exception as exc:
        return _error_decision(raw_text, ontology, "normalization", exc)

    presence = {
        "url": bool(metadata and metadata.get("url")),
        "title": bool(metadata and metadata.get("title")),
        "path": bool(metadata and metadata.get("path")),
        "domain": bool(metadata and metadata.get("domain")),
    }

    if not metadata or not metadata.get("url"):
        return _decision(
            "quarantined",
            normalized,
            ontology,
            source_metadata_presence=presence,
            quarantine_reason_code="missing_required_source_metadata",
        )

    source_view = _build_source_view(metadata)
    if source_view is None:
        return _decision(
            "quarantined",
            normalized,
            ontology,
            source_metadata_presence=presence,
            quarantine_reason_code="malformed_source_metadata",
        )

    hits: list[FilterHit] = []
    try:
        hits.extend(_match_source_rules(source_view, ontology))
        hits.extend(_match_text_rules(normalized, ontology))
    except TimeoutError:
        return _decision(
            "quarantined",
            normalized,
            ontology,
            source_metadata_presence=presence,
            quarantine_reason_code="regex_match_timeout",
        )
    except Exception as exc:
        return _error_decision(raw_text, ontology, "deterministic_matching", exc)

    status = (
        "rejected"
        if any(hit.severity in REJECTING_SEVERITIES for hit in hits)
        else "pending_required_gates"
    )
    return _decision(
        status,
        normalized,
        ontology,
        hits=tuple(hits),
        source_metadata_presence=presence,
    )


def _match_source_rules(source_view: str, ontology: RiskOntology) -> list[FilterHit]:
    hits: list[FilterHit] = []
    for rule in ontology.rules:
        if rule.view not in {"source_metadata", "url"} and rule.type != "url_regex":
            continue
        hits.extend(_match_rule(rule, source_view, "source_metadata", None))
    return hits


def _match_text_rules(
    normalized: NormalizedDocument,
    ontology: RiskOntology,
) -> list[FilterHit]:
    hits: list[FilterHit] = []
    text_rules = [
        rule
        for rule in ontology.rules
        if rule.view not in {"source_metadata", "url"} and rule.type != "url_regex"
    ]
    literal_needles = {
        rule.id: _literal_needle(rule)
        for rule in text_rules
        if rule.type == "literal"
    }

    for rule in text_rules:
        view_text = normalized.view(rule.view)
        if rule.type == "literal" and not _literal_possible(literal_needles[rule.id], view_text):
            continue
        hits.extend(
            _match_rule(
                rule,
                view_text,
                "text",
                normalized,
                scan_scope="whole_document",
            )
        )

    windows = _document_windows(normalized.canonical_text)
    window_views_needed = {
        rule.view
        for rule in text_rules
        if rule.view in {"canonical", "ascii_folded", "depunct_compact", "escape_decoded"}
    }
    for window in windows:
        window_views = _window_views(window.text, window_views_needed)
        for rule in text_rules:
            view_text = window_views[rule.view]
            if rule.type == "literal" and not _literal_possible(literal_needles[rule.id], view_text):
                continue
            span_offset = window.start_char if rule.view == "canonical" else 0
            hits.extend(
                _match_rule(
                    rule,
                    view_text,
                    "text",
                    normalized,
                    scan_scope="window",
                    window=window,
                    span_offset=span_offset,
                )
            )
    return hits


def _literal_needle(rule: RiskRule) -> str:
    return normalize_text(rule.value or "").view(rule.view)


def _literal_possible(needle: str, text: str) -> bool:
    return needle in text


def _match_rule(
    rule: RiskRule,
    text: str,
    source_field: str,
    normalized: NormalizedDocument | None,
    *,
    scan_scope: str = "whole_document",
    window: TextWindow | None = None,
    span_offset: int = 0,
) -> list[FilterHit]:
    matches = []
    try:
        iterator = rule.compiled.finditer(text, timeout=MATCH_TIMEOUT_MS / 1000)
        for match in iterator:
            matches.append(match)
            if len(matches) >= max(1, rule.threshold):
                break
    except TimeoutError:
        raise

    if rule.severity == "density_reject" and len(matches) < rule.threshold:
        return []

    hits: list[FilterHit] = []
    for match in matches:
        span = (match.start() + span_offset, match.end() + span_offset)
        raw_span = None
        raw_context = ""
        if normalized is not None and rule.view == "canonical":
            raw_span, raw_context = normalized.project_canonical_span(*span)
        elif normalized is not None:
            raw_context = "raw_context_unavailable_for_noncanonical_view"

        hits.append(
            FilterHit(
                category=rule.category,
                rule_id=rule.id,
                severity=rule.severity,
                reason=rule.reason,
                source_field=source_field if source_field != "text" else "text",
                match_view=rule.view,
                matched_text=_stable_context(text, match.start(), match.end(), 48),
                span_normalized=span,
                raw_span=raw_span,
                raw_context_snippet=raw_context,
                span_projection_version=SPAN_PROJECTION_VERSION,
                pattern_type=rule.type,
                hit_weight=rule.weight,
                scan_scope=scan_scope,
                window_index=window.index if window else None,
                window_start_normalized=window.start_char if window else None,
                window_end_normalized=window.end_char if window else None,
                window_word_start=window.start_word if window else None,
                window_word_end=window.end_word if window else None,
            )
        )
    return hits


def _document_windows(text: str) -> tuple[TextWindow, ...]:
    word_matches = tuple(re.finditer(r"\S+", text))
    if not word_matches:
        return (
            TextWindow(
                index=0,
                start_char=0,
                end_char=0,
                start_word=0,
                end_word=0,
                text="",
            ),
        )

    window_words = int(WINDOW_CONFIG["window_words"])
    overlap_words = int(WINDOW_CONFIG["window_overlap_words"])
    if window_words <= 0 or overlap_words < 0 or overlap_words >= window_words:
        raise ValueError("invalid document window configuration")

    step = window_words - overlap_words
    windows: list[TextWindow] = []
    start_word = 0
    while start_word < len(word_matches):
        end_word = min(len(word_matches), start_word + window_words)
        start_char = word_matches[start_word].start()
        end_char = word_matches[end_word - 1].end()
        windows.append(
            TextWindow(
                index=len(windows),
                start_char=start_char,
                end_char=end_char,
                start_word=start_word,
                end_word=end_word,
                text=text[start_char:end_char],
            )
        )
        if end_word == len(word_matches):
            break
        start_word += step

    max_windows = WINDOW_CONFIG["max_windows_per_document"]
    if max_windows is not None and len(windows) > int(max_windows):
        raise ValueError("document exceeds configured window scan cap")
    return tuple(windows)


def _window_views(canonical_text: str, views: set[str]) -> dict[str, str]:
    window_views: dict[str, str] = {}
    if "canonical" in views:
        window_views["canonical"] = canonical_text
    derived_views = views - {"canonical"}
    if derived_views:
        normalized = normalize_text(canonical_text)
        for view in derived_views:
            window_views[view] = normalized.view(view)
    return window_views


def _stable_context(text: str, start: int, end: int, context_chars: int = 80) -> str:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return prefix + text[left:right] + suffix


def _build_source_view(metadata: dict) -> str | None:
    try:
        url = str(metadata.get("url", ""))
        title = str(metadata.get("title", ""))
        path = str(metadata.get("path", ""))
        normalized_url = _normalize_url(url)
        if normalized_url is None:
            return None
        parts = [normalized_url, title, path, str(metadata.get("domain", ""))]
        return normalize_text("\n".join(parts)).canonical_text
    except Exception:
        return None


def _normalize_url(url: str) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    candidate = url.strip().replace("\\", "/")
    split = urlsplit(candidate)
    if not split.scheme or not split.netloc:
        return None
    try:
        host = split.hostname
        if not host:
            return None
        if split.port is not None:
            port = f":{split.port}"
        else:
            port = ""
        ascii_host = host.rstrip(".").encode("idna").decode("ascii").casefold()
    except Exception:
        return None

    path = _bounded_unquote(split.path or "")
    query = _bounded_unquote(split.query or "")
    normalized = f"{split.scheme.casefold()}://{ascii_host}{port}{path}"
    if query:
        normalized += f"?{query}"
    return normalized


def _bounded_unquote(value: str, passes: int = 3) -> str:
    current = value
    for _ in range(passes):
        decoded = unquote(current)
        if decoded == current:
            break
        current = decoded
    return current


def _decision(
    status: str,
    normalized: NormalizedDocument,
    ontology: RiskOntology,
    *,
    hits: tuple[FilterHit, ...] = (),
    source_metadata_presence: dict[str, bool] | None = None,
    quarantine_reason_code: str | None = None,
) -> FoundationDecision:
    return FoundationDecision(
        decision_status=status,
        raw_text_sha256=normalized.raw_sha256,
        normalized_text_sha256=normalized.canonical_sha256,
        view_sha256s=normalized.view_sha256s,
        ontology_semantic_hash=ontology.semantic_hash,
        ontology_source_tree_hash=ontology.source_tree_hash,
        config_hash=FOUNDATION_CONFIG_HASH,
        hits=hits,
        normalizer_version=normalized.normalization_version,
        normalization_steps_applied=normalized.steps_applied,
        regex_engine=ontology.regex_engine,
        regex_engine_version=ontology.regex_engine_version,
        yaml_parser_version=ontology.yaml_parser_version,
        python_unicode_version=ontology.python_unicode_version,
        match_timeout_ms=ontology.match_timeout_ms,
        source_metadata_presence=source_metadata_presence or {},
        quarantine_reason_code=quarantine_reason_code,
    )


def _error_decision(raw_text: str, ontology: RiskOntology, stage: str, exc: Exception) -> FoundationDecision:
    try:
        normalized = normalize_text(raw_text if isinstance(raw_text, str) else "")
    except Exception:
        normalized = normalize_text("")
    return FoundationDecision(
        decision_status="error",
        raw_text_sha256=normalized.raw_sha256,
        normalized_text_sha256=normalized.canonical_sha256,
        view_sha256s=normalized.view_sha256s,
        ontology_semantic_hash=ontology.semantic_hash,
        ontology_source_tree_hash=ontology.source_tree_hash,
        config_hash=FOUNDATION_CONFIG_HASH,
        normalizer_version=normalized.normalization_version,
        normalization_steps_applied=normalized.steps_applied,
        regex_engine=ontology.regex_engine,
        regex_engine_version=ontology.regex_engine_version,
        yaml_parser_version=ontology.yaml_parser_version,
        python_unicode_version=ontology.python_unicode_version,
        match_timeout_ms=ontology.match_timeout_ms,
        error_class=type(exc).__name__,
        error_stage=stage,
        error_message_hash=hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
    )
