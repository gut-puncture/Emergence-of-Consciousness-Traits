"""Deterministic natural-block segmentation for chunked clean-run data.

This module is the local, no-raw-public foundation for ``chunked_clean_run_v1``.
It segments private in-memory text into raw-span-backed natural blocks and
builds public summaries that cannot be mistaken for tokenizer, packing, or
training readiness.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from typing import Any

from twc.data.decision_schema import stable_json_hash
from twc.data.normalize import NORMALIZER_VERSION, normalize_canonical_text, normalize_text
from twc.pretraining.safety import validate_no_raw_public_payload


CHUNKED_CLEAN_RUN_SCHEMA_VERSION = 1
SEGMENTATION_POLICY_VERSION = "chunked_clean_run_segmenter_v1"
RAW_LINE_POLICY_VERSION = "raw_line_keepends_v1"
PROXY_TOKEN_ESTIMATOR_VERSION = "word_char_proxy_v1"
BLOCK_ID_POLICY_VERSION = "source_policy_index_span_hash_v1"
PUBLIC_REPORT_SCHEMA_VERSION = 1
PUBLIC_REPORT_ARTIFACT_KIND = "chunked_clean_run_segmentation_report"
PUBLIC_REPORT_STATUS = "schema_segmentation_validated_only"
BLOCKED_PUBLIC_REPORT_STATUS = "blocked_review_required"

ALLOWED_SPLITS = {"train", "validation"}
BLOCK_KINDS = {
    "heading",
    "prose",
    "list",
    "table",
    "fenced_code",
    "indented_code",
    "html_pre",
    "html_block",
    "log_run",
    "stack_trace",
    "ui_chrome",
    "oversized_soft_wrapped_prose",
    "oversized_unsegmentable",
}
TERMINATORS = {"\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\u0085", "\u2028", "\u2029"}

_SHA256_LENGTH = 64
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)")
_SETEXT_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_LIST_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$")
_LOG_RE = re.compile(
    r"^\s*(?:\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}|"
    r"\[[A-Z][A-Z0-9_-]{1,12}\]|(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL)\b)"
)
_STACK_TRACE_RE = re.compile(r"^\s*(?:Traceback\b|File \"|at\s+\S+|Caused by:)")
_HTML_BLOCK_RE = re.compile(r"^\s*</?[A-Za-z][A-Za-z0-9:-]*(?:\s|>|/>)")
_SAFE_WRAP_RE = re.compile(r"[\s.!?,;:\u3002\uff0c\uff01\uff1f\uff1b\uff1a]+")
_SAFE_PUBLIC_KEY_RE = re.compile(r"^[a-z0-9_:-]+$")

@dataclass(frozen=True)
class ChunkedSegmentationConfig:
    max_single_line_chars: int = 8_192
    soft_wrap_target_chars: int = 4_096
    max_log_run_lines: int = 256
    max_ui_chrome_lines: int = 128


@dataclass(frozen=True)
class SourceDocumentRecord:
    source_document_id_hash: str
    split: str
    raw_text: object
    source_lineage_hash: str | None = None
    source_token_count_hint: int | None = None


@dataclass(frozen=True)
class RawLineRecord:
    line_index: int
    char_start: int
    char_end: int
    content_end: int
    terminator: str
    line_kind: str


@dataclass(frozen=True)
class NaturalBlockRecord:
    block_id_hash: str
    source_document_id_hash: str
    segmentation_policy_hash: str
    block_index: int
    block_kind: str
    content_char_start: int
    content_char_end: int
    owned_char_start: int
    owned_char_end: int
    start_line_index: int
    end_line_index_exclusive: int
    raw_text_sha256: str
    normalized_text_sha256: str
    proxy_token_count: int
    warning_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class TriviaSpanRecord:
    source_document_id_hash: str
    char_start: int
    char_end: int
    start_line_index: int
    end_line_index_exclusive: int
    trivia_kind: str
    left_block_index: int | None
    right_block_index: int | None


@dataclass(frozen=True)
class SegmentationReceipt:
    source_document_id_hash: str
    split: str
    source_lineage_hash: str | None
    source_token_count_hint: int | None
    document_status: str
    raw_text_sha256: str | None
    normalized_text_sha256: str | None
    raw_char_count: int
    raw_line_count: int
    block_count: int
    trivia_span_count: int
    total_proxy_token_count: int
    block_kind_counts: dict[str, int]
    warning_code_counts: dict[str, int]
    segmentation_policy_version: str
    segmentation_policy_hash: str
    raw_line_policy_hash: str
    normalizer_version: str
    proxy_token_estimator_version: str
    block_id_policy_hash: str
    raw_coverage_validation: bool
    reconstruction_validation: bool
    block_order_validation: bool
    raw_text_serialized: bool
    raw_url_serialized: bool
    token_serialized: bool
    receipt_hash: str


@dataclass(frozen=True)
class SegmentationResult:
    source: SourceDocumentRecord
    lines: tuple[RawLineRecord, ...]
    blocks: tuple[NaturalBlockRecord, ...]
    trivia_spans: tuple[TriviaSpanRecord, ...]
    receipt: SegmentationReceipt


@lru_cache(maxsize=32)
def segmentation_policy_hash(config: ChunkedSegmentationConfig | None = None) -> str:
    config = config or ChunkedSegmentationConfig()
    return stable_json_hash(
        {
            "schema_version": CHUNKED_CLEAN_RUN_SCHEMA_VERSION,
            "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
            "raw_line_policy_version": RAW_LINE_POLICY_VERSION,
            "max_single_line_chars": config.max_single_line_chars,
            "soft_wrap_target_chars": config.soft_wrap_target_chars,
            "max_log_run_lines": config.max_log_run_lines,
            "max_ui_chrome_lines": config.max_ui_chrome_lines,
        }
    )


@lru_cache(maxsize=1)
def raw_line_policy_hash() -> str:
    return stable_json_hash(
        {
            "raw_line_policy_version": RAW_LINE_POLICY_VERSION,
            "terminators": sorted(TERMINATORS | {"\r\n"}),
            "keepends": True,
        }
    )


@lru_cache(maxsize=1)
def block_id_policy_hash() -> str:
    return stable_json_hash({"block_id_policy_version": BLOCK_ID_POLICY_VERSION})


def public_report_schema_hash() -> str:
    return stable_json_hash(
        {
            "public_report_schema_version": PUBLIC_REPORT_SCHEMA_VERSION,
            "artifact_kind": PUBLIC_REPORT_ARTIFACT_KIND,
            "status": PUBLIC_REPORT_STATUS,
        }
    )


def build_raw_lines(raw: str) -> tuple[RawLineRecord, ...]:
    if not isinstance(raw, str):
        raise TypeError("build_raw_lines requires a str input")

    lines: list[RawLineRecord] = []
    start = 0
    index = 0
    while start < len(raw):
        cursor = start
        terminator = ""
        while cursor < len(raw):
            char = raw[cursor]
            if char == "\r" and cursor + 1 < len(raw) and raw[cursor + 1] == "\n":
                terminator = "\r\n"
                content_end = cursor
                cursor += 2
                break
            if char in TERMINATORS:
                terminator = char
                content_end = cursor
                cursor += 1
                break
            cursor += 1
        else:
            content_end = cursor

        content = raw[start:content_end]
        lines.append(
            RawLineRecord(
                line_index=index,
                char_start=start,
                char_end=cursor,
                content_end=content_end,
                terminator=terminator,
                line_kind=_classify_line_kind(content),
            )
        )
        index += 1
        start = cursor
    return tuple(lines)


def segment_source_document(
    source: SourceDocumentRecord,
    *,
    config: ChunkedSegmentationConfig | None = None,
) -> SegmentationResult:
    return _segment_source_document(source, config=config, validate_result=True)


def _segment_source_document_prefilter_fast(
    source: SourceDocumentRecord,
    *,
    config: ChunkedSegmentationConfig | None = None,
) -> SegmentationResult:
    return _segment_source_document(source, config=config, validate_result=False)


def _segment_source_document(
    source: SourceDocumentRecord,
    *,
    config: ChunkedSegmentationConfig | None = None,
    validate_result: bool = True,
) -> SegmentationResult:
    config = _validate_config(config or ChunkedSegmentationConfig())
    _require_sha256(source.source_document_id_hash, "source_document_id_hash")
    if source.split not in ALLOWED_SPLITS:
        raise ValueError("source document split must be train or validation")
    if source.source_lineage_hash is not None:
        _require_sha256(source.source_lineage_hash, "source_lineage_hash")
    if source.source_token_count_hint is not None:
        _require_nonnegative_int(source.source_token_count_hint, "source_token_count_hint")

    if source.raw_text is None:
        return _terminal_result(source, "missing_text", config, validate_result=validate_result)
    if not isinstance(source.raw_text, str):
        return _terminal_result(source, "non_string_text", config, validate_result=validate_result)
    if _is_raw_blank(source.raw_text):
        return _terminal_result(source, "empty_or_whitespace", config, validate_result=validate_result)

    raw = source.raw_text
    fast_parts = _segment_single_line_common_prose_fast(source, raw, config)
    if fast_parts is None:
        lines = build_raw_lines(raw)
        blocks, trivia_spans = _segment_lines(source, lines, config)
    else:
        lines, blocks, trivia_spans = fast_parts
    block_tuple = tuple(blocks)
    trivia_tuple = tuple(trivia_spans)
    result = SegmentationResult(
        source=source,
        lines=lines,
        blocks=block_tuple,
        trivia_spans=trivia_tuple,
        receipt=_build_receipt(
            source,
            lines,
            block_tuple,
            trivia_tuple,
            "segmented",
            config,
            normalized_text_sha256=_single_block_receipt_normalized_sha256(raw, block_tuple),
        ),
    )
    if validate_result:
        validate_segmentation_result(result)
    return result


def _segment_single_line_common_prose_fast(
    source: SourceDocumentRecord,
    raw: str,
    config: ChunkedSegmentationConfig,
) -> tuple[tuple[RawLineRecord, ...], list[NaturalBlockRecord], list[TriviaSpanRecord]] | None:
    if any(terminator in raw for terminator in TERMINATORS):
        return None
    if len(raw) > config.max_single_line_chars:
        return None
    line_kind = _classify_line_kind(raw)
    if line_kind != "prose":
        return None
    lines = (
        RawLineRecord(
            line_index=0,
            char_start=0,
            char_end=len(raw),
            content_end=len(raw),
            terminator="",
            line_kind=line_kind,
        ),
    )
    return (
        lines,
        [
            _make_line_block(
                raw,
                source.source_document_id_hash,
                0,
                "prose",
                lines,
                0,
                1,
                (),
                segmentation_policy_hash(config),
            )
        ],
        [],
    )


def _single_block_receipt_normalized_sha256(raw: str, blocks: tuple[NaturalBlockRecord, ...]) -> str | None:
    if len(blocks) != 1:
        return None
    block = blocks[0]
    if raw[: block.content_char_start] and not _is_raw_blank(raw[: block.content_char_start]):
        return None
    if raw[block.content_char_end :] and not _is_raw_blank(raw[block.content_char_end :]):
        return None
    return block.normalized_text_sha256


def validate_segmentation_result(result: SegmentationResult) -> None:
    source = result.source
    _require_sha256(source.source_document_id_hash, "source_document_id_hash")
    if source.split not in ALLOWED_SPLITS:
        raise ValueError("segmentation result source split is unsupported")
    raw = source.raw_text if isinstance(source.raw_text, str) else None
    terminal = result.receipt.document_status in {"missing_text", "non_string_text", "empty_or_whitespace"}

    if terminal:
        _validate_receipt_identity(result)
        _validate_terminal_receipt_matches_source(result)
        if result.blocks or result.trivia_spans or result.lines:
            raise ValueError("terminal segmentation documents must not have blocks, trivia, or lines")
        if result.receipt.block_count != 0 or result.receipt.trivia_span_count != 0:
            raise ValueError("terminal segmentation receipt count mismatch")
        _validate_receipt_hash(result.receipt)
        return

    if raw is None:
        raise ValueError("non-terminal segmentation requires raw text")
    if result.receipt.document_status != "segmented":
        raise ValueError("non-terminal segmentation must use segmented status")
    _validate_receipt_identity(result)
    if tuple(build_raw_lines(raw)) != result.lines:
        raise ValueError("segmentation raw line records do not match source text")
    if [block.block_index for block in result.blocks] != list(range(len(result.blocks))):
        raise ValueError("natural block indexes must be contiguous")

    coverage: list[tuple[int, int, str]] = []
    for block in result.blocks:
        if block.source_document_id_hash != source.source_document_id_hash:
            raise ValueError("natural block source id does not match parent source")
        _validate_block_record(block, raw)
        if block.end_line_index_exclusive > len(result.lines):
            raise ValueError("natural block line range is outside raw lines")
        _validate_line_range_for_span(
            result.lines,
            block.start_line_index,
            block.end_line_index_exclusive,
            block.owned_char_start,
            block.owned_char_end,
            "natural block",
            require_exact_boundaries=block.block_kind != "oversized_soft_wrapped_prose",
        )
        coverage.append((block.owned_char_start, block.owned_char_end, f"block:{block.block_index}"))
    for trivia in result.trivia_spans:
        if trivia.source_document_id_hash != source.source_document_id_hash:
            raise ValueError("trivia source id does not match parent source")
        _validate_trivia_record(trivia, raw)
        if trivia.end_line_index_exclusive > len(result.lines):
            raise ValueError("trivia line range is outside raw lines")
        _validate_line_range_for_span(
            result.lines,
            trivia.start_line_index,
            trivia.end_line_index_exclusive,
            trivia.char_start,
            trivia.char_end,
            "trivia",
            require_exact_boundaries=True,
        )
        coverage.append((trivia.char_start, trivia.char_end, "trivia"))
    _validate_coverage(raw, coverage)
    _validate_trivia_adjacency(result.blocks, result.trivia_spans, raw_end=len(raw))
    _validate_receipt_matches_result(result)
    _validate_receipt_hash(result.receipt)


def build_public_chunked_segmentation_report(
    *,
    document_count: int,
    status_counts: dict[str, int],
    split_counts: dict[str, int],
    total_block_count: int,
    total_trivia_span_count: int,
    block_kind_counts: dict[str, int],
    warning_code_counts: dict[str, int],
    proxy_token_histogram: dict[str, int],
    artifact_status: str = PUBLIC_REPORT_STATUS,
    source_manifest_hash: str | None = None,
    source_schedule_hash: str | None = None,
) -> dict[str, Any]:
    if artifact_status not in {PUBLIC_REPORT_STATUS, BLOCKED_PUBLIC_REPORT_STATUS}:
        raise ValueError("unsupported chunked segmentation report status")
    for optional_hash, field in (
        (source_manifest_hash, "source_manifest_hash"),
        (source_schedule_hash, "source_schedule_hash"),
    ):
        if optional_hash is not None:
            _require_sha256(optional_hash, field)
    if artifact_status == PUBLIC_REPORT_STATUS:
        artifact_status = BLOCKED_PUBLIC_REPORT_STATUS

    payload: dict[str, Any] = {
        "artifact_kind": PUBLIC_REPORT_ARTIFACT_KIND,
        "artifact_status": artifact_status,
        "schema_version": PUBLIC_REPORT_SCHEMA_VERSION,
        "chunked_clean_run_schema_version": CHUNKED_CLEAN_RUN_SCHEMA_VERSION,
        "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
        "segmentation_policy_hash": segmentation_policy_hash(),
        "raw_line_policy_hash": raw_line_policy_hash(),
        "normalizer_version": NORMALIZER_VERSION,
        "proxy_token_estimator_version": PROXY_TOKEN_ESTIMATOR_VERSION,
        "block_id_policy_hash": block_id_policy_hash(),
        "public_report_schema_hash": public_report_schema_hash(),
        "source_manifest_hash": source_manifest_hash,
        "source_schedule_hash": source_schedule_hash,
        "sqlite_aggregate_hash": None,
        "document_count": _require_nonnegative_int(document_count, "document_count"),
        "status_counts": _sorted_count_dict(status_counts, "status_counts"),
        "split_counts": _sorted_count_dict(split_counts, "split_counts"),
        "total_block_count": _require_nonnegative_int(total_block_count, "total_block_count"),
        "total_trivia_span_count": _require_nonnegative_int(
            total_trivia_span_count,
            "total_trivia_span_count",
        ),
        "block_kind_counts": _sorted_count_dict(block_kind_counts, "block_kind_counts"),
        "warning_code_counts": _sorted_count_dict(warning_code_counts, "warning_code_counts"),
        "proxy_token_histogram": _sorted_count_dict(proxy_token_histogram, "proxy_token_histogram"),
        "sqlite_source_of_truth": False,
        "raw_coverage_validation": True,
        "reconstruction_validation": True,
        "block_order_validation": True,
        "public_allowlist_validation": True,
        "future_lane_tables_empty": True,
        "accepted_manifest_created": False,
        "clean_run_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "ready_for_tokenizer_and_packer": False,
        "ready_for_training": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["report_hash"] = _report_hash(payload)
    validate_public_chunked_segmentation_report(payload)
    return payload


def validate_public_chunked_segmentation_report(payload: dict[str, Any]) -> None:
    _validate_public_chunked_segmentation_report_payload(payload, allow_validated_report=False)


def _validate_public_chunked_segmentation_report_payload(
    payload: dict[str, Any],
    *,
    allow_validated_report: bool,
) -> None:
    validate_no_raw_public_payload(payload)
    if not isinstance(payload, dict):
        raise ValueError("chunked segmentation report must be a mapping")
    extra = set(payload) - _PUBLIC_REPORT_ALLOWED_KEYS
    if extra:
        raise ValueError(f"unsupported chunked segmentation public fields: {sorted(extra)}")
    missing = _PUBLIC_REPORT_ALLOWED_KEYS - set(payload)
    if missing:
        raise ValueError(f"missing chunked segmentation public fields: {sorted(missing)}")
    if payload.get("artifact_kind") != PUBLIC_REPORT_ARTIFACT_KIND:
        raise ValueError("unsupported chunked segmentation report kind")
    artifact_status = payload.get("artifact_status")
    if artifact_status not in {PUBLIC_REPORT_STATUS, BLOCKED_PUBLIC_REPORT_STATUS}:
        raise ValueError("unsupported chunked segmentation report status")
    if payload.get("schema_version") != PUBLIC_REPORT_SCHEMA_VERSION:
        raise ValueError("chunked segmentation report schema_version mismatch")
    if payload.get("chunked_clean_run_schema_version") != CHUNKED_CLEAN_RUN_SCHEMA_VERSION:
        raise ValueError("chunked segmentation report chunked schema version mismatch")
    if payload.get("segmentation_policy_version") != SEGMENTATION_POLICY_VERSION:
        raise ValueError("chunked segmentation report policy version mismatch")
    if payload.get("normalizer_version") != NORMALIZER_VERSION:
        raise ValueError("chunked segmentation report normalizer version mismatch")
    if payload.get("proxy_token_estimator_version") != PROXY_TOKEN_ESTIMATOR_VERSION:
        raise ValueError("chunked segmentation report proxy estimator version mismatch")
    for field in _PUBLIC_REPORT_INT_FIELDS:
        _require_nonnegative_int(payload.get(field), field)
    count_dicts = {field: _sorted_count_dict(payload.get(field), field) for field in _PUBLIC_REPORT_COUNT_DICT_FIELDS}
    document_count = int(payload["document_count"])
    total_block_count = int(payload["total_block_count"])
    if set(count_dicts["status_counts"]) - _PUBLIC_REPORT_ALLOWED_STATUSES:
        raise ValueError("chunked segmentation report contains unsupported document status")
    if set(count_dicts["split_counts"]) - ALLOWED_SPLITS:
        raise ValueError("chunked segmentation report contains unsupported split")
    if set(count_dicts["block_kind_counts"]) - BLOCK_KINDS:
        raise ValueError("chunked segmentation report contains unsupported block kind")
    if set(count_dicts["proxy_token_histogram"]) - _PUBLIC_REPORT_PROXY_BUCKETS:
        raise ValueError("chunked segmentation report contains unsupported proxy bucket")
    if set(count_dicts["warning_code_counts"]) - _PUBLIC_REPORT_WARNING_CODES:
        raise ValueError("chunked segmentation report contains unsupported warning code")
    if sum(count_dicts["status_counts"].values()) != document_count:
        raise ValueError("chunked segmentation report status counts do not match document_count")
    if sum(count_dicts["split_counts"].values()) != document_count:
        raise ValueError("chunked segmentation report split counts do not match document_count")
    if sum(count_dicts["block_kind_counts"].values()) != total_block_count:
        raise ValueError("chunked segmentation report block kind counts do not match total_block_count")
    if sum(count_dicts["proxy_token_histogram"].values()) != total_block_count:
        raise ValueError("chunked segmentation report proxy histogram does not match total_block_count")
    for field in _PUBLIC_REPORT_TRUE_FIELDS:
        if payload.get(field) is not True:
            raise ValueError(f"chunked segmentation report must prove {field}=true")
    for field in _PUBLIC_REPORT_FALSE_FIELDS:
        if payload.get(field) is not False:
            raise ValueError(f"chunked segmentation report must keep {field}=false")
    for field in _PUBLIC_REPORT_HASH_FIELDS:
        value = payload.get(field)
        if value is not None:
            _require_sha256(value, field)
    if payload["segmentation_policy_hash"] != segmentation_policy_hash():
        raise ValueError("chunked segmentation report segmentation_policy_hash mismatch")
    if payload["raw_line_policy_hash"] != raw_line_policy_hash():
        raise ValueError("chunked segmentation report raw_line_policy_hash mismatch")
    if payload["block_id_policy_hash"] != block_id_policy_hash():
        raise ValueError("chunked segmentation report block_id_policy_hash mismatch")
    if payload["public_report_schema_hash"] != public_report_schema_hash():
        raise ValueError("chunked segmentation report public_report_schema_hash mismatch")
    if artifact_status == PUBLIC_REPORT_STATUS:
        if not allow_validated_report:
            raise ValueError("validated chunked segmentation report requires SQLite state validation")
        if payload.get("sqlite_source_of_truth") is not True:
            raise ValueError("validated chunked segmentation report must be SQLite-derived")
        if payload.get("sqlite_aggregate_hash") is None:
            raise ValueError("validated chunked segmentation report requires sqlite_aggregate_hash")
    if artifact_status == BLOCKED_PUBLIC_REPORT_STATUS:
        if payload.get("sqlite_source_of_truth") is True:
            raise ValueError("blocked chunked segmentation report cannot claim SQLite source of truth")
        if payload.get("sqlite_aggregate_hash") is not None:
            raise ValueError("blocked chunked segmentation report cannot carry sqlite_aggregate_hash")
    if payload.get("report_hash") != _report_hash(payload):
        raise ValueError("chunked segmentation report hash mismatch")


def _validate_config(config: ChunkedSegmentationConfig) -> ChunkedSegmentationConfig:
    _require_positive_int(config.max_single_line_chars, "max_single_line_chars", minimum=16)
    _require_positive_int(config.soft_wrap_target_chars, "soft_wrap_target_chars", minimum=8)
    _require_positive_int(config.max_log_run_lines, "max_log_run_lines")
    _require_positive_int(config.max_ui_chrome_lines, "max_ui_chrome_lines")
    if config.soft_wrap_target_chars > config.max_single_line_chars:
        raise ValueError("soft_wrap_target_chars cannot exceed max_single_line_chars")
    return config


def _terminal_result(
    source: SourceDocumentRecord,
    status: str,
    config: ChunkedSegmentationConfig,
    *,
    validate_result: bool = True,
) -> SegmentationResult:
    raw_hash = _sha256_text(source.raw_text) if isinstance(source.raw_text, str) else None
    normalized_hash = normalize_text(source.raw_text).canonical_sha256 if isinstance(source.raw_text, str) else None
    raw_char_count = len(source.raw_text) if isinstance(source.raw_text, str) else 0
    receipt = _build_receipt(
        source,
        (),
        (),
        (),
        status,
        config,
        raw_text_sha256=raw_hash,
        normalized_text_sha256=normalized_hash,
        raw_char_count=raw_char_count,
    )
    result = SegmentationResult(source=source, lines=(), blocks=(), trivia_spans=(), receipt=receipt)
    if validate_result:
        validate_segmentation_result(result)
    return result


def _segment_lines(
    source: SourceDocumentRecord,
    lines: tuple[RawLineRecord, ...],
    config: ChunkedSegmentationConfig,
) -> tuple[list[NaturalBlockRecord], list[TriviaSpanRecord]]:
    common_prose = _segment_common_prose_lines_fast(source, lines, config)
    if common_prose is not None:
        return common_prose

    raw = source.raw_text
    assert isinstance(raw, str)
    blocks: list[NaturalBlockRecord] = []
    trivia_spans: list[TriviaSpanRecord] = []
    pending_trivia: list[tuple[int, int, int, int, str, int | None]] = []
    last_block_index: int | None = None
    index = 0

    def flush_pending(right_block_index: int | None) -> None:
        nonlocal pending_trivia
        for char_start, char_end, start_line, end_line, kind, left_block_index in pending_trivia:
            trivia_spans.append(
                TriviaSpanRecord(
                    source_document_id_hash=source.source_document_id_hash,
                    char_start=char_start,
                    char_end=char_end,
                    start_line_index=start_line,
                    end_line_index_exclusive=end_line,
                    trivia_kind=kind,
                    left_block_index=left_block_index,
                    right_block_index=right_block_index,
                )
            )
        pending_trivia = []

    while index < len(lines):
        line = lines[index]
        if line.line_kind == "blank":
            start_index = index
            while index < len(lines) and lines[index].line_kind == "blank":
                index += 1
            pending_trivia.append(
                (
                    lines[start_index].char_start,
                    lines[index - 1].char_end,
                    start_index,
                    index,
                    "blank_separator",
                    last_block_index,
                )
            )
            continue

        next_block_index = len(blocks)
        flush_pending(next_block_index)
        new_blocks, index = _parse_next_block(
            raw,
            source_document_id_hash=source.source_document_id_hash,
            lines=lines,
            start_index=index,
            first_block_index=next_block_index,
            config=config,
            policy_hash=segmentation_policy_hash(config),
        )
        blocks.extend(new_blocks)
        last_block_index = blocks[-1].block_index

    flush_pending(None)
    return blocks, trivia_spans


def _segment_common_prose_lines_fast(
    source: SourceDocumentRecord,
    lines: tuple[RawLineRecord, ...],
    config: ChunkedSegmentationConfig,
) -> tuple[list[NaturalBlockRecord], list[TriviaSpanRecord]] | None:
    raw = source.raw_text
    assert isinstance(raw, str)
    for line in lines:
        if line.line_kind not in {"blank", "prose"}:
            return None
        if line.line_kind == "prose" and line.content_end - line.char_start > config.max_single_line_chars:
            return None

    blocks: list[NaturalBlockRecord] = []
    trivia_spans: list[TriviaSpanRecord] = []
    pending_trivia: list[tuple[int, int, int, int, str, int | None]] = []
    last_block_index: int | None = None
    index = 0
    policy_hash = segmentation_policy_hash(config)

    def flush_pending(right_block_index: int | None) -> None:
        nonlocal pending_trivia
        for char_start, char_end, start_line, end_line, kind, left_block_index in pending_trivia:
            trivia_spans.append(
                TriviaSpanRecord(
                    source_document_id_hash=source.source_document_id_hash,
                    char_start=char_start,
                    char_end=char_end,
                    start_line_index=start_line,
                    end_line_index_exclusive=end_line,
                    trivia_kind=kind,
                    left_block_index=left_block_index,
                    right_block_index=right_block_index,
                )
            )
        pending_trivia = []

    while index < len(lines):
        line = lines[index]
        if line.line_kind == "blank":
            start_index = index
            while index < len(lines) and lines[index].line_kind == "blank":
                index += 1
            pending_trivia.append(
                (
                    lines[start_index].char_start,
                    lines[index - 1].char_end,
                    start_index,
                    index,
                    "blank_separator",
                    last_block_index,
                )
            )
            continue

        start_index = index
        while index < len(lines) and lines[index].line_kind == "prose":
            index += 1
        next_block_index = len(blocks)
        flush_pending(next_block_index)
        block = _make_line_block(
            raw,
            source.source_document_id_hash,
            next_block_index,
            "prose",
            lines,
            start_index,
            index,
            (),
            policy_hash,
        )
        blocks.append(block)
        last_block_index = block.block_index

    flush_pending(None)
    return blocks, trivia_spans


def _parse_next_block(
    raw: str,
    *,
    source_document_id_hash: str,
    lines: tuple[RawLineRecord, ...],
    start_index: int,
    first_block_index: int,
    config: ChunkedSegmentationConfig,
    policy_hash: str,
) -> tuple[list[NaturalBlockRecord], int]:
    line = lines[start_index]
    content = raw[line.char_start : line.content_end]
    stripped = content.strip()

    if line.content_end - line.char_start > config.max_single_line_chars:
        return (
            _oversized_line_blocks(
                raw,
                source_document_id_hash=source_document_id_hash,
                line=line,
                first_block_index=first_block_index,
                config=config,
                policy_hash=policy_hash,
            ),
            start_index + 1,
        )
    fence = _fence_marker(stripped)
    if fence:
        end_index, closed = _find_fence_end(raw, lines, start_index, fence)
        warnings = () if closed else ("unclosed_fenced_code",)
        return (
            [
                _make_line_block(
                    raw,
                    source_document_id_hash,
                    first_block_index,
                    "fenced_code",
                    lines,
                    start_index,
                    end_index,
                    warnings,
                    policy_hash,
                )
            ],
            end_index,
        )
    if _is_html_pre_start(stripped):
        end_index, closed = _find_html_pre_end(raw, lines, start_index)
        warnings = () if closed else ("unclosed_html_pre",)
        return (
            [
                _make_line_block(
                    raw,
                    source_document_id_hash,
                    first_block_index,
                    "html_pre",
                    lines,
                    start_index,
                    end_index,
                    warnings,
                    policy_hash,
                )
            ],
            end_index,
        )
    if _is_markdown_heading(stripped):
        return (
            [_make_line_block(raw, source_document_id_hash, first_block_index, "heading", lines, start_index, start_index + 1, (), policy_hash)],
            start_index + 1,
        )
    if _is_setext_heading(raw, lines, start_index):
        return (
            [_make_line_block(raw, source_document_id_hash, first_block_index, "heading", lines, start_index, start_index + 2, (), policy_hash)],
            start_index + 2,
        )
    table_end = _table_end(raw, lines, start_index)
    if table_end > start_index:
        return (
            [_make_line_block(raw, source_document_id_hash, first_block_index, "table", lines, start_index, table_end, (), policy_hash)],
            table_end,
        )
    if _LIST_RE.match(content):
        end_index = _list_end(raw, lines, start_index)
        return (
            [_make_line_block(raw, source_document_id_hash, first_block_index, "list", lines, start_index, end_index, (), policy_hash)],
            end_index,
        )
    if _STACK_TRACE_RE.match(content):
        end_index = _shape_run_end(raw, lines, start_index, lambda text: bool(_STACK_TRACE_RE.match(text)))
        return (
            [_make_line_block(raw, source_document_id_hash, first_block_index, "stack_trace", lines, start_index, end_index, (), policy_hash)],
            end_index,
        )
    if _is_indented_code(content):
        end_index = _shape_run_end(raw, lines, start_index, _is_indented_code)
        return (
            [_make_line_block(raw, source_document_id_hash, first_block_index, "indented_code", lines, start_index, end_index, (), policy_hash)],
            end_index,
        )
    if _LOG_RE.match(content):
        end_index = _capped_shape_run_end(raw, lines, start_index, lambda text: bool(_LOG_RE.match(text)), config.max_log_run_lines)
        return (
            [_make_line_block(raw, source_document_id_hash, first_block_index, "log_run", lines, start_index, end_index, (), policy_hash)],
            end_index,
        )
    if _HTML_BLOCK_RE.match(content):
        end_index = _shape_run_end(raw, lines, start_index, lambda text: bool(_HTML_BLOCK_RE.match(text)))
        return (
            [_make_line_block(raw, source_document_id_hash, first_block_index, "html_block", lines, start_index, end_index, (), policy_hash)],
            end_index,
        )
    ui_end = _ui_chrome_end(raw, lines, start_index, config.max_ui_chrome_lines)
    if ui_end > start_index:
        return (
            [_make_line_block(raw, source_document_id_hash, first_block_index, "ui_chrome", lines, start_index, ui_end, (), policy_hash)],
            ui_end,
        )
    prose_end = _prose_end(raw, lines, start_index)
    return (
        [_make_line_block(raw, source_document_id_hash, first_block_index, "prose", lines, start_index, prose_end, (), policy_hash)],
        prose_end,
    )


def _oversized_line_blocks(
    raw: str,
    *,
    source_document_id_hash: str,
    line: RawLineRecord,
    first_block_index: int,
    config: ChunkedSegmentationConfig,
    policy_hash: str,
) -> list[NaturalBlockRecord]:
    content = raw[line.char_start : line.content_end]
    if not _has_safe_wrap_boundaries(content):
        return [
            _make_span_block(
                raw,
                source_document_id_hash,
                first_block_index,
                "oversized_unsegmentable",
                line.char_start,
                line.char_end,
                line.char_start,
                line.content_end,
                line.line_index,
                line.line_index + 1,
                ("oversized_unsegmentable",),
                policy_hash,
            )
        ]

    spans = _soft_wrap_spans(content, line.char_start, config.soft_wrap_target_chars)
    if spans is None:
        return [
            _make_span_block(
                raw,
                source_document_id_hash,
                first_block_index,
                "oversized_unsegmentable",
                line.char_start,
                line.char_end,
                line.char_start,
                line.content_end,
                line.line_index,
                line.line_index + 1,
                ("oversized_unsegmentable",),
                policy_hash,
            )
        ]
    blocks: list[NaturalBlockRecord] = []
    for offset, (start, end) in enumerate(spans):
        owned_end = line.char_end if offset == len(spans) - 1 else end
        content_end = line.content_end if offset == len(spans) - 1 else end
        blocks.append(
            _make_span_block(
                raw,
                source_document_id_hash,
                first_block_index + offset,
                "oversized_soft_wrapped_prose",
                start,
                owned_end,
                start,
                content_end,
                line.line_index,
                line.line_index + 1,
                ("soft_wrap_single_line_prose",),
                policy_hash,
            )
        )
    return blocks


def _make_line_block(
    raw: str,
    source_document_id_hash: str,
    block_index: int,
    block_kind: str,
    lines: tuple[RawLineRecord, ...],
    start_index: int,
    end_index_exclusive: int,
    warning_codes: tuple[str, ...] = (),
    policy_hash: str | None = None,
) -> NaturalBlockRecord:
    owned_start = lines[start_index].char_start
    owned_end = lines[end_index_exclusive - 1].char_end
    content_start, content_end = _trim_content_span(
        raw,
        owned_start,
        lines[end_index_exclusive - 1].content_end,
    )
    return _make_span_block(
        raw,
        source_document_id_hash,
        block_index,
        block_kind,
        owned_start,
        owned_end,
        content_start,
        content_end,
        start_index,
        end_index_exclusive,
        warning_codes,
        policy_hash or segmentation_policy_hash(),
    )


def _make_span_block(
    raw: str,
    source_document_id_hash: str,
    block_index: int,
    block_kind: str,
    owned_start: int,
    owned_end: int,
    content_start: int,
    content_end: int,
    start_line_index: int,
    end_line_index_exclusive: int,
    warning_codes: tuple[str, ...] = (),
    policy_hash: str | None = None,
) -> NaturalBlockRecord:
    if block_kind not in BLOCK_KINDS:
        raise ValueError(f"unsupported natural block kind: {block_kind}")
    content_start, content_end = _trim_content_span(raw, content_start, content_end)
    content_text = raw[content_start:content_end]
    normalized = normalize_canonical_text(content_text)
    raw_hash = normalized.raw_sha256
    normalized_hash = normalized.canonical_sha256
    block = NaturalBlockRecord(
        block_id_hash="",
        source_document_id_hash=source_document_id_hash,
        segmentation_policy_hash=policy_hash or segmentation_policy_hash(),
        block_index=block_index,
        block_kind=block_kind,
        content_char_start=content_start,
        content_char_end=content_end,
        owned_char_start=owned_start,
        owned_char_end=owned_end,
        start_line_index=start_line_index,
        end_line_index_exclusive=end_line_index_exclusive,
        raw_text_sha256=raw_hash,
        normalized_text_sha256=normalized_hash,
        proxy_token_count=_estimate_proxy_tokens_from_normalized(normalized.canonical_text),
        warning_codes=tuple(sorted(set(warning_codes))),
    )
    return NaturalBlockRecord(**{**asdict(block), "block_id_hash": _block_id_hash(block)})


def estimate_proxy_tokens(text: str) -> int:
    if not isinstance(text, str):
        raise TypeError("estimate_proxy_tokens requires a str input")
    normalized = normalize_text(text).canonical_text
    return _estimate_proxy_tokens_from_normalized(normalized)


def _estimate_proxy_tokens_from_normalized(normalized: str) -> int:
    word_count = 0
    non_ws_chars = 0
    in_word = False
    for char in normalized:
        if char.isspace():
            in_word = False
            continue
        non_ws_chars += 1
        if not in_word:
            word_count += 1
            in_word = True
    if non_ws_chars == 0:
        return 0
    return int(math.ceil(max(word_count / 0.75, non_ws_chars / 4.0)))


def _build_receipt(
    source: SourceDocumentRecord,
    lines: tuple[RawLineRecord, ...],
    blocks: tuple[NaturalBlockRecord, ...],
    trivia_spans: tuple[TriviaSpanRecord, ...],
    status: str,
    config: ChunkedSegmentationConfig,
    *,
    raw_text_sha256: str | None = None,
    normalized_text_sha256: str | None = None,
    raw_char_count: int | None = None,
) -> SegmentationReceipt:
    raw_text = source.raw_text if isinstance(source.raw_text, str) else None
    if raw_text is not None:
        if raw_text_sha256 is None and normalized_text_sha256 is None:
            normalized = normalize_canonical_text(raw_text)
            raw_text_sha256 = normalized.raw_sha256
            normalized_text_sha256 = normalized.canonical_sha256
        elif raw_text_sha256 is None:
            raw_text_sha256 = _sha256_text(raw_text)
        elif normalized_text_sha256 is None:
            normalized_text_sha256 = normalize_canonical_text(raw_text).canonical_sha256
        raw_char_count = len(raw_text) if raw_char_count is None else raw_char_count
    warning_counts = Counter(code for block in blocks for code in block.warning_codes)
    block_kind_counts = Counter(block.block_kind for block in blocks)
    payload = {
        "source_document_id_hash": source.source_document_id_hash,
        "split": source.split,
        "source_lineage_hash": source.source_lineage_hash,
        "source_token_count_hint": source.source_token_count_hint,
        "document_status": status,
        "raw_text_sha256": raw_text_sha256,
        "normalized_text_sha256": normalized_text_sha256,
        "raw_char_count": int(raw_char_count or 0),
        "raw_line_count": len(lines),
        "block_count": len(blocks),
        "trivia_span_count": len(trivia_spans),
        "total_proxy_token_count": sum(block.proxy_token_count for block in blocks),
        "block_kind_counts": dict(sorted(block_kind_counts.items())),
        "warning_code_counts": dict(sorted(warning_counts.items())),
        "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
        "segmentation_policy_hash": segmentation_policy_hash(config),
        "raw_line_policy_hash": raw_line_policy_hash(),
        "normalizer_version": NORMALIZER_VERSION,
        "proxy_token_estimator_version": PROXY_TOKEN_ESTIMATOR_VERSION,
        "block_id_policy_hash": block_id_policy_hash(),
        "raw_coverage_validation": status != "segmented" or bool(blocks or trivia_spans),
        "reconstruction_validation": status != "segmented" or bool(blocks or trivia_spans),
        "block_order_validation": True,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return SegmentationReceipt(**payload)


def _validate_receipt_matches_result(result: SegmentationResult) -> None:
    receipt = result.receipt
    raw = result.source.raw_text
    assert isinstance(raw, str)
    expected_status = "segmented"
    if receipt.document_status != expected_status:
        raise ValueError("segmentation receipt status mismatch")
    if receipt.raw_text_sha256 != _sha256_text(raw):
        raise ValueError("segmentation receipt raw hash mismatch")
    if receipt.normalized_text_sha256 != normalize_text(raw).canonical_sha256:
        raise ValueError("segmentation receipt normalized hash mismatch")
    if receipt.raw_char_count != len(raw):
        raise ValueError("segmentation receipt raw char count mismatch")
    if receipt.raw_line_count != len(result.lines):
        raise ValueError("segmentation receipt raw line count mismatch")
    if receipt.block_count != len(result.blocks):
        raise ValueError("segmentation receipt block count mismatch")
    if receipt.trivia_span_count != len(result.trivia_spans):
        raise ValueError("segmentation receipt trivia span count mismatch")
    if receipt.total_proxy_token_count != sum(block.proxy_token_count for block in result.blocks):
        raise ValueError("segmentation receipt proxy token count mismatch")
    if receipt.block_kind_counts != dict(sorted(Counter(block.block_kind for block in result.blocks).items())):
        raise ValueError("segmentation receipt block kind count mismatch")
    if receipt.warning_code_counts != dict(
        sorted(Counter(code for block in result.blocks for code in block.warning_codes).items())
    ):
        raise ValueError("segmentation receipt warning count mismatch")
    if any(block.segmentation_policy_hash != receipt.segmentation_policy_hash for block in result.blocks):
        raise ValueError("natural block segmentation policy hash mismatch")
    if receipt.raw_coverage_validation is not True:
        raise ValueError("segmentation receipt raw coverage validation must be true")
    if receipt.reconstruction_validation is not True:
        raise ValueError("segmentation receipt reconstruction validation must be true")
    if receipt.block_order_validation is not True:
        raise ValueError("segmentation receipt block order validation must be true")


def _validate_receipt_identity(result: SegmentationResult) -> None:
    receipt = result.receipt
    if receipt.source_document_id_hash != result.source.source_document_id_hash:
        raise ValueError("segmentation receipt source id mismatch")
    if receipt.split != result.source.split:
        raise ValueError("segmentation receipt split mismatch")
    if receipt.source_lineage_hash != result.source.source_lineage_hash:
        raise ValueError("segmentation receipt source lineage mismatch")
    if receipt.source_token_count_hint != result.source.source_token_count_hint:
        raise ValueError("segmentation receipt source token count hint mismatch")


def _validate_terminal_receipt_matches_source(result: SegmentationResult) -> None:
    receipt = result.receipt
    raw = result.source.raw_text
    if isinstance(raw, str):
        if receipt.raw_text_sha256 != _sha256_text(raw):
            raise ValueError("terminal segmentation receipt raw hash mismatch")
        if receipt.normalized_text_sha256 != normalize_text(raw).canonical_sha256:
            raise ValueError("terminal segmentation receipt normalized hash mismatch")
        if receipt.raw_char_count != len(raw):
            raise ValueError("terminal segmentation receipt raw char count mismatch")
    else:
        if receipt.raw_text_sha256 is not None or receipt.normalized_text_sha256 is not None:
            raise ValueError("terminal non-string receipt must not carry text hashes")
        if receipt.raw_char_count != 0:
            raise ValueError("terminal non-string receipt raw char count mismatch")
    if receipt.raw_line_count != 0:
        raise ValueError("terminal segmentation receipt raw line count mismatch")
    if receipt.total_proxy_token_count != 0:
        raise ValueError("terminal segmentation receipt proxy token count mismatch")
    if receipt.block_kind_counts or receipt.warning_code_counts:
        raise ValueError("terminal segmentation receipt aggregate counts must be empty")
    if receipt.raw_coverage_validation is not True:
        raise ValueError("terminal segmentation receipt raw coverage validation must be true")
    if receipt.reconstruction_validation is not True:
        raise ValueError("terminal segmentation receipt reconstruction validation must be true")
    if receipt.block_order_validation is not True:
        raise ValueError("terminal segmentation receipt block order validation must be true")


def _validate_receipt_hash(receipt: SegmentationReceipt) -> None:
    payload = asdict(receipt)
    receipt_hash = payload.pop("receipt_hash")
    if receipt_hash != stable_json_hash(payload):
        raise ValueError("segmentation receipt hash mismatch")
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if getattr(receipt, field) is not False:
            raise ValueError(f"segmentation receipt must keep {field}=false")


def _validate_block_record(block: NaturalBlockRecord, raw: str) -> None:
    _require_sha256(block.block_id_hash, "block_id_hash")
    _require_sha256(block.source_document_id_hash, "source_document_id_hash")
    _require_sha256(block.segmentation_policy_hash, "segmentation_policy_hash")
    if block.block_kind not in BLOCK_KINDS:
        raise ValueError("unsupported natural block kind")
    for field_name in (
        "block_index",
        "content_char_start",
        "content_char_end",
        "owned_char_start",
        "owned_char_end",
        "start_line_index",
        "end_line_index_exclusive",
        "proxy_token_count",
    ):
        _require_nonnegative_int(getattr(block, field_name), field_name)
    if not (0 <= block.owned_char_start < block.owned_char_end <= len(raw)):
        raise ValueError("natural block owned span is outside raw document")
    if not block.start_line_index < block.end_line_index_exclusive:
        raise ValueError("natural block line range is invalid")
    if not (block.owned_char_start <= block.content_char_start <= block.content_char_end <= block.owned_char_end):
        raise ValueError("natural block content span must be inside owned span")
    content = raw[block.content_char_start : block.content_char_end]
    if block.raw_text_sha256 != _sha256_text(content):
        raise ValueError("natural block raw hash mismatch")
    if block.normalized_text_sha256 != normalize_text(content).canonical_sha256:
        raise ValueError("natural block normalized hash mismatch")
    if block.block_id_hash != _block_id_hash(block):
        raise ValueError("natural block id hash mismatch")


def _validate_trivia_record(trivia: TriviaSpanRecord, raw: str) -> None:
    _require_sha256(trivia.source_document_id_hash, "source_document_id_hash")
    if trivia.trivia_kind != "blank_separator":
        raise ValueError("unsupported trivia kind")
    if not (0 <= trivia.char_start < trivia.char_end <= len(raw)):
        raise ValueError("trivia span is outside raw document")
    if not trivia.start_line_index < trivia.end_line_index_exclusive:
        raise ValueError("trivia line range is invalid")
    if not _is_raw_blank(raw[trivia.char_start : trivia.char_end]):
        raise ValueError("trivia span must be blank/separator text")


def _validate_coverage(raw: str, coverage: list[tuple[int, int, str]]) -> None:
    cursor = 0
    for start, end, label in sorted(coverage):
        if start != cursor:
            raise ValueError(f"segmentation coverage gap before {label}")
        if end <= start:
            raise ValueError(f"segmentation coverage span is empty for {label}")
        cursor = end
    if cursor != len(raw):
        raise ValueError("segmentation coverage does not reach end of raw document")


def _validate_trivia_adjacency(
    blocks: tuple[NaturalBlockRecord, ...],
    trivia_spans: tuple[TriviaSpanRecord, ...],
    *,
    raw_end: int,
) -> None:
    blocks_by_start = {block.owned_char_start: block for block in blocks}
    blocks_by_end = {block.owned_char_end: block for block in blocks}
    ordered_blocks = sorted(blocks, key=lambda block: block.owned_char_start)
    if [block.block_index for block in ordered_blocks] != list(range(len(blocks))):
        raise ValueError("natural block indexes must match span order")
    for trivia in trivia_spans:
        left_block = blocks_by_end.get(trivia.char_start)
        right_block = blocks_by_start.get(trivia.char_end)
        expected_left = left_block.block_index if left_block is not None else None
        expected_right = right_block.block_index if right_block is not None else None
        if expected_left is None and trivia.char_start != 0:
            raise ValueError("trivia span is not adjacent to a left block")
        if expected_right is None and trivia.char_end != raw_end:
            raise ValueError("trivia span is not adjacent to a right block")
        if trivia.left_block_index != expected_left or trivia.right_block_index != expected_right:
            raise ValueError("trivia adjacency indexes do not match neighboring blocks")
        if expected_left is not None and expected_right is not None and expected_right != expected_left + 1:
            raise ValueError("trivia adjacency indexes must bridge consecutive blocks")


def _validate_line_range_for_span(
    lines: tuple[RawLineRecord, ...],
    start_line_index: int,
    end_line_index_exclusive: int,
    span_start: int,
    span_end: int,
    label: str,
    *,
    require_exact_boundaries: bool,
) -> None:
    if not start_line_index < end_line_index_exclusive <= len(lines):
        raise ValueError(f"{label} line range is invalid")
    line_start = lines[start_line_index].char_start
    line_end = lines[end_line_index_exclusive - 1].char_end
    if not line_start <= span_start < span_end <= line_end:
        raise ValueError(f"{label} line range does not contain span")
    minimal_start = next(
        index for index, line in enumerate(lines) if line.char_start <= span_start < line.char_end
    )
    minimal_end = next(
        index + 1 for index, line in enumerate(lines) if line.char_start < span_end <= line.char_end
    )
    if start_line_index != minimal_start or end_line_index_exclusive != minimal_end:
        raise ValueError(f"{label} line range is not minimal")
    if require_exact_boundaries and (span_start != line_start or span_end != line_end):
        raise ValueError(f"{label} line range does not match line boundaries")


def _block_id_hash(block: NaturalBlockRecord) -> str:
    return stable_json_hash(
        {
            "block_id_policy_version": BLOCK_ID_POLICY_VERSION,
            "source_document_id_hash": block.source_document_id_hash,
            "segmentation_policy_hash": block.segmentation_policy_hash,
            "block_index": block.block_index,
            "block_kind": block.block_kind,
            "content_char_start": block.content_char_start,
            "content_char_end": block.content_char_end,
            "owned_char_start": block.owned_char_start,
            "owned_char_end": block.owned_char_end,
            "raw_text_sha256": block.raw_text_sha256,
            "normalized_text_sha256": block.normalized_text_sha256,
        }
    )


def natural_block_id_hash(block: NaturalBlockRecord) -> str:
    return _block_id_hash(block)


def _classify_line_kind(content: str) -> str:
    if _is_raw_blank(content):
        return "blank"
    stripped = content.strip()
    if _fence_marker(stripped):
        return "fence"
    if _is_html_pre_start(stripped):
        return "html_pre"
    if _is_markdown_heading(stripped) or _SETEXT_RE.match(stripped):
        return "heading"
    if "|" in content:
        return "table_like"
    if _LIST_RE.match(content):
        return "list"
    if _STACK_TRACE_RE.match(content):
        return "stack_trace"
    if _LOG_RE.match(content):
        return "log"
    if _is_indented_code(content):
        return "indented_code"
    if _HTML_BLOCK_RE.match(content):
        return "html"
    if _is_short_ui_line(content):
        return "ui_chrome"
    return "prose"


def _is_raw_blank(text: str) -> bool:
    for char in text:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category == "Cf" or category.startswith("C"):
            continue
        return False
    return True


def _fence_marker(stripped: str) -> str | None:
    match = _FENCE_RE.match(stripped)
    return match.group(1) if match else None


def _find_fence_end(
    raw: str,
    lines: tuple[RawLineRecord, ...],
    start_index: int,
    fence: str,
) -> tuple[int, bool]:
    marker = fence[0] * len(fence)
    for index in range(start_index + 1, len(lines)):
        stripped = raw[lines[index].char_start : lines[index].content_end].strip()
        if stripped.startswith(marker):
            return index + 1, True
    return len(lines), False


def _is_html_pre_start(stripped: str) -> bool:
    lowered = stripped.casefold()
    return lowered.startswith("<pre") and ">" in lowered


def _find_html_pre_end(raw: str, lines: tuple[RawLineRecord, ...], start_index: int) -> tuple[int, bool]:
    for index in range(start_index, len(lines)):
        lowered = raw[lines[index].char_start : lines[index].content_end].casefold()
        if "</pre>" in lowered:
            return index + 1, True
    return len(lines), False


def _is_markdown_heading(stripped: str) -> bool:
    return bool(_MARKDOWN_HEADING_RE.match(stripped))


def _is_setext_heading(raw: str, lines: tuple[RawLineRecord, ...], start_index: int) -> bool:
    if start_index + 1 >= len(lines):
        return False
    current = raw[lines[start_index].char_start : lines[start_index].content_end]
    underline = raw[lines[start_index + 1].char_start : lines[start_index + 1].content_end]
    return bool(current.strip()) and bool(_SETEXT_RE.match(underline.strip()))


def _table_end(raw: str, lines: tuple[RawLineRecord, ...], start_index: int) -> int:
    if start_index + 1 >= len(lines):
        return start_index
    first = raw[lines[start_index].char_start : lines[start_index].content_end]
    second = raw[lines[start_index + 1].char_start : lines[start_index + 1].content_end]
    if "|" not in first or not _TABLE_SEPARATOR_RE.match(second):
        return start_index
    index = start_index + 2
    while index < len(lines):
        content = raw[lines[index].char_start : lines[index].content_end]
        if lines[index].line_kind == "blank" or "|" not in content:
            break
        index += 1
    return index


def _list_end(raw: str, lines: tuple[RawLineRecord, ...], start_index: int) -> int:
    index = start_index + 1
    while index < len(lines):
        content = raw[lines[index].char_start : lines[index].content_end]
        if lines[index].line_kind == "blank":
            break
        if _LIST_RE.match(content) or content.startswith((" ", "\t")):
            index += 1
            continue
        break
    return index


def _is_indented_code(content: str) -> bool:
    return content.startswith(("    ", "\t")) and bool(content.strip())


def _shape_run_end(
    raw: str,
    lines: tuple[RawLineRecord, ...],
    start_index: int,
    predicate,
) -> int:
    index = start_index
    while index < len(lines):
        content = raw[lines[index].char_start : lines[index].content_end]
        if lines[index].line_kind == "blank" or not predicate(content):
            break
        index += 1
    return index


def _capped_shape_run_end(
    raw: str,
    lines: tuple[RawLineRecord, ...],
    start_index: int,
    predicate,
    max_lines: int,
) -> int:
    return min(_shape_run_end(raw, lines, start_index, predicate), start_index + max_lines)


def _ui_chrome_end(raw: str, lines: tuple[RawLineRecord, ...], start_index: int, max_lines: int) -> int:
    index = start_index
    while index < len(lines) and index < start_index + max_lines:
        content = raw[lines[index].char_start : lines[index].content_end]
        if lines[index].line_kind == "blank" or not _is_short_ui_line(content):
            break
        index += 1
    return index if index - start_index >= 3 else start_index


def _is_short_ui_line(content: str) -> bool:
    stripped = content.strip()
    if not stripped or len(stripped) > 48:
        return False
    word_count = len(re.findall(r"\w+", stripped))
    return 1 <= word_count <= 4


def _prose_end(raw: str, lines: tuple[RawLineRecord, ...], start_index: int) -> int:
    index = start_index + 1
    while index < len(lines):
        if lines[index].line_kind == "blank":
            break
        content = raw[lines[index].char_start : lines[index].content_end]
        stripped = content.strip()
        if (
            _fence_marker(stripped)
            or _is_html_pre_start(stripped)
            or _is_markdown_heading(stripped)
            or _LIST_RE.match(content)
            or _LOG_RE.match(content)
            or _STACK_TRACE_RE.match(content)
            or _HTML_BLOCK_RE.match(content)
            or _is_indented_code(content)
        ):
            break
        if _table_end(raw, lines, index) > index:
            break
        index += 1
    return index


def _has_safe_wrap_boundaries(content: str) -> bool:
    return bool(_SAFE_WRAP_RE.search(content))


def _soft_wrap_spans(content: str, absolute_start: int, target_chars: int) -> list[tuple[int, int]] | None:
    spans: list[tuple[int, int]] = []
    local_start = 0
    while local_start < len(content):
        target = min(len(content), local_start + target_chars)
        if target == len(content):
            local_end = len(content)
        else:
            local_end = _last_safe_boundary(content, local_start, target)
            if local_end is None or local_end <= local_start:
                return None
        if not content[local_start:local_end].strip():
            return None
        spans.append((absolute_start + local_start, absolute_start + local_end))
        local_start = local_end
    return spans


def _last_safe_boundary(content: str, start: int, target: int) -> int | None:
    window = content[start:target]
    matches = list(_SAFE_WRAP_RE.finditer(window))
    if not matches:
        return None
    return start + matches[-1].end()


def _trim_content_span(raw: str, start: int, end: int) -> tuple[int, int]:
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return start, end


def _sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _require_sha256(value: str | None, field: str) -> str:
    if type(value) is not str or len(value) != _SHA256_LENGTH or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase sha256")
    return value


def _require_positive_int(value: Any, field: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer at least {minimum}")
    return int(value)


def _require_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _sorted_count_dict(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    output: dict[str, int] = {}
    for key, count in value.items():
        key_text = str(key)
        if not key_text or not _SAFE_PUBLIC_KEY_RE.match(key_text):
            raise ValueError(f"{field} contains unsafe key")
        output[key_text] = _require_nonnegative_int(count, f"{field}.{key_text}")
    return dict(sorted(output.items()))


def _report_hash(payload: dict[str, Any]) -> str:
    return stable_json_hash({key: value for key, value in payload.items() if key != "report_hash"})


_PUBLIC_REPORT_ALLOWED_KEYS = {
    "artifact_kind",
    "artifact_status",
    "schema_version",
    "chunked_clean_run_schema_version",
    "segmentation_policy_version",
    "segmentation_policy_hash",
    "raw_line_policy_hash",
    "normalizer_version",
    "proxy_token_estimator_version",
    "block_id_policy_hash",
    "public_report_schema_hash",
    "source_manifest_hash",
    "source_schedule_hash",
    "sqlite_aggregate_hash",
    "document_count",
    "status_counts",
    "split_counts",
    "total_block_count",
    "total_trivia_span_count",
    "block_kind_counts",
    "warning_code_counts",
    "proxy_token_histogram",
    "sqlite_source_of_truth",
    "raw_coverage_validation",
    "reconstruction_validation",
    "block_order_validation",
    "public_allowlist_validation",
    "future_lane_tables_empty",
    "accepted_manifest_created",
    "clean_run_manifest_created",
    "tokenizer_shards_created",
    "packed_shards_created",
    "training_shards_created",
    "base_pretraining_started",
    "ready_for_tokenizer_and_packer",
    "ready_for_training",
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "report_hash",
}
_PUBLIC_REPORT_INT_FIELDS = {
    "schema_version",
    "chunked_clean_run_schema_version",
    "document_count",
    "total_block_count",
    "total_trivia_span_count",
}
_PUBLIC_REPORT_COUNT_DICT_FIELDS = {
    "status_counts",
    "split_counts",
    "block_kind_counts",
    "warning_code_counts",
    "proxy_token_histogram",
}
_PUBLIC_REPORT_TRUE_FIELDS = {
    "raw_coverage_validation",
    "reconstruction_validation",
    "block_order_validation",
    "public_allowlist_validation",
    "future_lane_tables_empty",
}
_PUBLIC_REPORT_FALSE_FIELDS = {
    "accepted_manifest_created",
    "clean_run_manifest_created",
    "tokenizer_shards_created",
    "packed_shards_created",
    "training_shards_created",
    "base_pretraining_started",
    "ready_for_tokenizer_and_packer",
    "ready_for_training",
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
}
_PUBLIC_REPORT_HASH_FIELDS = {
    "segmentation_policy_hash",
    "raw_line_policy_hash",
    "block_id_policy_hash",
    "public_report_schema_hash",
    "source_manifest_hash",
    "source_schedule_hash",
    "sqlite_aggregate_hash",
    "report_hash",
}
_PUBLIC_REPORT_ALLOWED_STATUSES = {"segmented", "missing_text", "non_string_text", "empty_or_whitespace"}
_PUBLIC_REPORT_PROXY_BUCKETS = {"0", "1_128", "129_512", "513_1024", "1025_plus"}
_PUBLIC_REPORT_WARNING_CODES = {
    "oversized_unsegmentable",
    "soft_wrap_single_line_prose",
    "unclosed_fenced_code",
    "unclosed_html_pre",
}
