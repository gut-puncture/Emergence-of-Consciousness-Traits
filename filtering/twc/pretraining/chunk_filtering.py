"""Chunk-native classifier filtering and clean-run assembly.

This module owns the current production-facing filtering lane for TWC. It is
separate from the older whole-document accepted-corpus runner because the
current methodology filters classifier candidates built from natural blocks and
turns rejected/quarantined candidates into clean-run gaps.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import functools
import hashlib
import importlib
import json
import math
import os
import platform
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import yaml

from twc.classifiers.chunk_training import (
    CHUNK_SPARSE_MODEL_ID,
    CHUNK_WINDOW_MIL_MODEL_ID,
    WINDOW_INPUT_CORE_ONLY,
    WINDOW_INPUT_FULL_EVIDENCE,
    WINDOW_INPUT_SPLIT_HALO_CAPPED,
)
from twc.classifiers.feature_extraction import WHITESPACE_RE, WORD_RE
from twc.classifiers.filtering_handoff import build_chunk_classifier_decision
from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.classifiers.window_mil import (
    W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2,
    W1_ATTN_IMPLEMENTATION_SDPA,
    SUPPORTED_W1_ATTN_IMPLEMENTATIONS,
    normalize_w1_attn_implementation,
)
from twc.classifiers.precision_policy import (
    W1_PRECISION_AMP_BF16,
    W1_PRECISION_FP32,
    apply_w1_precision_policy,
    build_w1_precision_policy,
    normalize_w1_precision_mode,
    runtime_w1_precision_observation,
    validate_w1_precision_policy,
    w1_precision_context,
)
from twc.data.decision_log import HFFsspecParquetRowProvider, LocalParquetRowProvider
from twc.data.decision_schema import stable_json_hash
from twc.data.risk_ontology import RiskOntology, load_ontology
from twc.pretraining.chunked_clean_run import (
    SegmentationResult,
    SourceDocumentRecord,
    _estimate_proxy_tokens_from_normalized,
    _segment_source_document_prefilter_fast,
    segment_source_document,
    validate_segmentation_result,
)
from twc.pretraining.chunked_clean_run_state import (
    ChunkedCleanRunState,
    ChunkedCleanRunStateConfig,
    open_chunked_clean_run_state,
    store_segmentation_result,
    validate_chunked_clean_run_state,
)
from twc.pretraining.deterministic_filter import (
    DeterministicFilterRuntime,
    build_deterministic_filter_runtime,
    build_deterministic_filter_policy_payload,
    evaluate_chunk_candidate_deterministic_filter,
    evaluate_chunk_candidate_deterministic_filter_with_runtime,
    validate_deterministic_filter_policy_payload,
)


CHUNK_FILTERING_CONFIG_KIND = "twc_chunk_filtering_config"
CHUNK_FILTER_SCHEDULE_KIND = "twc_chunk_filter_canary_schedule"
CHUNK_FILTER_CANARY_REPORT_KIND = "twc_chunk_filter_canary_schedule_report"
LOCAL_PARQUET_EXECUTION_SCHEDULE_KIND = "twc_private_local_parquet_execution_schedule_v1"
LOCAL_PARQUET_MIRROR_REPORT_KIND = "twc_private_local_parquet_mirror_report_v1"
LOCAL_PARQUET_ALL_ROWS_PREFILTER_PLAN_KIND = "twc_local_parquet_all_rows_prefilter_plan_v1"
LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND = "twc_local_parquet_token_target_tranche_plan_v1"
PREFILTER_SUPERVISOR_MANIFEST_KIND = "twc_local_parquet_prefilter_supervisor_manifest_v1"
PREFILTER_SUPERVISOR_SUMMARY_KIND = "twc_local_parquet_prefilter_supervisor_summary_v1"
TOKEN_TARGET_TRANCHE_MODE = "token_target_stratified_tranche_v1"
CHUNK_INFERENCE_ARTIFACT_MANIFEST_KIND = "twc_chunk_classifier_inference_artifact_manifest"
CHUNK_FILTER_DOCUMENT_REPORT_KIND = "twc_chunk_filter_document_report"
CHUNK_FILTER_RUN_REPORT_KIND = "twc_chunk_filter_run_report"
PREFILTER_CPU_STAGE_KIND = "twc_chunk_filter_prefilter_cpu_stage_receipt_v1"
W1_SCORE_STAGE_KIND = "twc_chunk_filter_w1_score_stage_receipt_v1"
W1_RUNTIME_EXECUTION_ENVELOPE_KIND = "twc_chunk_filter_w1_runtime_execution_envelope_v1"
W1_RUNTIME_CAP_COMPARISON_REPORT_KIND = "twc_chunk_filter_w1_runtime_cap_comparison_report_v1"
W1_SCORE_RUN_COMPARISON_REPORT_KIND = "twc_chunk_filter_w1_score_run_comparison_report_v1"
W1_FA2_PACKAGE_CANDIDATE_MANIFEST_KIND = "twc_w1_fa2_package_candidate_manifest_v1"
W1_FA2_RUNTIME_COMPATIBILITY_MANIFEST_KIND = "twc_w1_fa2_runtime_compatibility_manifest_v1"
W1_SURVIVOR_LENGTH_DISTRIBUTION_KIND = "twc_w1_survivor_length_distribution_v1"
W1_SCORER_PERFORMANCE_LEDGER_KIND = "twc_w1_scorer_performance_ledger_v1"
W1_ACCEPTED_TOKEN_LEDGER_KIND = "twc_w1_accepted_token_ledger_v1"
W1_COMPACT_START_STATE_KIND = "twc_w1_compact_start_state_v1"
ASSEMBLED_STAGE_REPORT_KIND = "twc_chunk_filter_assembled_report_v1"
SHARDED_W1_HANDOFF_KIND = "twc_local_parquet_sharded_prefilter_w1_handoff_v1"
SHARDED_W1_SCORE_SUMMARY_KIND = "twc_chunk_filter_sharded_w1_score_summary_v1"
PARTIAL_SHARDED_W1_DIAGNOSTIC_HANDOFF_KIND = "twc_partial_shard_w1_diagnostic_handoff_v1"
PARTIAL_SHARDED_W1_DIAGNOSTIC_SUMMARY_KIND = "twc_partial_shard_w1_diagnostic_summary_v1"
SELECTED_POLICY_FREEZE_REPORT_KIND = "twc_chunk_classifier_freeze_v2_eval_report"
CHUNK_FILTERING_POLICY_VERSION = "chunked_clean_run_filtering_v1"
CHUNK_FILTERING_CONFIG_SCHEMA_VERSION = 2
CHUNK_FILTERING_STATE_SCHEMA_VERSION = 1
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
W1_PREPARE_PIPELINE_SERIAL = "serial_v1"
W1_PREPARE_PIPELINE_BOUNDED_OVERLAP = "bounded_tokenize_overlap_v1"
W1_BATCHING_FIXED_DOCUMENT_COUNT = "fixed_document_count_v1"
W1_BATCHING_ESTIMATED_WINDOW_BUDGET = "estimated_window_budget_v1"
W1_PIN_MEMORY_AUTO = "auto"
W1_PIN_MEMORY_ON = "on"
W1_PIN_MEMORY_OFF = "off"
W1_TOKENIZERS_PARALLELISM_INHERIT = "inherit"
W1_TOKENIZERS_PARALLELISM_FALSE = "false"
W1_TOKENIZERS_PARALLELISM_TRUE = "true"
W1_PERSIST_STRATEGY_PER_BATCH = "per_batch_commit_v1"
W1_PERSIST_STRATEGY_ROW_THRESHOLD = "row_threshold_v1"
LEGACY_CPU_PREFILTER_CONFIG_HASH = "83f3d4f7821c57333920b74636fb7947f791fbc4ce07504a38aeea553f5094c5"
LEGACY_CPU_PREFILTER_SELECTED_POLICY_HASH = "fee1a25ae53f498424aa12a51d5c628ece983c2123665d7c4224975145db299c"
LEGACY_CPU_PREFILTER_SELECTED_POLICY_REPORT_HASH = "ba47f27407ea8f547f3d15a57300f92c98fdc521552c14c19f66da00a1f0b960"
LEGACY_CPU_PREFILTER_W1_THRESHOLD = 0.976143479347229
DEFAULT_TARGET_PROXY_TOKENS = 1024
DEFAULT_HALO_CHARS = 512
DEFAULT_CANARY_HIGH_TOKEN_ROWS = 800
DEFAULT_CANARY_RANDOM_ROWS = 200
LEGACY_ACCEPTED_TOKEN_TARGET_LOWER_BOUND = 6_000_000_000
DEFAULT_ACCEPTED_TOKEN_TARGET_LOWER_BOUND = 4_000_000_000
DEFAULT_CANARY_BUDGET_CAP_USD = 3.0
DEFAULT_FULL_RUN_BUDGET_CAP_USD = 20.0
DEFAULT_TOKEN_BYTES_PER_TOKEN_LOWER_BOUND = 8
ALLOWED_DECISIONS = {"safe", "reject", "quarantine", "error"}
PUBLIC_FALSE_FLAGS = (
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "accepted_manifest_created",
    "accepted_manifest",
    "tokenizer_shards_created",
    "packed_shards_created",
    "training_shards_created",
    "base_pretraining_started",
    "pretraining_started",
)
LOCAL_PARQUET_PREFILTER_COLUMNS = ("text", "token_count", "id")
HASHED_NGRAM_COLUMN_CACHE_MAXSIZE = 262_144
ASCII_WORD_BYTES_RE = re.compile(rb"\w+", re.ASCII)
ASCII_WHITESPACE_TRANSLATION = bytes.maketrans(
    bytes(range(128)),
    bytes(32 if chr(value).isspace() else value for value in range(128)),
)
ASCII_LOWER_WHITESPACE_TRANSLATION = bytes.maketrans(
    bytes(range(128)),
    bytes(
        32
        if chr(value).isspace()
        else value + 32
        if 65 <= value <= 90
        else value
        for value in range(128)
    ),
)
PREFILTER_TIMING_SECOND_FIELDS = (
    "row_group_iteration",
    "parquet_fetch",
    "segmentation",
    "candidate_construction",
    "deterministic_filter",
    "s1_feature_score",
    "sqlite_write",
    "survivor_queue_export",
    "receipt_build",
)
PREFILTER_TIMING_COUNT_FIELDS = (
    "row_groups_considered",
    "row_groups_fetched",
    "rows_fetched",
    "segmentation_sampled_validations",
    "candidates_built",
    "deterministic_candidates",
    "s1_scored_candidates",
)
PREFILTER_ROW_GROUP_METRICS_FILENAME = "prefilter_row_group_metrics.jsonl"
PREFILTER_SUPERVISOR_SCHEMA_VERSION = 1
PREFILTER_SUPERVISOR_MODES = {"mergeable_tranche", "diagnostic_sweep"}
SEGMENTATION_VALIDATION_MODES = {"full", "sampled"}
DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS = 1000
ROW_GROUP_SHARDING_POLICY_ORDINAL = "ordinal_modulo_v1"
ROW_GROUP_SHARDING_POLICY_WEIGHTED_LPT = "weighted_lpt_v1"
ROW_GROUP_SHARDING_POLICY_SOURCE_LOCAL_WEIGHTED_LPT = "source_local_weighted_lpt_v1"
ROW_GROUP_SHARDING_POLICIES = {
    ROW_GROUP_SHARDING_POLICY_ORDINAL,
    ROW_GROUP_SHARDING_POLICY_WEIGHTED_LPT,
    ROW_GROUP_SHARDING_POLICY_SOURCE_LOCAL_WEIGHTED_LPT,
}
PREFILTER_SUPERVISOR_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _new_prefilter_runtime_metrics() -> dict[str, Any]:
    return {
        "seconds": {field: 0.0 for field in PREFILTER_TIMING_SECOND_FIELDS},
        "counts": Counter(),
    }


def _record_prefilter_elapsed(metrics: dict[str, Any], field: str, start: float) -> float:
    if field not in PREFILTER_TIMING_SECOND_FIELDS:
        raise ValueError(f"unsupported prefilter timing field: {field}")
    elapsed = time.monotonic() - start
    metrics["seconds"][field] = float(metrics["seconds"].get(field, 0.0)) + elapsed
    return elapsed


def _increment_prefilter_count(metrics: dict[str, Any], field: str, count: int = 1) -> None:
    if field not in PREFILTER_TIMING_COUNT_FIELDS:
        raise ValueError(f"unsupported prefilter count field: {field}")
    metrics["counts"][field] += int(count)


def _validate_segmentation_validation_mode(mode: str) -> str:
    value = str(mode)
    if value not in SEGMENTATION_VALIDATION_MODES:
        raise ValueError("segmentation validation mode must be full or sampled")
    return value


def _segmentation_sample_modulus(value: int) -> int:
    return _positive_int(value, "segmentation_validation_sample_modulus")


def _segment_for_prefilter(
    source: SourceDocumentRecord,
    *,
    validation_mode: str,
    sample_modulus: int,
    metrics: dict[str, Any] | None = None,
) -> SegmentationResult:
    mode = _validate_segmentation_validation_mode(validation_mode)
    modulus = _segmentation_sample_modulus(sample_modulus)
    if mode == "full":
        return segment_source_document(source)
    result = _segment_source_document_prefilter_fast(source)
    sampled_count = 0 if metrics is None else int(metrics["counts"].get("segmentation_sampled_validations", 0))
    if sampled_count == 0 or int(source.source_document_id_hash[:16], 16) % modulus == 0:
        validate_segmentation_result(result)
        if metrics is not None:
            _increment_prefilter_count(metrics, "segmentation_sampled_validations")
    return result


def _prefilter_runtime_receipt_fields(metrics: Mapping[str, Any]) -> dict[str, Any]:
    seconds = metrics.get("seconds", {})
    counts = metrics.get("counts", {})
    payload: dict[str, Any] = {"prefilter_timing_schema_version": 1}
    for field in PREFILTER_TIMING_SECOND_FIELDS:
        payload[f"prefilter_{field}_seconds"] = round(float(seconds.get(field, 0.0)), 6)
    for field in PREFILTER_TIMING_COUNT_FIELDS:
        payload[f"prefilter_{field}"] = int(counts.get(field, 0))
    payload["prefilter_private_w1_evidence_write_included_in_sqlite_write_seconds"] = True
    return payload


def _attach_prefilter_runtime_receipt_fields(report: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(report)
    payload.update(_prefilter_runtime_receipt_fields(metrics))
    if (
        payload.get("segmentation_validation_mode") == "sampled"
        and int(payload.get("segmented_rows", 0)) > 0
        and int(payload.get("prefilter_segmentation_sampled_validations", 0)) == 0
    ):
        raise ValueError("sampled segmentation validation produced zero validations")
    payload.pop("stage_receipt_hash", None)
    payload["stage_receipt_hash"] = stable_json_hash(payload)
    validate_public_no_raw_payload(payload)
    return payload


def _prefilter_row_group_metric_key(entry: Mapping[str, Any]) -> tuple[str, int]:
    return (stable_json_hash(str(entry.get("source_file_id") or "")), int(entry.get("row_group_index", 0)))


def _new_prefilter_row_group_metric_state(entry: Mapping[str, Any]) -> dict[str, Any]:
    source_file_id_hash, row_group_index = _prefilter_row_group_metric_key(entry)
    return {
        "entry": dict(entry),
        "source_schedule_hash": str(
            entry.get("row_group_metric_source_schedule_hash")
            or entry.get("source_schedule_hash")
            or ""
        ),
        "input_schedule_hash": str(entry.get("source_schedule_hash") or ""),
        "schedule_entry_hash": str(entry.get("entry_hash") or ""),
        "source_file_id_hash": source_file_id_hash,
        "row_group_index": int(row_group_index),
        "row_group_num_rows": int(entry.get("row_group_num_rows", 0)),
        "rows_fetched": 0,
        "terminal_rows": 0,
        "candidate_count": 0,
        "survivor_count": 0,
        "route_counts": Counter(),
        "seconds": {field: 0.0 for field in PREFILTER_TIMING_SECOND_FIELDS},
    }


def _add_prefilter_row_group_elapsed(state: dict[str, Any], field: str, elapsed: float) -> None:
    if field not in PREFILTER_TIMING_SECOND_FIELDS:
        raise ValueError(f"unsupported prefilter row-group timing field: {field}")
    seconds = state.setdefault("seconds", {})
    seconds[field] = float(seconds.get(field, 0.0)) + float(elapsed)


def _prefilter_row_group_metric_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "artifact_kind": "twc_chunk_filter_prefilter_row_group_metrics_v1",
        "schema_version": 1,
        "row_group_metric_id": stable_json_hash(
            {
                "source_schedule_hash": state["source_schedule_hash"],
                "input_schedule_hash": state["input_schedule_hash"],
                "source_file_id_hash": state["source_file_id_hash"],
                "row_group_index": int(state["row_group_index"]),
                "row_group_num_rows": int(state["row_group_num_rows"]),
            }
        ),
        "source_schedule_hash": _require_sha256(state["source_schedule_hash"], "row group source_schedule_hash"),
        "input_schedule_hash": _require_sha256(state["input_schedule_hash"], "row group input_schedule_hash"),
        "schedule_entry_hash": _require_sha256(state["schedule_entry_hash"], "row group schedule_entry_hash"),
        "source_file_id_hash": _require_sha256(state["source_file_id_hash"], "row group source_file_id_hash"),
        "row_group_index": int(state["row_group_index"]),
        "row_group_num_rows": int(state["row_group_num_rows"]),
        "rows_fetched": int(state.get("rows_fetched", 0)),
        "terminal_rows": int(state.get("terminal_rows", 0)),
        "candidate_count": int(state.get("candidate_count", 0)),
        "survivor_count": int(state.get("survivor_count", 0)),
        "route_counts": _sorted_counter(Counter(dict(state.get("route_counts") or {}))),
        "stage_seconds": {
            field: round(float((state.get("seconds") or {}).get(field, 0.0)), 6)
            for field in PREFILTER_TIMING_SECOND_FIELDS
        },
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["row_group_metric_hash"] = stable_json_hash(payload)
    validate_public_no_raw_payload(payload)
    return payload


def _load_prefilter_row_group_metrics(run_path: str | Path) -> dict[str, dict[str, Any]]:
    ready = Path(run_path) / f"{PREFILTER_ROW_GROUP_METRICS_FILENAME}.ready"
    if not ready.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with ready.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"prefilter row-group metrics row {line_number} is not an object")
            validate_public_no_raw_payload(payload)
            if payload.get("artifact_kind") != "twc_chunk_filter_prefilter_row_group_metrics_v1":
                raise ValueError("prefilter row-group metrics kind mismatch")
            if payload.get("row_group_metric_hash") != stable_json_hash(
                {key: value for key, value in payload.items() if key != "row_group_metric_hash"}
            ):
                raise ValueError("prefilter row-group metrics hash mismatch")
            metric_id = str(payload.get("row_group_metric_id") or "")
            _require_sha256(metric_id, "row_group_metric_id")
            if metric_id in rows:
                raise ValueError("prefilter row-group metrics contain duplicate metric ids")
            rows[metric_id] = dict(payload)
    return rows


def _write_prefilter_row_group_metrics(
    run_path: str | Path,
    metrics_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = sorted(
        (dict(row) for row in metrics_by_id.values()),
        key=lambda row: (str(row["source_file_id_hash"]), int(row["row_group_index"])),
    )
    base_path = Path(run_path) / PREFILTER_ROW_GROUP_METRICS_FILENAME
    digest, count = _write_jsonl_ready_streaming(base_path, rows)
    return {
        "prefilter_row_group_metrics_filename": f"{PREFILTER_ROW_GROUP_METRICS_FILENAME}.ready",
        "prefilter_row_group_metrics_row_count": int(count),
        "prefilter_row_group_metrics_sha256": digest,
    }


class ChunkFilteringScorer(Protocol):
    model_id: str

    def score_many(self, texts: list[str], metadata: list[dict[str, Any]]) -> list[float]:
        """Return reject-risk scores in the same order as ``texts``."""


def _hashed_ngram_column_counts(
    text: str,
    *,
    n_features: int = 2**18,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
    sort_output: bool = True,
) -> dict[int, int]:
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    min_word, max_word = word_ngram_range
    if min_word < 1 or max_word < min_word:
        raise ValueError("invalid word_ngram_range")
    min_char, max_char = char_ngram_range
    if min_char < 1 or max_char < min_char:
        raise ValueError("invalid char_ngram_range")

    if text.isascii():
        return _hashed_ngram_column_counts_ascii(
            text,
            n_features=n_features,
            word_ngram_range=word_ngram_range,
            char_ngram_range=char_ngram_range,
            sort_output=sort_output,
        )

    normalized = WHITESPACE_RE.sub(" ", text.casefold()).strip()
    words = WORD_RE.findall(normalized)
    counts: dict[int, int] = {}

    for ngram_size in range(min_word, max_word + 1):
        if len(words) < ngram_size:
            continue
        prefix = f"w{ngram_size}"
        if ngram_size == 1:
            for word in words:
                column = _hashed_ngram_column(prefix, word, n_features=n_features)
                counts[column] = counts.get(column, 0) + 1
            continue
        if ngram_size == 2:
            for index in range(0, len(words) - 1):
                column = _hashed_ngram_column(prefix, words[index] + " " + words[index + 1], n_features=n_features)
                counts[column] = counts.get(column, 0) + 1
            continue
        for index in range(0, len(words) - ngram_size + 1):
            ngram = " ".join(words[index : index + ngram_size])
            column = _hashed_ngram_column(prefix, ngram, n_features=n_features)
            counts[column] = counts.get(column, 0) + 1

    char_view = f" {normalized} "
    if char_view.isascii():
        char_bytes = char_view.encode("ascii")
        for ngram_size in range(min_char, max_char + 1):
            if len(char_bytes) < ngram_size:
                continue
            prefix_bytes = f"c{ngram_size}:".encode("ascii")
            for index in range(0, len(char_bytes) - ngram_size + 1):
                column = _hashed_ngram_column_bytes(
                    prefix_bytes,
                    char_bytes[index : index + ngram_size],
                    n_features=n_features,
                )
                counts[column] = counts.get(column, 0) + 1
    else:
        for ngram_size in range(min_char, max_char + 1):
            if len(char_view) < ngram_size:
                continue
            prefix = f"c{ngram_size}"
            for index in range(0, len(char_view) - ngram_size + 1):
                column = _hashed_ngram_column(prefix, char_view[index : index + ngram_size], n_features=n_features)
                counts[column] = counts.get(column, 0) + 1
    if sort_output:
        return dict(sorted(counts.items()))
    return counts


def _hashed_ngram_column_counts_ascii(
    text: str,
    *,
    n_features: int,
    word_ngram_range: tuple[int, int],
    char_ngram_range: tuple[int, int],
    sort_output: bool,
) -> dict[int, int]:
    normalized = b" ".join(text.encode("ascii").translate(ASCII_LOWER_WHITESPACE_TRANSLATION).split())
    words = ASCII_WORD_BYTES_RE.findall(normalized)
    counts: dict[int, int] = {}

    min_word, max_word = word_ngram_range
    for ngram_size in range(min_word, max_word + 1):
        if len(words) < ngram_size:
            continue
        prefix_bytes = f"w{ngram_size}:".encode("ascii")
        if ngram_size == 1:
            for word in words:
                column = _hashed_ngram_column_bytes(prefix_bytes, word, n_features=n_features)
                counts[column] = counts.get(column, 0) + 1
            continue
        if ngram_size == 2:
            for index in range(0, len(words) - 1):
                column = _hashed_word_bigram_column_bytes(
                    words[index],
                    words[index + 1],
                    n_features=n_features,
                )
                counts[column] = counts.get(column, 0) + 1
            continue
        for index in range(0, len(words) - ngram_size + 1):
            column = _hashed_ngram_column_bytes(
                prefix_bytes,
                b" ".join(words[index : index + ngram_size]),
                n_features=n_features,
            )
            counts[column] = counts.get(column, 0) + 1

    min_char, max_char = char_ngram_range
    char_view = b" " + normalized + b" "
    for ngram_size in range(min_char, max_char + 1):
        if len(char_view) < ngram_size:
            continue
        prefix_bytes = f"c{ngram_size}:".encode("ascii")
        for index in range(0, len(char_view) - ngram_size + 1):
            column = _hashed_ngram_column_bytes(
                prefix_bytes,
                char_view[index : index + ngram_size],
                n_features=n_features,
            )
            counts[column] = counts.get(column, 0) + 1

    if sort_output:
        return dict(sorted(counts.items()))
    return counts


@functools.lru_cache(maxsize=HASHED_NGRAM_COLUMN_CACHE_MAXSIZE)
def _hashed_ngram_column(prefix: str, value: str, *, n_features: int) -> int:
    digest = hashlib.blake2b(f"{prefix}:{value}".encode("utf-8", errors="surrogatepass"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % int(n_features)


@functools.lru_cache(maxsize=HASHED_NGRAM_COLUMN_CACHE_MAXSIZE)
def _hashed_ngram_column_bytes(prefix_with_separator: bytes, value: bytes, *, n_features: int) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(prefix_with_separator)
    hasher.update(value)
    digest = hasher.digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % int(n_features)


@functools.lru_cache(maxsize=HASHED_NGRAM_COLUMN_CACHE_MAXSIZE)
def _hashed_word_bigram_column_bytes(left: bytes, right: bytes, *, n_features: int) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(b"w2:")
    hasher.update(left)
    hasher.update(b" ")
    hasher.update(right)
    digest = hasher.digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % int(n_features)


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
    sparse_model_id: str = CHUNK_SPARSE_MODEL_ID
    window_mil_model_id: str = CHUNK_WINDOW_MIL_MODEL_ID
    window_mil_input_mode: str = WINDOW_INPUT_FULL_EVIDENCE
    window_mil_pooling_mode: str = "topk_mean_logits"
    window_mil_pooling_top_k: int = 3
    window_mil_precision: str = LOCKED_W1_PRECISION
    deterministic_filter_policy_hash: str | None = None
    target_proxy_tokens: int = DEFAULT_TARGET_PROXY_TOKENS
    halo_chars: int = DEFAULT_HALO_CHARS
    canary_high_token_rows: int = DEFAULT_CANARY_HIGH_TOKEN_ROWS
    canary_random_rows: int = DEFAULT_CANARY_RANDOM_ROWS
    accepted_token_target_lower_bound: int = DEFAULT_ACCEPTED_TOKEN_TARGET_LOWER_BOUND
    canary_budget_cap_usd: float = DEFAULT_CANARY_BUDGET_CAP_USD
    full_run_budget_cap_usd: float = DEFAULT_FULL_RUN_BUDGET_CAP_USD


class ChunkSparseRiskScorer:
    model_id = CHUNK_SPARSE_MODEL_ID

    def __init__(self, *, model: Any, threshold: float, n_features: int) -> None:
        if not 0 <= float(threshold) <= 1:
            raise ValueError("sparse threshold must be in [0, 1]")
        if int(n_features) < 2:
            raise ValueError("sparse n_features must be at least 2")
        self.model = model
        self.threshold = float(threshold)
        self.n_features = int(n_features)
        self._fused_log_loss_params = self._resolve_fused_log_loss_params(model)

    @classmethod
    def from_report(cls, report: Mapping[str, Any], model_path: str | Path) -> "ChunkSparseRiskScorer":
        validate_sparse_inference_artifacts(report, model_path)
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("joblib is required to load chunk sparse classifier") from exc
        _install_sklearn_loss_pickle_compat_alias()
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            model = joblib.load(model_path)
        for warning_record in warning_records:
            if type(warning_record.message).__name__ == "InconsistentVersionWarning":
                raise ValueError(
                    "chunk sparse model scikit-learn version mismatch; install the model training runtime before S1 filtering"
                )
        feature_config = report.get("feature_config")
        if not isinstance(feature_config, Mapping):
            raise ValueError("chunk sparse report missing feature_config")
        return cls(
            model=model,
            threshold=float(report["threshold"]),
            n_features=int(feature_config["n_features"]),
        )

    def score_many(self, texts: list[str], metadata: list[dict[str, Any]]) -> list[float]:
        del metadata
        if not texts:
            return []
        fused = self._fused_log_loss_params
        if fused is not None:
            coef, intercept, numpy_module = fused
            score_by_text: dict[str, float] = {}
            scores: list[float] = []
            for text in texts:
                cached = score_by_text.get(text)
                if cached is None:
                    cached = _sigmoid(
                        intercept
                        + _normalized_hashed_ngram_dot(
                            text,
                            n_features=self.n_features,
                            coef=coef,
                            numpy_module=numpy_module,
                        )
                    )
                    score_by_text[text] = cached
                scores.append(cached)
            return scores
        try:
            import numpy
            from scipy.sparse import csr_matrix
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("numpy and scipy are required for chunk sparse inference") from exc
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        for text in texts:
            row_values: list[tuple[int, float]] = []
            for column, count in _hashed_ngram_column_counts(
                text,
                n_features=self.n_features,
                sort_output=False,
            ).items():
                row_values.append((column, _sparse_tf_value(count)))
            if not row_values:
                row_values.append((0, 0.0))
            norm = math.sqrt(sum(value * value for _column, value in row_values)) or 1.0
            for column, value in row_values:
                indices.append(column)
                data.append(value / norm)
            indptr.append(len(indices))
        matrix = csr_matrix(
            (
                numpy.array(data, dtype=numpy.float64),
                numpy.array(indices, dtype=numpy.int32),
                numpy.array(indptr, dtype=numpy.int32),
            ),
            shape=(len(texts), self.n_features),
        )
        probabilities = self.model.predict_proba(matrix)
        return [_runtime_score(score[1]) for score in probabilities]

    def _resolve_fused_log_loss_params(self, model: Any) -> tuple[Any, float, Any] | None:
        """Return exact binary SGD log-loss parameters when direct scoring is safe."""

        if getattr(model, "loss", None) != "log_loss":
            return None
        classes = getattr(model, "classes_", None)
        coef = getattr(model, "coef_", None)
        intercept = getattr(model, "intercept_", None)
        n_features_in = getattr(model, "n_features_in_", self.n_features)
        try:
            import numpy
        except ImportError:  # pragma: no cover - production sparse deps include numpy.
            return None
        if classes is None or coef is None or intercept is None:
            return None
        class_values = list(classes.tolist() if hasattr(classes, "tolist") else classes)
        if class_values != [0, 1]:
            return None
        if getattr(coef, "shape", None) != (1, self.n_features):
            return None
        if getattr(intercept, "shape", None) != (1,):
            return None
        if int(n_features_in) != self.n_features:
            return None
        coef_row = numpy.asarray(coef[0], dtype=numpy.float64)
        intercept_value = float(numpy.asarray(intercept, dtype=numpy.float64)[0])
        if not bool(numpy.isfinite(intercept_value)) or not bool(numpy.isfinite(coef_row).all()):
            return None
        return coef_row, intercept_value, numpy


def _sparse_tf_value(count: int) -> float:
    if int(count) == 1:
        return 1.0
    return 1.0 + math.log(float(count))


def _normalized_hashed_ngram_dot(text: str, *, n_features: int, coef: Any, numpy_module: Any) -> float:
    counts = _hashed_ngram_column_counts(text, n_features=n_features, sort_output=False)
    if not counts:
        return 0.0
    columns = numpy_module.fromiter(counts.keys(), dtype=numpy_module.intp, count=len(counts))
    values = numpy_module.fromiter(
        (_sparse_tf_value(count) for count in counts.values()),
        dtype=numpy_module.float64,
        count=len(counts),
    )
    norm = math.sqrt(float(numpy_module.dot(values, values))) or 1.0
    return float(numpy_module.dot(coef[columns], values)) / norm


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        z = math.exp(-float(logit))
        return 1.0 / (1.0 + z)
    z = math.exp(float(logit))
    return z / (1.0 + z)


class ChunkWindowMilRiskScorer:
    model_id = CHUNK_WINDOW_MIL_MODEL_ID

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        device: Any,
        threshold: float,
        max_length: int,
        stride: int,
        max_windows_per_device_batch: int,
        artifact_max_windows_per_device_batch: int | None = None,
        requested_max_windows_per_device_batch: int | None = None,
        window_variant_config: Mapping[str, Any] | None = None,
        window_mil_precision: str = W1_PRECISION_FP32,
        synchronize_cuda_telemetry: bool = False,
        w1_attn_implementation_requested: str = W1_ATTN_IMPLEMENTATION_SDPA,
        w1_attn_implementation_observed: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.device = device
        self.threshold = float(threshold)
        self.max_length = int(max_length)
        self.stride = int(stride)
        self.artifact_max_windows_per_device_batch = int(
            artifact_max_windows_per_device_batch
            if artifact_max_windows_per_device_batch is not None
            else max_windows_per_device_batch
        )
        self.max_windows_per_device_batch = int(max_windows_per_device_batch)
        self.requested_max_windows_per_device_batch = (
            None
            if requested_max_windows_per_device_batch is None
            else int(requested_max_windows_per_device_batch)
        )
        self.window_mil_precision = normalize_w1_precision_mode(window_mil_precision)
        self.precision_policy = build_w1_precision_policy(self.window_mil_precision)
        self.synchronize_cuda_telemetry = bool(synchronize_cuda_telemetry)
        self.w1_attn_implementation_requested = normalize_w1_attn_implementation(
            w1_attn_implementation_requested
        )
        observed_attn = w1_attn_implementation_observed or self.w1_attn_implementation_requested
        self.w1_attn_implementation_observed = normalize_w1_attn_implementation(observed_attn)
        self.window_variant_config = dict(window_variant_config or {
            "input_mode": WINDOW_INPUT_FULL_EVIDENCE,
            "pooling_mode": "max",
            "pooling_top_k": 1,
            "halo_weight": 0.25,
            "halo_logit_clamp": 2.0,
        })

    @classmethod
    def from_report(
        cls,
        report: Mapping[str, Any],
        model_path: str | Path,
        *,
        device: str = "auto",
        require_cuda: bool = False,
        runtime_max_windows_per_device_batch: int | None = None,
        window_mil_precision: str = W1_PRECISION_FP32,
        synchronize_cuda_telemetry: bool = False,
        w1_attn_implementation: str = W1_ATTN_IMPLEMENTATION_SDPA,
    ) -> "ChunkWindowMilRiskScorer":
        validate_window_mil_inference_artifacts(report, model_path)
        try:
            from twc.classifiers.window_mil import (
                DEFAULT_MODEL_NAME,
                _load_model_bundle,
                _load_torch,
                _resolve_device,
            )
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("Window-MIL inference requires classifier window_mil dependencies") from exc
        torch = _load_torch()
        resolved_device = _resolve_device(torch, device=device, require_cuda=require_cuda)
        checkpoint = torch.load(model_path, map_location=resolved_device)
        if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
            raise ValueError("chunk Window-MIL checkpoint is malformed")
        model_name = str(checkpoint.get("model_name") or report.get("base_model_name") or DEFAULT_MODEL_NAME)
        requested_attn_implementation = normalize_w1_attn_implementation(w1_attn_implementation)
        tokenizer, model, resolved_model_name = _load_model_bundle(
            model_name=model_name,
            model_bundle_factory=None,
            torch=torch,
            attn_implementation=requested_attn_implementation,
        )
        observed_attn_implementation = normalize_w1_attn_implementation(
            getattr(model, "w1_attn_implementation_observed", requested_attn_implementation)
        )
        if str(report.get("base_model_name") or resolved_model_name) != resolved_model_name:
            raise ValueError("chunk Window-MIL base model name drift")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(resolved_device)
        model.eval()
        precision_mode = normalize_w1_precision_mode(window_mil_precision)
        if precision_mode == W1_PRECISION_AMP_BF16 and str(getattr(resolved_device, "type", "")) != "cuda":
            raise RuntimeError("amp_bf16 W1 precision requires a CUDA device")
        apply_w1_precision_policy(torch, precision_mode)
        artifact_max_windows = int(report["max_windows_per_device_batch"])
        requested_max_windows = (
            None
            if runtime_max_windows_per_device_batch is None
            else int(runtime_max_windows_per_device_batch)
        )
        if requested_max_windows is not None and requested_max_windows < 1:
            raise ValueError("runtime max_windows_per_device_batch must be positive")
        effective_max_windows = requested_max_windows if requested_max_windows is not None else artifact_max_windows
        return cls(
            model=model,
            tokenizer=tokenizer,
            torch_module=torch,
            device=resolved_device,
            threshold=float(report["threshold"]),
            max_length=int(report["max_length"]),
            stride=int(report["stride"]),
            max_windows_per_device_batch=effective_max_windows,
            artifact_max_windows_per_device_batch=artifact_max_windows,
            requested_max_windows_per_device_batch=requested_max_windows,
            window_variant_config=report.get("window_variant_config"),
            window_mil_precision=precision_mode,
            synchronize_cuda_telemetry=synchronize_cuda_telemetry,
            w1_attn_implementation_requested=requested_attn_implementation,
            w1_attn_implementation_observed=observed_attn_implementation,
        )

    def score_many(self, texts: list[str], metadata: list[dict[str, Any]]) -> list[float]:
        if not texts:
            self.last_score_many_telemetry = {}
            return []
        from twc.classifiers.window_mil import _score_document_batch, _score_split_halo_document_batch

        telemetry: dict[str, Any] = {
            "synchronize_cuda_timing": bool(self.synchronize_cuda_telemetry),
            "cuda_synchronized_timing": False,
            "w1_precision_policy": self.precision_policy,
            "w1_precision_policy_hash": self.precision_policy["precision_policy_hash"],
        }
        try:
            inference_context = getattr(self.torch, "inference_mode", None)
            with (inference_context() if inference_context is not None else self.torch.no_grad()):
                with w1_precision_context(
                    self.torch,
                    mode=self.window_mil_precision,
                    device_type=str(getattr(self.device, "type", "")),
                ):
                    input_mode = str(self.window_variant_config.get("input_mode") or WINDOW_INPUT_FULL_EVIDENCE)
                    pooling_mode = str(self.window_variant_config.get("pooling_mode") or "max")
                    pooling_top_k = int(self.window_variant_config.get("pooling_top_k", 1))
                    if input_mode == WINDOW_INPUT_SPLIT_HALO_CAPPED:
                        core_texts, halo_texts = _runtime_core_and_halo_texts(texts, metadata)
                        scores = _score_split_halo_document_batch(
                            torch=self.torch,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            core_texts=core_texts,
                            halo_texts=halo_texts,
                            device=self.device,
                            max_length=self.max_length,
                            stride=self.stride,
                            max_windows_per_device_batch=self.max_windows_per_device_batch,
                            pooling_mode=pooling_mode,
                            pooling_top_k=pooling_top_k,
                            halo_weight=float(self.window_variant_config.get("halo_weight", 0.25)),
                            halo_logit_clamp=float(self.window_variant_config.get("halo_logit_clamp", 2.0)),
                            telemetry=telemetry,
                        )
                    else:
                        scores = _score_document_batch(
                            torch=self.torch,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            texts=_runtime_texts_for_input_mode(texts, metadata, input_mode=input_mode),
                            device=self.device,
                            max_length=self.max_length,
                            stride=self.stride,
                            max_windows_per_device_batch=self.max_windows_per_device_batch,
                            pooling_mode=pooling_mode,
                            pooling_top_k=pooling_top_k,
                            telemetry=telemetry,
                        )
        finally:
            self.last_score_many_telemetry = dict(telemetry)
        return [_runtime_score(score) for score in scores]

    def prepare_many(self, texts: list[str], metadata: list[dict[str, Any]], *, pin_memory: bool = False) -> Any:
        if not texts:
            raise ValueError("W1 prepared scoring requires at least one text")
        if len(texts) != len(metadata):
            raise ValueError("W1 prepared scoring metadata count mismatch")
        input_mode = str(self.window_variant_config.get("input_mode") or WINDOW_INPUT_FULL_EVIDENCE)
        if input_mode == WINDOW_INPUT_SPLIT_HALO_CAPPED:
            raise NotImplementedError("bounded W1 prepare pipeline does not support split-halo input mode")
        pooling_mode = str(self.window_variant_config.get("pooling_mode") or "max")
        pooling_top_k = int(self.window_variant_config.get("pooling_top_k", 1))
        from twc.classifiers.window_mil import _prepare_document_logits_batch

        return _prepare_document_logits_batch(
            torch=self.torch,
            tokenizer=self.tokenizer,
            texts=_runtime_texts_for_input_mode(texts, metadata, input_mode=input_mode),
            max_length=self.max_length,
            stride=self.stride,
            pooling_mode=pooling_mode,
            pooling_top_k=pooling_top_k,
            pin_memory=bool(pin_memory),
            metadata=metadata,
        )

    def score_prepared(self, prepared_batch: Any) -> list[float]:
        if prepared_batch is None:
            raise ValueError("prepared W1 batch is required")
        from twc.classifiers.window_mil import _document_logits_from_prepared_batch

        telemetry: dict[str, Any] = {
            "synchronize_cuda_timing": bool(self.synchronize_cuda_telemetry),
            "cuda_synchronized_timing": False,
            "w1_precision_policy": self.precision_policy,
            "w1_precision_policy_hash": self.precision_policy["precision_policy_hash"],
            "prepared_batch_scored": True,
        }
        try:
            inference_context = getattr(self.torch, "inference_mode", None)
            with (inference_context() if inference_context is not None else self.torch.no_grad()):
                with w1_precision_context(
                    self.torch,
                    mode=self.window_mil_precision,
                    device_type=str(getattr(self.device, "type", "")),
                ):
                    doc_logits = _document_logits_from_prepared_batch(
                        torch=self.torch,
                        model=self.model,
                        prepared=prepared_batch,
                        device=self.device,
                        max_windows_per_device_batch=self.max_windows_per_device_batch,
                        telemetry=telemetry,
                    )
                    from twc.classifiers.window_mil import _synchronize_cuda_for_telemetry, _telemetry_add

                    _synchronize_cuda_for_telemetry(torch=self.torch, device=self.device, telemetry=telemetry)
                    postprocess_at = time.monotonic()
                    scores = [float(value) for value in self.torch.sigmoid(doc_logits).detach().cpu().tolist()]
                    _synchronize_cuda_for_telemetry(torch=self.torch, device=self.device, telemetry=telemetry)
                    _telemetry_add(telemetry, "postprocess_seconds", time.monotonic() - postprocess_at)
        finally:
            self.last_score_many_telemetry = dict(telemetry)
        return [_runtime_score(score) for score in scores]


def _runtime_texts_for_input_mode(
    texts: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
    *,
    input_mode: str,
) -> list[str]:
    if input_mode == WINDOW_INPUT_FULL_EVIDENCE:
        return [str(text) for text in texts]
    if input_mode == WINDOW_INPUT_CORE_ONLY:
        core_texts, _halo_texts = _runtime_core_and_halo_texts(texts, metadata)
        return core_texts
    raise ValueError(f"unsupported Window-MIL input mode: {input_mode}")


def _runtime_core_and_halo_texts(
    texts: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    if len(texts) != len(metadata):
        raise ValueError("Window-MIL runtime metadata count mismatch")
    core_texts: list[str] = []
    halo_texts: list[str] = []
    for evidence_text, row_metadata in zip(texts, metadata, strict=True):
        core_start = int(row_metadata.get("core_char_start", -1))
        core_end = int(row_metadata.get("core_char_end", -1))
        if core_start < 0 or core_end <= core_start or core_end > len(evidence_text):
            raise ValueError("Window-MIL runtime metadata missing valid core span")
        core_text = str(evidence_text)[core_start:core_end]
        expected_core_hash = row_metadata.get("core_text_sha256")
        if expected_core_hash is not None and hashlib.sha256(core_text.encode("utf-8", errors="surrogatepass")).hexdigest() != str(expected_core_hash):
            raise ValueError("Window-MIL runtime core hash drift")
        core_texts.append(core_text)
        halo_texts.append((str(evidence_text)[:core_start] + "\n" + str(evidence_text)[core_end:]).strip())
    return core_texts, halo_texts


def build_chunk_filtering_config_payload(config: ChunkFilteringConfig | Mapping[str, Any]) -> dict[str, Any]:
    config = _coerce_config(config)
    precision_mode = normalize_w1_precision_mode(config.window_mil_precision)
    payload = {
        "artifact_kind": CHUNK_FILTERING_CONFIG_KIND,
        "schema_version": CHUNK_FILTERING_CONFIG_SCHEMA_VERSION,
        "filtering_policy_version": CHUNK_FILTERING_POLICY_VERSION,
        "classifier_freeze_hash": _require_sha256(config.classifier_freeze_hash, "classifier_freeze_hash"),
        "selected_policy_report_hash": _require_sha256(
            config.selected_policy_report_hash,
            "selected_policy_report_hash",
        ),
        "selected_policy_hash": _require_sha256(config.selected_policy_hash, "selected_policy_hash"),
        "classifier_manifest_hash": _require_sha256(config.classifier_manifest_hash, "classifier_manifest_hash"),
        "private_evidence_cache_hash": _require_sha256(
            config.private_evidence_cache_hash,
            "private_evidence_cache_hash",
        ),
        "sparse_report_hash": _require_sha256(config.sparse_report_hash, "sparse_report_hash"),
        "sparse_model_file_sha256": _require_sha256(config.sparse_model_file_sha256, "sparse_model_file_sha256"),
        "window_mil_report_hash": _require_sha256(config.window_mil_report_hash, "window_mil_report_hash"),
        "window_mil_model_file_sha256": _require_sha256(
            config.window_mil_model_file_sha256,
            "window_mil_model_file_sha256",
        ),
        "source_metadata_hash": _require_sha256(config.source_metadata_hash, "source_metadata_hash"),
        "selected_eval_deterministic_filter_policy_hash": _require_sha256(
            config.selected_eval_deterministic_filter_policy_hash,
            "selected_eval_deterministic_filter_policy_hash",
        ),
        "sparse_model_id": str(config.sparse_model_id),
        "window_mil_model_id": str(config.window_mil_model_id),
        "window_mil_input_mode": str(config.window_mil_input_mode),
        "window_mil_pooling_mode": str(config.window_mil_pooling_mode),
        "window_mil_pooling_top_k": _positive_int(config.window_mil_pooling_top_k, "window_mil_pooling_top_k"),
        "sparse_threshold": _threshold(config.sparse_threshold, "sparse_threshold"),
        "window_mil_threshold": _threshold(config.window_mil_threshold, "window_mil_threshold"),
        "target_proxy_tokens": _positive_int(config.target_proxy_tokens, "target_proxy_tokens"),
        "halo_chars": _nonnegative_int(config.halo_chars, "halo_chars"),
        "canary_high_token_rows": _positive_int(config.canary_high_token_rows, "canary_high_token_rows"),
        "canary_random_rows": _positive_int(config.canary_random_rows, "canary_random_rows"),
        "canary_physical_row_count": int(config.canary_high_token_rows) + int(config.canary_random_rows),
        "accepted_token_target_lower_bound": _positive_int(
            config.accepted_token_target_lower_bound,
            "accepted_token_target_lower_bound",
        ),
        "canary_budget_cap_usd": _nonnegative_float(config.canary_budget_cap_usd, "canary_budget_cap_usd"),
        "full_run_budget_cap_usd": _nonnegative_float(config.full_run_budget_cap_usd, "full_run_budget_cap_usd"),
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
        payload["deterministic_filter_policy_hash"] = _require_sha256(
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
    validate_public_no_raw_payload(payload)
    if payload.get("artifact_kind") != CHUNK_FILTERING_CONFIG_KIND:
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
        _require_sha256(payload.get(field), field)
    if payload.get("deterministic_filter_policy_hash") is not None:
        _require_sha256(payload.get("deterministic_filter_policy_hash"), "deterministic_filter_policy_hash")
    for field in (
        "target_proxy_tokens",
        "canary_high_token_rows",
        "canary_random_rows",
        "accepted_token_target_lower_bound",
    ):
        _positive_int(payload.get(field), field)
    _nonnegative_int(payload.get("halo_chars"), "halo_chars")
    if payload.get("sparse_model_id") != CHUNK_SPARSE_MODEL_ID:
        raise ValueError("chunk filtering config sparse_model_id mismatch")
    if payload.get("window_mil_model_id") != CHUNK_WINDOW_MIL_MODEL_ID:
        raise ValueError("chunk filtering config window_mil_model_id mismatch")
    if payload.get("window_mil_input_mode") != WINDOW_INPUT_FULL_EVIDENCE:
        raise ValueError("chunk filtering config window_mil_input_mode mismatch")
    if payload.get("window_mil_pooling_mode") != "topk_mean_logits":
        raise ValueError("chunk filtering config window_mil_pooling_mode mismatch")
    if _positive_int(payload.get("window_mil_pooling_top_k"), "window_mil_pooling_top_k") != 3:
        raise ValueError("chunk filtering config window_mil_pooling_top_k mismatch")
    precision_present = (
        payload.get("window_mil_precision") is not None
        or payload.get("window_mil_precision_policy") is not None
    )
    config_precision = normalize_w1_precision_mode(str(payload.get("window_mil_precision") or W1_PRECISION_FP32))
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
    _require_false_flags(payload, PUBLIC_FALSE_FLAGS, "chunk filtering config")
    if payload.get("config_hash") != stable_json_hash({key: value for key, value in payload.items() if key != "config_hash"}):
        raise ValueError("chunk filtering config hash mismatch")


def _validate_prefilter_input_config_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("config_hash") != LEGACY_CPU_PREFILTER_CONFIG_HASH:
        validate_chunk_filtering_config_payload(payload)
        return
    validate_public_no_raw_payload(payload)
    if payload.get("artifact_kind") != CHUNK_FILTERING_CONFIG_KIND:
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
        _require_sha256(payload.get(field), f"legacy prefilter {field}")
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
    _require_false_flags(payload, PUBLIC_FALSE_FLAGS, "legacy prefilter config")


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
    if payload.get("deterministic_filter_policy_hash") != payload.get("selected_eval_deterministic_filter_policy_hash"):
        raise ValueError("locked chunk filtering config selected/executed deterministic_filter_policy_hash mismatch")
    if float(payload.get("sparse_threshold")) != LOCKED_S1_THRESHOLD:
        raise ValueError("locked chunk filtering config sparse_threshold mismatch")
    if float(payload.get("window_mil_threshold")) != LOCKED_W1_THRESHOLD:
        raise ValueError("locked chunk filtering config window_mil_threshold mismatch")
    if normalize_w1_precision_mode(str(payload.get("window_mil_precision") or W1_PRECISION_FP32)) != LOCKED_W1_PRECISION:
        raise ValueError("locked chunk filtering config window_mil_precision mismatch")


def _retarget_accepted_token_lower_bound(
    config_payload: Mapping[str, Any],
    target: int,
) -> dict[str, Any]:
    payload = dict(config_payload)
    payload.pop("config_hash", None)
    payload["accepted_token_target_lower_bound"] = _positive_int(
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
    observed = _require_sha256(observed_config_hash, f"{label} config_hash")
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


def build_chunk_filter_canary_schedule_from_rows(
    *,
    source_schedule: Mapping[str, Any],
    row_metadata: Sequence[Mapping[str, Any]],
    high_token_rows: int,
    random_rows: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if int(high_token_rows) < 1 or int(random_rows) < 1:
        raise ValueError("high_token_rows and random_rows must be positive")
    if source_schedule.get("raw_text_serialized") is not False or source_schedule.get("token_serialized") is not False:
        raise ValueError("source schedule must be no-raw")
    entry_by_group = _source_entries_by_group(source_schedule)
    candidates: list[dict[str, Any]] = []
    seen_physical: set[str] = set()
    duplicate_physical_rows_skipped = 0
    invalid_or_missing_token_count = 0
    for row in row_metadata:
        try:
            source_file_id = str(row.get("source_file_id"))
            row_group_index = int(row.get("row_group_index"))
            offset = int(row.get("row_group_local_row_index", row.get("offset")))
        except (TypeError, ValueError) as exc:
            raise ValueError("row metadata must include source_file_id, row_group_index, and offset") from exc
        entry = entry_by_group.get((source_file_id, row_group_index))
        if entry is None or offset not in {int(value) for value in entry.get("row_group_local_offsets", [])}:
            continue
        physical_key = _physical_row_key(entry, offset)
        if physical_key in seen_physical:
            duplicate_physical_rows_skipped += 1
            continue
        seen_physical.add(physical_key)
        token_count = _metadata_token_count(row)
        if token_count is None:
            invalid_or_missing_token_count += 1
        candidates.append(
            {
                "entry": entry,
                "source_file_id": source_file_id,
                "row_group_index": row_group_index,
                "offset": offset,
                "physical_row_key_hash": physical_key,
                "token_count": token_count,
                "token_count_valid": token_count is not None,
                "random_key": stable_json_hash({"seed": int(seed), "physical_row_key_hash": physical_key}),
            }
        )
    requested = int(high_token_rows) + int(random_rows)
    if len(candidates) < requested:
        raise ValueError("not enough unique physical rows for chunk filter canary")

    selected_keys: set[str] = set()
    high_pool = sorted(
        [candidate for candidate in candidates if candidate["token_count_valid"]],
        key=lambda item: (-int(item["token_count"]), str(item["random_key"])),
    )
    selected: list[dict[str, Any]] = []
    for item in high_pool:
        if len(selected) >= int(high_token_rows):
            break
        selected.append(item | {"slice": "high_token"})
        selected_keys.add(str(item["physical_row_key_hash"]))
    if len(selected) != int(high_token_rows):
        raise ValueError("not enough valid-token rows for high-token canary slice")

    random_pool = sorted(
        [candidate for candidate in candidates if candidate["physical_row_key_hash"] not in selected_keys],
        key=lambda item: str(item["random_key"]),
    )
    for item in random_pool:
        if sum(1 for row in selected if row["slice"] == "random_control") >= int(random_rows):
            break
        selected.append(item | {"slice": "random_control"})
        selected_keys.add(str(item["physical_row_key_hash"]))
    if len(selected) != requested:
        raise ValueError("not enough unique rows for random-control canary slice")

    selected_rows = [_public_selected_row(item, index=index) for index, item in enumerate(selected)]
    entries = _public_schedule_entries_from_selected(selected)
    token_values = [int(item["token_count"]) for item in selected if item["token_count"] is not None]
    schedule = {
        "artifact_kind": CHUNK_FILTER_SCHEDULE_KIND,
        "schema_version": 1,
        "filtering_policy_version": CHUNK_FILTERING_POLICY_VERSION,
        "dataset_hash": stable_json_hash(source_schedule.get("dataset")),
        "source_metadata_hash": _require_sha256(source_schedule.get("source_metadata_hash"), "source_metadata_hash"),
        "seed": int(seed),
        "entries": entries,
        "selected_rows": selected_rows,
        "sampling_policy": {
            "mode": "chunk_filter_canary_high_token_plus_random_control_v1",
            "target_row_count": requested,
            "selected_row_count": len(selected_rows),
            "high_token_row_count": sum(1 for row in selected_rows if row["slice"] == "high_token"),
            "random_row_count": sum(1 for row in selected_rows if row["slice"] == "random_control"),
            "duplicate_physical_rows_skipped": int(duplicate_physical_rows_skipped),
            "invalid_or_missing_token_count": int(invalid_or_missing_token_count),
            "source_file_count_selected": len({row["source_file_id_hash"] for row in selected_rows}),
            "row_group_count_selected": len({(row["source_file_id_hash"], row["row_group_index"]) for row in selected_rows}),
            "token_count_summary": _token_count_summary(token_values),
        },
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
    schedule["schedule_hash"] = stable_json_hash({key: value for key, value in schedule.items() if key != "schedule_hash"})
    validate_chunk_filter_canary_schedule(schedule)
    report = {
        "artifact_kind": CHUNK_FILTER_CANARY_REPORT_KIND,
        "schema_version": 1,
        "status": "ready",
        "schedule_hash": schedule["schedule_hash"],
        "source_metadata_hash": schedule["source_metadata_hash"],
        "target_row_count": requested,
        "selected_row_count": len(selected_rows),
        "high_token_row_count": schedule["sampling_policy"]["high_token_row_count"],
        "random_row_count": schedule["sampling_policy"]["random_row_count"],
        "duplicate_physical_rows_skipped": int(duplicate_physical_rows_skipped),
        "invalid_or_missing_token_count": int(invalid_or_missing_token_count),
        "token_count_summary": schedule["sampling_policy"]["token_count_summary"],
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
    report["report_hash"] = stable_json_hash({key: value for key, value in report.items() if key != "report_hash"})
    validate_chunk_filter_canary_report(report)
    return schedule, report


def validate_chunk_filter_canary_schedule(schedule: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(schedule)
    if schedule.get("artifact_kind") != CHUNK_FILTER_SCHEDULE_KIND:
        raise ValueError("unsupported chunk filter canary schedule kind")
    _require_sha256(schedule.get("source_metadata_hash"), "source_metadata_hash")
    _require_false_flags(schedule, PUBLIC_FALSE_FLAGS, "chunk filter canary schedule")
    entries = list(schedule.get("entries") or [])
    selected_rows = list(schedule.get("selected_rows") or [])
    if not selected_rows:
        raise ValueError("chunk filter canary schedule requires selected rows")
    if any(not isinstance(row, Mapping) for row in selected_rows):
        raise ValueError("chunk filter canary selected rows must be mappings")
    physical_keys = [str(row.get("physical_row_key_hash")) for row in selected_rows]
    for key in physical_keys:
        _require_sha256(key, "physical_row_key_hash")
    if len(physical_keys) != len(set(physical_keys)):
        raise ValueError("chunk filter canary selected rows contain duplicate physical rows")
    slices = Counter(str(row.get("slice")) for row in selected_rows)
    if set(slices) != {"high_token", "random_control"}:
        raise ValueError("chunk filter canary selected rows must contain high_token and random_control slices")
    policy = schedule.get("sampling_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("chunk filter canary schedule missing sampling_policy")
    if int(policy.get("selected_row_count", -1)) != len(selected_rows):
        raise ValueError("chunk filter canary selected_row_count mismatch")
    if int(policy.get("target_row_count", -1)) != len(selected_rows):
        raise ValueError("chunk filter canary target row count mismatch")
    if int(policy.get("high_token_row_count", -1)) != slices["high_token"]:
        raise ValueError("chunk filter canary high-token slice count mismatch")
    if int(policy.get("random_row_count", -1)) != slices["random_control"]:
        raise ValueError("chunk filter canary random slice count mismatch")
    for row in selected_rows:
        _require_sha256(row.get("source_file_id_hash"), "source_file_id_hash")
        _nonnegative_int(row.get("row_group_index"), "row_group_index")
        _nonnegative_int(row.get("row_group_local_row_index"), "row_group_local_row_index")
        if row.get("slice") == "high_token" and row.get("token_count_valid") is not True:
            raise ValueError("high-token canary row must have valid token_count")
        if row.get("token_count_valid") is True:
            _positive_int(row.get("token_count"), "token_count")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("chunk filter schedule entries must be mappings")
        _require_sha256(entry.get("source_file_id_hash"), "entry source_file_id_hash")
        _nonnegative_int(entry.get("row_group_index"), "entry row_group_index")
        offsets = entry.get("row_group_local_offsets")
        if not isinstance(offsets, list) or not offsets:
            raise ValueError("chunk filter schedule entry must contain row offsets")
        if sorted(int(offset) for offset in offsets) != [int(offset) for offset in offsets]:
            raise ValueError("chunk filter schedule entry offsets must be sorted")
        _require_sha256(entry.get("entry_hash"), "entry_hash")
    for value in _walk_string_values(schedule):
        lowered = value.casefold()
        if "http://" in lowered or "https://" in lowered:
            raise ValueError("chunk filter canary schedule contains raw URL-like content")
    if schedule.get("schedule_hash") != stable_json_hash({key: value for key, value in schedule.items() if key != "schedule_hash"}):
        raise ValueError("chunk filter canary schedule hash mismatch")


def validate_chunk_filter_canary_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_FILTER_CANARY_REPORT_KIND:
        raise ValueError("unsupported chunk filter canary report kind")
    if report.get("status") != "ready":
        raise ValueError("chunk filter canary report must be ready")
    _require_sha256(report.get("schedule_hash"), "schedule_hash")
    _require_sha256(report.get("source_metadata_hash"), "source_metadata_hash")
    selected = _positive_int(report.get("selected_row_count"), "selected_row_count")
    if int(report.get("target_row_count", -1)) != selected:
        raise ValueError("chunk filter canary report target row mismatch")
    if int(report.get("high_token_row_count", -1)) + int(report.get("random_row_count", -1)) != selected:
        raise ValueError("chunk filter canary report slice count mismatch")
    _require_false_flags(report, PUBLIC_FALSE_FLAGS, "chunk filter canary report")
    if report.get("report_hash") != stable_json_hash({key: value for key, value in report.items() if key != "report_hash"}):
        raise ValueError("chunk filter canary report hash mismatch")


_PRIVATE_ROW_GROUP_IDENTITY_FIELDS = (
    "dataset_repo_id",
    "config",
    "split",
    "resolved_repo_sha",
    "source_file_id",
    "parquet_url_or_path",
    "parquet_identity",
    "row_group_index",
    "row_group_num_rows",
    "row_selection_mode",
    "freezable_required",
)


def _merge_private_source_schedule_entry(existing: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    for field in _PRIVATE_ROW_GROUP_IDENTITY_FIELDS:
        if existing.get(field) != incoming.get(field):
            raise ValueError("duplicate private source row-group identity drift")
    merged = dict(existing)
    offsets = sorted(
        {
            int(offset)
            for offset in list(existing.get("row_group_local_offsets") or [])
            + list(incoming.get("row_group_local_offsets") or [])
        }
    )
    if not offsets:
        raise ValueError("private source row-group has no offsets")
    merged["row_group_local_offsets"] = offsets
    merged["entry_hash"] = stable_json_hash({key: value for key, value in merged.items() if key != "entry_hash"})
    return merged


def hydrate_chunk_filter_execution_schedule(
    canary_schedule: Mapping[str, Any],
    private_source_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a private fetchable schedule from the no-raw public canary schedule.

    The returned schedule may contain private source locators copied from the
    source schedule. It must not be committed or treated as a public artifact.
    """

    validate_chunk_filter_canary_schedule(canary_schedule)
    source_by_hash_group: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in private_source_schedule.get("entries") or []:
        source_file_id = str(entry.get("source_file_id") or "")
        if not source_file_id:
            continue
        key = (stable_json_hash(source_file_id), int(entry.get("row_group_index")))
        if key in source_by_hash_group:
            source_by_hash_group[key] = _merge_private_source_schedule_entry(source_by_hash_group[key], entry)
            continue
        source_by_hash_group[key] = dict(entry)
    entries: list[dict[str, Any]] = []
    for public_entry in canary_schedule.get("entries") or []:
        key = (str(public_entry["source_file_id_hash"]), int(public_entry["row_group_index"]))
        source_entry = source_by_hash_group.get(key)
        if source_entry is None:
            raise ValueError("private source schedule is missing canary row-group")
        source_offsets = {int(offset) for offset in source_entry.get("row_group_local_offsets") or []}
        requested_offsets = [int(offset) for offset in public_entry.get("row_group_local_offsets") or []]
        if not set(requested_offsets) <= source_offsets:
            raise ValueError("private source schedule is missing canary offsets")
        hydrated = dict(source_entry)
        hydrated["global_schedule_index"] = int(public_entry["global_schedule_index"])
        hydrated["row_selection_mode"] = "explicit_offsets"
        hydrated["row_group_local_offsets"] = requested_offsets
        hydrated["public_canary_entry_hash"] = public_entry["entry_hash"]
        hydrated["entry_hash"] = stable_json_hash(
            {key: value for key, value in hydrated.items() if key not in {"entry_hash"}}
        )
        entries.append(hydrated)
    return {
        "schema_version": 1,
        "artifact_kind": "twc_chunk_filter_private_execution_schedule",
        "schedule_hash": canary_schedule["schedule_hash"],
        "source_metadata_hash": canary_schedule["source_metadata_hash"],
        "entries": entries,
        "private_source_locator_serialized": True,
    }


def _local_parquet_mirror_path(mirror_dir: str | Path, source_file_id: str) -> Path:
    if not source_file_id:
        raise ValueError("local mirror source_file_id is required")
    return Path(mirror_dir) / f"{stable_json_hash(source_file_id)}.parquet"


def _schedule_hash_payload(schedule: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in schedule.items() if key not in {"schedule_hash", "generated_at_utc"}}


def _validate_private_source_schedule_hash(schedule: Mapping[str, Any]) -> str:
    schedule_hash = _require_sha256(schedule.get("schedule_hash"), "schedule_hash")
    current_payload_hash = stable_json_hash({key: value for key, value in schedule.items() if key != "schedule_hash"})
    source_schedule_payload_hash = stable_json_hash(_schedule_hash_payload(schedule))
    if schedule_hash not in {current_payload_hash, source_schedule_payload_hash}:
        raise ValueError("private source schedule hash mismatch")
    return schedule_hash


def _local_parquet_metadata(path: Path) -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional production dependency
        raise RuntimeError("pyarrow is required to validate local Parquet mirror files") from exc
    return pq.ParquetFile(path).metadata


def _validate_local_parquet_entry(path: Path, entry: Mapping[str, Any], metadata_cache: dict[Path, Any]) -> None:
    if not path.exists():
        source_file_id_hash = stable_json_hash(str(entry.get("source_file_id") or ""))
        raise FileNotFoundError(f"local mirror file missing for source_file_id_hash {source_file_id_hash}")
    if path.is_symlink():
        raise ValueError("local mirror file must not be a symlink")
    if not path.is_file():
        raise ValueError("local mirror path must be a file")
    metadata = metadata_cache.get(path)
    if metadata is None:
        metadata = _local_parquet_metadata(path)
        metadata_cache[path] = metadata
    row_group_index = int(entry["row_group_index"])
    if row_group_index < 0 or row_group_index >= int(metadata.num_row_groups):
        raise ValueError("local mirror row_group_index out of bounds")
    expected_num_rows = entry.get("row_group_num_rows")
    if expected_num_rows is not None and int(expected_num_rows) != int(metadata.row_group(row_group_index).num_rows):
        raise ValueError("local mirror row_group_num_rows drift")


def _local_parquet_identity(entry: Mapping[str, Any], path: Path, *, file_sha256: str) -> dict[str, Any]:
    identity = dict(entry.get("parquet_identity") or {})
    for field in ("url", "source_path", "filename"):
        identity.pop(field, None)
    identity.update(
        {
            "source_file_id_hash": stable_json_hash(str(entry["source_file_id"])),
            "local_mirror_file_name_hash": stable_json_hash(path.name),
            "local_mirror_file_size_bytes": int(path.stat().st_size),
            "local_mirror_file_sha256": file_sha256,
            "raw_url_serialized": False,
        }
    )
    return identity


def _local_mirror_file_ready_payload(
    *,
    source_file_id: str,
    parent_schedule_hash: str,
    source_metadata_hash: str,
    local_file: Path,
    entries: Sequence[Mapping[str, Any]],
    file_sha256: str,
) -> dict[str, Any]:
    row_group_indexes = sorted({int(entry["row_group_index"]) for entry in entries})
    offset_count = sum(len(entry.get("row_group_local_offsets") or []) for entry in entries)
    return {
        "artifact_kind": "twc_private_local_parquet_mirror_file_ready_v1",
        "parent_schedule_hash": parent_schedule_hash,
        "source_metadata_hash": source_metadata_hash,
        "source_file_id_hash": stable_json_hash(source_file_id),
        "source_entry_hashes": sorted(str(entry.get("entry_hash") or "") for entry in entries),
        "row_group_indexes": row_group_indexes,
        "requested_offset_count": int(offset_count),
        "local_mirror_file_name_hash": stable_json_hash(local_file.name),
        "local_mirror_file_size_bytes": int(local_file.stat().st_size),
        "local_mirror_file_sha256": file_sha256,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }


def _write_local_mirror_ready(path: Path, payload: Mapping[str, Any]) -> None:
    ready_payload = dict(payload)
    ready_payload["ready_hash"] = stable_json_hash(ready_payload)
    ready_path = path.with_suffix(path.suffix + ".ready.json")
    _write_json_atomic(ready_path, ready_payload)


def _validate_local_mirror_ready(path: Path, expected_payload: Mapping[str, Any]) -> None:
    ready_path = path.with_suffix(path.suffix + ".ready.json")
    if not ready_path.exists():
        raise ValueError("local mirror ready file missing")
    ready = _load_json(ready_path)
    if not isinstance(ready, Mapping):
        raise ValueError("local mirror ready file must be a JSON object")
    validate_public_no_raw_payload(ready)
    if ready.get("artifact_kind") != "twc_private_local_parquet_mirror_file_ready_v1":
        raise ValueError("local mirror ready file kind mismatch")
    if ready.get("ready_hash") != stable_json_hash({key: value for key, value in ready.items() if key != "ready_hash"}):
        raise ValueError("local mirror ready file hash mismatch")
    for key, value in expected_payload.items():
        if ready.get(key) != value:
            raise ValueError("local mirror ready file hash mismatch")


def build_local_parquet_execution_schedule(
    private_source_schedule: Mapping[str, Any],
    mirror_dir: str | Path,
) -> dict[str, Any]:
    """Rewrite a private HTTPS execution schedule to a private local-Parquet schedule.

    The returned schedule is still private because it contains local filesystem
    paths. It strips raw URL/source-path locators that are not needed after the
    mirror has been validated.
    """

    parent_schedule_hash = _validate_private_source_schedule_hash(private_source_schedule)
    source_metadata_hash = _require_sha256(private_source_schedule.get("source_metadata_hash"), "source_metadata_hash")
    entries = private_source_schedule.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("private source schedule entries must be a non-empty list")
    mirror_path = Path(mirror_dir)
    validate_private_source_cache_dir(mirror_path)
    metadata_cache: dict[Path, Any] = {}
    file_sha_cache: dict[Path, str] = {}
    local_entries: list[dict[str, Any]] = []
    local_files: set[Path] = set()
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"private source schedule entry {entry_index} is not an object")
        source_file_id = str(entry.get("source_file_id") or "")
        local_file = _local_parquet_mirror_path(mirror_path, source_file_id)
        _validate_local_parquet_entry(local_file, entry, metadata_cache)
        file_sha256 = file_sha_cache.get(local_file)
        if file_sha256 is None:
            file_sha256 = _sha256_file(local_file)
            file_sha_cache[local_file] = file_sha256
        local_files.add(local_file)
        local_entry = dict(entry)
        local_entry["parquet_url_or_path"] = str(local_file)
        local_entry["parquet_identity"] = _local_parquet_identity(entry, local_file, file_sha256=file_sha256)
        local_entry["entry_hash"] = stable_json_hash(
            {key: value for key, value in local_entry.items() if key != "entry_hash"}
        )
        local_entries.append(local_entry)
    local_schedule = dict(private_source_schedule)
    local_schedule.update(
        {
            "artifact_kind": LOCAL_PARQUET_EXECUTION_SCHEDULE_KIND,
            "parent_schedule_hash": parent_schedule_hash,
            "source_metadata_hash": source_metadata_hash,
            "local_mirror_dir_sha256": _sha256_text(str(mirror_path)),
            "local_mirror_file_count": len(local_files),
            "entries": local_entries,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
    )
    local_schedule["schedule_hash"] = stable_json_hash(_schedule_hash_payload(local_schedule))
    return local_schedule


def _source_metadata_hash(source_metadata_manifest: Mapping[str, Any]) -> str:
    value = source_metadata_manifest.get("source_metadata_hash") or source_metadata_manifest.get("manifest_hash")
    return _require_sha256(value, "source_metadata_hash")


def _source_metadata_files_by_id(source_metadata_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    files = source_metadata_manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source metadata files must be a non-empty list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("source metadata file entries must be mappings")
        file_id = str(item.get("file_id") or "")
        if not file_id:
            raise ValueError("source metadata file entry missing file_id")
        if file_id in by_id:
            raise ValueError("source metadata contains duplicate file_id")
        by_id[file_id] = item
    return by_id


def _source_metadata_footers_by_id(source_metadata_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    footers = source_metadata_manifest.get("parquet_footers")
    if not isinstance(footers, list) or not footers:
        raise ValueError("source metadata parquet_footers must be a non-empty list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in footers:
        if not isinstance(item, Mapping):
            raise ValueError("source metadata footer entries must be mappings")
        file_id = str(item.get("file_id") or "")
        if not file_id:
            raise ValueError("source metadata footer entry missing file_id")
        if file_id in by_id:
            raise ValueError("source metadata contains duplicate footer file_id")
        by_id[file_id] = item
    return by_id


def _local_schedule_entries_by_source_file(local_schedule: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if local_schedule.get("artifact_kind") != LOCAL_PARQUET_EXECUTION_SCHEDULE_KIND:
        raise ValueError("local all-row prefilter requires a local Parquet execution schedule")
    entries = local_schedule.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("local schedule entries must be a non-empty list")
    by_file: dict[str, Mapping[str, Any]] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError("local schedule entries must be mappings")
        file_id = str(item.get("source_file_id") or "")
        if not file_id:
            raise ValueError("local schedule entry missing source_file_id")
        by_file.setdefault(file_id, item)
    return by_file


def build_local_parquet_all_rows_prefilter_plan(
    *,
    local_schedule: Mapping[str, Any],
    source_metadata_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a compact private plan for filtering every local mirrored row.

    The plan has one entry per row group, not one offset per row, so the full
    corpus can be streamed without a 97M-element JSON schedule.
    """

    local_schedule_hash = _require_sha256(local_schedule.get("schedule_hash"), "local_schedule_hash")
    source_metadata_hash = _source_metadata_hash(source_metadata_manifest)
    if local_schedule.get("source_metadata_hash") != source_metadata_hash:
        raise ValueError("local schedule source metadata hash mismatch")
    files_by_id = _source_metadata_files_by_id(source_metadata_manifest)
    footers_by_id = _source_metadata_footers_by_id(source_metadata_manifest)
    local_by_file = _local_schedule_entries_by_source_file(local_schedule)
    metadata_file_ids = set(files_by_id)
    local_file_ids = set(local_by_file)
    if local_file_ids != metadata_file_ids:
        raise ValueError("local mirror coverage does not match source metadata files")
    if set(footers_by_id) != metadata_file_ids:
        raise ValueError("source metadata footer coverage does not match files")

    entries: list[dict[str, Any]] = []
    metadata_cache: dict[Path, Any] = {}
    expected_rows = 0
    for source_file_id in sorted(metadata_file_ids):
        local_prototype = dict(local_by_file[source_file_id])
        footer = footers_by_id[source_file_id]
        row_groups = footer.get("row_groups")
        if not isinstance(row_groups, list):
            raise ValueError("source metadata footer row_groups must be a list")
        for group in sorted(row_groups, key=lambda item: int(item.get("row_group_index", -1))):
            if not isinstance(group, Mapping):
                raise ValueError("source metadata row group entries must be mappings")
            num_rows = int(group.get("num_rows") or 0)
            if num_rows <= 0:
                continue
            entry = dict(local_prototype)
            entry["row_group_index"] = int(group["row_group_index"])
            entry["row_group_num_rows"] = num_rows
            entry["row_group_size_estimate_bytes"] = {
                "compressed": int(group.get("compressed_byte_size") or 0),
                "total": int(group.get("total_byte_size") or 0),
            }
            entry["row_selection_mode"] = "all_rows_in_row_group"
            entry.pop("row_group_local_offsets", None)
            entry["offset_recipe"] = {
                "mode": "all_rows_in_row_group",
                "count": num_rows,
                "modulus": num_rows,
            }
            entry["entry_hash"] = stable_json_hash(
                {key: value for key, value in entry.items() if key != "entry_hash"}
            )
            _validate_local_parquet_entry(Path(str(entry["parquet_url_or_path"])), entry, metadata_cache)
            entries.append(entry)
            expected_rows += num_rows
    if not entries or expected_rows <= 0:
        raise ValueError("local all-row prefilter plan has no rows")
    plan = {
        "artifact_kind": LOCAL_PARQUET_ALL_ROWS_PREFILTER_PLAN_KIND,
        "schema_version": 1,
        "source_metadata_hash": source_metadata_hash,
        "local_schedule_hash": local_schedule_hash,
        "parent_schedule_hash": local_schedule.get("parent_schedule_hash"),
        "local_mirror_file_count": int(local_schedule.get("local_mirror_file_count") or 0),
        "source_file_count": len(metadata_file_ids),
        "row_group_count": len(entries),
        "expected_physical_rows": int(expected_rows),
        "row_selection_mode": "all_rows_in_row_group",
        "entries": entries,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    plan["schedule_hash"] = stable_json_hash({key: value for key, value in plan.items() if key != "schedule_hash"})
    validate_public_no_raw_payload(
        {key: value for key, value in plan.items() if key not in {"entries"}}
    )
    return plan


def _local_parquet_token_count_stats(path: Path, row_group_index: int, *, expected_rows: int) -> dict[str, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional production dependency
        raise RuntimeError("pyarrow is required to build local Parquet token tranche plans") from exc
    try:
        table = pq.ParquetFile(path).read_row_group(int(row_group_index), columns=["token_count"])
    except Exception as exc:
        raise ValueError("token_count column is required for token tranche planning") from exc
    if "token_count" not in table.column_names:
        raise ValueError("token_count column is required for token tranche planning")
    values = table.column("token_count").to_pylist()
    if len(values) != int(expected_rows):
        raise ValueError("token_count row count drift in local Parquet row group")
    counts: list[int] = []
    for value in values:
        if value is None:
            raise ValueError("token_count contains null values")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("token_count values must be integers") from exc
        if parsed < 0:
            raise ValueError("token_count values must be nonnegative")
        counts.append(parsed)
    if not counts:
        raise ValueError("token_count row group is empty")
    return {
        "row_group_source_token_sum": int(sum(counts)),
        "row_group_token_count_min": int(min(counts)),
        "row_group_token_count_max": int(max(counts)),
    }


def _median_int(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("median requires non-empty values")
    ordered = sorted(int(value) for value in values)
    return ordered[len(ordered) // 2]


def _seeded_round_robin_by_file(entries: Sequence[Mapping[str, Any]], *, seed: str, band: str) -> list[dict[str, Any]]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_file.setdefault(str(entry["source_file_id"]), []).append(dict(entry))
    file_order = sorted(
        by_file,
        key=lambda source_file_id: stable_json_hash(
            {
                "mode": TOKEN_TARGET_TRANCHE_MODE,
                "seed": seed,
                "band": band,
                "source_file_id_hash": stable_json_hash(source_file_id),
            }
        ),
    )
    for source_file_id in file_order:
        by_file[source_file_id].sort(
            key=lambda entry: stable_json_hash(
                {
                    "mode": TOKEN_TARGET_TRANCHE_MODE,
                    "seed": seed,
                    "band": band,
                    "entry_hash": entry.get("entry_hash"),
                    "row_group_index": int(entry["row_group_index"]),
                }
            )
        )
    ordered: list[dict[str, Any]] = []
    while file_order:
        next_file_order: list[str] = []
        for source_file_id in file_order:
            pending = by_file[source_file_id]
            if pending:
                ordered.append(pending.pop(0))
            if pending:
                next_file_order.append(source_file_id)
        file_order = next_file_order
    return ordered


def build_local_parquet_token_target_tranche_plan(
    *,
    local_schedule: Mapping[str, Any],
    source_metadata_manifest: Mapping[str, Any],
    estimated_source_token_target: int,
    seed: str,
) -> dict[str, Any]:
    """Build a deterministic non-census row-group tranche plan.

    The plan selects whole row groups using only numeric ``token_count`` values,
    leaving raw text untouched until the normal private prefilter runner.
    """

    target = _positive_int(estimated_source_token_target, "estimated_source_token_target")
    seed_value = str(seed or "")
    if not seed_value:
        raise ValueError("token tranche seed is required")
    population = build_local_parquet_all_rows_prefilter_plan(
        local_schedule=local_schedule,
        source_metadata_manifest=source_metadata_manifest,
    )
    entries: list[dict[str, Any]] = []
    for entry in population["entries"]:
        local_file = Path(str(entry["parquet_url_or_path"]))
        token_stats = _local_parquet_token_count_stats(
            local_file,
            int(entry["row_group_index"]),
            expected_rows=int(entry["row_group_num_rows"]),
        )
        enriched = dict(entry)
        enriched.update(token_stats)
        entries.append(enriched)
    population_token_sum = sum(int(entry["row_group_source_token_sum"]) for entry in entries)
    if target >= population_token_sum:
        raise ValueError("tranche target would select the full population; use the all-row plan")
    median_tokens = _median_int([int(entry["row_group_source_token_sum"]) for entry in entries])
    bands: dict[str, list[dict[str, Any]]] = {"high_token": [], "wide_coverage": []}
    for entry in entries:
        band = "high_token" if int(entry["row_group_source_token_sum"]) >= median_tokens else "wide_coverage"
        enriched = dict(entry)
        enriched["token_target_tranche_band"] = band
        enriched["row_selection_mode"] = "all_rows_in_selected_row_groups"
        enriched["offset_recipe"] = {
            "mode": "all_rows_in_selected_row_groups",
            "count": int(enriched["row_group_num_rows"]),
            "modulus": int(enriched["row_group_num_rows"]),
        }
        enriched["entry_hash"] = stable_json_hash({key: value for key, value in enriched.items() if key != "entry_hash"})
        bands[band].append(enriched)
    ordered_by_band = {
        band: _seeded_round_robin_by_file(items, seed=seed_value, band=band)
        for band, items in bands.items()
    }
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()
    selected_token_sum = 0
    band_cycle = ["high_token", "wide_coverage"]
    while selected_token_sum < target and any(ordered_by_band.values()):
        progressed = False
        for band in band_cycle:
            pending = ordered_by_band[band]
            if not pending:
                continue
            entry = pending.pop(0)
            key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
            if key in selected_keys:
                raise ValueError("token tranche selected duplicate row group")
            selected.append(entry)
            selected_keys.add(key)
            selected_token_sum += int(entry["row_group_source_token_sum"])
            progressed = True
            if selected_token_sum >= target:
                break
        if not progressed:
            break
    if not selected:
        raise ValueError("token tranche selected no row groups")
    if len(selected) == len(entries):
        raise ValueError("tranche target would select the full population; use the all-row plan")
    selected_band_counts = _sorted_counter(Counter(str(entry["token_target_tranche_band"]) for entry in selected))
    sampling_policy = {
        "mode": TOKEN_TARGET_TRANCHE_MODE,
        "seed": seed_value,
        "estimated_source_token_target": target,
        "population_estimated_source_tokens_lower_bound": int(population_token_sum),
        "population_estimated_source_tokens_midpoint": int(population_token_sum),
        "population_estimated_source_tokens_upper_bound": int(population_token_sum),
        "selected_estimated_source_tokens_lower_bound": int(selected_token_sum),
        "selected_estimated_source_tokens_midpoint": int(selected_token_sum),
        "selected_estimated_source_tokens_upper_bound": int(selected_token_sum),
        "population_row_group_count": int(population["row_group_count"]),
        "population_physical_row_count": int(population["expected_physical_rows"]),
        "selected_row_group_count": len(selected),
        "selected_physical_row_count": sum(int(entry["row_group_num_rows"]) for entry in selected),
        "selected_row_group_counts_by_band": selected_band_counts,
        "row_group_selection_unit": "whole_row_group",
        "row_group_ordering": "seeded_file_round_robin_by_band_v1",
    }
    plan = {
        "artifact_kind": LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND,
        "schema_version": 1,
        "source_metadata_hash": population["source_metadata_hash"],
        "local_schedule_hash": population["local_schedule_hash"],
        "parent_schedule_hash": population.get("parent_schedule_hash"),
        "local_mirror_file_count": int(population.get("local_mirror_file_count") or 0),
        "source_file_count": int(population["source_file_count"]),
        "row_group_count": len(selected),
        "expected_physical_rows": int(sampling_policy["selected_physical_row_count"]),
        "row_selection_mode": "all_rows_in_selected_row_groups",
        "population_row_group_count": int(population["row_group_count"]),
        "population_expected_physical_rows": int(population["expected_physical_rows"]),
        "sampling_policy": sampling_policy,
        "sampling_policy_hash": stable_json_hash(sampling_policy),
        "full_corpus_census": False,
        "selected_equals_population": False,
        "claims_scope": "tranche_only",
        "entries": selected,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    plan["schedule_hash"] = stable_json_hash({key: value for key, value in plan.items() if key != "schedule_hash"})
    validate_local_parquet_token_target_tranche_plan(plan)
    return plan


def validate_local_parquet_token_target_tranche_plan(plan: Mapping[str, Any], *, allow_shard: bool = False) -> None:
    if plan.get("artifact_kind") != LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND:
        raise ValueError("token tranche plan kind mismatch")
    if plan.get("full_corpus_census") is not False:
        raise ValueError("token tranche plan must declare full_corpus_census=false")
    if plan.get("selected_equals_population") is not False:
        raise ValueError("token tranche plan must declare selected_equals_population=false")
    if plan.get("claims_scope") != "tranche_only":
        raise ValueError("token tranche plan claims_scope must be tranche_only")
    sampling_policy = plan.get("sampling_policy")
    if not isinstance(sampling_policy, Mapping):
        raise ValueError("token tranche plan missing sampling_policy")
    if sampling_policy.get("mode") != TOKEN_TARGET_TRANCHE_MODE:
        raise ValueError("token tranche plan sampling mode mismatch")
    if plan.get("sampling_policy_hash") != stable_json_hash(sampling_policy):
        raise ValueError("token tranche plan sampling_policy_hash mismatch")
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("token tranche plan entries must be a non-empty list")
    if int(plan.get("row_group_count") or 0) != len(entries):
        raise ValueError("token tranche plan row_group_count mismatch")
    if not allow_shard and int(plan.get("population_row_group_count") or 0) <= len(entries):
        raise ValueError("token tranche plan must not cover the full row-group population")
    expected_rows = sum(int(entry["row_group_num_rows"]) for entry in entries)
    if int(plan.get("expected_physical_rows") or -1) != expected_rows:
        raise ValueError("token tranche plan expected_physical_rows mismatch")
    if not allow_shard and int(sampling_policy.get("selected_physical_row_count") or -1) != expected_rows:
        raise ValueError("token tranche plan selected_physical_row_count mismatch")
    selected_token_sum = sum(int(entry["row_group_source_token_sum"]) for entry in entries)
    if not allow_shard and int(sampling_policy.get("selected_estimated_source_tokens_midpoint") or -1) != selected_token_sum:
        raise ValueError("token tranche plan selected token count mismatch")
    if not allow_shard and selected_token_sum < int(sampling_policy.get("estimated_source_token_target") or 0):
        raise ValueError("token tranche plan does not satisfy token target")
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        key = (str(entry.get("source_file_id") or ""), int(entry.get("row_group_index")))
        if not key[0]:
            raise ValueError("token tranche plan entry missing source_file_id")
        if key in seen:
            raise ValueError("token tranche plan contains duplicate row groups")
        seen.add(key)
        if entry.get("row_selection_mode") != "all_rows_in_selected_row_groups":
            raise ValueError("token tranche plan entry row_selection_mode mismatch")
        for field in ("row_group_source_token_sum", "row_group_token_count_min", "row_group_token_count_max"):
            _nonnegative_int(entry.get(field), field)
    schedule_hash = _require_sha256(plan.get("schedule_hash"), "token_tranche_plan_hash")
    if schedule_hash != stable_json_hash({key: value for key, value in plan.items() if key != "schedule_hash"}):
        raise ValueError("token tranche plan hash mismatch")
    validate_public_no_raw_payload({key: value for key, value in plan.items() if key != "entries"})


def _local_parquet_token_tranche_plan_report(plan: Mapping[str, Any]) -> dict[str, Any]:
    validate_local_parquet_token_target_tranche_plan(plan)
    sampling_policy = dict(plan["sampling_policy"])
    report = {
        "artifact_kind": "twc_local_parquet_token_target_tranche_plan_report_v1",
        "schema_version": 1,
        "status": "ready",
        "plan_hash": plan["schedule_hash"],
        "source_metadata_hash": plan["source_metadata_hash"],
        "local_schedule_hash": plan["local_schedule_hash"],
        "parent_schedule_hash": plan.get("parent_schedule_hash"),
        "claims_scope": "tranche_only",
        "full_corpus_census": False,
        "selected_equals_population": False,
        "row_group_selection_unit": "whole_row_group",
        "row_selection_mode": plan["row_selection_mode"],
        "source_file_count": int(plan["source_file_count"]),
        "selected_row_group_count": int(plan["row_group_count"]),
        "selected_physical_row_count": int(plan["expected_physical_rows"]),
        "population_row_group_count": int(plan["population_row_group_count"]),
        "population_physical_row_count": int(plan["population_expected_physical_rows"]),
        "sampling_policy_hash": plan["sampling_policy_hash"],
        "sampling_mode": sampling_policy["mode"],
        "seed_hash": stable_json_hash(str(sampling_policy["seed"])),
        "estimated_source_token_target": int(sampling_policy["estimated_source_token_target"]),
        "selected_estimated_source_tokens": int(sampling_policy["selected_estimated_source_tokens_midpoint"]),
        "population_estimated_source_tokens": int(sampling_policy["population_estimated_source_tokens_midpoint"]),
        "selected_row_group_counts_by_band": dict(sampling_policy["selected_row_group_counts_by_band"]),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    report["report_hash"] = stable_json_hash({key: value for key, value in report.items() if key != "report_hash"})
    validate_public_no_raw_payload(report)
    return report


def _validate_local_parquet_prefilter_plan(plan: Mapping[str, Any], *, allow_shard: bool = False) -> None:
    kind = plan.get("artifact_kind")
    if kind == LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND:
        validate_local_parquet_token_target_tranche_plan(plan, allow_shard=allow_shard)
        return
    if kind != LOCAL_PARQUET_ALL_ROWS_PREFILTER_PLAN_KIND:
        raise ValueError("local Parquet prefilter plan kind mismatch")
    schedule_hash = _require_sha256(plan.get("schedule_hash"), "local_parquet_prefilter_plan_hash")
    if schedule_hash != stable_json_hash({key: value for key, value in plan.items() if key != "schedule_hash"}):
        raise ValueError("local Parquet all-row prefilter plan hash mismatch")
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("local Parquet prefilter plan entries must be a non-empty list")
    if int(plan.get("row_group_count") or -1) != len(entries):
        raise ValueError("local Parquet prefilter plan row_group_count mismatch")
    expected_rows = sum(int(entry.get("row_group_num_rows") or 0) for entry in entries)
    if int(plan.get("expected_physical_rows") or -1) != expected_rows:
        raise ValueError("local Parquet prefilter plan expected_physical_rows mismatch")
    seen: set[tuple[str, int]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("local Parquet prefilter plan entries must be mappings")
        key = (str(entry.get("source_file_id") or ""), int(entry.get("row_group_index")))
        if not key[0]:
            raise ValueError("local Parquet prefilter plan entry missing source_file_id")
        if key in seen:
            raise ValueError("local Parquet prefilter plan contains duplicate row groups")
        seen.add(key)
        if entry.get("row_selection_mode") != "all_rows_in_row_group":
            raise ValueError("local Parquet prefilter plan row_selection_mode mismatch")
        num_rows = _positive_int(entry.get("row_group_num_rows"), "row_group_num_rows")
        offset_recipe = entry.get("offset_recipe")
        if not isinstance(offset_recipe, Mapping):
            raise ValueError("local Parquet prefilter plan offset_recipe missing")
        if offset_recipe.get("mode") != "all_rows_in_row_group":
            raise ValueError("local Parquet prefilter plan offset_recipe mode mismatch")
        if int(offset_recipe.get("count") or -1) != num_rows or int(offset_recipe.get("modulus") or -1) != num_rows:
            raise ValueError("local Parquet prefilter plan offset_recipe count mismatch")
        if entry.get("entry_hash") != stable_json_hash({key: value for key, value in entry.items() if key != "entry_hash"}):
            raise ValueError("local Parquet prefilter plan entry_hash mismatch")
    _require_false_flags(
        plan,
        (
            "raw_text_serialized",
            "raw_url_serialized",
            "token_serialized",
            "accepted_manifest_created",
            "tokenizer_shards_created",
            "packed_shards_created",
            "training_shards_created",
            "pretraining_started",
        ),
        "local Parquet prefilter plan",
    )
    validate_public_no_raw_payload({key: value for key, value in plan.items() if key != "entries"})


def _shard_local_parquet_all_rows_prefilter_plan(
    plan: Mapping[str, Any],
    *,
    shard_count: int,
    shard_index: int,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
) -> dict[str, Any]:
    return _shard_local_parquet_prefilter_plan(
        plan,
        shard_count=shard_count,
        shard_index=shard_index,
        row_group_sharding_policy=row_group_sharding_policy,
    )


def _validate_row_group_sharding_policy(policy: str | None) -> str:
    value = ROW_GROUP_SHARDING_POLICY_ORDINAL if policy is None else str(policy)
    if value not in ROW_GROUP_SHARDING_POLICIES:
        raise ValueError(
            "row_group_sharding_policy must be ordinal_modulo_v1, weighted_lpt_v1, "
            "or source_local_weighted_lpt_v1"
        )
    return value


def _row_group_sharding_weight(entry: Mapping[str, Any]) -> int:
    value = entry.get("row_group_source_token_sum", entry.get("row_group_num_rows"))
    try:
        weight = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("row group sharding weight must be an integer") from exc
    if weight < 0:
        raise ValueError("row group sharding weight must be nonnegative")
    return max(1, weight)


def _row_group_sharding_source_hash(entry: Mapping[str, Any]) -> str:
    existing = entry.get("source_file_id_hash")
    if isinstance(existing, str) and len(existing) == 64:
        return existing
    return stable_json_hash(str(entry.get("source_file_id") or ""))


def _source_local_lpt_units(entries: Sequence[Mapping[str, Any]], *, shard_count: int) -> list[list[int]]:
    total_weight = sum(_row_group_sharding_weight(entry) for entry in entries)
    target_weight = max(1, math.ceil(total_weight / max(1, int(shard_count))))
    groups: dict[str, list[int]] = {}
    for ordinal, entry in enumerate(entries):
        groups.setdefault(_row_group_sharding_source_hash(entry), []).append(ordinal)

    units: list[list[int]] = []
    for source_hash in sorted(groups):
        ordinals = sorted(
            groups[source_hash],
            key=lambda ordinal: (int(entries[ordinal].get("row_group_index", 0)), ordinal),
        )
        current: list[int] = []
        current_weight = 0
        for ordinal in ordinals:
            weight = _row_group_sharding_weight(entries[ordinal])
            if current and current_weight + weight > target_weight:
                units.append(current)
                current = []
                current_weight = 0
            current.append(ordinal)
            current_weight += weight
        if current:
            units.append(current)

    while len(units) < int(shard_count):
        splittable_indexes = [index for index, unit in enumerate(units) if len(unit) > 1]
        if not splittable_indexes:
            break
        split_index = max(
            splittable_indexes,
            key=lambda index: (
                sum(_row_group_sharding_weight(entries[ordinal]) for ordinal in units[index]),
                len(units[index]),
                -min(units[index]),
            ),
        )
        unit = units.pop(split_index)
        left: list[int] = []
        right: list[int] = []
        half_weight = sum(_row_group_sharding_weight(entries[ordinal]) for ordinal in unit) / 2.0
        running_weight = 0
        for ordinal in unit:
            if left and running_weight >= half_weight:
                right.append(ordinal)
            else:
                left.append(ordinal)
                running_weight += _row_group_sharding_weight(entries[ordinal])
        if not left or not right:
            midpoint = max(1, len(unit) // 2)
            left = unit[:midpoint]
            right = unit[midpoint:]
        units.extend([left, right])
    return units


def _assigned_row_group_ordinals(
    entries: Sequence[Mapping[str, Any]],
    *,
    shard_count: int,
    row_group_sharding_policy: str,
) -> list[list[int]]:
    count = _positive_int(shard_count, "row_group_shard_count")
    policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    if count > len(entries):
        raise ValueError("row_group_shard_count cannot exceed row_group_count")
    if policy == ROW_GROUP_SHARDING_POLICY_ORDINAL:
        return [[ordinal for ordinal in range(len(entries)) if ordinal % count == index] for index in range(count)]

    shard_ordinals: list[list[int]] = [[] for _ in range(count)]
    shard_weights = [0 for _ in range(count)]
    if policy == ROW_GROUP_SHARDING_POLICY_SOURCE_LOCAL_WEIGHTED_LPT:
        units = _source_local_lpt_units(entries, shard_count=count)
    else:
        units = [[ordinal] for ordinal in range(len(entries))]
    ordered_units = sorted(
        units,
        key=lambda unit: (
            -sum(_row_group_sharding_weight(entries[ordinal]) for ordinal in unit),
            _row_group_sharding_source_hash(entries[unit[0]]),
            int(entries[unit[0]].get("row_group_index", 0)),
            unit[0],
        ),
    )
    for unit in ordered_units:
        shard_index = min(range(count), key=lambda index: (shard_weights[index], index))
        shard_ordinals[shard_index].extend(unit)
        shard_weights[shard_index] += sum(_row_group_sharding_weight(entries[ordinal]) for ordinal in unit)
    return [sorted(ordinals) for ordinals in shard_ordinals]


def _shard_local_parquet_prefilter_plan(
    plan: Mapping[str, Any],
    *,
    shard_count: int,
    shard_index: int,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
) -> dict[str, Any]:
    _validate_local_parquet_prefilter_plan(plan)
    count = _positive_int(shard_count, "row_group_shard_count")
    index = int(shard_index)
    if index < 0 or index >= count:
        raise ValueError("row_group_shard_index must be within row_group_shard_count")
    policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("local all-row prefilter plan has no entries to shard")
    assigned_ordinals = _assigned_row_group_ordinals(
        entries,
        shard_count=count,
        row_group_sharding_policy=policy,
    )
    return _shard_local_parquet_prefilter_plan_from_assigned_ordinals(
        plan,
        entries=entries,
        assigned_ordinals=assigned_ordinals,
        shard_count=count,
        shard_index=index,
        row_group_sharding_policy=policy,
    )


def _shard_local_parquet_prefilter_plan_from_assigned_ordinals(
    plan: Mapping[str, Any],
    *,
    entries: Sequence[Mapping[str, Any]],
    assigned_ordinals: Sequence[Sequence[int]],
    shard_count: int,
    shard_index: int,
    row_group_sharding_policy: str,
) -> dict[str, Any]:
    count = _positive_int(shard_count, "row_group_shard_count")
    index = int(shard_index)
    if index < 0 or index >= count:
        raise ValueError("row_group_shard_index must be within row_group_shard_count")
    if len(assigned_ordinals) != count:
        raise ValueError("row group shard assignment count mismatch")
    policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    shard_entries = [dict(entries[ordinal]) for ordinal in assigned_ordinals[index]]
    if not shard_entries:
        raise ValueError("row group shard has no entries")
    plan_hash = _require_sha256(plan.get("schedule_hash"), "local_parquet_prefilter_plan_hash")
    shard_plan = dict(plan)
    shard_plan.update(
        {
            "entries": shard_entries,
            "row_group_count": len(shard_entries),
            "expected_physical_rows": sum(int(entry["row_group_num_rows"]) for entry in shard_entries),
            "row_group_sharding_enabled": count > 1,
            "row_group_shard_count": count,
            "row_group_shard_index": index,
            "local_parquet_prefilter_plan_hash": plan_hash,
            "local_parquet_prefilter_plan_kind": plan["artifact_kind"],
            "source_plan_row_group_count": int(plan["row_group_count"]),
            "source_plan_expected_physical_rows": int(plan["expected_physical_rows"]),
        }
    )
    if plan["artifact_kind"] == LOCAL_PARQUET_ALL_ROWS_PREFILTER_PLAN_KIND:
        shard_plan.update(
            {
                "all_rows_prefilter_plan_hash": plan_hash,
                "all_rows_row_group_count": int(plan["row_group_count"]),
                "all_rows_expected_physical_rows": int(plan["expected_physical_rows"]),
            }
        )
    else:
        shard_plan.update(
            {
                "token_target_tranche_plan_hash": plan_hash,
                "token_target_tranche_row_group_count": int(plan["row_group_count"]),
                "token_target_tranche_expected_physical_rows": int(plan["expected_physical_rows"]),
            }
        )
    if policy != ROW_GROUP_SHARDING_POLICY_ORDINAL:
        shard_plan["row_group_sharding_policy"] = policy
    shard_plan["schedule_hash"] = stable_json_hash({key: value for key, value in shard_plan.items() if key != "schedule_hash"})
    validate_public_no_raw_payload({key: value for key, value in shard_plan.items() if key != "entries"})
    _validate_local_parquet_prefilter_plan(shard_plan, allow_shard=True)
    return shard_plan


def _prefilter_supervisor_manifest_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_hash"}


def _prefilter_supervisor_shard_command_hash(
    *,
    manifest_mode: str,
    shard_count: int,
    shard_index: int,
    row_group_batch_size: int,
    shard_max_rows: int | None,
    min_free_bytes: int,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> str:
    payload = {
        "command": "prefilter-local-parquet-plan",
        "manifest_mode": manifest_mode,
        "row_group_shard_count": int(shard_count),
        "row_group_shard_index": int(shard_index),
        "row_group_batch_size": int(row_group_batch_size),
        "max_rows": None if shard_max_rows is None else int(shard_max_rows),
        "min_free_bytes": int(min_free_bytes),
        "segmentation_validation_mode": _validate_segmentation_validation_mode(segmentation_validation_mode),
        "segmentation_validation_sample_modulus": _segmentation_sample_modulus(
            segmentation_validation_sample_modulus
        ),
        "thread_env": PREFILTER_SUPERVISOR_THREAD_ENV,
    }
    policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    if policy != ROW_GROUP_SHARDING_POLICY_ORDINAL:
        payload["row_group_sharding_policy"] = policy
    return stable_json_hash(payload)


def _prefilter_supervisor_shard_dir(supervisor_run_dir: str | Path, shard: Mapping[str, Any]) -> Path:
    name = str(shard.get("shard_run_dir_name") or "")
    relative = Path(name)
    if not name or relative.is_absolute() or relative.name != name:
        raise ValueError("prefilter supervisor shard_run_dir_name must be a simple relative name")
    return Path(supervisor_run_dir) / relative


def _prefilter_supervisor_run_dir_sha256(path: str | Path) -> str:
    return _sha256_text(str(Path(path).expanduser().resolve(strict=False)))


def _allocate_prefilter_supervisor_diagnostic_row_caps(
    shard_expected_rows: Sequence[int],
    total_max_rows: int | None,
) -> list[int | None]:
    if total_max_rows is None:
        return [None for _ in shard_expected_rows]
    budget = _positive_int(total_max_rows, "max_rows")
    caps: list[int | None] = []
    remaining_budget = budget
    remaining_shards = len(shard_expected_rows)
    for expected_rows in shard_expected_rows:
        if remaining_budget <= 0:
            cap = 0
        else:
            cap = min(int(expected_rows), math.ceil(remaining_budget / max(1, remaining_shards)))
        caps.append(cap)
        remaining_budget -= int(cap)
        remaining_shards -= 1
    if sum(int(cap or 0) for cap in caps) > budget:
        raise ValueError("prefilter supervisor diagnostic row budget allocation overflow")
    return caps


def _expected_prefilter_supervisor_shards(
    *,
    prefilter_plan: Mapping[str, Any],
    shard_count: int,
    row_group_batch_size: int,
    manifest_mode: str,
    max_rows: int | None,
    min_free_bytes: int,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> list[dict[str, Any]]:
    policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    plan = dict(prefilter_plan)
    _validate_local_parquet_prefilter_plan(plan)
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("local all-row prefilter plan has no entries to shard")
    assigned_ordinals = _assigned_row_group_ordinals(
        entries,
        shard_count=shard_count,
        row_group_sharding_policy=policy,
    )
    shard_plans = [
        _shard_local_parquet_prefilter_plan_from_assigned_ordinals(
            plan,
            entries=entries,
            assigned_ordinals=assigned_ordinals,
            shard_count=shard_count,
            shard_index=shard_index,
            row_group_sharding_policy=policy,
        )
        for shard_index in range(shard_count)
    ]
    shard_caps = _allocate_prefilter_supervisor_diagnostic_row_caps(
        [int(shard_plan["expected_physical_rows"]) for shard_plan in shard_plans],
        max_rows if manifest_mode == "diagnostic_sweep" else None,
    )
    shards: list[dict[str, Any]] = []
    for shard_index, (shard_plan, shard_cap) in enumerate(zip(shard_plans, shard_caps, strict=True)):
        shards.append(
            {
                "shard_index": shard_index,
                "shard_run_dir_name": f"shard_{shard_index:05d}",
                "row_group_shard_count": shard_count,
                "expected_row_group_count": int(shard_plan["row_group_count"]),
                "expected_physical_rows": int(
                    shard_plan["expected_physical_rows"]
                    if shard_cap is None
                    else min(shard_cap, int(shard_plan["expected_physical_rows"]))
                ),
                "source_plan_expected_physical_rows": int(shard_plan["expected_physical_rows"]),
                "max_rows": shard_cap,
                "command_hash": _prefilter_supervisor_shard_command_hash(
                    manifest_mode=manifest_mode,
                    shard_count=shard_count,
                    shard_index=shard_index,
                    row_group_batch_size=row_group_batch_size,
                    row_group_sharding_policy=policy,
                    shard_max_rows=shard_cap,
                    min_free_bytes=min_free_bytes,
                    segmentation_validation_mode=segmentation_validation_mode,
                    segmentation_validation_sample_modulus=segmentation_validation_sample_modulus,
                ),
            }
        )
    return shards


def build_prefilter_local_parquet_plan_supervisor_manifest(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    shard_count: int,
    worker_count: int,
    row_group_batch_size: int,
    manifest_mode: str,
    max_rows: int | None = None,
    min_free_bytes: int = 1,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    plan = dict(prefilter_plan)
    _validate_local_parquet_prefilter_plan(plan)
    if plan["source_metadata_hash"] != config_payload["source_metadata_hash"]:
        raise ValueError("prefilter supervisor source metadata hash mismatch")
    mode = str(manifest_mode)
    if mode not in PREFILTER_SUPERVISOR_MODES:
        raise ValueError("prefilter supervisor manifest_mode mismatch")
    count = _positive_int(shard_count, "shard_count")
    workers = _positive_int(worker_count, "worker_count")
    if workers > count:
        raise ValueError("worker_count cannot exceed shard_count")
    batch_size = _positive_int(row_group_batch_size, "row_group_batch_size")
    free_bytes = _positive_int(min_free_bytes, "min_free_bytes")
    shard_policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    segment_mode = _validate_segmentation_validation_mode(segmentation_validation_mode)
    segment_sample_modulus = _segmentation_sample_modulus(segmentation_validation_sample_modulus)
    row_cap = None if max_rows is None else _positive_int(max_rows, "max_rows")
    if mode == "mergeable_tranche" and row_cap is not None:
        raise ValueError("mergeable prefilter supervisor manifests cannot use max_rows")
    merge_allowed = mode == "mergeable_tranche"
    claims_scope = "mergeable_prefilter_tranche" if merge_allowed else "throughput_diagnostic_only"
    plan_hash = _require_sha256(plan.get("schedule_hash"), "prefilter_plan_hash")
    shards = _expected_prefilter_supervisor_shards(
        prefilter_plan=plan,
        shard_count=count,
        row_group_batch_size=batch_size,
        row_group_sharding_policy=shard_policy,
        manifest_mode=mode,
        max_rows=row_cap,
        min_free_bytes=free_bytes,
        segmentation_validation_mode=segment_mode,
        segmentation_validation_sample_modulus=segment_sample_modulus,
    )
    manifest = {
        "artifact_kind": PREFILTER_SUPERVISOR_MANIFEST_KIND,
        "schema_version": PREFILTER_SUPERVISOR_SCHEMA_VERSION,
        "manifest_mode": mode,
        "status": "planned",
        "claims_scope": claims_scope,
        "merge_allowed": merge_allowed,
        "prefilter_plan_hash": plan_hash,
        "prefilter_plan_kind": str(plan["artifact_kind"]),
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "deterministic_filter_policy_hash": config_payload["deterministic_filter_policy_hash"],
        "sparse_report_hash": config_payload["sparse_report_hash"],
        "sparse_model_file_sha256": config_payload["sparse_model_file_sha256"],
        "sparse_threshold": float(config_payload["sparse_threshold"]),
        "supervisor_run_dir_sha256": _prefilter_supervisor_run_dir_sha256(supervisor_run_dir),
        "shard_count": count,
        "worker_count": workers,
        "row_group_batch_size": batch_size,
        "row_group_sharding_policy": shard_policy,
        "segmentation_validation_mode": segment_mode,
        "segmentation_validation_sample_modulus": segment_sample_modulus,
        "max_rows": row_cap,
        "min_free_bytes": free_bytes,
        "expected_row_group_count": int(plan["row_group_count"]),
        "expected_physical_rows": int(plan["expected_physical_rows"]),
        "shards": shards,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    manifest["manifest_hash"] = stable_json_hash(_prefilter_supervisor_manifest_hash_payload(manifest))
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
    )
    return manifest


def validate_prefilter_local_parquet_plan_supervisor_manifest(
    manifest: Mapping[str, Any],
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    supervisor_run_dir: str | Path | None = None,
    expect_merge_allowed: bool | None = None,
) -> None:
    if manifest.get("artifact_kind") != PREFILTER_SUPERVISOR_MANIFEST_KIND:
        raise ValueError("prefilter supervisor manifest kind mismatch")
    if int(manifest.get("schema_version") or 0) != PREFILTER_SUPERVISOR_SCHEMA_VERSION:
        raise ValueError("prefilter supervisor manifest schema_version mismatch")
    if manifest.get("manifest_hash") != stable_json_hash(_prefilter_supervisor_manifest_hash_payload(manifest)):
        raise ValueError("prefilter supervisor manifest hash mismatch")
    _validate_prefilter_input_config_payload(config_payload)
    plan = dict(prefilter_plan)
    _validate_local_parquet_prefilter_plan(plan)
    plan_hash = _require_sha256(plan.get("schedule_hash"), "prefilter_plan_hash")
    if manifest.get("prefilter_plan_hash") != plan_hash:
        raise ValueError("prefilter supervisor plan hash mismatch")
    if supervisor_run_dir is not None and manifest.get("supervisor_run_dir_sha256") != _prefilter_supervisor_run_dir_sha256(
        supervisor_run_dir
    ):
        raise ValueError("prefilter supervisor run dir mismatch")
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "deterministic_filter_policy_hash",
        "sparse_report_hash",
        "sparse_model_file_sha256",
    ):
        if manifest.get(field) != config_payload.get(field):
            raise ValueError(f"prefilter supervisor {field} mismatch")
    if float(manifest.get("sparse_threshold")) != float(config_payload["sparse_threshold"]):
        raise ValueError("prefilter supervisor sparse_threshold mismatch")
    mode = str(manifest.get("manifest_mode") or "")
    if mode not in PREFILTER_SUPERVISOR_MODES:
        raise ValueError("prefilter supervisor manifest_mode mismatch")
    merge_allowed = bool(manifest.get("merge_allowed"))
    if expect_merge_allowed is not None and merge_allowed != bool(expect_merge_allowed):
        raise ValueError("prefilter supervisor merge_allowed mismatch")
    if mode == "mergeable_tranche":
        if not merge_allowed:
            raise ValueError("prefilter supervisor mergeable manifest is not merge_allowed")
        if manifest.get("max_rows") is not None:
            raise ValueError("mergeable prefilter supervisor manifests cannot use max_rows")
        if manifest.get("claims_scope") != "mergeable_prefilter_tranche":
            raise ValueError("prefilter supervisor claims_scope mismatch")
    else:
        if merge_allowed:
            raise ValueError("diagnostic prefilter supervisor manifest is not mergeable")
        if manifest.get("claims_scope") != "throughput_diagnostic_only":
            raise ValueError("prefilter supervisor claims_scope mismatch")
    count = _positive_int(manifest.get("shard_count"), "shard_count")
    workers = _positive_int(manifest.get("worker_count"), "worker_count")
    if workers > count:
        raise ValueError("worker_count cannot exceed shard_count")
    batch_size = _positive_int(manifest.get("row_group_batch_size"), "row_group_batch_size")
    free_bytes = _positive_int(manifest.get("min_free_bytes"), "min_free_bytes")
    shard_policy = _validate_row_group_sharding_policy(
        manifest.get("row_group_sharding_policy", ROW_GROUP_SHARDING_POLICY_ORDINAL)
    )
    segment_mode = _validate_segmentation_validation_mode(str(manifest.get("segmentation_validation_mode", "full")))
    segment_sample_modulus = _segmentation_sample_modulus(
        int(manifest.get("segmentation_validation_sample_modulus", DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS))
    )
    row_cap = None if manifest.get("max_rows") is None else _positive_int(manifest.get("max_rows"), "max_rows")
    if int(manifest.get("expected_row_group_count") or -1) != int(plan["row_group_count"]):
        raise ValueError("prefilter supervisor expected row-group count mismatch")
    if int(manifest.get("expected_physical_rows") or -1) != int(plan["expected_physical_rows"]):
        raise ValueError("prefilter supervisor expected physical rows mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != count:
        raise ValueError("prefilter supervisor shard set mismatch")
    expected_shards_by_index = {
        int(shard["shard_index"]): shard
        for shard in _expected_prefilter_supervisor_shards(
            prefilter_plan=plan,
            shard_count=count,
            row_group_batch_size=batch_size,
            row_group_sharding_policy=shard_policy,
            manifest_mode=mode,
            max_rows=row_cap,
            min_free_bytes=free_bytes,
            segmentation_validation_mode=segment_mode,
            segmentation_validation_sample_modulus=segment_sample_modulus,
        )
    }
    seen: set[int] = set()
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("prefilter supervisor shard must be an object")
        shard_index = int(shard.get("shard_index", -1))
        if shard_index < 0 or shard_index >= count:
            raise ValueError("prefilter supervisor shard index mismatch")
        if shard_index in seen:
            raise ValueError("prefilter supervisor duplicate shard index")
        seen.add(shard_index)
        _prefilter_supervisor_shard_dir("supervisor", shard)
        expected = expected_shards_by_index[shard_index]
        for field, expected_value in expected.items():
            if shard.get(field) != expected_value:
                raise ValueError(f"prefilter supervisor shard {field} mismatch")
    if seen != set(range(count)):
        raise ValueError("prefilter supervisor shard set mismatch")
    if row_cap is not None and sum(int(shard.get("max_rows") or 0) for shard in shards) > row_cap:
        raise ValueError("prefilter supervisor diagnostic row budget exceeds max_rows")
    _require_false_flags(manifest, PUBLIC_FALSE_FLAGS, "prefilter supervisor manifest")
    validate_public_no_raw_payload(manifest)


def build_prefilter_local_parquet_plan_supervisor_shard_commands(
    *,
    manifest: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    config_path: str | Path,
    selected_policy_report_path: str | Path,
    plan_path: str | Path,
    cache_dir: str | Path,
    sparse_report_path: str | Path,
    sparse_model_path: str | Path,
    deterministic_ontology_path: str | Path,
    python_executable: str | Path | None = None,
) -> list[dict[str, Any]]:
    if manifest.get("artifact_kind") != PREFILTER_SUPERVISOR_MANIFEST_KIND:
        raise ValueError("prefilter supervisor manifest kind mismatch")
    if manifest.get("manifest_hash") != stable_json_hash(_prefilter_supervisor_manifest_hash_payload(manifest)):
        raise ValueError("prefilter supervisor manifest hash mismatch")
    _require_false_flags(manifest, PUBLIC_FALSE_FLAGS, "prefilter supervisor manifest")
    validate_public_no_raw_payload(manifest)
    executable = str(python_executable or sys.executable)
    shard_count = _positive_int(manifest.get("shard_count"), "shard_count")
    row_group_batch_size = _positive_int(manifest.get("row_group_batch_size"), "row_group_batch_size")
    min_free_bytes = _positive_int(manifest.get("min_free_bytes"), "min_free_bytes")
    row_group_sharding_policy = _validate_row_group_sharding_policy(
        manifest.get("row_group_sharding_policy", ROW_GROUP_SHARDING_POLICY_ORDINAL)
    )
    segmentation_validation_mode = _validate_segmentation_validation_mode(
        str(manifest.get("segmentation_validation_mode", "full"))
    )
    segmentation_validation_sample_modulus = _segmentation_sample_modulus(
        int(manifest.get("segmentation_validation_sample_modulus", DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS))
    )
    commands: list[dict[str, Any]] = []
    for shard in manifest.get("shards") or []:
        if not isinstance(shard, Mapping):
            raise ValueError("prefilter supervisor shard must be an object")
        shard_index = int(shard["shard_index"])
        shard_max_rows = None if shard.get("max_rows") is None else _nonnegative_int(shard.get("max_rows"), "shard max_rows")
        if shard_max_rows == 0:
            continue
        shard_run_dir = _prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
        command = [
            executable,
            "-m",
            "twc.pretraining.chunk_filtering",
            "prefilter-local-parquet-plan",
            "--config",
            str(config_path),
            "--selected-policy-report",
            str(selected_policy_report_path),
            "--plan",
            str(plan_path),
            "--supervisor-manifest",
            str(Path(supervisor_run_dir) / "prefilter_supervisor_manifest.json"),
            "--run-dir",
            str(shard_run_dir),
            "--cache-dir",
            str(cache_dir),
            "--sparse-report",
            str(sparse_report_path),
            "--sparse-model",
            str(sparse_model_path),
            "--deterministic-ontology",
            str(deterministic_ontology_path),
            "--min-free-bytes",
            str(min_free_bytes),
            "--row-group-batch-size",
            str(row_group_batch_size),
            "--row-group-shard-count",
            str(shard_count),
            "--row-group-shard-index",
            str(shard_index),
            "--row-group-sharding-policy",
            row_group_sharding_policy,
            "--segmentation-validation-mode",
            segmentation_validation_mode,
            "--segmentation-validation-sample-modulus",
            str(segmentation_validation_sample_modulus),
            "--supervisor-manifest-hash",
            str(manifest["manifest_hash"]),
            "--supervisor-command-hash",
            str(shard["command_hash"]),
            "--supervisor-run-dir-sha256",
            str(manifest["supervisor_run_dir_sha256"]),
        ]
        if shard_max_rows is not None:
            command.extend(["--max-rows", str(_positive_int(shard_max_rows, "shard max_rows"))])
        commands.append(
            {
                "shard_index": shard_index,
                "shard_run_dir": shard_run_dir,
                "command": command,
                "command_hash": str(shard["command_hash"]),
                "env_overrides": dict(PREFILTER_SUPERVISOR_THREAD_ENV),
            }
        )
    expected_command_count = sum(
        1
        for shard in manifest.get("shards") or []
        if shard.get("max_rows") is None or int(shard.get("max_rows") or 0) > 0
    )
    if len(commands) != expected_command_count:
        raise ValueError("prefilter supervisor command set mismatch")
    return sorted(commands, key=lambda item: int(item["shard_index"]))


def _validate_prefilter_supervisor_child_binding(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    supervisor_manifest_path: str | Path,
    supervisor_manifest_hash: str | None,
    supervisor_command_hash: str | None,
    supervisor_run_dir_sha256: str | None,
    row_group_shard_count: int,
    row_group_shard_index: int,
    row_group_batch_size: int,
    max_rows: int | None,
    min_free_bytes: int,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> dict[str, Any]:
    claimed_manifest_hash = _require_sha256(supervisor_manifest_hash, "supervisor_manifest_hash")
    claimed_command_hash = _require_sha256(supervisor_command_hash, "supervisor_command_hash")
    claimed_run_dir_hash = _require_sha256(supervisor_run_dir_sha256, "supervisor_run_dir_sha256")
    manifest_path = Path(supervisor_manifest_path)
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("prefilter supervisor manifest must be an object")
    if manifest.get("manifest_hash") != claimed_manifest_hash:
        raise ValueError("prefilter supervisor manifest hash mismatch")
    if manifest.get("supervisor_run_dir_sha256") != claimed_run_dir_hash:
        raise ValueError("prefilter supervisor run dir hash mismatch")
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=prefilter_plan,
        config_payload=config_payload,
        supervisor_run_dir=manifest_path.parent,
    )
    count = _positive_int(row_group_shard_count, "row_group_shard_count")
    index = _nonnegative_int(row_group_shard_index, "row_group_shard_index")
    shard_policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    if count != int(manifest["shard_count"]):
        raise ValueError("prefilter supervisor command hash shard count mismatch")
    if index >= count:
        raise ValueError("prefilter supervisor command hash shard index mismatch")
    shards_by_index = {int(shard["shard_index"]): shard for shard in manifest["shards"]}
    shard = shards_by_index.get(index)
    if shard is None:
        raise ValueError("prefilter supervisor command hash shard missing from manifest")
    expected_command_hash = _prefilter_supervisor_shard_command_hash(
        manifest_mode=str(manifest["manifest_mode"]),
        shard_count=count,
        shard_index=index,
        row_group_batch_size=_positive_int(row_group_batch_size, "row_group_batch_size"),
        shard_max_rows=max_rows,
        min_free_bytes=min_free_bytes,
        row_group_sharding_policy=shard_policy,
        segmentation_validation_mode=segmentation_validation_mode,
        segmentation_validation_sample_modulus=segmentation_validation_sample_modulus,
    )
    if claimed_command_hash != expected_command_hash:
        raise ValueError("prefilter supervisor command hash mismatch for child arguments")
    if shard.get("command_hash") != expected_command_hash:
        raise ValueError("prefilter supervisor command hash mismatch for manifest shard")
    return {
        "supervisor_manifest_hash": claimed_manifest_hash,
        "supervisor_command_hash": claimed_command_hash,
        "supervisor_run_dir_sha256": claimed_run_dir_hash,
    }


def _append_prefilter_supervisor_event(path: Path, payload: Mapping[str, Any]) -> None:
    event = dict(payload)
    event.setdefault("schema_version", 1)
    event.setdefault("timestamp_unix", round(time.time(), 6))
    validate_public_no_raw_payload(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
    _chmod_private_file(path)


def _prefilter_supervisor_shard_terminal_status(shard_run_dir: str | Path) -> str:
    receipt_path = Path(shard_run_dir) / "prefilter_cpu_receipt.json"
    if not receipt_path.exists():
        return "missing_receipt"
    try:
        receipt = _load_json(receipt_path)
    except (OSError, json.JSONDecodeError):
        return "invalid_receipt"
    if not isinstance(receipt, Mapping):
        return "invalid_receipt"
    status = str(receipt.get("status") or "")
    if status in {"prefilter_shard_ready_for_merge", "compact_fresh_import"}:
        return "ready"
    if status == "stopped":
        return "stopped"
    if status == "prefilter_blocked_errors":
        return "blocked_errors"
    return status or "unknown_receipt_status"


def _w1_start_prefilter_expected_status(receipt: Mapping[str, Any]) -> str:
    if _is_w1_compact_prefilter_receipt(receipt):
        return "compact_fresh_import"
    return "prefilter_shard_ready_for_merge"


def _w1_start_prefilter_expected_status_for_run_dir(run_dir: str | Path) -> str:
    receipt = _load_json(Path(run_dir) / "prefilter_cpu_receipt.json")
    if not isinstance(receipt, Mapping):
        raise ValueError("W1 start prefilter receipt must be an object")
    return _w1_start_prefilter_expected_status(receipt)


def _validate_prefilter_supervisor_ready_shard(
    *,
    shard_run_dir: str | Path,
    shard: Mapping[str, Any],
    manifest: Mapping[str, Any],
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    seen_survivor_candidate_ids: set[str] | None = None,
    seen_private_evidence_ids: set[str] | None = None,
    audit_sqlite_state: bool = True,
) -> dict[str, Any]:
    shard_path = Path(shard_run_dir)
    validate_private_filtering_run_dir(shard_path)
    receipt = _load_json(shard_path / "prefilter_cpu_receipt.json")
    if not isinstance(receipt, Mapping):
        raise ValueError("prefilter supervisor shard receipt must be an object")
    expected_status = _w1_start_prefilter_expected_status(receipt)
    compact_receipt = expected_status == "compact_fresh_import"
    prefilter_config_payload = (
        _prefilter_input_config_payload_for_hash(
            config_payload,
            receipt.get("config_hash"),
            label="compact prefilter supervisor shard receipt",
        )
        if compact_receipt
        else config_payload
    )
    proof_ready = _prefilter_receipt_has_survivor_queue_id_proof(receipt)
    id_index_ready = _prefilter_receipt_has_survivor_queue_id_index_proof(receipt)
    needs_cross_shard_id_scan = seen_survivor_candidate_ids is not None or seen_private_evidence_ids is not None
    queue_rows: list[dict[str, Any]] | None = None
    if compact_receipt:
        if not audit_sqlite_state and id_index_ready:
            queue_row_count = _validate_survivor_queue_id_index_ready_file(
                shard_path / "w1_survivor_queue.jsonl",
                allow_empty=True,
                expected_hash=receipt.get("survivor_queue_id_index_hash"),
                expected_row_count=int(receipt.get("survivor_queue_id_index_row_count", receipt.get("survivor_count", 0))),
                expected_candidate_id_set_hash=receipt.get("survivor_candidate_id_set_hash"),
                expected_private_evidence_id_set_hash=receipt.get("survivor_private_evidence_id_set_hash"),
                seen_survivor_candidate_ids=seen_survivor_candidate_ids,
                seen_private_evidence_ids=seen_private_evidence_ids,
            )
        else:
            queue_row_count = _validate_survivor_queue_ready_file(
                shard_path / "w1_survivor_queue.jsonl",
                prefilter_config_payload,
                allow_empty=True,
                seen_survivor_candidate_ids=seen_survivor_candidate_ids,
                seen_private_evidence_ids=seen_private_evidence_ids,
            )
            queue_rows = _load_ready_jsonl(shard_path / "w1_survivor_queue.jsonl", "prefilter survivor queue")
            if len(queue_rows) != queue_row_count:
                raise ValueError("compact start state survivor queue count changed during validation")
    elif audit_sqlite_state or not proof_ready:
        queue_row_count = _validate_survivor_queue_ready_file(
            shard_path / "w1_survivor_queue.jsonl",
            config_payload,
            allow_empty=True,
            seen_survivor_candidate_ids=seen_survivor_candidate_ids,
            seen_private_evidence_ids=seen_private_evidence_ids,
        )
    elif needs_cross_shard_id_scan and id_index_ready:
        queue_row_count = _validate_survivor_queue_id_index_ready_file(
            shard_path / "w1_survivor_queue.jsonl",
            allow_empty=True,
            expected_hash=receipt.get("survivor_queue_id_index_hash"),
            expected_row_count=int(receipt.get("survivor_queue_id_index_row_count", receipt.get("survivor_count", 0))),
            expected_candidate_id_set_hash=receipt.get("survivor_candidate_id_set_hash"),
            expected_private_evidence_id_set_hash=receipt.get("survivor_private_evidence_id_set_hash"),
            seen_survivor_candidate_ids=seen_survivor_candidate_ids,
            seen_private_evidence_ids=seen_private_evidence_ids,
        )
    elif needs_cross_shard_id_scan:
        queue_row_count = _validate_survivor_queue_ready_file(
            shard_path / "w1_survivor_queue.jsonl",
            prefilter_config_payload,
            allow_empty=True,
            seen_survivor_candidate_ids=seen_survivor_candidate_ids,
            seen_private_evidence_ids=seen_private_evidence_ids,
        )
    else:
        queue_row_count = int(receipt.get("survivor_queue_row_count", receipt.get("survivor_count", 0)) or 0)
    if (compact_receipt and queue_rows is not None) or audit_sqlite_state or not proof_ready:
        conn = _open_stage_state(shard_path, prefilter_config_payload)
    else:
        conn = None
    try:
        if conn is None:
            if compact_receipt:
                _validate_w1_compact_receipt_proof_payload(
                    receipt,
                    shard_path,
                    prefilter_config_payload,
                    queue_row_count=queue_row_count,
                    expected_status=expected_status,
                    require_empty_w1_scores=False,
                )
            else:
                _validate_prefilter_ready_receipt_payload(
                    receipt,
                    shard_path,
                    prefilter_config_payload,
                    queue_row_count=queue_row_count,
                    expected_status=expected_status,
                )
        else:
            _validate_prefilter_ready_receipt(
                conn,
                shard_path,
                prefilter_config_payload,
                queue_rows=queue_rows,
                queue_row_count=None if queue_rows is not None else queue_row_count,
                expected_status=expected_status,
            )
        shard_index = int(shard["shard_index"])
        shard_count = int(manifest["shard_count"])
        shard_policy = _validate_row_group_sharding_policy(
            manifest.get("row_group_sharding_policy", ROW_GROUP_SHARDING_POLICY_ORDINAL)
        )
        expected_shard_plan = _shard_local_parquet_prefilter_plan(
            prefilter_plan,
            shard_count=shard_count,
            shard_index=shard_index,
            row_group_sharding_policy=shard_policy,
        )
        expected_fields = _local_parquet_prefilter_plan_receipt_fields(expected_shard_plan)
        for field, expected_value in expected_fields.items():
            if receipt.get(field) != expected_value:
                raise ValueError(f"prefilter supervisor shard receipt {field} mismatch")
        if receipt.get("input_schedule_hash") != expected_shard_plan["schedule_hash"]:
            raise ValueError("prefilter supervisor shard input schedule hash mismatch")
        if int(receipt.get("terminal_rows", -1)) != int(shard["expected_physical_rows"]):
            raise ValueError("prefilter supervisor shard terminal row count mismatch")
        if receipt.get("supervisor_manifest_hash") != manifest["manifest_hash"]:
            raise ValueError("prefilter supervisor shard supervisor manifest hash mismatch")
        if receipt.get("supervisor_command_hash") != shard["command_hash"]:
            raise ValueError("prefilter supervisor shard command hash mismatch")
        if receipt.get("supervisor_run_dir_sha256") != manifest["supervisor_run_dir_sha256"]:
            raise ValueError("prefilter supervisor shard run dir hash mismatch")
        return dict(receipt)
    finally:
        if conn is not None:
            conn.close()


def _request_prefilter_supervisor_shard_stop(shard_run_dir: str | Path) -> None:
    stop_path = Path(shard_run_dir) / "STOP_REQUESTED"
    stop_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path.write_text("prefilter supervisor stop requested\n", encoding="utf-8")
    _chmod_private_file(stop_path)


def _clear_prefilter_supervisor_owned_shard_stop(shard_run_dir: str | Path) -> None:
    stop_path = Path(shard_run_dir) / "STOP_REQUESTED"
    if not stop_path.exists():
        return
    try:
        text = stop_path.read_text(encoding="utf-8")
    except OSError:
        return
    if text == "prefilter supervisor stop requested\n":
        stop_path.unlink()


def _prefilter_supervisor_summary(
    *,
    manifest: Mapping[str, Any],
    status: str,
    elapsed_seconds: float,
    shard_status_counts: Mapping[str, int],
    launched_shards: int,
    skipped_ready_shards: int,
    merge_receipt_hash: str | None = None,
) -> dict[str, Any]:
    summary = {
        "artifact_kind": PREFILTER_SUPERVISOR_SUMMARY_KIND,
        "schema_version": 1,
        "status": status,
        "manifest_hash": manifest["manifest_hash"],
        "manifest_mode": manifest["manifest_mode"],
        "merge_allowed": bool(manifest["merge_allowed"]),
        "claims_scope": manifest["claims_scope"],
        "shard_count": int(manifest["shard_count"]),
        "worker_count": int(manifest["worker_count"]),
        "row_group_batch_size": int(manifest["row_group_batch_size"]),
        "row_group_sharding_policy": _validate_row_group_sharding_policy(
            manifest.get("row_group_sharding_policy", ROW_GROUP_SHARDING_POLICY_ORDINAL)
        ),
        "segmentation_validation_mode": str(manifest.get("segmentation_validation_mode", "full")),
        "segmentation_validation_sample_modulus": int(
            manifest.get("segmentation_validation_sample_modulus", DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS)
        ),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "launched_shards": int(launched_shards),
        "skipped_ready_shards": int(skipped_ready_shards),
        "shard_status_counts": {str(key): int(value) for key, value in sorted(shard_status_counts.items())},
        "merge_receipt_hash": merge_receipt_hash,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    summary["summary_hash"] = stable_json_hash({key: value for key, value in summary.items() if key != "summary_hash"})
    validate_public_no_raw_payload(summary)
    return summary


def run_prefilter_local_parquet_plan_supervisor(
    *,
    config_path: str | Path,
    selected_policy_report_path: str | Path,
    plan_path: str | Path,
    supervisor_run_dir: str | Path,
    cache_dir: str | Path,
    sparse_report_path: str | Path,
    sparse_model_path: str | Path,
    deterministic_ontology_path: str | Path,
    shard_count: int,
    worker_count: int,
    row_group_batch_size: int,
    manifest_mode: str,
    max_rows: int | None = None,
    min_free_bytes: int = 1,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
    merge_run_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
    poll_interval_seconds: float = 1.0,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> dict[str, Any]:
    config_payload = build_chunk_filtering_config_payload(_load_yaml(config_path))
    validate_locked_chunk_filtering_config_payload(config_payload)
    _load_and_validate_selected_policy_report(selected_policy_report_path, config_payload, required=True)
    plan = _load_json(plan_path)
    supervisor_path = Path(supervisor_run_dir)
    _ensure_private_dir(supervisor_path)
    validate_private_source_cache_dir(cache_dir)
    _check_disk_watermark([supervisor_path, Path(cache_dir)], min_free_bytes=min_free_bytes)
    manifest = build_prefilter_local_parquet_plan_supervisor_manifest(
        prefilter_plan=plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_path,
        shard_count=shard_count,
        worker_count=worker_count,
        row_group_batch_size=row_group_batch_size,
        manifest_mode=manifest_mode,
        max_rows=max_rows,
        min_free_bytes=min_free_bytes,
        row_group_sharding_policy=row_group_sharding_policy,
        segmentation_validation_mode=segmentation_validation_mode,
        segmentation_validation_sample_modulus=segmentation_validation_sample_modulus,
    )
    manifest_path = supervisor_path / "prefilter_supervisor_manifest.json"
    if manifest_path.exists():
        existing = _load_json(manifest_path)
        if existing != manifest:
            raise ValueError("prefilter supervisor manifest drift; resume requires the exact original manifest")
        manifest = existing
    else:
        _write_json_atomic(manifest_path, manifest)
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_path,
    )
    existing_merge_receipt: dict[str, Any] | None = None
    if merge_run_dir is not None:
        if not bool(manifest["merge_allowed"]):
            raise ValueError("prefilter supervisor manifest is not mergeable")
        merge_path = Path(merge_run_dir)
        if merge_path.exists() and any(merge_path.iterdir()):
            try:
                existing_merge_receipt = _validate_existing_prefilter_supervisor_merge(
                    prefilter_plan=plan,
                    config_payload=config_payload,
                    manifest=manifest,
                    supervisor_run_dir=supervisor_path,
                    run_dir=merge_path,
                )
            except ValueError as exc:
                raise ValueError(
                    "prefilter supervisor merge run dir contains incomplete or invalid existing merge"
                ) from exc
        else:
            _check_disk_watermark([merge_path], min_free_bytes=min_free_bytes)
    commands = build_prefilter_local_parquet_plan_supervisor_shard_commands(
        manifest=manifest,
        supervisor_run_dir=supervisor_path,
        config_path=config_path,
        selected_policy_report_path=selected_policy_report_path,
        plan_path=plan_path,
        cache_dir=cache_dir,
        sparse_report_path=sparse_report_path,
        sparse_model_path=sparse_model_path,
        deterministic_ontology_path=deterministic_ontology_path,
        python_executable=python_executable,
    )
    events_path = supervisor_path / "prefilter_supervisor_events.jsonl"
    summary_path = supervisor_path / "prefilter_supervisor_summary.json"
    stop_path = supervisor_path / "STOP_REQUESTED"
    if not stop_path.exists():
        for spec in commands:
            _clear_prefilter_supervisor_owned_shard_stop(spec["shard_run_dir"])
    start = time.monotonic()
    launched = 0
    skipped = 0
    shard_statuses: dict[int, str] = {}
    pending: list[dict[str, Any]] = []
    shards_by_index = {int(shard["shard_index"]): shard for shard in manifest["shards"]}
    command_shard_indexes = {int(spec["shard_index"]) for spec in commands}
    for shard in shards_by_index.values():
        shard_index = int(shard["shard_index"])
        if shard_index not in command_shard_indexes:
            shard_statuses[shard_index] = "not_launched_zero_budget"
            _append_prefilter_supervisor_event(
                events_path,
                {
                    "event_type": "shard_skipped_zero_budget",
                    "manifest_hash": manifest["manifest_hash"],
                    "shard_index": shard_index,
                    "command_hash": str(shard["command_hash"]),
                },
            )
    for spec in commands:
        status = _prefilter_supervisor_shard_terminal_status(spec["shard_run_dir"])
        if status == "ready":
            shard = shards_by_index[int(spec["shard_index"])]
            _validate_prefilter_supervisor_ready_shard(
                shard_run_dir=spec["shard_run_dir"],
                shard=shard,
                manifest=manifest,
                prefilter_plan=plan,
                config_payload=config_payload,
            )
            skipped += 1
            shard_statuses[int(spec["shard_index"])] = "ready"
            _append_prefilter_supervisor_event(
                events_path,
                {
                    "event_type": "shard_skipped_ready",
                    "manifest_hash": manifest["manifest_hash"],
                    "shard_index": int(spec["shard_index"]),
                    "command_hash": spec["command_hash"],
                },
            )
        else:
            pending.append(spec)
    running: dict[subprocess.Popen[Any], tuple[dict[str, Any], Any]] = {}
    stop_requested = False
    failure_seen = False
    code_root = Path(__file__).resolve().parents[2]
    try:
        while pending or running:
            if stop_path.exists():
                stop_requested = True
                for spec, _handle in running.values():
                    _request_prefilter_supervisor_shard_stop(spec["shard_run_dir"])
            while (
                not stop_requested
                and not failure_seen
                and pending
                and len(running) < int(manifest["worker_count"])
            ):
                spec = pending.pop(0)
                shard_dir = Path(spec["shard_run_dir"])
                _ensure_private_dir(shard_dir)
                log_path = shard_dir / "prefilter_shard_stdout_stderr.log"
                log_handle = log_path.open("a", encoding="utf-8")
                _chmod_private_file(log_path)
                env = os.environ.copy()
                env.update({str(key): str(value) for key, value in spec["env_overrides"].items()})
                try:
                    process = subprocess.Popen(
                        spec["command"],
                        cwd=code_root,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        text=True,
                    )
                except BaseException:
                    log_handle.close()
                    shard_statuses[int(spec["shard_index"])] = "launch_failed"
                    raise
                running[process] = (spec, log_handle)
                launched += 1
                _append_prefilter_supervisor_event(
                    events_path,
                    {
                        "event_type": "shard_launched",
                        "manifest_hash": manifest["manifest_hash"],
                        "shard_index": int(spec["shard_index"]),
                        "command_hash": spec["command_hash"],
                    },
                )
            completed: list[subprocess.Popen[Any]] = []
            for process, (spec, log_handle) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                completed.append(process)
                log_handle.close()
                status = _prefilter_supervisor_shard_terminal_status(spec["shard_run_dir"])
                if status == "ready":
                    try:
                        shard = shards_by_index[int(spec["shard_index"])]
                        _validate_prefilter_supervisor_ready_shard(
                            shard_run_dir=spec["shard_run_dir"],
                            shard=shard,
                            manifest=manifest,
                            prefilter_plan=plan,
                            config_payload=config_payload,
                        )
                    except ValueError:
                        status = "failed"
                if int(return_code) != 0 and status != "ready":
                    status = "failed"
                if status == "stopped":
                    stop_requested = True
                elif status != "ready":
                    failure_seen = True
                shard_statuses[int(spec["shard_index"])] = status
                _append_prefilter_supervisor_event(
                    events_path,
                    {
                        "event_type": "shard_finished",
                        "manifest_hash": manifest["manifest_hash"],
                        "shard_index": int(spec["shard_index"]),
                        "command_hash": spec["command_hash"],
                        "return_code": int(return_code),
                        "shard_status": status,
                    },
                )
            for process in completed:
                running.pop(process, None)
            if failure_seen:
                for spec, _handle in running.values():
                    _request_prefilter_supervisor_shard_stop(spec["shard_run_dir"])
            if running and not completed:
                time.sleep(max(0.05, float(poll_interval_seconds)))
            if stop_requested and not running:
                break
        if stop_requested:
            for spec in pending:
                shard_statuses[int(spec["shard_index"])] = "not_launched_stop_requested"
            status = "supervisor_stopped"
            merge_receipt_hash = None
        elif failure_seen:
            for spec in pending:
                shard_statuses[int(spec["shard_index"])] = "not_launched_after_failure"
            status = "supervisor_failed"
            merge_receipt_hash = None
        else:
            for spec in commands:
                shard_index = int(spec["shard_index"])
                if shard_index not in shard_statuses:
                    status = _prefilter_supervisor_shard_terminal_status(spec["shard_run_dir"])
                    if status == "ready":
                        shard = shards_by_index[shard_index]
                        _validate_prefilter_supervisor_ready_shard(
                            shard_run_dir=spec["shard_run_dir"],
                            shard=shard,
                            manifest=manifest,
                            prefilter_plan=plan,
                            config_payload=config_payload,
                        )
                    shard_statuses[shard_index] = status
            successful_statuses = {"ready", "not_launched_zero_budget"}
            if all(value in successful_statuses for value in shard_statuses.values()):
                if manifest["manifest_mode"] == "diagnostic_sweep":
                    status = "supervisor_diagnostic_complete"
                    merge_receipt_hash = None
                elif merge_run_dir is not None:
                    if existing_merge_receipt is None:
                        merged = merge_prefilter_local_parquet_plan_supervisor_manifest(
                            prefilter_plan=plan,
                            config_payload=config_payload,
                            manifest=manifest,
                            supervisor_run_dir=supervisor_path,
                            run_dir=merge_run_dir,
                            min_free_bytes=min_free_bytes,
                        )
                    else:
                        merged = existing_merge_receipt
                    status = "supervisor_merged"
                    merge_receipt_hash = str(merged["stage_receipt_hash"])
                else:
                    status = "supervisor_ready_for_merge"
                    merge_receipt_hash = None
            else:
                status = "supervisor_failed"
                merge_receipt_hash = None
        counts = Counter(shard_statuses.values())
        summary = _prefilter_supervisor_summary(
            manifest=manifest,
            status=status,
            elapsed_seconds=time.monotonic() - start,
            shard_status_counts=counts,
            launched_shards=launched,
            skipped_ready_shards=skipped,
            merge_receipt_hash=merge_receipt_hash,
        )
        _write_json_atomic(summary_path, summary)
        return summary
    except BaseException:
        for spec in pending:
            shard_statuses[int(spec["shard_index"])] = "not_launched_after_exception"
        for process, (spec, log_handle) in list(running.items()):
            _request_prefilter_supervisor_shard_stop(spec["shard_run_dir"])
            shard_statuses[int(spec["shard_index"])] = "stop_requested_after_exception"
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass
            try:
                log_handle.close()
            except Exception:
                pass
        counts = Counter(shard_statuses.values())
        summary = _prefilter_supervisor_summary(
            manifest=manifest,
            status="supervisor_failed_exception",
            elapsed_seconds=time.monotonic() - start,
            shard_status_counts=counts,
            launched_shards=launched,
            skipped_ready_shards=skipped,
            merge_receipt_hash=None,
        )
        try:
            _write_json_atomic(summary_path, summary)
            _append_prefilter_supervisor_event(
                events_path,
                {
                    "event_type": "supervisor_exception",
                    "manifest_hash": manifest["manifest_hash"],
                    "summary_hash": summary["summary_hash"],
                },
            )
        except Exception:
            pass
        raise
    finally:
        for _process, (_spec, log_handle) in list(running.items()):
            try:
                log_handle.close()
            except Exception:
                pass


def merge_prefilter_local_parquet_plan_supervisor_manifest(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    run_dir: str | Path,
    min_free_bytes: int = 1,
) -> dict[str, Any]:
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=prefilter_plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
    )
    if not bool(manifest.get("merge_allowed")):
        raise ValueError("prefilter supervisor manifest is not mergeable")
    shard_dirs = []
    for shard in manifest["shards"]:
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
        _validate_prefilter_supervisor_ready_shard(
            shard_run_dir=shard_dir,
            shard=shard,
            manifest=manifest,
            prefilter_plan=prefilter_plan,
            config_payload=config_payload,
        )
        shard_dirs.append(shard_dir)
    return merge_prefilter_local_parquet_plan_shards(
        prefilter_plan=prefilter_plan,
        config_payload=config_payload,
        shard_run_dirs=shard_dirs,
        run_dir=run_dir,
        min_free_bytes=min_free_bytes,
        row_group_sharding_policy=_validate_row_group_sharding_policy(
            manifest.get("row_group_sharding_policy", ROW_GROUP_SHARDING_POLICY_ORDINAL)
        ),
    )


def _local_all_rows_prefilter_plan_receipt_fields(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _local_parquet_prefilter_plan_receipt_fields(plan)


def _local_parquet_prefilter_plan_receipt_fields(plan: Mapping[str, Any]) -> dict[str, Any]:
    _validate_local_parquet_prefilter_plan(plan, allow_shard=bool(plan.get("row_group_sharding_enabled", False)))
    plan_hash = _require_sha256(plan.get("schedule_hash"), "prefilter_plan_hash")
    fields = {
        "local_parquet_prefilter_plan_hash": plan_hash,
        "local_parquet_prefilter_plan_kind": str(plan["artifact_kind"]),
        "local_all_rows_prefilter_plan_hash": plan_hash,
        "all_rows_prefilter_plan_hash": _require_sha256(
            plan.get("all_rows_prefilter_plan_hash") or plan_hash,
            "all_rows_prefilter_plan_hash",
        ),
        "row_group_sharding_enabled": bool(plan.get("row_group_sharding_enabled", False)),
        "row_group_shard_count": int(plan.get("row_group_shard_count", 1)),
        "row_group_shard_index": int(plan.get("row_group_shard_index", 0)),
        "row_group_shard_row_group_count": int(plan["row_group_count"]),
        "row_group_shard_expected_physical_rows": int(plan["expected_physical_rows"]),
        "all_rows_row_group_count": int(plan.get("all_rows_row_group_count", plan["row_group_count"])),
        "all_rows_expected_physical_rows": int(
            plan.get("all_rows_expected_physical_rows", plan["expected_physical_rows"])
        ),
    }
    if plan["artifact_kind"] == LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND:
        source_plan_hash = _require_sha256(
            plan.get("token_target_tranche_plan_hash") or plan.get("local_parquet_prefilter_plan_hash") or plan_hash,
            "token_target_tranche_plan_hash",
        )
        sampling_policy = plan.get("sampling_policy")
        if not isinstance(sampling_policy, Mapping):
            raise ValueError("token tranche plan missing sampling_policy")
        fields.update(
            {
                "local_all_rows_prefilter_plan_hash": None,
                "all_rows_prefilter_plan_hash": None,
                "all_rows_row_group_count": int(plan.get("population_row_group_count", plan["row_group_count"])),
                "all_rows_expected_physical_rows": int(
                    plan.get("population_expected_physical_rows", plan["expected_physical_rows"])
                ),
                "token_target_tranche_plan_hash": source_plan_hash,
                "token_target_tranche_source_plan_hash": source_plan_hash,
                "token_target_tranche_mode": TOKEN_TARGET_TRANCHE_MODE,
                "token_target_tranche_seed_hash": stable_json_hash(str(sampling_policy.get("seed") or "")),
                "token_target_tranche_estimated_source_token_target": int(
                    sampling_policy["estimated_source_token_target"]
                ),
                "token_target_tranche_selected_estimated_source_tokens": int(
                    sampling_policy["selected_estimated_source_tokens_midpoint"]
                ),
                "token_target_tranche_population_estimated_source_tokens": int(
                    sampling_policy["population_estimated_source_tokens_midpoint"]
                ),
                "token_target_tranche_row_group_count": int(
                    plan.get("token_target_tranche_row_group_count", plan["row_group_count"])
                ),
                "token_target_tranche_expected_physical_rows": int(
                    plan.get("token_target_tranche_expected_physical_rows", plan["expected_physical_rows"])
                ),
                "population_row_group_count": int(plan.get("population_row_group_count", plan["row_group_count"])),
                "population_expected_physical_rows": int(
                    plan.get("population_expected_physical_rows", plan["expected_physical_rows"])
                ),
                "full_corpus_census": False,
                "selected_equals_population": False,
                "claims_scope": "tranche_only",
            }
        )
    return fields


def _copy_private_schedule_source(source: str, part_path: Path, *, allow_local_sources: bool) -> tuple[str, int]:
    part_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    bytes_written = 0

    def copy_stream(src: Any) -> None:
        nonlocal bytes_written
        with part_path.open("wb") as dst:
            for chunk in iter(lambda: src.read(16 * 1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
                bytes_written += len(chunk)
                dst.write(chunk)

    if source.startswith("https://"):
        try:
            import fsspec
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("fsspec is required to mirror HTTPS Parquet sources") from exc
        with fsspec.open(source, mode="rb") as src:
            copy_stream(src)
        _chmod_private_file(part_path)
        return digest.hexdigest(), bytes_written
    if allow_local_sources:
        local_source = Path(source)
        if not local_source.exists() or not local_source.is_file():
            raise FileNotFoundError("local test source file missing")
        with local_source.open("rb") as src:
            copy_stream(src)
        _chmod_private_file(part_path)
        return digest.hexdigest(), bytes_written
    raise ValueError("source mirroring requires HTTPS sources")


def mirror_private_schedule_sources(
    private_source_schedule: Mapping[str, Any],
    mirror_dir: str | Path,
    *,
    allow_local_sources: bool = False,
    stop_file: str | Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Mirror private schedule Parquet files and build a local execution schedule.

    The report is no-raw by construction: it records hashes, counts, byte sizes,
    and statuses, not source URLs or local paths.
    """

    parent_schedule_hash = _validate_private_source_schedule_hash(private_source_schedule)
    source_metadata_hash = _require_sha256(private_source_schedule.get("source_metadata_hash"), "source_metadata_hash")
    entries = private_source_schedule.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("private source schedule entries must be a non-empty list")
    mirror_path = Path(mirror_dir)
    _ensure_private_cache_dir(mirror_path)
    stop_path = Path(stop_file) if stop_file is not None else None
    grouped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("private source schedule entries must be mappings")
        source_file_id = str(entry.get("source_file_id") or "")
        if not source_file_id:
            raise ValueError("private source schedule entry missing source_file_id")
        source = str(entry.get("parquet_url_or_path") or "")
        if not source:
            raise ValueError("private source schedule entry missing parquet_url_or_path")
        group = grouped.setdefault(source_file_id, {"source": source, "entries": []})
        if group["source"] != source:
            raise ValueError("source_file_id maps to multiple source locators")
        group["entries"].append(entry)
    file_reports: list[dict[str, Any]] = []
    completed = True
    metadata_cache: dict[Path, Any] = {}
    for source_file_id, group in sorted(grouped.items()):
        if stop_path is not None and stop_path.exists():
            completed = False
            break
        local_file = _local_parquet_mirror_path(mirror_path, source_file_id)
        ready_path = local_file.with_suffix(local_file.suffix + ".ready.json")
        status = "reused" if local_file.exists() and ready_path.exists() else "mirrored"
        if status == "mirrored":
            part_path = local_file.with_suffix(local_file.suffix + ".part")
            if part_path.exists():
                part_path.unlink()
            file_sha256, copied_bytes = _copy_private_schedule_source(
                str(group["source"]), part_path, allow_local_sources=allow_local_sources
            )
            if int(part_path.stat().st_size) != copied_bytes:
                raise ValueError("local mirror copied byte count drift")
            os.replace(part_path, local_file)
            _chmod_private_file(local_file)
        else:
            file_sha256 = _sha256_file(local_file)
        for entry in group["entries"]:
            _validate_local_parquet_entry(local_file, entry, metadata_cache)
        ready_payload = _local_mirror_file_ready_payload(
            source_file_id=source_file_id,
            parent_schedule_hash=parent_schedule_hash,
            source_metadata_hash=source_metadata_hash,
            local_file=local_file,
            entries=group["entries"],
            file_sha256=file_sha256,
        )
        if status == "reused":
            _validate_local_mirror_ready(local_file, ready_payload)
        else:
            _write_local_mirror_ready(local_file, ready_payload)
        file_reports.append(
            {
                "source_file_id_hash": stable_json_hash(source_file_id),
                "local_mirror_file_name_hash": stable_json_hash(local_file.name),
                "local_mirror_file_size_bytes": int(local_file.stat().st_size),
                "local_mirror_file_sha256": file_sha256,
                "entry_count": len(group["entries"]),
                "status": status,
            }
        )
    local_schedule = build_local_parquet_execution_schedule(private_source_schedule, mirror_path) if completed else None
    report = {
        "artifact_kind": LOCAL_PARQUET_MIRROR_REPORT_KIND,
        "schema_version": 1,
        "status": "ready" if completed else "stopped",
        "parent_schedule_hash": parent_schedule_hash,
        "local_schedule_hash": local_schedule.get("schedule_hash") if local_schedule is not None else None,
        "source_metadata_hash": source_metadata_hash,
        "local_mirror_dir_sha256": _sha256_text(str(mirror_path)),
        "source_file_count": len(grouped),
        "mirrored_or_reused_file_count": len(file_reports),
        "entry_count": len(entries),
        "total_mirrored_bytes": sum(int(item["local_mirror_file_size_bytes"]) for item in file_reports),
        "file_reports": file_reports,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    report["report_hash"] = stable_json_hash({key: value for key, value in report.items() if key != "report_hash"})
    validate_public_no_raw_payload(report)
    return local_schedule, report


def validate_sparse_inference_artifacts(report: Mapping[str, Any], model_path: str | Path) -> None:
    validate_public_no_raw_payload(report)
    _validate_report_self_hash_if_present(report, "chunk sparse report")
    if report.get("artifact_kind") != "twc_chunk_hashed_ngram_sparse_report":
        raise ValueError("unsupported chunk sparse report kind")
    if report.get("status") != "complete":
        raise ValueError("chunk sparse report must be complete")
    if report.get("model_id") != CHUNK_SPARSE_MODEL_ID:
        raise ValueError("chunk sparse model_id mismatch")
    _threshold(report.get("threshold"), "chunk sparse threshold")
    path = Path(model_path)
    if str(report.get("model_file_name") or "") != path.name:
        raise ValueError("chunk sparse model file name drift")
    if not path.exists():
        raise ValueError("chunk sparse model file missing")
    if _sha256_file(path) != _require_sha256(report.get("model_file_sha256"), "chunk sparse model_file_sha256"):
        raise ValueError("chunk sparse model file hash drift")
    feature_config = report.get("feature_config")
    if not isinstance(feature_config, Mapping) or int(feature_config.get("n_features", 0)) < 2:
        raise ValueError("chunk sparse report missing valid feature_config")
    for field in ("classifier_manifest_hash", "evidence_cache_hash", "feature_config_hash", "model_payload_hash"):
        _require_sha256(report.get(field), field)


def validate_window_mil_inference_artifacts(report: Mapping[str, Any], model_path: str | Path) -> None:
    validate_public_no_raw_payload(report)
    _validate_report_self_hash_if_present(report, "chunk Window-MIL report")
    if report.get("artifact_kind") != "twc_chunk_modernbert_window_mil_report":
        raise ValueError("unsupported chunk Window-MIL report kind")
    if report.get("status") != "complete":
        raise ValueError("chunk Window-MIL report must be complete")
    if report.get("model_id") != CHUNK_WINDOW_MIL_MODEL_ID:
        raise ValueError("chunk Window-MIL model_id mismatch")
    _threshold(report.get("threshold"), "chunk Window-MIL threshold")
    path = Path(model_path)
    if str(report.get("model_file_name") or "") != path.name:
        raise ValueError("chunk Window-MIL model file name drift")
    if not path.exists():
        raise ValueError("chunk Window-MIL model file missing")
    if _sha256_file(path) != _require_sha256(report.get("model_file_sha256"), "chunk Window-MIL model_file_sha256"):
        raise ValueError("chunk Window-MIL model file hash drift")
    if not str(report.get("base_model_name") or ""):
        raise ValueError("chunk Window-MIL report missing base_model_name")
    if _positive_int(report.get("max_length"), "max_length") < 8:
        raise ValueError("chunk Window-MIL max_length must be at least 8")
    stride = _nonnegative_int(report.get("stride"), "stride")
    if stride >= int(report["max_length"]):
        raise ValueError("chunk Window-MIL stride must be less than max_length")
    _positive_int(report.get("max_windows_per_device_batch"), "max_windows_per_device_batch")
    for field in ("classifier_manifest_hash", "evidence_cache_hash", "model_payload_hash"):
        _require_sha256(report.get(field), field)


def _validate_report_self_hash_if_present(report: Mapping[str, Any], label: str) -> None:
    if report.get("report_hash") is None:
        return
    if report.get("report_hash") != stable_json_hash({key: value for key, value in report.items() if key != "report_hash"}):
        raise ValueError(f"{label} report_hash content mismatch")


def _validate_report_self_hash_required(report: Mapping[str, Any], label: str) -> None:
    if report.get("report_hash") is None:
        raise ValueError(f"{label} missing report_hash")
    if report.get("report_hash") != stable_json_hash({key: value for key, value in report.items() if key != "report_hash"}):
        raise ValueError(f"{label} report_hash content mismatch")


def _validate_locked_sparse_runtime_artifact(
    report: Mapping[str, Any],
    model_path: str | Path,
    config_payload: Mapping[str, Any],
) -> None:
    validate_sparse_inference_artifacts(report, model_path)
    if report.get("report_hash") is None:
        raise ValueError("S1 sparse report missing report_hash")
    if report.get("report_hash") != config_payload["sparse_report_hash"]:
        raise ValueError("S1 sparse report hash does not match locked config")
    if report.get("model_file_sha256") != config_payload["sparse_model_file_sha256"]:
        raise ValueError("S1 sparse model hash does not match locked config")
    if report.get("classifier_manifest_hash") != config_payload["classifier_manifest_hash"]:
        raise ValueError("S1 sparse classifier manifest hash mismatch")
    if report.get("evidence_cache_hash") != config_payload["private_evidence_cache_hash"]:
        raise ValueError("S1 sparse evidence cache hash mismatch")


def _validate_locked_w1_runtime_artifact(
    report: Mapping[str, Any],
    model_path: str | Path,
    config_payload: Mapping[str, Any],
) -> None:
    validate_window_mil_inference_artifacts(report, model_path)
    if report.get("report_hash") is None:
        raise ValueError("W1 report missing report_hash")
    if report.get("report_hash") != config_payload["window_mil_report_hash"]:
        raise ValueError("W1 report hash does not match locked config")
    if report.get("model_file_sha256") != config_payload["window_mil_model_file_sha256"]:
        raise ValueError("W1 model hash does not match locked config")
    if report.get("classifier_manifest_hash") != config_payload["classifier_manifest_hash"]:
        raise ValueError("W1 classifier manifest hash mismatch")
    if report.get("evidence_cache_hash") != config_payload["private_evidence_cache_hash"]:
        raise ValueError("W1 evidence cache hash mismatch")
    variant = report.get("window_variant_config")
    if not isinstance(variant, Mapping):
        raise ValueError("W1 report missing window_variant_config")
    if str(variant.get("input_mode")) != config_payload["window_mil_input_mode"]:
        raise ValueError("W1 input_mode mismatch")
    if str(variant.get("pooling_mode")) != config_payload["window_mil_pooling_mode"]:
        raise ValueError("W1 pooling_mode mismatch")
    if int(variant.get("pooling_top_k", -1)) != int(config_payload["window_mil_pooling_top_k"]):
        raise ValueError("W1 pooling_top_k mismatch")


def build_chunk_inference_artifact_manifest(
    *,
    freeze_report: Mapping[str, Any],
    sparse_report: Mapping[str, Any],
    sparse_model_path: str | Path,
    window_mil_report: Mapping[str, Any],
    window_mil_model_path: str | Path,
    config_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_public_no_raw_payload(freeze_report)
    _validate_report_self_hash_required(freeze_report, "freeze report")
    validate_sparse_inference_artifacts(sparse_report, sparse_model_path)
    validate_window_mil_inference_artifacts(window_mil_report, window_mil_model_path)
    _validate_report_self_hash_required(sparse_report, "sparse report")
    _validate_report_self_hash_required(window_mil_report, "Window-MIL report")
    if freeze_report.get("artifact_kind") == SELECTED_POLICY_FREEZE_REPORT_KIND:
        if config_payload is None:
            raise ValueError("selected policy inference manifest requires config_payload")
        _validate_selected_policy_freeze_report(freeze_report, config_payload)
        if sparse_report.get("report_hash") != config_payload["sparse_report_hash"]:
            raise ValueError("S1 sparse report hash does not match selected policy config")
        if sparse_report.get("model_file_sha256") != config_payload["sparse_model_file_sha256"]:
            raise ValueError("S1 sparse model hash does not match selected policy config")
        if window_mil_report.get("report_hash") != config_payload["window_mil_report_hash"]:
            raise ValueError("W1 report hash does not match selected policy config")
        if window_mil_report.get("model_file_sha256") != config_payload["window_mil_model_file_sha256"]:
            raise ValueError("W1 model hash does not match selected policy config")
        thresholds = _selected_policy_thresholds_by_model_id(freeze_report)
    elif freeze_report.get("artifact_kind") == "twc_chunk_classifier_freeze_report":
        if freeze_report.get("status") != "ready":
            raise ValueError("chunk classifier freeze report must be ready")
        thresholds = _freeze_thresholds_by_model_id(freeze_report)
    else:
        raise ValueError("unsupported chunk classifier freeze report")
    if freeze_report.get("sparse_report_hash") is not None and sparse_report.get("report_hash") != freeze_report.get("sparse_report_hash"):
        raise ValueError("sparse report hash does not match freeze report")
    if freeze_report.get("window_mil_report_hash") is not None and window_mil_report.get("report_hash") != freeze_report.get("window_mil_report_hash"):
        raise ValueError("Window-MIL report hash does not match freeze report")
    for report_name, report in (("sparse", sparse_report), ("Window-MIL", window_mil_report)):
        if report.get("classifier_manifest_hash") != freeze_report.get("classifier_manifest_hash"):
            raise ValueError(f"{report_name} classifier manifest hash does not match freeze report")
        if report.get("evidence_cache_hash") != freeze_report.get("evidence_cache_hash"):
            raise ValueError(f"{report_name} evidence cache hash does not match freeze report")
    window_variant = window_mil_report.get("window_variant_config")
    if not isinstance(window_variant, Mapping):
        raise ValueError("Window-MIL report missing window_variant_config")
    manifest = {
        "artifact_kind": CHUNK_INFERENCE_ARTIFACT_MANIFEST_KIND,
        "schema_version": 1,
        "status": "ready",
        "classifier_freeze_hash": _require_sha256(freeze_report.get("report_hash"), "classifier_freeze_hash"),
        "selected_policy_report_hash": _require_sha256(
            freeze_report.get("report_hash"),
            "selected_policy_report_hash",
        ),
        "selected_policy_hash": _require_sha256(
            freeze_report.get("selected_policy_hash", freeze_report.get("report_hash")),
            "selected_policy_hash",
        ),
        "deterministic_filter_policy_hash": _require_sha256(
            freeze_report.get("deterministic_filter_policy_hash"),
            "deterministic_filter_policy_hash",
        ),
        "classifier_manifest_hash": _require_sha256(freeze_report.get("classifier_manifest_hash"), "classifier_manifest_hash"),
        "evidence_cache_hash": _require_sha256(freeze_report.get("evidence_cache_hash"), "evidence_cache_hash"),
        "sparse_report_hash": _require_sha256(sparse_report.get("report_hash"), "sparse_report_hash"),
        "sparse_model_file_sha256": _require_sha256(sparse_report.get("model_file_sha256"), "sparse_model_file_sha256"),
        "window_mil_report_hash": _require_sha256(window_mil_report.get("report_hash"), "window_mil_report_hash"),
        "window_mil_model_file_sha256": _require_sha256(
            window_mil_report.get("model_file_sha256"),
            "window_mil_model_file_sha256",
        ),
        "window_mil_input_mode": str(window_variant.get("input_mode")),
        "window_mil_pooling_mode": str(window_variant.get("pooling_mode")),
        "window_mil_pooling_top_k": _positive_int(window_variant.get("pooling_top_k"), "window_mil_pooling_top_k"),
        "thresholds": {
            CHUNK_SPARSE_MODEL_ID: float(thresholds[CHUNK_SPARSE_MODEL_ID]),
            CHUNK_WINDOW_MIL_MODEL_ID: float(thresholds[CHUNK_WINDOW_MIL_MODEL_ID]),
        },
        "decision_rule": "reject_if_either_model_rejects_else_quarantine_on_missing_or_invalid_else_safe",
        "classifier_can_approve_corpus": False,
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
    manifest["inference_artifact_manifest_hash"] = stable_json_hash(manifest)
    validate_chunk_inference_artifact_manifest(manifest)
    return manifest


def validate_chunk_inference_artifact_manifest(manifest: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(manifest)
    if manifest.get("artifact_kind") != CHUNK_INFERENCE_ARTIFACT_MANIFEST_KIND:
        raise ValueError("unsupported chunk inference artifact manifest kind")
    if manifest.get("status") != "ready":
        raise ValueError("chunk inference artifact manifest must be ready")
    for field in (
        "classifier_freeze_hash",
        "selected_policy_report_hash",
        "selected_policy_hash",
        "deterministic_filter_policy_hash",
        "classifier_manifest_hash",
        "evidence_cache_hash",
        "sparse_report_hash",
        "sparse_model_file_sha256",
        "window_mil_report_hash",
        "window_mil_model_file_sha256",
    ):
        _require_sha256(manifest.get(field), field)
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("chunk inference artifact manifest missing thresholds")
    if set(thresholds) != {CHUNK_SPARSE_MODEL_ID, CHUNK_WINDOW_MIL_MODEL_ID}:
        raise ValueError("chunk inference artifact manifest threshold model set mismatch")
    for model_id, value in thresholds.items():
        _threshold(value, f"threshold.{model_id}")
    if manifest.get("window_mil_input_mode") != WINDOW_INPUT_FULL_EVIDENCE:
        raise ValueError("chunk inference artifact manifest window_mil_input_mode mismatch")
    if manifest.get("window_mil_pooling_mode") != "topk_mean_logits":
        raise ValueError("chunk inference artifact manifest window_mil_pooling_mode mismatch")
    if _positive_int(manifest.get("window_mil_pooling_top_k"), "window_mil_pooling_top_k") != 3:
        raise ValueError("chunk inference artifact manifest window_mil_pooling_top_k mismatch")
    if manifest.get("classifier_can_approve_corpus") is not False:
        raise ValueError("classifier must not approve corpus")
    _require_false_flags(manifest, PUBLIC_FALSE_FLAGS, "chunk inference artifact manifest")
    if manifest.get("inference_artifact_manifest_hash") != stable_json_hash({key: value for key, value in manifest.items() if key != "inference_artifact_manifest_hash"}):
        raise ValueError("chunk inference artifact manifest hash mismatch")


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
    _require_sha256(config_hash, "config_hash")
    if deterministic_policy_hash is not None:
        _require_sha256(deterministic_policy_hash, "deterministic_policy_hash")
    target = _positive_int(target_proxy_tokens, "target_proxy_tokens")
    halo = _nonnegative_int(halo_chars, "halo_chars")
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
    scorers: Sequence[ChunkFilteringScorer],
    thresholds: Mapping[str, float],
    config_hash: str,
    source_chunk_manifest_hash: str,
    deterministic_policy_payload: Mapping[str, Any] | None = None,
    deterministic_ontology: RiskOntology | None = None,
) -> dict[str, Any]:
    _require_sha256(config_hash, "config_hash")
    _require_sha256(source_chunk_manifest_hash, "source_chunk_manifest_hash")
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
        deterministic_policy_hash = _require_sha256(
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
            parsed = _safe_runtime_score(score)
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
        "artifact_kind": CHUNK_FILTER_DOCUMENT_REPORT_KIND,
        "schema_version": 1,
        "filtering_policy_version": CHUNK_FILTERING_POLICY_VERSION,
        "source_document_id_hash": result.source.source_document_id_hash,
        "segmentation_receipt_hash": result.receipt.receipt_hash,
        "config_hash": config_hash,
        "candidate_count": len(candidate_rows),
        "decision_counts": _sorted_counter(Counter(str(decision["classifier_decision"]) for decision in decisions)),
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
    validate_public_no_raw_payload(document_report)
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


def store_chunk_filtering_outputs(state: ChunkedCleanRunState, outputs: Mapping[str, Any]) -> None:
    conn = state.conn
    _ensure_filtering_state_schema(conn)
    with conn:
        for candidate in outputs.get("candidates") or []:
            _insert_filtering_row(
                conn,
                table="classifier_candidates",
                key_column="candidate_id_hash",
                key_value=str(candidate["candidate_id_hash"]),
                json_column="candidate_json",
                columns=(
                    "candidate_id_hash",
                    "source_document_id_hash",
                    "segmentation_receipt_hash",
                    "candidate_policy_hash",
                    "core_block_indexes_json",
                    "core_char_start",
                    "core_char_end",
                    "evidence_char_start",
                    "evidence_char_end",
                    "core_text_sha256",
                    "evidence_text_sha256",
                    "status",
                    "candidate_json",
                ),
                values=(
                    candidate["candidate_id_hash"],
                    candidate["source_document_id_hash"],
                    candidate["segmentation_receipt_hash"],
                    candidate["candidate_policy_hash"],
                    json.dumps(candidate["core_block_indexes"]),
                    int(candidate["core_char_start"]),
                    int(candidate["core_char_end"]),
                    int(candidate["evidence_char_start"]),
                    int(candidate["evidence_char_end"]),
                    candidate["core_text_sha256"],
                    candidate["evidence_text_sha256"],
                    "complete",
                    json.dumps(candidate, sort_keys=True),
                ),
            )
        for decision in outputs.get("decisions") or []:
            _insert_filtering_row(
                conn,
                table="chunk_decisions",
                key_column="decision_id_hash",
                key_value=str(decision["decision_id_hash"]),
                json_column="decision_json",
                columns=(
                    "decision_id_hash",
                    "candidate_id_hash",
                    "source_document_id_hash",
                    "status",
                    "decision_json",
                ),
                values=(
                    decision["decision_id_hash"],
                    decision["candidate_id"],
                    outputs["source_document_id_hash"],
                    decision["classifier_decision"],
                    json.dumps(decision, sort_keys=True),
                ),
            )
        for gap in outputs.get("gap_events") or []:
            _insert_filtering_row(
                conn,
                table="gap_events",
                key_column="gap_event_id_hash",
                key_value=str(gap["gap_event_id_hash"]),
                json_column="gap_json",
                columns=(
                    "gap_event_id_hash",
                    "candidate_id_hash",
                    "source_document_id_hash",
                    "block_indexes_json",
                    "status",
                    "gap_json",
                ),
                values=(
                    gap["gap_event_id_hash"],
                    gap["candidate_id_hash"],
                    gap["source_document_id_hash"],
                    json.dumps(gap["block_indexes"]),
                    gap["classifier_decision"],
                    json.dumps(gap, sort_keys=True),
                ),
            )
        for run in outputs.get("clean_runs") or []:
            _insert_filtering_row(
                conn,
                table="clean_runs",
                key_column="clean_run_id_hash",
                key_value=str(run["clean_run_id_hash"]),
                json_column="clean_run_json",
                columns=(
                    "clean_run_id_hash",
                    "source_document_id_hash",
                    "block_indexes_json",
                    "status",
                    "clean_run_json",
                ),
                values=(
                    run["clean_run_id_hash"],
                    run["source_document_id_hash"],
                    json.dumps(run["block_indexes"]),
                    "safe",
                    json.dumps(run, sort_keys=True),
                ),
            )


def validate_chunked_filtering_state(state: ChunkedCleanRunState) -> None:
    validate_chunked_clean_run_state(state, lane_mode="filtering")


def validate_private_clean_run_output_path(path: str | Path) -> None:
    resolved = Path(path).expanduser().resolve(strict=False)
    for parent in (resolved, *resolved.parents):
        if parent.name == "code" and (parent / "twc").exists() and (parent / "tests").exists():
            raise ValueError("private clean-run output must be outside the public repo")


def validate_private_filtering_run_dir(path: str | Path) -> None:
    resolved = Path(path).expanduser().resolve(strict=False)
    for parent in (resolved, *resolved.parents):
        if parent.name == "code" and (parent / "twc").exists() and (parent / "tests").exists():
            raise ValueError("private filtering run dir must be outside the public repo")
    if Path(path).expanduser().is_symlink():
        raise ValueError("private filtering run dir must not be a symlink")


def validate_private_source_cache_dir(path: str | Path) -> None:
    resolved = Path(path).expanduser().resolve(strict=False)
    for parent in (resolved, *resolved.parents):
        if parent.name == "code" and (parent / "twc").exists() and (parent / "tests").exists():
            raise ValueError("private source cache dir must be outside the public repo")
    if Path(path).expanduser().is_symlink():
        raise ValueError("private source cache dir must not be a symlink")


def _ensure_private_dir(path: Path) -> None:
    validate_private_filtering_run_dir(path)
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(path, 0o700)


def _ensure_private_cache_dir(path: Path) -> None:
    validate_private_source_cache_dir(path)
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(path, 0o700)


def _chmod_private_file(path: str | Path) -> None:
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except FileNotFoundError:
            return


def _chmod_private_sqlite_files(path: str | Path) -> None:
    base = Path(path)
    _chmod_private_file(base)
    _chmod_private_file(Path(str(base) + "-wal"))
    _chmod_private_file(Path(str(base) + "-shm"))


def _path_free_bytes(path: str | Path) -> int:
    probe = Path(path)
    existing = probe if probe.exists() else probe.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return int(os.statvfs(existing).f_bavail * os.statvfs(existing).f_frsize)


def _check_disk_watermark(paths: Sequence[str | Path], *, min_free_bytes: int) -> None:
    minimum = _nonnegative_int(min_free_bytes, "min_free_bytes")
    for path in paths:
        if _path_free_bytes(path) < minimum:
            raise RuntimeError("disk hard stop: free bytes below configured minimum")


def _expected_prefilter_schedule_rows(
    schedule: Mapping[str, Any],
    *,
    max_rows: int | None,
) -> tuple[str, list[str]]:
    schedule_hash = _require_sha256(schedule.get("schedule_hash"), "schedule_hash")
    expected: list[str] = []
    seen: set[str] = set()
    row_cap = None if max_rows is None else _positive_int(max_rows, "max_rows")
    for entry in schedule.get("entries") or []:
        if not isinstance(entry, Mapping):
            raise ValueError("prefilter-cpu schedule entries must be mappings")
        offsets = [int(offset) for offset in entry.get("row_group_local_offsets") or []]
        if len(offsets) != len(set(offsets)):
            raise ValueError("prefilter-cpu schedule entry contains duplicate offsets")
        for offset in offsets:
            key = _physical_row_key(entry, offset)
            if key in seen:
                raise ValueError("prefilter-cpu schedule contains duplicate physical row keys")
            seen.add(key)
            expected.append(key)
            if row_cap is not None and len(expected) >= row_cap:
                break
        if row_cap is not None and len(expected) >= row_cap:
            break
    if not expected:
        raise ValueError("prefilter-cpu schedule has no expected physical rows")
    return schedule_hash, expected


def _record_prefilter_expected_schedule(
    conn: sqlite3.Connection,
    input_schedule_hash: str,
    expected_physical_rows: Sequence[str],
) -> None:
    _record_prefilter_expected_counts(
        conn,
        input_schedule_hash=input_schedule_hash,
        expected_physical_row_count=len(expected_physical_rows),
        expected_physical_row_key_set_hash=stable_json_hash(sorted(expected_physical_rows)),
    )


def _record_prefilter_expected_counts(
    conn: sqlite3.Connection,
    *,
    input_schedule_hash: str,
    expected_physical_row_count: int,
    expected_physical_row_key_set_hash: str | None,
) -> None:
    existing = {
        str(row["key"]): json.loads(str(row["value_json"]))
        for row in conn.execute(
            "select key, value_json from stage_metadata where key in ('input_schedule_hash', 'expected_physical_row_count', 'expected_physical_row_key_set_hash')"
        )
    }
    expected_values = {
        "input_schedule_hash": input_schedule_hash,
        "expected_physical_row_count": _positive_int(expected_physical_row_count, "expected_physical_row_count"),
        "expected_physical_row_key_set_hash": (
            None
            if expected_physical_row_key_set_hash is None
            else _require_sha256(expected_physical_row_key_set_hash, "expected_physical_row_key_set_hash")
        ),
    }
    for key, value in expected_values.items():
        if key in existing and existing[key] != value:
            raise ValueError(f"prefilter-cpu {key} drift")
    with conn:
        for key, value in expected_values.items():
            conn.execute(
                "insert or replace into stage_metadata(key, value_json) values (?, ?)",
                (key, json.dumps(value, sort_keys=True)),
            )


def run_prefilter_cpu_stage_for_documents(
    *,
    results: Sequence[SegmentationResult],
    config_payload: Mapping[str, Any],
    sparse_scorer: ChunkFilteringScorer,
    run_dir: str | Path,
    deterministic_policy_payload: Mapping[str, Any],
    deterministic_ontology: RiskOntology,
    forbidden_w1_scorer: ChunkFilteringScorer | None = None,
) -> dict[str, Any]:
    del forbidden_w1_scorer
    validate_chunk_filtering_config_payload(config_payload)
    if sparse_scorer.model_id != CHUNK_SPARSE_MODEL_ID:
        raise ValueError("prefilter-cpu requires the S1 sparse scorer only")
    if config_payload.get("deterministic_filter_policy_hash") != deterministic_policy_payload.get(
        "deterministic_filter_policy_hash"
    ):
        raise ValueError("prefilter-cpu deterministic filter policy hash mismatch")
    validate_deterministic_filter_policy_payload(deterministic_policy_payload)
    run_path = Path(run_dir)
    _ensure_private_dir(run_path)
    start = time.monotonic()
    conn = _open_stage_state(run_path, config_payload)
    stop_requested = False
    try:
        for result in results:
            raw_text = str(result.source.raw_text) if isinstance(result.source.raw_text, str) else ""
            if result.receipt.document_status != "segmented":
                continue
            _prefilter_segmented_result(
                conn,
                result=result,
                raw_text=raw_text,
                config_payload=config_payload,
                sparse_scorer=sparse_scorer,
                deterministic_policy_payload=deterministic_policy_payload,
                deterministic_ontology=deterministic_ontology,
            )
            if (run_path / "STOP_REQUESTED").exists():
                stop_requested = True
                break
        queue_path = run_path / "w1_survivor_queue.jsonl"
        if stop_requested:
            _write_jsonl_path_streaming(
                queue_path.with_suffix(queue_path.suffix + ".part"),
                _iter_prefilter_survivor_queue_rows(conn, config_payload),
            )
            report = _build_prefilter_receipt(
                conn,
                config_payload,
                run_path=run_path,
                status="stopped",
                stop_reason="stop_requested",
                elapsed_seconds=time.monotonic() - start,
                cache_dir=None,
                min_free_bytes=1,
            )
        else:
            queue_proof = _write_survivor_queue_ready_streaming(
                queue_path,
                _iter_prefilter_survivor_queue_rows(conn, config_payload),
            )
            report = _build_prefilter_receipt(
                conn,
                config_payload,
                run_path=run_path,
                status="prefilter_ready_for_w1",
                stop_reason="prefilter_complete",
                elapsed_seconds=time.monotonic() - start,
                cache_dir=None,
                min_free_bytes=1,
                survivor_queue_ready=True,
                survivor_queue_part=False,
                survivor_queue_hash=queue_proof.survivor_queue_hash,
                survivor_queue_row_count=queue_proof.survivor_queue_row_count,
                survivor_queue_id_index_hash=queue_proof.survivor_queue_id_index_hash,
                survivor_queue_id_index_row_count=queue_proof.survivor_queue_id_index_row_count,
                survivor_candidate_id_set_hash=queue_proof.survivor_candidate_id_set_hash,
                survivor_private_evidence_id_set_hash=queue_proof.survivor_private_evidence_id_set_hash,
            )
        _write_json_atomic(run_path / "prefilter_cpu_receipt.json", report)
        return report
    finally:
        conn.close()


def run_prefilter_cpu_stage_for_schedule(
    *,
    schedule: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    row_provider: Any,
    sparse_scorer: ChunkFilteringScorer,
    run_dir: str | Path,
    deterministic_policy_payload: Mapping[str, Any],
    deterministic_ontology: RiskOntology,
    max_rows: int | None = None,
    cache_dir: str | Path | None = None,
    min_free_bytes: int = 1,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    if sparse_scorer.model_id != CHUNK_SPARSE_MODEL_ID:
        raise ValueError("prefilter-cpu requires the S1 sparse scorer only")
    if config_payload.get("deterministic_filter_policy_hash") != deterministic_policy_payload.get(
        "deterministic_filter_policy_hash"
    ):
        raise ValueError("prefilter-cpu deterministic filter policy hash mismatch")
    validate_deterministic_filter_policy_payload(deterministic_policy_payload)
    if schedule.get("source_metadata_hash") != config_payload["source_metadata_hash"]:
        raise ValueError("prefilter-cpu schedule source metadata hash mismatch")
    input_schedule_hash, expected_physical_rows = _expected_prefilter_schedule_rows(schedule, max_rows=max_rows)
    expected_physical_row_set = set(expected_physical_rows)
    run_path = Path(run_dir)
    _ensure_private_dir(run_path)
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        _ensure_private_cache_dir(cache_path)
    _check_disk_watermark([run_path, *( [cache_path] if cache_path is not None else [] )], min_free_bytes=min_free_bytes)
    conn = _open_stage_state(run_path, config_payload)
    _record_prefilter_expected_schedule(conn, input_schedule_hash, expected_physical_rows)
    start = time.monotonic()
    stop_requested = False
    try:
        for source_entry in schedule.get("entries") or []:
            entry = dict(source_entry)
            entry["source_schedule_hash"] = input_schedule_hash
            offsets = [int(offset) for offset in entry.get("row_group_local_offsets") or []]
            if not offsets:
                continue
            pending_offsets: list[int] = []
            for offset in offsets:
                physical_key = _physical_row_key(entry, offset)
                if physical_key not in expected_physical_row_set:
                    continue
                existing = conn.execute(
                    "select row_status from prefilter_source_rows where physical_row_key_hash = ?",
                    (physical_key,),
                ).fetchone()
                if existing is not None and str(existing["row_status"]) in {
                    "prefiltered",
                    "segmentation_error",
                    "fetch_error",
                    "non_mapping_row",
                    "missing_text",
                }:
                    continue
                _upsert_prefilter_source_row(
                    conn,
                    entry=entry,
                    offset=offset,
                    row_status="scheduled",
                    error_class=None,
                    segmentation_receipt_hash=None,
                    candidate_count=0,
                    survivor_count=0,
                )
                pending_offsets.append(offset)
            if not pending_offsets:
                continue
            if (run_path / "STOP_REQUESTED").exists():
                stop_requested = True
                break
            _check_disk_watermark([run_path, *( [cache_path] if cache_path is not None else [] )], min_free_bytes=min_free_bytes)
            try:
                fetched = row_provider.fetch_rows(dict(entry), pending_offsets)
            except BaseException as exc:
                for offset in pending_offsets:
                    _upsert_prefilter_source_row(
                        conn,
                        entry=entry,
                        offset=offset,
                        row_status="fetch_error",
                        error_class=type(exc).__name__,
                        segmentation_receipt_hash=None,
                        candidate_count=0,
                        survivor_count=0,
                    )
                continue
            for offset in pending_offsets:
                row = fetched.get(offset)
                if isinstance(row, BaseException):
                    _upsert_prefilter_source_row(
                        conn,
                        entry=entry,
                        offset=offset,
                        row_status="fetch_error",
                        error_class=type(row).__name__,
                        segmentation_receipt_hash=None,
                        candidate_count=0,
                        survivor_count=0,
                    )
                    continue
                if not isinstance(row, Mapping):
                    _upsert_prefilter_source_row(
                        conn,
                        entry=entry,
                        offset=offset,
                        row_status="non_mapping_row",
                        error_class="non_mapping_row",
                        segmentation_receipt_hash=None,
                        candidate_count=0,
                        survivor_count=0,
                    )
                    continue
                text = row.get("text")
                if not isinstance(text, str):
                    result = segment_source_document(
                        SourceDocumentRecord(
                            source_document_id_hash=_source_document_id_for_schedule_row(entry, offset),
                            split="train",
                            raw_text=text,
                            source_lineage_hash=_physical_row_key(entry, offset),
                            source_token_count_hint=_metadata_token_count(row),
                        )
                    )
                    _upsert_prefilter_source_row(
                        conn,
                        entry=entry,
                        offset=offset,
                        row_status="missing_text",
                        error_class=str(result.receipt.document_status),
                        segmentation_receipt_hash=result.receipt.receipt_hash,
                        candidate_count=0,
                        survivor_count=0,
                    )
                    continue
                result = segment_source_document(
                    SourceDocumentRecord(
                        source_document_id_hash=_source_document_id_for_schedule_row(entry, offset),
                        split="train",
                        raw_text=text,
                        source_lineage_hash=_physical_row_key(entry, offset),
                        source_token_count_hint=_metadata_token_count(row),
                    )
                )
                if result.receipt.document_status != "segmented":
                    _upsert_prefilter_source_row(
                        conn,
                        entry=entry,
                        offset=offset,
                        row_status="segmentation_error",
                        error_class=str(result.receipt.document_status),
                        segmentation_receipt_hash=result.receipt.receipt_hash,
                        candidate_count=0,
                        survivor_count=0,
                    )
                    continue
                before = _prefilter_candidate_counts(conn)
                _prefilter_segmented_result(
                    conn,
                    result=result,
                    raw_text=text,
                    config_payload=config_payload,
                    sparse_scorer=sparse_scorer,
                    deterministic_policy_payload=deterministic_policy_payload,
                    deterministic_ontology=deterministic_ontology,
                )
                after = _prefilter_candidate_counts(conn)
                _upsert_prefilter_source_row(
                    conn,
                    entry=entry,
                    offset=offset,
                    row_status="prefiltered",
                    error_class=None,
                    segmentation_receipt_hash=result.receipt.receipt_hash,
                    candidate_count=after["candidate_count"] - before["candidate_count"],
                    survivor_count=after["survivor_count"] - before["survivor_count"],
                )
                _check_disk_watermark([run_path, *( [cache_path] if cache_path is not None else [] )], min_free_bytes=min_free_bytes)
                if (run_path / "STOP_REQUESTED").exists():
                    stop_requested = True
                    break
            if stop_requested:
                break
        queue_path = run_path / "w1_survivor_queue.jsonl"
        if stop_requested:
            _write_jsonl_path_streaming(
                queue_path.with_suffix(queue_path.suffix + ".part"),
                _iter_prefilter_survivor_queue_rows(conn, config_payload),
            )
            report = _build_prefilter_receipt(
                conn,
                config_payload,
                run_path=run_path,
                status="stopped",
                stop_reason="stop_requested",
                elapsed_seconds=time.monotonic() - start,
                cache_dir=cache_path,
                min_free_bytes=min_free_bytes,
            )
        else:
            final_row_counts = _prefilter_source_row_summary(conn)
            blocked_by_errors = bool(final_row_counts["row_error_counts"])
            if blocked_by_errors:
                queue_hash = None
                queue_proof = None
                _write_jsonl_path_streaming(
                    queue_path.with_suffix(queue_path.suffix + ".part"),
                    _iter_prefilter_survivor_queue_rows(conn, config_payload),
                )
            else:
                queue_proof = _write_survivor_queue_ready_streaming(
                    queue_path,
                    _iter_prefilter_survivor_queue_rows(conn, config_payload),
                )
                queue_hash = queue_proof.survivor_queue_hash
            report = _build_prefilter_receipt(
                conn,
                config_payload,
                run_path=run_path,
                status="prefilter_blocked_errors" if blocked_by_errors else "prefilter_ready_for_w1",
                stop_reason="prefilter_row_errors" if blocked_by_errors else "prefilter_complete",
                elapsed_seconds=time.monotonic() - start,
                cache_dir=cache_path,
                min_free_bytes=min_free_bytes,
                survivor_queue_ready=not blocked_by_errors,
                survivor_queue_part=blocked_by_errors,
                survivor_queue_hash=queue_hash,
                survivor_queue_row_count=None if queue_proof is None else queue_proof.survivor_queue_row_count,
                survivor_queue_id_index_hash=None if queue_proof is None else queue_proof.survivor_queue_id_index_hash,
                survivor_queue_id_index_row_count=None
                if queue_proof is None
                else queue_proof.survivor_queue_id_index_row_count,
                survivor_candidate_id_set_hash=None if queue_proof is None else queue_proof.survivor_candidate_id_set_hash,
                survivor_private_evidence_id_set_hash=None if queue_proof is None else queue_proof.survivor_private_evidence_id_set_hash,
            )
        _write_json_atomic(run_path / "prefilter_cpu_receipt.json", report)
        return report
    finally:
        conn.close()


def run_prefilter_cpu_stage_for_local_parquet_all_rows(
    *,
    local_schedule: Mapping[str, Any],
    source_metadata_manifest: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    row_provider: Any,
    sparse_scorer: ChunkFilteringScorer,
    run_dir: str | Path,
    deterministic_policy_payload: Mapping[str, Any],
    deterministic_ontology: RiskOntology,
    max_rows: int | None = None,
    cache_dir: str | Path | None = None,
    min_free_bytes: int = 1,
    row_group_shard_count: int = 1,
    row_group_shard_index: int = 0,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
    supervisor_manifest_hash: str | None = None,
    supervisor_command_hash: str | None = None,
    supervisor_run_dir_sha256: str | None = None,
    supervisor_manifest_path: str | Path | None = None,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> dict[str, Any]:
    plan = build_local_parquet_all_rows_prefilter_plan(
        local_schedule=local_schedule,
        source_metadata_manifest=source_metadata_manifest,
    )
    return run_prefilter_cpu_stage_for_local_parquet_plan(
        prefilter_plan=plan,
        config_payload=config_payload,
        row_provider=row_provider,
        sparse_scorer=sparse_scorer,
        run_dir=run_dir,
        deterministic_policy_payload=deterministic_policy_payload,
        deterministic_ontology=deterministic_ontology,
        max_rows=max_rows,
        cache_dir=cache_dir,
        min_free_bytes=min_free_bytes,
        row_group_shard_count=row_group_shard_count,
        row_group_shard_index=row_group_shard_index,
        row_group_sharding_policy=row_group_sharding_policy,
        supervisor_manifest_hash=supervisor_manifest_hash,
        supervisor_command_hash=supervisor_command_hash,
        supervisor_run_dir_sha256=supervisor_run_dir_sha256,
        supervisor_manifest_path=supervisor_manifest_path,
        segmentation_validation_mode=segmentation_validation_mode,
        segmentation_validation_sample_modulus=segmentation_validation_sample_modulus,
    )


def run_prefilter_cpu_stage_for_local_parquet_plan(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    row_provider: Any,
    sparse_scorer: ChunkFilteringScorer,
    run_dir: str | Path,
    deterministic_policy_payload: Mapping[str, Any],
    deterministic_ontology: RiskOntology,
    max_rows: int | None = None,
    cache_dir: str | Path | None = None,
    min_free_bytes: int = 1,
    row_group_shard_count: int = 1,
    row_group_shard_index: int = 0,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
    supervisor_manifest_hash: str | None = None,
    supervisor_command_hash: str | None = None,
    supervisor_run_dir_sha256: str | None = None,
    supervisor_manifest_path: str | Path | None = None,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    if sparse_scorer.model_id != CHUNK_SPARSE_MODEL_ID:
        raise ValueError("prefilter-cpu local all-row mode requires the S1 sparse scorer only")
    if not isinstance(row_provider, LocalParquetRowProvider):
        raise ValueError("local Parquet prefilter requires LocalParquetRowProvider")
    if config_payload.get("deterministic_filter_policy_hash") != deterministic_policy_payload.get(
        "deterministic_filter_policy_hash"
    ):
        raise ValueError("prefilter-cpu deterministic filter policy hash mismatch")
    validate_deterministic_filter_policy_payload(deterministic_policy_payload)
    deterministic_runtime = build_deterministic_filter_runtime(deterministic_ontology, deterministic_policy_payload)
    sparse_scorer_metadata_required = _sparse_scorer_requires_metadata(sparse_scorer)
    segment_mode = _validate_segmentation_validation_mode(segmentation_validation_mode)
    segment_sample_modulus = _segmentation_sample_modulus(segmentation_validation_sample_modulus)
    plan = dict(prefilter_plan)
    _validate_local_parquet_prefilter_plan(plan)
    if plan["source_metadata_hash"] != config_payload["source_metadata_hash"]:
        raise ValueError("prefilter-cpu local Parquet plan source metadata hash mismatch")
    if plan["artifact_kind"] == LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND and max_rows is not None:
        raise ValueError("token tranche prefilter does not support max_rows; build a smaller tranche plan")
    row_cap = None if max_rows is None else _positive_int(max_rows, "max_rows")
    row_group_fetch_batch_size = max(1, int(getattr(row_provider, "row_group_batch_size", 1)))
    shard_policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    supervisor_binding_values = (
        supervisor_manifest_hash,
        supervisor_command_hash,
        supervisor_run_dir_sha256,
        supervisor_manifest_path,
    )
    supervisor_receipt_fields: dict[str, Any] = {}
    if any(value is not None for value in supervisor_binding_values):
        if not all(value is not None for value in supervisor_binding_values):
            raise ValueError("prefilter supervisor receipt binding requires manifest path, manifest, command, and run-dir hashes")
        supervisor_receipt_fields = _validate_prefilter_supervisor_child_binding(
            prefilter_plan=plan,
            config_payload=config_payload,
            supervisor_manifest_path=supervisor_manifest_path,
            supervisor_manifest_hash=supervisor_manifest_hash,
            supervisor_command_hash=supervisor_command_hash,
            supervisor_run_dir_sha256=supervisor_run_dir_sha256,
            row_group_shard_count=row_group_shard_count,
            row_group_shard_index=row_group_shard_index,
            row_group_batch_size=row_group_fetch_batch_size,
            max_rows=row_cap,
            min_free_bytes=min_free_bytes,
            row_group_sharding_policy=shard_policy,
            segmentation_validation_mode=segment_mode,
            segmentation_validation_sample_modulus=segment_sample_modulus,
        )
    plan = _shard_local_parquet_prefilter_plan(
        plan,
        shard_count=row_group_shard_count,
        shard_index=row_group_shard_index,
        row_group_sharding_policy=shard_policy,
    )
    expected_rows = int(plan["expected_physical_rows"] if row_cap is None else min(row_cap, int(plan["expected_physical_rows"])))
    diagnostic_receipt_fields = (
        {
            "row_cap_applied": True,
            "max_rows": int(row_cap),
            "claims_scope": "throughput_diagnostic_only",
            "merge_allowed": False,
        }
        if row_cap is not None
        else {}
    )
    segmentation_receipt_fields = {
        "segmentation_validation_mode": segment_mode,
        "segmentation_validation_sample_modulus": segment_sample_modulus,
    }
    run_path = Path(run_dir)
    _ensure_private_dir(run_path)
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        _ensure_private_cache_dir(cache_path)
    _check_disk_watermark([run_path, *([cache_path] if cache_path is not None else [])], min_free_bytes=min_free_bytes)
    conn = _open_stage_state(run_path, config_payload)
    _record_prefilter_expected_counts(
        conn,
        input_schedule_hash=str(plan["schedule_hash"]),
        expected_physical_row_count=expected_rows,
        expected_physical_row_key_set_hash=None,
    )
    start = time.monotonic()
    runtime_metrics = _new_prefilter_runtime_metrics()
    row_group_metrics_by_id = {
        metric_id: row
        for metric_id, row in _load_prefilter_row_group_metrics(run_path).items()
        if row.get("input_schedule_hash", row.get("source_schedule_hash")) == plan["schedule_hash"]
    }
    stop_requested = False
    rows_accounted = int(_prefilter_source_row_summary(conn)["terminal_rows"])
    try:
        entries = list(plan["entries"])
        entry_index = 0
        while entry_index < len(entries):
            fetch_batch: list[tuple[dict[str, Any], list[int]]] = []
            pending_batch_rows = 0
            while entry_index < len(entries) and len(fetch_batch) < row_group_fetch_batch_size:
                iteration_start = time.monotonic()
                if row_cap is not None and rows_accounted + pending_batch_rows >= row_cap:
                    _record_prefilter_elapsed(runtime_metrics, "row_group_iteration", iteration_start)
                    entry_index = len(entries)
                    break
                _increment_prefilter_count(runtime_metrics, "row_groups_considered")
                entry = dict(entries[entry_index])
                entry["source_schedule_hash"] = str(plan["schedule_hash"])
                entry["row_group_metric_source_schedule_hash"] = str(
                    plan.get("local_parquet_prefilter_plan_hash") or plan["schedule_hash"]
                )
                num_rows = int(entry["row_group_num_rows"])
                candidate_offsets: list[int] = []
                for offset in range(num_rows):
                    if row_cap is not None and rows_accounted + pending_batch_rows + len(candidate_offsets) >= row_cap:
                        break
                    candidate_offsets.append(offset)
                terminal_offsets = _prefilter_terminal_offsets_for_entry(conn, entry, candidate_offsets)
                rows_accounted += len(terminal_offsets)
                pending_offsets = [offset for offset in candidate_offsets if offset not in terminal_offsets]
                entry_index += 1
                if not pending_offsets:
                    _record_prefilter_elapsed(runtime_metrics, "row_group_iteration", iteration_start)
                    continue
                if (run_path / "STOP_REQUESTED").exists():
                    stop_requested = True
                    _record_prefilter_elapsed(runtime_metrics, "row_group_iteration", iteration_start)
                    break
                fetch_batch.append((entry, pending_offsets))
                pending_batch_rows += len(pending_offsets)
                _record_prefilter_elapsed(runtime_metrics, "row_group_iteration", iteration_start)
            if stop_requested:
                break
            if not fetch_batch:
                continue
            batch_group_metrics: dict[tuple[str, int], dict[str, Any]] = {}
            raw_group_to_metric_key: dict[tuple[str, int], tuple[str, int]] = {}
            for entry, _pending_offsets in fetch_batch:
                raw_group = (str(entry["source_file_id"]), int(entry["row_group_index"]))
                metric_key = _prefilter_row_group_metric_key(entry)
                raw_group_to_metric_key[raw_group] = metric_key
                batch_group_metrics[metric_key] = _new_prefilter_row_group_metric_state(entry)
            _check_disk_watermark([run_path, *([cache_path] if cache_path is not None else [])], min_free_bytes=min_free_bytes)
            fetch_start = time.monotonic()
            fetch_error_row_payloads: list[dict[str, Any]] = []
            fetch_error_group_keys: set[tuple[str, int]] = set()
            try:
                if hasattr(row_provider, "fetch_row_groups"):
                    fetched_by_group = row_provider.fetch_row_groups(
                        [(dict(entry), list(pending_offsets)) for entry, pending_offsets in fetch_batch]
                    )
                else:
                    fetched_by_group = {}
                    for entry, pending_offsets in fetch_batch:
                        fetched_by_group[(str(entry["source_file_id"]), int(entry["row_group_index"]))] = row_provider.fetch_rows(
                            dict(entry),
                            pending_offsets,
                        )
            except Exception as exc:
                fetched_by_group = {}
                for entry, pending_offsets in fetch_batch:
                    try:
                        fetched_by_group[(str(entry["source_file_id"]), int(entry["row_group_index"]))] = row_provider.fetch_rows(
                            dict(entry),
                            list(pending_offsets),
                        )
                    except Exception as row_group_exc:
                        failed_key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
                        fetch_error_group_keys.add(failed_key)
                        fetch_error_row_payloads.extend(
                            [
                                _prefilter_source_row_payload(
                                    entry=entry,
                                    offset=offset,
                                    row_status="fetch_error",
                                    error_class=type(row_group_exc).__name__,
                                    segmentation_receipt_hash=None,
                                    candidate_count=0,
                                    survivor_count=0,
                                )
                                for offset in pending_offsets
                            ]
                        )
                if not fetched_by_group and not fetch_error_row_payloads:
                    fetch_error_row_payloads = [
                        _prefilter_source_row_payload(
                            entry=entry,
                            offset=offset,
                            row_status="fetch_error",
                            error_class=type(exc).__name__,
                            segmentation_receipt_hash=None,
                            candidate_count=0,
                            survivor_count=0,
                        )
                        for entry, pending_offsets in fetch_batch
                        for offset in pending_offsets
                    ]
            fetch_elapsed = _record_prefilter_elapsed(runtime_metrics, "parquet_fetch", fetch_start)
            for state in batch_group_metrics.values():
                _add_prefilter_row_group_elapsed(
                    state,
                    "parquet_fetch",
                    fetch_elapsed / max(1, len(batch_group_metrics)),
                )
            for raw_group, rows in fetched_by_group.items():
                metric_key = raw_group_to_metric_key.get(raw_group)
                if metric_key in batch_group_metrics:
                    batch_group_metrics[metric_key]["rows_fetched"] = len(rows)
            _increment_prefilter_count(runtime_metrics, "row_groups_fetched", len(fetched_by_group))
            _increment_prefilter_count(runtime_metrics, "rows_fetched", sum(len(rows) for rows in fetched_by_group.values()))
            row_payloads: list[dict[str, Any]] = list(fetch_error_row_payloads)
            candidate_records: list[tuple[tuple[Any, ...], tuple[Any, ...] | None]] = []
            scorer_items: list[tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any], str]] = []
            scorer_texts: list[str] = []
            scorer_metadata: list[dict[str, Any]] = []
            row_infos: list[dict[str, Any]] = []
            for entry, pending_offsets in fetch_batch:
                group_key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
                row_group_metric_key = raw_group_to_metric_key.get(group_key)
                row_group_metric = (
                    batch_group_metrics.get(row_group_metric_key)
                    if row_group_metric_key is not None
                    else None
                )
                if group_key not in fetched_by_group:
                    if group_key not in fetch_error_group_keys:
                        row_payloads.extend(
                            _prefilter_source_row_payload(
                                entry=entry,
                                offset=offset,
                                row_status="fetch_error",
                                error_class="missing_row_group_fetch_result",
                                segmentation_receipt_hash=None,
                                candidate_count=0,
                                survivor_count=0,
                            )
                            for offset in pending_offsets
                        )
                    continue
                fetched = fetched_by_group[group_key]
                for offset in pending_offsets:
                    row = fetched.get(offset)
                    if isinstance(row, BaseException):
                        row_payloads.append(
                            _prefilter_source_row_payload(
                                entry=entry,
                                offset=offset,
                                row_status="fetch_error",
                                error_class=type(row).__name__,
                                segmentation_receipt_hash=None,
                                candidate_count=0,
                                survivor_count=0,
                            )
                        )
                        continue
                    if not isinstance(row, Mapping):
                        row_payloads.append(
                            _prefilter_source_row_payload(
                                entry=entry,
                                offset=offset,
                                row_status="non_mapping_row",
                                error_class="non_mapping_row",
                                segmentation_receipt_hash=None,
                                candidate_count=0,
                                survivor_count=0,
                            )
                        )
                        continue
                    text = row.get("text")
                    if not isinstance(text, str):
                        segmentation_start = time.monotonic()
                        result = _segment_for_prefilter(
                            SourceDocumentRecord(
                                source_document_id_hash=_source_document_id_for_schedule_row(entry, offset),
                                split="train",
                                raw_text=text,
                                source_lineage_hash=_physical_row_key(entry, offset),
                                source_token_count_hint=_metadata_token_count(row),
                            ),
                            validation_mode=segment_mode,
                            sample_modulus=segment_sample_modulus,
                            metrics=runtime_metrics,
                        )
                        segment_elapsed = _record_prefilter_elapsed(runtime_metrics, "segmentation", segmentation_start)
                        if row_group_metric is not None:
                            _add_prefilter_row_group_elapsed(row_group_metric, "segmentation", segment_elapsed)
                        row_payloads.append(
                            _prefilter_source_row_payload(
                                entry=entry,
                                offset=offset,
                                row_status="missing_text",
                                error_class=str(result.receipt.document_status),
                                segmentation_receipt_hash=result.receipt.receipt_hash,
                                candidate_count=0,
                                survivor_count=0,
                            )
                        )
                        continue
                    segmentation_start = time.monotonic()
                    result = _segment_for_prefilter(
                        SourceDocumentRecord(
                            source_document_id_hash=_source_document_id_for_schedule_row(entry, offset),
                            split="train",
                            raw_text=text,
                            source_lineage_hash=_physical_row_key(entry, offset),
                            source_token_count_hint=_metadata_token_count(row),
                        ),
                        validation_mode=segment_mode,
                        sample_modulus=segment_sample_modulus,
                        metrics=runtime_metrics,
                    )
                    segment_elapsed = _record_prefilter_elapsed(runtime_metrics, "segmentation", segmentation_start)
                    if row_group_metric is not None:
                        _add_prefilter_row_group_elapsed(row_group_metric, "segmentation", segment_elapsed)
                    if result.receipt.document_status != "segmented":
                        row_payloads.append(
                            _prefilter_source_row_payload(
                                entry=entry,
                                offset=offset,
                                row_status="segmentation_error",
                                error_class=str(result.receipt.document_status),
                                segmentation_receipt_hash=result.receipt.receipt_hash,
                                candidate_count=0,
                                survivor_count=0,
                            )
                        )
                        continue
                    candidate_start = time.monotonic()
                    candidates = build_filtering_candidates(
                        result=result,
                        raw_text=text,
                        config_hash=str(config_payload["config_hash"]),
                        target_proxy_tokens=int(config_payload["target_proxy_tokens"]),
                        halo_chars=int(config_payload["halo_chars"]),
                        deterministic_policy_hash=str(deterministic_policy_payload["deterministic_filter_policy_hash"]),
                    )
                    candidate_elapsed = _record_prefilter_elapsed(runtime_metrics, "candidate_construction", candidate_start)
                    if row_group_metric is not None:
                        _add_prefilter_row_group_elapsed(
                            row_group_metric,
                            "candidate_construction",
                            candidate_elapsed,
                        )
                    _increment_prefilter_count(runtime_metrics, "candidates_built", len(candidates))
                    row_info = {
                        "entry": entry,
                        "offset": offset,
                        "row_group_metric_key": row_group_metric_key,
                        "segmentation_receipt_hash": result.receipt.receipt_hash,
                        "candidate_count": len(candidates),
                        "survivor_count": 0,
                    }
                    row_infos.append(row_info)
                    deterministic_normalization_cache: dict[tuple[Any, ...], Any] = {}
                    for candidate in candidates:
                        deterministic_start = time.monotonic()
                        deterministic_result = evaluate_chunk_candidate_deterministic_filter_with_runtime(
                            candidate,
                            raw_text=text,
                            runtime=deterministic_runtime,
                            normalization_cache=deterministic_normalization_cache,
                        )
                        deterministic_elapsed = _record_prefilter_elapsed(
                            runtime_metrics,
                            "deterministic_filter",
                            deterministic_start,
                        )
                        if row_group_metric is not None:
                            _add_prefilter_row_group_elapsed(
                                row_group_metric,
                                "deterministic_filter",
                                deterministic_elapsed,
                            )
                        _increment_prefilter_count(runtime_metrics, "deterministic_candidates")
                        deterministic_decision = _deterministic_candidate_decision(candidate)
                        if deterministic_result.get("terminal_decision") == "reject":
                            if row_group_metric is not None:
                                row_group_metric["route_counts"]["deterministic_reject"] += 1
                            candidate_records.append(
                                _prefilter_candidate_insert_record(
                                    candidate,
                                    prefilter_route="deterministic_reject",
                                    deterministic_result=deterministic_result,
                                    s1_score=None,
                                    private_evidence_text=None,
                                )
                            )
                            continue
                        if deterministic_decision == "quarantine":
                            if row_group_metric is not None:
                                row_group_metric["route_counts"]["prefilter_quarantine"] += 1
                            candidate_records.append(
                                _prefilter_candidate_insert_record(
                                    candidate,
                                    prefilter_route="prefilter_quarantine",
                                    deterministic_result=deterministic_result,
                                    s1_score=None,
                                    private_evidence_text=None,
                                )
                            )
                            continue
                        evidence_text = text[int(candidate["evidence_char_start"]) : int(candidate["evidence_char_end"])]
                        scorer_items.append((row_info, candidate, deterministic_result, evidence_text))
                        scorer_texts.append(evidence_text)
                        if sparse_scorer_metadata_required:
                            scorer_metadata.append(_sparse_scorer_metadata(sparse_scorer, candidate))
            if scorer_items:
                s1_start = time.monotonic()
                scores = list(
                    sparse_scorer.score_many(
                        scorer_texts,
                        scorer_metadata if sparse_scorer_metadata_required else [],
                    )
                )
                s1_elapsed = _record_prefilter_elapsed(runtime_metrics, "s1_feature_score", s1_start)
                _increment_prefilter_count(runtime_metrics, "s1_scored_candidates", len(scorer_items))
                s1_group_counts = Counter(
                    row_info.get("row_group_metric_key")
                    for row_info, _candidate, _deterministic_result, _evidence_text in scorer_items
                    if row_info.get("row_group_metric_key") is not None
                )
                for metric_key, group_count in s1_group_counts.items():
                    state = batch_group_metrics.get(metric_key)
                    if state is not None:
                        _add_prefilter_row_group_elapsed(
                            state,
                            "s1_feature_score",
                            s1_elapsed * (int(group_count) / max(1, len(scorer_items))),
                        )
                if len(scores) != len(scorer_items):
                    scores = [math.nan] * len(scorer_items)
                for (row_info, candidate, deterministic_result, evidence_text), score in zip(
                    scorer_items, scores, strict=True
                ):
                    metric_key = row_info.get("row_group_metric_key")
                    row_group_metric = batch_group_metrics.get(metric_key) if metric_key is not None else None
                    parsed = _safe_runtime_score(score)
                    if parsed is None:
                        route = "prefilter_error"
                    elif parsed >= float(config_payload["sparse_threshold"]):
                        route = "s1_reject"
                    else:
                        route = "prefilter_survivor"
                        row_info["survivor_count"] = int(row_info["survivor_count"]) + 1
                    if row_group_metric is not None:
                        row_group_metric["route_counts"][route] += 1
                    candidate_records.append(
                        _prefilter_candidate_insert_record(
                            candidate,
                            prefilter_route=route,
                            deterministic_result=deterministic_result,
                            s1_score=parsed,
                            private_evidence_text=evidence_text if route == "prefilter_survivor" else None,
                        )
                    )
            row_payloads.extend(
                _prefilter_source_row_payload(
                    entry=row_info["entry"],
                    offset=int(row_info["offset"]),
                    row_status="prefiltered",
                    error_class=None,
                    segmentation_receipt_hash=str(row_info["segmentation_receipt_hash"]),
                    candidate_count=int(row_info["candidate_count"]),
                    survivor_count=int(row_info["survivor_count"]),
                    )
                for row_info in row_infos
            )
            for row_payload in row_payloads:
                metric_key = (str(row_payload["source_file_id_hash"]), int(row_payload["row_group_index"]))
                state = batch_group_metrics.get(metric_key)
                if state is None:
                    continue
                state["terminal_rows"] = int(state.get("terminal_rows", 0)) + 1
                state["candidate_count"] = int(state.get("candidate_count", 0)) + int(row_payload["candidate_count"])
                state["survivor_count"] = int(state.get("survivor_count", 0)) + int(row_payload["survivor_count"])
            sqlite_start = time.monotonic()
            _bulk_insert_prefilter_batch(conn, source_row_payloads=row_payloads, candidate_records=candidate_records)
            sqlite_elapsed = _record_prefilter_elapsed(runtime_metrics, "sqlite_write", sqlite_start)
            terminal_rows_in_batch = sum(int(state.get("terminal_rows", 0)) for state in batch_group_metrics.values())
            for state in batch_group_metrics.values():
                share = int(state.get("terminal_rows", 0)) / max(1, terminal_rows_in_batch)
                _add_prefilter_row_group_elapsed(state, "sqlite_write", sqlite_elapsed * share)
                metric_payload = _prefilter_row_group_metric_payload(state)
                row_group_metrics_by_id[str(metric_payload["row_group_metric_id"])] = metric_payload
            rows_accounted += pending_batch_rows
            _check_disk_watermark([run_path, *([cache_path] if cache_path is not None else [])], min_free_bytes=min_free_bytes)
            if (run_path / "STOP_REQUESTED").exists():
                stop_requested = True
                break
        queue_path = run_path / "w1_survivor_queue.jsonl"
        if stop_requested:
            queue_start = time.monotonic()
            _write_jsonl_path_streaming(
                queue_path.with_suffix(queue_path.suffix + ".part"),
                _iter_prefilter_survivor_queue_rows(conn, config_payload),
            )
            _record_prefilter_elapsed(runtime_metrics, "survivor_queue_export", queue_start)
            row_group_metric_fields = _write_prefilter_row_group_metrics(run_path, row_group_metrics_by_id)
            receipt_start = time.monotonic()
            report = _build_prefilter_receipt(
                conn,
                config_payload,
                run_path=run_path,
                status="stopped",
                stop_reason="stop_requested",
                elapsed_seconds=time.monotonic() - start,
                cache_dir=cache_path,
                min_free_bytes=min_free_bytes,
                extra_fields=_local_parquet_prefilter_plan_receipt_fields(plan)
                | supervisor_receipt_fields
                | diagnostic_receipt_fields
                | segmentation_receipt_fields
                | row_group_metric_fields,
            )
            _record_prefilter_elapsed(runtime_metrics, "receipt_build", receipt_start)
            report = _attach_prefilter_runtime_receipt_fields(report, runtime_metrics)
        else:
            final_row_counts = _prefilter_source_row_summary(conn)
            blocked_by_errors = bool(final_row_counts["row_error_counts"])
            queue_proof = None
            if blocked_by_errors or row_cap is not None:
                queue_hash = None
                queue_start = time.monotonic()
                _write_jsonl_path_streaming(
                    queue_path.with_suffix(queue_path.suffix + ".part"),
                    _iter_prefilter_survivor_queue_rows(conn, config_payload),
                )
                _record_prefilter_elapsed(runtime_metrics, "survivor_queue_export", queue_start)
            else:
                queue_start = time.monotonic()
                queue_proof = _write_survivor_queue_ready_streaming(
                    queue_path,
                    _iter_prefilter_survivor_queue_rows(conn, config_payload),
                )
                queue_hash = queue_proof.survivor_queue_hash
                _record_prefilter_elapsed(runtime_metrics, "survivor_queue_export", queue_start)
            if row_cap is not None:
                ready_status = "prefilter_diagnostic_complete"
            else:
                ready_status = (
                    "prefilter_shard_ready_for_merge"
                    if int(plan.get("row_group_shard_count", 1)) > 1
                    else "prefilter_ready_for_w1"
                )
            row_group_metric_fields = _write_prefilter_row_group_metrics(run_path, row_group_metrics_by_id)
            receipt_start = time.monotonic()
            report = _build_prefilter_receipt(
                conn,
                config_payload,
                run_path=run_path,
                status="prefilter_blocked_errors" if blocked_by_errors else ready_status,
                stop_reason=(
                    "prefilter_row_errors"
                    if blocked_by_errors
                    else "diagnostic_row_cap_reached"
                    if row_cap is not None
                    else "prefilter_complete"
                ),
                elapsed_seconds=time.monotonic() - start,
                cache_dir=cache_path,
                min_free_bytes=min_free_bytes,
                survivor_queue_ready=not blocked_by_errors and row_cap is None,
                survivor_queue_part=blocked_by_errors or row_cap is not None,
                survivor_queue_hash=queue_hash,
                survivor_queue_row_count=None if queue_proof is None else queue_proof.survivor_queue_row_count,
                survivor_queue_id_index_hash=None if queue_proof is None else queue_proof.survivor_queue_id_index_hash,
                survivor_queue_id_index_row_count=None
                if queue_proof is None
                else queue_proof.survivor_queue_id_index_row_count,
                survivor_candidate_id_set_hash=None if queue_proof is None else queue_proof.survivor_candidate_id_set_hash,
                survivor_private_evidence_id_set_hash=None if queue_proof is None else queue_proof.survivor_private_evidence_id_set_hash,
                extra_fields=_local_parquet_prefilter_plan_receipt_fields(plan)
                | supervisor_receipt_fields
                | diagnostic_receipt_fields
                | segmentation_receipt_fields
                | row_group_metric_fields,
            )
            _record_prefilter_elapsed(runtime_metrics, "receipt_build", receipt_start)
            report = _attach_prefilter_runtime_receipt_fields(report, runtime_metrics)
        _write_json_atomic(run_path / "prefilter_cpu_receipt.json", report)
        return report
    finally:
        conn.close()


def _copy_prefilter_table_rows(
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    *,
    table: str,
    columns: Sequence[str],
    batch_size: int = 10_000,
) -> None:
    if int(batch_size) < 1:
        raise ValueError("prefilter merge copy batch_size must be positive")
    placeholders = ", ".join("?" for _ in columns)
    cursor = source.execute(f"select {', '.join(columns)} from {table} order by {columns[0]}")
    while True:
        rows = cursor.fetchmany(int(batch_size))
        if not rows:
            return
        try:
            target.executemany(
                f"insert into {table}({', '.join(columns)}) values ({placeholders})",
                [tuple(row[column] for column in columns) for row in rows],
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"merged prefilter shard contains duplicate or conflicting {table} rows") from exc


def _prefilter_merged_plan_receipt_fields(
    plan: Mapping[str, Any],
    receipts_by_index: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    shard_count = len(receipts_by_index)
    extra_fields = _local_parquet_prefilter_plan_receipt_fields(plan)
    extra_fields.update({
        "merged_prefilter_shards": True,
        "merged_shard_count": shard_count,
        "merged_shard_indexes": sorted(receipts_by_index),
        "merged_shard_receipt_hashes": sorted(str(receipt["stage_receipt_hash"]) for receipt in receipts_by_index.values()),
        "row_group_sharding_enabled": True,
        "row_group_shard_count": shard_count,
        "row_group_shard_row_group_count": int(plan["row_group_count"]),
        "row_group_shard_expected_physical_rows": int(plan["expected_physical_rows"]),
        "all_rows_row_group_count": int(plan["row_group_count"]),
        "all_rows_expected_physical_rows": int(plan["expected_physical_rows"]),
    })
    if plan["artifact_kind"] == LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND:
        extra_fields.update(
            {
                "all_rows_row_group_count": int(plan["population_row_group_count"]),
                "all_rows_expected_physical_rows": int(plan["population_expected_physical_rows"]),
                "full_corpus_census": False,
                "selected_equals_population": False,
                "claims_scope": "tranche_only",
            }
        )
    return extra_fields


def _validate_existing_prefilter_supervisor_merge(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    validate_private_filtering_run_dir(run_path)
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=prefilter_plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
        expect_merge_allowed=True,
    )
    shard_receipts_by_index: dict[int, dict[str, Any]] = {}
    shard_dirs = []
    for shard in manifest["shards"]:
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
        shard_dirs.append(shard_dir)
        receipt = _validate_prefilter_supervisor_ready_shard(
            shard_run_dir=shard_dir,
            shard=shard,
            manifest=manifest,
            prefilter_plan=prefilter_plan,
            config_payload=config_payload,
        )
        shard_receipts_by_index[int(shard["shard_index"])] = receipt
    if any(path.resolve(strict=False) == run_path.resolve(strict=False) for path in shard_dirs):
        raise ValueError("merged prefilter run dir must be separate from shard run dirs")
    queue_row_count = _validate_survivor_queue_ready_file(
        run_path / "w1_survivor_queue.jsonl",
        config_payload,
        allow_empty=True,
    )
    conn = _open_stage_state(run_path, config_payload)
    try:
        _validate_prefilter_ready_receipt(
            conn,
            run_path,
            config_payload,
            queue_row_count=queue_row_count,
        )
        receipt = _load_json(run_path / "prefilter_cpu_receipt.json")
    finally:
        conn.close()
    if not isinstance(receipt, Mapping):
        raise ValueError("merged prefilter receipt must be an object")
    validate_public_no_raw_payload(receipt)
    expected_fields = _prefilter_merged_plan_receipt_fields(prefilter_plan, shard_receipts_by_index)
    for field, expected_value in expected_fields.items():
        if receipt.get(field) != expected_value:
            raise ValueError(f"merged prefilter receipt {field} mismatch")
    if receipt.get("input_schedule_hash") != prefilter_plan["schedule_hash"]:
        raise ValueError("merged prefilter input schedule hash mismatch")
    if int(receipt.get("terminal_rows", -1)) != int(prefilter_plan["expected_physical_rows"]):
        raise ValueError("merged prefilter terminal row count mismatch")
    if int(receipt.get("survivor_count", -1)) != sum(
        int(shard_receipt.get("survivor_count", 0)) for shard_receipt in shard_receipts_by_index.values()
    ):
        raise ValueError("merged prefilter survivor count mismatch")
    return dict(receipt)


def _w1_stop_requested(run_path: Path, extra_stop_paths: Sequence[str | Path] = ()) -> bool:
    if (run_path / "STOP_REQUESTED").exists():
        return True
    return any(Path(path).exists() for path in extra_stop_paths)


def _iter_sized_chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    chunk_size = max(1, int(size))
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _fetch_w1_private_evidence_texts(
    conn: sqlite3.Connection,
    private_evidence_ids: Sequence[str],
) -> dict[str, str]:
    fetched: dict[str, str] = {}
    for chunk in _iter_sized_chunks(list(dict.fromkeys(private_evidence_ids)), 900):
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"""
            select private_evidence_row_id, private_evidence_text
            from private_w1_evidence
            where private_evidence_row_id in ({placeholders})
            """,
            tuple(chunk),
        ):
            fetched[str(row["private_evidence_row_id"])] = str(row["private_evidence_text"])
    return fetched


def _fetch_w1_candidate_jsons(
    conn: sqlite3.Connection,
    candidate_ids: Sequence[str],
) -> dict[str, str]:
    fetched: dict[str, str] = {}
    for chunk in _iter_sized_chunks(list(dict.fromkeys(candidate_ids)), 900):
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"""
            select candidate_id_hash, candidate_json
            from prefilter_candidates
            where candidate_id_hash in ({placeholders})
            """,
            tuple(chunk),
        ):
            fetched[str(row["candidate_id_hash"])] = str(row["candidate_json"])
    return fetched


@dataclass(frozen=True)
class _W1ScoringInputRecord:
    candidate_json: str | None
    evidence_candidate_id_hash: str | None
    private_evidence_text: str | None


def _fetch_w1_scoring_input_records(
    conn: sqlite3.Connection,
    survivor_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], _W1ScoringInputRecord]:
    requested_pairs = list(
        dict.fromkeys(
            (
                _require_sha256(row.get("candidate_id_hash"), "candidate_id_hash"),
                _require_sha256(row.get("private_evidence_row_id"), "private_evidence_row_id"),
            )
            for row in survivor_rows
        )
    )
    fetched: dict[tuple[str, str], _W1ScoringInputRecord] = {}
    for chunk in _iter_sized_chunks(requested_pairs, 450):
        placeholders = ",".join("(?, ?)" for _ in chunk)
        params: list[str] = []
        for candidate_id, evidence_id in chunk:
            params.extend([candidate_id, evidence_id])
        for row in conn.execute(
            f"""
            with requested(candidate_id_hash, private_evidence_row_id) as (
                values {placeholders}
            )
            select r.candidate_id_hash as requested_candidate_id_hash,
                   r.private_evidence_row_id as requested_private_evidence_row_id,
                   c.candidate_json as candidate_json,
                   e.candidate_id_hash as evidence_candidate_id_hash,
                   e.private_evidence_text as private_evidence_text
            from requested r
            left join prefilter_candidates c
              on c.candidate_id_hash = r.candidate_id_hash
            left join private_w1_evidence e
              on e.private_evidence_row_id = r.private_evidence_row_id
            """,
            tuple(params),
        ):
            candidate_id = _require_sha256(row["requested_candidate_id_hash"], "candidate_id_hash")
            evidence_id = _require_sha256(row["requested_private_evidence_row_id"], "private_evidence_row_id")
            fetched[(candidate_id, evidence_id)] = _W1ScoringInputRecord(
                candidate_json=None if row["candidate_json"] is None else str(row["candidate_json"]),
                evidence_candidate_id_hash=(
                    None if row["evidence_candidate_id_hash"] is None else str(row["evidence_candidate_id_hash"])
                ),
                private_evidence_text=None if row["private_evidence_text"] is None else str(row["private_evidence_text"]),
            )
    return fetched


def _w1_batch_scoring_inputs(
    conn: sqlite3.Connection,
    survivor_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[Mapping[str, Any]],
    list[str],
    list[dict[str, Any]],
    list[tuple[Mapping[str, Any], float | None, str, str | None]],
]:
    if not survivor_rows:
        return [], [], [], []
    input_records = _fetch_w1_scoring_input_records(conn, survivor_rows)
    scorable_rows: list[Mapping[str, Any]] = []
    texts: list[str] = []
    metadata: list[dict[str, Any]] = []
    error_records: list[tuple[Mapping[str, Any], float | None, str, str | None]] = []
    for row in survivor_rows:
        candidate_id = _require_sha256(row.get("candidate_id_hash"), "candidate_id_hash")
        evidence_id = _require_sha256(row.get("private_evidence_row_id"), "private_evidence_row_id")
        input_record = input_records[(candidate_id, evidence_id)]
        text = input_record.private_evidence_text
        if text is None:
            error_records.append((row, None, "w1_error", "missing_private_evidence"))
            continue
        if input_record.evidence_candidate_id_hash != candidate_id:
            raise ValueError("W1 private evidence candidate binding mismatch")
        if _sha256_text(text) != row["evidence_text_sha256"]:
            error_records.append((row, None, "w1_error", "evidence_hash_mismatch"))
            continue
        candidate_json = input_record.candidate_json
        if candidate_json is None:
            raise ValueError("W1 candidate metadata is missing from SQLite state")
        candidate = json.loads(candidate_json)
        if str(candidate.get("candidate_id_hash")) != candidate_id:
            raise ValueError("W1 candidate metadata id mismatch")
        scorable_rows.append(row)
        texts.append(text)
        metadata.append(_candidate_public_metadata(candidate))
    return scorable_rows, texts, metadata, error_records


def _is_cuda_out_of_memory_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "cuda" in message
        and "out of memory" in message
    ) or "cublas_status_alloc_failed" in message or "cuda_error_out_of_memory" in message


def _clear_cuda_cache_for_scorer(w1_scorer: ChunkFilteringScorer) -> None:
    torch_module = getattr(w1_scorer, "torch", None)
    cuda_module = getattr(torch_module, "cuda", None)
    empty_cache = getattr(cuda_module, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


_W1_INNER_TELEMETRY_SUM_FIELDS = {
    "tokenizer_call_count": "score_tokenizer_call_count",
    "tokenized_window_count": "score_tokenized_window_count",
    "tokenized_input_cell_count": "score_tokenized_input_cell_count",
    "tokenized_attention_token_count": "score_tokenized_attention_token_count",
    "tokenized_padding_cell_count": "score_tokenized_padding_cell_count",
    "tokenization_seconds": "score_tokenization_seconds",
    "device_transfer_seconds": "score_device_transfer_seconds",
    "window_forward_call_count": "score_window_forward_call_count",
    "window_forward_window_count": "score_window_forward_window_count",
    "window_forward_seconds": "score_window_forward_seconds",
    "pooling_seconds": "score_pooling_seconds",
    "postprocess_seconds": "score_postprocess_seconds",
    "cuda_synchronization_count": "score_cuda_synchronization_count",
    "pooling_ragged_topk_call_count": "score_pooling_ragged_topk_call_count",
    "pooling_ragged_topk_iteration_count": "score_pooling_ragged_topk_iteration_count",
    "pooling_ragged_topk_input_window_count": "score_pooling_ragged_topk_input_window_count",
    "pooling_ragged_topk_document_count": "score_pooling_ragged_topk_document_count",
    "pooling_dense_padded_cell_count_avoided": "score_pooling_dense_padded_cell_count_avoided",
    "pin_memory_seconds": "score_pin_memory_seconds",
}
_W1_INNER_TELEMETRY_MAX_FIELDS = {
    "max_windows_per_forward_call": "score_max_windows_per_forward_call",
    "max_tokenized_windows_per_document_batch": "score_max_tokenized_windows_per_document_batch",
    "max_tokenized_sequence_length": "score_max_tokenized_sequence_length",
    "max_windows_per_document": "score_max_windows_per_document",
}
_W1_INNER_TELEMETRY_BOOL_OR_FIELDS = {
    "cuda_synchronized_timing": "score_cuda_synchronized_timing_observed",
}


def _estimated_w1_window_count_for_text(text: str) -> int:
    return max(1, math.ceil(_estimate_proxy_tokens_from_normalized(text) / 512))


def _merge_w1_scorer_inner_telemetry(
    runtime_telemetry: dict[str, Any],
    w1_scorer: ChunkFilteringScorer,
) -> None:
    payload = getattr(w1_scorer, "last_score_many_telemetry", None)
    if not isinstance(payload, Mapping):
        return
    validate_public_no_raw_payload(dict(payload))
    for source_field, target_field in _W1_INNER_TELEMETRY_SUM_FIELDS.items():
        value = payload.get(source_field)
        if value is None:
            continue
        runtime_telemetry[target_field] = runtime_telemetry.get(target_field, 0) + value
    for source_field, target_field in _W1_INNER_TELEMETRY_MAX_FIELDS.items():
        value = payload.get(source_field)
        if value is None:
            continue
        runtime_telemetry[target_field] = max(int(runtime_telemetry.get(target_field, 0)), int(value))
    for source_field, target_field in _W1_INNER_TELEMETRY_BOOL_OR_FIELDS.items():
        value = payload.get(source_field)
        if value is None:
            continue
        runtime_telemetry[target_field] = bool(runtime_telemetry.get(target_field, False)) or bool(value)
        runtime_telemetry["score_cuda_synchronized_timing"] = bool(
            runtime_telemetry.get("score_cuda_synchronized_timing_observed", False)
        )
    if bool(payload.get("pin_memory_requested", False)):
        runtime_telemetry["score_pin_memory_requested_count"] = int(
            runtime_telemetry.get("score_pin_memory_requested_count", 0)
        ) + 1
    if bool(payload.get("pin_memory_effective", False)):
        runtime_telemetry["score_pin_memory_effective_count"] = int(
            runtime_telemetry.get("score_pin_memory_effective_count", 0)
        ) + 1


def _w1_positive_int_attr(w1_scorer: ChunkFilteringScorer, attr: str) -> int:
    value = getattr(w1_scorer, attr, None)
    if value is None:
        raise ValueError(f"W1 scorer missing {attr}")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"W1 scorer {attr} must be positive")
    return parsed


def _normalize_w1_prepare_pipeline(value: str | None) -> str:
    normalized = str(value or "serial").strip().lower().replace("_", "-")
    if normalized in {"serial", "serial-v1"}:
        return W1_PREPARE_PIPELINE_SERIAL
    if normalized in {"bounded-tokenize-overlap", "bounded-tokenize-overlap-v1"}:
        return W1_PREPARE_PIPELINE_BOUNDED_OVERLAP
    raise ValueError("w1_prepare_pipeline must be serial or bounded-tokenize-overlap")


def _normalize_w1_pin_memory(value: str | None) -> str:
    normalized = str(value or W1_PIN_MEMORY_AUTO).strip().lower()
    if normalized not in {W1_PIN_MEMORY_AUTO, W1_PIN_MEMORY_ON, W1_PIN_MEMORY_OFF}:
        raise ValueError("w1_pin_memory must be auto, on, or off")
    return normalized


def _normalize_w1_tokenizers_parallelism(value: str | None) -> str:
    normalized = str(value or W1_TOKENIZERS_PARALLELISM_INHERIT).strip().lower()
    allowed = {
        W1_TOKENIZERS_PARALLELISM_INHERIT,
        W1_TOKENIZERS_PARALLELISM_FALSE,
        W1_TOKENIZERS_PARALLELISM_TRUE,
    }
    if normalized not in allowed:
        raise ValueError("w1_tokenizers_parallelism must be inherit, false, or true")
    return normalized


def _apply_w1_tokenizers_parallelism(value: str) -> tuple[bool, str | None, str | None]:
    requested = _normalize_w1_tokenizers_parallelism(value)
    had_original = "TOKENIZERS_PARALLELISM" in os.environ
    original = os.environ.get("TOKENIZERS_PARALLELISM")
    if requested != W1_TOKENIZERS_PARALLELISM_INHERIT:
        os.environ["TOKENIZERS_PARALLELISM"] = requested
        return had_original, original, requested
    return had_original, original, original


def _restore_w1_tokenizers_parallelism(had_original: bool, original: str | None) -> None:
    if had_original:
        os.environ["TOKENIZERS_PARALLELISM"] = str(original)
    else:
        os.environ.pop("TOKENIZERS_PARALLELISM", None)


def _w1_pin_memory_requested_for_batch(*, pin_memory_mode: str, w1_scorer: ChunkFilteringScorer) -> bool:
    mode = _normalize_w1_pin_memory(pin_memory_mode)
    if mode == W1_PIN_MEMORY_OFF:
        return False
    if mode == W1_PIN_MEMORY_ON:
        return True
    device = getattr(w1_scorer, "device", None)
    device_type = getattr(device, "type", None)
    if device_type is None and device is not None:
        device_type = str(device).split(":", 1)[0]
    return str(device_type or "").lower() == "cuda"


def _positive_optional_int(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{label} must be positive")
    return parsed


def _w1_batching_strategy(max_estimated_windows_per_prepare_batch: int | None) -> str:
    return (
        W1_BATCHING_ESTIMATED_WINDOW_BUDGET
        if max_estimated_windows_per_prepare_batch is not None
        else W1_BATCHING_FIXED_DOCUMENT_COUNT
    )


def _w1_persist_strategy(w1_persist_commit_row_threshold: int | None) -> str:
    return (
        W1_PERSIST_STRATEGY_ROW_THRESHOLD
        if w1_persist_commit_row_threshold is not None
        else W1_PERSIST_STRATEGY_PER_BATCH
    )


def _w1_runtime_execution_envelope(
    *,
    config_payload: Mapping[str, Any],
    w1_scorer: ChunkFilteringScorer,
    stage_batch_size: int,
    w1_prepare_pipeline: str = W1_PREPARE_PIPELINE_SERIAL,
    max_estimated_windows_per_prepare_batch: int | None = None,
    prepare_queue_depth: int = 1,
    prepare_worker_count: int = 0,
    w1_pin_memory: str = W1_PIN_MEMORY_AUTO,
    w1_tokenizers_parallelism_requested: str = W1_TOKENIZERS_PARALLELISM_INHERIT,
    w1_tokenizers_parallelism_effective: str | None = None,
    w1_persist_commit_row_threshold: int | None = None,
) -> dict[str, Any]:
    artifact_max_windows = _w1_positive_int_attr(w1_scorer, "artifact_max_windows_per_device_batch")
    effective_max_windows = _w1_positive_int_attr(w1_scorer, "max_windows_per_device_batch")
    requested_raw = getattr(w1_scorer, "requested_max_windows_per_device_batch", None)
    requested_max_windows = None if requested_raw is None else int(requested_raw)
    if requested_max_windows is not None and requested_max_windows < 1:
        raise ValueError("W1 requested_max_windows_per_device_batch must be positive")
    override_applied = effective_max_windows != artifact_max_windows
    override_source = "cli_explicit" if requested_max_windows is not None else "artifact_default"
    requested_attn_implementation = normalize_w1_attn_implementation(
        getattr(w1_scorer, "w1_attn_implementation_requested", W1_ATTN_IMPLEMENTATION_SDPA)
    )
    observed_attn_implementation = normalize_w1_attn_implementation(
        getattr(w1_scorer, "w1_attn_implementation_observed", requested_attn_implementation)
    )
    normalized_tokenizers_parallelism = _normalize_w1_tokenizers_parallelism(w1_tokenizers_parallelism_requested)
    normalized_persist_threshold = _positive_optional_int(
        w1_persist_commit_row_threshold,
        "w1_persist_commit_row_threshold",
    )
    precision_mode = normalize_w1_precision_mode(str(config_payload.get("window_mil_precision") or LOCKED_W1_PRECISION))
    configured_precision_policy = config_payload.get("window_mil_precision_policy")
    if configured_precision_policy is None:
        precision_policy = build_w1_precision_policy(precision_mode)
    else:
        if not isinstance(configured_precision_policy, Mapping):
            raise ValueError("W1 runtime envelope precision policy must be an object")
        validate_w1_precision_policy(configured_precision_policy)
        precision_policy = dict(configured_precision_policy)
        if precision_mode != str(precision_policy["precision_mode"]):
            raise ValueError("W1 runtime envelope precision policy mismatch")
    payload = {
        "artifact_kind": W1_RUNTIME_EXECUTION_ENVELOPE_KIND,
        "schema_version": 1,
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "window_mil_report_hash": config_payload["window_mil_report_hash"],
        "window_mil_model_file_sha256": config_payload["window_mil_model_file_sha256"],
        "window_mil_model_id": config_payload["window_mil_model_id"],
        "window_mil_threshold": float(config_payload["window_mil_threshold"]),
        "window_mil_input_mode": config_payload["window_mil_input_mode"],
        "window_mil_pooling_mode": config_payload["window_mil_pooling_mode"],
        "window_mil_pooling_top_k": int(config_payload["window_mil_pooling_top_k"]),
        "window_mil_precision": precision_mode,
        "window_mil_precision_policy": precision_policy,
        "window_mil_precision_policy_hash": precision_policy["precision_policy_hash"],
        "window_mil_precision_runtime": runtime_w1_precision_observation(
            getattr(w1_scorer, "torch", None),
            mode=str(getattr(w1_scorer, "window_mil_precision", precision_mode)),
            device=getattr(w1_scorer, "device", None),
        ),
        "w1_attn_implementation_requested": requested_attn_implementation,
        "w1_attn_implementation_observed": observed_attn_implementation,
        "torch_compile": "off_deferred",
        "stage_batch_size": max(1, int(stage_batch_size)),
        "w1_prepare_pipeline": _normalize_w1_prepare_pipeline(w1_prepare_pipeline),
        "w1_batching_strategy": _w1_batching_strategy(max_estimated_windows_per_prepare_batch),
        "max_estimated_windows_per_prepare_batch": _positive_optional_int(
            max_estimated_windows_per_prepare_batch,
            "max_estimated_windows_per_prepare_batch",
        ),
        "prepare_queue_depth": max(1, int(prepare_queue_depth)),
        "prepare_worker_count": max(0, int(prepare_worker_count)),
        "pin_memory_requested": _normalize_w1_pin_memory(w1_pin_memory),
        "w1_tokenizers_parallelism_requested": normalized_tokenizers_parallelism,
        "w1_tokenizers_parallelism_effective": w1_tokenizers_parallelism_effective,
        "w1_persist_strategy": _w1_persist_strategy(normalized_persist_threshold),
        "w1_persist_commit_row_threshold": normalized_persist_threshold,
        "artifact_max_windows_per_device_batch": artifact_max_windows,
        "requested_max_windows_per_device_batch": requested_max_windows,
        "effective_max_windows_per_device_batch": effective_max_windows,
        "override_source": override_source,
        "override_applied": bool(override_applied),
        "claim_scope": "execution_envelope_only",
        "locked_report_artifacts_mutated": False,
    }
    validate_public_no_raw_payload(payload)
    return payload


def _w1_runtime_execution_envelope_hash(envelope: Mapping[str, Any]) -> str:
    envelope_hash = stable_json_hash(dict(envelope))
    return _require_sha256(envelope_hash, "w1_runtime_envelope_hash")


def _validated_w1_runtime_envelope_from_receipt(
    receipt: Mapping[str, Any],
    *,
    requested_w1_runtime_envelope_hash: str | None = None,
    label: str = "W1 score receipt",
) -> tuple[dict[str, Any], str]:
    w1_runtime_envelope = receipt.get("w1_runtime_envelope")
    if not isinstance(w1_runtime_envelope, Mapping):
        raise ValueError(f"{label} missing runtime envelope")
    w1_runtime_envelope_hash = _require_sha256(
        receipt.get("w1_runtime_envelope_hash"),
        "w1_runtime_envelope_hash",
    )
    if _w1_runtime_execution_envelope_hash(w1_runtime_envelope) != w1_runtime_envelope_hash:
        raise ValueError(f"{label} runtime envelope hash mismatch")
    if (
        requested_w1_runtime_envelope_hash is not None
        and w1_runtime_envelope_hash != requested_w1_runtime_envelope_hash
    ):
        raise ValueError(f"{label} runtime envelope mismatch")
    return dict(w1_runtime_envelope), w1_runtime_envelope_hash


def _w1_score_row_runtime_hashes(score_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for row in score_rows:
        hashes.add(_require_sha256(row.get("w1_runtime_envelope_hash"), "w1_runtime_envelope_hash"))
    return hashes


def _validate_existing_w1_runtime_envelope_state(
    *,
    conn: sqlite3.Connection,
    run_path: Path,
    requested_w1_runtime_envelope_hash: str,
) -> None:
    existing_score_rows = [
        json.loads(str(row["score_json"]))
        for row in conn.execute("select score_json from w1_scores order by candidate_id_hash")
    ]
    if existing_score_rows:
        hashes = _w1_score_row_runtime_hashes(existing_score_rows)
        if hashes != {requested_w1_runtime_envelope_hash}:
            raise ValueError("W1 runtime envelope mismatch for existing W1 score rows")
    receipt_path = run_path / "w1_score_receipt.json"
    if not receipt_path.exists():
        return
    receipt = _load_json(receipt_path)
    if not isinstance(receipt, Mapping):
        raise ValueError("existing W1 score receipt must be an object")
    receipt_hash = receipt.get("w1_runtime_envelope_hash")
    if receipt_hash is None:
        if existing_score_rows:
            raise ValueError("W1 runtime envelope missing from existing W1 score receipt")
        return
    _validated_w1_runtime_envelope_from_receipt(
        receipt,
        requested_w1_runtime_envelope_hash=requested_w1_runtime_envelope_hash,
        label="existing W1 score receipt",
    )


def _w1_scorer_device_observation(w1_scorer: ChunkFilteringScorer) -> dict[str, Any]:
    device = getattr(w1_scorer, "device", None)
    payload: dict[str, Any] = {
        "device_present": device is not None,
        "device": None,
        "device_type": None,
        "device_index": None,
        "cuda_available_at_scoring_start": None,
        "cuda_device_name": None,
    }
    if device is not None:
        payload["device"] = str(device)
        device_type = getattr(device, "type", None)
        if device_type is None:
            device_type = str(device).split(":", 1)[0]
        payload["device_type"] = str(device_type)
        device_index = getattr(device, "index", None)
        payload["device_index"] = None if device_index is None else int(device_index)
    torch_module = getattr(w1_scorer, "torch", None)
    cuda_module = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda_module, "is_available", None)
    if callable(is_available):
        payload["cuda_available_at_scoring_start"] = bool(is_available())
    if payload["device_type"] == "cuda" and cuda_module is not None:
        try:
            index = payload["device_index"]
            if index is None:
                current_device = getattr(cuda_module, "current_device", None)
                index = int(current_device()) if callable(current_device) else 0
            get_device_name = getattr(cuda_module, "get_device_name", None)
            if callable(get_device_name):
                payload["cuda_device_name"] = str(get_device_name(index))
        except Exception:
            payload["cuda_device_name"] = None
    validate_public_no_raw_payload(payload)
    return payload


def _score_w1_texts_with_oom_backoff(
    *,
    w1_scorer: ChunkFilteringScorer,
    texts: list[str],
    metadata: list[dict[str, Any]],
    runtime_telemetry: dict[str, Any],
    estimated_window_counts: Sequence[int] | None = None,
) -> list[float]:
    if not texts:
        return []
    if estimated_window_counts is None:
        attempted_window_counts = [0 for _ in texts]
    else:
        attempted_window_counts = [int(value) for value in estimated_window_counts]
        if len(attempted_window_counts) != len(texts):
            raise ValueError("W1 estimated_window_counts must match texts")
    runtime_telemetry["score_model_call_count"] += 1
    model_at = time.monotonic()
    try:
        setattr(w1_scorer, "last_score_many_telemetry", {})
    except Exception:
        pass
    try:
        scores = list(w1_scorer.score_many(texts, metadata))
        _merge_w1_scorer_inner_telemetry(runtime_telemetry, w1_scorer)
    except Exception as exc:
        runtime_telemetry["score_model_seconds"] += time.monotonic() - model_at
        if len(texts) > 1 and _is_cuda_out_of_memory_error(exc):
            _merge_w1_scorer_inner_telemetry(runtime_telemetry, w1_scorer)
            attempted_windows = sum(attempted_window_counts)
            runtime_telemetry["score_oom_split_count"] += 1
            runtime_telemetry["score_oom_failed_attempt_count"] += 1
            runtime_telemetry["score_oom_failed_attempt_docs"] += len(texts)
            runtime_telemetry["score_oom_failed_attempt_estimated_windows"] += attempted_windows
            runtime_telemetry["score_oom_failed_attempt_max_docs"] = max(
                int(runtime_telemetry.get("score_oom_failed_attempt_max_docs", 0)),
                len(texts),
            )
            runtime_telemetry["score_oom_failed_attempt_max_estimated_windows"] = max(
                int(runtime_telemetry.get("score_oom_failed_attempt_max_estimated_windows", 0)),
                attempted_windows,
            )
            _clear_cuda_cache_for_scorer(w1_scorer)
            midpoint = max(1, len(texts) // 2)
            return _score_w1_texts_with_oom_backoff(
                w1_scorer=w1_scorer,
                texts=texts[:midpoint],
                metadata=metadata[:midpoint],
                runtime_telemetry=runtime_telemetry,
                estimated_window_counts=attempted_window_counts[:midpoint],
            ) + _score_w1_texts_with_oom_backoff(
                w1_scorer=w1_scorer,
                texts=texts[midpoint:],
                metadata=metadata[midpoint:],
                runtime_telemetry=runtime_telemetry,
                estimated_window_counts=attempted_window_counts[midpoint:],
            )
        raise
    runtime_telemetry["score_model_seconds"] += time.monotonic() - model_at
    runtime_telemetry["scorer_batch_count"] += 1
    runtime_telemetry["scorer_input_docs"] += len(texts)
    runtime_telemetry["score_max_effective_batch_docs"] = max(
        int(runtime_telemetry.get("score_max_effective_batch_docs", 0)),
        len(texts),
    )
    return scores


@dataclass(frozen=True)
class _FetchedW1StageBatch:
    survivor_rows: tuple[Mapping[str, Any], ...]
    scorable_rows: tuple[Mapping[str, Any], ...]
    texts: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]
    score_records: tuple[tuple[Mapping[str, Any], float | None, str, str | None], ...]
    estimated_window_counts: tuple[int, ...]


@dataclass(frozen=True)
class _PreparedW1StageBatch:
    fetched: _FetchedW1StageBatch
    prepared: Any | None
    input_fetch_seconds: float = 0.0


def _queue_row_estimated_w1_window_count(row: Mapping[str, Any]) -> int | None:
    raw = row.get("estimated_w1_window_count")
    if raw is None:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError("survivor queue estimated_w1_window_count must be positive")
    return value


def _scorable_w1_estimated_window_counts(
    scorable_rows: Sequence[Mapping[str, Any]],
    texts: Sequence[str],
) -> list[int]:
    estimated_window_counts: list[int] = []
    for row, text in zip(scorable_rows, texts, strict=True):
        estimated = None
        if row.get("evidence_proxy_token_count") is not None:
            estimated = _queue_row_estimated_w1_window_count(row)
        if estimated is None:
            estimated = _estimated_w1_window_count_for_text(text)
        estimated_window_counts.append(int(estimated))
    return estimated_window_counts


def _iter_w1_pending_batches(
    pending: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_estimated_windows_per_prepare_batch: int | None,
) -> Iterator[tuple[list[Mapping[str, Any]], int, bool]]:
    batch = max(1, int(batch_size))
    window_budget = _positive_optional_int(
        max_estimated_windows_per_prepare_batch,
        "max_estimated_windows_per_prepare_batch",
    )
    if window_budget is None:
        for start_index in range(0, len(pending), batch):
            chunk = list(pending[start_index:start_index + batch])
            yield chunk, sum(_queue_row_estimated_w1_window_count(row) or 0 for row in chunk), False
        return
    if any(_queue_row_estimated_w1_window_count(row) is None for row in pending):
        for start_index in range(0, len(pending), batch):
            chunk = list(pending[start_index:start_index + batch])
            yield chunk, sum(_queue_row_estimated_w1_window_count(row) or 0 for row in chunk), False
        return
    current: list[Mapping[str, Any]] = []
    current_windows = 0
    for row in pending:
        estimated = int(_queue_row_estimated_w1_window_count(row) or 1)
        if current and (len(current) >= batch or current_windows + estimated > window_budget):
            yield current, current_windows, len(current) == 1 and current_windows > window_budget
            current = []
            current_windows = 0
        current.append(row)
        current_windows += estimated
        if len(current) >= batch:
            yield current, current_windows, len(current) == 1 and current_windows > window_budget
            current = []
            current_windows = 0
    if current:
        yield current, current_windows, len(current) == 1 and current_windows > window_budget


def _w1_prepare_pipeline_available(w1_scorer: ChunkFilteringScorer) -> bool:
    return callable(getattr(w1_scorer, "prepare_many", None)) and callable(getattr(w1_scorer, "score_prepared", None))


def _w1_prepare_pipeline_disabled_reason(w1_scorer: ChunkFilteringScorer) -> str | None:
    if not _w1_prepare_pipeline_available(w1_scorer):
        return "scorer_prepare_api_unavailable"
    variant = getattr(w1_scorer, "window_variant_config", None)
    input_mode = None
    if isinstance(variant, Mapping):
        input_mode = variant.get("input_mode")
    if str(input_mode or WINDOW_INPUT_FULL_EVIDENCE) == WINDOW_INPUT_SPLIT_HALO_CAPPED:
        return "split_halo_not_supported"
    return None


def _fetch_w1_stage_batch(
    conn: sqlite3.Connection,
    chunk: Sequence[Mapping[str, Any]],
) -> _FetchedW1StageBatch:
    scorable_rows, texts, metadata, score_records = _w1_batch_scoring_inputs(conn, chunk)
    estimated_window_counts = _scorable_w1_estimated_window_counts(scorable_rows, texts)
    return _FetchedW1StageBatch(
        survivor_rows=tuple(chunk),
        scorable_rows=tuple(scorable_rows),
        texts=tuple(texts),
        metadata=tuple(metadata),
        score_records=tuple(score_records),
        estimated_window_counts=tuple(int(value) for value in estimated_window_counts),
    )


def _prepare_w1_stage_batch(
    *,
    w1_scorer: ChunkFilteringScorer,
    fetched: _FetchedW1StageBatch,
    pin_memory: bool,
) -> _PreparedW1StageBatch:
    if not fetched.texts:
        return _PreparedW1StageBatch(fetched=fetched, prepared=None)
    prepared = w1_scorer.prepare_many(list(fetched.texts), list(fetched.metadata), pin_memory=bool(pin_memory))  # type: ignore[attr-defined]
    return _PreparedW1StageBatch(fetched=fetched, prepared=prepared)


def _fetch_and_prepare_w1_stage_batch(
    *,
    conn: sqlite3.Connection,
    chunk: Sequence[Mapping[str, Any]],
    w1_scorer: ChunkFilteringScorer,
    pin_memory: bool,
) -> _PreparedW1StageBatch:
    input_fetch_at = time.monotonic()
    fetched = _fetch_w1_stage_batch(conn, chunk)
    input_fetch_seconds = time.monotonic() - input_fetch_at
    if not fetched.texts:
        return _PreparedW1StageBatch(
            fetched=fetched,
            prepared=None,
            input_fetch_seconds=input_fetch_seconds,
        )
    prepared = w1_scorer.prepare_many(list(fetched.texts), list(fetched.metadata), pin_memory=bool(pin_memory))  # type: ignore[attr-defined]
    return _PreparedW1StageBatch(
        fetched=fetched,
        prepared=prepared,
        input_fetch_seconds=input_fetch_seconds,
    )


def _slice_prepared_w1_batch(prepared: Any, start: int, end: int) -> Any:
    slice_method = getattr(prepared, "slice", None)
    if callable(slice_method):
        return slice_method(start, end)
    metadata = list(getattr(prepared, "metadata", ()))
    texts = list(getattr(prepared, "texts", ()))
    candidate_ids = list(getattr(prepared, "candidate_ids", ()))
    if not metadata and not texts:
        raise ValueError("prepared W1 batch cannot be sliced")
    sliced_metadata = metadata[start:end]
    sliced_texts = texts[start:end] if texts else []
    window_count = max(1, int(getattr(prepared, "window_count", max(1, len(sliced_metadata)))) * max(1, end - start) // max(1, int(getattr(prepared, "document_count", len(metadata) or len(texts) or 1))))
    return SimpleNamespace(
        texts=sliced_texts,
        metadata=sliced_metadata,
        candidate_ids=candidate_ids[start:end],
        document_count=end - start,
        window_count=window_count,
        padded_token_count=max(1, int(getattr(prepared, "padded_token_count", window_count))),
        pin_memory_requested=bool(getattr(prepared, "pin_memory_requested", False)),
        pin_memory_effective=bool(getattr(prepared, "pin_memory_effective", False)),
        telemetry=dict(getattr(prepared, "telemetry", {}) or {}),
    )


def _score_w1_prepared_with_oom_backoff(
    *,
    w1_scorer: ChunkFilteringScorer,
    prepared: Any,
    runtime_telemetry: dict[str, Any],
    estimated_window_counts: Sequence[int],
) -> list[float]:
    document_count = int(getattr(prepared, "document_count", len(estimated_window_counts)))
    if document_count < 1:
        return []
    if len(estimated_window_counts) != document_count:
        raise ValueError("prepared W1 estimated_window_counts must match document_count")
    runtime_telemetry["score_model_call_count"] += 1
    model_at = time.monotonic()
    try:
        setattr(w1_scorer, "last_score_many_telemetry", {})
    except Exception:
        pass
    try:
        scores = list(w1_scorer.score_prepared(prepared))  # type: ignore[attr-defined]
        _merge_w1_scorer_inner_telemetry(runtime_telemetry, w1_scorer)
    except Exception as exc:
        runtime_telemetry["score_model_seconds"] += time.monotonic() - model_at
        if document_count > 1 and _is_cuda_out_of_memory_error(exc):
            _merge_w1_scorer_inner_telemetry(runtime_telemetry, w1_scorer)
            attempted_windows = sum(int(value) for value in estimated_window_counts)
            runtime_telemetry["score_oom_split_count"] += 1
            runtime_telemetry["score_prepared_oom_split_count"] += 1
            runtime_telemetry["score_oom_failed_attempt_count"] += 1
            runtime_telemetry["score_oom_failed_attempt_docs"] += document_count
            runtime_telemetry["score_oom_failed_attempt_estimated_windows"] += attempted_windows
            runtime_telemetry["score_oom_failed_attempt_max_docs"] = max(
                int(runtime_telemetry.get("score_oom_failed_attempt_max_docs", 0)),
                document_count,
            )
            runtime_telemetry["score_oom_failed_attempt_max_estimated_windows"] = max(
                int(runtime_telemetry.get("score_oom_failed_attempt_max_estimated_windows", 0)),
                attempted_windows,
            )
            _clear_cuda_cache_for_scorer(w1_scorer)
            midpoint = max(1, document_count // 2)
            return _score_w1_prepared_with_oom_backoff(
                w1_scorer=w1_scorer,
                prepared=_slice_prepared_w1_batch(prepared, 0, midpoint),
                runtime_telemetry=runtime_telemetry,
                estimated_window_counts=estimated_window_counts[:midpoint],
            ) + _score_w1_prepared_with_oom_backoff(
                w1_scorer=w1_scorer,
                prepared=_slice_prepared_w1_batch(prepared, midpoint, document_count),
                runtime_telemetry=runtime_telemetry,
                estimated_window_counts=estimated_window_counts[midpoint:],
            )
        raise
    runtime_telemetry["score_model_seconds"] += time.monotonic() - model_at
    runtime_telemetry["scorer_batch_count"] += 1
    runtime_telemetry["score_prepared_batch_count"] += 1
    runtime_telemetry["scorer_input_docs"] += document_count
    runtime_telemetry["score_max_effective_batch_docs"] = max(
        int(runtime_telemetry.get("score_max_effective_batch_docs", 0)),
        document_count,
    )
    return scores


def _append_w1_score_records_from_scores(
    *,
    config_payload: Mapping[str, Any],
    scorable_rows: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    score_records: list[tuple[Mapping[str, Any], float | None, str, str | None]],
) -> None:
    if len(scores) != len(scorable_rows):
        scores = [math.nan] * len(scorable_rows)
    for row, score in zip(scorable_rows, scores, strict=True):
        parsed = _safe_runtime_score(score)
        if parsed is None:
            score_records.append((row, None, "w1_error", "nonfinite_w1_score"))
        elif parsed >= float(config_payload["window_mil_threshold"]):
            score_records.append((row, parsed, "w1_reject", None))
        else:
            score_records.append((row, parsed, "w1_pass", None))


def _persist_w1_stage_batch_scores(
    *,
    conn: sqlite3.Connection,
    config_payload: Mapping[str, Any],
    runtime_telemetry: dict[str, Any],
    score_records: Sequence[tuple[Mapping[str, Any], float | None, str, str | None]],
    w1_runtime_envelope_hash: str,
) -> None:
    persist_at = time.monotonic()
    runtime_telemetry["score_rows_attempted"] += len(score_records)
    runtime_telemetry["score_rows_inserted"] += _insert_w1_scores(
        conn,
        list(score_records),
        config_payload=config_payload,
        w1_runtime_envelope_hash=w1_runtime_envelope_hash,
    )
    runtime_telemetry["score_persist_seconds"] += time.monotonic() - persist_at


def _sqlite_connection_path(conn: sqlite3.Connection) -> Path | None:
    db_list = conn.execute("pragma database_list").fetchone()
    if db_list is None:
        return None
    path = str(db_list["file"])
    return Path(path) if path else None


def _sqlite_pragma_evidence(conn: sqlite3.Connection) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name in ("journal_mode", "synchronous", "wal_autocheckpoint", "temp_store", "cache_size", "locking_mode"):
        try:
            row = conn.execute(f"pragma {name}").fetchone()
        except sqlite3.DatabaseError:
            continue
        if row is not None:
            evidence[name] = row[0]
    return evidence


def _flush_w1_stage_score_buffer(
    *,
    conn: sqlite3.Connection,
    config_payload: Mapping[str, Any],
    runtime_telemetry: dict[str, Any],
    pending_score_records: list[tuple[Mapping[str, Any], float | None, str, str | None]],
    w1_runtime_envelope_hash: str,
) -> None:
    if not pending_score_records:
        runtime_telemetry["score_persist_pending_rows"] = 0
        return
    rows_in_commit = len(pending_score_records)
    persist_at = time.monotonic()
    runtime_telemetry["score_rows_attempted"] += rows_in_commit
    runtime_telemetry["score_rows_inserted"] += _insert_w1_scores(
        conn,
        list(pending_score_records),
        config_payload=config_payload,
        w1_runtime_envelope_hash=w1_runtime_envelope_hash,
    )
    runtime_telemetry["score_persist_seconds"] += time.monotonic() - persist_at
    runtime_telemetry["score_persist_commit_count"] += 1
    runtime_telemetry["score_persist_rows_per_commit"].append(rows_in_commit)
    runtime_telemetry["score_persist_max_rows_per_commit"] = max(
        int(runtime_telemetry.get("score_persist_max_rows_per_commit", 0)),
        rows_in_commit,
    )
    pending_score_records.clear()
    runtime_telemetry["score_persist_pending_rows"] = 0


def _buffer_w1_stage_batch_scores(
    *,
    conn: sqlite3.Connection,
    config_payload: Mapping[str, Any],
    runtime_telemetry: dict[str, Any],
    pending_score_records: list[tuple[Mapping[str, Any], float | None, str, str | None]],
    score_records: Sequence[tuple[Mapping[str, Any], float | None, str, str | None]],
    w1_runtime_envelope_hash: str,
    commit_row_threshold: int | None,
) -> None:
    pending_score_records.extend(score_records)
    runtime_telemetry["score_persist_pending_rows"] = len(pending_score_records)
    if commit_row_threshold is None or len(pending_score_records) >= int(commit_row_threshold):
        _flush_w1_stage_score_buffer(
            conn=conn,
            config_payload=config_payload,
            runtime_telemetry=runtime_telemetry,
            pending_score_records=pending_score_records,
            w1_runtime_envelope_hash=w1_runtime_envelope_hash,
        )


def _w1_score_records_accepted_token_lower_bound(
    score_records: Sequence[tuple[Mapping[str, Any], float | None, str, str | None]],
) -> int:
    return sum(
        int(row.get("core_proxy_token_count", 0) or 0)
        for row, _score, route, _error_class in score_records
        if route == "w1_pass"
    )


def _w1_score_rows_accepted_token_lower_bound(
    score_rows: Sequence[Mapping[str, Any]],
    *,
    survivor_by_id: Mapping[str, Mapping[str, Any]],
) -> int:
    total = 0
    for row in score_rows:
        if row.get("w1_route") != "w1_pass":
            continue
        survivor = survivor_by_id.get(str(row.get("candidate_id_hash")))
        if survivor is None:
            raise ValueError("accepted-token ledger score row missing survivor queue row")
        total += int(survivor.get("core_proxy_token_count", 0) or 0)
    return total


def _truncate_w1_score_records_at_accepted_token_target(
    score_records: Sequence[tuple[Mapping[str, Any], float | None, str, str | None]],
    *,
    accepted_token_count_lower_bound: int,
    accepted_token_target_lower_bound: int,
) -> tuple[list[tuple[Mapping[str, Any], float | None, str, str | None]], int, bool]:
    target = _positive_int(accepted_token_target_lower_bound, "accepted_token_target_lower_bound")
    accepted = int(accepted_token_count_lower_bound)
    if accepted >= target:
        return [], accepted, True
    kept: list[tuple[Mapping[str, Any], float | None, str, str | None]] = []
    for record in score_records:
        kept.append(record)
        row, _score, route, _error_class = record
        if route == "w1_pass":
            accepted += int(row.get("core_proxy_token_count", 0) or 0)
            if accepted >= target:
                return kept, accepted, True
    return kept, accepted, False


def _w1_token_ledger_state(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        """
        select next_commit_ordinal, accepted_token_count_lower_bound,
               accepted_token_target_reached, stop_ordinal, total_expected_rows,
               accepted_token_target_lower_bound, updated_at
        from ledger_state where id = 1
        """
    ).fetchone()
    if row is None:
        raise ValueError("accepted-token ledger state is missing")
    return {
        "next_commit_ordinal": int(row["next_commit_ordinal"]),
        "accepted_token_count_lower_bound": int(row["accepted_token_count_lower_bound"]),
        "accepted_token_target_reached": bool(row["accepted_token_target_reached"]),
        "stop_ordinal": None if row["stop_ordinal"] is None else int(row["stop_ordinal"]),
        "total_expected_rows": int(row["total_expected_rows"]),
        "accepted_token_target_lower_bound": int(row["accepted_token_target_lower_bound"]),
        "updated_at": float(row["updated_at"]),
    }


def _w1_token_ledger_setup(conn: sqlite3.Connection) -> None:
    conn.execute("pragma busy_timeout=60000")
    try:
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma synchronous=normal")
    except sqlite3.DatabaseError:
        pass
    conn.execute(
        """
        create table if not exists ledger_metadata(
            key text primary key,
            value_json text not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists ledger_state(
            id integer primary key check (id = 1),
            next_commit_ordinal integer not null,
            accepted_token_count_lower_bound integer not null,
            accepted_token_target_reached integer not null,
            stop_ordinal integer,
            total_expected_rows integer not null,
            accepted_token_target_lower_bound integer not null,
            updated_at real not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists ledger_rows(
            global_ordinal integer primary key,
            shard_index integer not null,
            shard_row_index integer not null,
            candidate_id_hash text not null unique,
            w1_route text not null,
            accepted_token_count integer not null,
            score_json text not null,
            score_json_hash text not null,
            commit_decision text not null,
            producer_rank_assignment_hash text,
            created_at real not null,
            updated_at real not null
        )
        """
    )
    conn.execute("create index if not exists ledger_rows_shard_decision_idx on ledger_rows(shard_index, commit_decision)")
    conn.execute(
        "create index if not exists ledger_rows_shard_decision_row_idx "
        "on ledger_rows(shard_index, commit_decision, shard_row_index)"
    )
    conn.execute("create index if not exists ledger_rows_shard_route_idx on ledger_rows(shard_index, w1_route)")
    conn.execute("create index if not exists ledger_rows_decision_ordinal_idx on ledger_rows(commit_decision, global_ordinal)")


def _connect_w1_accepted_token_ledger(
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> sqlite3.Connection:
    ledger_path = Path(path).expanduser().resolve(strict=False)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ledger_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    _w1_token_ledger_setup(conn)
    target = _positive_int(metadata.get("accepted_token_target_lower_bound"), "accepted_token_target_lower_bound")
    total_expected_rows = _nonnegative_int(metadata.get("survivor_count", 0), "accepted-token ledger survivor_count")
    metadata_payload = dict(metadata)
    metadata_payload["artifact_kind"] = W1_ACCEPTED_TOKEN_LEDGER_KIND
    metadata_payload["schema_version"] = 1
    metadata_payload["accepted_token_target_lower_bound"] = target
    metadata_payload["survivor_count"] = total_expected_rows
    metadata_payload["raw_text_serialized"] = False
    metadata_payload["raw_url_serialized"] = False
    metadata_payload["token_serialized"] = False
    metadata_payload["ledger_binding_hash"] = stable_json_hash(
        {key: value for key, value in metadata_payload.items() if key != "ledger_binding_hash"}
    )
    validate_public_no_raw_payload(metadata_payload)
    expected = {key: json.dumps(value, sort_keys=True) for key, value in sorted(metadata_payload.items())}
    now = time.time()
    with conn:
        for key, value_json in expected.items():
            conn.execute(
                "insert or ignore into ledger_metadata(key, value_json) values (?, ?)",
                (key, value_json),
            )
        existing = {str(row["key"]): str(row["value_json"]) for row in conn.execute("select key, value_json from ledger_metadata")}
        for key, value_json in expected.items():
            if existing.get(key) != value_json:
                raise ValueError(f"accepted-token ledger metadata mismatch for {key}")
        unexpected = sorted(set(existing).difference(expected))
        if unexpected:
            raise ValueError(f"accepted-token ledger has unexpected metadata keys: {unexpected[:5]}")
        conn.execute(
            """
            insert or ignore into ledger_state(
                id, next_commit_ordinal, accepted_token_count_lower_bound,
                accepted_token_target_reached, stop_ordinal, total_expected_rows,
                accepted_token_target_lower_bound, updated_at
            ) values (1, 0, 0, 0, null, ?, ?, ?)
            """,
            (total_expected_rows, target, now),
        )
        state = _w1_token_ledger_state(conn)
        if int(state["total_expected_rows"]) != total_expected_rows:
            raise ValueError("accepted-token ledger survivor_count mismatch")
        if int(state["accepted_token_target_lower_bound"]) != target:
            raise ValueError("accepted-token ledger target mismatch")
    _chmod_private_sqlite_files(str(ledger_path))
    return conn


def _advance_w1_token_ledger_locked(conn: sqlite3.Connection) -> dict[str, Any]:
    state = _w1_token_ledger_state(conn)
    target = int(state["accepted_token_target_lower_bound"])
    total_expected_rows = int(state["total_expected_rows"])
    next_ordinal = int(state["next_commit_ordinal"])
    accepted = int(state["accepted_token_count_lower_bound"])
    target_reached = bool(state["accepted_token_target_reached"])
    stop_ordinal = state["stop_ordinal"]
    if target_reached:
        if stop_ordinal is not None:
            conn.execute(
                """
                update ledger_rows
                set commit_decision = 'after_stop', updated_at = ?
                where global_ordinal > ? and commit_decision = 'pending'
                """,
                (time.time(), int(stop_ordinal)),
            )
        return _w1_token_ledger_state(conn)
    while next_ordinal < total_expected_rows:
        rows = list(
            conn.execute(
                """
                select global_ordinal, w1_route, accepted_token_count, commit_decision
                from ledger_rows
                where global_ordinal >= ?
                order by global_ordinal
                limit 4096
                """,
                (next_ordinal,),
            )
        )
        if not rows:
            break
        update_now = time.time()
        range_start = next_ordinal
        range_end: int | None = None
        consumed_rows = 0
        for row in rows:
            row_ordinal = int(row["global_ordinal"])
            if row_ordinal != next_ordinal:
                break
            if str(row["commit_decision"]) == "after_stop":
                raise ValueError("accepted-token ledger encountered after-stop row before target")
            if str(row["commit_decision"]) != "persist":
                if str(row["w1_route"]) == "w1_pass":
                    accepted += int(row["accepted_token_count"])
            range_end = next_ordinal
            consumed_rows += 1
            if accepted >= target:
                target_reached = True
                stop_ordinal = next_ordinal
                next_ordinal += 1
                break
            next_ordinal += 1
        if range_end is not None:
            conn.execute(
                """
                update ledger_rows
                set commit_decision = 'persist', updated_at = ?
                where global_ordinal between ? and ? and commit_decision != 'persist'
                """,
                (update_now, int(range_start), int(range_end)),
            )
        if target_reached or consumed_rows < len(rows):
            break
    now = time.time()
    conn.execute(
        """
        update ledger_state
        set next_commit_ordinal = ?, accepted_token_count_lower_bound = ?,
            accepted_token_target_reached = ?, stop_ordinal = ?, updated_at = ?
        where id = 1
        """,
        (
            next_ordinal,
            accepted,
            1 if target_reached else 0,
            None if stop_ordinal is None else int(stop_ordinal),
            now,
        ),
    )
    if target_reached and stop_ordinal is not None:
        conn.execute(
            """
            update ledger_rows
            set commit_decision = 'after_stop', updated_at = ?
            where global_ordinal > ? and commit_decision = 'pending'
            """,
            (now, int(stop_ordinal)),
        )
    return _w1_token_ledger_state(conn)


def _submit_w1_token_ledger_records(
    conn: sqlite3.Connection,
    *,
    score_records: Sequence[tuple[Mapping[str, Any], float | None, str, str | None]],
    config_payload: Mapping[str, Any],
    w1_runtime_envelope_hash: str,
    global_ordinal_by_candidate_id: Mapping[str, int],
    global_ordinal_base: int,
    shard_index: int,
    rank_assignment_hash: str | None,
) -> dict[str, Any]:
    if not score_records:
        return _w1_token_ledger_state(conn)
    now = time.time()
    ordinal_base = _nonnegative_int(global_ordinal_base, "accepted-token ledger global ordinal base")
    with conn:
        state = _w1_token_ledger_state(conn)
        target_reached = bool(state["accepted_token_target_reached"])
        stop_ordinal = state["stop_ordinal"]
        prepared_rows: list[tuple[int, str, str, int, str, str, str, int]] = []
        for survivor_row, score, route, error_class in score_records:
            payload = _build_w1_score_payload(
                survivor_row,
                config_payload=config_payload,
                score=score,
                route=route,
                error_class=error_class,
                w1_runtime_envelope_hash=w1_runtime_envelope_hash,
            )
            candidate_id = str(payload["candidate_id_hash"])
            if candidate_id not in global_ordinal_by_candidate_id:
                raise ValueError("accepted-token ledger missing candidate ordinal")
            global_ordinal = int(global_ordinal_by_candidate_id[candidate_id])
            accepted_tokens = int(survivor_row.get("core_proxy_token_count", 0) or 0) if route == "w1_pass" else 0
            score_json = json.dumps(payload, sort_keys=True)
            score_json_hash = stable_json_hash(payload)
            decision = "pending"
            if target_reached and stop_ordinal is not None and global_ordinal > int(stop_ordinal):
                decision = "after_stop"
            shard_row_index = global_ordinal - ordinal_base
            if shard_row_index < 0:
                raise ValueError("accepted-token ledger shard row index is negative")
            prepared_rows.append(
                (
                    global_ordinal,
                    candidate_id,
                    str(route),
                    int(accepted_tokens),
                    score_json,
                    score_json_hash,
                    decision,
                    int(shard_row_index),
                )
            )
        existing_by_ordinal: dict[int, tuple[str, str]] = {}
        ordinals = [row[0] for row in prepared_rows]
        for chunk in _iter_sized_chunks(list(dict.fromkeys(ordinals)), 900):
            placeholders = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"""
                select global_ordinal, candidate_id_hash, score_json
                from ledger_rows where global_ordinal in ({placeholders})
                """,
                tuple(chunk),
            ):
                existing_by_ordinal[int(row["global_ordinal"])] = (
                    str(row["candidate_id_hash"]),
                    str(row["score_json"]),
                )
        existing_by_candidate_id: dict[str, tuple[int, str]] = {}
        candidate_ids = [row[1] for row in prepared_rows]
        for chunk in _iter_sized_chunks(list(dict.fromkeys(candidate_ids)), 900):
            placeholders = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"""
                select candidate_id_hash, global_ordinal, score_json
                from ledger_rows where candidate_id_hash in ({placeholders})
                """,
                tuple(chunk),
            ):
                existing_by_candidate_id[str(row["candidate_id_hash"])] = (
                    int(row["global_ordinal"]),
                    str(row["score_json"]),
                )
        rows_to_insert: list[tuple[int, int, int, str, str, int, str, str, str, str | None, float, float]] = []
        seen_ordinals: dict[int, tuple[str, str]] = {}
        seen_candidate_ids: dict[str, tuple[int, str]] = {}
        for (
            global_ordinal,
            candidate_id,
            route,
            accepted_tokens,
            score_json,
            score_json_hash,
            decision,
            shard_row_index,
        ) in prepared_rows:
            batch_ordinal = seen_ordinals.get(global_ordinal)
            if batch_ordinal is not None:
                if batch_ordinal != (candidate_id, score_json):
                    raise ValueError("conflicting accepted-token ledger row for global ordinal")
                continue
            batch_candidate = seen_candidate_ids.get(candidate_id)
            if batch_candidate is not None:
                if batch_candidate != (global_ordinal, score_json):
                    raise ValueError("conflicting accepted-token ledger row for candidate")
                continue
            existing = existing_by_ordinal.get(global_ordinal)
            if existing is not None:
                if existing != (candidate_id, score_json):
                    raise ValueError("conflicting accepted-token ledger row for global ordinal")
                seen_ordinals[global_ordinal] = (candidate_id, score_json)
                seen_candidate_ids[candidate_id] = (global_ordinal, score_json)
                continue
            existing_by_id = existing_by_candidate_id.get(candidate_id)
            if existing_by_id is not None:
                if existing_by_id != (global_ordinal, score_json):
                    raise ValueError("conflicting accepted-token ledger row for candidate")
                seen_ordinals[global_ordinal] = (candidate_id, score_json)
                seen_candidate_ids[candidate_id] = (global_ordinal, score_json)
                continue
            rows_to_insert.append(
                (
                    global_ordinal,
                    int(shard_index),
                    int(shard_row_index),
                    candidate_id,
                    route,
                    accepted_tokens,
                    score_json,
                    score_json_hash,
                    decision,
                    rank_assignment_hash,
                    now,
                    now,
                )
            )
            seen_ordinals[global_ordinal] = (candidate_id, score_json)
            seen_candidate_ids[candidate_id] = (global_ordinal, score_json)
        if rows_to_insert:
            conn.executemany(
                """
                insert into ledger_rows(
                    global_ordinal, shard_index, shard_row_index, candidate_id_hash,
                    w1_route, accepted_token_count, score_json, score_json_hash,
                    commit_decision, producer_rank_assignment_hash, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
        return _advance_w1_token_ledger_locked(conn)


def _persist_w1_token_ledger_grants_for_shard(
    *,
    ledger_conn: sqlite3.Connection,
    shard_conn: sqlite3.Connection,
    shard_index: int,
    survivor_by_id: Mapping[str, Mapping[str, Any]],
    config_payload: Mapping[str, Any],
) -> int:
    inserted, _max_shard_row_index = _persist_w1_token_ledger_grants_for_shard_delta(
        ledger_conn=ledger_conn,
        shard_conn=shard_conn,
        shard_index=shard_index,
        survivor_by_id=survivor_by_id,
        config_payload=config_payload,
        after_shard_row_index=None,
    )
    return inserted


def _persist_w1_token_ledger_grants_for_shard_delta(
    *,
    ledger_conn: sqlite3.Connection,
    shard_conn: sqlite3.Connection,
    shard_index: int,
    survivor_by_id: Mapping[str, Mapping[str, Any]],
    config_payload: Mapping[str, Any],
    after_shard_row_index: int | None,
) -> tuple[int, int | None]:
    where = "shard_index = ? and commit_decision = 'persist'"
    params: list[Any] = [int(shard_index)]
    if after_shard_row_index is not None:
        where += " and shard_row_index > ?"
        params.append(_nonnegative_int(after_shard_row_index, "after_shard_row_index"))
    rows = [
        (int(row["shard_row_index"]), json.loads(str(row["score_json"])))
        for row in ledger_conn.execute(
            f"""
            select shard_row_index, score_json from ledger_rows
            where {where}
            order by shard_row_index
            """,
            tuple(params),
        )
    ]
    if not rows:
        return 0, after_shard_row_index
    payloads = [payload for _row_index, payload in rows]
    max_shard_row_index = max(row_index for row_index, _payload in rows)
    _validate_w1_score_rows(payloads, config_payload, survivor_by_id=survivor_by_id, allow_empty=True)
    return _insert_w1_score_payloads(shard_conn, payloads), max_shard_row_index


def _w1_token_ledger_row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts = {
        str(row["commit_decision"]): int(row["row_count"])
        for row in conn.execute(
            """
            select commit_decision, count(*) as row_count
            from ledger_rows
            group by commit_decision
            """
        )
    }
    return {
        "persist": int(counts.get("persist", 0)),
        "pending": int(counts.get("pending", 0)),
        "after_stop": int(counts.get("after_stop", 0)),
    }


def _w1_token_ledger_candidate_ids_for_shard(conn: sqlite3.Connection, shard_index: int) -> set[str]:
    return {
        str(row["candidate_id_hash"])
        for row in conn.execute(
            "select candidate_id_hash from ledger_rows where shard_index = ?",
            (int(shard_index),),
        )
    }


def _w1_token_ledger_shard_summary(conn: sqlite3.Connection, shard_index: int) -> dict[str, Any]:
    row = conn.execute(
        """
        select count(*) as row_count, coalesce(sum(accepted_token_count), 0) as accepted_token_count
        from ledger_rows
        where shard_index = ?
        """,
        (int(shard_index),),
    ).fetchone()
    route_counts = {
        str(route_row["w1_route"]): int(route_row["row_count"])
        for route_row in conn.execute(
            """
            select w1_route, count(*) as row_count
            from ledger_rows
            where shard_index = ?
            group by w1_route
            """,
            (int(shard_index),),
        )
    }
    commit_decision_counts = {
        str(decision_row["commit_decision"]): int(decision_row["row_count"])
        for decision_row in conn.execute(
            """
            select commit_decision, count(*) as row_count
            from ledger_rows
            where shard_index = ?
            group by commit_decision
            """,
            (int(shard_index),),
        )
    }
    return {
        "row_count": 0 if row is None else int(row["row_count"]),
        "accepted_token_count_lower_bound": 0 if row is None else int(row["accepted_token_count"]),
        "route_counts": _sorted_counter(Counter(route_counts)),
        "commit_decision_counts": _sorted_counter(Counter(commit_decision_counts)),
    }


def _w1_compact_forbidden_fresh_files(run_path: Path) -> list[str]:
    forbidden = []
    for name in (
        "STOP_REQUESTED",
        "w1_score_receipt.json",
        "w1_scores.jsonl.ready",
        "w1_scores.jsonl.part",
    ):
        if (run_path / name).exists():
            forbidden.append(name)
    return forbidden


def _copy_selected_w1_compact_rows(
    *,
    source: sqlite3.Connection,
    target: sqlite3.Connection,
    candidate_ids: Sequence[str],
    private_evidence_ids: Sequence[str],
) -> None:
    candidate_id_set = list(dict.fromkeys(_require_sha256(item, "compact candidate_id_hash") for item in candidate_ids))
    evidence_id_set = list(dict.fromkeys(_require_sha256(item, "compact private_evidence_row_id") for item in private_evidence_ids))
    if len(candidate_id_set) != len(candidate_ids):
        raise ValueError("compact start state survivor queue contains duplicate candidate ids")
    if len(evidence_id_set) != len(private_evidence_ids):
        raise ValueError("compact start state survivor queue contains duplicate private evidence ids")
    with target:
        for row in source.execute("select key, value_json from stage_metadata"):
            target.execute(
                "insert or replace into stage_metadata(key, value_json) values (?, ?)",
                (str(row["key"]), str(row["value_json"])),
            )
        for chunk in _iter_sized_chunks(candidate_id_set, 900):
            placeholders = ",".join("?" for _ in chunk)
            rows = list(
                source.execute(
                    f"""
                    select candidate_id_hash, source_document_id_hash, segmentation_receipt_hash,
                           candidate_json, deterministic_result_json, prefilter_route, s1_score, created_at
                    from prefilter_candidates
                    where candidate_id_hash in ({placeholders})
                    order by candidate_id_hash
                    """,
                    tuple(chunk),
                )
            )
            target.executemany(
                """
                insert into prefilter_candidates(
                    candidate_id_hash, source_document_id_hash, segmentation_receipt_hash,
                    candidate_json, deterministic_result_json, prefilter_route, s1_score, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(row["candidate_id_hash"]),
                        str(row["source_document_id_hash"]),
                        str(row["segmentation_receipt_hash"]),
                        str(row["candidate_json"]),
                        str(row["deterministic_result_json"]),
                        str(row["prefilter_route"]),
                        row["s1_score"],
                        float(row["created_at"]),
                    )
                    for row in rows
                ],
            )
        for chunk in _iter_sized_chunks(evidence_id_set, 900):
            placeholders = ",".join("?" for _ in chunk)
            rows = list(
                source.execute(
                    f"""
                    select private_evidence_row_id, candidate_id_hash, evidence_text_sha256, private_evidence_text
                    from private_w1_evidence
                    where private_evidence_row_id in ({placeholders})
                    order by private_evidence_row_id
                    """,
                    tuple(chunk),
                )
            )
            target.executemany(
                """
                insert into private_w1_evidence(
                    private_evidence_row_id, candidate_id_hash, evidence_text_sha256, private_evidence_text
                ) values (?, ?, ?, ?)
                """,
                [
                    (
                        str(row["private_evidence_row_id"]),
                        str(row["candidate_id_hash"]),
                        str(row["evidence_text_sha256"]),
                        str(row["private_evidence_text"]),
                    )
                    for row in rows
                ],
            )
    copied_candidates = int(target.execute("select count(*) from prefilter_candidates").fetchone()[0])
    copied_evidence = int(target.execute("select count(*) from private_w1_evidence").fetchone()[0])
    if copied_candidates != len(candidate_id_set):
        raise ValueError("compact start state is missing survivor candidate rows")
    if copied_evidence != len(evidence_id_set):
        raise ValueError("compact start state is missing private evidence rows")
    non_survivor = target.execute(
        "select count(*) from prefilter_candidates where prefilter_route != 'prefilter_survivor'"
    ).fetchone()[0]
    if int(non_survivor) != 0:
        raise ValueError("compact start state may contain only survivor candidate rows")


def _compact_sqlite_sha256(conn: sqlite3.Connection, run_path: Path) -> str:
    conn.execute("pragma wal_checkpoint(TRUNCATE)")
    db_path = _stage_state_path(run_path)
    return _sha256_file(db_path)


def _compact_destination_supervisor_receipt_fields(
    *,
    dest_run_dir: Path,
    destination_supervisor_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if destination_supervisor_manifest is None:
        return {}
    manifest = dict(destination_supervisor_manifest)
    if manifest.get("manifest_hash") != stable_json_hash(_prefilter_supervisor_manifest_hash_payload(manifest)):
        raise ValueError("compact start destination supervisor manifest hash mismatch")
    supervisor_run_dir = dest_run_dir.parent
    if manifest.get("supervisor_run_dir_sha256") != _prefilter_supervisor_run_dir_sha256(supervisor_run_dir):
        raise ValueError("compact start destination supervisor run dir hash mismatch")
    matching_shards = [
        shard
        for shard in manifest.get("shards", [])
        if str(shard.get("shard_run_dir_name")) == dest_run_dir.name
    ]
    if len(matching_shards) != 1:
        raise ValueError("compact start destination supervisor manifest does not uniquely identify shard dir")
    shard = matching_shards[0]
    return {
        "supervisor_manifest_hash": manifest["manifest_hash"],
        "supervisor_command_hash": shard["command_hash"],
        "supervisor_run_dir_sha256": manifest["supervisor_run_dir_sha256"],
    }


def _is_w1_compact_prefilter_receipt(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("compact_start_state_artifact_kind") == W1_COMPACT_START_STATE_KIND


def _validate_w1_compact_ready_db(
    conn: sqlite3.Connection,
    run_path: Path,
    config_payload: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    queue_rows: Sequence[Mapping[str, Any]],
    require_empty_w1_scores: bool = True,
) -> None:
    if receipt.get("compact_start_state_status") != "compact_fresh_import":
        raise ValueError("compact start state status mismatch")
    compact_candidate_count = int(
        conn.execute("select count(*) from prefilter_candidates").fetchone()[0] or 0
    )
    compact_survivor_count = int(
        conn.execute(
            "select count(*) from prefilter_candidates where prefilter_route = 'prefilter_survivor'"
        ).fetchone()[0] or 0
    )
    compact_evidence_count = int(
        conn.execute("select count(*) from private_w1_evidence").fetchone()[0] or 0
    )
    compact_score_count = int(conn.execute("select count(*) from w1_scores").fetchone()[0] or 0)
    survivor_count = len(queue_rows)
    if compact_candidate_count != survivor_count or compact_survivor_count != survivor_count:
        raise ValueError("compact start state survivor candidate count mismatch")
    if compact_evidence_count != survivor_count:
        raise ValueError("compact start state private evidence count mismatch")
    if require_empty_w1_scores and compact_score_count != 0:
        raise ValueError("compact start state must have empty w1_scores")
    if int(receipt.get("compact_candidate_count", -1)) != compact_candidate_count:
        raise ValueError("compact start state compact candidate count mismatch")
    if int(receipt.get("compact_private_evidence_count", -1)) != compact_evidence_count:
        raise ValueError("compact start state compact private evidence count mismatch")
    if int(receipt.get("survivor_count", -1)) != survivor_count:
        raise ValueError("compact start state survivor count mismatch")
    db_queue_rows = _prefilter_survivor_queue_rows(conn, config_payload)
    if not _survivor_queue_rows_match_live_db(queue_rows, db_queue_rows):
        raise ValueError("compact start state survivor queue export does not match SQLite state")
    queue_by_evidence_id = {
        _require_sha256(row.get("private_evidence_row_id"), "private_evidence_row_id"): row
        for row in queue_rows
    }
    observed_evidence_ids: set[str] = set()
    for chunk in _iter_sized_chunks(list(queue_by_evidence_id), 500):
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"""
            select private_evidence_row_id, candidate_id_hash, evidence_text_sha256, private_evidence_text
            from private_w1_evidence
            where private_evidence_row_id in ({placeholders})
            """,
            tuple(chunk),
        ):
            evidence_id = str(row["private_evidence_row_id"])
            observed_evidence_ids.add(evidence_id)
            expected = queue_by_evidence_id[evidence_id]
            if str(row["candidate_id_hash"]) != str(expected["candidate_id_hash"]):
                raise ValueError("compact start state private evidence candidate mismatch")
            expected_text_hash = _require_sha256(expected.get("evidence_text_sha256"), "evidence_text_sha256")
            if str(row["evidence_text_sha256"]) != expected_text_hash:
                raise ValueError("compact start state private evidence stored hash mismatch")
            if _sha256_text(str(row["private_evidence_text"])) != expected_text_hash:
                raise ValueError("compact start state private evidence text hash mismatch")
    if observed_evidence_ids != set(queue_by_evidence_id):
        raise ValueError("compact start state private evidence id set mismatch")
    _require_sha256(receipt.get("source_prefilter_stage_receipt_hash"), "source_prefilter_stage_receipt_hash")
    _require_sha256(receipt.get("compact_sqlite_sha256"), "compact_sqlite_sha256")


def build_w1_compact_start_state(
    *,
    config_payload: Mapping[str, Any],
    source_run_dir: str | Path,
    dest_run_dir: str | Path,
    expected_prefilter_status: str = "prefilter_ready_for_w1",
    destination_supervisor_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    source_path = Path(source_run_dir)
    dest_path = Path(dest_run_dir)
    validate_private_filtering_run_dir(source_path)
    if dest_path.exists() and any(dest_path.iterdir()):
        raise ValueError("compact start state destination must be fresh and empty")
    if source_path.resolve(strict=False) == dest_path.resolve(strict=False):
        raise ValueError("compact start state destination must differ from source")
    if (source_path / "STOP_REQUESTED").exists():
        raise ValueError("compact start state source contains STOP_REQUESTED")
    source_queue_rows = _load_ready_jsonl(source_path / "w1_survivor_queue.jsonl", "survivor queue")
    observed_prefilter_config_hash = (
        source_queue_rows[0].get("config_hash") if source_queue_rows else config_payload["config_hash"]
    )
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        observed_prefilter_config_hash,
        label="compact start state survivor queue",
    )
    _validate_survivor_queue_rows(source_queue_rows, prefilter_config_payload, allow_empty=True)
    source_conn = _open_existing_stage_state(source_path, prefilter_config_payload)
    try:
        _validate_prefilter_ready_receipt(
            source_conn,
            source_path,
            prefilter_config_payload,
            queue_rows=source_queue_rows,
            expected_status=expected_prefilter_status,
        )
        if not _survivor_queue_rows_match_live_db(
            source_queue_rows,
            _prefilter_survivor_queue_rows(source_conn, prefilter_config_payload),
        ):
            raise ValueError("compact start state source survivor queue export does not match SQLite state")
        source_w1_score_count = int(source_conn.execute("select count(*) from w1_scores").fetchone()[0] or 0)
        if source_w1_score_count != 0:
            raise ValueError("compact start state source must not contain W1 scores")
        source_receipt = _load_json(source_path / "prefilter_cpu_receipt.json")
        if not isinstance(source_receipt, Mapping):
            raise ValueError("compact start state source receipt must be an object")
        destination_supervisor_fields = _compact_destination_supervisor_receipt_fields(
            dest_run_dir=dest_path,
            destination_supervisor_manifest=destination_supervisor_manifest,
        )
        _ensure_private_dir(dest_path)
        dest_conn = _open_stage_state(dest_path, prefilter_config_payload)
        try:
            _copy_selected_w1_compact_rows(
                source=source_conn,
                target=dest_conn,
                candidate_ids=[str(row["candidate_id_hash"]) for row in source_queue_rows],
                private_evidence_ids=[str(row["private_evidence_row_id"]) for row in source_queue_rows],
            )
            queue_proof = _write_survivor_queue_ready_streaming(
                dest_path / "w1_survivor_queue.jsonl",
                source_queue_rows,
            )
            compact_sqlite_hash = _compact_sqlite_sha256(dest_conn, dest_path)
        finally:
            dest_conn.close()
    finally:
        source_conn.close()
    payload = dict(source_receipt)
    payload.pop("stage_receipt_hash", None)
    payload.update(
        {
            "status": "compact_fresh_import",
            "stop_reason": "compact_start_state_built",
            "next_required_phase": "w1-score",
            "compact_start_state_artifact_kind": W1_COMPACT_START_STATE_KIND,
            "compact_start_state_status": "compact_fresh_import",
            "source_prefilter_stage_receipt_hash": source_receipt["stage_receipt_hash"],
            "source_prefilter_status": source_receipt.get("status"),
            "source_supervisor_manifest_hash": source_receipt.get("supervisor_manifest_hash"),
            "source_supervisor_command_hash": source_receipt.get("supervisor_command_hash"),
            "source_supervisor_run_dir_sha256": source_receipt.get("supervisor_run_dir_sha256"),
            "source_run_dir_sha256": stable_json_hash(str(source_path.resolve(strict=False))),
            "compact_candidate_count": len(source_queue_rows),
            "compact_private_evidence_count": len(source_queue_rows),
            "compact_w1_scores_count": 0,
            "compact_db_contains_only_survivor_prefilter_candidates": True,
            "compact_non_survivor_prefilter_candidates_omitted": int(source_receipt.get("candidate_count", 0))
            - len(source_queue_rows),
            "compact_prefilter_source_rows_omitted": True,
            "compact_sqlite_sha256": compact_sqlite_hash,
            "survivor_queue_hash": queue_proof.survivor_queue_hash,
            "survivor_queue_row_count": queue_proof.survivor_queue_row_count,
            "survivor_queue_id_index_hash": queue_proof.survivor_queue_id_index_hash,
            "survivor_queue_id_index_row_count": queue_proof.survivor_queue_id_index_row_count,
            "survivor_candidate_id_set_hash": queue_proof.survivor_candidate_id_set_hash,
            "survivor_private_evidence_id_set_hash": queue_proof.survivor_private_evidence_id_set_hash,
            "survivor_queue_proof_kind": "jsonl_sha256_and_id_set_hashes_v1",
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
    )
    payload.update(destination_supervisor_fields)
    payload["stage_receipt_hash"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "stage_receipt_hash"}
    )
    validate_public_no_raw_payload(payload)
    _write_json_atomic(dest_path / "prefilter_cpu_receipt.json", payload)
    return validate_w1_compact_start_state(config_payload=config_payload, run_dir=dest_path)


def validate_w1_compact_start_state(
    *,
    config_payload: Mapping[str, Any],
    run_dir: str | Path,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    run_path = Path(run_dir)
    validate_private_filtering_run_dir(run_path)
    if _w1_compact_forbidden_fresh_files(run_path):
        raise ValueError("compact start state contains stale W1 output or STOP_REQUESTED")
    queue_rows = _load_ready_jsonl(run_path / "w1_survivor_queue.jsonl", "survivor queue")
    observed_prefilter_config_hash = queue_rows[0].get("config_hash") if queue_rows else config_payload["config_hash"]
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        observed_prefilter_config_hash,
        label="compact start state",
    )
    _validate_survivor_queue_rows(queue_rows, prefilter_config_payload, allow_empty=True)
    receipt = _load_json(run_path / "prefilter_cpu_receipt.json")
    if not isinstance(receipt, Mapping):
        raise ValueError("compact start state receipt must be an object")
    current_sqlite_hash = _sha256_file(_stage_state_path(run_path))
    if receipt.get("compact_sqlite_sha256") != current_sqlite_hash:
        raise ValueError("compact start state SQLite hash mismatch")
    conn = _open_existing_stage_state(run_path, prefilter_config_payload)
    try:
        _validate_prefilter_ready_receipt(
            conn,
            run_path,
            prefilter_config_payload,
            queue_rows=queue_rows,
            expected_status="compact_fresh_import",
        )
        payload = {
            "artifact_kind": W1_COMPACT_START_STATE_KIND,
            "schema_version": 1,
            "status": "compact_fresh_import",
            "config_hash": prefilter_config_payload["config_hash"],
            "source_prefilter_stage_receipt_hash": receipt["source_prefilter_stage_receipt_hash"],
            "compact_prefilter_stage_receipt_hash": receipt["stage_receipt_hash"],
            "full_candidate_count": int(receipt["candidate_count"]),
            "survivor_count": int(receipt["survivor_count"]),
            "compact_candidate_count": int(receipt["compact_candidate_count"]),
            "compact_private_evidence_count": int(receipt["compact_private_evidence_count"]),
            "route_counts": dict(receipt.get("route_counts") or {}),
            "compact_sqlite_sha256": receipt["compact_sqlite_sha256"],
            "survivor_queue_hash": receipt["survivor_queue_hash"],
            "accepted_manifest_created": False,
            "accepted_manifest": False,
            "tokenizer_shards_created": False,
            "packed_shards_created": False,
            "training_shards_created": False,
            "base_pretraining_started": False,
            "pretraining_started": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        payload["compact_start_state_hash"] = stable_json_hash(
            {key: value for key, value in payload.items() if key != "compact_start_state_hash"}
        )
        validate_public_no_raw_payload(payload)
        return payload
    finally:
        conn.close()


W1_PRIVATE_SCORE_STATE_ARCHIVE_KIND = "twc_w1_private_score_state_archive_v1"


def _iter_w1_private_score_state_files(run_path: Path) -> Iterator[Path]:
    allowed_names = {
        "filtering_stage_state.sqlite",
        "filtering_stage_state.sqlite-wal",
        "filtering_stage_state.sqlite-shm",
        "prefilter_cpu_receipt.json",
        "w1_survivor_queue.jsonl.ready",
        "w1_survivor_queue.ids.tsv.ready",
        "w1_scores.jsonl.ready",
        "w1_scores.jsonl.part",
        "w1_score_receipt.json",
        "assembled_filtering_report.json",
        "sharded_w1_handoff.json",
        "prefilter_supervisor_manifest.json",
        "prefilter_supervisor_summary.json",
        "w1_sharded_summary.json",
        "w1_assembled_report.json",
        "launch_plan.json",
        "launcher_receipt.json",
        "commands.jsonl",
    }
    for path in sorted(run_path.rglob("*")):
        if not path.is_file():
            continue
        if path.name in allowed_names or re.fullmatch(r"rank_\d+_w1_summary\.json", path.name):
            yield path


def _iter_w1_extra_state_files(paths: Sequence[str | Path] | None) -> Iterator[tuple[Path, str]]:
    seen_archive_names: set[str] = set()
    for item in list(paths or []):
        base = Path(item)
        if not base.exists() or not base.is_file():
            raise ValueError("W1 private score archive required extra state file missing")
        for path in (base, Path(str(base) + "-wal"), Path(str(base) + "-shm")):
            if not path.exists():
                continue
            if not path.is_file():
                raise ValueError("W1 private score archive extra state path is not a file")
            archive_name = f"extra_state/{path.name}"
            if archive_name in seen_archive_names:
                continue
            seen_archive_names.add(archive_name)
            yield path, archive_name


def archive_w1_private_score_state(
    *,
    config_payload: Mapping[str, Any],
    run_dir: str | Path,
    archive_dir: str | Path,
    extra_state_files: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    run_path = Path(run_dir)
    archive_path = Path(archive_dir)
    validate_private_filtering_run_dir(run_path)
    validate_private_filtering_run_dir(archive_path)
    _ensure_private_dir(archive_path)
    files = list(_iter_w1_private_score_state_files(run_path))
    extra_files = list(_iter_w1_extra_state_files(extra_state_files))
    if not files:
        raise ValueError("W1 private score archive found no allowlisted files")
    sqlite_files = [path for path in files if path.name == "filtering_stage_state.sqlite"]
    if not sqlite_files:
        raise ValueError("W1 private score archive requires SQLite state")
    score_receipts = [path for path in files if path.name == "w1_score_receipt.json"]
    if not score_receipts:
        raise ValueError("W1 private score archive requires W1 score receipt")
    w1_scored_count = 0
    w1_statuses: set[str] = set()
    for receipt_path in score_receipts:
        receipt = _load_json(receipt_path)
        if not isinstance(receipt, Mapping):
            raise ValueError("W1 private score archive receipt must be an object")
        if receipt.get("artifact_kind") != W1_SCORE_STAGE_KIND:
            raise ValueError("W1 private score archive receipt kind mismatch")
        if receipt.get("config_hash") != config_payload["config_hash"]:
            raise ValueError("W1 private score archive config hash mismatch")
        w1_scored_count += int(receipt.get("scored_count", 0))
        w1_statuses.add(str(receipt.get("status")))
    zip_path = archive_path / "w1_private_score_state.zip"
    manifest_path = archive_path / "w1_private_score_state_manifest.json"
    file_payloads: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            relative = path.relative_to(run_path).as_posix()
            archive.write(path, arcname=relative)
            file_payloads.append(
                {
                    "relative_path": relative,
                    "source_kind": "run_dir",
                    "sha256": _sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                }
            )
        for path, relative in extra_files:
            archive.write(path, arcname=relative)
            file_payloads.append(
                {
                    "relative_path": relative,
                    "source_kind": "extra_state",
                    "sha256": _sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                }
            )
    _chmod_private_file(zip_path)
    payload = {
        "artifact_kind": W1_PRIVATE_SCORE_STATE_ARCHIVE_KIND,
        "schema_version": 1,
        "status": "w1_private_score_state_archived",
        "config_hash": config_payload["config_hash"],
        "run_dir_sha256": stable_json_hash(str(run_path.resolve(strict=False))),
        "archive_file": zip_path.name,
        "archive_sha256": _sha256_file(zip_path),
        "archive_size_bytes": int(zip_path.stat().st_size),
        "file_count": len(file_payloads),
        "files": file_payloads,
        "extra_state_file_count": len(extra_files),
        "private_archive_contains_accepted_token_ledger_state": any(
            relative == "extra_state/w1_accepted_token_ledger.sqlite"
            for _path, relative in extra_files
        ),
        "w1_score_receipt_count": len(score_receipts),
        "w1_statuses": sorted(w1_statuses),
        "w1_scored_count": int(w1_scored_count),
        "private_archive_contains_sqlite_state": True,
        "private_archive_may_contain_private_evidence_text": True,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["archive_manifest_hash"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "archive_manifest_hash"}
    )
    validate_public_no_raw_payload(payload)
    _write_json_atomic(manifest_path, payload)
    return payload


def merge_prefilter_local_parquet_all_row_shards(
    *,
    local_schedule: Mapping[str, Any],
    source_metadata_manifest: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    shard_run_dirs: Sequence[str | Path],
    run_dir: str | Path,
    min_free_bytes: int = 1,
) -> dict[str, Any]:
    plan = build_local_parquet_all_rows_prefilter_plan(
        local_schedule=local_schedule,
        source_metadata_manifest=source_metadata_manifest,
    )
    return merge_prefilter_local_parquet_plan_shards(
        prefilter_plan=plan,
        config_payload=config_payload,
        shard_run_dirs=shard_run_dirs,
        run_dir=run_dir,
        min_free_bytes=min_free_bytes,
    )


def merge_prefilter_local_parquet_plan_shards(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    shard_run_dirs: Sequence[str | Path],
    run_dir: str | Path,
    min_free_bytes: int = 1,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    plan = dict(prefilter_plan)
    _validate_local_parquet_prefilter_plan(plan)
    shard_policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    if plan["source_metadata_hash"] != config_payload["source_metadata_hash"]:
        raise ValueError("prefilter shard merge source metadata hash mismatch")
    shard_paths = [Path(path) for path in shard_run_dirs]
    if not shard_paths:
        raise ValueError("complete shard set is required for prefilter merge")
    run_path = Path(run_dir)
    if run_path.exists() and any(run_path.iterdir()):
        raise ValueError("merged prefilter run dir must be fresh and empty")
    if any(path.resolve(strict=False) == run_path.resolve(strict=False) for path in shard_paths):
        raise ValueError("merged prefilter run dir must be separate from shard run dirs")
    _ensure_private_dir(run_path)
    _check_disk_watermark([run_path], min_free_bytes=min_free_bytes)
    start = time.monotonic()
    shard_count = len(shard_paths)
    expected_indexes = set(range(shard_count))
    receipts_by_index: dict[int, dict[str, Any]] = {}
    shard_conns: list[sqlite3.Connection] = []
    try:
        for shard_path in shard_paths:
            validate_private_filtering_run_dir(shard_path)
            queue_row_count = _validate_survivor_queue_ready_file(
                shard_path / "w1_survivor_queue.jsonl",
                config_payload,
                allow_empty=True,
            )
            conn = _open_stage_state(shard_path, config_payload)
            shard_conns.append(conn)
            _validate_prefilter_ready_receipt(
                conn,
                shard_path,
                config_payload,
                queue_row_count=queue_row_count,
                expected_status="prefilter_shard_ready_for_merge",
            )
            receipt = _load_json(shard_path / "prefilter_cpu_receipt.json")
            if not isinstance(receipt, Mapping):
                raise ValueError("prefilter shard receipt must be an object")
            validate_public_no_raw_payload(receipt)
            declared_count = int(receipt.get("row_group_shard_count", 1))
            if declared_count != shard_count:
                raise ValueError("complete shard set is required for prefilter merge")
            shard_index = int(receipt.get("row_group_shard_index", -1))
            if shard_index not in expected_indexes:
                raise ValueError("prefilter shard index is outside the complete shard set")
            if shard_index in receipts_by_index:
                raise ValueError("prefilter shard merge received duplicate shard index")
            expected_shard_plan = _shard_local_parquet_prefilter_plan(
                plan,
                shard_count=shard_count,
                shard_index=shard_index,
                row_group_sharding_policy=shard_policy,
            )
            expected_fields = _local_parquet_prefilter_plan_receipt_fields(expected_shard_plan)
            for field, expected_value in expected_fields.items():
                if receipt.get(field) != expected_value:
                    raise ValueError(f"prefilter shard receipt {field} mismatch")
            if receipt.get("input_schedule_hash") != expected_shard_plan["schedule_hash"]:
                raise ValueError("prefilter shard input schedule hash mismatch")
            if int(receipt.get("terminal_rows", -1)) != int(expected_shard_plan["expected_physical_rows"]):
                raise ValueError("prefilter shard terminal row count mismatch")
            receipts_by_index[shard_index] = dict(receipt)
        if set(receipts_by_index) != expected_indexes:
            raise ValueError("complete shard set is required for prefilter merge")
        if sum(int(item["expected_physical_rows"]) for item in receipts_by_index.values()) != int(
            plan["expected_physical_rows"]
        ):
            raise ValueError("prefilter shard expected rows do not cover the all-row plan")
        if sum(int(item["row_group_shard_row_group_count"]) for item in receipts_by_index.values()) != int(
            plan["row_group_count"]
        ):
            raise ValueError("prefilter shard row groups do not cover the all-row plan")

        merged = _open_stage_state(run_path, config_payload)
        try:
            _record_prefilter_expected_counts(
                merged,
                input_schedule_hash=str(plan["schedule_hash"]),
                expected_physical_row_count=int(plan["expected_physical_rows"]),
                expected_physical_row_key_set_hash=None,
            )
            with merged:
                for conn in shard_conns:
                    _copy_prefilter_table_rows(
                        conn,
                        merged,
                        table="prefilter_source_rows",
                        columns=(
                            "physical_row_key_hash",
                            "source_schedule_hash",
                            "schedule_entry_hash",
                            "source_file_id_hash",
                            "row_group_index",
                            "row_group_local_offset",
                            "source_document_id_hash",
                            "row_status",
                            "error_class",
                            "segmentation_receipt_hash",
                            "candidate_count",
                            "survivor_count",
                            "row_receipt_hash",
                            "created_at",
                            "updated_at",
                        ),
                    )
                    _copy_prefilter_table_rows(
                        conn,
                        merged,
                        table="prefilter_candidates",
                        columns=(
                            "candidate_id_hash",
                            "source_document_id_hash",
                            "segmentation_receipt_hash",
                            "candidate_json",
                            "deterministic_result_json",
                            "prefilter_route",
                            "s1_score",
                            "created_at",
                        ),
                    )
                    _copy_prefilter_table_rows(
                        conn,
                        merged,
                        table="private_w1_evidence",
                        columns=(
                            "private_evidence_row_id",
                            "candidate_id_hash",
                            "evidence_text_sha256",
                            "private_evidence_text",
                        ),
                    )
            queue_proof = _write_survivor_queue_ready_streaming(
                run_path / "w1_survivor_queue.jsonl",
                _iter_prefilter_survivor_queue_rows(merged, config_payload),
            )
            queue_hash = queue_proof.survivor_queue_hash
            queue_count = queue_proof.survivor_queue_row_count
            if queue_count != sum(int(item.get("survivor_count", 0)) for item in receipts_by_index.values()):
                raise ValueError("merged prefilter survivor queue row count mismatch")
            extra_fields = _prefilter_merged_plan_receipt_fields(plan, receipts_by_index)
            report = _build_prefilter_receipt(
                merged,
                config_payload,
                run_path=run_path,
                status="prefilter_ready_for_w1",
                stop_reason="prefilter_shards_merged",
                elapsed_seconds=time.monotonic() - start,
                cache_dir=None,
                min_free_bytes=min_free_bytes,
                survivor_queue_ready=True,
                survivor_queue_part=False,
                survivor_queue_hash=queue_hash,
                survivor_queue_row_count=queue_proof.survivor_queue_row_count,
                survivor_queue_id_index_hash=queue_proof.survivor_queue_id_index_hash,
                survivor_queue_id_index_row_count=queue_proof.survivor_queue_id_index_row_count,
                survivor_candidate_id_set_hash=queue_proof.survivor_candidate_id_set_hash,
                survivor_private_evidence_id_set_hash=queue_proof.survivor_private_evidence_id_set_hash,
                extra_fields=extra_fields,
            )
            _write_json_atomic(run_path / "prefilter_cpu_receipt.json", report)
            return report
        finally:
            merged.close()
    finally:
        for conn in shard_conns:
            conn.close()


def run_w1_score_stage(
    *,
    config_payload: Mapping[str, Any],
    run_dir: str | Path,
    w1_scorer: ChunkFilteringScorer,
    batch_size: int = 16,
    expected_prefilter_status: str = "prefilter_ready_for_w1",
    extra_stop_paths: Sequence[str | Path] = (),
    w1_prepare_pipeline: str = "serial",
    max_estimated_windows_per_prepare_batch: int | None = None,
    w1_prepare_queue_depth: int = 1,
    w1_pin_memory: str = W1_PIN_MEMORY_AUTO,
    w1_tokenizers_parallelism: str = W1_TOKENIZERS_PARALLELISM_INHERIT,
    w1_persist_commit_row_threshold: int | None = None,
    enforce_accepted_token_target: bool = False,
    accepted_token_count_initial_lower_bound: int = 0,
    accepted_token_ledger_path: str | Path | None = None,
    accepted_token_ledger_metadata: Mapping[str, Any] | None = None,
    accepted_token_ledger_shard_index: int | None = None,
    accepted_token_ledger_global_ordinal_base: int | None = None,
    accepted_token_ledger_rank_assignment_hash: str | None = None,
    allow_queue_exhaustion_before_token_target: bool = False,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    if w1_scorer.model_id != CHUNK_WINDOW_MIL_MODEL_ID:
        raise ValueError("w1-score requires the W1 Window-MIL scorer")
    requested_prepare_pipeline = _normalize_w1_prepare_pipeline(w1_prepare_pipeline)
    prepare_pipeline_disabled_reason = None
    normalized_prepare_pipeline = requested_prepare_pipeline
    if requested_prepare_pipeline == W1_PREPARE_PIPELINE_BOUNDED_OVERLAP:
        prepare_pipeline_disabled_reason = _w1_prepare_pipeline_disabled_reason(w1_scorer)
        if prepare_pipeline_disabled_reason is not None:
            normalized_prepare_pipeline = W1_PREPARE_PIPELINE_SERIAL
    normalized_pin_memory = _normalize_w1_pin_memory(w1_pin_memory)
    normalized_tokenizers_parallelism = _normalize_w1_tokenizers_parallelism(w1_tokenizers_parallelism)
    normalized_persist_threshold = _positive_optional_int(
        w1_persist_commit_row_threshold,
        "w1_persist_commit_row_threshold",
    )
    accepted_token_initial_lower_bound = int(accepted_token_count_initial_lower_bound)
    if accepted_token_initial_lower_bound < 0:
        raise ValueError("accepted_token_count_initial_lower_bound must be non-negative")
    ledger_enabled = accepted_token_ledger_path is not None
    if ledger_enabled and not bool(enforce_accepted_token_target):
        raise ValueError("accepted-token ledger requires accepted token target enforcement")
    if ledger_enabled:
        if accepted_token_ledger_metadata is None:
            raise ValueError("accepted-token ledger metadata is required")
        if accepted_token_ledger_shard_index is None:
            raise ValueError("accepted-token ledger shard index is required")
        if accepted_token_ledger_global_ordinal_base is None:
            raise ValueError("accepted-token ledger global ordinal base is required")
    max_estimated_window_budget = _positive_optional_int(
        max_estimated_windows_per_prepare_batch,
        "max_estimated_windows_per_prepare_batch",
    )
    prepare_queue_depth = max(1, int(w1_prepare_queue_depth))
    if requested_prepare_pipeline == W1_PREPARE_PIPELINE_BOUNDED_OVERLAP and prepare_queue_depth != 1:
        raise ValueError("bounded W1 prepare overlap currently supports prepare_queue_depth=1")
    prepare_worker_count = 1 if normalized_prepare_pipeline == W1_PREPARE_PIPELINE_BOUNDED_OVERLAP else 0
    run_path = Path(run_dir)
    validate_private_filtering_run_dir(run_path)
    queue_rows = _load_ready_jsonl(run_path / "w1_survivor_queue.jsonl", "survivor queue")
    observed_prefilter_config_hash = queue_rows[0].get("config_hash") if queue_rows else config_payload["config_hash"]
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        observed_prefilter_config_hash,
        label="survivor queue",
    )
    _validate_survivor_queue_rows(queue_rows, prefilter_config_payload, allow_empty=True)
    conn = _open_w1_stage_state(
        run_path,
        config_payload=config_payload,
        prefilter_config_payload=prefilter_config_payload,
        exclusive_locking=normalized_prepare_pipeline != W1_PREPARE_PIPELINE_BOUNDED_OVERLAP,
    )
    tokenizers_had_original, tokenizers_original, effective_tokenizers_parallelism = _apply_w1_tokenizers_parallelism(
        normalized_tokenizers_parallelism
    )
    start = time.monotonic()
    ledger_conn: sqlite3.Connection | None = None
    try:
        _validate_prefilter_ready_receipt(
            conn,
            run_path,
            prefilter_config_payload,
            queue_rows=queue_rows,
            expected_status=expected_prefilter_status,
            require_compact_w1_scores_empty=False,
        )
        db_queue_rows = _prefilter_survivor_queue_rows(conn, prefilter_config_payload)
        if not _survivor_queue_rows_match_live_db(queue_rows, db_queue_rows):
            raise ValueError("survivor queue export does not match SQLite state")
        batch = max(1, int(batch_size))
        w1_runtime_envelope = _w1_runtime_execution_envelope(
            config_payload=config_payload,
            w1_scorer=w1_scorer,
            stage_batch_size=batch,
            w1_prepare_pipeline=normalized_prepare_pipeline,
            max_estimated_windows_per_prepare_batch=max_estimated_window_budget,
            prepare_queue_depth=prepare_queue_depth,
            prepare_worker_count=prepare_worker_count,
            w1_pin_memory=normalized_pin_memory,
            w1_tokenizers_parallelism_requested=normalized_tokenizers_parallelism,
            w1_tokenizers_parallelism_effective=effective_tokenizers_parallelism,
            w1_persist_commit_row_threshold=normalized_persist_threshold,
        )
        w1_runtime_envelope_hash = _w1_runtime_execution_envelope_hash(w1_runtime_envelope)
        _validate_existing_w1_runtime_envelope_state(
            conn=conn,
            run_path=run_path,
            requested_w1_runtime_envelope_hash=w1_runtime_envelope_hash,
        )
        survivor_by_id = {str(row["candidate_id_hash"]): row for row in queue_rows}
        accepted_token_target_lower_bound = int(config_payload["accepted_token_target_lower_bound"])
        ledger_state: dict[str, Any] | None = None
        ledger_row_counts = {"persist": 0, "pending": 0, "after_stop": 0}
        ledger_grant_rows_inserted_initial = 0
        global_ordinal_by_candidate_id: dict[str, int] = {}
        shard_index_for_ledger: int | None = None
        ledger_candidate_ids_for_shard: set[str] = set()
        ledger_shard_summary: dict[str, Any] = {
            "row_count": 0,
            "accepted_token_count_lower_bound": 0,
            "route_counts": {},
            "commit_decision_counts": {},
        }
        ledger_grants_materialized_through_shard_row_index: int | None = None
        if ledger_enabled:
            ordinal_base = _nonnegative_int(
                accepted_token_ledger_global_ordinal_base,
                "accepted_token_ledger_global_ordinal_base",
            )
            shard_index_for_ledger = _nonnegative_int(
                accepted_token_ledger_shard_index,
                "accepted_token_ledger_shard_index",
            )
            global_ordinal_by_candidate_id = {
                str(row["candidate_id_hash"]): ordinal_base + offset
                for offset, row in enumerate(queue_rows)
            }
            ledger_metadata = dict(accepted_token_ledger_metadata or {})
            ledger_metadata["w1_runtime_envelope_hash"] = w1_runtime_envelope_hash
            ledger_conn = _connect_w1_accepted_token_ledger(
                Path(accepted_token_ledger_path),
                metadata=ledger_metadata,
            )
            (
                ledger_grant_rows_inserted_initial,
                ledger_grants_materialized_through_shard_row_index,
            ) = _persist_w1_token_ledger_grants_for_shard_delta(
                ledger_conn=ledger_conn,
                shard_conn=conn,
                shard_index=shard_index_for_ledger,
                survivor_by_id=survivor_by_id,
                config_payload=config_payload,
                after_shard_row_index=ledger_grants_materialized_through_shard_row_index,
            )
            ledger_state = _w1_token_ledger_state(ledger_conn)
            ledger_row_counts = _w1_token_ledger_row_counts(ledger_conn)
            ledger_candidate_ids_for_shard = _w1_token_ledger_candidate_ids_for_shard(
                ledger_conn,
                shard_index_for_ledger,
            )
            ledger_shard_summary = _w1_token_ledger_shard_summary(ledger_conn, shard_index_for_ledger)
        scored_existing = {
            str(row["candidate_id_hash"]): json.loads(str(row["score_json"]))
            for row in conn.execute("select candidate_id_hash, score_json from w1_scores")
        }
        if ledger_enabled and ledger_state is not None:
            accepted_stage_token_count_lower_bound = _w1_score_rows_accepted_token_lower_bound(
                list(scored_existing.values()),
                survivor_by_id=survivor_by_id,
            )
            accepted_token_count_lower_bound = int(ledger_state["accepted_token_count_lower_bound"])
        else:
            accepted_stage_token_count_lower_bound = _w1_score_rows_accepted_token_lower_bound(
                list(scored_existing.values()),
                survivor_by_id=survivor_by_id,
            )
            accepted_token_count_lower_bound = (
                accepted_token_initial_lower_bound + accepted_stage_token_count_lower_bound
            )
        pending = [
            row
            for row in queue_rows
            if str(row["candidate_id_hash"]) not in scored_existing
            and (not ledger_enabled or str(row["candidate_id_hash"]) not in ledger_candidate_ids_for_shard)
        ]
        if not pending:
            score_rows = _w1_score_export_rows(conn)
            existing_complete_receipt = _load_existing_complete_w1_receipt(
                run_path,
                config_payload,
                queue_rows=queue_rows,
                score_rows=score_rows,
                w1_runtime_envelope_hash=w1_runtime_envelope_hash,
            )
            if existing_complete_receipt is not None:
                return existing_complete_receipt
        runtime_telemetry = {
            "stage_batch_size": batch,
            "w1_prepare_pipeline_requested": requested_prepare_pipeline,
            "w1_prepare_pipeline": normalized_prepare_pipeline,
            "w1_prepare_pipeline_disabled_reason": prepare_pipeline_disabled_reason,
            "w1_batching_strategy": _w1_batching_strategy(max_estimated_window_budget),
            "max_estimated_windows_per_prepare_batch": max_estimated_window_budget,
            "prepare_queue_depth": prepare_queue_depth,
            "prepare_worker_count": prepare_worker_count,
            "pin_memory_requested": normalized_pin_memory,
            "w1_tokenizers_parallelism_requested": normalized_tokenizers_parallelism,
            "w1_tokenizers_parallelism_effective": effective_tokenizers_parallelism,
            "score_persist_strategy": _w1_persist_strategy(normalized_persist_threshold),
            "score_persist_commit_row_threshold": normalized_persist_threshold,
            "score_persist_commit_count": 0,
            "score_persist_rows_per_commit": [],
            "score_persist_max_rows_per_commit": 0,
            "score_persist_pending_rows": 0,
            "score_persist_database_path_hash": (
                stable_json_hash(str(_sqlite_connection_path(conn).resolve(strict=False)))
                if _sqlite_connection_path(conn) is not None
                else None
            ),
            "accepted_token_target_enforced": bool(enforce_accepted_token_target),
            "accepted_token_target_lower_bound": accepted_token_target_lower_bound,
            "accepted_token_initial_lower_bound": accepted_token_initial_lower_bound,
            "accepted_estimated_token_count_lower_bound_initial": accepted_token_count_lower_bound,
            "accepted_estimated_token_count_lower_bound_initial_stage": accepted_stage_token_count_lower_bound,
            "accepted_token_ledger_enabled": bool(ledger_enabled),
            "accepted_token_ledger_path_hash": (
                stable_json_hash(str(Path(accepted_token_ledger_path).expanduser().resolve(strict=False)))
                if ledger_enabled
                else None
            ),
            "accepted_token_ledger_grant_rows_inserted_initial": int(ledger_grant_rows_inserted_initial),
            "accepted_token_ledger_grant_rows_inserted": int(ledger_grant_rows_inserted_initial),
            "accepted_token_ledger_submitted_rows": 0,
            "accepted_token_ledger_rows_for_shard_initial": int(ledger_shard_summary["row_count"]),
            "accepted_token_ledger_rows_for_shard": int(ledger_shard_summary["row_count"]),
            "accepted_token_ledger_accepted_token_count_lower_bound_for_shard": int(
                ledger_shard_summary["accepted_token_count_lower_bound"]
            ),
            "accepted_token_ledger_route_counts_for_shard": dict(ledger_shard_summary["route_counts"]),
            "accepted_token_ledger_commit_decision_counts_for_shard": dict(
                ledger_shard_summary["commit_decision_counts"]
            ),
            "accepted_token_ledger_persist_rows": int(ledger_row_counts.get("persist", 0)),
            "accepted_token_ledger_pending_rows": int(ledger_row_counts.get("pending", 0)),
            "accepted_token_ledger_after_stop_rows": int(ledger_row_counts.get("after_stop", 0)),
            "accepted_token_ledger_next_commit_ordinal": (
                None if ledger_state is None else int(ledger_state["next_commit_ordinal"])
            ),
            "accepted_token_ledger_stop_ordinal": None if ledger_state is None else ledger_state["stop_ordinal"],
            "score_persist_sqlite_pragmas": _sqlite_pragma_evidence(conn),
            "survivor_queue_rows": len(queue_rows),
            "scored_existing_initial": len(scored_existing),
            "pending_initial": len(pending),
            "scorer_batch_count": 0,
            "scorer_input_docs": 0,
            "score_model_call_count": 0,
            "score_oom_split_count": 0,
            "score_oom_failed_attempt_count": 0,
            "score_oom_failed_attempt_docs": 0,
            "score_oom_failed_attempt_estimated_windows": 0,
            "score_oom_failed_attempt_max_docs": 0,
            "score_oom_failed_attempt_max_estimated_windows": 0,
            "score_max_effective_batch_docs": 0,
            "score_prepared_batch_count": 0,
            "score_prepared_oom_split_count": 0,
            "score_prepare_submit_count": 0,
            "score_prepare_completed_count": 0,
            "score_prepare_failed_count": 0,
            "score_prepare_wait_seconds": 0.0,
            "score_window_budget_batch_count": 0,
            "score_window_budget_oversized_single_doc_batch_count": 0,
            "score_window_budget_max_estimated_windows_per_batch": 0,
            "score_pin_memory_requested_count": 0,
            "score_pin_memory_effective_count": 0,
            "score_pin_memory_seconds": 0.0,
            "score_rows_attempted": 0,
            "score_rows_inserted": 0,
            "score_input_fetch_seconds": 0.0,
            "score_model_seconds": 0.0,
            "score_persist_seconds": 0.0,
            "score_cuda_synchronized_timing": False,
            "score_cuda_synchronized_timing_requested": bool(getattr(w1_scorer, "synchronize_cuda_telemetry", False)),
            "score_cuda_synchronized_timing_observed": False,
            "score_tokenizer_call_count": 0,
            "score_tokenized_window_count": 0,
            "score_tokenization_seconds": 0.0,
            "score_device_transfer_seconds": 0.0,
            "score_window_forward_call_count": 0,
            "score_window_forward_window_count": 0,
            "score_window_forward_seconds": 0.0,
            "score_pooling_seconds": 0.0,
            "score_postprocess_seconds": 0.0,
            "score_cuda_synchronization_count": 0,
            "score_pooling_ragged_topk_call_count": 0,
            "score_pooling_ragged_topk_iteration_count": 0,
            "score_pooling_ragged_topk_input_window_count": 0,
            "score_pooling_ragged_topk_document_count": 0,
            "score_pooling_dense_padded_cell_count_avoided": 0,
            "score_max_windows_per_forward_call": 0,
            "score_max_tokenized_windows_per_document_batch": 0,
            "estimated_w1_window_count": 0,
            "scorer_device": _w1_scorer_device_observation(w1_scorer),
        }
        stop_requested = False
        accepted_token_target_reached = bool(
            enforce_accepted_token_target
            and accepted_token_count_lower_bound >= accepted_token_target_lower_bound
        )
        if ledger_enabled and ledger_state is not None:
            accepted_token_target_reached = bool(ledger_state["accepted_token_target_reached"])
        pending_score_records: list[tuple[Mapping[str, Any], float | None, str, str | None]] = []

        def refresh_ledger_shard_telemetry() -> None:
            nonlocal ledger_shard_summary
            if not ledger_enabled:
                return
            if ledger_conn is None or shard_index_for_ledger is None:
                raise ValueError("accepted-token ledger is not open")
            ledger_shard_summary = _w1_token_ledger_shard_summary(ledger_conn, shard_index_for_ledger)
            runtime_telemetry["accepted_token_ledger_rows_for_shard"] = int(ledger_shard_summary["row_count"])
            runtime_telemetry["accepted_token_ledger_accepted_token_count_lower_bound_for_shard"] = int(
                ledger_shard_summary["accepted_token_count_lower_bound"]
            )
            runtime_telemetry["accepted_token_ledger_route_counts_for_shard"] = dict(
                ledger_shard_summary["route_counts"]
            )
            runtime_telemetry["accepted_token_ledger_commit_decision_counts_for_shard"] = dict(
                ledger_shard_summary["commit_decision_counts"]
            )

        def persist_or_buffer_score_records(
            score_records: Sequence[tuple[Mapping[str, Any], float | None, str, str | None]],
        ) -> None:
            nonlocal accepted_token_count_lower_bound, accepted_token_target_reached, ledger_state, ledger_row_counts
            nonlocal ledger_grants_materialized_through_shard_row_index
            if not score_records:
                return
            if ledger_enabled:
                if ledger_conn is None or shard_index_for_ledger is None:
                    raise ValueError("accepted-token ledger is not open")
                runtime_telemetry["accepted_token_ledger_submitted_rows"] += len(score_records)
                ledger_state = _submit_w1_token_ledger_records(
                    ledger_conn,
                    score_records=score_records,
                    config_payload=config_payload,
                    w1_runtime_envelope_hash=w1_runtime_envelope_hash,
                    global_ordinal_by_candidate_id=global_ordinal_by_candidate_id,
                    global_ordinal_base=accepted_token_ledger_global_ordinal_base,
                    shard_index=shard_index_for_ledger,
                    rank_assignment_hash=accepted_token_ledger_rank_assignment_hash,
                )
                accepted_token_count_lower_bound = int(ledger_state["accepted_token_count_lower_bound"])
                accepted_token_target_reached = bool(ledger_state["accepted_token_target_reached"])
                runtime_telemetry["accepted_token_ledger_next_commit_ordinal"] = int(
                    ledger_state["next_commit_ordinal"]
                )
                runtime_telemetry["accepted_token_ledger_stop_ordinal"] = ledger_state["stop_ordinal"]
                runtime_telemetry["score_rows_attempted"] += len(score_records)
                return
            _buffer_w1_stage_batch_scores(
                conn=conn,
                config_payload=config_payload,
                runtime_telemetry=runtime_telemetry,
                pending_score_records=pending_score_records,
                score_records=score_records,
                w1_runtime_envelope_hash=w1_runtime_envelope_hash,
                commit_row_threshold=normalized_persist_threshold,
            )

        def materialize_ledger_grants_for_shard() -> None:
            nonlocal accepted_token_count_lower_bound, accepted_token_target_reached, ledger_state, ledger_row_counts
            nonlocal ledger_grants_materialized_through_shard_row_index
            if not ledger_enabled:
                return
            if ledger_conn is None or shard_index_for_ledger is None:
                raise ValueError("accepted-token ledger is not open")
            ledger_state = _w1_token_ledger_state(ledger_conn)
            accepted_token_count_lower_bound = int(ledger_state["accepted_token_count_lower_bound"])
            accepted_token_target_reached = bool(ledger_state["accepted_token_target_reached"])
            persist_at = time.monotonic()
            (
                inserted,
                next_materialized_shard_row_index,
            ) = _persist_w1_token_ledger_grants_for_shard_delta(
                ledger_conn=ledger_conn,
                shard_conn=conn,
                shard_index=shard_index_for_ledger,
                survivor_by_id=survivor_by_id,
                config_payload=config_payload,
                after_shard_row_index=ledger_grants_materialized_through_shard_row_index,
            )
            if next_materialized_shard_row_index is not None:
                ledger_grants_materialized_through_shard_row_index = next_materialized_shard_row_index
            runtime_telemetry["score_persist_seconds"] += time.monotonic() - persist_at
            runtime_telemetry["score_rows_inserted"] += int(inserted)
            runtime_telemetry["accepted_token_ledger_grant_rows_inserted"] += int(inserted)
            if inserted:
                runtime_telemetry["score_persist_commit_count"] += 1
                runtime_telemetry["score_persist_rows_per_commit"].append(int(inserted))
                runtime_telemetry["score_persist_max_rows_per_commit"] = max(
                    int(runtime_telemetry.get("score_persist_max_rows_per_commit", 0)),
                    int(inserted),
                )
            ledger_row_counts = _w1_token_ledger_row_counts(ledger_conn)
            runtime_telemetry["accepted_token_ledger_persist_rows"] = int(ledger_row_counts.get("persist", 0))
            runtime_telemetry["accepted_token_ledger_pending_rows"] = int(ledger_row_counts.get("pending", 0))
            runtime_telemetry["accepted_token_ledger_after_stop_rows"] = int(ledger_row_counts.get("after_stop", 0))
            runtime_telemetry["accepted_token_ledger_next_commit_ordinal"] = int(ledger_state["next_commit_ordinal"])
            runtime_telemetry["accepted_token_ledger_stop_ordinal"] = ledger_state["stop_ordinal"]
            refresh_ledger_shard_telemetry()

        def refresh_ledger_state_for_stop() -> None:
            nonlocal accepted_token_count_lower_bound, accepted_token_target_reached, ledger_state
            if not ledger_enabled:
                return
            if ledger_conn is None:
                raise ValueError("accepted-token ledger is not open")
            ledger_state = _w1_token_ledger_state(ledger_conn)
            accepted_token_count_lower_bound = int(ledger_state["accepted_token_count_lower_bound"])
            accepted_token_target_reached = bool(ledger_state["accepted_token_target_reached"])
            runtime_telemetry["accepted_token_ledger_next_commit_ordinal"] = int(ledger_state["next_commit_ordinal"])
            runtime_telemetry["accepted_token_ledger_stop_ordinal"] = ledger_state["stop_ordinal"]

        pending_batches = _iter_w1_pending_batches(
            pending,
            batch_size=batch,
            max_estimated_windows_per_prepare_batch=max_estimated_window_budget,
        )
        if normalized_prepare_pipeline == W1_PREPARE_PIPELINE_SERIAL:
            processed_pending_rows = 0
            for chunk, estimated_batch_windows, oversized_single in pending_batches:
                refresh_ledger_state_for_stop()
                if accepted_token_target_reached or _w1_stop_requested(run_path, extra_stop_paths):
                    stop_requested = True
                    break
                processed_pending_rows += len(chunk)
                runtime_telemetry["score_window_budget_batch_count"] += 1
                runtime_telemetry["score_window_budget_max_estimated_windows_per_batch"] = max(
                    int(runtime_telemetry.get("score_window_budget_max_estimated_windows_per_batch", 0)),
                    int(estimated_batch_windows),
                )
                if oversized_single:
                    runtime_telemetry["score_window_budget_oversized_single_doc_batch_count"] += 1
                input_fetch_at = time.monotonic()
                fetched = _fetch_w1_stage_batch(conn, chunk)
                runtime_telemetry["score_input_fetch_seconds"] += time.monotonic() - input_fetch_at
                score_records = list(fetched.score_records)
                if fetched.texts:
                    runtime_telemetry["estimated_w1_window_count"] += sum(fetched.estimated_window_counts)
                    scores = _score_w1_texts_with_oom_backoff(
                        w1_scorer=w1_scorer,
                        texts=list(fetched.texts),
                        metadata=list(fetched.metadata),
                        runtime_telemetry=runtime_telemetry,
                        estimated_window_counts=fetched.estimated_window_counts,
                    )
                    _append_w1_score_records_from_scores(
                        config_payload=config_payload,
                        scorable_rows=fetched.scorable_rows,
                        scores=scores,
                        score_records=score_records,
                    )
                if enforce_accepted_token_target and not ledger_enabled and score_records:
                    score_records, accepted_token_count_lower_bound, accepted_token_target_reached = (
                        _truncate_w1_score_records_at_accepted_token_target(
                            score_records,
                            accepted_token_count_lower_bound=accepted_token_count_lower_bound,
                            accepted_token_target_lower_bound=accepted_token_target_lower_bound,
                        )
                    )
                persist_or_buffer_score_records(score_records)
                if accepted_token_target_reached:
                    stop_requested = True
                    break
                if _w1_stop_requested(run_path, extra_stop_paths) and processed_pending_rows < len(pending):
                    stop_requested = True
                    break
        else:
            pin_memory = _w1_pin_memory_requested_for_batch(
                pin_memory_mode=normalized_pin_memory,
                w1_scorer=w1_scorer,
            )
            prepare_fetch_conn = _open_w1_stage_state_readonly(
                run_path,
                prefilter_config_payload=prefilter_config_payload,
                check_same_thread=False,
            )

            def fetch_and_submit(
                executor: ThreadPoolExecutor,
                item: tuple[list[Mapping[str, Any]], int, bool],
            ) -> Future[_PreparedW1StageBatch]:
                chunk, estimated_batch_windows, oversized_single = item
                runtime_telemetry["score_window_budget_batch_count"] += 1
                runtime_telemetry["score_window_budget_max_estimated_windows_per_batch"] = max(
                    int(runtime_telemetry.get("score_window_budget_max_estimated_windows_per_batch", 0)),
                    int(estimated_batch_windows),
                )
                if oversized_single:
                    runtime_telemetry["score_window_budget_oversized_single_doc_batch_count"] += 1
                runtime_telemetry["score_prepare_submit_count"] += 1
                return executor.submit(
                    _fetch_and_prepare_w1_stage_batch,
                    conn=prepare_fetch_conn,
                    chunk=tuple(chunk),
                    w1_scorer=w1_scorer,
                    pin_memory=pin_memory,
                )

            try:
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="w1-prepare") as executor:
                    batch_iter = iter(pending_batches)
                    current_future: Future[_PreparedW1StageBatch] | None = None
                    processed_pending_rows = 0
                    try:
                        refresh_ledger_state_for_stop()
                        if not accepted_token_target_reached and not _w1_stop_requested(run_path, extra_stop_paths):
                            current_future = fetch_and_submit(executor, next(batch_iter))
                        else:
                            stop_requested = True
                    except StopIteration:
                        current_future = None
                    while current_future is not None:
                        wait_at = time.monotonic()
                        try:
                            prepared_stage_batch = current_future.result()
                        except Exception:
                            runtime_telemetry["score_prepare_failed_count"] += 1
                            raise
                        runtime_telemetry["score_prepare_wait_seconds"] += time.monotonic() - wait_at
                        runtime_telemetry["score_input_fetch_seconds"] += prepared_stage_batch.input_fetch_seconds
                        runtime_telemetry["score_prepare_completed_count"] += 1
                        next_future: Future[_PreparedW1StageBatch] | None = None
                        try:
                            refresh_ledger_state_for_stop()
                            if not accepted_token_target_reached and not _w1_stop_requested(run_path, extra_stop_paths):
                                next_future = fetch_and_submit(executor, next(batch_iter))
                        except StopIteration:
                            next_future = None
                        fetched = prepared_stage_batch.fetched
                        processed_pending_rows += len(fetched.survivor_rows)
                        score_records = list(fetched.score_records)
                        if prepared_stage_batch.prepared is not None:
                            runtime_telemetry["estimated_w1_window_count"] += sum(fetched.estimated_window_counts)
                            scores = _score_w1_prepared_with_oom_backoff(
                                w1_scorer=w1_scorer,
                                prepared=prepared_stage_batch.prepared,
                                runtime_telemetry=runtime_telemetry,
                                estimated_window_counts=fetched.estimated_window_counts,
                            )
                            _append_w1_score_records_from_scores(
                                config_payload=config_payload,
                                scorable_rows=fetched.scorable_rows,
                                scores=scores,
                                score_records=score_records,
                            )
                        if enforce_accepted_token_target and not ledger_enabled and score_records:
                            score_records, accepted_token_count_lower_bound, accepted_token_target_reached = (
                                _truncate_w1_score_records_at_accepted_token_target(
                                    score_records,
                                    accepted_token_count_lower_bound=accepted_token_count_lower_bound,
                                    accepted_token_target_lower_bound=accepted_token_target_lower_bound,
                                )
                            )
                        persist_or_buffer_score_records(score_records)
                        if accepted_token_target_reached:
                            stop_requested = True
                            break
                        if _w1_stop_requested(run_path, extra_stop_paths) and (
                            next_future is not None or processed_pending_rows < len(pending)
                        ):
                            stop_requested = True
                            break
                        current_future = next_future
            finally:
                prepare_fetch_conn.close()
        _flush_w1_stage_score_buffer(
            conn=conn,
            config_payload=config_payload,
            runtime_telemetry=runtime_telemetry,
            pending_score_records=pending_score_records,
            w1_runtime_envelope_hash=w1_runtime_envelope_hash,
        )
        materialize_ledger_grants_for_shard()
        score_rows = _w1_score_export_rows(conn)
        if len(score_rows) == len(queue_rows):
            _write_jsonl_ready(run_path / "w1_scores.jsonl", score_rows)
        else:
            _write_jsonl_path((run_path / "w1_scores.jsonl").with_suffix(".jsonl.part"), score_rows)
        elapsed_seconds = time.monotonic() - start
        runtime_telemetry["score_stage_wall_seconds"] = elapsed_seconds
        stage_final_accepted = _w1_score_rows_accepted_token_lower_bound(
            score_rows,
            survivor_by_id=survivor_by_id,
        )
        if ledger_enabled and ledger_conn is not None:
            ledger_state = _w1_token_ledger_state(ledger_conn)
            accepted_token_count_lower_bound = int(ledger_state["accepted_token_count_lower_bound"])
            accepted_token_target_reached = bool(ledger_state["accepted_token_target_reached"])
            runtime_telemetry["accepted_token_ledger_next_commit_ordinal"] = int(ledger_state["next_commit_ordinal"])
            runtime_telemetry["accepted_token_ledger_stop_ordinal"] = ledger_state["stop_ordinal"]
            refresh_ledger_shard_telemetry()
            receipt_initial_lower_bound = max(0, accepted_token_count_lower_bound - int(stage_final_accepted))
        else:
            receipt_initial_lower_bound = accepted_token_initial_lower_bound
            accepted_token_count_lower_bound = int(stage_final_accepted) + accepted_token_initial_lower_bound
        runtime_telemetry["accepted_estimated_token_count_lower_bound_final"] = int(
            accepted_token_count_lower_bound
        )
        report = _build_w1_receipt(
            conn,
            config_payload,
            queue_rows=queue_rows,
            run_path=run_path,
            elapsed_seconds=elapsed_seconds,
            stop_requested=stop_requested,
            runtime_telemetry=runtime_telemetry,
            w1_runtime_envelope=w1_runtime_envelope,
            w1_runtime_envelope_hash=w1_runtime_envelope_hash,
            enforce_accepted_token_target=bool(enforce_accepted_token_target),
            accepted_token_count_initial_lower_bound=int(receipt_initial_lower_bound),
            allow_queue_exhaustion_before_token_target=bool(allow_queue_exhaustion_before_token_target),
        )
        _write_json_atomic(run_path / "w1_score_receipt.json", report)
        return report
    finally:
        _restore_w1_tokenizers_parallelism(tokenizers_had_original, tokenizers_original)
        if ledger_conn is not None:
            ledger_conn.close()
        conn.close()


def assemble_filtering_stage_report(
    *,
    config_payload: Mapping[str, Any],
    run_dir: str | Path,
    expected_prefilter_status: str = "prefilter_ready_for_w1",
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    run_path = Path(run_dir)
    validate_private_filtering_run_dir(run_path)
    early_w1_receipt_path = run_path / "w1_score_receipt.json"
    if early_w1_receipt_path.exists():
        early_w1_receipt = _load_json(early_w1_receipt_path)
        if isinstance(early_w1_receipt, Mapping) and early_w1_receipt.get("status") in {
            "accepted_token_target_reached_partial",
            "queue_exhausted_before_token_target",
        }:
            raise ValueError("assembled report cannot consume partial token-target W1 state before accepted token target review")
    queue_rows = _load_ready_jsonl(run_path / "w1_survivor_queue.jsonl", "survivor queue")
    score_rows = _load_ready_jsonl(run_path / "w1_scores.jsonl", "W1 scores")
    observed_prefilter_config_hash = queue_rows[0].get("config_hash") if queue_rows else config_payload["config_hash"]
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        observed_prefilter_config_hash,
        label="survivor queue",
    )
    _validate_survivor_queue_rows(queue_rows, prefilter_config_payload, allow_empty=True)
    survivor_by_id = {str(row["candidate_id_hash"]): row for row in queue_rows}
    _validate_w1_score_rows(score_rows, config_payload, survivor_by_id=survivor_by_id, allow_empty=True)
    survivor_ids = {str(row["candidate_id_hash"]) for row in queue_rows}
    scored_ids = {str(row["candidate_id_hash"]) for row in score_rows}
    score_by_id = {str(row["candidate_id_hash"]): row for row in score_rows}
    if survivor_ids != scored_ids:
        raise ValueError("W1 scored set must equal survivor set")
    conn = _open_existing_stage_state(run_path, prefilter_config_payload)
    try:
        _validate_prefilter_ready_receipt(
            conn,
            run_path,
            prefilter_config_payload,
            queue_rows=queue_rows,
            expected_status=expected_prefilter_status,
            require_compact_w1_scores_empty=False,
        )
        db_queue_rows = _prefilter_survivor_queue_rows(conn, prefilter_config_payload)
        if not _survivor_queue_rows_match_live_db(queue_rows, db_queue_rows):
            raise ValueError("survivor queue export does not match SQLite state")
        db_score_rows = _w1_score_export_rows(conn)
        if score_rows != db_score_rows:
            raise ValueError("W1 score export does not match SQLite state")
        w1_receipt_path = run_path / "w1_score_receipt.json"
        if not w1_receipt_path.exists():
            raise ValueError("assembled report requires W1 score receipt")
        w1_receipt = _load_json(w1_receipt_path)
        if not isinstance(w1_receipt, Mapping):
            raise ValueError("W1 score receipt must be an object")
        if w1_receipt.get("stage_receipt_hash") != stable_json_hash(
            {key: value for key, value in w1_receipt.items() if key != "stage_receipt_hash"}
        ):
            raise ValueError("W1 score receipt hash mismatch")
        w1_runtime_envelope, w1_runtime_envelope_hash = _validated_w1_runtime_envelope_from_receipt(w1_receipt)
        if score_rows and _w1_score_row_runtime_hashes(score_rows) != {w1_runtime_envelope_hash}:
            raise ValueError("W1 score row runtime envelope mismatch")
        db_survivor_ids = {
            str(row["candidate_id_hash"])
            for row in conn.execute(
                "select candidate_id_hash from prefilter_candidates where prefilter_route = 'prefilter_survivor'"
            )
        }
        if db_survivor_ids != survivor_ids:
            raise ValueError("DB survivor set must equal survivor queue set")
        db_score_ids = {str(row["candidate_id_hash"]) for row in conn.execute("select candidate_id_hash from w1_scores")}
        if db_score_ids != scored_ids:
            raise ValueError("DB W1 score set must equal W1 score export set")
        route_counts: Counter[str] = Counter()
        candidate_count = 0
        for row in conn.execute("select candidate_id_hash, prefilter_route from prefilter_candidates"):
            candidate_count += 1
            candidate_id = str(row["candidate_id_hash"])
            prefilter_route = str(row["prefilter_route"])
            if prefilter_route != "prefilter_survivor":
                if prefilter_route.endswith("error"):
                    route_counts["final_error"] += 1
                elif prefilter_route.endswith("quarantine"):
                    route_counts["final_quarantine"] += 1
                else:
                    route_counts["final_reject"] += 1
                continue
            score = score_by_id[candidate_id]
            if score["w1_route"] == "w1_pass":
                route_counts["final_safe"] += 1
            elif score["w1_route"] == "w1_error":
                route_counts["final_error"] += 1
            else:
                route_counts["final_reject"] += 1
        if candidate_count < len(survivor_ids):
            raise ValueError("candidate state is missing survivor rows")
        payload = {
            "artifact_kind": ASSEMBLED_STAGE_REPORT_KIND,
            "schema_version": 1,
            "status": "assembled_ready_for_review",
            "stop_reason": "assembled_complete",
            "next_required_phase": "coordinator_review_before_any_accepted_manifest_or_rental_scale",
            "config_hash": config_payload["config_hash"],
            "prefilter_producer_config_hash": prefilter_config_payload["config_hash"],
            "w1_consumer_config_hash": config_payload["config_hash"],
            "selected_policy_hash": config_payload["selected_policy_hash"],
            "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
            "source_metadata_hash": config_payload["source_metadata_hash"],
            "deterministic_filter_policy_hash": config_payload.get("deterministic_filter_policy_hash"),
            "candidate_count": int(candidate_count),
            "survivor_count": len(survivor_ids),
            "w1_scored_count": len(scored_ids),
            "survivor_w1_set_equality_validated": True,
            "w1_runtime_envelope": dict(w1_runtime_envelope),
            "w1_runtime_envelope_hash": w1_runtime_envelope_hash,
            "route_counts": _sorted_counter(route_counts),
            "accepted_manifest_created": False,
            "accepted_manifest": False,
            "tokenizer_shards_created": False,
            "packed_shards_created": False,
            "training_shards_created": False,
            "base_pretraining_started": False,
            "pretraining_started": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        payload["report_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "report_hash"})
        validate_public_no_raw_payload(payload)
        _write_json_atomic(run_path / "assembled_filtering_report.json", payload)
        return payload
    finally:
        conn.close()


def _elapsed_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"max": None, "p50": None, "p90": None}
    ordered = sorted(float(value) for value in values)
    return {
        "max": round(ordered[-1], 6),
        "p50": round(ordered[len(ordered) // 2], 6),
        "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 6),
    }


def _validate_sharded_w1_handoff_payload(
    handoff: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    manifest: Mapping[str, Any] | None = None,
    require_live_shard_state: bool = True,
    require_compact_w1_scores_empty: bool = False,
) -> None:
    if handoff.get("artifact_kind") != SHARDED_W1_HANDOFF_KIND:
        raise ValueError("sharded W1 handoff kind mismatch")
    if int(handoff.get("schema_version") or 0) != 1:
        raise ValueError("sharded W1 handoff schema_version mismatch")
    if handoff.get("handoff_hash") != stable_json_hash({key: value for key, value in handoff.items() if key != "handoff_hash"}):
        raise ValueError("sharded W1 handoff hash mismatch")
    if handoff.get("status") != "sharded_prefilter_ready_for_w1":
        raise ValueError("sharded W1 handoff is not ready")
    if handoff.get("supervisor_run_dir_sha256") != _prefilter_supervisor_run_dir_sha256(supervisor_run_dir):
        raise ValueError("sharded W1 handoff supervisor run dir mismatch")
    supervisor_path = Path(supervisor_run_dir)
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        handoff.get("config_hash"),
        label="sharded W1 handoff",
    )
    for field in (
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "deterministic_filter_policy_hash",
        "sparse_report_hash",
        "sparse_model_file_sha256",
    ):
        if handoff.get(field) != prefilter_config_payload.get(field):
            raise ValueError(f"sharded W1 handoff {field} mismatch")
    if handoff.get("w1_report_hash") != prefilter_config_payload["window_mil_report_hash"]:
        raise ValueError("sharded W1 handoff W1 report hash mismatch")
    if handoff.get("w1_model_file_sha256") != prefilter_config_payload["window_mil_model_file_sha256"]:
        raise ValueError("sharded W1 handoff W1 model hash mismatch")
    shards = handoff.get("shards")
    if not isinstance(shards, list) or len(shards) != int(handoff.get("shard_count", -1)):
        raise ValueError("sharded W1 handoff shard set mismatch")
    live_manifest = manifest
    if live_manifest is None and require_live_shard_state:
        manifest_path = supervisor_path / "prefilter_supervisor_manifest.json"
        if not manifest_path.exists():
            raise ValueError("sharded W1 handoff requires the supervisor manifest in the supervisor run dir")
        loaded_manifest = _load_json(manifest_path)
        if not isinstance(loaded_manifest, Mapping):
            raise ValueError("sharded W1 handoff supervisor manifest must be an object")
        live_manifest = loaded_manifest
    manifest_shards_by_index: dict[int, Mapping[str, Any]] = {}
    if live_manifest is not None:
        if live_manifest.get("manifest_hash") != stable_json_hash(_prefilter_supervisor_manifest_hash_payload(live_manifest)):
            raise ValueError("sharded W1 handoff supervisor manifest hash mismatch")
        if live_manifest.get("manifest_hash") != handoff.get("supervisor_manifest_hash"):
            raise ValueError("sharded W1 handoff supervisor_manifest_hash mismatch")
        if live_manifest.get("prefilter_plan_hash") != handoff.get("prefilter_plan_hash"):
            raise ValueError("sharded W1 handoff prefilter_plan_hash mismatch")
        if live_manifest.get("prefilter_plan_kind") != handoff.get("prefilter_plan_kind"):
            raise ValueError("sharded W1 handoff prefilter_plan_kind mismatch")
        if live_manifest.get("supervisor_run_dir_sha256") != handoff.get("supervisor_run_dir_sha256"):
            raise ValueError("sharded W1 handoff supervisor run dir mismatch")
        if int(live_manifest.get("shard_count", -1)) != int(handoff.get("shard_count", -1)):
            raise ValueError("sharded W1 handoff manifest shard_count mismatch")
        manifest_shards = live_manifest.get("shards")
        if not isinstance(manifest_shards, list):
            raise ValueError("sharded W1 handoff supervisor manifest shard set mismatch")
        manifest_shards_by_index = {int(shard["shard_index"]): shard for shard in manifest_shards}
    seen: set[int] = set()
    expected_physical_rows = 0
    terminal_rows = 0
    candidate_count = 0
    survivor_count = 0
    route_counts: Counter[str] = Counter()
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("sharded W1 handoff shard must be an object")
        index = int(shard.get("shard_index", -1))
        if index < 0 or index in seen:
            raise ValueError("sharded W1 handoff duplicate shard index")
        seen.add(index)
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_path, shard)
        _require_sha256(shard.get("prefilter_stage_receipt_hash"), "prefilter_stage_receipt_hash")
        _require_sha256(shard.get("survivor_queue_hash"), "survivor_queue_hash")
        expected_physical_rows += int(shard.get("expected_physical_rows", 0))
        terminal_rows += int(shard.get("terminal_rows", 0))
        candidate_count += int(shard.get("candidate_count", 0))
        survivor_count += int(shard.get("survivor_count", 0))
        route_counts.update({str(key): int(value) for key, value in dict(shard.get("route_counts") or {}).items()})
        manifest_shard = manifest_shards_by_index.get(index)
        if manifest_shard is not None:
            if manifest_shard.get("shard_run_dir_name") != shard.get("shard_run_dir_name"):
                raise ValueError("sharded W1 handoff shard_run_dir_name mismatch")
            if int(manifest_shard.get("expected_physical_rows", -1)) != int(shard.get("expected_physical_rows", -2)):
                raise ValueError("sharded W1 handoff shard expected_physical_rows mismatch")
        if require_live_shard_state:
            receipt_path = shard_dir / "prefilter_cpu_receipt.json"
            if not receipt_path.exists():
                raise ValueError("sharded W1 handoff shard prefilter receipt is missing")
            receipt = _load_json(receipt_path)
            if not isinstance(receipt, Mapping):
                raise ValueError("sharded W1 handoff shard prefilter receipt must be an object")
            expected_status = _w1_start_prefilter_expected_status(receipt)
            shard_prefilter_config_payload = _prefilter_input_config_payload_for_hash(
                prefilter_config_payload,
                receipt.get("config_hash"),
                label="sharded W1 handoff shard prefilter receipt",
            )
            if _is_w1_compact_prefilter_receipt(receipt):
                if _prefilter_receipt_has_survivor_queue_id_index_proof(receipt):
                    queue_count = _validate_survivor_queue_id_index_ready_file(
                        shard_dir / "w1_survivor_queue.jsonl",
                        allow_empty=True,
                        expected_hash=receipt.get("survivor_queue_id_index_hash"),
                        expected_row_count=int(
                            receipt.get("survivor_queue_id_index_row_count", receipt.get("survivor_count", 0))
                        ),
                        expected_candidate_id_set_hash=receipt.get("survivor_candidate_id_set_hash"),
                        expected_private_evidence_id_set_hash=receipt.get("survivor_private_evidence_id_set_hash"),
                    )
                    _validate_w1_compact_receipt_proof_payload(
                        receipt,
                        shard_dir,
                        shard_prefilter_config_payload,
                        queue_row_count=queue_count,
                        expected_status=expected_status,
                        require_empty_w1_scores=require_compact_w1_scores_empty,
                    )
                else:
                    queue_rows = _load_ready_jsonl(shard_dir / "w1_survivor_queue.jsonl", "prefilter survivor queue")
                    conn = _open_existing_stage_state(shard_dir, shard_prefilter_config_payload)
                    try:
                        _validate_prefilter_ready_receipt(
                            conn,
                            shard_dir,
                            shard_prefilter_config_payload,
                            queue_rows=queue_rows,
                            expected_status=expected_status,
                            require_compact_w1_scores_empty=require_compact_w1_scores_empty,
                        )
                    finally:
                        conn.close()
            elif _prefilter_receipt_has_survivor_queue_id_proof(receipt):
                _validate_prefilter_ready_receipt_payload(
                    receipt,
                    shard_dir,
                    shard_prefilter_config_payload,
                    queue_row_count=int(shard.get("survivor_count", 0)),
                    expected_status=expected_status,
                )
            else:
                queue_count = _validate_survivor_queue_ready_file(
                    shard_dir / "w1_survivor_queue.jsonl",
                    shard_prefilter_config_payload,
                    allow_empty=True,
                )
                conn = _open_existing_stage_state(shard_dir, shard_prefilter_config_payload)
                try:
                    _validate_prefilter_ready_receipt(
                        conn,
                        shard_dir,
                        shard_prefilter_config_payload,
                        queue_row_count=queue_count,
                        expected_status=expected_status,
                    )
                finally:
                    conn.close()
            if receipt.get("stage_receipt_hash") != shard.get("prefilter_stage_receipt_hash"):
                raise ValueError("sharded W1 handoff shard prefilter_stage_receipt_hash mismatch")
            for field in ("expected_physical_rows", "terminal_rows", "candidate_count", "survivor_count", "route_counts"):
                if receipt.get(field) != shard.get(field):
                    raise ValueError(f"sharded W1 handoff shard {field} mismatch")
            for field in (
                "survivor_queue_row_count",
                "survivor_queue_id_index_hash",
                "survivor_queue_id_index_row_count",
                "survivor_candidate_id_set_hash",
                "survivor_private_evidence_id_set_hash",
            ):
                if shard.get(field) is not None and receipt.get(field) != shard.get(field):
                    raise ValueError(f"sharded W1 handoff shard {field} mismatch")
    if seen != set(range(int(handoff["shard_count"]))):
        raise ValueError("sharded W1 handoff shard indexes are not contiguous")
    aggregate_expected = {
        "expected_physical_rows": int(expected_physical_rows),
        "terminal_rows": int(terminal_rows),
        "candidate_count": int(candidate_count),
        "survivor_count": int(survivor_count),
        "route_counts": _sorted_counter(route_counts),
    }
    for field, expected_value in aggregate_expected.items():
        if handoff.get(field) != expected_value:
            raise ValueError(f"sharded W1 handoff {field} mismatch")
    _require_false_flags(handoff, PUBLIC_FALSE_FLAGS, "sharded W1 handoff")
    validate_public_no_raw_payload(handoff)


def build_sharded_w1_handoff(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    handoff_path: str | Path | None = None,
    full_audit: bool = False,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    plan = dict(prefilter_plan)
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
        expect_merge_allowed=True,
    )
    seen_candidates: set[str] | None = set()
    seen_evidence: set[str] | None = set()
    route_counts: Counter[str] = Counter()
    timing_sums: Counter[str] = Counter()
    shard_payloads: list[dict[str, Any]] = []
    elapsed_values: list[float] = []
    candidate_count = 0
    survivor_count = 0
    terminal_rows = 0
    prefilter_producer_config_hashes: set[str] = set()
    for shard in sorted(manifest["shards"], key=lambda item: int(item["shard_index"])):
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
        receipt = _validate_prefilter_supervisor_ready_shard(
            shard_run_dir=shard_dir,
            shard=shard,
            manifest=manifest,
            prefilter_plan=plan,
            config_payload=config_payload,
            seen_survivor_candidate_ids=seen_candidates,
            seen_private_evidence_ids=seen_evidence,
            audit_sqlite_state=full_audit,
        )
        prefilter_producer_config_hashes.add(
            _require_sha256(receipt.get("config_hash"), "sharded W1 handoff shard producer config_hash")
        )
        if full_audit:
            shard_prefilter_config_payload = _prefilter_input_config_payload_for_hash(
                config_payload,
                receipt.get("config_hash"),
                label="sharded W1 handoff shard prefilter receipt",
            )
            queue_rows = _load_ready_jsonl(shard_dir / "w1_survivor_queue.jsonl", "prefilter survivor queue")
            _validate_survivor_queue_rows(queue_rows, shard_prefilter_config_payload, allow_empty=True)
            conn = _open_stage_state(shard_dir, shard_prefilter_config_payload)
            try:
                receipt = _load_json(shard_dir / "prefilter_cpu_receipt.json")
                if not isinstance(receipt, Mapping):
                    raise ValueError("sharded W1 handoff shard prefilter receipt must be an object")
                _validate_prefilter_ready_receipt(
                    conn,
                    shard_dir,
                    shard_prefilter_config_payload,
                    queue_rows=queue_rows,
                    expected_status=_w1_start_prefilter_expected_status(receipt),
                )
                if not _survivor_queue_rows_match_live_db(
                    queue_rows,
                    _prefilter_survivor_queue_rows(conn, shard_prefilter_config_payload),
                ):
                    raise ValueError("sharded W1 handoff survivor queue export does not match SQLite state")
            finally:
                conn.close()
        shard_route_counts = {str(key): int(value) for key, value in dict(receipt.get("route_counts") or {}).items()}
        route_counts.update(shard_route_counts)
        for field in PREFILTER_TIMING_SECOND_FIELDS:
            timing_sums[field] += float(receipt.get(f"prefilter_{field}_seconds", 0.0) or 0.0)
        candidate_count += int(receipt.get("candidate_count", 0))
        survivor_count += int(receipt.get("survivor_count", 0))
        terminal_rows += int(receipt.get("terminal_rows", 0))
        elapsed_values.append(float(receipt.get("elapsed_seconds", 0.0) or 0.0))
        has_queue_proof = _prefilter_receipt_has_survivor_queue_id_proof(receipt)
        shard_payloads.append(
            {
                "shard_index": int(shard["shard_index"]),
                "shard_run_dir_name": str(shard["shard_run_dir_name"]),
                "expected_physical_rows": int(shard["expected_physical_rows"]),
                "terminal_rows": int(receipt.get("terminal_rows", 0)),
                "candidate_count": int(receipt.get("candidate_count", 0)),
                "survivor_count": int(receipt.get("survivor_count", 0)),
                "route_counts": shard_route_counts,
                "elapsed_seconds": round(float(receipt.get("elapsed_seconds", 0.0) or 0.0), 6),
                "prefilter_stage_receipt_hash": receipt["stage_receipt_hash"],
                "survivor_queue_hash": receipt["survivor_queue_hash"],
                "survivor_queue_row_count": int(receipt["survivor_queue_row_count"]) if has_queue_proof else None,
                "survivor_queue_id_index_hash": receipt.get("survivor_queue_id_index_hash") if has_queue_proof else None,
                "survivor_queue_id_index_row_count": (
                    int(receipt["survivor_queue_id_index_row_count"])
                    if has_queue_proof and receipt.get("survivor_queue_id_index_row_count") is not None
                    else None
                ),
                "survivor_candidate_id_set_hash": receipt.get("survivor_candidate_id_set_hash") if has_queue_proof else None,
                "survivor_private_evidence_id_set_hash": receipt.get("survivor_private_evidence_id_set_hash") if has_queue_proof else None,
            }
        )
    if len(prefilter_producer_config_hashes) != 1:
        raise ValueError("sharded W1 handoff shards have inconsistent producer config hashes")
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        next(iter(prefilter_producer_config_hashes)),
        label="sharded W1 handoff shard producer",
    )
    payload = {
        "artifact_kind": SHARDED_W1_HANDOFF_KIND,
        "schema_version": 1,
        "status": "sharded_prefilter_ready_for_w1",
        "next_required_phase": "w1-score-sharded",
        "prefilter_plan_hash": plan["schedule_hash"],
        "prefilter_plan_kind": plan["artifact_kind"],
        "supervisor_manifest_hash": manifest["manifest_hash"],
        "supervisor_run_dir_sha256": manifest["supervisor_run_dir_sha256"],
        "config_hash": prefilter_config_payload["config_hash"],
        "selected_policy_hash": prefilter_config_payload["selected_policy_hash"],
        "selected_policy_report_hash": prefilter_config_payload["selected_policy_report_hash"],
        "source_metadata_hash": prefilter_config_payload["source_metadata_hash"],
        "deterministic_filter_policy_hash": prefilter_config_payload["deterministic_filter_policy_hash"],
        "sparse_report_hash": prefilter_config_payload["sparse_report_hash"],
        "sparse_model_file_sha256": prefilter_config_payload["sparse_model_file_sha256"],
        "w1_report_hash": prefilter_config_payload["window_mil_report_hash"],
        "w1_model_file_sha256": prefilter_config_payload["window_mil_model_file_sha256"],
        "shard_count": int(manifest["shard_count"]),
        "expected_physical_rows": int(manifest["expected_physical_rows"]),
        "terminal_rows": int(terminal_rows),
        "candidate_count": int(candidate_count),
        "survivor_count": int(survivor_count),
        "route_counts": _sorted_counter(route_counts),
        "prefilter_timing_seconds": {field: round(float(timing_sums[field]), 6) for field in PREFILTER_TIMING_SECOND_FIELDS},
        "shard_elapsed_seconds_summary": _elapsed_summary(elapsed_values),
        "shards": shard_payloads,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["handoff_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "handoff_hash"})
    _validate_sharded_w1_handoff_payload(
        payload,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
        manifest=manifest,
        require_compact_w1_scores_empty=True,
    )
    if handoff_path is not None:
        _write_json_atomic(handoff_path, payload)
    return payload


def _validate_partial_sharded_w1_diagnostic_handoff_payload(
    handoff: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
    supervisor_run_dir: str | Path,
) -> None:
    if handoff.get("artifact_kind") != PARTIAL_SHARDED_W1_DIAGNOSTIC_HANDOFF_KIND:
        raise ValueError("partial W1 diagnostic handoff artifact_kind mismatch")
    if handoff.get("diagnostic_handoff_hash") != stable_json_hash(
        {key: value for key, value in handoff.items() if key != "diagnostic_handoff_hash"}
    ):
        raise ValueError("partial W1 diagnostic handoff hash mismatch")
    if handoff.get("status") != "partial_shard_prefilter_ready_for_w1_diagnostic":
        raise ValueError("partial W1 diagnostic handoff status mismatch")
    if handoff.get("claims_scope") != "throughput_diagnostic_only":
        raise ValueError("partial W1 diagnostic handoff claims_scope mismatch")
    if bool(handoff.get("canonical_full_handoff_satisfied")):
        raise ValueError("partial W1 diagnostic handoff must not claim canonical full handoff")
    source_supervisor_hash = _require_sha256(
        handoff.get("source_supervisor_run_dir_sha256"),
        "source_supervisor_run_dir_sha256",
    )
    local_supervisor_hash = _require_sha256(
        handoff.get("local_supervisor_run_dir_sha256"),
        "local_supervisor_run_dir_sha256",
    )
    if source_supervisor_hash == local_supervisor_hash:
        raise ValueError("partial W1 diagnostic handoff requires a copied supervisor run dir")
    if int(handoff.get("selected_shard_count", 0)) < 1:
        raise ValueError("partial W1 diagnostic handoff requires selected shards")
    if int(handoff.get("selected_shard_count", 0)) != len(list(handoff.get("shards") or [])):
        raise ValueError("partial W1 diagnostic selected_shard_count mismatch")
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        handoff.get("config_hash"),
        label="partial W1 diagnostic handoff",
    )
    for field in (
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "deterministic_filter_policy_hash",
        "sparse_report_hash",
        "sparse_model_file_sha256",
        "w1_report_hash",
        "w1_model_file_sha256",
    ):
        expected = {
            "w1_report_hash": "window_mil_report_hash",
            "w1_model_file_sha256": "window_mil_model_file_sha256",
        }.get(field, field)
        expected_value = prefilter_config_payload[expected]
        if handoff.get(field) != expected_value:
            raise ValueError(f"partial W1 diagnostic handoff {field} mismatch")
    selected_indexes = [int(item) for item in handoff.get("selected_shard_indexes") or []]
    shards = list(handoff.get("shards") or [])
    shard_indexes = [int(shard.get("shard_index", -1)) for shard in shards]
    if selected_indexes != shard_indexes:
        raise ValueError("partial W1 diagnostic selected shard indexes mismatch")
    aggregate_route_counts: Counter[str] = Counter()
    aggregate_expected = {
        "expected_physical_rows": 0,
        "terminal_rows": 0,
        "candidate_count": 0,
        "survivor_count": 0,
    }
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("partial W1 diagnostic handoff shard must be an object")
        for field in aggregate_expected:
            aggregate_expected[field] += int(shard.get(field, 0))
        aggregate_route_counts.update(
            {str(key): int(value) for key, value in dict(shard.get("route_counts") or {}).items()}
        )
    aggregate_expected["route_counts"] = _sorted_counter(aggregate_route_counts)
    for field, expected_value in aggregate_expected.items():
        if handoff.get(field) != expected_value:
            raise ValueError(f"partial W1 diagnostic handoff {field} mismatch")
    if local_supervisor_hash != _prefilter_supervisor_run_dir_sha256(supervisor_run_dir):
        raise ValueError("partial W1 diagnostic local supervisor_run_dir_sha256 mismatch")
    _require_false_flags(handoff, PUBLIC_FALSE_FLAGS, "partial W1 diagnostic handoff")
    validate_public_no_raw_payload(handoff)


def _validate_partial_sharded_w1_diagnostic_live_shard_state(
    handoff: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
    supervisor_run_dir: str | Path,
) -> None:
    supervisor_path = Path(supervisor_run_dir)
    for shard in handoff.get("shards") or []:
        if not isinstance(shard, Mapping):
            raise ValueError("partial W1 diagnostic live shard must be an object")
        shard_index = int(shard.get("shard_index", -1))
        if shard_index < 0:
            raise ValueError("partial W1 diagnostic live shard index mismatch")
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_path, shard)
        receipt_path = shard_dir / "prefilter_cpu_receipt.json"
        if not receipt_path.exists():
            raise ValueError("partial W1 diagnostic live shard prefilter receipt is missing")
        receipt = _load_json(receipt_path)
        if not isinstance(receipt, Mapping):
            raise ValueError("partial W1 diagnostic live shard prefilter receipt must be an object")
        if _prefilter_receipt_has_survivor_queue_id_proof(receipt):
            prefilter_config_payload = _prefilter_input_config_payload_for_hash(
                config_payload,
                receipt.get("config_hash"),
                label="partial W1 diagnostic live shard prefilter receipt",
            )
            _validate_prefilter_ready_receipt_payload(
                receipt,
                shard_dir,
                prefilter_config_payload,
                queue_row_count=int(shard.get("survivor_count", 0)),
                expected_status="prefilter_shard_ready_for_merge",
            )
        else:
            prefilter_config_payload = _prefilter_input_config_payload_for_hash(
                config_payload,
                receipt.get("config_hash"),
                label="partial W1 diagnostic live shard prefilter receipt",
            )
            queue_count = _validate_survivor_queue_ready_file(
                shard_dir / "w1_survivor_queue.jsonl",
                prefilter_config_payload,
                allow_empty=True,
            )
            conn = _open_existing_stage_state(shard_dir, prefilter_config_payload)
            try:
                _validate_prefilter_ready_receipt(
                    conn,
                    shard_dir,
                    prefilter_config_payload,
                    queue_row_count=queue_count,
                    expected_status="prefilter_shard_ready_for_merge",
                )
            finally:
                conn.close()
        if receipt.get("stage_receipt_hash") != shard.get("prefilter_stage_receipt_hash"):
            raise ValueError("partial W1 diagnostic live shard prefilter_stage_receipt_hash mismatch")
        if int(receipt.get("row_group_shard_index", shard_index)) != shard_index:
            raise ValueError("partial W1 diagnostic live shard index mismatch")
        if receipt.get("supervisor_manifest_hash") is not None and receipt.get("supervisor_manifest_hash") != handoff.get(
            "supervisor_manifest_hash"
        ):
            raise ValueError("partial W1 diagnostic live shard supervisor_manifest_hash mismatch")
        for field in ("expected_physical_rows", "terminal_rows", "candidate_count", "survivor_count", "route_counts"):
            if receipt.get(field) != shard.get(field):
                raise ValueError(f"partial W1 diagnostic live shard {field} mismatch")
        for field in (
            "survivor_queue_row_count",
            "survivor_queue_id_index_hash",
            "survivor_queue_id_index_row_count",
            "survivor_candidate_id_set_hash",
            "survivor_private_evidence_id_set_hash",
        ):
            if shard.get(field) is not None and receipt.get(field) != shard.get(field):
                raise ValueError(f"partial W1 diagnostic live shard {field} mismatch")


def build_partial_sharded_w1_diagnostic_handoff(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    shard_indexes: Sequence[int],
    handoff_path: str | Path | None = None,
    full_audit: bool = False,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    plan = dict(prefilter_plan)
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        manifest.get("config_hash"),
        label="partial W1 diagnostic supervisor manifest",
    )
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=plan,
        config_payload=prefilter_config_payload,
        expect_merge_allowed=True,
    )
    selected_index_set = {int(index) for index in shard_indexes}
    if not selected_index_set:
        raise ValueError("partial W1 diagnostic handoff requires at least one shard index")
    manifest_shards = {int(shard["shard_index"]): shard for shard in manifest["shards"]}
    missing_indexes = sorted(selected_index_set.difference(manifest_shards))
    if missing_indexes:
        raise ValueError("partial W1 diagnostic shard index is not present in supervisor manifest")
    seen_candidates: set[str] | None = set()
    seen_evidence: set[str] | None = set()
    source_supervisor_run_dir_hashes: set[str] = set()
    route_counts: Counter[str] = Counter()
    timing_sums: Counter[str] = Counter()
    shard_payloads: list[dict[str, Any]] = []
    elapsed_values: list[float] = []
    candidate_count = 0
    survivor_count = 0
    terminal_rows = 0
    for shard_index in sorted(selected_index_set):
        shard = manifest_shards[shard_index]
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
        receipt = _validate_prefilter_supervisor_ready_shard(
            shard_run_dir=shard_dir,
            shard=shard,
            manifest=manifest,
            prefilter_plan=plan,
            config_payload=prefilter_config_payload,
            seen_survivor_candidate_ids=seen_candidates,
            seen_private_evidence_ids=seen_evidence,
            audit_sqlite_state=full_audit,
        )
        if receipt.get("source_supervisor_run_dir_sha256") is not None:
            source_supervisor_run_dir_hashes.add(
                _require_sha256(
                    receipt.get("source_supervisor_run_dir_sha256"),
                    "partial W1 diagnostic source_supervisor_run_dir_sha256",
                )
            )
        shard_route_counts = {str(key): int(value) for key, value in dict(receipt.get("route_counts") or {}).items()}
        route_counts.update(shard_route_counts)
        for field in PREFILTER_TIMING_SECOND_FIELDS:
            timing_sums[field] += float(receipt.get(f"prefilter_{field}_seconds", 0.0) or 0.0)
        candidate_count += int(receipt.get("candidate_count", 0))
        survivor_count += int(receipt.get("survivor_count", 0))
        terminal_rows += int(receipt.get("terminal_rows", 0))
        elapsed_values.append(float(receipt.get("elapsed_seconds", 0.0) or 0.0))
        has_queue_proof = _prefilter_receipt_has_survivor_queue_id_proof(receipt)
        shard_payloads.append(
            {
                "shard_index": int(shard["shard_index"]),
                "shard_run_dir_name": str(shard["shard_run_dir_name"]),
                "expected_physical_rows": int(shard["expected_physical_rows"]),
                "terminal_rows": int(receipt.get("terminal_rows", 0)),
                "candidate_count": int(receipt.get("candidate_count", 0)),
                "survivor_count": int(receipt.get("survivor_count", 0)),
                "route_counts": shard_route_counts,
                "elapsed_seconds": round(float(receipt.get("elapsed_seconds", 0.0) or 0.0), 6),
                "prefilter_stage_receipt_hash": receipt["stage_receipt_hash"],
                "survivor_queue_hash": receipt["survivor_queue_hash"],
                "survivor_queue_row_count": int(receipt["survivor_queue_row_count"]) if has_queue_proof else None,
                "survivor_queue_id_index_hash": receipt.get("survivor_queue_id_index_hash") if has_queue_proof else None,
                "survivor_queue_id_index_row_count": (
                    int(receipt["survivor_queue_id_index_row_count"])
                    if has_queue_proof and receipt.get("survivor_queue_id_index_row_count") is not None
                    else None
                ),
                "survivor_candidate_id_set_hash": receipt.get("survivor_candidate_id_set_hash") if has_queue_proof else None,
                "survivor_private_evidence_id_set_hash": receipt.get("survivor_private_evidence_id_set_hash") if has_queue_proof else None,
            }
        )
    source_supervisor_run_dir_sha256 = _require_sha256(
        manifest["supervisor_run_dir_sha256"],
        "partial W1 diagnostic manifest supervisor_run_dir_sha256",
    )
    if source_supervisor_run_dir_hashes:
        if len(source_supervisor_run_dir_hashes) != 1:
            raise ValueError("partial W1 diagnostic compact shards have inconsistent source supervisor run dir hashes")
        source_supervisor_run_dir_sha256 = next(iter(source_supervisor_run_dir_hashes))
    payload = {
        "artifact_kind": PARTIAL_SHARDED_W1_DIAGNOSTIC_HANDOFF_KIND,
        "schema_version": 1,
        "status": "partial_shard_prefilter_ready_for_w1_diagnostic",
        "claims_scope": "throughput_diagnostic_only",
        "canonical_full_handoff_satisfied": False,
        "next_required_phase": "w1-score-partial-sharded-diagnostic",
        "prefilter_plan_hash": plan["schedule_hash"],
        "prefilter_plan_kind": plan["artifact_kind"],
        "supervisor_manifest_hash": manifest["manifest_hash"],
        "source_supervisor_run_dir_sha256": source_supervisor_run_dir_sha256,
        "local_supervisor_run_dir_sha256": _prefilter_supervisor_run_dir_sha256(supervisor_run_dir),
        "config_hash": prefilter_config_payload["config_hash"],
        "selected_policy_hash": prefilter_config_payload["selected_policy_hash"],
        "selected_policy_report_hash": prefilter_config_payload["selected_policy_report_hash"],
        "source_metadata_hash": prefilter_config_payload["source_metadata_hash"],
        "deterministic_filter_policy_hash": prefilter_config_payload["deterministic_filter_policy_hash"],
        "sparse_report_hash": prefilter_config_payload["sparse_report_hash"],
        "sparse_model_file_sha256": prefilter_config_payload["sparse_model_file_sha256"],
        "w1_report_hash": prefilter_config_payload["window_mil_report_hash"],
        "w1_model_file_sha256": prefilter_config_payload["window_mil_model_file_sha256"],
        "full_shard_count": int(manifest["shard_count"]),
        "selected_shard_count": len(shard_payloads),
        "selected_shard_indexes": [int(shard["shard_index"]) for shard in shard_payloads],
        "expected_physical_rows": int(terminal_rows),
        "terminal_rows": int(terminal_rows),
        "candidate_count": int(candidate_count),
        "survivor_count": int(survivor_count),
        "route_counts": _sorted_counter(route_counts),
        "prefilter_timing_seconds": {field: round(float(timing_sums[field]), 6) for field in PREFILTER_TIMING_SECOND_FIELDS},
        "shard_elapsed_seconds_summary": _elapsed_summary(elapsed_values),
        "shards": shard_payloads,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["diagnostic_handoff_hash"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "diagnostic_handoff_hash"}
    )
    _validate_partial_sharded_w1_diagnostic_handoff_payload(
        payload,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
    )
    if handoff_path is not None:
        _write_json_atomic(handoff_path, payload)
    return payload


def _w1_report_ledger_shard_scored_count(report: Mapping[str, Any]) -> int:
    telemetry = report.get("w1_runtime_telemetry")
    if not isinstance(telemetry, Mapping):
        return int(report.get("scored_count", 0) or 0)
    return max(
        int(report.get("scored_count", 0) or 0),
        int(telemetry.get("accepted_token_ledger_rows_for_shard", 0) or 0),
    )


def _w1_report_complete_for_sharded_rank(report: Mapping[str, Any], *, ledger_enabled: bool) -> bool:
    if report.get("status") == "w1_scored":
        return True
    if not ledger_enabled:
        return False
    survivor_count = int(report.get("survivor_count", 0) or 0)
    if survivor_count < 1:
        return False
    return _w1_report_ledger_shard_scored_count(report) == survivor_count


def _w1_report_route_counts_for_sharded_rank(report: Mapping[str, Any], *, ledger_enabled: bool) -> dict[str, int]:
    if ledger_enabled:
        telemetry = report.get("w1_runtime_telemetry")
        if isinstance(telemetry, Mapping):
            ledger_route_counts = telemetry.get("accepted_token_ledger_route_counts_for_shard")
            if isinstance(ledger_route_counts, Mapping):
                return {str(key): int(value) for key, value in ledger_route_counts.items()}
    return {str(key): int(value) for key, value in dict(report.get("route_counts") or {}).items()}


def _build_w1_ledger_fast_skip_receipt(
    *,
    config_payload: Mapping[str, Any],
    handoff: Mapping[str, Any],
    shard: Mapping[str, Any],
    ledger_state: Mapping[str, Any],
    ledger_shard_summary: Mapping[str, Any],
    w1_runtime_envelope: Mapping[str, Any],
    w1_runtime_envelope_hash: str,
    rank_assignment_hash: str | None,
) -> dict[str, Any]:
    survivor_count = int(shard.get("survivor_count", 0))
    ledger_row_count = int(ledger_shard_summary.get("row_count", 0) or 0)
    telemetry = _normalize_w1_runtime_telemetry(
        {
            "survivor_queue_rows": survivor_count,
            "pending_initial": 0,
            "accepted_token_ledger_enabled": True,
            "accepted_token_ledger_rows_for_shard_initial": ledger_row_count,
            "accepted_token_ledger_rows_for_shard": ledger_row_count,
            "accepted_token_ledger_accepted_token_count_lower_bound_for_shard": int(
                ledger_shard_summary.get("accepted_token_count_lower_bound", 0) or 0
            ),
            "accepted_token_ledger_route_counts_for_shard": dict(
                ledger_shard_summary.get("route_counts") or {}
            ),
            "accepted_token_ledger_commit_decision_counts_for_shard": dict(
                ledger_shard_summary.get("commit_decision_counts") or {}
            ),
            "accepted_token_ledger_next_commit_ordinal": int(ledger_state["next_commit_ordinal"]),
            "accepted_token_ledger_stop_ordinal": ledger_state["stop_ordinal"],
            "accepted_token_ledger_fast_skipped_shards": 1,
            "accepted_token_ledger_fast_skipped_rows": ledger_row_count,
            "score_stage_wall_seconds": 0.0,
        }
    )
    payload = {
        "artifact_kind": W1_SCORE_STAGE_KIND,
        "schema_version": 1,
        "phase": "w1-score",
        "status": "w1_ledger_fast_skipped",
        "stop_reason": "accepted_token_ledger_complete_shard_fast_skip",
        "next_required_phase": "resume_w1_score_sharded",
        "config_hash": config_payload["config_hash"],
        "prefilter_producer_config_hash": handoff["config_hash"],
        "w1_consumer_config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "survivor_count": survivor_count,
        "survivor_queue_hash": _require_sha256(shard.get("survivor_queue_hash"), "survivor_queue_hash"),
        "scored_count": ledger_row_count,
        "accepted_token_target_enforced": True,
        "accepted_token_target_lower_bound": int(config_payload["accepted_token_target_lower_bound"]),
        "accepted_token_initial_lower_bound": 0,
        "accepted_estimated_token_count_lower_bound_stage": int(
            ledger_shard_summary.get("accepted_token_count_lower_bound", 0) or 0
        ),
        "accepted_estimated_token_count_lower_bound": int(
            ledger_state["accepted_token_count_lower_bound"]
        ),
        "accepted_token_target_reached": bool(ledger_state["accepted_token_target_reached"]),
        "accepted_token_queue_exhaustion_deferred_to_sharded_summary": False,
        "route_counts": dict(ledger_shard_summary.get("route_counts") or {}),
        "w1_scores_hash": None,
        "w1_scores_part_hash": None,
        "elapsed_seconds": 0.0,
        "docs_scored_this_invocation": 0,
        "docs_per_second": 0.0,
        "batching_controls": {"batch_size_defaulted_or_requested": True},
        "w1_runtime_envelope": dict(w1_runtime_envelope),
        "w1_runtime_envelope_hash": _require_sha256(w1_runtime_envelope_hash, "w1_runtime_envelope_hash"),
        "w1_runtime_telemetry": telemetry,
        "rank_assignment_hash": rank_assignment_hash,
        "gpu_required_for_local_tests": False,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["stage_receipt_hash"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "stage_receipt_hash"}
    )
    validate_public_no_raw_payload(payload)
    return payload


def _w1_homogeneous_complete_ledger_shard_state(
    *,
    ledger_conn: sqlite3.Connection,
    shard: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    ledger_state = _w1_token_ledger_state(ledger_conn)
    ledger_shard_summary = _w1_token_ledger_shard_summary(ledger_conn, int(shard["shard_index"]))
    commit_decision_counts = {
        str(decision): int(count)
        for decision, count in dict(ledger_shard_summary.get("commit_decision_counts") or {}).items()
        if int(count) > 0
    }
    if (
        int(ledger_shard_summary["row_count"]) == int(shard.get("survivor_count", 0))
        and len(commit_decision_counts) == 1
    ):
        return ledger_state, ledger_shard_summary
    return None


def _w1_homogeneous_complete_ledger_shard_state_from_path(
    *,
    ledger_path: str | Path,
    ledger_metadata: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    ledger_conn = _connect_w1_accepted_token_ledger(
        Path(ledger_path),
        metadata=ledger_metadata,
    )
    try:
        return _w1_homogeneous_complete_ledger_shard_state(ledger_conn=ledger_conn, shard=shard)
    finally:
        ledger_conn.close()


def run_w1_score_sharded(
    *,
    config_payload: Mapping[str, Any],
    handoff: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    w1_scorer: ChunkFilteringScorer,
    batch_size: int = 16,
    shard_index_modulus: int = 1,
    shard_index_remainder: int = 0,
    assigned_shard_indexes: Sequence[int] | None = None,
    rank_assignment_hash: str | None = None,
    summary_path: str | Path | None = None,
    w1_prepare_pipeline: str = "serial",
    max_estimated_windows_per_prepare_batch: int | None = None,
    w1_prepare_queue_depth: int = 1,
    w1_pin_memory: str = W1_PIN_MEMORY_AUTO,
    w1_tokenizers_parallelism: str = W1_TOKENIZERS_PARALLELISM_INHERIT,
    w1_persist_commit_row_threshold: int | None = None,
    accepted_token_ledger_path: str | Path | None = None,
    enforce_accepted_token_target: bool = False,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    if w1_scorer.model_id != CHUNK_WINDOW_MIL_MODEL_ID:
        raise ValueError("w1-score-sharded requires the W1 Window-MIL scorer")
    _validate_sharded_w1_handoff_payload(handoff, config_payload=config_payload, supervisor_run_dir=supervisor_run_dir)
    shard_modulus = int(shard_index_modulus)
    shard_remainder = int(shard_index_remainder)
    if shard_modulus < 1:
        raise ValueError("shard_index_modulus must be positive")
    if shard_remainder < 0 or shard_remainder >= shard_modulus:
        raise ValueError("shard_index_remainder must be in [0, shard_index_modulus)")
    all_shards = sorted(handoff["shards"], key=lambda item: int(item["shard_index"]))
    all_shard_indexes = [int(shard["shard_index"]) for shard in all_shards]
    if assigned_shard_indexes is not None:
        if shard_modulus != 1 or shard_remainder != 0:
            raise ValueError("assigned_shard_indexes cannot combine with shard_index_modulus/remainder")
        requested_indexes = [int(item) for item in assigned_shard_indexes]
        if not requested_indexes:
            raise ValueError("assigned_shard_indexes must not be empty")
        if len(set(requested_indexes)) != len(requested_indexes):
            raise ValueError("duplicate assigned_shard_indexes")
        known_indexes = set(all_shard_indexes)
        unknown = sorted(index for index in requested_indexes if index not in known_indexes)
        if unknown:
            raise ValueError(f"unknown assigned_shard_indexes: {unknown}")
        requested_index_set = set(requested_indexes)
        selected_shards = [shard for shard in all_shards if int(shard["shard_index"]) in requested_index_set]
        partition_selection_mode = "explicit_shard_indexes_v1"
        summary_shard_modulus: int | None = None
        summary_shard_remainder: int | None = None
    else:
        selected_shards = [
            shard
            for shard in all_shards
            if int(shard["shard_index"]) % shard_modulus == shard_remainder
        ]
        partition_selection_mode = "shard_index_modulo_v1"
        summary_shard_modulus = shard_modulus
        summary_shard_remainder = shard_remainder
    if not selected_shards:
        raise ValueError("W1 shard partition selected no shards")
    selected_shard_indexes = [int(shard["shard_index"]) for shard in selected_shards]
    partition_active = selected_shard_indexes != all_shard_indexes
    ledger_enabled = accepted_token_ledger_path is not None
    if bool(enforce_accepted_token_target) and partition_active and not ledger_enabled:
        raise ValueError(
            "accepted token target enforcement requires unpartitioned sharded W1 scoring "
            "until deterministic multi-rank token ledger support exists"
        )
    if ledger_enabled and not bool(enforce_accepted_token_target):
        raise ValueError("accepted-token ledger requires accepted token target enforcement")
    selected_survivor_count = sum(int(shard.get("survivor_count", 0)) for shard in selected_shards)
    validated_rank_assignment_hash = (
        _require_sha256(rank_assignment_hash, "rank_assignment_hash")
        if rank_assignment_hash is not None
        else None
    )
    supervisor_path = Path(supervisor_run_dir)
    stop_path = supervisor_path / "STOP_REQUESTED"
    requested_prepare_pipeline = _normalize_w1_prepare_pipeline(w1_prepare_pipeline)
    normalized_prepare_pipeline = requested_prepare_pipeline
    if requested_prepare_pipeline == W1_PREPARE_PIPELINE_BOUNDED_OVERLAP:
        if _w1_prepare_pipeline_disabled_reason(w1_scorer) is not None:
            normalized_prepare_pipeline = W1_PREPARE_PIPELINE_SERIAL
    normalized_pin_memory = _normalize_w1_pin_memory(w1_pin_memory)
    normalized_tokenizers_parallelism = _normalize_w1_tokenizers_parallelism(w1_tokenizers_parallelism)
    tokenizers_effective = (
        os.environ.get("TOKENIZERS_PARALLELISM")
        if normalized_tokenizers_parallelism == W1_TOKENIZERS_PARALLELISM_INHERIT
        else normalized_tokenizers_parallelism
    )
    normalized_persist_threshold = _positive_optional_int(
        w1_persist_commit_row_threshold,
        "w1_persist_commit_row_threshold",
    )
    max_estimated_window_budget = _positive_optional_int(
        max_estimated_windows_per_prepare_batch,
        "max_estimated_windows_per_prepare_batch",
    )
    prepare_queue_depth = max(1, int(w1_prepare_queue_depth))
    prepare_worker_count = 1 if normalized_prepare_pipeline == W1_PREPARE_PIPELINE_BOUNDED_OVERLAP else 0
    requested_w1_runtime_envelope = _w1_runtime_execution_envelope(
        config_payload=config_payload,
        w1_scorer=w1_scorer,
        stage_batch_size=max(1, int(batch_size)),
        w1_prepare_pipeline=normalized_prepare_pipeline,
        max_estimated_windows_per_prepare_batch=max_estimated_window_budget,
        prepare_queue_depth=prepare_queue_depth,
        prepare_worker_count=prepare_worker_count,
        w1_pin_memory=normalized_pin_memory,
        w1_tokenizers_parallelism_requested=normalized_tokenizers_parallelism,
        w1_tokenizers_parallelism_effective=tokenizers_effective,
        w1_persist_commit_row_threshold=normalized_persist_threshold,
    )
    requested_w1_runtime_envelope_hash = _w1_runtime_execution_envelope_hash(requested_w1_runtime_envelope)
    ordinal_bases: dict[int, int] = {}
    ordinal_cursor = 0
    shard_survivor_counts: list[dict[str, int]] = []
    for shard in all_shards:
        shard_index = int(shard["shard_index"])
        survivor_count = int(shard.get("survivor_count", 0))
        ordinal_bases[shard_index] = ordinal_cursor
        ordinal_cursor += survivor_count
        shard_survivor_counts.append(
            {"shard_index": shard_index, "survivor_count": survivor_count}
        )
    if ordinal_cursor != int(handoff["survivor_count"]):
        raise ValueError("sharded W1 handoff survivor ordinal count mismatch")
    accepted_token_ledger_metadata: dict[str, Any] | None = None
    if ledger_enabled:
        accepted_token_ledger_metadata = {
            "sharded_w1_handoff_hash": handoff["handoff_hash"],
            "config_hash": config_payload["config_hash"],
            "prefilter_producer_config_hash": handoff["config_hash"],
            "w1_consumer_config_hash": config_payload["config_hash"],
            "selected_policy_hash": config_payload["selected_policy_hash"],
            "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
            "source_metadata_hash": config_payload["source_metadata_hash"],
            "shard_count": int(handoff["shard_count"]),
            "survivor_count": int(handoff["survivor_count"]),
            "accepted_token_target_lower_bound": int(config_payload["accepted_token_target_lower_bound"]),
            "rank_assignment_hash": validated_rank_assignment_hash,
            "w1_runtime_envelope_hash": requested_w1_runtime_envelope_hash,
            "shard_survivor_counts_hash": stable_json_hash(shard_survivor_counts),
        }
    ledger_complete_shard_summaries: dict[int, dict[str, Any]] = {}
    if ledger_enabled:
        if accepted_token_ledger_metadata is None:
            raise ValueError("accepted-token ledger metadata is required")
        ledger_conn = _connect_w1_accepted_token_ledger(
            Path(accepted_token_ledger_path),
            metadata=accepted_token_ledger_metadata,
        )
        try:
            for shard in selected_shards:
                ledger_snapshot = _w1_homogeneous_complete_ledger_shard_state(
                    ledger_conn=ledger_conn,
                    shard=shard,
                )
                if ledger_snapshot is not None:
                    _ledger_state, ledger_shard_summary = ledger_snapshot
                    ledger_complete_shard_summaries[int(shard["shard_index"])] = ledger_shard_summary
        finally:
            ledger_conn.close()
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        handoff.get("config_hash"),
        label="sharded W1 handoff",
    )
    for shard in selected_shards:
        if int(shard["shard_index"]) in ledger_complete_shard_summaries:
            continue
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_path, shard)
        conn = _open_w1_stage_state(
            shard_dir,
            config_payload=config_payload,
            prefilter_config_payload=prefilter_config_payload,
        )
        try:
            _validate_existing_w1_runtime_envelope_state(
                conn=conn,
                run_path=shard_dir,
                requested_w1_runtime_envelope_hash=requested_w1_runtime_envelope_hash,
            )
        finally:
            conn.close()
    reports: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    accepted_token_target_lower_bound = int(config_payload["accepted_token_target_lower_bound"])
    accepted_token_count_lower_bound = 0
    accepted_token_target_reached = False
    start = time.monotonic()
    for shard in selected_shards:
        if stop_path.exists():
            break
        if (
            bool(enforce_accepted_token_target)
            and not ledger_enabled
            and accepted_token_count_lower_bound >= accepted_token_target_lower_bound
        ):
            accepted_token_target_reached = True
            break
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_path, shard)
        ledger_snapshot = (
            None
            if not ledger_enabled
            else _w1_homogeneous_complete_ledger_shard_state_from_path(
                ledger_path=accepted_token_ledger_path,
                ledger_metadata=accepted_token_ledger_metadata or {},
                shard=shard,
            )
        )
        if ledger_snapshot is not None:
            ledger_state_for_fast_skip, ledger_shard_summary = ledger_snapshot
            report = _build_w1_ledger_fast_skip_receipt(
                config_payload=config_payload,
                handoff=handoff,
                shard=shard,
                ledger_state=ledger_state_for_fast_skip,
                ledger_shard_summary=ledger_shard_summary,
                w1_runtime_envelope=requested_w1_runtime_envelope,
                w1_runtime_envelope_hash=requested_w1_runtime_envelope_hash,
                rank_assignment_hash=validated_rank_assignment_hash,
            )
        else:
            report = run_w1_score_stage(
                config_payload=config_payload,
                run_dir=shard_dir,
                w1_scorer=w1_scorer,
                batch_size=batch_size,
                expected_prefilter_status=_w1_start_prefilter_expected_status_for_run_dir(shard_dir),
                extra_stop_paths=(stop_path,),
                w1_prepare_pipeline=normalized_prepare_pipeline,
                max_estimated_windows_per_prepare_batch=max_estimated_window_budget,
                w1_prepare_queue_depth=prepare_queue_depth,
                w1_pin_memory=normalized_pin_memory,
                w1_tokenizers_parallelism=normalized_tokenizers_parallelism,
                w1_persist_commit_row_threshold=normalized_persist_threshold,
                enforce_accepted_token_target=bool(enforce_accepted_token_target),
                accepted_token_count_initial_lower_bound=(
                    0 if ledger_enabled else accepted_token_count_lower_bound
                ),
                accepted_token_ledger_path=accepted_token_ledger_path if ledger_enabled else None,
                accepted_token_ledger_metadata=accepted_token_ledger_metadata,
                accepted_token_ledger_shard_index=int(shard["shard_index"]) if ledger_enabled else None,
                accepted_token_ledger_global_ordinal_base=(
                    ordinal_bases[int(shard["shard_index"])] if ledger_enabled else None
                ),
                accepted_token_ledger_rank_assignment_hash=validated_rank_assignment_hash,
                allow_queue_exhaustion_before_token_target=True,
            )
        reports.append(report)
        route_counts.update(_w1_report_route_counts_for_sharded_rank(report, ledger_enabled=ledger_enabled))
        stage_accepted = int(
            report.get(
                "accepted_estimated_token_count_lower_bound_stage",
                report.get("accepted_estimated_token_count_lower_bound", 0),
            )
            or 0
        )
        if ledger_enabled:
            accepted_token_count_lower_bound = int(report.get("accepted_estimated_token_count_lower_bound", 0) or 0)
        else:
            accepted_token_count_lower_bound += stage_accepted
        accepted_token_target_reached = bool(
            enforce_accepted_token_target
            and (
                accepted_token_count_lower_bound >= accepted_token_target_lower_bound
                or report.get("accepted_token_target_reached")
            )
        )
        if accepted_token_target_reached and not ledger_enabled:
            break
        if not _w1_report_complete_for_sharded_rank(report, ledger_enabled=ledger_enabled):
            break
    scored_count = sum(
        _w1_report_ledger_shard_scored_count(report) if ledger_enabled else int(report.get("scored_count", 0))
        for report in reports
    )
    selected_complete = len(reports) == len(selected_shards) and all(
        _w1_report_complete_for_sharded_rank(report, ledger_enabled=ledger_enabled) for report in reports
    )
    queue_exhausted_before_token_target = bool(
        enforce_accepted_token_target
        and selected_complete
        and accepted_token_count_lower_bound < accepted_token_target_lower_bound
    )
    complete = not partition_active and selected_complete
    if complete and scored_count != int(handoff["survivor_count"]):
        raise ValueError("sharded W1 score summary survivor_count mismatch")
    if selected_complete and scored_count != int(selected_survivor_count):
        raise ValueError("sharded W1 score summary selected survivor_count mismatch")
    status = "w1_sharded_scored" if complete else "w1_sharded_partial"
    stop_reason = "w1_sharded_complete" if complete else "stop_requested_or_pending_shards"
    next_required_phase = "assemble-sharded-report" if complete else "resume_w1_score_sharded"
    if accepted_token_target_reached:
        status = "accepted_token_target_reached_partial"
        stop_reason = "accepted_token_target_reached"
        next_required_phase = "archive_w1_private_score_state_before_gpu_teardown"
    elif partition_active and selected_complete:
        status = "w1_sharded_partition_scored"
        stop_reason = "w1_shard_partition_complete"
        next_required_phase = "resume_w1_score_sharded_or_aggregate_partitions"
    elif queue_exhausted_before_token_target and not partition_active:
        status = "queue_exhausted_before_token_target"
        stop_reason = "w1_survivor_queue_exhausted_before_accepted_token_target"
        next_required_phase = "coordinator_review_token_shortfall_before_downstream_unlock"
    w1_runtime_envelope_hashes = sorted(
        {
            _require_sha256(report.get("w1_runtime_envelope_hash"), "w1_runtime_envelope_hash")
            for report in reports
            if report.get("w1_runtime_envelope_hash") is not None
        }
    )
    if reports and len(w1_runtime_envelope_hashes) != 1:
        raise ValueError("W1 runtime envelope mismatch across sharded W1 receipts")
    w1_runtime_envelope = (
        dict(reports[0]["w1_runtime_envelope"])
        if reports and isinstance(reports[0].get("w1_runtime_envelope"), Mapping)
        else None
    )
    payload = {
        "artifact_kind": SHARDED_W1_SCORE_SUMMARY_KIND,
        "schema_version": 1,
        "status": status,
        "stop_reason": stop_reason,
        "next_required_phase": next_required_phase,
        "config_hash": config_payload["config_hash"],
        "prefilter_producer_config_hash": handoff["config_hash"],
        "w1_consumer_config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "sharded_w1_handoff_hash": handoff["handoff_hash"],
        "shard_count": int(handoff["shard_count"]),
        "selected_shard_count": len(selected_shards),
        "selected_survivor_count": int(selected_survivor_count),
        "partition_selection_mode": partition_selection_mode,
        "partition_shard_indexes": selected_shard_indexes,
        "shard_index_modulus": summary_shard_modulus,
        "shard_index_remainder": summary_shard_remainder,
        "rank_assignment_hash": validated_rank_assignment_hash,
        "processed_shard_count": len(reports),
        "survivor_count": int(handoff["survivor_count"]),
        "scored_count": int(scored_count),
        "accepted_token_target_enforced": bool(enforce_accepted_token_target),
        "accepted_token_target_lower_bound": accepted_token_target_lower_bound,
        "accepted_estimated_token_count_lower_bound": int(accepted_token_count_lower_bound),
        "accepted_token_target_reached": bool(accepted_token_target_reached),
        "accepted_token_ledger_enabled": bool(ledger_enabled),
        "accepted_token_ledger_path_hash": (
            stable_json_hash(str(Path(accepted_token_ledger_path).expanduser().resolve(strict=False)))
            if ledger_enabled
            else None
        ),
        "route_counts": _sorted_counter(route_counts),
        "w1_score_receipt_hashes": [str(report["stage_receipt_hash"]) for report in reports],
        "w1_runtime_envelope": w1_runtime_envelope,
        "w1_runtime_envelope_hashes": w1_runtime_envelope_hashes,
        "w1_runtime_envelope_hash": w1_runtime_envelope_hashes[0] if len(w1_runtime_envelope_hashes) == 1 else None,
        "w1_runtime_telemetry": _aggregate_w1_runtime_telemetry(reports),
        "elapsed_seconds": round(time.monotonic() - start, 6),
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["summary_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "summary_hash"})
    validate_public_no_raw_payload(payload)
    if summary_path is not None:
        _write_json_atomic(summary_path, payload)
    return payload


def run_w1_score_partial_sharded_diagnostic(
    *,
    config_payload: Mapping[str, Any],
    handoff: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    w1_scorer: ChunkFilteringScorer,
    batch_size: int = 64,
    summary_path: str | Path | None = None,
    w1_prepare_pipeline: str = "serial",
    max_estimated_windows_per_prepare_batch: int | None = None,
    w1_prepare_queue_depth: int = 1,
    w1_pin_memory: str = W1_PIN_MEMORY_AUTO,
    w1_tokenizers_parallelism: str = W1_TOKENIZERS_PARALLELISM_INHERIT,
    w1_persist_commit_row_threshold: int | None = None,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    if w1_scorer.model_id != CHUNK_WINDOW_MIL_MODEL_ID:
        raise ValueError("w1-score-partial-sharded-diagnostic requires the W1 Window-MIL scorer")
    _validate_partial_sharded_w1_diagnostic_handoff_payload(
        handoff,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
    )
    _validate_partial_sharded_w1_diagnostic_live_shard_state(
        handoff,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
    )
    supervisor_path = Path(supervisor_run_dir)
    stop_path = supervisor_path / "STOP_REQUESTED"
    requested_prepare_pipeline = _normalize_w1_prepare_pipeline(w1_prepare_pipeline)
    normalized_prepare_pipeline = requested_prepare_pipeline
    if requested_prepare_pipeline == W1_PREPARE_PIPELINE_BOUNDED_OVERLAP:
        if _w1_prepare_pipeline_disabled_reason(w1_scorer) is not None:
            normalized_prepare_pipeline = W1_PREPARE_PIPELINE_SERIAL
    normalized_pin_memory = _normalize_w1_pin_memory(w1_pin_memory)
    normalized_tokenizers_parallelism = _normalize_w1_tokenizers_parallelism(w1_tokenizers_parallelism)
    tokenizers_effective = (
        os.environ.get("TOKENIZERS_PARALLELISM")
        if normalized_tokenizers_parallelism == W1_TOKENIZERS_PARALLELISM_INHERIT
        else normalized_tokenizers_parallelism
    )
    normalized_persist_threshold = _positive_optional_int(
        w1_persist_commit_row_threshold,
        "w1_persist_commit_row_threshold",
    )
    max_estimated_window_budget = _positive_optional_int(
        max_estimated_windows_per_prepare_batch,
        "max_estimated_windows_per_prepare_batch",
    )
    prepare_queue_depth = max(1, int(w1_prepare_queue_depth))
    prepare_worker_count = 1 if normalized_prepare_pipeline == W1_PREPARE_PIPELINE_BOUNDED_OVERLAP else 0
    requested_w1_runtime_envelope = _w1_runtime_execution_envelope(
        config_payload=config_payload,
        w1_scorer=w1_scorer,
        stage_batch_size=max(1, int(batch_size)),
        w1_prepare_pipeline=normalized_prepare_pipeline,
        max_estimated_windows_per_prepare_batch=max_estimated_window_budget,
        prepare_queue_depth=prepare_queue_depth,
        prepare_worker_count=prepare_worker_count,
        w1_pin_memory=normalized_pin_memory,
        w1_tokenizers_parallelism_requested=normalized_tokenizers_parallelism,
        w1_tokenizers_parallelism_effective=tokenizers_effective,
        w1_persist_commit_row_threshold=normalized_persist_threshold,
    )
    requested_w1_runtime_envelope_hash = _w1_runtime_execution_envelope_hash(requested_w1_runtime_envelope)
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        handoff.get("config_hash"),
        label="partial W1 diagnostic handoff",
    )
    for shard in sorted(handoff["shards"], key=lambda item: int(item["shard_index"])):
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_path, shard)
        conn = _open_w1_stage_state(
            shard_dir,
            config_payload=config_payload,
            prefilter_config_payload=prefilter_config_payload,
        )
        try:
            _validate_existing_w1_runtime_envelope_state(
                conn=conn,
                run_path=shard_dir,
                requested_w1_runtime_envelope_hash=requested_w1_runtime_envelope_hash,
            )
        finally:
            conn.close()
    reports: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    start = time.monotonic()
    for shard in sorted(handoff["shards"], key=lambda item: int(item["shard_index"])):
        if stop_path.exists():
            break
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_path, shard)
        report = run_w1_score_stage(
            config_payload=config_payload,
            run_dir=shard_dir,
            w1_scorer=w1_scorer,
            batch_size=batch_size,
            expected_prefilter_status="prefilter_shard_ready_for_merge",
            extra_stop_paths=(stop_path,),
            w1_prepare_pipeline=normalized_prepare_pipeline,
            max_estimated_windows_per_prepare_batch=max_estimated_window_budget,
            w1_prepare_queue_depth=prepare_queue_depth,
            w1_pin_memory=normalized_pin_memory,
            w1_tokenizers_parallelism=normalized_tokenizers_parallelism,
            w1_persist_commit_row_threshold=normalized_persist_threshold,
        )
        reports.append(report)
        route_counts.update({str(key): int(value) for key, value in dict(report.get("route_counts") or {}).items()})
        if report["status"] != "w1_scored":
            break
    scored_count = sum(int(report.get("scored_count", 0)) for report in reports)
    complete = len(reports) == int(handoff["selected_shard_count"]) and all(
        report.get("status") == "w1_scored" for report in reports
    )
    if complete and scored_count != int(handoff["survivor_count"]):
        raise ValueError("partial W1 diagnostic score summary survivor_count mismatch")
    w1_runtime_envelope_hashes = sorted(
        {
            _require_sha256(report.get("w1_runtime_envelope_hash"), "w1_runtime_envelope_hash")
            for report in reports
            if report.get("w1_runtime_envelope_hash") is not None
        }
    )
    if reports and len(w1_runtime_envelope_hashes) != 1:
        raise ValueError("W1 runtime envelope mismatch across partial W1 diagnostic receipts")
    w1_runtime_envelope = (
        dict(reports[0]["w1_runtime_envelope"])
        if reports and isinstance(reports[0].get("w1_runtime_envelope"), Mapping)
        else None
    )
    payload = {
        "artifact_kind": PARTIAL_SHARDED_W1_DIAGNOSTIC_SUMMARY_KIND,
        "schema_version": 1,
        "status": "w1_partial_shard_diagnostic_scored" if complete else "w1_partial_shard_diagnostic_partial",
        "stop_reason": "selected_shards_scored" if complete else "stop_requested_or_pending_selected_shards",
        "claims_scope": "throughput_diagnostic_only",
        "canonical_full_handoff_satisfied": False,
        "next_required_phase": "diagnostic_review_only" if complete else "resume_w1_score_partial_sharded_diagnostic",
        "config_hash": config_payload["config_hash"],
        "prefilter_producer_config_hash": handoff["config_hash"],
        "w1_consumer_config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "partial_sharded_w1_diagnostic_handoff_hash": handoff["diagnostic_handoff_hash"],
        "supervisor_manifest_hash": handoff["supervisor_manifest_hash"],
        "full_shard_count": int(handoff["full_shard_count"]),
        "selected_shard_count": int(handoff["selected_shard_count"]),
        "selected_shard_indexes": [int(index) for index in handoff["selected_shard_indexes"]],
        "processed_shard_count": len(reports),
        "selected_survivor_count": int(handoff["survivor_count"]),
        "scored_count": int(scored_count),
        "route_counts": _sorted_counter(route_counts),
        "w1_score_receipt_hashes": [str(report["stage_receipt_hash"]) for report in reports],
        "w1_runtime_envelope": w1_runtime_envelope,
        "w1_runtime_envelope_hashes": w1_runtime_envelope_hashes,
        "w1_runtime_envelope_hash": w1_runtime_envelope_hashes[0] if len(w1_runtime_envelope_hashes) == 1 else None,
        "w1_runtime_telemetry": _aggregate_w1_runtime_telemetry(reports),
        "elapsed_seconds": round(time.monotonic() - start, 6),
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["summary_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "summary_hash"})
    validate_public_no_raw_payload(payload)
    if summary_path is not None:
        _write_json_atomic(summary_path, payload)
    return payload


def _aggregate_w1_runtime_telemetry(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    telemetry_rows = [
        dict(report.get("w1_runtime_telemetry") or {})
        for report in reports
        if isinstance(report.get("w1_runtime_telemetry"), Mapping)
    ]
    totals = {
        "shard_report_count": len(reports),
        "telemetry_report_count": len(telemetry_rows),
        "survivor_queue_rows": 0,
        "scored_existing_initial": 0,
        "pending_initial": 0,
        "scorer_batch_count": 0,
        "scorer_input_docs": 0,
        "score_model_call_count": 0,
        "score_oom_split_count": 0,
        "score_oom_failed_attempt_count": 0,
        "score_oom_failed_attempt_docs": 0,
        "score_oom_failed_attempt_estimated_windows": 0,
        "score_oom_failed_attempt_max_docs": 0,
        "score_oom_failed_attempt_max_estimated_windows": 0,
        "score_max_effective_batch_docs": 0,
        "score_prepared_batch_count": 0,
        "score_prepared_oom_split_count": 0,
        "score_prepare_submit_count": 0,
        "score_prepare_completed_count": 0,
        "score_prepare_failed_count": 0,
        "score_window_budget_batch_count": 0,
        "score_window_budget_oversized_single_doc_batch_count": 0,
        "score_window_budget_max_estimated_windows_per_batch": 0,
        "score_pin_memory_requested_count": 0,
        "score_pin_memory_effective_count": 0,
        "score_rows_attempted": 0,
        "score_rows_inserted": 0,
        "accepted_token_ledger_rows_for_shard_initial": 0,
        "accepted_token_ledger_rows_for_shard": 0,
        "accepted_token_ledger_accepted_token_count_lower_bound_for_shard": 0,
        "accepted_token_ledger_fast_skipped_shards": 0,
        "accepted_token_ledger_fast_skipped_rows": 0,
        "score_persist_commit_count": 0,
        "score_persist_rows_per_commit": [],
        "score_persist_max_rows_per_commit": 0,
        "score_persist_pending_rows": 0,
        "score_persist_database_path_hash": None,
        "score_persist_sqlite_pragmas": {},
        "score_persist_strategy": W1_PERSIST_STRATEGY_PER_BATCH,
        "score_persist_commit_row_threshold": None,
        "w1_tokenizers_parallelism_requested": W1_TOKENIZERS_PARALLELISM_INHERIT,
        "w1_tokenizers_parallelism_effective": None,
        "score_stage_wall_seconds": 0.0,
        "score_input_fetch_seconds": 0.0,
        "score_model_seconds": 0.0,
        "score_persist_seconds": 0.0,
        "score_prepare_wait_seconds": 0.0,
        "score_cuda_synchronized_timing": False,
        "score_cuda_synchronized_timing_requested": False,
        "score_cuda_synchronized_timing_observed": False,
        "score_cuda_synchronized_timing_requested_values": [],
        "score_cuda_synchronized_timing_observed_values": [],
        "score_tokenizer_call_count": 0,
        "score_tokenized_window_count": 0,
        "score_tokenized_input_cell_count": 0,
        "score_tokenized_attention_token_count": 0,
        "score_tokenized_padding_cell_count": 0,
        "score_tokenization_seconds": 0.0,
        "score_device_transfer_seconds": 0.0,
        "score_window_forward_call_count": 0,
        "score_window_forward_window_count": 0,
        "score_window_forward_seconds": 0.0,
        "score_pooling_seconds": 0.0,
        "score_postprocess_seconds": 0.0,
        "score_pin_memory_seconds": 0.0,
        "score_cuda_synchronization_count": 0,
        "score_pooling_ragged_topk_call_count": 0,
        "score_pooling_ragged_topk_iteration_count": 0,
        "score_pooling_ragged_topk_input_window_count": 0,
        "score_pooling_ragged_topk_document_count": 0,
        "score_pooling_dense_padded_cell_count_avoided": 0,
        "score_max_windows_per_forward_call": 0,
        "score_max_tokenized_windows_per_document_batch": 0,
        "score_max_tokenized_sequence_length": 0,
        "score_max_windows_per_document": 0,
        "estimated_w1_window_count": 0,
        "stage_batch_size_max": 0,
        "scorer_device_observations": [],
        "scorer_device_types": [],
        "cuda_device_observed": False,
    }
    device_observations: list[dict[str, Any]] = []
    sync_requested_values: set[bool] = set()
    sync_observed_values: set[bool] = set()
    persist_rows_per_commit: list[int] = []
    persist_strategy_values: set[str] = set()
    tokenizers_requested_values: set[str] = set()
    tokenizers_effective_values: set[str] = set()
    for row in telemetry_rows:
        totals["stage_batch_size_max"] = max(int(totals["stage_batch_size_max"]), int(row.get("stage_batch_size", 0)))
        legacy_sync = bool(row.get("score_cuda_synchronized_timing", False))
        sync_requested_values.add(bool(row.get("score_cuda_synchronized_timing_requested", legacy_sync)))
        sync_observed_values.add(bool(row.get("score_cuda_synchronized_timing_observed", legacy_sync)))
        scorer_device = row.get("scorer_device")
        if isinstance(scorer_device, Mapping):
            observation = dict(scorer_device)
            if observation not in device_observations:
                device_observations.append(observation)
            if observation.get("device_type") == "cuda":
                totals["cuda_device_observed"] = True
        for key in (
            "survivor_queue_rows",
            "scored_existing_initial",
            "pending_initial",
            "scorer_batch_count",
            "scorer_input_docs",
            "score_model_call_count",
            "score_oom_split_count",
            "score_oom_failed_attempt_count",
            "score_oom_failed_attempt_docs",
            "score_oom_failed_attempt_estimated_windows",
            "score_prepared_batch_count",
            "score_prepared_oom_split_count",
            "score_prepare_submit_count",
            "score_prepare_completed_count",
            "score_prepare_failed_count",
            "score_window_budget_batch_count",
            "score_window_budget_oversized_single_doc_batch_count",
            "score_pin_memory_requested_count",
            "score_pin_memory_effective_count",
            "score_rows_attempted",
            "score_rows_inserted",
            "accepted_token_ledger_rows_for_shard_initial",
            "accepted_token_ledger_rows_for_shard",
            "accepted_token_ledger_accepted_token_count_lower_bound_for_shard",
            "accepted_token_ledger_fast_skipped_shards",
            "accepted_token_ledger_fast_skipped_rows",
            "score_persist_commit_count",
            "score_tokenizer_call_count",
            "score_tokenized_window_count",
            "score_tokenized_input_cell_count",
            "score_tokenized_attention_token_count",
            "score_tokenized_padding_cell_count",
            "score_window_forward_call_count",
            "score_window_forward_window_count",
            "score_cuda_synchronization_count",
            "score_pooling_ragged_topk_call_count",
            "score_pooling_ragged_topk_iteration_count",
            "score_pooling_ragged_topk_input_window_count",
            "score_pooling_ragged_topk_document_count",
            "score_pooling_dense_padded_cell_count_avoided",
            "estimated_w1_window_count",
        ):
            totals[key] = int(totals[key]) + int(row.get(key, 0))
        totals["score_max_effective_batch_docs"] = max(
            int(totals["score_max_effective_batch_docs"]),
            int(row.get("score_max_effective_batch_docs", 0)),
        )
        totals["score_oom_failed_attempt_max_docs"] = max(
            int(totals["score_oom_failed_attempt_max_docs"]),
            int(row.get("score_oom_failed_attempt_max_docs", 0)),
        )
        totals["score_oom_failed_attempt_max_estimated_windows"] = max(
            int(totals["score_oom_failed_attempt_max_estimated_windows"]),
            int(row.get("score_oom_failed_attempt_max_estimated_windows", 0)),
        )
        totals["score_max_windows_per_forward_call"] = max(
            int(totals["score_max_windows_per_forward_call"]),
            int(row.get("score_max_windows_per_forward_call", 0)),
        )
        totals["score_max_tokenized_windows_per_document_batch"] = max(
            int(totals["score_max_tokenized_windows_per_document_batch"]),
            int(row.get("score_max_tokenized_windows_per_document_batch", 0)),
        )
        totals["score_max_tokenized_sequence_length"] = max(
            int(totals["score_max_tokenized_sequence_length"]),
            int(row.get("score_max_tokenized_sequence_length", 0)),
        )
        totals["score_window_budget_max_estimated_windows_per_batch"] = max(
            int(totals["score_window_budget_max_estimated_windows_per_batch"]),
            int(row.get("score_window_budget_max_estimated_windows_per_batch", 0)),
        )
        totals["score_max_windows_per_document"] = max(
            int(totals["score_max_windows_per_document"]),
            int(row.get("score_max_windows_per_document", 0)),
        )
        totals["score_persist_max_rows_per_commit"] = max(
            int(totals["score_persist_max_rows_per_commit"]),
            int(row.get("score_persist_max_rows_per_commit", 0)),
        )
        totals["score_persist_pending_rows"] = max(
            int(totals["score_persist_pending_rows"]),
            int(row.get("score_persist_pending_rows", 0)),
        )
        persist_rows_per_commit.extend(int(value) for value in list(row.get("score_persist_rows_per_commit") or []))
        if row.get("score_persist_strategy") is not None:
            persist_strategy_values.add(str(row.get("score_persist_strategy")))
        if row.get("w1_tokenizers_parallelism_requested") is not None:
            tokenizers_requested_values.add(str(row.get("w1_tokenizers_parallelism_requested")))
        if row.get("w1_tokenizers_parallelism_effective") is not None:
            tokenizers_effective_values.add(str(row.get("w1_tokenizers_parallelism_effective")))
        for key in (
            "score_input_fetch_seconds",
            "score_stage_wall_seconds",
            "score_model_seconds",
            "score_persist_seconds",
            "score_prepare_wait_seconds",
            "score_tokenization_seconds",
            "score_device_transfer_seconds",
            "score_window_forward_seconds",
            "score_pooling_seconds",
            "score_postprocess_seconds",
            "score_pin_memory_seconds",
        ):
            totals[key] = float(totals[key]) + float(row.get(key, 0.0))
    sorted_sync_requested_values = sorted(sync_requested_values)
    sorted_sync_observed_values = sorted(sync_observed_values)
    totals["score_cuda_synchronized_timing_requested_values"] = sorted_sync_requested_values
    totals["score_cuda_synchronized_timing_observed_values"] = sorted_sync_observed_values
    totals["score_cuda_synchronized_timing_requested"] = sorted_sync_requested_values == [True]
    totals["score_cuda_synchronized_timing_observed"] = sorted_sync_observed_values == [True]
    totals["score_cuda_synchronized_timing"] = bool(totals["score_cuda_synchronized_timing_observed"])
    for key in (
        "score_input_fetch_seconds",
        "score_stage_wall_seconds",
        "score_model_seconds",
        "score_persist_seconds",
        "score_prepare_wait_seconds",
        "score_tokenization_seconds",
        "score_device_transfer_seconds",
        "score_window_forward_seconds",
        "score_pooling_seconds",
        "score_postprocess_seconds",
        "score_pin_memory_seconds",
    ):
        totals[key] = round(float(totals[key]), 6)
    totals["scorer_device_observations"] = device_observations
    totals["scorer_device_types"] = sorted(
        {str(item["device_type"]) for item in device_observations if item.get("device_type") is not None}
    )
    totals["score_persist_rows_per_commit"] = persist_rows_per_commit
    totals["score_persist_strategy_values"] = sorted(persist_strategy_values)
    totals["w1_tokenizers_parallelism_requested_values"] = sorted(tokenizers_requested_values)
    totals["w1_tokenizers_parallelism_effective_values"] = sorted(tokenizers_effective_values)
    _add_w1_runtime_derived_metrics(totals)
    validate_public_no_raw_payload(totals)
    return totals


def _w1_rate(numerator: int | float, denominator: int | float) -> float:
    denominator_float = float(denominator)
    if denominator_float <= 0.0:
        return 0.0
    return round(float(numerator) / denominator_float, 6)


def _add_w1_runtime_derived_metrics(payload: dict[str, Any]) -> None:
    scorer_input_docs = int(payload.get("scorer_input_docs", 0))
    model_call_count = int(payload.get("score_model_call_count", 0))
    tokenized_window_count = int(payload.get("score_tokenized_window_count", 0))
    tokenized_input_cell_count = int(payload.get("score_tokenized_input_cell_count", 0))
    tokenized_attention_token_count = int(payload.get("score_tokenized_attention_token_count", 0))
    tokenized_padding_cell_count = int(payload.get("score_tokenized_padding_cell_count", 0))
    stage_wall_seconds = round(float(payload.get("score_stage_wall_seconds", 0.0)), 6)
    input_fetch_seconds = round(float(payload.get("score_input_fetch_seconds", 0.0)), 6)
    tokenization_seconds = float(payload.get("score_tokenization_seconds", 0.0))
    device_transfer_seconds = float(payload.get("score_device_transfer_seconds", 0.0))
    forward_call_count = int(payload.get("score_window_forward_call_count", 0))
    forward_window_count = int(payload.get("score_window_forward_window_count", 0))
    forward_seconds = float(payload.get("score_window_forward_seconds", 0.0))
    pooling_seconds = float(payload.get("score_pooling_seconds", 0.0))
    postprocess_seconds = float(payload.get("score_postprocess_seconds", 0.0))
    model_seconds = float(payload.get("score_model_seconds", 0.0))
    persist_seconds = round(float(payload.get("score_persist_seconds", 0.0)), 6)
    prepare_seconds = round(tokenization_seconds, 6)
    h2d_seconds = round(device_transfer_seconds, 6)
    gpu_forward_seconds = round(forward_seconds, 6)
    pool_seconds = round(pooling_seconds, 6)
    postprocess_seconds = round(postprocess_seconds, 6)
    model_inner_accounted_seconds = round(
        prepare_seconds + h2d_seconds + gpu_forward_seconds + pool_seconds + postprocess_seconds,
        6,
    )
    phase_accounted_seconds = round(
        input_fetch_seconds + model_inner_accounted_seconds + persist_seconds,
        6,
    )
    phase_seconds = {
        "input_fetch": input_fetch_seconds,
        "prepare_tokenize": prepare_seconds,
        "h2d": h2d_seconds,
        "forward": gpu_forward_seconds,
        "pool": pool_seconds,
        "postprocess": postprocess_seconds,
        "write": persist_seconds,
        "other": round(max(0.0, stage_wall_seconds - phase_accounted_seconds), 6),
    }
    bottleneck_candidates = {key: value for key, value in phase_seconds.items() if key != "other"}
    payload["score_stage_wall_seconds"] = stage_wall_seconds
    payload["score_prepare_seconds"] = prepare_seconds
    payload["score_h2d_seconds"] = h2d_seconds
    payload["score_gpu_forward_seconds"] = gpu_forward_seconds
    payload["score_pool_seconds"] = pool_seconds
    payload["score_write_seconds"] = persist_seconds
    payload["score_input_wait_seconds"] = input_fetch_seconds
    payload["score_model_inner_accounted_seconds"] = model_inner_accounted_seconds
    payload["score_phase_accounted_seconds"] = phase_accounted_seconds
    payload["score_phase_seconds"] = phase_seconds
    payload["score_bottleneck_phase"] = (
        max(bottleneck_candidates, key=lambda key: float(bottleneck_candidates[key]))
        if bottleneck_candidates
        else None
    )
    payload["score_gpu_idle_estimate_scope"] = "serial_non_forward_wall_upper_bound"
    payload["score_gpu_idle_estimate_seconds"] = round(max(0.0, stage_wall_seconds - gpu_forward_seconds), 6)
    payload["score_model_docs_per_second"] = _w1_rate(scorer_input_docs, model_seconds)
    payload["score_wall_docs_per_second"] = _w1_rate(scorer_input_docs, stage_wall_seconds)
    payload["score_wall_windows_per_second"] = _w1_rate(tokenized_window_count, stage_wall_seconds)
    payload["score_wall_live_tokens_per_second"] = _w1_rate(tokenized_attention_token_count, stage_wall_seconds)
    payload["score_wall_padded_tokens_per_second"] = _w1_rate(tokenized_input_cell_count, stage_wall_seconds)
    payload["score_tokenized_windows_per_doc"] = _w1_rate(tokenized_window_count, scorer_input_docs)
    payload["score_tokenized_padding_fraction"] = _w1_rate(tokenized_padding_cell_count, tokenized_input_cell_count)
    payload["score_padded_to_live_token_ratio"] = _w1_rate(
        tokenized_input_cell_count,
        tokenized_attention_token_count,
    )
    payload["score_tokenized_windows_per_tokenization_second"] = _w1_rate(
        tokenized_window_count,
        tokenization_seconds,
    )
    payload["score_forward_calls_per_model_call"] = _w1_rate(forward_call_count, model_call_count)
    payload["score_windows_per_forward_call"] = _w1_rate(forward_window_count, forward_call_count)
    payload["score_window_forward_windows_per_second"] = _w1_rate(forward_window_count, forward_seconds)


def _w1_runtime_phase_evidence_present(telemetry: Mapping[str, Any]) -> bool:
    phase_seconds = telemetry.get("score_phase_seconds")
    if not isinstance(phase_seconds, Mapping):
        return False
    for phase in ("input_fetch", "prepare_tokenize", "h2d", "forward", "pool", "postprocess", "write", "other"):
        try:
            if float(phase_seconds.get(phase)) < 0.0:
                return False
        except (TypeError, ValueError):
            return False
    try:
        return (
            float(telemetry.get("score_stage_wall_seconds", 0.0) or 0.0) > 0.0
            and float(telemetry.get("score_wall_docs_per_second", 0.0) or 0.0) > 0.0
            and float(telemetry.get("score_wall_live_tokens_per_second", 0.0) or 0.0) > 0.0
            and float(telemetry.get("score_wall_padded_tokens_per_second", 0.0) or 0.0) > 0.0
            and str(telemetry.get("score_gpu_idle_estimate_scope") or "") == "serial_non_forward_wall_upper_bound"
        )
    except (TypeError, ValueError):
        return False


def _summary_for_nonnegative_ints(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "mean": 0.0}
    ordered = sorted(_nonnegative_int(value, "summary value") for value in values)

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, max(0, math.ceil((len(ordered) * fraction) - 1)))
        return int(ordered[index])

    return {
        "count": len(ordered),
        "min": int(ordered[0]),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": int(ordered[-1]),
        "mean": round(float(sum(ordered)) / float(len(ordered)), 6),
    }


def _w1_window_count_bucket(value: int) -> str:
    parsed = max(1, int(value))
    if parsed <= 1:
        return "1"
    if parsed <= 2:
        return "2"
    if parsed <= 3:
        return "3"
    if parsed <= 7:
        return "4-7"
    if parsed <= 15:
        return "8-15"
    return "16+"


def build_w1_survivor_length_distribution_report(
    *,
    config_payload: Mapping[str, Any],
    run_dir: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    queue_rows = _load_ready_jsonl(run_path / "w1_survivor_queue.jsonl", "prefilter survivor queue")
    _validate_survivor_queue_rows(queue_rows, config_payload, allow_empty=True)
    evidence_counts = [
        _nonnegative_int(
            row.get("evidence_proxy_token_count", row.get("core_proxy_token_count", 0)),
            "survivor queue evidence_proxy_token_count",
        )
        for row in queue_rows
    ]
    estimated_windows = [
        max(1, _nonnegative_int(row.get("estimated_w1_window_count", 1), "survivor queue estimated_w1_window_count"))
        for row in queue_rows
    ]
    buckets = Counter(_w1_window_count_bucket(value) for value in estimated_windows)
    queue_ready = run_path / "w1_survivor_queue.jsonl.ready"
    payload = {
        "artifact_kind": W1_SURVIVOR_LENGTH_DISTRIBUTION_KIND,
        "schema_version": 1,
        "status": "ready",
        "claim_scope": "survivor_queue_no_raw_length_distribution",
        "coverage_scope": "single_run_survivor_queue",
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "window_mil_report_hash": config_payload["window_mil_report_hash"],
        "window_mil_model_file_sha256": config_payload["window_mil_model_file_sha256"],
        "survivor_count": len(queue_rows),
        "survivor_queue_hash": _sha256_file(queue_ready),
        "evidence_proxy_token_count_summary": _summary_for_nonnegative_ints(evidence_counts),
        "estimated_w1_window_count_summary": _summary_for_nonnegative_ints(estimated_windows),
        "estimated_w1_window_count_buckets": dict(sorted(buckets.items())),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["length_distribution_hash"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "length_distribution_hash"}
    )
    validate_w1_survivor_length_distribution_report(payload, config_payload=config_payload)
    if report_path is not None:
        _write_json_atomic(report_path, payload)
    return payload


def validate_w1_survivor_length_distribution_report(
    report: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
) -> None:
    if report.get("artifact_kind") != W1_SURVIVOR_LENGTH_DISTRIBUTION_KIND:
        raise ValueError("unsupported W1 survivor length distribution kind")
    validate_public_no_raw_payload(report)
    expected_hash = stable_json_hash(
        {key: value for key, value in dict(report).items() if key != "length_distribution_hash"}
    )
    if report.get("length_distribution_hash") != expected_hash:
        raise ValueError("W1 survivor length distribution hash mismatch")
    for field in ("config_hash", "selected_policy_hash", "selected_policy_report_hash", "source_metadata_hash"):
        if report.get(field) != config_payload.get(field):
            raise ValueError(f"length distribution {field} mismatch")
    _require_sha256(report.get("survivor_queue_hash"), "length distribution survivor_queue_hash")
    if report.get("coverage_scope") != "single_run_survivor_queue":
        raise ValueError("unsupported length distribution coverage_scope")
    buckets = report.get("estimated_w1_window_count_buckets")
    if not isinstance(buckets, Mapping):
        raise ValueError("length distribution buckets must be an object")
    bucket_total = sum(int(value) for value in buckets.values())
    if bucket_total != int(report.get("survivor_count", -1)):
        raise ValueError("length distribution bucket count mismatch")


def _validate_w1_microbench_report(report: Mapping[str, Any]) -> str:
    if report.get("artifact_kind") != "twc_window_mil_local_microbench_v1":
        raise ValueError("unsupported W1 scorer microbench kind")
    validate_public_no_raw_payload(report)
    expected_hash = stable_json_hash({key: value for key, value in dict(report).items() if key != "microbench_hash"})
    microbench_hash = _require_sha256(report.get("microbench_hash"), "microbench_hash")
    if microbench_hash != expected_hash:
        raise ValueError("W1 scorer microbench hash mismatch")
    if int(report.get("schema_version", 0)) != 1:
        raise ValueError("W1 scorer microbench schema_version mismatch")
    if report.get("status") != "complete":
        raise ValueError("W1 scorer microbench status mismatch")
    if report.get("claim_scope") != "synthetic_local_microbench_only":
        raise ValueError("W1 scorer microbench claim_scope mismatch")
    document_count = _positive_int(report.get("document_count"), "W1 scorer microbench document_count")
    _positive_int(report.get("max_length"), "W1 scorer microbench max_length")
    _nonnegative_int(report.get("stride"), "W1 scorer microbench stride")
    _positive_int(
        report.get("max_windows_per_device_batch"),
        "W1 scorer microbench max_windows_per_device_batch",
    )
    if report.get("pooling_mode") != "topk_mean_logits":
        raise ValueError("W1 scorer microbench pooling_mode mismatch")
    pooling_top_k = _positive_int(report.get("pooling_top_k"), "W1 scorer microbench pooling_top_k")
    tokenized_window_count = _positive_int(
        report.get("tokenized_window_count"),
        "W1 scorer microbench tokenized_window_count",
    )
    if tokenized_window_count < document_count:
        raise ValueError("W1 scorer microbench tokenized_window_count mismatch")
    max_windows_per_document = _positive_int(
        report.get("max_windows_per_document"),
        "W1 scorer microbench max_windows_per_document",
    )
    if max_windows_per_document > tokenized_window_count:
        raise ValueError("W1 scorer microbench max_windows_per_document mismatch")
    tokenized_input_cell_count = _positive_int(
        report.get("tokenized_input_cell_count"),
        "W1 scorer microbench tokenized_input_cell_count",
    )
    tokenized_attention_token_count = _positive_int(
        report.get("tokenized_attention_token_count"),
        "W1 scorer microbench tokenized_attention_token_count",
    )
    tokenized_padding_cell_count = _nonnegative_int(
        report.get("tokenized_padding_cell_count"),
        "W1 scorer microbench tokenized_padding_cell_count",
    )
    if tokenized_attention_token_count + tokenized_padding_cell_count != tokenized_input_cell_count:
        raise ValueError("W1 scorer microbench tokenized cell accounting mismatch")
    _require_sha256(report.get("synthetic_profile_hash"), "W1 scorer microbench synthetic_profile_hash")
    _positive_int(report.get("window_forward_call_count"), "W1 scorer microbench window_forward_call_count")
    _positive_int(report.get("max_windows_per_forward_call"), "W1 scorer microbench max_windows_per_forward_call")
    if _positive_int(report.get("score_count"), "W1 scorer microbench score_count") != document_count:
        raise ValueError("W1 scorer microbench score_count mismatch")
    try:
        score_min = float(report.get("score_min"))
        score_max = float(report.get("score_max"))
    except (TypeError, ValueError) as exc:
        raise ValueError("W1 scorer microbench score range must be numeric") from exc
    if not (math.isfinite(score_min) and math.isfinite(score_max) and 0.0 <= score_min <= score_max <= 1.0):
        raise ValueError("W1 scorer microbench score range mismatch")
    window_length_summary = report.get("window_length_summary")
    if not isinstance(window_length_summary, Mapping) or int(window_length_summary.get("count", -1)) != tokenized_window_count:
        raise ValueError("W1 scorer microbench window_length_summary mismatch")
    if report.get("pooling_backend") != "vectorized_topk_mean":
        raise ValueError("W1 scorer microbench pooling_backend mismatch")
    if _positive_int(
        report.get("pooling_vectorized_topk_call_count"),
        "W1 scorer microbench pooling_vectorized_topk_call_count",
    ) != 1:
        raise ValueError("W1 scorer microbench pooling_vectorized_topk_call_count mismatch")
    if _positive_int(
        report.get("pooling_vectorized_topk_input_window_count"),
        "W1 scorer microbench pooling_vectorized_topk_input_window_count",
    ) != tokenized_window_count:
        raise ValueError("W1 scorer microbench pooling_vectorized_topk_input_window_count mismatch")
    if _positive_int(
        report.get("pooling_vectorized_topk_document_count"),
        "W1 scorer microbench pooling_vectorized_topk_document_count",
    ) != document_count:
        raise ValueError("W1 scorer microbench pooling_vectorized_topk_document_count mismatch")
    if _positive_int(
        report.get("pooling_vectorized_padded_cell_count"),
        "W1 scorer microbench pooling_vectorized_padded_cell_count",
    ) != document_count * max_windows_per_document:
        raise ValueError("W1 scorer microbench pooling_vectorized_padded_cell_count mismatch")
    phase_seconds = report.get("phase_seconds")
    if not isinstance(phase_seconds, Mapping):
        raise ValueError("W1 scorer microbench phase_seconds must be an object")
    for phase in ("tokenization", "fake_forward", "pooling", "postprocess", "total"):
        _nonnegative_float(phase_seconds.get(phase), f"W1 scorer microbench phase_seconds.{phase}")
    phase_rates = report.get("phase_rates")
    if not isinstance(phase_rates, Mapping):
        raise ValueError("W1 scorer microbench phase_rates must be an object")
    for field in (
        "documents_per_total_second",
        "windows_per_total_second",
        "live_tokens_per_total_second",
        "padded_tokens_per_total_second",
    ):
        if _nonnegative_float(phase_rates.get(field), f"W1 scorer microbench phase_rates.{field}") <= 0.0:
            raise ValueError(f"W1 scorer microbench phase_rates.{field} must be positive")
    for field in ("windows_per_fake_forward_second", "windows_per_pooling_second"):
        _nonnegative_float(phase_rates.get(field), f"W1 scorer microbench phase_rates.{field}")
    if str(report.get("bottleneck_phase") or "") not in {"tokenization", "fake_forward", "pooling", "postprocess"}:
        raise ValueError("W1 scorer microbench bottleneck_phase mismatch")
    if not str(report.get("device_type") or ""):
        raise ValueError("W1 scorer microbench device_type missing")
    return microbench_hash


def _w1_live_artifact_evidence_hash(report: Mapping[str, Any], run_dir: str | Path) -> str:
    run_path = Path(run_dir)
    receipt_path = run_path / "w1_score_receipt.json"
    if not receipt_path.exists():
        raise ValueError("missing live W1 score receipt")
    receipt = _load_json(receipt_path)
    if not isinstance(receipt, Mapping):
        raise ValueError("live W1 score receipt must be an object")
    validate_public_no_raw_payload(receipt)
    expected_receipt_hash = stable_json_hash(
        {key: value for key, value in dict(receipt).items() if key != "stage_receipt_hash"}
    )
    report_hash = _require_sha256(report.get("stage_receipt_hash"), "stage_receipt_hash")
    if receipt.get("stage_receipt_hash") != expected_receipt_hash or report_hash != expected_receipt_hash:
        raise ValueError("live W1 score receipt hash mismatch")
    scores_ready = run_path / "w1_scores.jsonl.ready"
    if not scores_ready.exists():
        raise ValueError("missing live W1 score export")
    if report.get("w1_scores_hash") != _sha256_file(scores_ready):
        raise ValueError("live W1 score export hash mismatch")
    queue_ready = run_path / "w1_survivor_queue.jsonl.ready"
    if not queue_ready.exists():
        raise ValueError("missing live W1 survivor queue")
    if report.get("survivor_queue_hash") != _sha256_file(queue_ready):
        raise ValueError("live W1 survivor queue hash mismatch")
    return stable_json_hash(
        {
            "stage_receipt_hash": report_hash,
            "w1_scores_hash": _require_sha256(report.get("w1_scores_hash"), "w1_scores_hash"),
            "survivor_queue_hash": _require_sha256(report.get("survivor_queue_hash"), "survivor_queue_hash"),
        }
    )


def _validated_w1_performance_report(
    report: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
) -> dict[str, Any]:
    validate_public_no_raw_payload(report)
    artifact_kind = str(report.get("artifact_kind") or "")
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
    ):
        if report.get(field) != config_payload.get(field):
            raise ValueError(f"W1 report {field} mismatch")
    if artifact_kind == W1_SCORE_STAGE_KIND:
        report_hash = _require_sha256(report.get("stage_receipt_hash"), "stage_receipt_hash")
        expected_hash = stable_json_hash({key: value for key, value in dict(report).items() if key != "stage_receipt_hash"})
        if report_hash != expected_hash:
            raise ValueError("W1 stage receipt hash mismatch")
        runtime_envelope, _runtime_envelope_hash = _validated_w1_runtime_envelope_from_receipt(
            report,
            label="W1 performance ledger receipt",
        )
        _validate_w1_performance_runtime_envelope(runtime_envelope, config_payload=config_payload)
        status = str(report.get("status") or "")
        blocking_reasons = []
        if status != "w1_scored":
            blocking_reasons.append("incomplete_w1_report")
        scored_count = int(report.get("scored_count", -1))
        if scored_count != int(report.get("survivor_count", -2)):
            blocking_reasons.append("incomplete_w1_report")
        telemetry = dict(report.get("w1_runtime_telemetry") or {})
        docs_scored_this_invocation = int(report.get("docs_scored_this_invocation", 0) or 0)
        scored_existing_initial = int(telemetry.get("scored_existing_initial", -1) or 0)
        score_rows_inserted = int(telemetry.get("score_rows_inserted", -1) or 0)
        if scored_existing_initial != 0 or docs_scored_this_invocation != scored_count or score_rows_inserted != scored_count:
            blocking_reasons.append("not_fresh_scoring_invocation")
        survivor_queue_hash = report.get("survivor_queue_hash")
        if survivor_queue_hash is None:
            blocking_reasons.append("missing_w1_survivor_queue_hash")
            normalized_survivor_queue_hash = None
        else:
            normalized_survivor_queue_hash = _require_sha256(survivor_queue_hash, "survivor_queue_hash")
        docs_per_second = round(float(report.get("docs_per_second", 0.0) or 0.0), 6)
        return {
            "artifact_kind": artifact_kind,
            "report_hash": report_hash,
            "status": status,
            "coverage_scope": "single_run_survivor_queue",
            "survivor_count": int(report.get("survivor_count", 0)),
            "survivor_queue_hash": normalized_survivor_queue_hash,
            "scored_count": int(report.get("scored_count", 0)),
            "docs_per_second": docs_per_second,
            "blocking_reasons": sorted(set(blocking_reasons)),
        }
    if artifact_kind == SHARDED_W1_SCORE_SUMMARY_KIND:
        report_hash = _require_sha256(report.get("summary_hash"), "summary_hash")
        expected_hash = stable_json_hash({key: value for key, value in dict(report).items() if key != "summary_hash"})
        if report_hash != expected_hash:
            raise ValueError("W1 sharded summary hash mismatch")
        w1_runtime_envelope = report.get("w1_runtime_envelope")
        if not isinstance(w1_runtime_envelope, Mapping):
            raise ValueError("W1 sharded summary missing runtime envelope")
        w1_runtime_envelope_hash = _require_sha256(
            report.get("w1_runtime_envelope_hash"),
            "w1_runtime_envelope_hash",
        )
        if _w1_runtime_execution_envelope_hash(w1_runtime_envelope) != w1_runtime_envelope_hash:
            raise ValueError("W1 sharded summary runtime envelope hash mismatch")
        _validate_w1_performance_runtime_envelope(w1_runtime_envelope, config_payload=config_payload)
        status = str(report.get("status") or "")
        blocking_reasons = ["missing_command_wall_time_evidence"]
        if status != "w1_sharded_scored":
            blocking_reasons.append("incomplete_w1_report")
        if int(report.get("scored_count", -1)) != int(report.get("survivor_count", -2)):
            blocking_reasons.append("incomplete_w1_report")
        telemetry = dict(report.get("w1_runtime_telemetry") or {})
        docs_per_second = round(float(telemetry.get("score_model_docs_per_second", 0.0) or 0.0), 6)
        return {
            "artifact_kind": artifact_kind,
            "report_hash": report_hash,
            "status": status,
            "coverage_scope": "sharded_w1_handoff",
            "survivor_count": int(report.get("survivor_count", 0)),
            "survivor_queue_hash": report.get("survivor_queue_hash"),
            "scored_count": int(report.get("scored_count", 0)),
            "docs_per_second": docs_per_second,
            "blocking_reasons": sorted(set(blocking_reasons)),
        }
    raise ValueError("unsupported W1 performance report kind")


def _validate_w1_performance_runtime_envelope(
    envelope: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
) -> None:
    validate_public_no_raw_payload(envelope)
    field_pairs = (
        ("config_hash", "config_hash"),
        ("selected_policy_hash", "selected_policy_hash"),
        ("selected_policy_report_hash", "selected_policy_report_hash"),
        ("source_metadata_hash", "source_metadata_hash"),
        ("window_mil_report_hash", "window_mil_report_hash"),
        ("window_mil_model_file_sha256", "window_mil_model_file_sha256"),
        ("window_mil_model_id", "window_mil_model_id"),
        ("window_mil_input_mode", "window_mil_input_mode"),
        ("window_mil_pooling_mode", "window_mil_pooling_mode"),
    )
    for envelope_field, config_field in field_pairs:
        if envelope.get(envelope_field) != config_payload.get(config_field):
            raise ValueError(f"W1 runtime envelope {envelope_field} mismatch")
    if float(envelope.get("window_mil_threshold")) != float(config_payload["window_mil_threshold"]):
        raise ValueError("W1 runtime envelope threshold mismatch")
    if int(envelope.get("window_mil_pooling_top_k")) != int(config_payload["window_mil_pooling_top_k"]):
        raise ValueError("W1 runtime envelope pooling_top_k mismatch")
    precision_mode = normalize_w1_precision_mode(str(config_payload.get("window_mil_precision") or LOCKED_W1_PRECISION))
    if normalize_w1_precision_mode(str(envelope.get("window_mil_precision") or "")) != precision_mode:
        raise ValueError("W1 runtime envelope precision mismatch")
    configured_precision_policy = config_payload.get("window_mil_precision_policy")
    expected_policy = (
        build_w1_precision_policy(precision_mode)
        if configured_precision_policy is None
        else dict(configured_precision_policy)
    )
    validate_w1_precision_policy(expected_policy)
    if envelope.get("window_mil_precision_policy_hash") != expected_policy["precision_policy_hash"]:
        raise ValueError("W1 runtime envelope precision policy hash mismatch")


def build_w1_scorer_performance_ledger(
    *,
    config_payload: Mapping[str, Any],
    w1_reports: Sequence[Mapping[str, Any]],
    w1_run_dirs: Sequence[str | Path] = (),
    microbench_reports: Sequence[Mapping[str, Any]] = (),
    length_distribution_report: Mapping[str, Any] | None = None,
    target_docs_per_second: float = 1000.0,
    require_synchronized_cuda: bool = True,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    if float(target_docs_per_second) <= 0.0:
        raise ValueError("target_docs_per_second must be positive")
    if w1_run_dirs and len(w1_run_dirs) != len(w1_reports):
        raise ValueError("w1_run_dirs must match w1_reports when provided")
    blocking_reasons: list[str] = []
    w1_report_infos = [
        _validated_w1_performance_report(report, config_payload=config_payload)
        for report in w1_reports
    ]
    normalized_w1_reports = [dict(report) for report in w1_reports]
    if not w1_report_infos:
        blocking_reasons.append("missing_w1_report")
    for info in w1_report_infos:
        blocking_reasons.extend(str(reason) for reason in info.get("blocking_reasons", []))
    telemetry = _aggregate_w1_runtime_telemetry(normalized_w1_reports) if normalized_w1_reports else {}
    if normalized_w1_reports and not bool(telemetry.get("cuda_device_observed", False)):
        blocking_reasons.append("missing_cuda_w1_evidence")
    if normalized_w1_reports and not _w1_runtime_phase_evidence_present(telemetry):
        blocking_reasons.append("missing_w1_phase_breakdown")
    if bool(require_synchronized_cuda):
        if not bool(telemetry.get("score_cuda_synchronized_timing_observed", False)):
            blocking_reasons.append("missing_synchronized_cuda_timing")
        if int(telemetry.get("score_cuda_synchronization_count", 0) or 0) <= 0:
            if "missing_synchronized_cuda_timing" not in blocking_reasons:
                blocking_reasons.append("missing_synchronized_cuda_timing")

    live_artifact_hashes: list[str | None] = []
    run_dirs_by_index = [Path(path) for path in w1_run_dirs]
    for index, (report, info) in enumerate(zip(normalized_w1_reports, w1_report_infos, strict=True)):
        if str(info.get("artifact_kind")) != W1_SCORE_STAGE_KIND:
            live_artifact_hashes.append(None)
            continue
        if not run_dirs_by_index:
            blocking_reasons.append("missing_live_w1_artifact_evidence")
            live_artifact_hashes.append(None)
            continue
        try:
            live_artifact_hashes.append(_w1_live_artifact_evidence_hash(report, run_dirs_by_index[index]))
        except ValueError:
            blocking_reasons.append("missing_live_w1_artifact_evidence")
            live_artifact_hashes.append(None)

    microbench_hashes = [_validate_w1_microbench_report(report) for report in microbench_reports]
    if not microbench_hashes:
        blocking_reasons.append("missing_microbench_report")

    length_distribution_hash = None
    length_distribution_survivor_queue_hash = None
    length_survivor_count = None
    if length_distribution_report is None:
        blocking_reasons.append("missing_length_distribution_report")
    else:
        validate_w1_survivor_length_distribution_report(
            length_distribution_report,
            config_payload=config_payload,
        )
        length_distribution_hash = _require_sha256(
            length_distribution_report.get("length_distribution_hash"),
            "length_distribution_hash",
        )
        length_distribution_survivor_queue_hash = _require_sha256(
            length_distribution_report.get("survivor_queue_hash"),
            "length_distribution_survivor_queue_hash",
        )
        length_survivor_count = int(length_distribution_report.get("survivor_count", 0))
        length_scope = str(length_distribution_report.get("coverage_scope") or "")
        report_scopes = {str(info.get("coverage_scope") or "") for info in w1_report_infos}
        if report_scopes and report_scopes != {length_scope}:
            blocking_reasons.append("length_distribution_scope_mismatch")
        if length_scope == "single_run_survivor_queue":
            if len(w1_report_infos) != 1:
                blocking_reasons.append("single_run_length_distribution_report_count_mismatch")
            elif int(w1_report_infos[0].get("survivor_count", -1)) != length_survivor_count:
                blocking_reasons.append("length_distribution_survivor_count_mismatch")
            elif w1_report_infos[0].get("survivor_queue_hash") is None:
                blocking_reasons.append("missing_w1_survivor_queue_hash")
            elif w1_report_infos[0].get("survivor_queue_hash") != length_distribution_survivor_queue_hash:
                blocking_reasons.append("length_distribution_survivor_queue_hash_mismatch")

    observed_docs_per_second_values = [float(info.get("docs_per_second", 0.0) or 0.0) for info in w1_report_infos]
    observed_docs_per_second = min(observed_docs_per_second_values) if observed_docs_per_second_values else 0.0
    if w1_report_infos and observed_docs_per_second < float(target_docs_per_second):
        blocking_reasons.append("single_gpu_target_not_met")

    payload = {
        "artifact_kind": W1_SCORER_PERFORMANCE_LEDGER_KIND,
        "schema_version": 1,
        "status": "blocked" if blocking_reasons else "ready_for_paid_canary",
        "blocking_reasons": sorted(set(blocking_reasons)),
        "claim_scope": "w1_scorer_performance_evidence_gate",
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "window_mil_report_hash": config_payload["window_mil_report_hash"],
        "window_mil_model_file_sha256": config_payload["window_mil_model_file_sha256"],
        "window_mil_threshold": float(config_payload["window_mil_threshold"]),
        "window_mil_precision": config_payload.get("window_mil_precision"),
        "target_docs_per_second": round(float(target_docs_per_second), 6),
        "observed_docs_per_second": round(float(observed_docs_per_second), 6),
        "w1_report_count": len(w1_report_infos),
        "w1_report_hashes": [str(info["report_hash"]) for info in w1_report_infos],
        "w1_report_artifact_kinds": [str(info["artifact_kind"]) for info in w1_report_infos],
        "w1_report_statuses": [str(info["status"]) for info in w1_report_infos],
        "w1_report_coverage_scopes": [str(info["coverage_scope"]) for info in w1_report_infos],
        "w1_report_survivor_queue_hashes": [info.get("survivor_queue_hash") for info in w1_report_infos],
        "w1_report_live_artifact_evidence_hashes": live_artifact_hashes,
        "w1_report_live_artifact_evidence_count": sum(1 for value in live_artifact_hashes if value is not None),
        "microbench_report_count": len(microbench_hashes),
        "microbench_report_hashes": microbench_hashes,
        "length_distribution_hash": length_distribution_hash,
        "length_distribution_survivor_queue_hash": length_distribution_survivor_queue_hash,
        "length_distribution_survivor_count": length_survivor_count,
        "require_synchronized_cuda": bool(require_synchronized_cuda),
        "aggregated_w1_runtime_telemetry": telemetry,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["performance_ledger_hash"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "performance_ledger_hash"}
    )
    validate_w1_scorer_performance_ledger(payload, config_payload=config_payload)
    if report_path is not None:
        _write_json_atomic(report_path, payload)
    return payload


def validate_w1_scorer_performance_ledger(
    ledger: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
) -> None:
    if ledger.get("artifact_kind") != W1_SCORER_PERFORMANCE_LEDGER_KIND:
        raise ValueError("unsupported W1 scorer performance ledger kind")
    validate_public_no_raw_payload(ledger)
    expected_hash = stable_json_hash(
        {key: value for key, value in dict(ledger).items() if key != "performance_ledger_hash"}
    )
    if ledger.get("performance_ledger_hash") != expected_hash:
        raise ValueError("W1 scorer performance ledger hash mismatch")
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "window_mil_report_hash",
        "window_mil_model_file_sha256",
    ):
        if ledger.get(field) != config_payload.get(field):
            raise ValueError(f"W1 scorer performance ledger {field} mismatch")
    try:
        if float(ledger.get("window_mil_threshold")) != float(config_payload["window_mil_threshold"]):
            raise ValueError("W1 scorer performance ledger threshold mismatch")
    except (TypeError, ValueError) as exc:
        raise ValueError("W1 scorer performance ledger threshold mismatch") from exc
    ledger_precision = normalize_w1_precision_mode(str(ledger.get("window_mil_precision") or ""))
    config_precision = normalize_w1_precision_mode(str(config_payload.get("window_mil_precision") or LOCKED_W1_PRECISION))
    if ledger_precision != config_precision:
        raise ValueError("W1 scorer performance ledger precision mismatch")
    if ledger.get("status") not in {"blocked", "ready_for_paid_canary"}:
        raise ValueError("unsupported W1 scorer performance ledger status")
    blocking_reasons = ledger.get("blocking_reasons")
    if not isinstance(blocking_reasons, list):
        raise ValueError("W1 scorer performance ledger blocking_reasons must be a list")
    target_docs_per_second = float(ledger.get("target_docs_per_second", 0.0) or 0.0)
    if target_docs_per_second <= 0.0:
        raise ValueError("W1 scorer performance ledger target_docs_per_second must be positive")
    observed_docs_per_second = float(ledger.get("observed_docs_per_second", 0.0) or 0.0)
    if observed_docs_per_second < 0.0:
        raise ValueError("W1 scorer performance ledger observed_docs_per_second must be nonnegative")
    w1_report_count = _nonnegative_int(ledger.get("w1_report_count", -1), "W1 scorer performance ledger w1_report_count")
    w1_report_hashes = ledger.get("w1_report_hashes")
    if not isinstance(w1_report_hashes, list) or len(w1_report_hashes) != w1_report_count:
        raise ValueError("W1 scorer performance ledger w1_report_hashes count mismatch")
    for index, report_hash in enumerate(w1_report_hashes):
        _require_sha256(report_hash, f"W1 scorer performance ledger w1_report_hashes[{index}]")
    for field in ("w1_report_artifact_kinds", "w1_report_statuses", "w1_report_coverage_scopes"):
        values = ledger.get(field)
        if not isinstance(values, list) or len(values) != w1_report_count:
            raise ValueError(f"W1 scorer performance ledger {field} count mismatch")
    w1_survivor_queue_hashes = ledger.get("w1_report_survivor_queue_hashes")
    if not isinstance(w1_survivor_queue_hashes, list) or len(w1_survivor_queue_hashes) != w1_report_count:
        raise ValueError("W1 scorer performance ledger w1_report_survivor_queue_hashes count mismatch")
    for index, survivor_queue_hash in enumerate(w1_survivor_queue_hashes):
        if survivor_queue_hash is not None:
            _require_sha256(
                survivor_queue_hash,
                f"W1 scorer performance ledger w1_report_survivor_queue_hashes[{index}]",
            )
    live_artifact_hashes = ledger.get("w1_report_live_artifact_evidence_hashes")
    if live_artifact_hashes is None:
        live_artifact_hashes = []
    if not isinstance(live_artifact_hashes, list):
        raise ValueError("W1 scorer performance ledger w1_report_live_artifact_evidence_hashes must be a list")
    if live_artifact_hashes and len(live_artifact_hashes) != w1_report_count:
        raise ValueError("W1 scorer performance ledger live artifact evidence count mismatch")
    for index, live_artifact_hash in enumerate(live_artifact_hashes):
        if live_artifact_hash is not None:
            _require_sha256(
                live_artifact_hash,
                f"W1 scorer performance ledger w1_report_live_artifact_evidence_hashes[{index}]",
            )
    live_artifact_count = _nonnegative_int(
        ledger.get("w1_report_live_artifact_evidence_count", 0),
        "W1 scorer performance ledger w1_report_live_artifact_evidence_count",
    )
    if live_artifact_hashes and live_artifact_count != sum(1 for value in live_artifact_hashes if value is not None):
        raise ValueError("W1 scorer performance ledger live artifact evidence count mismatch")
    microbench_report_count = _nonnegative_int(
        ledger.get("microbench_report_count", -1),
        "W1 scorer performance ledger microbench_report_count",
    )
    microbench_hashes = ledger.get("microbench_report_hashes")
    if not isinstance(microbench_hashes, list) or len(microbench_hashes) != microbench_report_count:
        raise ValueError("W1 scorer performance ledger microbench_report_hashes count mismatch")
    for index, microbench_hash in enumerate(microbench_hashes):
        _require_sha256(microbench_hash, f"W1 scorer performance ledger microbench_report_hashes[{index}]")
    length_distribution_hash = ledger.get("length_distribution_hash")
    if length_distribution_hash is not None:
        _require_sha256(length_distribution_hash, "W1 scorer performance ledger length_distribution_hash")
    length_distribution_survivor_queue_hash = ledger.get("length_distribution_survivor_queue_hash")
    if length_distribution_survivor_queue_hash is not None:
        _require_sha256(
            length_distribution_survivor_queue_hash,
            "W1 scorer performance ledger length_distribution_survivor_queue_hash",
        )
    telemetry = ledger.get("aggregated_w1_runtime_telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("W1 scorer performance ledger aggregated_w1_runtime_telemetry must be an object")
    if ledger.get("status") == "blocked" and not blocking_reasons:
        raise ValueError("blocked W1 scorer performance ledger must include blocking reasons")
    if ledger.get("status") == "ready_for_paid_canary" and blocking_reasons:
        raise ValueError("ready W1 scorer performance ledger must not include blocking reasons")
    if ledger.get("status") == "ready_for_paid_canary":
        if w1_report_count <= 0:
            raise ValueError("ready W1 scorer performance ledger missing W1 report evidence")
        if microbench_report_count <= 0:
            raise ValueError("ready W1 scorer performance ledger missing microbench evidence")
        if length_distribution_hash is None:
            raise ValueError("ready W1 scorer performance ledger missing length distribution evidence")
        if length_distribution_survivor_queue_hash is None:
            raise ValueError("ready W1 scorer performance ledger missing length distribution survivor queue evidence")
        if any(hash_value is None for hash_value in w1_survivor_queue_hashes):
            raise ValueError("ready W1 scorer performance ledger missing W1 survivor queue evidence")
        if (
            not live_artifact_hashes
            or len(live_artifact_hashes) != w1_report_count
            or any(hash_value is None for hash_value in live_artifact_hashes)
            or live_artifact_count != w1_report_count
        ):
            raise ValueError("ready W1 scorer performance ledger missing live W1 artifact evidence")
        if set(w1_survivor_queue_hashes) != {length_distribution_survivor_queue_hash}:
            raise ValueError("ready W1 scorer performance ledger survivor queue mismatch")
        if observed_docs_per_second < target_docs_per_second:
            raise ValueError("ready W1 scorer performance ledger target was not met")
        if not _w1_runtime_phase_evidence_present(telemetry):
            raise ValueError("ready W1 scorer performance ledger missing phase breakdown")
        if not bool(telemetry.get("cuda_device_observed", False)):
            raise ValueError("ready W1 scorer performance ledger missing CUDA evidence")
        if bool(ledger.get("require_synchronized_cuda", True)):
            if not bool(telemetry.get("score_cuda_synchronized_timing_observed", False)):
                raise ValueError("ready W1 scorer performance ledger missing synchronized CUDA timing")
            if int(telemetry.get("score_cuda_synchronization_count", 0) or 0) <= 0:
                raise ValueError("ready W1 scorer performance ledger missing synchronized CUDA timing")


def _load_jsonl_path(path: str | Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(payload)
    return rows


def _require_git_commit(value: Any, label: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{40}", text):
        raise ValueError(f"{label} must be a 40-character lowercase Git commit")
    return text


def build_w1_fa2_package_candidate_manifest(
    *,
    code_commit: str,
    pyproject_hash: str,
    uv_lock_hash: str,
    python_version: str,
    platform_tag: str,
    torch_version_expected: str,
    torch_cuda_expected: str,
    torch_cxx11_abi_expected: str | int | bool | None,
    flash_attn_version: str,
    flash_attn_wheel_filename: str,
    flash_attn_wheel_sha256: str,
    install_command_hash: str,
    install_log_summary_hash: str,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_kind": W1_FA2_PACKAGE_CANDIDATE_MANIFEST_KIND,
        "schema_version": 1,
        "status": "w1_fa2_package_candidate_ready",
        "claims_scope": "fa2_package_candidate_only",
        "code_commit": _require_git_commit(code_commit, "code_commit"),
        "pyproject_hash": _require_sha256(pyproject_hash, "pyproject_hash"),
        "uv_lock_hash": _require_sha256(uv_lock_hash, "uv_lock_hash"),
        "python_version": str(python_version),
        "platform_tag": str(platform_tag),
        "torch_version_expected": str(torch_version_expected),
        "torch_cuda_expected": str(torch_cuda_expected),
        "torch_cxx11_abi_expected": None
        if torch_cxx11_abi_expected is None
        else str(torch_cxx11_abi_expected),
        "flash_attn_package": "flash-attn",
        "flash_attn_version": str(flash_attn_version),
        "flash_attn_wheel_filename": str(flash_attn_wheel_filename),
        "flash_attn_wheel_sha256": _require_sha256(
            flash_attn_wheel_sha256,
            "flash_attn_wheel_sha256",
        ),
        "install_mode": "wheelhouse_no_deps",
        "dependency_resolution_enabled": False,
        "source_build_detected": False,
        "install_command_hash": _require_sha256(install_command_hash, "install_command_hash"),
        "install_log_summary_hash": _require_sha256(
            install_log_summary_hash,
            "install_log_summary_hash",
        ),
        "full_install_log_serialized": False,
        **{field: False for field in PUBLIC_FALSE_FLAGS},
    }
    payload["manifest_hash"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    validate_w1_fa2_package_candidate_manifest(payload)
    if report_path is not None:
        _write_json_atomic(report_path, payload)
    return payload


W1_FA2_SOURCE_BUILD_MARKERS = (
    "--no-build-isolation",
    "--no-binary",
    "building wheel",
    "building editable",
    "source build",
    "from source",
    "setup.py",
    "sdist",
    ".tar.gz",
    ".zip",
)


def _read_small_text_file(path: Path, *, label: str, max_bytes: int) -> str:
    if not path.exists():
        raise ValueError(f"{label} is missing")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"{label} is empty")
    if size > int(max_bytes):
        raise ValueError(f"{label} is too large for a public-hash workflow")
    return path.read_text(encoding="utf-8")


def _verify_clean_code_commit(code_root: str | Path, expected_commit: str) -> None:
    root = Path(code_root).expanduser().resolve(strict=False)
    if not root.exists():
        raise RuntimeError("W1 FA2 runtime code root is missing")
    expected = _require_git_commit(expected_commit, "code_commit")

    def _run_git(args: Sequence[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if int(getattr(result, "returncode", 1)) != 0:
            raise RuntimeError("W1 FA2 runtime git proof failed")
        return str(getattr(result, "stdout", "") or "").strip()

    observed = _run_git(["rev-parse", "HEAD"])
    if observed != expected:
        raise RuntimeError("W1 FA2 runtime code commit mismatch")
    dirty = _run_git(["status", "--short"])
    if dirty:
        raise RuntimeError("W1 FA2 runtime code tree is dirty")


def _hash_flash_attn_extension_member_from_wheel(wheel_path: str | Path) -> str:
    resolved = Path(wheel_path).expanduser().resolve(strict=False)
    if not resolved.exists() or not resolved.is_file():
        raise RuntimeError("W1 FA2 runtime candidate wheel is missing")
    try:
        with zipfile.ZipFile(resolved) as wheel:
            members = [
                info
                for info in wheel.infolist()
                if Path(info.filename).name.startswith("flash_attn_2_cuda")
                and Path(info.filename).suffix in {".so", ".pyd"}
            ]
            if len(members) != 1:
                raise RuntimeError("W1 FA2 runtime wheel must contain exactly one flash_attn_2_cuda extension")
            return hashlib.sha256(wheel.read(members[0])).hexdigest()
    except zipfile.BadZipFile as exc:
        raise RuntimeError("W1 FA2 runtime candidate wheel is not a valid wheel archive") from exc


def _build_fa2_package_runtime_binding(
    *,
    package_candidate_manifest: Mapping[str, Any],
    flash_attn_wheel: str | Path,
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    validate_w1_fa2_package_candidate_manifest(package_candidate_manifest)
    wheel_path = Path(flash_attn_wheel).expanduser().resolve(strict=False)
    if not wheel_path.exists() or not wheel_path.is_file():
        raise RuntimeError("W1 FA2 runtime candidate wheel is missing")
    wheel_hash = _sha256_file(wheel_path)
    expected_wheel_hash = str(package_candidate_manifest.get("flash_attn_wheel_sha256") or "")
    if wheel_hash != expected_wheel_hash:
        raise RuntimeError("W1 FA2 runtime candidate wheel hash mismatch")
    wheel_extension_hash = _hash_flash_attn_extension_member_from_wheel(wheel_path)
    installed_extension_hash = _require_sha256(
        runtime_environment.get("flash_attn_extension_file_hash"),
        "flash_attn_extension_file_hash",
    )
    observed_version = str(runtime_environment.get("flash_attn_version") or "")
    expected_version = str(package_candidate_manifest.get("flash_attn_version") or "")
    if observed_version != expected_version:
        raise RuntimeError("W1 FA2 runtime flash-attn version mismatch")
    if installed_extension_hash != wheel_extension_hash:
        raise RuntimeError("W1 FA2 runtime installed extension does not match candidate wheel extension")
    binding = {
        "package_candidate_manifest_hash": str(package_candidate_manifest["manifest_hash"]),
        "flash_attn_wheel_filename": str(package_candidate_manifest.get("flash_attn_wheel_filename") or ""),
        "flash_attn_wheel_sha256": wheel_hash,
        "flash_attn_extension_wheel_member_sha256": wheel_extension_hash,
        "flash_attn_installed_extension_sha256": installed_extension_hash,
        "flash_attn_version_expected": expected_version,
        "flash_attn_version_observed": observed_version,
        "flash_attn_version_match": True,
        "flash_attn_extension_match": True,
    }
    _validate_fa2_package_runtime_binding(
        binding,
        package_candidate_manifest=package_candidate_manifest,
        runtime_environment=runtime_environment,
    )
    return binding


def _validate_w1_fa2_package_candidate_workflow_inputs(
    *,
    wheel_path: Path,
    command_text: str,
    log_summary_text: str,
) -> None:
    wheel_name = wheel_path.name
    if not wheel_path.exists() or not wheel_path.is_file():
        raise ValueError("W1 FA2 package candidate wheel is missing")
    if not wheel_name.endswith(".whl"):
        raise ValueError("W1 FA2 package candidate requires a wheel file")
    command_lower = command_text.lower()
    summary_lower = log_summary_text.lower()
    combined = f"{command_lower}\n{summary_lower}"
    if any(marker in combined for marker in W1_FA2_SOURCE_BUILD_MARKERS):
        raise ValueError("W1 FA2 package candidate source build markers detected")
    if "--no-deps" not in command_lower:
        raise ValueError("W1 FA2 package candidate install command must include --no-deps")
    if wheel_name.lower() not in command_lower:
        raise ValueError("W1 FA2 package candidate install command must reference the candidate wheel")


def build_w1_fa2_package_candidate_manifest_from_files(
    *,
    code_root: str | Path,
    code_commit: str,
    flash_attn_wheel: str | Path,
    flash_attn_version: str,
    install_command_file: str | Path,
    install_log_summary: str | Path,
    python_version: str | None = None,
    platform_tag: str | None = None,
    torch_version_expected: str = "2.6.0+cu124",
    torch_cuda_expected: str = "12.4",
    torch_cxx11_abi_expected: str | int | bool | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(code_root).expanduser().resolve(strict=False)
    pyproject = root / "pyproject.toml"
    uv_lock = root / "uv.lock"
    wheel_path = Path(flash_attn_wheel).expanduser().resolve(strict=False)
    command_path = Path(install_command_file).expanduser().resolve(strict=False)
    log_summary_path = Path(install_log_summary).expanduser().resolve(strict=False)
    command_text = _read_small_text_file(command_path, label="W1 FA2 install command file", max_bytes=8192)
    log_summary_text = _read_small_text_file(
        log_summary_path,
        label="W1 FA2 install log summary",
        max_bytes=65536,
    )
    _validate_w1_fa2_package_candidate_workflow_inputs(
        wheel_path=wheel_path,
        command_text=command_text,
        log_summary_text=log_summary_text,
    )
    if not pyproject.exists():
        raise ValueError("W1 FA2 package candidate pyproject.toml is missing")
    if not uv_lock.exists():
        raise ValueError("W1 FA2 package candidate uv.lock is missing")
    resolved_python_version = python_version or (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    resolved_platform_tag = platform_tag or f"{sys.platform}-{platform.machine()}"
    return build_w1_fa2_package_candidate_manifest(
        code_commit=code_commit,
        pyproject_hash=_sha256_file(pyproject),
        uv_lock_hash=_sha256_file(uv_lock),
        python_version=resolved_python_version,
        platform_tag=resolved_platform_tag,
        torch_version_expected=torch_version_expected,
        torch_cuda_expected=torch_cuda_expected,
        torch_cxx11_abi_expected=torch_cxx11_abi_expected,
        flash_attn_version=flash_attn_version,
        flash_attn_wheel_filename=wheel_path.name,
        flash_attn_wheel_sha256=_sha256_file(wheel_path),
        install_command_hash=_sha256_file(command_path),
        install_log_summary_hash=_sha256_file(log_summary_path),
        report_path=report_path,
    )


def validate_w1_fa2_package_candidate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("artifact_kind") != W1_FA2_PACKAGE_CANDIDATE_MANIFEST_KIND:
        raise ValueError("W1 FA2 package candidate manifest kind mismatch")
    if int(manifest.get("schema_version") or 0) != 1:
        raise ValueError("W1 FA2 package candidate manifest schema_version mismatch")
    if manifest.get("manifest_hash") != stable_json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    ):
        raise ValueError("W1 FA2 package candidate manifest hash mismatch")
    if manifest.get("status") != "w1_fa2_package_candidate_ready":
        raise ValueError("W1 FA2 package candidate manifest status mismatch")
    if manifest.get("claims_scope") != "fa2_package_candidate_only":
        raise ValueError("W1 FA2 package candidate claims_scope mismatch")
    _require_git_commit(manifest.get("code_commit"), "code_commit")
    if str(manifest.get("flash_attn_package") or "") != "flash-attn":
        raise ValueError("W1 FA2 package candidate package mismatch")
    if str(manifest.get("install_mode") or "") != "wheelhouse_no_deps":
        raise ValueError("W1 FA2 package candidate install_mode mismatch")
    if bool(manifest.get("dependency_resolution_enabled", True)):
        raise ValueError("W1 FA2 package candidate dependency resolution must be disabled")
    if bool(manifest.get("source_build_detected", True)):
        raise ValueError("W1 FA2 package candidate source build must not be used")
    if not str(manifest.get("python_version") or "").startswith("3.11"):
        raise ValueError("W1 FA2 package candidate must target Python 3.11")
    if str(manifest.get("torch_version_expected") or "") != "2.6.0+cu124":
        raise ValueError("W1 FA2 package candidate torch version mismatch")
    if str(manifest.get("torch_cuda_expected") or "") != "12.4":
        raise ValueError("W1 FA2 package candidate CUDA runtime mismatch")
    wheel_filename = str(manifest.get("flash_attn_wheel_filename") or "")
    flash_attn_version = str(manifest.get("flash_attn_version") or "")
    if (
        not wheel_filename.startswith("flash_attn-")
        or not wheel_filename.endswith(".whl")
        or "cp311" not in wheel_filename
        or "linux" not in wheel_filename
        or f"flash_attn-{flash_attn_version}" not in wheel_filename
    ):
        raise ValueError("W1 FA2 package candidate wheel filename mismatch")
    _require_sha256(manifest.get("pyproject_hash"), "pyproject_hash")
    _require_sha256(manifest.get("uv_lock_hash"), "uv_lock_hash")
    _require_sha256(manifest.get("flash_attn_wheel_sha256"), "flash_attn_wheel_sha256")
    _require_sha256(manifest.get("install_command_hash"), "install_command_hash")
    _require_sha256(manifest.get("install_log_summary_hash"), "install_log_summary_hash")
    if manifest.get("full_install_log_serialized") is not False:
        raise ValueError("W1 FA2 package candidate must not serialize full install logs")
    _require_false_flags(manifest, PUBLIC_FALSE_FLAGS, "W1 FA2 package candidate manifest")
    validate_public_no_raw_payload(manifest)


def build_w1_fa2_runtime_compatibility_manifest(
    *,
    config_payload: Mapping[str, Any],
    package_candidate_manifest: Mapping[str, Any],
    code_commit: str,
    code_commit_verified: bool,
    code_tree_clean: bool,
    package_runtime_binding: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    synthetic_forward_smoke: Mapping[str, Any],
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    validate_locked_chunk_filtering_config_payload(config_payload)
    validate_w1_fa2_package_candidate_manifest(package_candidate_manifest)
    environment = dict(runtime_environment)
    smoke = dict(synthetic_forward_smoke)
    payload: dict[str, Any] = {
        "artifact_kind": W1_FA2_RUNTIME_COMPATIBILITY_MANIFEST_KIND,
        "schema_version": 1,
        "status": "w1_fa2_runtime_compatibility_ready",
        "claims_scope": "fa2_environment_load_and_forward_smoke_only",
        "package_candidate_manifest_hash": package_candidate_manifest["manifest_hash"],
        "code_commit": _require_git_commit(code_commit, "code_commit"),
        "code_commit_verified": bool(code_commit_verified),
        "code_tree_clean": bool(code_tree_clean),
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "w1_report_hash": config_payload["window_mil_report_hash"],
        "w1_model_file_sha256": config_payload["window_mil_model_file_sha256"],
        "w1_threshold": _threshold(config_payload["window_mil_threshold"], "w1_threshold"),
        "window_mil_precision": str(config_payload["window_mil_precision"]),
        "window_mil_precision_policy_hash": config_payload["window_mil_precision_policy"][
            "precision_policy_hash"
        ],
        "w1_attn_implementation_requested": W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2,
        "w1_attn_implementation_observed": normalize_w1_attn_implementation(
            str(environment.get("w1_attn_implementation_observed") or "")
        ),
        "runtime_environment": environment,
        "package_runtime_binding": dict(package_runtime_binding),
        "synthetic_forward_smoke": smoke,
        "install_mode": "wheelhouse_no_deps",
        "dependency_resolution_enabled": False,
        "source_build_detected": False,
        "full_install_log_serialized": False,
        **{field: False for field in PUBLIC_FALSE_FLAGS},
    }
    payload["manifest_hash"] = stable_json_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    validate_w1_fa2_runtime_compatibility_manifest(
        payload,
        config_payload=config_payload,
        package_candidate_manifest=package_candidate_manifest,
    )
    if report_path is not None:
        _write_json_atomic(report_path, payload)
    return payload


def build_w1_fa2_runtime_compatibility_manifest_from_scorer(
    *,
    config_payload: Mapping[str, Any],
    package_candidate_manifest: Mapping[str, Any],
    code_commit: str,
    code_root: str | Path,
    flash_attn_wheel: str | Path,
    w1_scorer: Any,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    _verify_clean_code_commit(code_root, code_commit)
    requested = normalize_w1_attn_implementation(
        str(getattr(w1_scorer, "w1_attn_implementation_requested", ""))
    )
    observed = normalize_w1_attn_implementation(
        str(getattr(w1_scorer, "w1_attn_implementation_observed", ""))
    )
    if requested != W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2 or observed != W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2:
        raise RuntimeError("W1 FA2 runtime smoke must load the scorer with flash_attention_2")
    torch_module = getattr(w1_scorer, "torch", None)
    device = getattr(w1_scorer, "device", None)
    if torch_module is None or device is None:
        raise RuntimeError("W1 FA2 runtime smoke requires a real torch-backed scorer")
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    device_type = str(getattr(device, "type", ""))
    if not cuda_available or device_type != "cuda":
        raise RuntimeError("W1 FA2 runtime smoke requires CUDA")
    is_bf16_supported = getattr(cuda, "is_bf16_supported", None)
    bf16_supported = bool(is_bf16_supported()) if callable(is_bf16_supported) else False
    if not bf16_supported:
        raise RuntimeError("W1 FA2 runtime smoke requires BF16 support")
    flash_attn_pkg = importlib.import_module("flash_attn")
    flash_attn_extension = importlib.import_module("flash_attn_2_cuda")
    extension_path = Path(str(getattr(flash_attn_extension, "__file__", ""))).expanduser()
    if not extension_path.exists():
        raise RuntimeError("flash_attn_2_cuda extension path is missing")
    current_device = int(cuda.current_device())
    capability = cuda.get_device_capability(current_device)
    torch_version = str(getattr(torch_module, "__version__", ""))
    torch_cuda_runtime = str(getattr(getattr(torch_module, "version", None), "cuda", "") or "")
    torch_cxx11_abi = getattr(getattr(torch_module, "_C", None), "_GLIBCXX_USE_CXX11_ABI", None)
    long_unit = "synthetic fa2 forward smoke window "
    texts = [
        long_unit * 220,
        "synthetic fa2 forward smoke short control",
    ]
    metadata = [
        {
            "candidate_id_hash": stable_json_hash("w1-fa2-smoke-0"),
            "core_char_start": 0,
            "core_char_end": len(texts[0]),
            "core_text_sha256": hashlib.sha256(texts[0].encode("utf-8")).hexdigest(),
        },
        {
            "candidate_id_hash": stable_json_hash("w1-fa2-smoke-1"),
            "core_char_start": 0,
            "core_char_end": len(texts[1]),
            "core_text_sha256": hashlib.sha256(texts[1].encode("utf-8")).hexdigest(),
        },
    ]
    scores = list(w1_scorer.score_many(texts, metadata))
    telemetry = dict(getattr(w1_scorer, "last_score_many_telemetry", {}) or {})
    tokenized_window_count = int(telemetry.get("tokenized_window_count", 0) or 0)
    runtime_environment = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform_tag": sys.platform,
        "torch_version": torch_version,
        "torch_cuda_runtime": torch_cuda_runtime,
        "torch_cxx11_abi": None if torch_cxx11_abi is None else str(torch_cxx11_abi),
        "cuda_available": True,
        "device_type": device_type,
        "gpu_name": str(cuda.get_device_name(current_device)),
        "gpu_compute_capability": [int(capability[0]), int(capability[1])],
        "bf16_supported": True,
        "flash_attn_version": str(getattr(flash_attn_pkg, "__version__", "")),
        "flash_attn_extension_file_hash": _sha256_file(extension_path),
        "w1_attn_implementation_observed": observed,
    }
    package_runtime_binding = _build_fa2_package_runtime_binding(
        package_candidate_manifest=package_candidate_manifest,
        flash_attn_wheel=flash_attn_wheel,
        runtime_environment=runtime_environment,
    )
    smoke = {
        "ran": True,
        "w1_scorer_path": "ChunkWindowMilRiskScorer.from_report",
        "document_count": len(texts),
        "score_count": len(scores),
        "tokenized_window_count": tokenized_window_count,
        "max_tokenized_sequence_length": int(telemetry.get("max_tokenized_sequence_length", 0) or 0),
        "overflow_window_observed": tokenized_window_count > len(texts),
        "attention_mask_observed": int(telemetry.get("tokenized_attention_token_count", 0) or 0) > 0,
        "bf16_runtime_observed": (
            normalize_w1_precision_mode(str(getattr(w1_scorer, "window_mil_precision", "")))
            == W1_PRECISION_AMP_BF16
            and device_type == "cuda"
        ),
        "score_values_serialized": False,
        **{field: False for field in PUBLIC_FALSE_FLAGS},
    }
    return build_w1_fa2_runtime_compatibility_manifest(
        config_payload=config_payload,
        package_candidate_manifest=package_candidate_manifest,
        code_commit=code_commit,
        code_commit_verified=True,
        code_tree_clean=True,
        package_runtime_binding=package_runtime_binding,
        runtime_environment=runtime_environment,
        synthetic_forward_smoke=smoke,
        report_path=report_path,
    )


def _validate_fa2_runtime_environment(environment: Mapping[str, Any]) -> None:
    if not bool(environment.get("cuda_available")):
        raise ValueError("W1 FA2 runtime compatibility missing CUDA availability")
    if str(environment.get("device_type") or "") != "cuda":
        raise ValueError("W1 FA2 runtime compatibility device_type mismatch")
    if not bool(environment.get("bf16_supported")):
        raise ValueError("W1 FA2 runtime compatibility missing BF16 support")
    if str(environment.get("torch_version") or "") != "2.6.0+cu124":
        raise ValueError("W1 FA2 runtime compatibility torch version mismatch")
    if str(environment.get("torch_cuda_runtime") or "") != "12.4":
        raise ValueError("W1 FA2 runtime compatibility CUDA runtime mismatch")
    if not str(environment.get("flash_attn_version") or ""):
        raise ValueError("W1 FA2 runtime compatibility flash-attn version missing")
    if normalize_w1_attn_implementation(
        str(environment.get("w1_attn_implementation_observed") or "")
    ) != W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2:
        raise ValueError("W1 FA2 runtime compatibility observed backend mismatch")
    _require_sha256(environment.get("flash_attn_extension_file_hash"), "flash_attn_extension_file_hash")
    validate_public_no_raw_payload(environment)


def _validate_fa2_synthetic_forward_smoke(smoke: Mapping[str, Any]) -> None:
    if not bool(smoke.get("ran")):
        raise ValueError("W1 FA2 runtime compatibility missing synthetic forward smoke")
    if str(smoke.get("w1_scorer_path") or "") != "ChunkWindowMilRiskScorer.from_report":
        raise ValueError("W1 FA2 runtime compatibility smoke must use W1 scorer path")
    document_count = int(smoke.get("document_count") or 0)
    score_count = int(smoke.get("score_count") or 0)
    tokenized_window_count = int(smoke.get("tokenized_window_count") or 0)
    max_sequence_length = int(smoke.get("max_tokenized_sequence_length") or 0)
    if document_count < 2:
        raise ValueError("W1 FA2 runtime compatibility smoke must score at least two documents")
    if score_count != document_count:
        raise ValueError("W1 FA2 runtime compatibility smoke score count mismatch")
    if tokenized_window_count < document_count:
        raise ValueError("W1 FA2 runtime compatibility smoke missing tokenized windows")
    if max_sequence_length <= 0:
        raise ValueError("W1 FA2 runtime compatibility smoke missing sequence length")
    if not bool(smoke.get("overflow_window_observed")):
        raise ValueError("W1 FA2 runtime compatibility smoke must observe an overflow window")
    if not bool(smoke.get("attention_mask_observed")):
        raise ValueError("W1 FA2 runtime compatibility smoke must observe an attention mask")
    if not bool(smoke.get("bf16_runtime_observed")):
        raise ValueError("W1 FA2 runtime compatibility smoke must observe BF16 runtime")
    if smoke.get("score_values_serialized") is not False:
        raise ValueError("W1 FA2 runtime compatibility smoke must not serialize score values")
    _require_false_flags(smoke, PUBLIC_FALSE_FLAGS, "W1 FA2 runtime compatibility smoke")
    validate_public_no_raw_payload(smoke)


def _validate_fa2_package_runtime_binding(
    binding: Mapping[str, Any],
    *,
    package_candidate_manifest: Mapping[str, Any] | None = None,
    runtime_environment: Mapping[str, Any],
) -> None:
    package_hash = _require_sha256(
        binding.get("package_candidate_manifest_hash"),
        "package_runtime_binding.package_candidate_manifest_hash",
    )
    wheel_hash = _require_sha256(
        binding.get("flash_attn_wheel_sha256"),
        "package_runtime_binding.flash_attn_wheel_sha256",
    )
    wheel_extension_hash = _require_sha256(
        binding.get("flash_attn_extension_wheel_member_sha256"),
        "package_runtime_binding.flash_attn_extension_wheel_member_sha256",
    )
    installed_extension_hash = _require_sha256(
        binding.get("flash_attn_installed_extension_sha256"),
        "package_runtime_binding.flash_attn_installed_extension_sha256",
    )
    if wheel_extension_hash != installed_extension_hash:
        raise ValueError("W1 FA2 runtime compatibility package/runtime extension mismatch")
    if bool(binding.get("flash_attn_extension_match")) is not True:
        raise ValueError("W1 FA2 runtime compatibility package/runtime extension mismatch")
    expected_version = str(binding.get("flash_attn_version_expected") or "")
    observed_version = str(binding.get("flash_attn_version_observed") or "")
    if not expected_version or observed_version != expected_version:
        raise ValueError("W1 FA2 runtime compatibility package/runtime version mismatch")
    if bool(binding.get("flash_attn_version_match")) is not True:
        raise ValueError("W1 FA2 runtime compatibility package/runtime version mismatch")
    if str(runtime_environment.get("flash_attn_version") or "") != observed_version:
        raise ValueError("W1 FA2 runtime compatibility runtime version binding mismatch")
    if runtime_environment.get("flash_attn_extension_file_hash") != installed_extension_hash:
        raise ValueError("W1 FA2 runtime compatibility runtime extension binding mismatch")
    wheel_filename = str(binding.get("flash_attn_wheel_filename") or "")
    if not wheel_filename.startswith("flash_attn-") or not wheel_filename.endswith(".whl"):
        raise ValueError("W1 FA2 runtime compatibility package/runtime wheel filename mismatch")
    if package_candidate_manifest is not None:
        validate_w1_fa2_package_candidate_manifest(package_candidate_manifest)
        if package_hash != package_candidate_manifest["manifest_hash"]:
            raise ValueError("W1 FA2 runtime compatibility package/runtime manifest mismatch")
        if wheel_filename != str(package_candidate_manifest.get("flash_attn_wheel_filename") or ""):
            raise ValueError("W1 FA2 runtime compatibility package/runtime wheel filename mismatch")
        if wheel_hash != str(package_candidate_manifest.get("flash_attn_wheel_sha256") or ""):
            raise ValueError("W1 FA2 runtime compatibility package/runtime wheel hash mismatch")
        if expected_version != str(package_candidate_manifest.get("flash_attn_version") or ""):
            raise ValueError("W1 FA2 runtime compatibility package/runtime version mismatch")
    validate_public_no_raw_payload(binding)


def validate_w1_fa2_runtime_compatibility_manifest(
    manifest: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any] | None = None,
    package_candidate_manifest: Mapping[str, Any] | None = None,
    expected_code_commit: str | None = None,
) -> None:
    if manifest.get("artifact_kind") != W1_FA2_RUNTIME_COMPATIBILITY_MANIFEST_KIND:
        raise ValueError("W1 FA2 runtime compatibility manifest kind mismatch")
    if int(manifest.get("schema_version") or 0) != 1:
        raise ValueError("W1 FA2 runtime compatibility manifest schema_version mismatch")
    if manifest.get("manifest_hash") != stable_json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    ):
        raise ValueError("W1 FA2 runtime compatibility manifest hash mismatch")
    if manifest.get("status") != "w1_fa2_runtime_compatibility_ready":
        raise ValueError("W1 FA2 runtime compatibility manifest status mismatch")
    if manifest.get("claims_scope") != "fa2_environment_load_and_forward_smoke_only":
        raise ValueError("W1 FA2 runtime compatibility claims_scope mismatch")
    code_commit = _require_git_commit(manifest.get("code_commit"), "code_commit")
    if expected_code_commit is not None and code_commit != str(expected_code_commit):
        raise ValueError("W1 FA2 runtime compatibility code_commit mismatch")
    if manifest.get("code_commit_verified") is not True:
        raise ValueError("W1 FA2 runtime compatibility code commit proof missing")
    if manifest.get("code_tree_clean") is not True:
        raise ValueError("W1 FA2 runtime compatibility clean code proof missing")
    if config_payload is not None:
        for field in ("config_hash", "selected_policy_hash", "selected_policy_report_hash", "source_metadata_hash"):
            if manifest.get(field) != config_payload[field]:
                raise ValueError(f"W1 FA2 runtime compatibility {field} mismatch")
        if manifest.get("w1_report_hash") != config_payload["window_mil_report_hash"]:
            raise ValueError("W1 FA2 runtime compatibility W1 report mismatch")
        if manifest.get("w1_model_file_sha256") != config_payload["window_mil_model_file_sha256"]:
            raise ValueError("W1 FA2 runtime compatibility W1 model mismatch")
        if manifest.get("window_mil_precision") != config_payload["window_mil_precision"]:
            raise ValueError("W1 FA2 runtime compatibility precision mismatch")
    if package_candidate_manifest is not None:
        validate_w1_fa2_package_candidate_manifest(package_candidate_manifest)
        if str(package_candidate_manifest.get("code_commit") or "") != code_commit:
            raise ValueError("W1 FA2 runtime compatibility package code_commit mismatch")
        if manifest.get("package_candidate_manifest_hash") != package_candidate_manifest["manifest_hash"]:
            raise ValueError("W1 FA2 runtime compatibility package manifest mismatch")
    else:
        _require_sha256(manifest.get("package_candidate_manifest_hash"), "package_candidate_manifest_hash")
    if normalize_w1_attn_implementation(
        str(manifest.get("w1_attn_implementation_requested") or "")
    ) != W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2:
        raise ValueError("W1 FA2 runtime compatibility requested backend mismatch")
    if normalize_w1_attn_implementation(
        str(manifest.get("w1_attn_implementation_observed") or "")
    ) != W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2:
        raise ValueError("W1 FA2 runtime compatibility observed backend mismatch")
    environment = manifest.get("runtime_environment")
    if not isinstance(environment, Mapping):
        raise ValueError("W1 FA2 runtime compatibility missing runtime environment")
    binding = manifest.get("package_runtime_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("W1 FA2 runtime compatibility package/runtime binding missing")
    smoke = manifest.get("synthetic_forward_smoke")
    if not isinstance(smoke, Mapping):
        raise ValueError("W1 FA2 runtime compatibility missing synthetic forward smoke")
    _validate_fa2_runtime_environment(environment)
    _validate_fa2_package_runtime_binding(
        binding,
        package_candidate_manifest=package_candidate_manifest,
        runtime_environment=environment,
    )
    _validate_fa2_synthetic_forward_smoke(smoke)
    if bool(manifest.get("dependency_resolution_enabled", True)):
        raise ValueError("W1 FA2 runtime compatibility dependency resolution must be disabled")
    if bool(manifest.get("source_build_detected", True)):
        raise ValueError("W1 FA2 runtime compatibility source build must not be used")
    if manifest.get("full_install_log_serialized") is not False:
        raise ValueError("W1 FA2 runtime compatibility must not serialize full install logs")
    _require_false_flags(manifest, PUBLIC_FALSE_FLAGS, "W1 FA2 runtime compatibility manifest")
    validate_public_no_raw_payload(manifest)


def _w1_fa2_runtime_compatibility_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_w1_fa2_runtime_compatibility_manifest(manifest)
    environment = dict(manifest.get("runtime_environment") or {})
    smoke = dict(manifest.get("synthetic_forward_smoke") or {})
    binding = dict(manifest.get("package_runtime_binding") or {})
    summary = {
        "artifact_kind": W1_FA2_RUNTIME_COMPATIBILITY_MANIFEST_KIND,
        "status": manifest["status"],
        "manifest_hash": manifest["manifest_hash"],
        "package_candidate_manifest_hash": manifest["package_candidate_manifest_hash"],
        "code_commit": str(manifest.get("code_commit") or ""),
        "code_commit_verified": manifest.get("code_commit_verified") is True,
        "code_tree_clean": manifest.get("code_tree_clean") is True,
        "w1_attn_implementation_requested": manifest["w1_attn_implementation_requested"],
        "w1_attn_implementation_observed": manifest["w1_attn_implementation_observed"],
        "torch_version": str(environment.get("torch_version") or ""),
        "torch_cuda_runtime": str(environment.get("torch_cuda_runtime") or ""),
        "device_type": str(environment.get("device_type") or ""),
        "gpu_name_hash": stable_json_hash(str(environment.get("gpu_name") or "")),
        "flash_attn_version": str(environment.get("flash_attn_version") or ""),
        "flash_attn_extension_file_hash": _require_sha256(
            environment.get("flash_attn_extension_file_hash"),
            "flash_attn_extension_file_hash",
        ),
        "package_runtime_binding_hash": stable_json_hash(binding),
        "flash_attn_version_match": binding.get("flash_attn_version_match") is True,
        "flash_attn_extension_match": binding.get("flash_attn_extension_match") is True,
        "synthetic_forward_smoke_ran": bool(smoke.get("ran")),
        "synthetic_forward_smoke_hash": stable_json_hash(smoke),
    }
    validate_public_no_raw_payload(summary)
    return summary


def _load_w1_fa2_runtime_compatibility_manifest(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    config_payload: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = (
        _load_json(manifest_or_path)
        if isinstance(manifest_or_path, str | Path)
        else dict(manifest_or_path)
    )
    if not isinstance(manifest, Mapping):
        raise ValueError("W1 FA2 runtime compatibility manifest must be an object")
    validate_w1_fa2_runtime_compatibility_manifest(manifest, config_payload=config_payload)
    return dict(manifest)


def _w1_score_run_for_comparison(
    *,
    run_dir: str | Path,
    config_payload: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    validate_private_filtering_run_dir(run_path)
    receipt_path = run_path / "w1_score_receipt.json"
    if not receipt_path.exists():
        raise ValueError(f"{label} W1 score receipt is missing")
    receipt = _load_json(receipt_path)
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{label} W1 score receipt must be an object")
    if receipt.get("stage_receipt_hash") != stable_json_hash(
        {key: value for key, value in receipt.items() if key != "stage_receipt_hash"}
    ):
        raise ValueError(f"{label} W1 score receipt hash mismatch")
    if receipt.get("artifact_kind") != W1_SCORE_STAGE_KIND:
        raise ValueError(f"{label} W1 score receipt kind mismatch")
    if receipt.get("phase") != "w1-score":
        raise ValueError(f"{label} W1 score receipt phase mismatch")
    for field in ("config_hash", "selected_policy_hash", "selected_policy_report_hash", "source_metadata_hash"):
        if receipt.get(field) != config_payload[field]:
            raise ValueError(f"{label} W1 score receipt {field} mismatch")
    w1_runtime_envelope, w1_runtime_envelope_hash = _validated_w1_runtime_envelope_from_receipt(
        receipt,
        label=f"{label} W1 score receipt",
    )
    ready_hash = receipt.get("w1_scores_hash")
    part_hash = receipt.get("w1_scores_part_hash")
    if ready_hash is not None:
        score_state = "ready"
        score_path = run_path / "w1_scores.jsonl.ready"
        expected_score_hash = _require_sha256(ready_hash, f"{label} w1_scores_hash")
    elif part_hash is not None:
        score_state = "part"
        score_path = run_path / "w1_scores.jsonl.part"
        expected_score_hash = _require_sha256(part_hash, f"{label} w1_scores_part_hash")
    else:
        raise ValueError(f"{label} W1 score receipt does not reference a score export")
    if not score_path.exists():
        raise ValueError(f"{label} W1 score export is missing")
    if _sha256_file(score_path) != expected_score_hash:
        raise ValueError(f"{label} W1 score export hash mismatch")
    score_rows = _load_jsonl_path(score_path, f"{label} W1 scores")
    survivor_rows = _load_ready_jsonl(run_path / "w1_survivor_queue.jsonl", f"{label} survivor queue")
    observed_prefilter_config_hash = (
        survivor_rows[0].get("config_hash") if survivor_rows else config_payload["config_hash"]
    )
    prefilter_config_payload = _prefilter_input_config_payload_for_hash(
        config_payload,
        observed_prefilter_config_hash,
        label=f"{label} survivor queue",
    )
    _validate_survivor_queue_rows(survivor_rows, prefilter_config_payload, allow_empty=False)
    survivor_by_id = {str(row["candidate_id_hash"]): row for row in survivor_rows}
    _validate_w1_score_rows(score_rows, config_payload, survivor_by_id=survivor_by_id, allow_empty=False)
    if score_rows and _w1_score_row_runtime_hashes(score_rows) != {w1_runtime_envelope_hash}:
        raise ValueError(f"{label} W1 score row runtime envelope mismatch")
    if int(receipt.get("scored_count", -1)) != len(score_rows):
        raise ValueError(f"{label} W1 score receipt scored_count mismatch")
    route_counts = _sorted_counter(Counter(str(row["w1_route"]) for row in score_rows))
    if dict(receipt.get("route_counts") or {}) != route_counts:
        raise ValueError(f"{label} W1 score receipt route_counts mismatch")
    _require_false_flags(receipt, PUBLIC_FALSE_FLAGS, f"{label} W1 score receipt")
    validate_public_no_raw_payload(receipt)
    return {
        "score_state": score_state,
        "score_export_hash": expected_score_hash,
        "receipt": dict(receipt),
        "score_rows": score_rows,
        "survivor_rows": survivor_rows,
        "w1_runtime_envelope": w1_runtime_envelope,
        "w1_runtime_envelope_hash": w1_runtime_envelope_hash,
    }


def _score_delta_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "max": None, "mean": None, "p50": None, "p95": None, "p99": None}
    ordered = sorted(float(value) for value in values)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
        return round(ordered[index], 10)

    return {
        "count": len(ordered),
        "max": round(ordered[-1], 10),
        "mean": round(sum(ordered) / len(ordered), 10),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def _w1_comparison_run_summary(run: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(run["receipt"])
    envelope = dict(run["w1_runtime_envelope"])
    telemetry = dict(receipt.get("w1_runtime_telemetry") or {})
    explicit_sync_observed_field = "score_cuda_synchronized_timing_observed" in telemetry
    explicit_sync_requested_field = "score_cuda_synchronized_timing_requested" in telemetry
    scored_count = int(receipt["scored_count"])
    docs_scored_this_invocation = int(receipt.get("docs_scored_this_invocation", 0))
    scored_existing_initial = int(telemetry.get("scored_existing_initial", 0))
    score_rows_inserted = int(telemetry.get("score_rows_inserted", 0))
    payload = {
        "status": receipt["status"],
        "stop_reason": receipt["stop_reason"],
        "score_export_state": run["score_state"],
        "stage_receipt_hash": receipt["stage_receipt_hash"],
        "w1_scores_export_hash": run["score_export_hash"],
        "scored_count": scored_count,
        "survivor_count": int(receipt["survivor_count"]),
        "docs_scored_this_invocation": docs_scored_this_invocation,
        "docs_per_second": float(receipt.get("docs_per_second", 0.0) or 0.0),
        "route_counts": dict(receipt.get("route_counts") or {}),
        "w1_runtime_envelope_hash": run["w1_runtime_envelope_hash"],
        "artifact_max_windows_per_device_batch": int(envelope.get("artifact_max_windows_per_device_batch", 0)),
        "requested_max_windows_per_device_batch": envelope.get("requested_max_windows_per_device_batch"),
        "effective_max_windows_per_device_batch": int(envelope.get("effective_max_windows_per_device_batch", 0)),
        "override_applied": bool(envelope.get("override_applied", False)),
        "window_mil_precision": str(envelope.get("window_mil_precision") or W1_PRECISION_FP32),
        "window_mil_precision_policy_hash": envelope.get("window_mil_precision_policy_hash"),
        "w1_attn_implementation_requested": normalize_w1_attn_implementation(
            str(envelope.get("w1_attn_implementation_requested") or W1_ATTN_IMPLEMENTATION_SDPA)
        ),
        "w1_attn_implementation_observed": normalize_w1_attn_implementation(
            str(
                envelope.get("w1_attn_implementation_observed")
                or envelope.get("w1_attn_implementation_requested")
                or W1_ATTN_IMPLEMENTATION_SDPA
            )
        ),
        "stage_batch_size": int(envelope.get("stage_batch_size", 0)),
        "score_cuda_synchronized_timing_requested_field_present": explicit_sync_requested_field,
        "score_cuda_synchronized_timing_observed_field_present": explicit_sync_observed_field,
        "score_cuda_synchronized_timing_requested": bool(telemetry.get("score_cuda_synchronized_timing_requested", False)),
        "score_cuda_synchronized_timing_observed": bool(telemetry.get("score_cuda_synchronized_timing_observed", False)),
        "score_cuda_synchronization_count": int(telemetry.get("score_cuda_synchronization_count", 0)),
        "scored_existing_initial": scored_existing_initial,
        "pending_initial": int(telemetry.get("pending_initial", 0)),
        "score_rows_inserted": score_rows_inserted,
        "fresh_scoring_invocation": (
            str(receipt["status"]) == "w1_scored"
            and run["score_state"] == "ready"
            and scored_existing_initial == 0
            and docs_scored_this_invocation == scored_count
            and score_rows_inserted == scored_count
        ),
        "score_oom_failed_attempt_count": int(telemetry.get("score_oom_failed_attempt_count", 0)),
        "score_tokenized_window_count": int(telemetry.get("score_tokenized_window_count", 0)),
        "score_window_forward_call_count": int(telemetry.get("score_window_forward_call_count", 0)),
        "score_window_forward_window_count": int(telemetry.get("score_window_forward_window_count", 0)),
        "score_max_windows_per_forward_call": int(telemetry.get("score_max_windows_per_forward_call", 0)),
        "score_model_docs_per_second": float(telemetry.get("score_model_docs_per_second", 0.0) or 0.0),
        "score_window_forward_windows_per_second": float(
            telemetry.get("score_window_forward_windows_per_second", 0.0) or 0.0
        ),
    }
    validate_public_no_raw_payload(payload)
    return payload


def _w1_scored_survivor_binding_hash(run: Mapping[str, Any]) -> str:
    survivor_by_id = {str(row["candidate_id_hash"]): row for row in run["survivor_rows"]}
    bindings: list[dict[str, str]] = []
    for row in sorted(run["score_rows"], key=lambda item: str(item["candidate_id_hash"])):
        candidate_id = str(row["candidate_id_hash"])
        survivor = survivor_by_id.get(candidate_id)
        if survivor is None:
            raise ValueError("W1 comparison score row missing survivor row")
        bindings.append(
            {
                "candidate_id_hash": candidate_id,
                "private_evidence_row_id": _require_sha256(
                    survivor.get("private_evidence_row_id"),
                    "private_evidence_row_id",
                ),
                "survivor_stage_receipt_hash": _require_sha256(
                    row.get("survivor_stage_receipt_hash"),
                    "survivor_stage_receipt_hash",
                ),
            }
        )
    return stable_json_hash(bindings)


def compare_w1_runtime_cap_score_runs(
    *,
    config_payload: Mapping[str, Any],
    baseline_run_dir: str | Path,
    comparison_run_dir: str | Path,
    report_path: str | Path | None = None,
    threshold_band_epsilon: float = 1e-4,
    max_allowed_score_delta: float = 1e-6,
    require_synchronized_cuda: bool = True,
    require_cap_exercised: bool = True,
    require_baseline_artifact_default_cap: bool = True,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    threshold_band = _nonnegative_float(threshold_band_epsilon, "threshold_band_epsilon")
    allowed_delta = _nonnegative_float(max_allowed_score_delta, "max_allowed_score_delta")
    baseline = _w1_score_run_for_comparison(
        run_dir=baseline_run_dir,
        config_payload=config_payload,
        label="baseline",
    )
    comparison = _w1_score_run_for_comparison(
        run_dir=comparison_run_dir,
        config_payload=config_payload,
        label="comparison",
    )
    baseline_rows_by_id = {str(row["candidate_id_hash"]): row for row in baseline["score_rows"]}
    comparison_rows_by_id = {str(row["candidate_id_hash"]): row for row in comparison["score_rows"]}
    baseline_ids = set(baseline_rows_by_id)
    comparison_ids = set(comparison_rows_by_id)
    shared_ids = sorted(baseline_ids.intersection(comparison_ids))
    candidate_set_equal = baseline_ids == comparison_ids
    route_mismatch_count = sum(
        1
        for candidate_id in shared_ids
        if baseline_rows_by_id[candidate_id].get("w1_route") != comparison_rows_by_id[candidate_id].get("w1_route")
    )
    score_delta_values: list[float] = []
    threshold_band_count = 0
    missing_score_pair_count = 0
    threshold = float(config_payload["window_mil_threshold"])
    for candidate_id in shared_ids:
        baseline_score = _safe_runtime_score(baseline_rows_by_id[candidate_id].get("w1_score"))
        comparison_score = _safe_runtime_score(comparison_rows_by_id[candidate_id].get("w1_score"))
        if baseline_score is None or comparison_score is None:
            missing_score_pair_count += 1
            continue
        score_delta_values.append(abs(comparison_score - baseline_score))
        if abs(baseline_score - threshold) <= threshold_band or abs(comparison_score - threshold) <= threshold_band:
            threshold_band_count += 1
    score_delta_summary = _score_delta_summary(score_delta_values)
    max_delta = score_delta_summary["max"]
    score_delta_within_tolerance = max_delta is not None and float(max_delta) <= allowed_delta
    baseline_summary = _w1_comparison_run_summary(baseline)
    comparison_summary = _w1_comparison_run_summary(comparison)
    baseline_complete_ready = baseline_summary["status"] == "w1_scored" and baseline_summary["score_export_state"] == "ready"
    comparison_complete_ready = (
        comparison_summary["status"] == "w1_scored" and comparison_summary["score_export_state"] == "ready"
    )
    baseline_fresh = bool(baseline_summary["fresh_scoring_invocation"])
    comparison_fresh = bool(comparison_summary["fresh_scoring_invocation"])
    baseline_survivor_binding_hash = _w1_scored_survivor_binding_hash(baseline)
    comparison_survivor_binding_hash = _w1_scored_survivor_binding_hash(comparison)
    same_survivor_inputs = (
        candidate_set_equal
        and baseline_survivor_binding_hash == comparison_survivor_binding_hash
    )
    baseline_uses_artifact_default_cap = (
        not bool(baseline_summary["override_applied"])
        and baseline_summary["requested_max_windows_per_device_batch"] is None
        and int(baseline_summary["effective_max_windows_per_device_batch"])
        == int(baseline_summary["artifact_max_windows_per_device_batch"])
    )
    baseline_effective_cap = int(baseline_summary["effective_max_windows_per_device_batch"])
    comparison_effective_cap = int(comparison_summary["effective_max_windows_per_device_batch"])
    comparison_forward_max = int(comparison_summary["score_max_windows_per_forward_call"])
    comparison_requested_cap = comparison_summary["requested_max_windows_per_device_batch"]
    cap_override_exceeded_default_cap = (
        comparison_effective_cap > baseline_effective_cap
        and bool(comparison_summary["override_applied"])
        and comparison_forward_max > baseline_effective_cap
    )
    requested_cap_exercised = (
        comparison_requested_cap is not None
        and comparison_forward_max >= int(comparison_requested_cap)
    )
    cap_override_exercised = cap_override_exceeded_default_cap and requested_cap_exercised
    baseline_sync_observed = (
        bool(baseline_summary["score_cuda_synchronized_timing_observed_field_present"])
        and bool(baseline_summary["score_cuda_synchronized_timing_observed"])
        and int(baseline_summary["score_cuda_synchronization_count"]) > 0
    )
    comparison_sync_observed = (
        bool(comparison_summary["score_cuda_synchronized_timing_observed_field_present"])
        and bool(comparison_summary["score_cuda_synchronized_timing_observed"])
        and int(comparison_summary["score_cuda_synchronization_count"]) > 0
    )
    synchronized_cuda_timing_observed = baseline_sync_observed and comparison_sync_observed
    oom_bound_diagnostic = (
        int(baseline_summary["score_oom_failed_attempt_count"]) > 0
        or int(comparison_summary["score_oom_failed_attempt_count"]) > 0
    )
    blocking_reasons: list[str] = []
    if not baseline_complete_ready:
        blocking_reasons.append("baseline_not_complete_ready")
    if not comparison_complete_ready:
        blocking_reasons.append("comparison_not_complete_ready")
    if not baseline_fresh:
        blocking_reasons.append("baseline_not_fresh_scoring_invocation")
    if not comparison_fresh:
        blocking_reasons.append("comparison_not_fresh_scoring_invocation")
    if require_baseline_artifact_default_cap and not baseline_uses_artifact_default_cap:
        blocking_reasons.append("baseline_not_artifact_default_cap")
    if not candidate_set_equal:
        blocking_reasons.append("candidate_sets_differ")
    if not same_survivor_inputs:
        blocking_reasons.append("survivor_inputs_differ")
    if not shared_ids:
        blocking_reasons.append("no_shared_candidates")
    if route_mismatch_count:
        blocking_reasons.append("route_mismatch")
    if missing_score_pair_count:
        blocking_reasons.append("missing_score_pair")
    if not score_delta_within_tolerance:
        blocking_reasons.append("score_delta_exceeds_tolerance")
    if require_synchronized_cuda and not synchronized_cuda_timing_observed:
        blocking_reasons.append("synchronized_cuda_timing_not_observed")
    if require_cap_exercised and not cap_override_exercised:
        blocking_reasons.append("requested_cap_not_exercised")
    if oom_bound_diagnostic:
        blocking_reasons.append("oom_backoff_attempts_present")
    throughput_comparison_ready = not blocking_reasons
    baseline_docs_per_second = float(baseline_summary["docs_per_second"])
    comparison_docs_per_second = float(comparison_summary["docs_per_second"])
    payload = {
        "artifact_kind": W1_RUNTIME_CAP_COMPARISON_REPORT_KIND,
        "schema_version": 1,
        "status": (
            "w1_runtime_cap_comparison_ready"
            if throughput_comparison_ready
            else "w1_runtime_cap_comparison_blocked"
        ),
        "claims_scope": "runtime_override_equivalence_diagnostic_only",
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "w1_report_hash": config_payload["window_mil_report_hash"],
        "w1_model_file_sha256": config_payload["window_mil_model_file_sha256"],
        "w1_threshold": _threshold(config_payload["window_mil_threshold"], "w1_threshold"),
        "threshold_band_epsilon": round(threshold_band, 10),
        "max_allowed_score_delta": round(allowed_delta, 10),
        "require_synchronized_cuda": bool(require_synchronized_cuda),
        "require_cap_exercised": bool(require_cap_exercised),
        "require_baseline_artifact_default_cap": bool(require_baseline_artifact_default_cap),
        "candidate_set_equal": bool(candidate_set_equal),
        "same_survivor_inputs": bool(same_survivor_inputs),
        "candidate_count": len(shared_ids) if candidate_set_equal else None,
        "baseline_candidate_count": len(baseline_ids),
        "comparison_candidate_count": len(comparison_ids),
        "baseline_candidate_set_hash": stable_json_hash(sorted(baseline_ids)),
        "comparison_candidate_set_hash": stable_json_hash(sorted(comparison_ids)),
        "baseline_scored_survivor_binding_hash": baseline_survivor_binding_hash,
        "comparison_scored_survivor_binding_hash": comparison_survivor_binding_hash,
        "complete_ready_exports": bool(baseline_complete_ready and comparison_complete_ready),
        "fresh_scoring_invocations": bool(baseline_fresh and comparison_fresh),
        "baseline_artifact_default_cap": bool(baseline_uses_artifact_default_cap),
        "compared_score_count": len(shared_ids),
        "route_mismatch_count": int(route_mismatch_count),
        "missing_score_pair_count": int(missing_score_pair_count),
        "score_delta_summary": score_delta_summary,
        "score_delta_within_tolerance": bool(score_delta_within_tolerance),
        "threshold_band_count": int(threshold_band_count),
        "route_equivalent": bool(candidate_set_equal and route_mismatch_count == 0),
        "score_equivalent": bool(
            candidate_set_equal
            and route_mismatch_count == 0
            and missing_score_pair_count == 0
            and score_delta_within_tolerance
        ),
        "baseline": baseline_summary,
        "comparison": comparison_summary,
        "cap_override_exceeded_default_cap": bool(cap_override_exceeded_default_cap),
        "requested_cap_exercised": bool(requested_cap_exercised),
        "cap_override_exercised": bool(cap_override_exercised),
        "synchronized_cuda_timing_observed": bool(synchronized_cuda_timing_observed),
        "oom_bound_diagnostic": bool(oom_bound_diagnostic),
        "docs_per_second_ratio": _w1_rate(comparison_docs_per_second, baseline_docs_per_second),
        "throughput_comparison_ready": bool(throughput_comparison_ready),
        "blocking_reasons": blocking_reasons,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["report_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "report_hash"})
    validate_w1_runtime_cap_comparison_report(payload, config_payload=config_payload)
    if report_path is not None:
        _write_json_atomic(report_path, payload)
    return payload


def compare_w1_score_runs(
    *,
    config_payload: Mapping[str, Any],
    baseline_run_dir: str | Path,
    comparison_run_dir: str | Path,
    report_path: str | Path | None = None,
    threshold_band_epsilon: float = 1e-4,
    max_allowed_score_delta: float = 1e-6,
    require_synchronized_cuda: bool = True,
    comparison_backend_runtime_compatibility_manifest: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    base = compare_w1_runtime_cap_score_runs(
        config_payload=config_payload,
        baseline_run_dir=baseline_run_dir,
        comparison_run_dir=comparison_run_dir,
        report_path=None,
        threshold_band_epsilon=threshold_band_epsilon,
        max_allowed_score_delta=max_allowed_score_delta,
        require_synchronized_cuda=require_synchronized_cuda,
        require_cap_exercised=False,
        require_baseline_artifact_default_cap=False,
    )
    blocking_reasons = list(base.get("blocking_reasons") or [])
    comparison_summary = base.get("comparison")
    if not isinstance(comparison_summary, Mapping):
        blocking_reasons.append("comparison_summary_missing")
        comparison_backend = W1_ATTN_IMPLEMENTATION_SDPA
    else:
        comparison_backend = normalize_w1_attn_implementation(
            str(
                comparison_summary.get("w1_attn_implementation_observed")
                or comparison_summary.get("w1_attn_implementation_requested")
                or W1_ATTN_IMPLEMENTATION_SDPA
            )
        )
    compatibility_summary: dict[str, Any] | None = None
    compatibility_required = comparison_backend != W1_ATTN_IMPLEMENTATION_SDPA
    if compatibility_required:
        if comparison_backend_runtime_compatibility_manifest is None:
            blocking_reasons.append("missing_backend_runtime_compatibility_manifest")
        else:
            try:
                compatibility_manifest = _load_w1_fa2_runtime_compatibility_manifest(
                    comparison_backend_runtime_compatibility_manifest,
                    config_payload=config_payload,
                )
                compatibility_summary = _w1_fa2_runtime_compatibility_summary(compatibility_manifest)
                observed = normalize_w1_attn_implementation(
                    str(compatibility_summary["w1_attn_implementation_observed"])
                )
                if observed != comparison_backend:
                    blocking_reasons.append("backend_runtime_compatibility_backend_mismatch")
            except ValueError:
                blocking_reasons.append("backend_runtime_compatibility_manifest_invalid")
    score_stability_ready = not blocking_reasons
    payload = dict(base)
    payload.pop("report_hash", None)
    payload.update(
        {
            "artifact_kind": W1_SCORE_RUN_COMPARISON_REPORT_KIND,
            "status": (
                "w1_score_run_comparison_ready"
                if score_stability_ready
                else "w1_score_run_comparison_blocked"
            ),
            "claims_scope": "score_run_equivalence_diagnostic_only",
            "score_stability_ready": bool(score_stability_ready),
            "throughput_comparison_ready": bool(score_stability_ready),
            "comparison_backend_runtime_compatibility_required": bool(compatibility_required),
            "comparison_backend_runtime_compatibility": compatibility_summary,
            "blocking_reasons": blocking_reasons,
        }
    )
    payload["report_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "report_hash"})
    validate_w1_score_run_comparison_report(payload, config_payload=config_payload)
    if report_path is not None:
        _write_json_atomic(report_path, payload)
    return payload


def validate_w1_runtime_cap_comparison_report(
    report: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any] | None = None,
) -> None:
    if report.get("artifact_kind") != W1_RUNTIME_CAP_COMPARISON_REPORT_KIND:
        raise ValueError("W1 runtime cap comparison report kind mismatch")
    if int(report.get("schema_version") or 0) != 1:
        raise ValueError("W1 runtime cap comparison report schema_version mismatch")
    if report.get("report_hash") != stable_json_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    ):
        raise ValueError("W1 runtime cap comparison report hash mismatch")
    if report.get("claims_scope") != "runtime_override_equivalence_diagnostic_only":
        raise ValueError("W1 runtime cap comparison claims_scope mismatch")
    if config_payload is not None:
        for field in ("config_hash", "selected_policy_hash", "selected_policy_report_hash", "source_metadata_hash"):
            if report.get(field) != config_payload[field]:
                raise ValueError(f"W1 runtime cap comparison {field} mismatch")
        if report.get("w1_report_hash") != config_payload["window_mil_report_hash"]:
            raise ValueError("W1 runtime cap comparison W1 report mismatch")
        if report.get("w1_model_file_sha256") != config_payload["window_mil_model_file_sha256"]:
            raise ValueError("W1 runtime cap comparison W1 model mismatch")
    blocking_reasons = list(report.get("blocking_reasons") or [])
    status = str(report.get("status") or "")
    ready = bool(report.get("throughput_comparison_ready", False))
    if status == "w1_runtime_cap_comparison_ready":
        if blocking_reasons or not ready:
            raise ValueError("ready W1 runtime cap comparison must have no blocking reasons")
        if not report.get("candidate_set_equal") or int(report.get("route_mismatch_count", -1)) != 0:
            raise ValueError("ready W1 runtime cap comparison must prove route equivalence")
        if not report.get("same_survivor_inputs"):
            raise ValueError("ready W1 runtime cap comparison must prove survivor input equivalence")
        if not report.get("complete_ready_exports") or not report.get("fresh_scoring_invocations"):
            raise ValueError("ready W1 runtime cap comparison must use complete fresh W1 runs")
        if bool(report.get("require_baseline_artifact_default_cap", True)) and not report.get(
            "baseline_artifact_default_cap"
        ):
            raise ValueError("ready W1 runtime cap comparison must use the artifact-default baseline cap")
        if not report.get("score_equivalent"):
            raise ValueError("ready W1 runtime cap comparison must prove score equivalence")
        if bool(report.get("require_synchronized_cuda")) and not report.get("synchronized_cuda_timing_observed"):
            raise ValueError("ready W1 runtime cap comparison must observe synchronized CUDA timing")
        if bool(report.get("require_cap_exercised")) and not report.get("cap_override_exercised"):
            raise ValueError("ready W1 runtime cap comparison must exercise the cap override")
    elif status == "w1_runtime_cap_comparison_blocked":
        if ready or not blocking_reasons:
            raise ValueError("blocked W1 runtime cap comparison must record blocking reasons")
    else:
        raise ValueError("W1 runtime cap comparison status mismatch")
    _require_false_flags(report, PUBLIC_FALSE_FLAGS, "W1 runtime cap comparison report")
    validate_public_no_raw_payload(report)


def validate_w1_score_run_comparison_report(
    report: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any] | None = None,
) -> None:
    if report.get("artifact_kind") != W1_SCORE_RUN_COMPARISON_REPORT_KIND:
        raise ValueError("W1 score run comparison report kind mismatch")
    if int(report.get("schema_version") or 0) != 1:
        raise ValueError("W1 score run comparison report schema_version mismatch")
    if report.get("report_hash") != stable_json_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    ):
        raise ValueError("W1 score run comparison report hash mismatch")
    if report.get("claims_scope") != "score_run_equivalence_diagnostic_only":
        raise ValueError("W1 score run comparison claims_scope mismatch")
    if config_payload is not None:
        for field in ("config_hash", "selected_policy_hash", "selected_policy_report_hash", "source_metadata_hash"):
            if report.get(field) != config_payload[field]:
                raise ValueError(f"W1 score run comparison {field} mismatch")
        if report.get("w1_report_hash") != config_payload["window_mil_report_hash"]:
            raise ValueError("W1 score run comparison W1 report mismatch")
        if report.get("w1_model_file_sha256") != config_payload["window_mil_model_file_sha256"]:
            raise ValueError("W1 score run comparison W1 model mismatch")
    blocking_reasons = list(report.get("blocking_reasons") or [])
    status = str(report.get("status") or "")
    ready = bool(report.get("score_stability_ready", False))
    comparison = report.get("comparison")
    comparison_backend = W1_ATTN_IMPLEMENTATION_SDPA
    if isinstance(comparison, Mapping):
        comparison_backend = normalize_w1_attn_implementation(
            str(
                comparison.get("w1_attn_implementation_observed")
                or comparison.get("w1_attn_implementation_requested")
                or W1_ATTN_IMPLEMENTATION_SDPA
            )
        )
    compatibility_required = bool(report.get("comparison_backend_runtime_compatibility_required", False))
    if comparison_backend != W1_ATTN_IMPLEMENTATION_SDPA and not compatibility_required:
        raise ValueError("W1 score run comparison missing backend runtime compatibility requirement")
    compatibility = report.get("comparison_backend_runtime_compatibility")
    if compatibility_required:
        if not isinstance(compatibility, Mapping):
            if status == "w1_score_run_comparison_ready":
                raise ValueError("ready W1 score run comparison missing backend runtime compatibility")
        elif compatibility.get("artifact_kind") != W1_FA2_RUNTIME_COMPATIBILITY_MANIFEST_KIND:
            raise ValueError("W1 score run comparison backend compatibility kind mismatch")
        else:
            _require_sha256(compatibility.get("manifest_hash"), "backend runtime compatibility manifest_hash")
            if normalize_w1_attn_implementation(
                str(compatibility.get("w1_attn_implementation_observed") or "")
            ) != comparison_backend:
                raise ValueError("W1 score run comparison backend compatibility mismatch")
    if status == "w1_score_run_comparison_ready":
        if blocking_reasons or not ready:
            raise ValueError("ready W1 score run comparison must have no blocking reasons")
        if not report.get("candidate_set_equal") or int(report.get("route_mismatch_count", -1)) != 0:
            raise ValueError("ready W1 score run comparison must prove route equivalence")
        if not report.get("same_survivor_inputs"):
            raise ValueError("ready W1 score run comparison must prove survivor input equivalence")
        if not report.get("complete_ready_exports") or not report.get("fresh_scoring_invocations"):
            raise ValueError("ready W1 score run comparison must use complete fresh W1 runs")
        if not report.get("score_equivalent"):
            raise ValueError("ready W1 score run comparison must prove score equivalence")
        if bool(report.get("require_synchronized_cuda")) and not report.get("synchronized_cuda_timing_observed"):
            raise ValueError("ready W1 score run comparison must observe synchronized CUDA timing")
        if compatibility_required and not isinstance(compatibility, Mapping):
            raise ValueError("ready W1 score run comparison must include backend runtime compatibility")
    elif status == "w1_score_run_comparison_blocked":
        if ready or not blocking_reasons:
            raise ValueError("blocked W1 score run comparison must record blocking reasons")
    else:
        raise ValueError("W1 score run comparison status mismatch")
    _require_false_flags(report, PUBLIC_FALSE_FLAGS, "W1 score run comparison report")
    validate_public_no_raw_payload(report)


def assemble_sharded_filtering_stage_report(
    *,
    config_payload: Mapping[str, Any],
    handoff: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    _validate_sharded_w1_handoff_payload(handoff, config_payload=config_payload, supervisor_run_dir=supervisor_run_dir)
    route_counts: Counter[str] = Counter()
    shard_report_hashes: list[str] = []
    w1_runtime_envelope_hashes: set[str] = set()
    w1_runtime_envelope: dict[str, Any] | None = None
    candidate_count = 0
    survivor_count = 0
    scored_count = 0
    for shard in sorted(handoff["shards"], key=lambda item: int(item["shard_index"])):
        shard_report = assemble_filtering_stage_report(
            config_payload=config_payload,
            run_dir=_prefilter_supervisor_shard_dir(supervisor_run_dir, shard),
            expected_prefilter_status=_w1_start_prefilter_expected_status_for_run_dir(
                _prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
            ),
        )
        route_counts.update({str(key): int(value) for key, value in dict(shard_report["route_counts"]).items()})
        shard_report_hashes.append(str(shard_report["report_hash"]))
        shard_envelope_hash = _require_sha256(
            shard_report.get("w1_runtime_envelope_hash"),
            "w1_runtime_envelope_hash",
        )
        w1_runtime_envelope_hashes.add(shard_envelope_hash)
        if w1_runtime_envelope is None:
            shard_envelope = shard_report.get("w1_runtime_envelope")
            if not isinstance(shard_envelope, Mapping):
                raise ValueError("sharded assembled report missing W1 runtime envelope")
            w1_runtime_envelope = dict(shard_envelope)
        candidate_count += int(shard_report["candidate_count"])
        survivor_count += int(shard_report["survivor_count"])
        scored_count += int(shard_report["w1_scored_count"])
    if len(w1_runtime_envelope_hashes) != 1:
        raise ValueError("sharded assembled report W1 runtime envelope mismatch")
    if candidate_count != int(handoff["candidate_count"]):
        raise ValueError("sharded assembled report candidate count mismatch")
    if survivor_count != int(handoff["survivor_count"]) or scored_count != int(handoff["survivor_count"]):
        raise ValueError("sharded assembled report survivor/W1 score count mismatch")
    payload = {
        "artifact_kind": ASSEMBLED_STAGE_REPORT_KIND,
        "schema_version": 1,
        "status": "assembled_ready_for_review",
        "stop_reason": "assembled_sharded_complete",
        "next_required_phase": "coordinator_review_before_any_accepted_manifest_or_rental_scale",
        "assembly_input_mode": "sharded_w1_handoff",
        "sharded_w1_handoff_hash": handoff["handoff_hash"],
        "config_hash": config_payload["config_hash"],
        "prefilter_producer_config_hash": handoff["config_hash"],
        "w1_consumer_config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "deterministic_filter_policy_hash": config_payload.get("deterministic_filter_policy_hash"),
        "candidate_count": int(candidate_count),
        "survivor_count": int(survivor_count),
        "w1_scored_count": int(scored_count),
        "survivor_w1_set_equality_validated": True,
        "w1_runtime_envelope": w1_runtime_envelope,
        "w1_runtime_envelope_hash": next(iter(w1_runtime_envelope_hashes)),
        "route_counts": _sorted_counter(route_counts),
        "shard_count": int(handoff["shard_count"]),
        "shard_report_hashes": shard_report_hashes,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["report_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "report_hash"})
    validate_public_no_raw_payload(payload)
    if report_path is not None:
        _write_json_atomic(report_path, payload)
    return payload


def _insert_filtering_row(
    conn: sqlite3.Connection,
    *,
    table: str,
    key_column: str,
    key_value: str,
    json_column: str,
    columns: Sequence[str],
    values: Sequence[Any],
) -> None:
    existing = conn.execute(
        f"select {json_column} from {table} where {key_column} = ?",
        (key_value,),
    ).fetchone()
    expected_json = str(values[tuple(columns).index(json_column)])
    if existing is not None:
        if str(existing[json_column]) != expected_json:
            raise ValueError(f"conflicting {table} row for existing id")
        return
    placeholders = ", ".join("?" for _ in columns)
    conn.execute(
        f"insert into {table}({', '.join(columns)}) values ({placeholders})",
        tuple(values),
    )


def _stage_state_path(run_dir: Path) -> Path:
    return run_dir / "filtering_stage_state.sqlite"


def _open_w1_stage_state_readonly(
    run_dir: Path,
    *,
    prefilter_config_payload: Mapping[str, Any],
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    state_path = _stage_state_path(run_dir)
    if not state_path.exists():
        raise ValueError("filtering stage state is missing")
    conn = sqlite3.connect(
        f"file:{state_path}?mode=ro",
        uri=True,
        timeout=60.0,
        check_same_thread=bool(check_same_thread),
    )
    conn.row_factory = sqlite3.Row
    conn.execute("pragma busy_timeout=60000")
    conn.execute("pragma temp_store=MEMORY")
    conn.execute("pragma cache_size=-262144")
    existing = conn.execute("select value_json from stage_metadata where key = 'config_hash'").fetchone()
    if existing is None:
        conn.close()
        raise ValueError("filtering stage state config hash missing")
    stored = json.loads(str(existing["value_json"]))
    if stored != prefilter_config_payload["config_hash"]:
        conn.close()
        raise ValueError("filtering stage state config hash mismatch")
    return conn


def _open_existing_stage_state(run_dir: Path, config_payload: Mapping[str, Any]) -> sqlite3.Connection:
    state_path = _stage_state_path(run_dir)
    if not state_path.exists():
        raise ValueError("filtering stage state is missing")
    conn = sqlite3.connect(f"file:{state_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    existing = conn.execute("select value_json from stage_metadata where key = 'config_hash'").fetchone()
    if existing is None:
        conn.close()
        raise ValueError("filtering stage state config hash missing")
    stored = json.loads(str(existing["value_json"]))
    if stored != config_payload["config_hash"]:
        conn.close()
        raise ValueError("filtering stage state config hash mismatch")
    return conn


def _open_w1_stage_state(
    run_dir: Path,
    *,
    config_payload: Mapping[str, Any],
    prefilter_config_payload: Mapping[str, Any],
    exclusive_locking: bool = True,
) -> sqlite3.Connection:
    _ensure_private_dir(run_dir)
    state_path = _stage_state_path(run_dir)
    if not state_path.exists():
        raise ValueError("filtering stage state is missing")
    conn = sqlite3.connect(state_path)
    _chmod_private_sqlite_files(state_path)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode=wal")
    conn.execute("pragma synchronous=normal")
    conn.execute("pragma wal_autocheckpoint=16384")
    conn.execute("pragma temp_store=MEMORY")
    conn.execute("pragma cache_size=-262144")
    if exclusive_locking:
        conn.execute("pragma locking_mode=EXCLUSIVE")
    else:
        conn.execute("pragma locking_mode=NORMAL")
    _create_stage_state_schema(conn)
    existing = conn.execute("select value_json from stage_metadata where key = 'config_hash'").fetchone()
    if existing is None:
        conn.close()
        raise ValueError("filtering stage state config hash missing")
    stored = json.loads(str(existing["value_json"]))
    if stored != prefilter_config_payload["config_hash"]:
        conn.close()
        raise ValueError("filtering stage state config hash mismatch")
    with conn:
        conn.execute(
            "insert or replace into stage_metadata(key, value_json) values (?, ?)",
            ("w1_config_hash", json.dumps(config_payload["config_hash"])),
        )
        conn.execute(
            "insert or replace into stage_metadata(key, value_json) values (?, ?)",
            ("w1_config_payload", json.dumps(config_payload, sort_keys=True)),
        )
    _chmod_private_sqlite_files(state_path)
    return conn


def _open_stage_state(run_dir: Path, config_payload: Mapping[str, Any]) -> sqlite3.Connection:
    _ensure_private_dir(run_dir)
    state_path = _stage_state_path(run_dir)
    conn = sqlite3.connect(state_path)
    _chmod_private_sqlite_files(state_path)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode=wal")
    conn.execute("pragma synchronous=normal")
    conn.execute("pragma wal_autocheckpoint=16384")
    conn.execute("pragma temp_store=MEMORY")
    conn.execute("pragma cache_size=-262144")
    conn.execute("pragma locking_mode=EXCLUSIVE")
    _chmod_private_sqlite_files(state_path)
    _create_stage_state_schema(conn)
    existing = conn.execute("select value_json from stage_metadata where key = 'config_hash'").fetchone()
    if existing is not None:
        stored = json.loads(str(existing["value_json"]))
        if stored != config_payload["config_hash"]:
            conn.close()
            raise ValueError("filtering stage state config hash mismatch")
    with conn:
        conn.execute(
            "insert or replace into stage_metadata(key, value_json) values (?, ?)",
            ("config_hash", json.dumps(config_payload["config_hash"])),
        )
        conn.execute(
            "insert or replace into stage_metadata(key, value_json) values (?, ?)",
            ("config_payload", json.dumps(config_payload, sort_keys=True)),
        )
    _chmod_private_sqlite_files(state_path)
    return conn


def _create_stage_state_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.executescript(
            """
            create table if not exists stage_metadata(
                key text primary key,
                value_json text not null
            );
            create table if not exists prefilter_source_rows(
                physical_row_key_hash text primary key,
                source_schedule_hash text,
                schedule_entry_hash text not null,
                source_file_id_hash text not null,
                row_group_index integer not null,
                row_group_local_offset integer not null,
                source_document_id_hash text not null,
                row_status text not null,
                error_class text,
                segmentation_receipt_hash text,
                candidate_count integer not null default 0,
                survivor_count integer not null default 0,
                row_receipt_hash text not null,
                created_at real not null,
                updated_at real not null
            );
            create table if not exists prefilter_candidates(
                candidate_id_hash text primary key,
                source_document_id_hash text not null,
                segmentation_receipt_hash text not null,
                candidate_json text not null,
                deterministic_result_json text not null,
                prefilter_route text not null,
                s1_score real,
                created_at real not null
            );
            create table if not exists private_w1_evidence(
                private_evidence_row_id text primary key,
                candidate_id_hash text not null unique,
                evidence_text_sha256 text not null,
                private_evidence_text text not null
            );
            create table if not exists w1_scores(
                candidate_id_hash text primary key,
                score_json text not null,
                created_at real not null
            );
            """
        )


PREFILTER_TERMINAL_ROW_STATUSES = frozenset(
    {
        "prefiltered",
        "segmentation_error",
        "fetch_error",
        "non_mapping_row",
        "missing_text",
    }
)


def _prefilter_source_row_payload(
    *,
    entry: Mapping[str, Any],
    offset: int,
    row_status: str,
    error_class: str | None,
    segmentation_receipt_hash: str | None,
    candidate_count: int,
    survivor_count: int,
) -> dict[str, Any]:
    payload = {
        "artifact_kind": "twc_chunk_filter_prefilter_source_row_receipt_v1",
        "schema_version": 1,
        "physical_row_key_hash": _physical_row_key(entry, offset),
        "source_schedule_hash": entry.get("source_schedule_hash"),
        "schedule_entry_hash": entry.get("entry_hash"),
        "source_file_id_hash": stable_json_hash(str(entry.get("source_file_id"))),
        "row_group_index": int(entry.get("row_group_index")),
        "row_group_local_offset": int(offset),
        "source_document_id_hash": _source_document_id_for_schedule_row(entry, offset),
        "row_status": row_status,
        "error_class": error_class,
        "segmentation_receipt_hash": segmentation_receipt_hash,
        "candidate_count": int(candidate_count),
        "survivor_count": int(survivor_count),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["row_receipt_hash"] = stable_json_hash(payload)
    validate_public_no_raw_payload(payload)
    return payload


def _bulk_upsert_prefilter_source_rows(conn: sqlite3.Connection, payloads: Sequence[Mapping[str, Any]]) -> None:
    if not payloads:
        return
    now = time.time()
    conn.executemany(
        """
        insert into prefilter_source_rows(
            physical_row_key_hash, source_schedule_hash, schedule_entry_hash, source_file_id_hash,
            row_group_index, row_group_local_offset, source_document_id_hash, row_status, error_class,
            segmentation_receipt_hash, candidate_count, survivor_count, row_receipt_hash, created_at, updated_at
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(physical_row_key_hash) do update set
            row_status=excluded.row_status,
            error_class=excluded.error_class,
            segmentation_receipt_hash=excluded.segmentation_receipt_hash,
            candidate_count=excluded.candidate_count,
            survivor_count=excluded.survivor_count,
            row_receipt_hash=excluded.row_receipt_hash,
            updated_at=excluded.updated_at
        """,
        [
            (
                payload["physical_row_key_hash"],
                payload["source_schedule_hash"],
                payload["schedule_entry_hash"],
                payload["source_file_id_hash"],
                int(payload["row_group_index"]),
                int(payload["row_group_local_offset"]),
                payload["source_document_id_hash"],
                payload["row_status"],
                payload["error_class"],
                payload["segmentation_receipt_hash"],
                int(payload["candidate_count"]),
                int(payload["survivor_count"]),
                payload["row_receipt_hash"],
                now,
                now,
            )
            for payload in payloads
        ],
    )


def _bulk_insert_prefilter_source_rows(conn: sqlite3.Connection, payloads: Sequence[Mapping[str, Any]]) -> None:
    if not payloads:
        return
    now = time.time()
    conn.executemany(
        """
        insert into prefilter_source_rows(
            physical_row_key_hash, source_schedule_hash, schedule_entry_hash, source_file_id_hash,
            row_group_index, row_group_local_offset, source_document_id_hash, row_status, error_class,
            segmentation_receipt_hash, candidate_count, survivor_count, row_receipt_hash, created_at, updated_at
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                payload["physical_row_key_hash"],
                payload["source_schedule_hash"],
                payload["schedule_entry_hash"],
                payload["source_file_id_hash"],
                int(payload["row_group_index"]),
                int(payload["row_group_local_offset"]),
                payload["source_document_id_hash"],
                payload["row_status"],
                payload["error_class"],
                payload["segmentation_receipt_hash"],
                int(payload["candidate_count"]),
                int(payload["survivor_count"]),
                payload["row_receipt_hash"],
                now,
                now,
            )
            for payload in payloads
        ],
    )


def _prefilter_terminal_offsets_for_entry(
    conn: sqlite3.Connection,
    entry: Mapping[str, Any],
    offsets: Sequence[int],
) -> set[int]:
    if not offsets:
        return set()
    key_to_offset = {_physical_row_key(entry, int(offset)): int(offset) for offset in offsets}
    terminal: set[int] = set()
    keys = list(key_to_offset)
    for start in range(0, len(keys), 500):
        chunk = keys[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"select physical_row_key_hash, row_status from prefilter_source_rows where physical_row_key_hash in ({placeholders})",
            chunk,
        ):
            if str(row["row_status"]) in PREFILTER_TERMINAL_ROW_STATUSES:
                terminal.add(key_to_offset[str(row["physical_row_key_hash"])])
    return terminal


def _upsert_prefilter_source_row(
    conn: sqlite3.Connection,
    *,
    entry: Mapping[str, Any],
    offset: int,
    row_status: str,
    error_class: str | None,
    segmentation_receipt_hash: str | None,
    candidate_count: int,
    survivor_count: int,
) -> None:
    payload = _prefilter_source_row_payload(
        entry=entry,
        offset=offset,
        row_status=row_status,
        error_class=error_class,
        segmentation_receipt_hash=segmentation_receipt_hash,
        candidate_count=candidate_count,
        survivor_count=survivor_count,
    )
    physical_key = str(payload["physical_row_key_hash"])
    existing = conn.execute(
        "select row_receipt_hash from prefilter_source_rows where physical_row_key_hash = ?",
        (physical_key,),
    ).fetchone()
    now = time.time()
    if existing is not None and str(existing["row_receipt_hash"]) == payload["row_receipt_hash"]:
        return
    with conn:
        conn.execute(
            """
            insert into prefilter_source_rows(
                physical_row_key_hash, source_schedule_hash, schedule_entry_hash, source_file_id_hash,
                row_group_index, row_group_local_offset, source_document_id_hash, row_status, error_class,
                segmentation_receipt_hash, candidate_count, survivor_count, row_receipt_hash, created_at, updated_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(physical_row_key_hash) do update set
                row_status=excluded.row_status,
                error_class=excluded.error_class,
                segmentation_receipt_hash=excluded.segmentation_receipt_hash,
                candidate_count=excluded.candidate_count,
                survivor_count=excluded.survivor_count,
                row_receipt_hash=excluded.row_receipt_hash,
                updated_at=excluded.updated_at
            """,
            (
                physical_key,
                payload["source_schedule_hash"],
                payload["schedule_entry_hash"],
                payload["source_file_id_hash"],
                payload["row_group_index"],
                payload["row_group_local_offset"],
                payload["source_document_id_hash"],
                row_status,
                error_class,
                segmentation_receipt_hash,
                int(candidate_count),
                int(survivor_count),
                payload["row_receipt_hash"],
                now,
                now,
            ),
        )


def _prefilter_candidate_insert_record(
    candidate: Mapping[str, Any],
    *,
    prefilter_route: str,
    deterministic_result: Mapping[str, Any],
    s1_score: float | None,
    private_evidence_text: str | None,
) -> tuple[tuple[Any, ...], tuple[Any, ...] | None]:
    if prefilter_route not in {"deterministic_reject", "s1_reject", "prefilter_survivor", "prefilter_error", "prefilter_quarantine"}:
        raise ValueError("unsupported prefilter route")
    validate_public_no_raw_payload(candidate)
    validate_public_no_raw_payload(deterministic_result)
    candidate_id = _require_sha256(candidate.get("candidate_id_hash"), "candidate_id_hash")
    candidate_json = json.dumps(dict(candidate), sort_keys=True)
    deterministic_json = json.dumps(dict(deterministic_result), sort_keys=True)
    candidate_row = (
        candidate_id,
        candidate["source_document_id_hash"],
        candidate["segmentation_receipt_hash"],
        candidate_json,
        deterministic_json,
        prefilter_route,
        s1_score,
        time.time(),
    )
    evidence_row: tuple[Any, ...] | None = None
    if prefilter_route == "prefilter_survivor":
        if private_evidence_text is None:
            raise ValueError("prefilter survivor requires private evidence text")
        evidence_hash = _sha256_text(private_evidence_text)
        if evidence_hash != candidate["evidence_text_sha256"]:
            raise ValueError("private W1 evidence hash mismatch")
        private_evidence_id = stable_json_hash(
            {
                "candidate_id_hash": candidate_id,
                "evidence_text_sha256": evidence_hash,
                "config_hash": candidate["candidate_policy_hash"],
            }
        )
        evidence_row = (private_evidence_id, candidate_id, evidence_hash, private_evidence_text)
    return candidate_row, evidence_row


def _bulk_insert_prefilter_candidates(
    conn: sqlite3.Connection,
    records: Sequence[tuple[tuple[Any, ...], tuple[Any, ...] | None]],
) -> None:
    if not records:
        return
    conn.executemany(
        """
        insert into prefilter_candidates(
            candidate_id_hash, source_document_id_hash, segmentation_receipt_hash,
            candidate_json, deterministic_result_json, prefilter_route, s1_score, created_at
        ) values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [record[0] for record in records],
    )
    evidence_rows = [record[1] for record in records if record[1] is not None]
    if evidence_rows:
        conn.executemany(
            """
            insert into private_w1_evidence(
                private_evidence_row_id, candidate_id_hash, evidence_text_sha256, private_evidence_text
            ) values (?, ?, ?, ?)
            """,
            evidence_rows,
        )


def _bulk_insert_prefilter_batch(
    conn: sqlite3.Connection,
    *,
    source_row_payloads: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[tuple[tuple[Any, ...], tuple[Any, ...] | None]],
) -> None:
    try:
        with conn:
            _bulk_insert_prefilter_source_rows(conn, source_row_payloads)
            _bulk_insert_prefilter_candidates(conn, candidate_records)
    except sqlite3.IntegrityError:
        with conn:
            _bulk_upsert_prefilter_source_rows(conn, source_row_payloads)
            _bulk_insert_prefilter_candidates(conn, candidate_records)
    db_list = conn.execute("pragma database_list").fetchone()
    if db_list is not None:
        _chmod_private_sqlite_files(str(db_list["file"]))


def _insert_prefilter_candidate(
    conn: sqlite3.Connection,
    candidate: Mapping[str, Any],
    *,
    prefilter_route: str,
    deterministic_result: Mapping[str, Any],
    s1_score: float | None,
    private_evidence_text: str | None,
) -> None:
    if prefilter_route not in {"deterministic_reject", "s1_reject", "prefilter_survivor", "prefilter_error", "prefilter_quarantine"}:
        raise ValueError("unsupported prefilter route")
    validate_public_no_raw_payload(candidate)
    validate_public_no_raw_payload(deterministic_result)
    candidate_id = _require_sha256(candidate.get("candidate_id_hash"), "candidate_id_hash")
    candidate_json = json.dumps(dict(candidate), sort_keys=True)
    deterministic_json = json.dumps(dict(deterministic_result), sort_keys=True)
    existing = conn.execute(
        "select candidate_json, deterministic_result_json, prefilter_route, s1_score from prefilter_candidates where candidate_id_hash = ?",
        (candidate_id,),
    ).fetchone()
    if existing is not None:
        if (
            str(existing["candidate_json"]) != candidate_json
            or str(existing["deterministic_result_json"]) != deterministic_json
            or str(existing["prefilter_route"]) != prefilter_route
        ):
            raise ValueError("conflicting prefilter candidate row for existing id")
        if s1_score is None and existing["s1_score"] is None:
            return
        if s1_score is not None and existing["s1_score"] is not None and float(existing["s1_score"]) == float(s1_score):
            return
        raise ValueError("conflicting prefilter candidate score for existing id")
    with conn:
        conn.execute(
            """
            insert into prefilter_candidates(
                candidate_id_hash, source_document_id_hash, segmentation_receipt_hash,
                candidate_json, deterministic_result_json, prefilter_route, s1_score, created_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                candidate["source_document_id_hash"],
                candidate["segmentation_receipt_hash"],
                candidate_json,
                deterministic_json,
                prefilter_route,
                s1_score,
                time.time(),
            ),
        )
        if prefilter_route == "prefilter_survivor":
            if private_evidence_text is None:
                raise ValueError("prefilter survivor requires private evidence text")
            evidence_hash = _sha256_text(private_evidence_text)
            if evidence_hash != candidate["evidence_text_sha256"]:
                raise ValueError("private W1 evidence hash mismatch")
            private_evidence_id = stable_json_hash(
                {
                    "candidate_id_hash": candidate_id,
                    "evidence_text_sha256": evidence_hash,
                    "config_hash": candidate["candidate_policy_hash"],
                }
            )
            conn.execute(
                """
                insert into private_w1_evidence(
                    private_evidence_row_id, candidate_id_hash, evidence_text_sha256, private_evidence_text
                ) values (?, ?, ?, ?)
                """,
                (private_evidence_id, candidate_id, evidence_hash, private_evidence_text),
            )
    db_list = conn.execute("pragma database_list").fetchone()
    if db_list is not None:
        _chmod_private_sqlite_files(str(db_list["file"]))


def _prefilter_segmented_result(
    conn: sqlite3.Connection,
    *,
    result: SegmentationResult,
    raw_text: str,
    config_payload: Mapping[str, Any],
    sparse_scorer: ChunkFilteringScorer,
    deterministic_policy_payload: Mapping[str, Any],
    deterministic_ontology: RiskOntology,
    deterministic_runtime: DeterministicFilterRuntime | None = None,
) -> None:
    candidates = build_filtering_candidates(
        result,
        raw_text=raw_text,
        config_hash=str(config_payload["config_hash"]),
        target_proxy_tokens=int(config_payload["target_proxy_tokens"]),
        halo_chars=int(config_payload["halo_chars"]),
        deterministic_policy_hash=str(deterministic_policy_payload["deterministic_filter_policy_hash"]),
    )
    scorer_items: list[tuple[Mapping[str, Any], Mapping[str, Any], str]] = []
    scorer_texts: list[str] = []
    scorer_metadata: list[dict[str, Any]] = []
    sparse_scorer_metadata_required = _sparse_scorer_requires_metadata(sparse_scorer)
    deterministic_normalization_cache: dict[tuple[Any, ...], Any] = {}
    for candidate in candidates:
        existing = conn.execute(
            "select candidate_json from prefilter_candidates where candidate_id_hash = ?",
            (candidate["candidate_id_hash"],),
        ).fetchone()
        if existing is not None:
            if json.loads(str(existing["candidate_json"])) != dict(candidate):
                raise ValueError("conflicting prefilter candidate row for existing id")
            continue
        if deterministic_runtime is None:
            deterministic_result = evaluate_chunk_candidate_deterministic_filter(
                candidate,
                raw_text=raw_text,
                ontology=deterministic_ontology,
                policy_payload=deterministic_policy_payload,
            )
        else:
            deterministic_result = evaluate_chunk_candidate_deterministic_filter_with_runtime(
                candidate,
                raw_text=raw_text,
                runtime=deterministic_runtime,
                normalization_cache=deterministic_normalization_cache,
            )
        deterministic_decision = _deterministic_candidate_decision(candidate)
        if deterministic_result.get("terminal_decision") == "reject":
            _insert_prefilter_candidate(
                conn,
                candidate,
                prefilter_route="deterministic_reject",
                deterministic_result=deterministic_result,
                s1_score=None,
                private_evidence_text=None,
            )
            continue
        if deterministic_decision == "quarantine":
            _insert_prefilter_candidate(
                conn,
                candidate,
                prefilter_route="prefilter_quarantine",
                deterministic_result=deterministic_result,
                s1_score=None,
                private_evidence_text=None,
            )
            continue
        evidence_text = raw_text[int(candidate["evidence_char_start"]) : int(candidate["evidence_char_end"])]
        scorer_items.append((candidate, deterministic_result, evidence_text))
        scorer_texts.append(evidence_text)
        if sparse_scorer_metadata_required:
            scorer_metadata.append(_sparse_scorer_metadata(sparse_scorer, candidate))
    if not scorer_items:
        return
    scores = list(
        sparse_scorer.score_many(
            scorer_texts,
            scorer_metadata if sparse_scorer_metadata_required else [],
        )
    )
    if len(scores) != len(scorer_items):
        scores = [math.nan] * len(scorer_items)
    for (candidate, deterministic_result, evidence_text), score in zip(scorer_items, scores, strict=True):
        parsed = _safe_runtime_score(score)
        if parsed is None:
            route = "prefilter_error"
        elif parsed >= float(config_payload["sparse_threshold"]):
            route = "s1_reject"
        else:
            route = "prefilter_survivor"
        _insert_prefilter_candidate(
            conn,
            candidate,
            prefilter_route=route,
            deterministic_result=deterministic_result,
            s1_score=parsed,
            private_evidence_text=evidence_text if route == "prefilter_survivor" else None,
        )


def _prefilter_candidate_counts(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        select count(*) as candidate_count,
               sum(case when prefilter_route = 'prefilter_survivor' then 1 else 0 end) as survivor_count
        from prefilter_candidates
        """
    ).fetchone()
    return {
        "candidate_count": int(row["candidate_count"] or 0),
        "survivor_count": int(row["survivor_count"] or 0),
    }


def _prefilter_source_row_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    metadata = {
        str(row["key"]): json.loads(str(row["value_json"]))
        for row in conn.execute(
            "select key, value_json from stage_metadata where key in ('input_schedule_hash', 'expected_physical_row_count', 'expected_physical_row_key_set_hash')"
        )
    }
    try:
        status_rows = list(conn.execute("select row_status, count(*) as row_count from prefilter_source_rows group by row_status"))
    except sqlite3.OperationalError:
        status_rows = []
    counts = Counter({str(row["row_status"]): int(row["row_count"]) for row in status_rows})
    error_counts = Counter(
        {
            "fetch_error": counts.get("fetch_error", 0),
            "missing_text": counts.get("missing_text", 0),
            "non_mapping_row": counts.get("non_mapping_row", 0),
            "segmentation_error": counts.get("segmentation_error", 0),
        }
    )
    error_counts = Counter({key: value for key, value in error_counts.items() if value})
    final_statuses = {
        "prefiltered",
        "fetch_error",
        "non_mapping_row",
        "missing_text",
        "segmentation_error",
    }
    unaccounted = sum(value for key, value in counts.items() if key not in final_statuses)
    terminal_rows = sum(value for key, value in counts.items() if key in final_statuses)
    pending_rows = sum(value for key, value in counts.items() if key not in final_statuses)
    scheduled_key_hash = None
    terminal_key_hash = None
    pending_key_hash = None
    if metadata.get("expected_physical_row_key_set_hash") is not None:
        rows = list(conn.execute("select physical_row_key_hash, row_status from prefilter_source_rows"))
        scheduled_keys = sorted(str(row["physical_row_key_hash"]) for row in rows)
        terminal_keys = sorted(
            str(row["physical_row_key_hash"]) for row in rows if str(row["row_status"]) in final_statuses
        )
        pending_keys = sorted(
            str(row["physical_row_key_hash"]) for row in rows if str(row["row_status"]) not in final_statuses
        )
        scheduled_key_hash = stable_json_hash(scheduled_keys)
        terminal_key_hash = stable_json_hash(terminal_keys)
        pending_key_hash = stable_json_hash(pending_keys)
    expected_rows = metadata.get("expected_physical_row_count")
    if expected_rows is None:
        expected_rows = int(sum(counts.values()))
    return {
        "input_schedule_hash": metadata.get("input_schedule_hash"),
        "expected_physical_row_key_set_hash": metadata.get("expected_physical_row_key_set_hash"),
        "expected_physical_rows": int(expected_rows),
        "scheduled_rows": int(sum(counts.values())),
        "scheduled_physical_row_key_set_hash": scheduled_key_hash,
        "terminal_rows": int(terminal_rows),
        "terminal_physical_row_key_set_hash": terminal_key_hash,
        "pending_rows": int(pending_rows),
        "pending_physical_row_key_set_hash": pending_key_hash,
        "fetched_rows": int(counts.get("prefiltered", 0) + counts.get("missing_text", 0) + counts.get("segmentation_error", 0)),
        "segmented_rows": int(counts.get("prefiltered", 0)),
        "row_error_counts": _sorted_counter(error_counts),
        "unaccounted_row_count": int(unaccounted),
    }


def _iter_prefilter_survivor_queue_rows(
    conn: sqlite3.Connection,
    config_payload: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    for row in conn.execute(
        """
        select p.candidate_id_hash, p.candidate_json, p.deterministic_result_json, p.s1_score,
               e.private_evidence_row_id, e.private_evidence_text
        from prefilter_candidates p
        join private_w1_evidence e on e.candidate_id_hash = p.candidate_id_hash
        where p.prefilter_route = 'prefilter_survivor'
        order by p.candidate_id_hash
        """
    ):
        candidate = json.loads(str(row["candidate_json"]))
        deterministic_result = json.loads(str(row["deterministic_result_json"]))
        core_proxy_token_count = int(candidate["core_proxy_token_count"])
        evidence_proxy_token_count = candidate.get("evidence_proxy_token_count")
        if evidence_proxy_token_count is None:
            evidence_proxy_token_count = _estimate_proxy_tokens_from_normalized(str(row["private_evidence_text"]))
        evidence_proxy_token_count = max(core_proxy_token_count, int(evidence_proxy_token_count))
        payload = {
            "artifact_kind": "twc_chunk_filter_w1_survivor_queue_row_v1",
            "candidate_id_hash": row["candidate_id_hash"],
            "source_document_id_hash": candidate["source_document_id_hash"],
            "segmentation_receipt_hash": candidate["segmentation_receipt_hash"],
            "candidate_policy_hash": candidate["candidate_policy_hash"],
            "source_metadata_hash": config_payload["source_metadata_hash"],
            "config_hash": config_payload["config_hash"],
            "selected_policy_hash": config_payload["selected_policy_hash"],
            "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
            "deterministic_filter_policy_hash": deterministic_result["deterministic_filter_policy_hash"],
            "deterministic_filter_result_hash": deterministic_result["deterministic_filter_result_hash"],
            "s1_score": _runtime_score(row["s1_score"]),
            "s1_threshold": _threshold(config_payload["sparse_threshold"], "s1_threshold"),
            "s1_report_hash": config_payload["sparse_report_hash"],
            "s1_model_file_sha256": config_payload["sparse_model_file_sha256"],
            "w1_threshold": _threshold(config_payload["window_mil_threshold"], "w1_threshold"),
            "w1_report_hash": config_payload["window_mil_report_hash"],
            "w1_model_file_sha256": config_payload["window_mil_model_file_sha256"],
            "core_text_sha256": candidate["core_text_sha256"],
            "evidence_text_sha256": candidate["evidence_text_sha256"],
            "core_proxy_token_count": core_proxy_token_count,
            "evidence_proxy_token_count": evidence_proxy_token_count,
            "estimated_w1_window_count": max(1, math.ceil(evidence_proxy_token_count / 512)),
            "private_evidence_row_id": row["private_evidence_row_id"],
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        payload["stage_receipt_hash"] = stable_json_hash(payload)
        validate_public_no_raw_payload(payload)
        yield payload


def _legacy_core_estimated_survivor_queue_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.pop("stage_receipt_hash", None)
    payload.pop("evidence_proxy_token_count", None)
    payload["estimated_w1_window_count"] = max(1, math.ceil(int(payload["core_proxy_token_count"]) / 512))
    payload["stage_receipt_hash"] = stable_json_hash(payload)
    return payload


def _survivor_queue_rows_match_live_db(
    exported_rows: Sequence[Mapping[str, Any]],
    db_rows: Sequence[Mapping[str, Any]],
) -> bool:
    if list(exported_rows) == list(db_rows):
        return True
    legacy_db_rows = [_legacy_core_estimated_survivor_queue_row(row) for row in db_rows]
    return list(exported_rows) == legacy_db_rows


def _prefilter_survivor_queue_rows(conn: sqlite3.Connection, config_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return list(_iter_prefilter_survivor_queue_rows(conn, config_payload))


def _build_prefilter_receipt(
    conn: sqlite3.Connection,
    config_payload: Mapping[str, Any],
    *,
    run_path: Path,
    status: str,
    stop_reason: str,
    elapsed_seconds: float,
    cache_dir: Path | None,
    min_free_bytes: int,
    survivor_queue_ready: bool | None = None,
    survivor_queue_part: bool | None = None,
    survivor_queue_hash: str | None = None,
    survivor_queue_row_count: int | None = None,
    survivor_queue_id_index_hash: str | None = None,
    survivor_queue_id_index_row_count: int | None = None,
    survivor_candidate_id_set_hash: str | None = None,
    survivor_private_evidence_id_set_hash: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    route_counts = Counter(
        str(row["prefilter_route"]) for row in conn.execute("select prefilter_route from prefilter_candidates")
    )
    survivor_count = int(route_counts.get("prefilter_survivor", 0))
    queue_ready = run_path / "w1_survivor_queue.jsonl.ready"
    queue_part = run_path / "w1_survivor_queue.jsonl.part"
    disk_bytes = _directory_size(run_path)
    cache_bytes = _directory_size(cache_dir) if cache_dir is not None else 0
    row_counts = _prefilter_source_row_summary(conn)
    accounted_rows = int(row_counts["terminal_rows"]) + int(row_counts["pending_rows"])
    expected_rows = int(row_counts["expected_physical_rows"])
    if accounted_rows > expected_rows:
        raise ValueError("prefilter expected source row accounting mismatch")
    unvisited_rows = expected_rows - accounted_rows
    ready_status = status in {"prefilter_ready_for_w1", "prefilter_shard_ready_for_merge"}
    if ready_status and int(row_counts["unaccounted_row_count"]) != 0:
        raise ValueError("prefilter readiness blocked by unaccounted source rows")
    if ready_status and int(row_counts["pending_rows"]) != 0:
        raise ValueError("prefilter readiness blocked by pending source rows")
    if ready_status and unvisited_rows != 0:
        raise ValueError("prefilter readiness blocked by unvisited source rows")
    if (
        ready_status
        and row_counts["expected_physical_row_key_set_hash"] is not None
        and row_counts["terminal_physical_row_key_set_hash"] != row_counts["expected_physical_row_key_set_hash"]
    ):
        raise ValueError("prefilter readiness blocked by expected physical row key-set mismatch")
    if ready_status and row_counts["row_error_counts"]:
        raise ValueError("prefilter readiness blocked by terminal row errors")
    next_phase = "w1-score" if status == "prefilter_ready_for_w1" else "resume_prefilter_cpu"
    if status == "prefilter_shard_ready_for_merge":
        next_phase = "build-sharded-w1-handoff"
    if status == "prefilter_blocked_errors":
        next_phase = "review_prefilter_row_errors"
    if status == "prefilter_diagnostic_complete":
        next_phase = "review_prefilter_diagnostic_only"
    if status == "prefilter_ready_for_w1" and survivor_count == 0:
        next_phase = "w1-score_or_assemble-report_zero_survivors"
    payload = {
        "artifact_kind": PREFILTER_CPU_STAGE_KIND,
        "schema_version": 1,
        "phase": "prefilter-cpu",
        "status": status,
        "stop_reason": stop_reason,
        "next_required_phase": next_phase,
        "config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "selected_eval_deterministic_filter_policy_hash": config_payload[
            "selected_eval_deterministic_filter_policy_hash"
        ],
        "executed_deterministic_filter_policy_hash": config_payload.get("deterministic_filter_policy_hash"),
        "candidate_count": int(sum(route_counts.values())),
        "survivor_count": survivor_count,
        "input_schedule_hash": row_counts["input_schedule_hash"],
        "expected_physical_rows": row_counts["expected_physical_rows"],
        "expected_physical_row_key_set_hash": row_counts["expected_physical_row_key_set_hash"],
        "scheduled_rows": row_counts["scheduled_rows"],
        "scheduled_physical_row_key_set_hash": row_counts["scheduled_physical_row_key_set_hash"],
        "terminal_rows": row_counts["terminal_rows"],
        "terminal_physical_row_key_set_hash": row_counts["terminal_physical_row_key_set_hash"],
        "pending_rows": row_counts["pending_rows"],
        "pending_physical_row_key_set_hash": row_counts["pending_physical_row_key_set_hash"],
        "unvisited_rows": int(unvisited_rows),
        "fetched_rows": row_counts["fetched_rows"],
        "segmented_rows": row_counts["segmented_rows"],
        "row_error_counts": row_counts["row_error_counts"],
        "unaccounted_row_count": row_counts["unaccounted_row_count"],
        "route_counts": _sorted_counter(route_counts),
        "survivor_queue_ready": queue_ready.exists() if survivor_queue_ready is None else bool(survivor_queue_ready),
        "survivor_queue_part": queue_part.exists() if survivor_queue_part is None else bool(survivor_queue_part),
        "survivor_queue_hash": (
            _sha256_file(queue_ready) if queue_ready.exists() else None
        )
        if survivor_queue_hash is None
        else _require_sha256(survivor_queue_hash, "survivor_queue_hash"),
        "survivor_queue_row_count": None if survivor_queue_row_count is None else _nonnegative_int(
            survivor_queue_row_count,
            "survivor_queue_row_count",
        ),
        "survivor_queue_id_index_hash": None
        if survivor_queue_id_index_hash is None
        else _require_sha256(survivor_queue_id_index_hash, "survivor_queue_id_index_hash"),
        "survivor_queue_id_index_row_count": None
        if survivor_queue_id_index_row_count is None
        else _nonnegative_int(
            survivor_queue_id_index_row_count,
            "survivor_queue_id_index_row_count",
        ),
        "survivor_candidate_id_set_hash": None
        if survivor_candidate_id_set_hash is None
        else _require_sha256(survivor_candidate_id_set_hash, "survivor_candidate_id_set_hash"),
        "survivor_private_evidence_id_set_hash": None
        if survivor_private_evidence_id_set_hash is None
        else _require_sha256(survivor_private_evidence_id_set_hash, "survivor_private_evidence_id_set_hash"),
        "survivor_queue_proof_kind": (
            "jsonl_sha256_and_id_set_hashes_v1"
            if (
                survivor_queue_row_count is not None
                and survivor_candidate_id_set_hash is not None
                and survivor_private_evidence_id_set_hash is not None
            )
            else None
        ),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "candidates_per_second": round(float(sum(route_counts.values()) / max(float(elapsed_seconds), 1e-9)), 6),
        "private_run_dir_disk_bytes": int(disk_bytes),
        "source_cache_disk_bytes": int(cache_bytes),
        "disk_free_bytes": min(_path_free_bytes(run_path), _path_free_bytes(cache_dir)) if cache_dir is not None else _path_free_bytes(run_path),
        "disk_min_free_bytes": int(min_free_bytes),
        "disk_watermark_guard_configured": True,
        "disk_watermark_guard_enforced": True,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    if extra_fields:
        validate_public_no_raw_payload(extra_fields)
        payload.update(dict(extra_fields))
    payload["stage_receipt_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "stage_receipt_hash"})
    validate_public_no_raw_payload(payload)
    return payload


def _build_w1_score_payload(
    survivor_row: Mapping[str, Any],
    *,
    config_payload: Mapping[str, Any],
    score: float | None,
    route: str,
    error_class: str | None,
    w1_runtime_envelope_hash: str,
) -> dict[str, Any]:
    candidate_id = _require_sha256(survivor_row.get("candidate_id_hash"), "candidate_id_hash")
    if route not in {"w1_pass", "w1_reject", "w1_error"}:
        raise ValueError("unsupported W1 route")
    payload = {
        "artifact_kind": "twc_chunk_filter_w1_score_row_v1",
        "candidate_id_hash": candidate_id,
        "config_hash": config_payload["config_hash"],
        "prefilter_producer_config_hash": survivor_row["config_hash"],
        "w1_consumer_config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "w1_score": None if score is None else _runtime_score(score),
        "w1_threshold": config_payload["window_mil_threshold"],
        "w1_route": route,
        "error_class": error_class,
        "survivor_stage_receipt_hash": survivor_row["stage_receipt_hash"],
        "w1_report_hash": config_payload["window_mil_report_hash"],
        "w1_model_file_sha256": config_payload["window_mil_model_file_sha256"],
        "w1_runtime_envelope_hash": _require_sha256(w1_runtime_envelope_hash, "w1_runtime_envelope_hash"),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["stage_receipt_hash"] = stable_json_hash(payload)
    validate_public_no_raw_payload(payload)
    return payload


def _insert_w1_scores(
    conn: sqlite3.Connection,
    records: Sequence[tuple[Mapping[str, Any], float | None, str, str | None]],
    *,
    config_payload: Mapping[str, Any],
    w1_runtime_envelope_hash: str,
) -> int:
    if not records:
        return 0
    payloads = [
        _build_w1_score_payload(
            survivor_row,
            config_payload=config_payload,
            score=score,
            route=route,
            error_class=error_class,
            w1_runtime_envelope_hash=w1_runtime_envelope_hash,
        )
        for survivor_row, score, route, error_class in records
    ]
    return _insert_w1_score_payloads(conn, payloads)


def _insert_w1_score_payloads(
    conn: sqlite3.Connection,
    payloads: Sequence[Mapping[str, Any]],
) -> int:
    if not payloads:
        return 0
    candidate_ids = [str(payload["candidate_id_hash"]) for payload in payloads]
    existing_by_id: dict[str, str] = {}
    for chunk in _iter_sized_chunks(list(dict.fromkeys(candidate_ids)), 900):
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"select candidate_id_hash, score_json from w1_scores where candidate_id_hash in ({placeholders})",
            tuple(chunk),
        ):
            existing_by_id[str(row["candidate_id_hash"])] = str(row["score_json"])
    created_at = time.time()
    rows_to_insert: list[tuple[str, str, float]] = []
    for payload in payloads:
        candidate_id = str(payload["candidate_id_hash"])
        score_json = json.dumps(payload, sort_keys=True)
        existing = existing_by_id.get(candidate_id)
        if existing is not None:
            if existing != score_json:
                raise ValueError("conflicting W1 score row for existing candidate")
            continue
        rows_to_insert.append((candidate_id, score_json, created_at))
    if rows_to_insert:
        with conn:
            conn.executemany(
                "insert into w1_scores(candidate_id_hash, score_json, created_at) values (?, ?, ?)",
                rows_to_insert,
            )
        db_list = conn.execute("pragma database_list").fetchone()
        if db_list is not None:
            _chmod_private_sqlite_files(str(db_list["file"]))
    return len(rows_to_insert)


def _w1_score_export_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        json.loads(str(row["score_json"]))
        for row in conn.execute("select score_json from w1_scores order by candidate_id_hash")
    ]


def _load_existing_complete_w1_receipt(
    run_path: Path,
    config_payload: Mapping[str, Any],
    *,
    queue_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    w1_runtime_envelope_hash: str,
) -> dict[str, Any] | None:
    receipt_path = run_path / "w1_score_receipt.json"
    if not receipt_path.exists():
        return None
    receipt = _load_json(receipt_path)
    if not isinstance(receipt, Mapping):
        raise ValueError("existing W1 score receipt must be an object")
    if receipt.get("status") != "w1_scored":
        return None
    if receipt.get("stage_receipt_hash") != stable_json_hash(
        {key: value for key, value in receipt.items() if key != "stage_receipt_hash"}
    ):
        raise ValueError("existing W1 score receipt hash mismatch")
    if receipt.get("artifact_kind") != W1_SCORE_STAGE_KIND:
        raise ValueError("existing W1 score receipt kind mismatch")
    if receipt.get("phase") != "w1-score":
        raise ValueError("existing W1 score receipt phase mismatch")
    if receipt.get("config_hash") != config_payload["config_hash"]:
        raise ValueError("existing W1 score receipt config hash mismatch")
    if receipt.get("selected_policy_hash") != config_payload["selected_policy_hash"]:
        raise ValueError("existing W1 score receipt selected policy mismatch")
    if receipt.get("selected_policy_report_hash") != config_payload["selected_policy_report_hash"]:
        raise ValueError("existing W1 score receipt selected policy report mismatch")
    if receipt.get("source_metadata_hash") != config_payload["source_metadata_hash"]:
        raise ValueError("existing W1 score receipt source metadata mismatch")
    _validated_w1_runtime_envelope_from_receipt(
        receipt,
        requested_w1_runtime_envelope_hash=w1_runtime_envelope_hash,
        label="existing W1 score receipt",
    )
    if int(receipt.get("survivor_count", -1)) != len(queue_rows):
        raise ValueError("existing W1 score receipt survivor count mismatch")
    if int(receipt.get("scored_count", -1)) != len(score_rows) or len(score_rows) != len(queue_rows):
        raise ValueError("existing W1 score receipt scored count mismatch")
    survivor_by_id = {str(row["candidate_id_hash"]): row for row in queue_rows}
    _validate_w1_score_rows(score_rows, config_payload, survivor_by_id=survivor_by_id, allow_empty=True)
    if score_rows and _w1_score_row_runtime_hashes(score_rows) != {w1_runtime_envelope_hash}:
        raise ValueError("existing W1 score receipt score row runtime envelope mismatch")
    score_ids = {str(row["candidate_id_hash"]) for row in score_rows}
    if score_ids != set(survivor_by_id):
        raise ValueError("existing W1 score receipt scored set mismatch")
    expected_route_counts = _sorted_counter(Counter(str(row["w1_route"]) for row in score_rows))
    if receipt.get("route_counts") != expected_route_counts:
        raise ValueError("existing W1 score receipt route counts mismatch")
    ready_path = run_path / "w1_scores.jsonl.ready"
    if not ready_path.exists():
        raise ValueError("existing complete W1 score receipt requires ready score export")
    if receipt.get("w1_scores_hash") != _sha256_file(ready_path):
        raise ValueError("existing W1 score receipt ready export hash mismatch")
    ready_queue_path = run_path / "w1_survivor_queue.jsonl.ready"
    current_survivor_queue_hash = _sha256_file(ready_queue_path)
    if receipt.get("survivor_queue_hash") is not None and receipt.get("survivor_queue_hash") != current_survivor_queue_hash:
        raise ValueError("existing W1 score receipt survivor queue hash mismatch")
    part_path = run_path / "w1_scores.jsonl.part"
    if receipt.get("w1_scores_part_hash") is not None:
        if not part_path.exists() or receipt.get("w1_scores_part_hash") != _sha256_file(part_path):
            raise ValueError("existing W1 score receipt part export hash mismatch")
    _require_false_flags(receipt, PUBLIC_FALSE_FLAGS, "existing W1 score receipt")
    validate_public_no_raw_payload(receipt)
    return dict(receipt)


def _build_w1_receipt(
    conn: sqlite3.Connection,
    config_payload: Mapping[str, Any],
    *,
    queue_rows: Sequence[Mapping[str, Any]],
    run_path: Path,
    elapsed_seconds: float,
    stop_requested: bool = False,
    runtime_telemetry: Mapping[str, Any] | None = None,
    w1_runtime_envelope: Mapping[str, Any],
    w1_runtime_envelope_hash: str,
    enforce_accepted_token_target: bool = False,
    accepted_token_count_initial_lower_bound: int = 0,
    allow_queue_exhaustion_before_token_target: bool = False,
) -> dict[str, Any]:
    score_rows = _w1_score_export_rows(conn)
    route_counts = Counter(str(row["w1_route"]) for row in score_rows)
    exhausted = len(score_rows) == len(queue_rows)
    survivor_by_id = {str(row["candidate_id_hash"]): row for row in queue_rows}
    accepted_token_target_lower_bound = int(config_payload["accepted_token_target_lower_bound"])
    accepted_token_initial_lower_bound = int(accepted_token_count_initial_lower_bound)
    if accepted_token_initial_lower_bound < 0:
        raise ValueError("accepted_token_count_initial_lower_bound must be non-negative")
    accepted_stage_token_count_lower_bound = _w1_score_rows_accepted_token_lower_bound(
        score_rows,
        survivor_by_id=survivor_by_id,
    )
    accepted_token_count_lower_bound = (
        accepted_token_initial_lower_bound + accepted_stage_token_count_lower_bound
    )
    accepted_target_reached = (
        bool(enforce_accepted_token_target)
        and accepted_token_count_lower_bound >= accepted_token_target_lower_bound
    )
    if accepted_target_reached and not exhausted:
        status = "accepted_token_target_reached_partial"
        stop_reason = "accepted_token_target_reached"
        next_required_phase = "archive_w1_private_score_state_before_gpu_teardown"
    elif (
        bool(enforce_accepted_token_target)
        and exhausted
        and accepted_token_count_lower_bound < accepted_token_target_lower_bound
        and not bool(allow_queue_exhaustion_before_token_target)
    ):
        status = "queue_exhausted_before_token_target"
        stop_reason = "w1_survivor_queue_exhausted_before_accepted_token_target"
        next_required_phase = "coordinator_review_token_shortfall_before_downstream_unlock"
    else:
        status = "w1_scored" if exhausted else "w1_partial"
        stop_reason = "w1_survivor_queue_exhausted" if exhausted else ("stop_requested" if stop_requested else "w1_pending_survivors")
        next_required_phase = "assemble-report" if exhausted else "resume_w1_score"
    normalized_telemetry = _normalize_w1_runtime_telemetry(runtime_telemetry)
    throughput_doc_count = (
        int(normalized_telemetry["score_rows_inserted"]) if runtime_telemetry is not None else len(score_rows)
    )
    prefilter_producer_config_hash = (
        _require_sha256(queue_rows[0].get("config_hash"), "prefilter_producer_config_hash")
        if queue_rows
        else config_payload["config_hash"]
    )
    payload = {
        "artifact_kind": W1_SCORE_STAGE_KIND,
        "schema_version": 1,
        "phase": "w1-score",
        "status": status,
        "stop_reason": stop_reason,
        "next_required_phase": next_required_phase,
        "config_hash": config_payload["config_hash"],
        "prefilter_producer_config_hash": prefilter_producer_config_hash,
        "w1_consumer_config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "survivor_count": len(queue_rows),
        "survivor_queue_hash": _sha256_file(run_path / "w1_survivor_queue.jsonl.ready"),
        "scored_count": len(score_rows),
        "accepted_token_target_enforced": bool(enforce_accepted_token_target),
        "accepted_token_target_lower_bound": accepted_token_target_lower_bound,
        "accepted_token_initial_lower_bound": accepted_token_initial_lower_bound,
        "accepted_estimated_token_count_lower_bound_stage": int(accepted_stage_token_count_lower_bound),
        "accepted_estimated_token_count_lower_bound": int(accepted_token_count_lower_bound),
        "accepted_token_target_reached": bool(accepted_target_reached),
        "accepted_token_queue_exhaustion_deferred_to_sharded_summary": bool(
            allow_queue_exhaustion_before_token_target
            and enforce_accepted_token_target
            and exhausted
            and accepted_token_count_lower_bound < accepted_token_target_lower_bound
        ),
        "route_counts": _sorted_counter(route_counts),
        "w1_scores_hash": _sha256_file(run_path / "w1_scores.jsonl.ready") if (run_path / "w1_scores.jsonl.ready").exists() else None,
        "w1_scores_part_hash": _sha256_file(run_path / "w1_scores.jsonl.part") if (run_path / "w1_scores.jsonl.part").exists() else None,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "docs_scored_this_invocation": throughput_doc_count,
        "docs_per_second": round(float(throughput_doc_count / max(float(elapsed_seconds), 1e-9)), 6),
        "batching_controls": {"batch_size_defaulted_or_requested": True},
        "w1_runtime_envelope": dict(w1_runtime_envelope),
        "w1_runtime_envelope_hash": _require_sha256(w1_runtime_envelope_hash, "w1_runtime_envelope_hash"),
        "w1_runtime_telemetry": normalized_telemetry,
        "gpu_required_for_local_tests": False,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["stage_receipt_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "stage_receipt_hash"})
    validate_public_no_raw_payload(payload)
    return payload


def _normalize_w1_runtime_telemetry(runtime_telemetry: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = {
        "stage_batch_size": 0,
        "survivor_queue_rows": 0,
        "scored_existing_initial": 0,
        "pending_initial": 0,
        "scorer_batch_count": 0,
        "scorer_input_docs": 0,
        "score_model_call_count": 0,
        "score_oom_split_count": 0,
        "score_oom_failed_attempt_count": 0,
        "score_oom_failed_attempt_docs": 0,
        "score_oom_failed_attempt_estimated_windows": 0,
        "score_oom_failed_attempt_max_docs": 0,
        "score_oom_failed_attempt_max_estimated_windows": 0,
        "score_max_effective_batch_docs": 0,
        "score_rows_attempted": 0,
        "score_rows_inserted": 0,
        "score_stage_wall_seconds": 0.0,
        "accepted_token_ledger_rows_for_shard_initial": 0,
        "accepted_token_ledger_rows_for_shard": 0,
        "accepted_token_ledger_accepted_token_count_lower_bound_for_shard": 0,
        "accepted_token_ledger_route_counts_for_shard": {},
        "accepted_token_ledger_commit_decision_counts_for_shard": {},
        "score_input_fetch_seconds": 0.0,
        "score_model_seconds": 0.0,
        "score_persist_seconds": 0.0,
        "score_cuda_synchronized_timing": False,
        "score_cuda_synchronized_timing_requested": False,
        "score_cuda_synchronized_timing_observed": False,
        "score_tokenizer_call_count": 0,
        "score_tokenized_window_count": 0,
        "score_tokenized_input_cell_count": 0,
        "score_tokenized_attention_token_count": 0,
        "score_tokenized_padding_cell_count": 0,
        "score_tokenization_seconds": 0.0,
        "score_device_transfer_seconds": 0.0,
        "score_window_forward_call_count": 0,
        "score_window_forward_window_count": 0,
        "score_window_forward_seconds": 0.0,
        "score_pooling_seconds": 0.0,
        "score_postprocess_seconds": 0.0,
        "score_cuda_synchronization_count": 0,
        "score_pooling_ragged_topk_call_count": 0,
        "score_pooling_ragged_topk_iteration_count": 0,
        "score_pooling_ragged_topk_input_window_count": 0,
        "score_pooling_ragged_topk_document_count": 0,
        "score_pooling_dense_padded_cell_count_avoided": 0,
        "score_max_windows_per_forward_call": 0,
        "score_max_tokenized_windows_per_document_batch": 0,
        "score_max_tokenized_sequence_length": 0,
        "score_max_windows_per_document": 0,
        "estimated_w1_window_count": 0,
    }
    if runtime_telemetry:
        payload.update(dict(runtime_telemetry))
    for key in (
        "stage_batch_size",
        "survivor_queue_rows",
        "scored_existing_initial",
        "pending_initial",
        "scorer_batch_count",
        "scorer_input_docs",
        "score_model_call_count",
        "score_oom_split_count",
        "score_oom_failed_attempt_count",
        "score_oom_failed_attempt_docs",
        "score_oom_failed_attempt_estimated_windows",
        "score_oom_failed_attempt_max_docs",
        "score_oom_failed_attempt_max_estimated_windows",
        "score_max_effective_batch_docs",
        "score_rows_attempted",
        "score_rows_inserted",
        "score_persist_commit_count",
        "score_persist_max_rows_per_commit",
        "score_persist_pending_rows",
        "score_tokenizer_call_count",
        "score_tokenized_window_count",
        "score_tokenized_input_cell_count",
        "score_tokenized_attention_token_count",
        "score_tokenized_padding_cell_count",
        "score_window_forward_call_count",
        "score_window_forward_window_count",
        "score_cuda_synchronization_count",
        "score_pooling_ragged_topk_call_count",
        "score_pooling_ragged_topk_iteration_count",
        "score_pooling_ragged_topk_input_window_count",
        "score_pooling_ragged_topk_document_count",
        "score_pooling_dense_padded_cell_count_avoided",
        "score_max_windows_per_forward_call",
        "score_max_tokenized_windows_per_document_batch",
        "score_max_tokenized_sequence_length",
        "score_max_windows_per_document",
        "estimated_w1_window_count",
        "accepted_token_ledger_rows_for_shard_initial",
        "accepted_token_ledger_rows_for_shard",
        "accepted_token_ledger_accepted_token_count_lower_bound_for_shard",
    ):
        payload[key] = int(payload.get(key, 0))
    legacy_sync = bool(payload.get("score_cuda_synchronized_timing", False))
    payload["score_cuda_synchronized_timing_requested"] = bool(
        payload.get("score_cuda_synchronized_timing_requested", legacy_sync)
    )
    payload["score_cuda_synchronized_timing_observed"] = bool(
        payload.get("score_cuda_synchronized_timing_observed", legacy_sync)
    )
    payload["score_cuda_synchronized_timing"] = bool(payload["score_cuda_synchronized_timing_observed"])
    for key in (
        "score_input_fetch_seconds",
        "score_stage_wall_seconds",
        "score_model_seconds",
        "score_persist_seconds",
        "score_tokenization_seconds",
        "score_device_transfer_seconds",
        "score_window_forward_seconds",
        "score_pooling_seconds",
        "score_postprocess_seconds",
    ):
        payload[key] = round(float(payload.get(key, 0.0)), 6)
    _add_w1_runtime_derived_metrics(payload)
    validate_public_no_raw_payload(payload)
    return payload


def run_chunk_filtering_schedule(
    *,
    schedule: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    row_provider: Any,
    scorers: Sequence[ChunkFilteringScorer],
    thresholds: Mapping[str, float],
    run_dir: str | Path,
    private_clean_runs_jsonl: str | Path,
    public_report_path: str | Path,
    max_rows: int | None = None,
    source_chunk_manifest_hash: str | None = None,
    deterministic_policy_payload: Mapping[str, Any] | None = None,
    deterministic_ontology: RiskOntology | None = None,
) -> dict[str, Any]:
    validate_chunk_filtering_config_payload(config_payload)
    validate_locked_chunk_filtering_config_payload(config_payload)
    configured_deterministic_hash = config_payload.get("deterministic_filter_policy_hash")
    if configured_deterministic_hash is not None:
        if deterministic_policy_payload is None or deterministic_ontology is None:
            raise ValueError("chunk filtering config binds deterministic filter but no policy/ontology was supplied")
        validate_deterministic_filter_policy_payload(deterministic_policy_payload)
        if deterministic_policy_payload.get("deterministic_filter_policy_hash") != configured_deterministic_hash:
            raise ValueError("chunk filtering deterministic filter policy hash mismatch")
    elif deterministic_policy_payload is not None or deterministic_ontology is not None:
        raise ValueError("deterministic filter policy supplied but config does not bind its hash")
    if max_rows is not None and int(max_rows) < 1:
        raise ValueError("max_rows must be positive when supplied")
    if not scorers:
        raise ValueError("chunk filtering schedule requires scorers")
    required_source_hash = str(config_payload["source_metadata_hash"])
    if schedule.get("source_metadata_hash") != required_source_hash:
        raise ValueError("chunk filtering schedule source metadata hash mismatch")
    start = time.monotonic()
    run_path = Path(run_dir)
    private_clean_path = Path(private_clean_runs_jsonl)
    public_report = Path(public_report_path)
    validate_private_clean_run_output_path(private_clean_path)
    state_config = ChunkedCleanRunStateConfig(
        run_dir=run_path,
        source_schedule_hash=str(schedule.get("schedule_hash") or "<none>"),
        source_manifest_hash=required_source_hash,
    )
    private_clean_path.parent.mkdir(parents=True, exist_ok=True)
    rows_seen = 0
    doc_reports: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    accepted_lower_bound = 0
    with open_chunked_clean_run_state(state_config, resume=True) as state:
        with _append_jsonl_atomic(private_clean_path) as private_writer:
            for entry in schedule.get("entries") or []:
                offsets = [int(offset) for offset in entry.get("row_group_local_offsets") or []]
                if not offsets:
                    continue
                fetched = row_provider.fetch_rows(dict(entry), offsets)
                for offset in offsets:
                    if max_rows is not None and rows_seen >= int(max_rows):
                        break
                    rows_seen += 1
                    row = fetched.get(offset)
                    if isinstance(row, BaseException):
                        error_counts[type(row).__name__] += 1
                        continue
                    if not isinstance(row, Mapping):
                        error_counts["non_mapping_row"] += 1
                        continue
                    text = row.get("text")
                    source_id = _source_document_id_for_schedule_row(entry, offset)
                    result = segment_source_document(
                        SourceDocumentRecord(
                            source_document_id_hash=source_id,
                            split="train",
                            raw_text=text,
                            source_lineage_hash=_physical_row_key(entry, offset),
                            source_token_count_hint=_metadata_token_count(row),
                        )
                    )
                    store_segmentation_result(state, result)
                    if result.receipt.document_status != "segmented":
                        outputs = _terminal_filtering_outputs(result, config_hash=str(config_payload["config_hash"]))
                    else:
                        raw_text = str(text)
                        candidates = build_filtering_candidates(
                            result,
                            raw_text=raw_text,
                            config_hash=str(config_payload["config_hash"]),
                            target_proxy_tokens=int(config_payload["target_proxy_tokens"]),
                            halo_chars=int(config_payload["halo_chars"]),
                            deterministic_policy_hash=(
                                str(configured_deterministic_hash)
                                if configured_deterministic_hash is not None
                                else None
                            ),
                        )
                        outputs = build_chunk_filtering_outputs_for_document(
                            result,
                            raw_text=raw_text,
                            candidates=candidates,
                            scorers=scorers,
                            thresholds=thresholds,
                            config_hash=str(config_payload["config_hash"]),
                            source_chunk_manifest_hash=source_chunk_manifest_hash or str(config_payload["classifier_freeze_hash"]),
                            deterministic_policy_payload=deterministic_policy_payload,
                            deterministic_ontology=deterministic_ontology,
                        )
                    store_chunk_filtering_outputs(state, outputs)
                    validate_chunked_filtering_state(state)
                    for private_row in outputs.get("private_clean_run_rows") or []:
                        private_writer(private_row)
                    report = dict(outputs["document_report"])
                    doc_reports.append(report)
                    route_counts.update(report["decision_counts"])
                    accepted_lower_bound += int(report["accepted_estimated_token_count_lower_bound"])
                if max_rows is not None and rows_seen >= int(max_rows):
                    break
        validate_chunked_filtering_state(state)
    report = {
        "artifact_kind": CHUNK_FILTER_RUN_REPORT_KIND,
        "schema_version": 1,
        "status": "complete" if not error_counts else "blocked_errors",
        "config_hash": config_payload["config_hash"],
        "source_metadata_hash": required_source_hash,
        "source_schedule_hash": schedule.get("schedule_hash"),
        "processed_physical_row_count": rows_seen,
        "document_report_count": len(doc_reports),
        "route_counts": _sorted_counter(route_counts),
        "error_counts": _sorted_counter(error_counts),
        "accepted_estimated_token_count_lower_bound": int(accepted_lower_bound),
        "private_clean_run_jsonl_hash": _sha256_file(private_clean_path) if private_clean_path.exists() else None,
        "elapsed_seconds": round(time.monotonic() - start, 6),
        "mechanically_valid": len(error_counts) == 0,
        "yield_viable": False,
        "review_clean": False,
        "full_run_ready": False,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    if configured_deterministic_hash is not None:
        report["deterministic_filter_policy_hash"] = configured_deterministic_hash
    report["report_hash"] = stable_json_hash({key: value for key, value in report.items() if key != "report_hash"})
    validate_public_no_raw_payload(report)
    _write_json_atomic(public_report, report)
    return report


def _terminal_filtering_outputs(result: SegmentationResult, *, config_hash: str) -> dict[str, Any]:
    source_id = result.source.source_document_id_hash
    decision = result.receipt.document_status
    status = "error" if decision in {"missing_text", "non_string_text"} else "quarantine"
    report = {
        "artifact_kind": CHUNK_FILTER_DOCUMENT_REPORT_KIND,
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
        candidate_policy["deterministic_filter_policy_hash"] = _require_sha256(
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
        "core_text_sha256": _sha256_text(core_text),
        "evidence_text_sha256": _sha256_text(evidence_text),
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
    validate_public_no_raw_payload(payload)
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
        _require_sha256(candidate.get(field), field)
    core_start = _nonnegative_int(candidate.get("core_char_start"), "core_char_start")
    core_end = _positive_int(candidate.get("core_char_end"), "core_char_end")
    evidence_start = _nonnegative_int(candidate.get("evidence_char_start"), "evidence_char_start")
    evidence_end = _positive_int(candidate.get("evidence_char_end"), "evidence_char_end")
    core_proxy_tokens = _nonnegative_int(candidate.get("core_proxy_token_count"), "core_proxy_token_count")
    evidence_proxy_tokens = _nonnegative_int(candidate.get("evidence_proxy_token_count"), "evidence_proxy_token_count")
    if evidence_proxy_tokens < core_proxy_tokens:
        raise ValueError("candidate evidence_proxy_token_count must cover core_proxy_token_count")
    if not evidence_start <= core_start < core_end <= evidence_end <= len(raw_text):
        raise ValueError("candidate core/evidence spans are invalid")
    if _sha256_text(raw_text[core_start:core_end]) != candidate["core_text_sha256"]:
        raise ValueError("candidate core text hash mismatch")
    if _sha256_text(raw_text[evidence_start:evidence_end]) != candidate["evidence_text_sha256"]:
        raise ValueError("candidate evidence text hash mismatch")


def _validate_candidate_block_coverage(result: SegmentationResult, candidates: Sequence[Mapping[str, Any]]) -> None:
    expected = list(range(len(result.blocks)))
    covered: list[int] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = _require_sha256(candidate.get("candidate_id_hash"), "candidate_id_hash")
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
        "candidate_id": _require_sha256(candidate_id, "candidate_id"),
        "candidate_id_hash": _require_sha256(candidate_id, "candidate_id_hash"),
        "source_chunk_manifest_hash": _require_sha256(source_chunk_manifest_hash, "source_chunk_manifest_hash"),
        "classifier_decision": status,
        "model_scores": {str(key): _runtime_score(value) for key, value in sorted(model_scores.items())},
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
    validate_public_no_raw_payload(payload)
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
    validate_public_no_raw_payload(payload)
    return payload


def _deterministic_filter_decision_fields(deterministic_filter_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "deterministic_filter_policy_hash": _require_sha256(
            deterministic_filter_result.get("deterministic_filter_policy_hash"),
            "deterministic_filter_policy_hash",
        ),
        "deterministic_filter_result_hash": _require_sha256(
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
            "clean_run_text_sha256": _sha256_text(private_text),
        }
    )
    public = {
        "clean_run_id_hash": clean_run_id,
        "source_document_id_hash": result.source.source_document_id_hash,
        "segmentation_receipt_hash": result.receipt.receipt_hash,
        "clean_run_index": int(clean_run_index),
        "block_indexes": block_indexes,
        "block_id_hashes": [str(block.block_id_hash) for block in blocks],
        "clean_run_text_sha256": _sha256_text(private_text),
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
    validate_public_no_raw_payload(public)
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
    validate_public_no_raw_payload(payload)
    return payload


def _ensure_filtering_state_schema(conn: sqlite3.Connection) -> None:
    columns = {
        "classifier_candidates": {
            "source_document_id_hash": "text",
            "segmentation_receipt_hash": "text",
            "candidate_policy_hash": "text",
            "core_block_indexes_json": "text",
            "core_char_start": "integer",
            "core_char_end": "integer",
            "evidence_char_start": "integer",
            "evidence_char_end": "integer",
            "core_text_sha256": "text",
            "evidence_text_sha256": "text",
            "candidate_json": "text",
        },
        "chunk_decisions": {
            "candidate_id_hash": "text",
            "source_document_id_hash": "text",
            "decision_json": "text",
        },
        "gap_events": {
            "candidate_id_hash": "text",
            "source_document_id_hash": "text",
            "block_indexes_json": "text",
            "gap_json": "text",
        },
        "clean_runs": {
            "source_document_id_hash": "text",
            "block_indexes_json": "text",
            "clean_run_json": "text",
        },
    }
    with conn:
        for table, table_columns in columns.items():
            existing = {str(row[1]) for row in conn.execute(f"pragma table_info({table})")}
            for column, ddl in table_columns.items():
                if column not in existing:
                    conn.execute(f"alter table {table} add column {column} {ddl}")


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
        sparse_model_id=str(config.get("sparse_model_id", CHUNK_SPARSE_MODEL_ID)),
        window_mil_model_id=str(config.get("window_mil_model_id", CHUNK_WINDOW_MIL_MODEL_ID)),
        window_mil_input_mode=str(config.get("window_mil_input_mode", WINDOW_INPUT_FULL_EVIDENCE)),
        window_mil_pooling_mode=str(config.get("window_mil_pooling_mode", "topk_mean_logits")),
        window_mil_pooling_top_k=int(config.get("window_mil_pooling_top_k", 3)),
        window_mil_precision=normalize_w1_precision_mode(str(config.get("window_mil_precision", LOCKED_W1_PRECISION))),
        deterministic_filter_policy_hash=(
            str(config["deterministic_filter_policy_hash"])
            if config.get("deterministic_filter_policy_hash") is not None
            else None
        ),
        target_proxy_tokens=int(config.get("target_proxy_tokens", DEFAULT_TARGET_PROXY_TOKENS)),
        halo_chars=int(config.get("halo_chars", DEFAULT_HALO_CHARS)),
        canary_high_token_rows=int(config.get("canary_high_token_rows", DEFAULT_CANARY_HIGH_TOKEN_ROWS)),
        canary_random_rows=int(config.get("canary_random_rows", DEFAULT_CANARY_RANDOM_ROWS)),
        accepted_token_target_lower_bound=int(
            config.get("accepted_token_target_lower_bound", DEFAULT_ACCEPTED_TOKEN_TARGET_LOWER_BOUND)
        ),
        canary_budget_cap_usd=float(config.get("canary_budget_cap_usd", DEFAULT_CANARY_BUDGET_CAP_USD)),
        full_run_budget_cap_usd=float(config.get("full_run_budget_cap_usd", DEFAULT_FULL_RUN_BUDGET_CAP_USD)),
    )


def _public_selected_row(item: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "selection_index": int(index),
        "slice": str(item["slice"]),
        "physical_row_key_hash": str(item["physical_row_key_hash"]),
        "source_file_id_hash": stable_json_hash(str(item["source_file_id"])),
        "row_group_index": int(item["row_group_index"]),
        "row_group_local_row_index": int(item["offset"]),
        "token_count": item["token_count"],
        "token_count_valid": bool(item["token_count_valid"]),
    }


def _public_schedule_entries_from_selected(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[int]] = {}
    for item in selected:
        key = (stable_json_hash(str(item["source_file_id"])), int(item["row_group_index"]))
        grouped.setdefault(key, []).append(int(item["offset"]))
    entries = []
    for index, ((source_file_id_hash, row_group_index), offsets) in enumerate(sorted(grouped.items())):
        entry = {
            "global_schedule_index": int(index),
            "source_file_id_hash": source_file_id_hash,
            "row_group_index": row_group_index,
            "row_selection_mode": "explicit_offsets",
            "row_group_local_offsets": sorted(set(offsets)),
        }
        entry["entry_hash"] = stable_json_hash(entry)
        entries.append(entry)
    return entries


def _source_entries_by_group(source_schedule: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    entries: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in source_schedule.get("entries") or []:
        key = (str(entry.get("source_file_id")), int(entry.get("row_group_index")))
        entries[key] = dict(entry)
    return entries


def _freeze_thresholds_by_model_id(freeze_report: Mapping[str, Any]) -> dict[str, float]:
    thresholds = freeze_report.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("freeze report missing thresholds")
    aliases = {
        CHUNK_SPARSE_MODEL_ID: (CHUNK_SPARSE_MODEL_ID, "sparse"),
        CHUNK_WINDOW_MIL_MODEL_ID: (CHUNK_WINDOW_MIL_MODEL_ID, "window_mil"),
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
    policy = freeze_report.get("selected_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("selected policy freeze report missing selected_policy")
    return {
        CHUNK_SPARSE_MODEL_ID: _threshold(policy.get("sparse_threshold"), f"threshold.{CHUNK_SPARSE_MODEL_ID}"),
        CHUNK_WINDOW_MIL_MODEL_ID: _threshold(
            policy.get("window_mil_threshold"),
            f"threshold.{CHUNK_WINDOW_MIL_MODEL_ID}",
        ),
    }


def _validate_selected_policy_freeze_report(
    freeze_report: Mapping[str, Any],
    config_payload: Mapping[str, Any] | None = None,
) -> None:
    validate_public_no_raw_payload(freeze_report)
    if freeze_report.get("artifact_kind") != SELECTED_POLICY_FREEZE_REPORT_KIND:
        raise ValueError("unsupported selected policy freeze report kind")
    if freeze_report.get("status") != "ready":
        raise ValueError("selected policy freeze report must be ready")
    _validate_report_self_hash_required(freeze_report, "selected policy freeze report")
    _require_false_flags(freeze_report, PUBLIC_FALSE_FLAGS, "selected policy freeze report")
    if freeze_report.get("classifier_can_approve_corpus") is not False:
        raise ValueError("selected policy freeze report must not approve corpus")
    if freeze_report.get("labels_are_corpus_approval") is not False:
        raise ValueError("selected policy freeze labels must not approve corpus")
    policy = freeze_report.get("selected_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("selected policy freeze report missing selected_policy")
    if freeze_report.get("selected_policy_hash") != stable_json_hash(policy):
        raise ValueError("selected policy hash mismatch")
    if policy.get("policy_family") != "deterministic_terminal_or_sparse_or_window_mil_veto":
        raise ValueError("selected policy family mismatch")
    if policy.get("selection_split") != "calibration":
        raise ValueError("selected policy must be selected on calibration")
    if policy.get("threshold_direction") != "reject_at_or_above":
        raise ValueError("selected policy threshold direction mismatch")
    if policy.get("deterministic_terminal_core_reject_fixed") is not True:
        raise ValueError("selected policy must fix deterministic terminal rejects")
    if policy.get("halo_hits_terminal") is not False:
        raise ValueError("selected policy must keep halo hits nonterminal")
    if policy.get("audit_only_hits_terminal") is not False:
        raise ValueError("selected policy must keep audit-only hits nonterminal")
    if policy.get("classifier_can_approve_corpus") is not False:
        raise ValueError("selected policy must not approve corpus")
    if policy.get("missing_or_non_finite_score_action") != "block_freeze":
        raise ValueError("selected policy must block missing or non-finite scores")
    thresholds = _selected_policy_thresholds_by_model_id(freeze_report)
    for field in (
        "report_hash",
        "selected_policy_hash",
        "classifier_manifest_hash",
        "evidence_cache_hash",
        "deterministic_filter_policy_hash",
    ):
        _require_sha256(freeze_report.get(field), field)
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
    config_precision = normalize_w1_precision_mode(str(config_payload.get("window_mil_precision") or W1_PRECISION_FP32))
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
    if thresholds[CHUNK_SPARSE_MODEL_ID] != float(config_payload["sparse_threshold"]):
        raise ValueError("selected policy freeze sparse threshold mismatch")
    if thresholds[CHUNK_WINDOW_MIL_MODEL_ID] != float(config_payload["window_mil_threshold"]):
        raise ValueError("selected policy freeze Window-MIL threshold mismatch")


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


def _sparse_scorer_metadata(sparse_scorer: ChunkFilteringScorer, candidate: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(sparse_scorer, ChunkSparseRiskScorer):
        return {}
    return _candidate_public_metadata(candidate)


def _sparse_scorer_requires_metadata(sparse_scorer: ChunkFilteringScorer) -> bool:
    return type(sparse_scorer) is not ChunkSparseRiskScorer


def _feature_id_to_column(feature_id: str, *, n_features: int) -> int:
    if not str(feature_id).startswith("h"):
        raise ValueError(f"unsupported feature id: {feature_id}")
    column = int(str(feature_id)[1:], 16)
    if not 0 <= column < int(n_features):
        raise ValueError(f"feature id out of configured range: {feature_id}")
    return column


def _physical_row_key(entry: Mapping[str, Any], offset: int) -> str:
    return stable_json_hash(
        {
            "dataset_repo_id": entry.get("dataset_repo_id"),
            "config": entry.get("config"),
            "split": entry.get("split"),
            "resolved_repo_sha": entry.get("resolved_repo_sha"),
            "source_file_id": entry.get("source_file_id"),
            "row_group_index": int(entry.get("row_group_index")),
            "offset": int(offset),
        }
    )


def _source_document_id_for_schedule_row(entry: Mapping[str, Any], offset: int) -> str:
    return stable_json_hash(
        {
            "kind": "chunk_filter_source_document",
            "physical_row_key_hash": _physical_row_key(entry, offset),
        }
    )


def _metadata_token_count(row: Mapping[str, Any]) -> int | None:
    try:
        value = int(row.get("token_count"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _estimated_token_lower_bound(text: str) -> int:
    return max(1, len(text.encode("utf-8", errors="surrogatepass")) // DEFAULT_TOKEN_BYTES_PER_TOKEN_LOWER_BOUND)


def _token_count_summary(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None}
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": int(sum(ordered) / len(ordered)),
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
    }


def _threshold(value: Any, label: str) -> float:
    try:
        score = _runtime_score(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric and finite") from exc
    if not 0 <= score <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return score


def _runtime_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("classifier score must be numeric") from exc
    if not math.isfinite(score):
        raise ValueError("classifier score must be finite")
    if score < 0 or score > 1:
        raise ValueError("classifier score must be in [0, 1]")
    return score


def _safe_runtime_score(value: Any) -> float | None:
    try:
        return _runtime_score(value)
    except ValueError:
        return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _install_sklearn_loss_pickle_compat_alias() -> None:
    """Support sparse artifacts pickled with sklearn's bare compiled _loss module."""

    import importlib
    import sys

    sys.modules.setdefault("_loss", importlib.import_module("sklearn._loss._loss"))


def _require_sha256(value: Any, label: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{label} must be a lowercase sha256 hex digest")
    return candidate


def _positive_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be nonnegative and finite")
    return result


def _require_false_flags(payload: Mapping[str, Any], fields: Sequence[str], context: str) -> None:
    for field in fields:
        if field in payload and payload[field] is not False:
            raise ValueError(f"{context} must keep {field}=false")


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _walk_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    strings: list[str] = []
    if isinstance(value, Mapping):
        for child in value.values():
            strings.extend(_walk_string_values(child))
    elif isinstance(value, list):
        for child in value:
            strings.extend(_walk_string_values(child))
    return strings


class _append_jsonl_atomic:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tmp_path: Path | None = None
        self.handle: Any | None = None
        self.clean_run_ids: set[str] = set()

    def __enter__(self) -> Any:
        existing = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        for line_number, line in enumerate(existing.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"existing private clean-run JSONL is corrupt at line {line_number}") from exc
            clean_run_id = row.get("clean_run_id_hash")
            if isinstance(clean_run_id, str) and clean_run_id:
                if clean_run_id in self.clean_run_ids:
                    raise ValueError("existing private clean-run JSONL contains duplicate clean_run_id_hash")
                self.clean_run_ids.add(clean_run_id)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        self.tmp_path = Path(tmp_name)
        self.handle = os.fdopen(fd, "w", encoding="utf-8")
        if existing:
            self.handle.write(existing)
            if not existing.endswith("\n"):
                self.handle.write("\n")
        return self.write

    def write(self, payload: Mapping[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("JSONL writer is closed")
        clean_run_id = str(payload.get("clean_run_id_hash") or "")
        if clean_run_id:
            _require_sha256(clean_run_id, "private clean_run_id_hash")
            if clean_run_id in self.clean_run_ids:
                return
            self.clean_run_ids.add(clean_run_id)
        self.handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.close()
        if self.tmp_path is None:
            return
        if exc_type is None:
            self.tmp_path.replace(self.path)
        else:
            try:
                self.tmp_path.unlink()
            except FileNotFoundError:
                pass


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        _chmod_private_file(tmp_name)
        Path(tmp_name).replace(output)
        _chmod_private_file(output)
    except Exception:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _write_jsonl_path(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
    _chmod_private_file(output)


def _jsonl_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _write_jsonl_path_streaming(output: str | Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    _chmod_private_file(path)
    return digest.hexdigest(), count


@dataclass(frozen=True)
class _SurvivorQueueWriteProof:
    survivor_queue_hash: str
    survivor_queue_row_count: int
    survivor_queue_id_index_hash: str
    survivor_queue_id_index_row_count: int
    survivor_candidate_id_set_hash: str
    survivor_private_evidence_id_set_hash: str


def _survivor_queue_id_index_base_path(base_path: str | Path) -> Path:
    base = Path(base_path)
    return base.with_suffix(".ids.tsv")


def _survivor_queue_id_index_part_path(base_path: str | Path) -> Path:
    base = _survivor_queue_id_index_base_path(base_path)
    return base.with_suffix(base.suffix + ".part")


def _survivor_queue_id_index_ready_path(base_path: str | Path) -> Path:
    base = _survivor_queue_id_index_base_path(base_path)
    return base.with_suffix(base.suffix + ".ready")


def _write_survivor_queue_ready_streaming(base_path: str | Path, rows: Iterable[Mapping[str, Any]]) -> _SurvivorQueueWriteProof:
    base = Path(base_path)
    part = base.with_suffix(base.suffix + ".part")
    ready = base.with_suffix(base.suffix + ".ready")
    id_index_part = _survivor_queue_id_index_part_path(base)
    id_index_ready = _survivor_queue_id_index_ready_path(base)
    part.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    id_index_digest = hashlib.sha256()
    candidate_ids: set[str] = set()
    evidence_ids: set[str] = set()
    count = 0
    with part.open("w", encoding="utf-8") as handle, id_index_part.open("w", encoding="ascii") as id_handle:
        for row in rows:
            candidate_id = _require_sha256(row.get("candidate_id_hash"), "survivor queue candidate_id_hash")
            evidence_id = _require_sha256(row.get("private_evidence_row_id"), "survivor queue private_evidence_row_id")
            if candidate_id in candidate_ids:
                raise ValueError("survivor queue contains duplicate candidate ids")
            if evidence_id in evidence_ids:
                raise ValueError("survivor queue contains duplicate private evidence ids")
            candidate_ids.add(candidate_id)
            evidence_ids.add(evidence_id)
            line = json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            id_line = f"{candidate_id}\t{evidence_id}\n"
            id_handle.write(id_line)
            id_index_digest.update(id_line.encode("ascii"))
            count += 1
    _chmod_private_file(part)
    _chmod_private_file(id_index_part)
    id_index_part.replace(id_index_ready)
    _chmod_private_file(id_index_ready)
    part.replace(ready)
    _chmod_private_file(ready)
    return _SurvivorQueueWriteProof(
        survivor_queue_hash=digest.hexdigest(),
        survivor_queue_row_count=count,
        survivor_queue_id_index_hash=id_index_digest.hexdigest(),
        survivor_queue_id_index_row_count=count,
        survivor_candidate_id_set_hash=stable_json_hash(sorted(candidate_ids)),
        survivor_private_evidence_id_set_hash=stable_json_hash(sorted(evidence_ids)),
    )


def _write_jsonl_ready(base_path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base = Path(base_path)
    part = base.with_suffix(base.suffix + ".part")
    ready = base.with_suffix(base.suffix + ".ready")
    _write_jsonl_path(part, rows)
    part.replace(ready)
    _chmod_private_file(ready)


def _write_jsonl_ready_streaming(base_path: str | Path, rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    base = Path(base_path)
    part = base.with_suffix(base.suffix + ".part")
    ready = base.with_suffix(base.suffix + ".ready")
    digest, count = _write_jsonl_path_streaming(part, rows)
    part.replace(ready)
    _chmod_private_file(ready)
    return digest, count


def _iter_ready_jsonl(base_path: str | Path, label: str) -> Iterator[dict[str, Any]]:
    base = Path(base_path)
    part = base.with_suffix(base.suffix + ".part")
    ready = base.with_suffix(base.suffix + ".ready")
    if part.exists():
        raise ValueError(f"{label} has untrusted .part export")
    if not ready.exists():
        raise ValueError(f"{label} is missing ready export")
    with ready.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            yield payload


def _load_ready_jsonl(base_path: str | Path, label: str) -> list[dict[str, Any]]:
    return list(_iter_ready_jsonl(base_path, label))


def _validate_survivor_queue_rows(
    rows: Sequence[Mapping[str, Any]],
    config_payload: Mapping[str, Any],
    *,
    allow_empty: bool = False,
) -> None:
    if not rows and not allow_empty:
        raise ValueError("survivor queue is empty or missing survivors")
    seen: set[str] = set()
    for row in rows:
        _validate_survivor_queue_row(row, config_payload, seen)


def _validate_survivor_queue_ready_file(
    base_path: str | Path,
    config_payload: Mapping[str, Any],
    *,
    allow_empty: bool = False,
    seen_survivor_candidate_ids: set[str] | None = None,
    seen_private_evidence_ids: set[str] | None = None,
) -> int:
    seen: set[str] = set()
    count = 0
    for row in _iter_ready_jsonl(base_path, "prefilter survivor queue"):
        _validate_survivor_queue_row(row, config_payload, seen)
        candidate_id = _require_sha256(row.get("candidate_id_hash"), "candidate_id_hash")
        evidence_id = _require_sha256(row.get("private_evidence_row_id"), "private_evidence_row_id")
        if seen_survivor_candidate_ids is not None:
            if candidate_id in seen_survivor_candidate_ids:
                raise ValueError("sharded W1 handoff contains duplicate survivor candidate ids")
            seen_survivor_candidate_ids.add(candidate_id)
        if seen_private_evidence_ids is not None:
            if evidence_id in seen_private_evidence_ids:
                raise ValueError("sharded W1 handoff contains duplicate private evidence ids")
            seen_private_evidence_ids.add(evidence_id)
        count += 1
    if count == 0 and not allow_empty:
        raise ValueError("survivor queue is empty or missing survivors")
    return count


def _validate_survivor_queue_id_index_ready_file(
    base_path: str | Path,
    *,
    allow_empty: bool = False,
    expected_hash: str | None = None,
    expected_row_count: int | None = None,
    expected_candidate_id_set_hash: str | None = None,
    expected_private_evidence_id_set_hash: str | None = None,
    seen_survivor_candidate_ids: set[str] | None = None,
    seen_private_evidence_ids: set[str] | None = None,
) -> int:
    part = _survivor_queue_id_index_part_path(base_path)
    ready = _survivor_queue_id_index_ready_path(base_path)
    if part.exists():
        raise ValueError("prefilter survivor queue ID index has untrusted .part export")
    if not ready.exists():
        raise ValueError("prefilter survivor queue ID index is missing ready export")
    if expected_hash is not None and _require_sha256(expected_hash, "survivor_queue_id_index_hash") != _sha256_file(ready):
        raise ValueError("survivor queue ID index hash mismatch")
    local_candidates: set[str] = set()
    local_evidence: set[str] = set()
    count = 0
    with ready.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) != 2:
                raise ValueError(f"prefilter survivor queue ID index row {line_number} is malformed")
            candidate_id = _require_sha256(parts[0], "survivor queue ID index candidate_id_hash")
            evidence_id = _require_sha256(parts[1], "survivor queue ID index private_evidence_row_id")
            if candidate_id in local_candidates:
                raise ValueError("survivor queue ID index contains duplicate candidate ids")
            if evidence_id in local_evidence:
                raise ValueError("survivor queue ID index contains duplicate private evidence ids")
            local_candidates.add(candidate_id)
            local_evidence.add(evidence_id)
            if seen_survivor_candidate_ids is not None:
                if candidate_id in seen_survivor_candidate_ids:
                    raise ValueError("sharded W1 handoff contains duplicate survivor candidate ids")
                seen_survivor_candidate_ids.add(candidate_id)
            if seen_private_evidence_ids is not None:
                if evidence_id in seen_private_evidence_ids:
                    raise ValueError("sharded W1 handoff contains duplicate private evidence ids")
                seen_private_evidence_ids.add(evidence_id)
            count += 1
    if count == 0 and not allow_empty:
        raise ValueError("survivor queue ID index is empty or missing survivors")
    if expected_row_count is not None and count != int(expected_row_count):
        raise ValueError("survivor queue ID index row count mismatch")
    if (
        expected_candidate_id_set_hash is not None
        and _require_sha256(expected_candidate_id_set_hash, "survivor_candidate_id_set_hash")
        != stable_json_hash(sorted(local_candidates))
    ):
        raise ValueError("survivor queue ID index candidate set hash mismatch")
    if (
        expected_private_evidence_id_set_hash is not None
        and _require_sha256(expected_private_evidence_id_set_hash, "survivor_private_evidence_id_set_hash")
        != stable_json_hash(sorted(local_evidence))
    ):
        raise ValueError("survivor queue ID index private evidence set hash mismatch")
    return count


def _validate_survivor_queue_row(
    row: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    seen: set[str],
) -> None:
    validate_public_no_raw_payload(row)
    candidate_id = _require_sha256(row.get("candidate_id_hash"), "candidate_id_hash")
    if candidate_id in seen:
        raise ValueError("survivor queue contains duplicate candidate ids")
    seen.add(candidate_id)
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "s1_report_hash",
        "s1_model_file_sha256",
        "w1_report_hash",
        "w1_model_file_sha256",
    ):
        expected = {
            "s1_report_hash": "sparse_report_hash",
            "s1_model_file_sha256": "sparse_model_file_sha256",
            "w1_report_hash": "window_mil_report_hash",
            "w1_model_file_sha256": "window_mil_model_file_sha256",
        }.get(field, field)
        if row.get(field) != config_payload.get(expected):
            raise ValueError(f"survivor queue {field} mismatch")
    if float(row.get("s1_threshold")) != float(config_payload["sparse_threshold"]):
        raise ValueError("survivor queue S1 threshold mismatch")
    if float(row.get("w1_threshold")) != float(config_payload["window_mil_threshold"]):
        raise ValueError("survivor queue W1 threshold mismatch")
    core_proxy_token_count = _nonnegative_int(row.get("core_proxy_token_count"), "survivor queue core_proxy_token_count")
    if row.get("evidence_proxy_token_count") is None:
        evidence_proxy_token_count = core_proxy_token_count
    else:
        evidence_proxy_token_count = _nonnegative_int(
            row.get("evidence_proxy_token_count"),
            "survivor queue evidence_proxy_token_count",
        )
        if evidence_proxy_token_count < core_proxy_token_count:
            raise ValueError("survivor queue evidence_proxy_token_count must cover core_proxy_token_count")
    expected_window_count = max(1, math.ceil(evidence_proxy_token_count / 512))
    if _positive_int(row.get("estimated_w1_window_count"), "survivor queue estimated_w1_window_count") != expected_window_count:
        raise ValueError("survivor queue estimated_w1_window_count mismatch")
    _require_sha256(row.get("private_evidence_row_id"), "private_evidence_row_id")
    _require_sha256(row.get("stage_receipt_hash"), "stage_receipt_hash")
    if row.get("stage_receipt_hash") != stable_json_hash(
        {key: value for key, value in row.items() if key != "stage_receipt_hash"}
    ):
        raise ValueError("survivor queue row receipt hash mismatch")


def _validate_w1_score_rows(
    rows: Sequence[Mapping[str, Any]],
    config_payload: Mapping[str, Any],
    *,
    survivor_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    allow_empty: bool = False,
) -> None:
    if not rows and not allow_empty:
        raise ValueError("W1 score export is empty")
    seen: set[str] = set()
    for row in rows:
        validate_public_no_raw_payload(row)
        candidate_id = _require_sha256(row.get("candidate_id_hash"), "candidate_id_hash")
        if candidate_id in seen:
            raise ValueError("W1 score export contains duplicate candidate ids")
        seen.add(candidate_id)
        if row.get("config_hash") != config_payload["config_hash"]:
            raise ValueError("W1 score config hash mismatch")
        if row.get("selected_policy_hash") != config_payload["selected_policy_hash"]:
            raise ValueError("W1 score selected policy mismatch")
        if row.get("selected_policy_report_hash") != config_payload["selected_policy_report_hash"]:
            raise ValueError("W1 score selected policy report mismatch")
        if row.get("source_metadata_hash") != config_payload["source_metadata_hash"]:
            raise ValueError("W1 score source metadata hash mismatch")
        if row.get("w1_report_hash") != config_payload["window_mil_report_hash"]:
            raise ValueError("W1 score report hash mismatch")
        if row.get("w1_model_file_sha256") != config_payload["window_mil_model_file_sha256"]:
            raise ValueError("W1 score model hash mismatch")
        _require_sha256(row.get("w1_runtime_envelope_hash"), "w1_runtime_envelope_hash")
        if float(row.get("w1_threshold")) != float(config_payload["window_mil_threshold"]):
            raise ValueError("W1 score threshold mismatch")
        if row.get("w1_route") not in {"w1_pass", "w1_reject", "w1_error"}:
            raise ValueError("W1 score route mismatch")
        score = row.get("w1_score")
        if row.get("w1_route") == "w1_error":
            if score is not None or not row.get("error_class"):
                raise ValueError("W1 error score row must carry only an error_class")
        else:
            parsed = _safe_runtime_score(score)
            if parsed is None:
                raise ValueError("W1 non-error score row must carry a finite score")
            if row.get("w1_route") == "w1_pass" and parsed >= float(config_payload["window_mil_threshold"]):
                raise ValueError("W1 score route contradicts threshold")
            if row.get("w1_route") == "w1_reject" and parsed < float(config_payload["window_mil_threshold"]):
                raise ValueError("W1 score route contradicts threshold")
        if survivor_by_id is not None:
            survivor = survivor_by_id.get(candidate_id)
            if survivor is None:
                raise ValueError("W1 score row missing matching survivor queue row")
            if row.get("survivor_stage_receipt_hash") != survivor.get("stage_receipt_hash"):
                raise ValueError("W1 score survivor receipt hash mismatch")
        _require_sha256(row.get("stage_receipt_hash"), "stage_receipt_hash")
        if row.get("stage_receipt_hash") != stable_json_hash(
            {key: value for key, value in row.items() if key != "stage_receipt_hash"}
        ):
            raise ValueError("W1 score row receipt hash mismatch")


def _prefilter_receipt_has_survivor_queue_id_proof(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("survivor_queue_proof_kind") != "jsonl_sha256_and_id_set_hashes_v1":
        return False
    return (
        receipt.get("survivor_queue_row_count") is not None
        and receipt.get("survivor_candidate_id_set_hash") is not None
        and receipt.get("survivor_private_evidence_id_set_hash") is not None
    )


def _prefilter_receipt_has_survivor_queue_id_index_proof(receipt: Mapping[str, Any]) -> bool:
    return (
        _prefilter_receipt_has_survivor_queue_id_proof(receipt)
        and receipt.get("survivor_queue_id_index_hash") is not None
        and receipt.get("survivor_queue_id_index_row_count") is not None
    )


def _validate_prefilter_ready_receipt_payload(
    receipt: Mapping[str, Any],
    run_path: Path,
    config_payload: Mapping[str, Any],
    *,
    queue_row_count: int | None = None,
    expected_status: str = "prefilter_ready_for_w1",
) -> None:
    validate_public_no_raw_payload(receipt)
    if receipt.get("artifact_kind") != PREFILTER_CPU_STAGE_KIND:
        raise ValueError("prefilter ready receipt kind mismatch")
    if expected_status not in {"prefilter_ready_for_w1", "prefilter_shard_ready_for_merge", "compact_fresh_import"}:
        raise ValueError("unsupported prefilter ready receipt status expectation")
    if receipt.get("status") != expected_status:
        raise ValueError("prefilter stage is not ready for the requested next phase")
    if receipt.get("stage_receipt_hash") != stable_json_hash(
        {key: value for key, value in receipt.items() if key != "stage_receipt_hash"}
    ):
        raise ValueError("prefilter ready receipt hash mismatch")
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "selected_eval_deterministic_filter_policy_hash",
    ):
        if receipt.get(field) != config_payload.get(field):
            raise ValueError(f"prefilter ready receipt {field} mismatch")
    if receipt.get("executed_deterministic_filter_policy_hash") != config_payload.get("deterministic_filter_policy_hash"):
        raise ValueError("prefilter ready receipt deterministic policy mismatch")
    route_counts = {str(key): int(value) for key, value in dict(receipt.get("route_counts") or {}).items()}
    if int(receipt.get("candidate_count", -1)) != sum(route_counts.values()):
        raise ValueError("prefilter ready receipt candidate count does not match route_counts")
    if int(receipt.get("survivor_count", -1)) != int(route_counts.get("prefilter_survivor", 0)):
        raise ValueError("prefilter ready receipt survivor count does not match route_counts")
    if queue_row_count is not None and int(receipt.get("survivor_count", -1)) != int(queue_row_count):
        raise ValueError("prefilter ready receipt survivor count mismatch")
    if receipt.get("survivor_queue_ready") is not True or receipt.get("survivor_queue_part") is not False:
        raise ValueError("prefilter ready receipt does not prove a ready survivor queue")
    ready_queue_path = run_path / "w1_survivor_queue.jsonl.ready"
    if receipt.get("survivor_queue_hash") != _sha256_file(ready_queue_path):
        raise ValueError("prefilter ready receipt survivor queue hash mismatch")
    if _prefilter_receipt_has_survivor_queue_id_proof(receipt):
        if int(receipt.get("survivor_queue_row_count", -1)) != int(receipt.get("survivor_count", -2)):
            raise ValueError("prefilter ready receipt survivor queue row count mismatch")
        _require_sha256(receipt.get("survivor_candidate_id_set_hash"), "survivor_candidate_id_set_hash")
        _require_sha256(receipt.get("survivor_private_evidence_id_set_hash"), "survivor_private_evidence_id_set_hash")
        if _prefilter_receipt_has_survivor_queue_id_index_proof(receipt):
            ready_index_path = _survivor_queue_id_index_ready_path(run_path / "w1_survivor_queue.jsonl")
            if receipt.get("survivor_queue_id_index_hash") != _sha256_file(ready_index_path):
                raise ValueError("prefilter ready receipt survivor queue ID index hash mismatch")
            if int(receipt.get("survivor_queue_id_index_row_count", -1)) != int(receipt.get("survivor_count", -2)):
                raise ValueError("prefilter ready receipt survivor queue ID index row count mismatch")


def _validate_w1_compact_receipt_proof_payload(
    receipt: Mapping[str, Any],
    run_path: Path,
    config_payload: Mapping[str, Any],
    *,
    queue_row_count: int | None = None,
    expected_status: str = "compact_fresh_import",
    require_empty_w1_scores: bool = False,
) -> None:
    """Validate compact W1-start receipt proof without rereading full private state.

    Compact export already performs queue/SQLite/private-evidence validation while
    building each shard. Global handoff and launch preflight can rely on the
    compact receipt self-hash plus the compact ID-index proof; shard scoring
    still performs deep queue/SQLite validation before reading private evidence.
    """
    validate_public_no_raw_payload(receipt)
    if receipt.get("artifact_kind") != PREFILTER_CPU_STAGE_KIND:
        raise ValueError("compact start receipt kind mismatch")
    if receipt.get("status") != expected_status:
        raise ValueError("compact start receipt status mismatch")
    if receipt.get("compact_start_state_artifact_kind") != W1_COMPACT_START_STATE_KIND:
        raise ValueError("compact start receipt artifact kind mismatch")
    if receipt.get("compact_start_state_status") != "compact_fresh_import":
        raise ValueError("compact start state status mismatch")
    if receipt.get("stage_receipt_hash") != stable_json_hash(
        {key: value for key, value in receipt.items() if key != "stage_receipt_hash"}
    ):
        raise ValueError("compact start receipt hash mismatch")
    for field in (
        "config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "selected_eval_deterministic_filter_policy_hash",
    ):
        if receipt.get(field) != config_payload.get(field):
            raise ValueError(f"compact start receipt {field} mismatch")
    if receipt.get("executed_deterministic_filter_policy_hash") != config_payload.get("deterministic_filter_policy_hash"):
        raise ValueError("compact start receipt deterministic policy mismatch")
    route_counts = {str(key): int(value) for key, value in dict(receipt.get("route_counts") or {}).items()}
    survivor_count = int(receipt.get("survivor_count", -1))
    if int(receipt.get("candidate_count", -1)) != sum(route_counts.values()):
        raise ValueError("compact start receipt candidate count does not match route_counts")
    if survivor_count != int(route_counts.get("prefilter_survivor", 0)):
        raise ValueError("compact start receipt survivor count does not match route_counts")
    if queue_row_count is not None and survivor_count != int(queue_row_count):
        raise ValueError("compact start receipt survivor count mismatch")
    if int(receipt.get("compact_candidate_count", -1)) != survivor_count:
        raise ValueError("compact start receipt compact candidate count mismatch")
    if int(receipt.get("compact_private_evidence_count", -1)) != survivor_count:
        raise ValueError("compact start receipt compact private evidence count mismatch")
    if bool(require_empty_w1_scores) and int(receipt.get("compact_w1_scores_count", -1)) != 0:
        raise ValueError("compact start receipt must prove empty w1_scores")
    if receipt.get("compact_db_contains_only_survivor_prefilter_candidates") is not True:
        raise ValueError("compact start receipt survivor-only DB flag mismatch")
    if receipt.get("compact_prefilter_source_rows_omitted") is not True:
        raise ValueError("compact start receipt source-row omission flag mismatch")
    if receipt.get("survivor_queue_ready") is not True or receipt.get("survivor_queue_part") is not False:
        raise ValueError("compact start receipt does not prove a ready survivor queue")
    if require_empty_w1_scores and _w1_compact_forbidden_fresh_files(run_path):
        raise ValueError("compact start state contains stale W1 output or STOP_REQUESTED")
    ready_queue_path = run_path / "w1_survivor_queue.jsonl.ready"
    if not ready_queue_path.exists():
        raise ValueError("compact start survivor queue is missing")
    if not _prefilter_receipt_has_survivor_queue_id_index_proof(receipt):
        raise ValueError("compact start receipt requires survivor queue ID-index proof")
    ready_index_path = _survivor_queue_id_index_ready_path(run_path / "w1_survivor_queue.jsonl")
    if receipt.get("survivor_queue_id_index_hash") != _sha256_file(ready_index_path):
        raise ValueError("compact start receipt survivor queue ID index hash mismatch")
    if int(receipt.get("survivor_queue_id_index_row_count", -1)) != survivor_count:
        raise ValueError("compact start receipt survivor queue ID index row count mismatch")
    if int(receipt.get("survivor_queue_row_count", -1)) != survivor_count:
        raise ValueError("compact start receipt survivor queue row count mismatch")
    for field in (
        "source_prefilter_stage_receipt_hash",
        "compact_sqlite_sha256",
        "survivor_queue_hash",
        "survivor_candidate_id_set_hash",
        "survivor_private_evidence_id_set_hash",
    ):
        _require_sha256(receipt.get(field), field)


def _validate_prefilter_ready_receipt(
    conn: sqlite3.Connection,
    run_path: Path,
    config_payload: Mapping[str, Any],
    *,
    queue_rows: Sequence[Mapping[str, Any]] | None = None,
    queue_row_count: int | None = None,
    expected_status: str = "prefilter_ready_for_w1",
    require_compact_w1_scores_empty: bool = True,
) -> None:
    receipt_path = run_path / "prefilter_cpu_receipt.json"
    if not receipt_path.exists():
        raise ValueError("prefilter ready receipt is missing")
    receipt = _load_json(receipt_path)
    if not isinstance(receipt, Mapping):
        raise ValueError("prefilter ready receipt must be an object")
    if queue_rows is None and queue_row_count is None:
        raise ValueError("prefilter ready receipt validation requires survivor queue evidence")
    survivor_queue_count = len(queue_rows) if queue_rows is not None else int(queue_row_count or 0)
    _validate_prefilter_ready_receipt_payload(
        receipt,
        run_path,
        config_payload,
        queue_row_count=survivor_queue_count,
        expected_status=expected_status,
    )
    if _is_w1_compact_prefilter_receipt(receipt):
        if queue_rows is None:
            raise ValueError("compact start state validation requires survivor queue rows")
        _validate_w1_compact_ready_db(
            conn,
            run_path,
            config_payload,
            receipt=receipt,
            queue_rows=queue_rows,
            require_empty_w1_scores=require_compact_w1_scores_empty,
        )
        return
    route_counts = Counter(
        str(row["prefilter_route"]) for row in conn.execute("select prefilter_route from prefilter_candidates")
    )
    candidate_counts = _prefilter_candidate_counts(conn)
    if int(receipt.get("candidate_count", -1)) != int(candidate_counts["candidate_count"]):
        raise ValueError("prefilter ready receipt candidate count mismatch")
    if int(receipt.get("survivor_count", -1)) != int(candidate_counts["survivor_count"]):
        raise ValueError("prefilter ready receipt DB survivor count mismatch")
    if dict(receipt.get("route_counts") or {}) != _sorted_counter(route_counts):
        raise ValueError("prefilter ready receipt route_counts mismatch")
    row_counts = _prefilter_source_row_summary(conn)
    if receipt.get("input_schedule_hash") != row_counts["input_schedule_hash"]:
        raise ValueError("prefilter ready receipt input schedule hash mismatch")
    for field in (
        "expected_physical_rows",
        "expected_physical_row_key_set_hash",
        "scheduled_rows",
        "scheduled_physical_row_key_set_hash",
        "terminal_rows",
        "terminal_physical_row_key_set_hash",
        "pending_rows",
        "pending_physical_row_key_set_hash",
        "fetched_rows",
        "segmented_rows",
        "row_error_counts",
        "unaccounted_row_count",
    ):
        if receipt.get(field) != row_counts[field]:
            raise ValueError(f"prefilter ready receipt {field} mismatch")
    accounted_rows = int(row_counts["terminal_rows"]) + int(row_counts["pending_rows"])
    unvisited_rows = int(row_counts["expected_physical_rows"]) - accounted_rows
    if int(receipt.get("unvisited_rows", -1)) != unvisited_rows:
        raise ValueError("prefilter ready receipt unvisited row mismatch")
    if int(row_counts["pending_rows"]) != 0 or int(row_counts["unaccounted_row_count"]) != 0 or unvisited_rows != 0:
        raise ValueError("prefilter ready receipt does not prove all expected rows reached terminal state")
    if (
        row_counts["expected_physical_row_key_set_hash"] is not None
        and row_counts["terminal_physical_row_key_set_hash"] != row_counts["expected_physical_row_key_set_hash"]
    ):
        raise ValueError("prefilter ready receipt physical row key-set mismatch")


def _directory_size(path: str | Path) -> int:
    root = Path(path)
    total = 0
    if not root.exists():
        return 0
    for item in root.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load_real_scorers(args: argparse.Namespace, config_payload: Mapping[str, Any] | None = None) -> tuple[list[ChunkFilteringScorer], dict[str, float]]:
    sparse_report = _load_json(args.sparse_report)
    window_report = _load_json(args.window_mil_report)
    if config_payload is not None:
        _validate_locked_sparse_runtime_artifact(sparse_report, args.sparse_model, config_payload)
        _validate_locked_w1_runtime_artifact(window_report, args.window_mil_model, config_payload)
    sparse = ChunkSparseRiskScorer.from_report(sparse_report, args.sparse_model)
    window = ChunkWindowMilRiskScorer.from_report(
        window_report,
        args.window_mil_model,
        device=args.device,
        require_cuda=bool(args.require_cuda),
    )
    if config_payload is not None:
        return [sparse, window], {
            CHUNK_SPARSE_MODEL_ID: float(config_payload["sparse_threshold"]),
            CHUNK_WINDOW_MIL_MODEL_ID: float(config_payload["window_mil_threshold"]),
        }
    return [sparse, window], {
        CHUNK_SPARSE_MODEL_ID: float(sparse_report["threshold"]),
        CHUNK_WINDOW_MIL_MODEL_ID: float(window_report["threshold"]),
    }


def _load_sparse_scorer(args: argparse.Namespace, config_payload: Mapping[str, Any]) -> ChunkSparseRiskScorer:
    sparse_report = _load_json(args.sparse_report)
    _validate_locked_sparse_runtime_artifact(sparse_report, args.sparse_model, config_payload)
    return ChunkSparseRiskScorer.from_report(sparse_report, args.sparse_model)


@dataclass(frozen=True)
class _PrefilterLocalParquetCliContext:
    config_payload: dict[str, Any]
    deterministic_ontology: RiskOntology
    deterministic_policy_payload: dict[str, Any]
    sparse_scorer: ChunkSparseRiskScorer


def _load_prefilter_local_parquet_cli_context(
    args: argparse.Namespace,
    *,
    command_name: str,
) -> _PrefilterLocalParquetCliContext:
    config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
    validate_locked_chunk_filtering_config_payload(config_payload)
    _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
    deterministic_ontology = load_ontology(args.deterministic_ontology)
    deterministic_policy_payload = build_deterministic_filter_policy_payload(deterministic_ontology)
    if config_payload.get("deterministic_filter_policy_hash") != deterministic_policy_payload["deterministic_filter_policy_hash"]:
        raise ValueError(f"{command_name} deterministic ontology hash does not match config")
    validate_private_source_cache_dir(args.cache_dir)
    return _PrefilterLocalParquetCliContext(
        config_payload=config_payload,
        deterministic_ontology=deterministic_ontology,
        deterministic_policy_payload=deterministic_policy_payload,
        sparse_scorer=_load_sparse_scorer(args, config_payload),
    )


def _load_w1_scorer(args: argparse.Namespace, config_payload: Mapping[str, Any]) -> ChunkWindowMilRiskScorer:
    window_report = _load_json(args.window_mil_report)
    _validate_locked_w1_runtime_artifact(window_report, args.window_mil_model, config_payload)
    return ChunkWindowMilRiskScorer.from_report(
        window_report,
        args.window_mil_model,
        device=args.device,
        require_cuda=bool(args.require_cuda),
        runtime_max_windows_per_device_batch=getattr(args, "max_windows_per_device_batch", None),
        window_mil_precision=str(config_payload.get("window_mil_precision") or W1_PRECISION_FP32),
        synchronize_cuda_telemetry=bool(getattr(args, "synchronize_cuda_telemetry", False)),
        w1_attn_implementation=str(getattr(args, "w1_attn_implementation", W1_ATTN_IMPLEMENTATION_SDPA)),
    )


def _load_and_validate_selected_policy_report(
    path: str | Path | None,
    config_payload: Mapping[str, Any],
    *,
    required: bool,
) -> dict[str, Any] | None:
    if path is None:
        if required:
            raise ValueError("selected policy freeze report is required for production filtering stages")
        return None
    report = _load_json(path)
    if not isinstance(report, Mapping):
        raise ValueError("selected policy freeze report must be a JSON object")
    _validate_selected_policy_freeze_report(report, config_payload)
    return dict(report)


def _build_schedule_row_provider(
    *,
    row_provider: str,
    cache_dir: str | Path,
    cache_delta_cap_bytes: int,
    source_concurrency: int,
    local_row_group_batch_size: int = 8,
) -> Any:
    if row_provider == "hf-fsspec":
        return HFFsspecParquetRowProvider(
            cache_dir=cache_dir,
            cache_delta_cap_bytes=cache_delta_cap_bytes,
            source_concurrency=source_concurrency,
            manage_hf_cache_env=source_concurrency == 1,
        )
    if row_provider == "local-parquet":
        return LocalParquetRowProvider(
            columns=LOCAL_PARQUET_PREFILTER_COLUMNS,
            row_group_batch_size=local_row_group_batch_size,
        )
    raise ValueError(f"unsupported row provider: {row_provider}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate_config = sub.add_parser("validate-config")
    validate_config.add_argument("--config", required=True)
    validate_config.add_argument("--selected-policy-report", default=None)

    build_schedule = sub.add_parser("build-canary-schedule")
    build_schedule.add_argument("--source-schedule", required=True)
    build_schedule.add_argument("--row-metadata", required=True)
    build_schedule.add_argument("--out", required=True)
    build_schedule.add_argument("--report-out", required=True)
    build_schedule.add_argument("--high-token-rows", type=int, default=DEFAULT_CANARY_HIGH_TOKEN_ROWS)
    build_schedule.add_argument("--random-rows", type=int, default=DEFAULT_CANARY_RANDOM_ROWS)
    build_schedule.add_argument("--seed", type=int, default=20260619)

    mirror_sources = sub.add_parser("mirror-source-files")
    mirror_sources.add_argument("--schedule", required=True)
    mirror_sources.add_argument("--mirror-dir", required=True)
    mirror_sources.add_argument("--local-schedule-out", required=True)
    mirror_sources.add_argument("--report-out", required=True)
    mirror_sources.add_argument("--stop-file", default=None)
    mirror_sources.add_argument("--allow-local-sources-for-tests", action="store_true")

    tranche_plan = sub.add_parser("build-local-parquet-token-tranche-plan")
    tranche_plan.add_argument("--local-schedule", required=True)
    tranche_plan.add_argument("--source-metadata", required=True)
    tranche_plan.add_argument("--estimated-source-token-target", type=int, required=True)
    tranche_plan.add_argument("--seed", required=True)
    tranche_plan.add_argument("--out", required=True)
    tranche_plan.add_argument("--report-out", required=True)

    run = sub.add_parser("run-schedule")
    run.add_argument("--config", required=True)
    run.add_argument("--selected-policy-report", required=True)
    run.add_argument("--schedule", required=True)
    run.add_argument("--source-schedule", default=None)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--private-clean-runs-jsonl", required=True)
    run.add_argument("--public-report", required=True)
    run.add_argument("--row-provider", choices=["hf-fsspec", "local-parquet"], default="hf-fsspec")
    run.add_argument("--cache-dir", required=True)
    run.add_argument("--sparse-report", required=True)
    run.add_argument("--sparse-model", required=True)
    run.add_argument("--window-mil-report", required=True)
    run.add_argument("--window-mil-model", required=True)
    run.add_argument("--deterministic-ontology", default=None)
    run.add_argument("--device", default="auto")
    run.add_argument("--require-cuda", action="store_true")
    run.add_argument("--max-rows", type=int, default=None)
    run.add_argument("--cache-delta-cap-bytes", type=int, default=10 * 1024 * 1024 * 1024)
    run.add_argument("--source-concurrency", type=int, default=1)
    run.add_argument("--local-row-group-batch-size", type=int, default=8)
    run.add_argument("--min-free-bytes", type=int, default=1)

    prefilter_cpu = sub.add_parser("prefilter-cpu")
    prefilter_cpu.add_argument("--config", required=True)
    prefilter_cpu.add_argument("--selected-policy-report", required=True)
    prefilter_cpu.add_argument("--schedule", required=True)
    prefilter_cpu.add_argument("--source-schedule", default=None)
    prefilter_cpu.add_argument("--run-dir", required=True)
    prefilter_cpu.add_argument("--row-provider", choices=["hf-fsspec", "local-parquet"], default="hf-fsspec")
    prefilter_cpu.add_argument("--cache-dir", required=True)
    prefilter_cpu.add_argument("--sparse-report", required=True)
    prefilter_cpu.add_argument("--sparse-model", required=True)
    prefilter_cpu.add_argument("--deterministic-ontology", required=True)
    prefilter_cpu.add_argument("--max-rows", type=int, default=None)
    prefilter_cpu.add_argument("--cache-delta-cap-bytes", type=int, default=10 * 1024 * 1024 * 1024)
    prefilter_cpu.add_argument("--source-concurrency", type=int, default=1)
    prefilter_cpu.add_argument("--local-row-group-batch-size", type=int, default=8)
    prefilter_cpu.add_argument("--min-free-bytes", type=int, default=1)

    prefilter_local_all = sub.add_parser("prefilter-local-parquet-all-rows")
    prefilter_local_all.add_argument("--config", required=True)
    prefilter_local_all.add_argument("--selected-policy-report", required=True)
    prefilter_local_all.add_argument("--local-schedule", required=True)
    prefilter_local_all.add_argument("--source-metadata", required=True)
    prefilter_local_all.add_argument("--run-dir", required=True)
    prefilter_local_all.add_argument("--cache-dir", required=True)
    prefilter_local_all.add_argument("--sparse-report", required=True)
    prefilter_local_all.add_argument("--sparse-model", required=True)
    prefilter_local_all.add_argument("--deterministic-ontology", required=True)
    prefilter_local_all.add_argument("--max-rows", type=int, default=None)
    prefilter_local_all.add_argument("--min-free-bytes", type=int, default=1)
    prefilter_local_all.add_argument("--row-group-batch-size", type=int, default=8)
    prefilter_local_all.add_argument("--row-group-shard-count", type=int, default=1)
    prefilter_local_all.add_argument("--row-group-shard-index", type=int, default=0)
    prefilter_local_all.add_argument(
        "--row-group-sharding-policy",
        choices=sorted(ROW_GROUP_SHARDING_POLICIES),
        default=ROW_GROUP_SHARDING_POLICY_ORDINAL,
    )
    prefilter_local_all.add_argument("--segmentation-validation-mode", choices=sorted(SEGMENTATION_VALIDATION_MODES), default="full")
    prefilter_local_all.add_argument(
        "--segmentation-validation-sample-modulus",
        type=int,
        default=DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
    )
    prefilter_local_all.add_argument("--supervisor-manifest", default=None, help=argparse.SUPPRESS)
    prefilter_local_all.add_argument("--supervisor-manifest-hash", default=None, help=argparse.SUPPRESS)
    prefilter_local_all.add_argument("--supervisor-command-hash", default=None, help=argparse.SUPPRESS)
    prefilter_local_all.add_argument("--supervisor-run-dir-sha256", default=None, help=argparse.SUPPRESS)

    prefilter_local_plan = sub.add_parser("prefilter-local-parquet-plan")
    prefilter_local_plan.add_argument("--config", required=True)
    prefilter_local_plan.add_argument("--selected-policy-report", required=True)
    prefilter_local_plan.add_argument("--plan", required=True)
    prefilter_local_plan.add_argument("--run-dir", required=True)
    prefilter_local_plan.add_argument("--cache-dir", required=True)
    prefilter_local_plan.add_argument("--sparse-report", required=True)
    prefilter_local_plan.add_argument("--sparse-model", required=True)
    prefilter_local_plan.add_argument("--deterministic-ontology", required=True)
    prefilter_local_plan.add_argument("--max-rows", type=int, default=None)
    prefilter_local_plan.add_argument("--min-free-bytes", type=int, default=1)
    prefilter_local_plan.add_argument("--row-group-batch-size", type=int, default=8)
    prefilter_local_plan.add_argument("--row-group-shard-count", type=int, default=1)
    prefilter_local_plan.add_argument("--row-group-shard-index", type=int, default=0)
    prefilter_local_plan.add_argument(
        "--row-group-sharding-policy",
        choices=sorted(ROW_GROUP_SHARDING_POLICIES),
        default=ROW_GROUP_SHARDING_POLICY_ORDINAL,
    )
    prefilter_local_plan.add_argument("--segmentation-validation-mode", choices=sorted(SEGMENTATION_VALIDATION_MODES), default="full")
    prefilter_local_plan.add_argument(
        "--segmentation-validation-sample-modulus",
        type=int,
        default=DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
    )
    prefilter_local_plan.add_argument("--supervisor-manifest", default=None, help=argparse.SUPPRESS)
    prefilter_local_plan.add_argument("--supervisor-manifest-hash", default=None, help=argparse.SUPPRESS)
    prefilter_local_plan.add_argument("--supervisor-command-hash", default=None, help=argparse.SUPPRESS)
    prefilter_local_plan.add_argument("--supervisor-run-dir-sha256", default=None, help=argparse.SUPPRESS)

    prefilter_supervisor_manifest = sub.add_parser("build-prefilter-plan-supervisor-manifest")
    prefilter_supervisor_manifest.add_argument("--config", required=True)
    prefilter_supervisor_manifest.add_argument("--selected-policy-report", required=True)
    prefilter_supervisor_manifest.add_argument("--plan", required=True)
    prefilter_supervisor_manifest.add_argument("--supervisor-run-dir", required=True)
    prefilter_supervisor_manifest.add_argument("--manifest-out", required=True)
    prefilter_supervisor_manifest.add_argument("--shard-count", type=int, required=True)
    prefilter_supervisor_manifest.add_argument("--worker-count", type=int, required=True)
    prefilter_supervisor_manifest.add_argument("--row-group-batch-size", type=int, default=8)
    prefilter_supervisor_manifest.add_argument(
        "--row-group-sharding-policy",
        choices=sorted(ROW_GROUP_SHARDING_POLICIES),
        default=ROW_GROUP_SHARDING_POLICY_ORDINAL,
    )
    prefilter_supervisor_manifest.add_argument(
        "--manifest-mode",
        choices=sorted(PREFILTER_SUPERVISOR_MODES),
        required=True,
    )
    prefilter_supervisor_manifest.add_argument("--max-rows", type=int, default=None)
    prefilter_supervisor_manifest.add_argument("--min-free-bytes", type=int, default=1)
    prefilter_supervisor_manifest.add_argument("--segmentation-validation-mode", choices=sorted(SEGMENTATION_VALIDATION_MODES), default="full")
    prefilter_supervisor_manifest.add_argument(
        "--segmentation-validation-sample-modulus",
        type=int,
        default=DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
    )

    run_prefilter_supervisor = sub.add_parser("run-prefilter-plan-supervisor")
    run_prefilter_supervisor.add_argument("--config", required=True)
    run_prefilter_supervisor.add_argument("--selected-policy-report", required=True)
    run_prefilter_supervisor.add_argument("--plan", required=True)
    run_prefilter_supervisor.add_argument("--supervisor-run-dir", required=True)
    run_prefilter_supervisor.add_argument("--cache-dir", required=True)
    run_prefilter_supervisor.add_argument("--sparse-report", required=True)
    run_prefilter_supervisor.add_argument("--sparse-model", required=True)
    run_prefilter_supervisor.add_argument("--deterministic-ontology", required=True)
    run_prefilter_supervisor.add_argument("--shard-count", type=int, required=True)
    run_prefilter_supervisor.add_argument("--worker-count", type=int, required=True)
    run_prefilter_supervisor.add_argument("--row-group-batch-size", type=int, default=8)
    run_prefilter_supervisor.add_argument(
        "--row-group-sharding-policy",
        choices=sorted(ROW_GROUP_SHARDING_POLICIES),
        default=ROW_GROUP_SHARDING_POLICY_ORDINAL,
    )
    run_prefilter_supervisor.add_argument(
        "--manifest-mode",
        choices=sorted(PREFILTER_SUPERVISOR_MODES),
        required=True,
    )
    run_prefilter_supervisor.add_argument("--max-rows", type=int, default=None)
    run_prefilter_supervisor.add_argument("--min-free-bytes", type=int, default=1)
    run_prefilter_supervisor.add_argument("--merge-run-dir", default=None)
    run_prefilter_supervisor.add_argument("--python-executable", default=None)
    run_prefilter_supervisor.add_argument("--poll-interval-seconds", type=float, default=1.0)
    run_prefilter_supervisor.add_argument("--segmentation-validation-mode", choices=sorted(SEGMENTATION_VALIDATION_MODES), default="full")
    run_prefilter_supervisor.add_argument(
        "--segmentation-validation-sample-modulus",
        type=int,
        default=DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
    )

    merge_prefilter = sub.add_parser("merge-prefilter-shards")
    merge_prefilter.add_argument("--config", required=True)
    merge_prefilter.add_argument("--selected-policy-report", required=True)
    merge_prefilter.add_argument("--local-schedule", required=True)
    merge_prefilter.add_argument("--source-metadata", required=True)
    merge_prefilter.add_argument("--run-dir", required=True)
    merge_prefilter.add_argument("--shard-run-dirs", nargs="+", required=True)
    merge_prefilter.add_argument("--min-free-bytes", type=int, default=1)

    merge_prefilter_plan = sub.add_parser("merge-prefilter-plan-shards")
    merge_prefilter_plan.add_argument("--config", required=True)
    merge_prefilter_plan.add_argument("--selected-policy-report", required=True)
    merge_prefilter_plan.add_argument("--plan", required=True)
    merge_prefilter_plan.add_argument("--run-dir", required=True)
    merge_prefilter_plan.add_argument("--shard-run-dirs", nargs="+", required=True)
    merge_prefilter_plan.add_argument("--min-free-bytes", type=int, default=1)
    merge_prefilter_plan.add_argument(
        "--row-group-sharding-policy",
        choices=sorted(ROW_GROUP_SHARDING_POLICIES),
        default=ROW_GROUP_SHARDING_POLICY_ORDINAL,
    )

    merge_prefilter_supervisor = sub.add_parser("merge-prefilter-plan-supervisor-manifest")
    merge_prefilter_supervisor.add_argument("--config", required=True)
    merge_prefilter_supervisor.add_argument("--selected-policy-report", required=True)
    merge_prefilter_supervisor.add_argument("--plan", required=True)
    merge_prefilter_supervisor.add_argument("--manifest", required=True)
    merge_prefilter_supervisor.add_argument("--supervisor-run-dir", required=True)
    merge_prefilter_supervisor.add_argument("--run-dir", required=True)
    merge_prefilter_supervisor.add_argument("--min-free-bytes", type=int, default=1)

    sharded_handoff = sub.add_parser("build-sharded-w1-handoff")
    sharded_handoff.add_argument("--config", required=True)
    sharded_handoff.add_argument("--selected-policy-report", required=True)
    sharded_handoff.add_argument("--plan", required=True)
    sharded_handoff.add_argument("--manifest", required=True)
    sharded_handoff.add_argument("--supervisor-run-dir", required=True)
    sharded_handoff.add_argument("--handoff-out", required=True)
    sharded_handoff.add_argument("--full-audit", action="store_true")

    partial_diagnostic_handoff = sub.add_parser("build-partial-sharded-w1-diagnostic-handoff")
    partial_diagnostic_handoff.add_argument("--config", required=True)
    partial_diagnostic_handoff.add_argument("--selected-policy-report", required=True)
    partial_diagnostic_handoff.add_argument("--plan", required=True)
    partial_diagnostic_handoff.add_argument("--manifest", required=True)
    partial_diagnostic_handoff.add_argument("--supervisor-run-dir", required=True)
    partial_diagnostic_handoff.add_argument("--handoff-out", required=True)
    partial_diagnostic_handoff.add_argument("--shard-index", type=int, action="append", required=True)
    partial_diagnostic_handoff.add_argument("--full-audit", action="store_true")

    compact_start = sub.add_parser("build-w1-compact-start-state")
    compact_start.add_argument("--config", required=True)
    compact_start.add_argument("--selected-policy-report", required=True)
    compact_start.add_argument("--source-run-dir", required=True)
    compact_start.add_argument("--dest-run-dir", required=True)
    compact_start.add_argument("--destination-supervisor-manifest")
    compact_start.add_argument(
        "--expected-prefilter-status",
        choices=("prefilter_ready_for_w1", "prefilter_shard_ready_for_merge"),
        default="prefilter_ready_for_w1",
    )
    compact_start.add_argument("--out", required=True)

    validate_compact_start = sub.add_parser("validate-w1-compact-start-state")
    validate_compact_start.add_argument("--config", required=True)
    validate_compact_start.add_argument("--selected-policy-report", required=True)
    validate_compact_start.add_argument("--run-dir", required=True)
    validate_compact_start.add_argument("--out", required=True)

    archive_w1_score_state = sub.add_parser("archive-w1-private-score-state")
    archive_w1_score_state.add_argument("--config", required=True)
    archive_w1_score_state.add_argument("--selected-policy-report", required=True)
    archive_w1_score_state.add_argument("--run-dir", required=True)
    archive_w1_score_state.add_argument("--archive-dir", required=True)
    archive_w1_score_state.add_argument("--extra-state-file", action="append", default=[])
    archive_w1_score_state.add_argument("--out", required=True)

    w1_score_sharded = sub.add_parser("w1-score-sharded")
    w1_score_sharded.add_argument("--config", required=True)
    w1_score_sharded.add_argument("--selected-policy-report", required=True)
    w1_score_sharded.add_argument("--handoff", required=True)
    w1_score_sharded.add_argument("--supervisor-run-dir", required=True)
    w1_score_sharded.add_argument("--window-mil-report", required=True)
    w1_score_sharded.add_argument("--window-mil-model", required=True)
    w1_score_sharded.add_argument("--summary-out", required=True)
    w1_score_sharded.add_argument("--device", default="auto")
    w1_score_sharded.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    w1_score_sharded.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    w1_score_sharded.add_argument("--batch-size", type=int, default=64)
    w1_score_sharded.add_argument("--max-windows-per-device-batch", type=int, default=None)
    w1_score_sharded.add_argument("--synchronize-cuda-telemetry", action="store_true")
    w1_score_sharded.add_argument(
        "--w1-attn-implementation",
        choices=sorted(SUPPORTED_W1_ATTN_IMPLEMENTATIONS),
        default=W1_ATTN_IMPLEMENTATION_SDPA,
    )
    w1_score_sharded.add_argument("--w1-prepare-pipeline", choices=("serial", "bounded-tokenize-overlap"), default="serial")
    w1_score_sharded.add_argument("--w1-max-estimated-windows-per-prepare-batch", type=int, default=None)
    w1_score_sharded.add_argument("--w1-prepare-queue-depth", type=int, default=1)
    w1_score_sharded.add_argument("--w1-pin-memory", choices=("auto", "on", "off"), default="auto")
    w1_score_sharded.add_argument(
        "--w1-tokenizers-parallelism",
        choices=("inherit", "false", "true"),
        default="inherit",
    )
    w1_score_sharded.add_argument("--w1-persist-commit-row-threshold", type=int, default=None)
    w1_score_sharded.add_argument("--enforce-accepted-token-target", action="store_true")
    w1_score_sharded.add_argument("--accepted-token-ledger", default=None)
    w1_score_sharded.add_argument("--shard-index-modulus", type=int, default=1)
    w1_score_sharded.add_argument("--shard-index-remainder", type=int, default=0)
    w1_score_sharded.add_argument("--assigned-shard-index", type=int, action="append", default=None)
    w1_score_sharded.add_argument("--rank-assignment-hash")

    w1_score_partial_diagnostic = sub.add_parser("w1-score-partial-sharded-diagnostic")
    w1_score_partial_diagnostic.add_argument("--config", required=True)
    w1_score_partial_diagnostic.add_argument("--selected-policy-report", required=True)
    w1_score_partial_diagnostic.add_argument("--handoff", required=True)
    w1_score_partial_diagnostic.add_argument("--supervisor-run-dir", required=True)
    w1_score_partial_diagnostic.add_argument("--window-mil-report", required=True)
    w1_score_partial_diagnostic.add_argument("--window-mil-model", required=True)
    w1_score_partial_diagnostic.add_argument("--summary-out", required=True)
    w1_score_partial_diagnostic.add_argument("--device", default="auto")
    w1_score_partial_diagnostic.add_argument("--require-cuda", dest="require_cuda", action="store_true", default=True)
    w1_score_partial_diagnostic.add_argument("--allow-cpu", dest="require_cuda", action="store_false")
    w1_score_partial_diagnostic.add_argument("--batch-size", type=int, default=64)
    w1_score_partial_diagnostic.add_argument("--max-windows-per-device-batch", type=int, default=None)
    w1_score_partial_diagnostic.add_argument("--synchronize-cuda-telemetry", action="store_true")
    w1_score_partial_diagnostic.add_argument(
        "--w1-attn-implementation",
        choices=sorted(SUPPORTED_W1_ATTN_IMPLEMENTATIONS),
        default=W1_ATTN_IMPLEMENTATION_SDPA,
    )
    w1_score_partial_diagnostic.add_argument("--w1-prepare-pipeline", choices=("serial", "bounded-tokenize-overlap"), default="serial")
    w1_score_partial_diagnostic.add_argument("--w1-max-estimated-windows-per-prepare-batch", type=int, default=None)
    w1_score_partial_diagnostic.add_argument("--w1-prepare-queue-depth", type=int, default=1)
    w1_score_partial_diagnostic.add_argument("--w1-pin-memory", choices=("auto", "on", "off"), default="auto")
    w1_score_partial_diagnostic.add_argument(
        "--w1-tokenizers-parallelism",
        choices=("inherit", "false", "true"),
        default="inherit",
    )
    w1_score_partial_diagnostic.add_argument("--w1-persist-commit-row-threshold", type=int, default=None)

    compare_w1_runtime_cap = sub.add_parser("compare-w1-runtime-cap-runs")
    compare_w1_runtime_cap.add_argument("--config", required=True)
    compare_w1_runtime_cap.add_argument("--selected-policy-report", required=True)
    compare_w1_runtime_cap.add_argument("--baseline-run-dir", required=True)
    compare_w1_runtime_cap.add_argument("--comparison-run-dir", required=True)
    compare_w1_runtime_cap.add_argument("--report-out", required=True)
    compare_w1_runtime_cap.add_argument("--threshold-band-epsilon", type=float, default=1e-4)
    compare_w1_runtime_cap.add_argument("--max-allowed-score-delta", type=float, default=1e-6)
    compare_w1_runtime_cap.add_argument("--require-synchronized-cuda", dest="require_synchronized_cuda", action="store_true", default=True)
    compare_w1_runtime_cap.add_argument("--allow-unsynchronized-cuda", dest="require_synchronized_cuda", action="store_false")
    compare_w1_runtime_cap.add_argument("--require-cap-exercised", dest="require_cap_exercised", action="store_true", default=True)
    compare_w1_runtime_cap.add_argument("--allow-unexercised-cap", dest="require_cap_exercised", action="store_false")

    compare_w1_score_runs_parser = sub.add_parser("compare-w1-score-runs")
    compare_w1_score_runs_parser.add_argument("--config", required=True)
    compare_w1_score_runs_parser.add_argument("--selected-policy-report", required=True)
    compare_w1_score_runs_parser.add_argument("--baseline-run-dir", required=True)
    compare_w1_score_runs_parser.add_argument("--comparison-run-dir", required=True)
    compare_w1_score_runs_parser.add_argument("--report-out", required=True)
    compare_w1_score_runs_parser.add_argument("--threshold-band-epsilon", type=float, default=1e-4)
    compare_w1_score_runs_parser.add_argument("--max-allowed-score-delta", type=float, default=1e-6)
    compare_w1_score_runs_parser.add_argument(
        "--require-synchronized-cuda",
        dest="require_synchronized_cuda",
        action="store_true",
        default=True,
    )
    compare_w1_score_runs_parser.add_argument(
        "--allow-unsynchronized-cuda",
        dest="require_synchronized_cuda",
        action="store_false",
    )
    compare_w1_score_runs_parser.add_argument("--comparison-backend-runtime-compatibility-manifest")

    w1_fa2_package = sub.add_parser("build-w1-fa2-package-candidate-manifest")
    w1_fa2_package.add_argument("--code-root", required=True)
    w1_fa2_package.add_argument("--code-commit", required=True)
    w1_fa2_package.add_argument("--flash-attn-wheel", required=True)
    w1_fa2_package.add_argument("--flash-attn-version", required=True)
    w1_fa2_package.add_argument("--install-command-file", required=True)
    w1_fa2_package.add_argument("--install-log-summary", required=True)
    w1_fa2_package.add_argument("--python-version")
    w1_fa2_package.add_argument("--platform-tag")
    w1_fa2_package.add_argument("--torch-version-expected", default="2.6.0+cu124")
    w1_fa2_package.add_argument("--torch-cuda-expected", default="12.4")
    w1_fa2_package.add_argument("--torch-cxx11-abi-expected")
    w1_fa2_package.add_argument("--out", required=True)

    w1_fa2_runtime = sub.add_parser("build-w1-fa2-runtime-compatibility-manifest")
    w1_fa2_runtime.add_argument("--config", required=True)
    w1_fa2_runtime.add_argument("--selected-policy-report", required=True)
    w1_fa2_runtime.add_argument("--package-candidate-manifest", required=True)
    w1_fa2_runtime.add_argument("--window-mil-report", required=True)
    w1_fa2_runtime.add_argument("--window-mil-model", required=True)
    w1_fa2_runtime.add_argument("--device", default="cuda:0")
    w1_fa2_runtime.add_argument("--code-commit", required=True)
    w1_fa2_runtime.add_argument("--code-root", required=True)
    w1_fa2_runtime.add_argument("--flash-attn-wheel", required=True)
    w1_fa2_runtime.add_argument("--out", required=True)

    w1_microbench = sub.add_parser("build-w1-scorer-local-microbench")
    w1_microbench.add_argument("--out", required=True)
    w1_microbench.add_argument("--document-count", type=int, default=64)
    w1_microbench.add_argument("--max-length", type=int, default=512)
    w1_microbench.add_argument("--stride", type=int, default=128)
    w1_microbench.add_argument("--max-windows-per-device-batch", type=int, default=512)
    w1_microbench.add_argument("--pooling-top-k", type=int, default=3)
    w1_microbench.add_argument("--device", default="cpu")
    w1_microbench.add_argument("--synchronize-cuda-telemetry", action="store_true")

    w1_length_distribution = sub.add_parser("build-w1-survivor-length-distribution")
    w1_length_distribution.add_argument("--config", required=True)
    w1_length_distribution.add_argument("--selected-policy-report", required=True)
    w1_length_distribution.add_argument("--run-dir", required=True)
    w1_length_distribution.add_argument("--out", required=True)

    w1_performance_ledger = sub.add_parser("build-w1-scorer-performance-ledger")
    w1_performance_ledger.add_argument("--config", required=True)
    w1_performance_ledger.add_argument("--selected-policy-report", required=True)
    w1_performance_ledger.add_argument("--w1-report", action="append", default=[])
    w1_performance_ledger.add_argument("--w1-run-dir", action="append", default=[])
    w1_performance_ledger.add_argument("--microbench-report", action="append", default=[])
    w1_performance_ledger.add_argument("--length-distribution-report", default=None)
    w1_performance_ledger.add_argument("--target-docs-per-second", type=float, default=1000.0)
    w1_performance_ledger.add_argument("--require-synchronized-cuda", dest="require_synchronized_cuda", action="store_true", default=True)
    w1_performance_ledger.add_argument("--allow-unsynchronized-cuda", dest="require_synchronized_cuda", action="store_false")
    w1_performance_ledger.add_argument("--out", required=True)

    assemble_sharded = sub.add_parser("assemble-sharded-report")
    assemble_sharded.add_argument("--config", required=True)
    assemble_sharded.add_argument("--selected-policy-report", required=True)
    assemble_sharded.add_argument("--handoff", required=True)
    assemble_sharded.add_argument("--supervisor-run-dir", required=True)
    assemble_sharded.add_argument("--report-out", required=True)

    w1_score = sub.add_parser("w1-score")
    w1_score.add_argument("--config", required=True)
    w1_score.add_argument("--selected-policy-report", required=True)
    w1_score.add_argument("--run-dir", required=True)
    w1_score.add_argument("--window-mil-report", required=True)
    w1_score.add_argument("--window-mil-model", required=True)
    w1_score.add_argument("--device", default="auto")
    w1_score.add_argument("--require-cuda", action="store_true")
    w1_score.add_argument("--batch-size", type=int, default=64)
    w1_score.add_argument("--max-windows-per-device-batch", type=int, default=None)
    w1_score.add_argument("--synchronize-cuda-telemetry", action="store_true")
    w1_score.add_argument(
        "--w1-attn-implementation",
        choices=sorted(SUPPORTED_W1_ATTN_IMPLEMENTATIONS),
        default=W1_ATTN_IMPLEMENTATION_SDPA,
    )
    w1_score.add_argument("--w1-prepare-pipeline", choices=("serial", "bounded-tokenize-overlap"), default="serial")
    w1_score.add_argument("--w1-max-estimated-windows-per-prepare-batch", type=int, default=None)
    w1_score.add_argument("--w1-prepare-queue-depth", type=int, default=1)
    w1_score.add_argument("--w1-pin-memory", choices=("auto", "on", "off"), default="auto")
    w1_score.add_argument(
        "--w1-tokenizers-parallelism",
        choices=("inherit", "false", "true"),
        default="inherit",
    )
    w1_score.add_argument("--w1-persist-commit-row-threshold", type=int, default=None)
    w1_score.add_argument("--enforce-accepted-token-target", action="store_true")

    assemble = sub.add_parser("assemble-report")
    assemble.add_argument("--config", required=True)
    assemble.add_argument("--selected-policy-report", required=True)
    assemble.add_argument("--run-dir", required=True)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "validate-config":
        payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(payload)
        selected_report = _load_and_validate_selected_policy_report(
            args.selected_policy_report,
            payload,
            required=False,
        )
        print(json.dumps({
            "valid": True,
            "config_hash": payload["config_hash"],
            "selected_policy_hash": payload["selected_policy_hash"],
            "selected_policy_report_validated": selected_report is not None,
        }, sort_keys=True))
    elif args.command == "build-canary-schedule":
        schedule, report = build_chunk_filter_canary_schedule_from_rows(
            source_schedule=_load_json(args.source_schedule),
            row_metadata=_load_json(args.row_metadata),
            high_token_rows=args.high_token_rows,
            random_rows=args.random_rows,
            seed=args.seed,
        )
        _write_json_atomic(args.out, schedule)
        _write_json_atomic(args.report_out, report)
        print(json.dumps({"schedule_hash": schedule["schedule_hash"], "report_hash": report["report_hash"]}, sort_keys=True))
    elif args.command == "mirror-source-files":
        validate_private_source_cache_dir(args.mirror_dir)
        validate_private_source_cache_dir(Path(args.local_schedule_out).parent)
        validate_private_source_cache_dir(Path(args.report_out).parent)
        local_schedule, report = mirror_private_schedule_sources(
            _load_json(args.schedule),
            args.mirror_dir,
            allow_local_sources=bool(args.allow_local_sources_for_tests),
            stop_file=args.stop_file,
        )
        if local_schedule is not None:
            _write_json_atomic(args.local_schedule_out, local_schedule)
        else:
            local_schedule_path = Path(args.local_schedule_out)
            if local_schedule_path.exists():
                local_schedule_path.unlink()
        _write_json_atomic(args.report_out, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report_hash": report["report_hash"],
                    "local_schedule_hash": report.get("local_schedule_hash"),
                },
                sort_keys=True,
            )
        )
    elif args.command == "build-local-parquet-token-tranche-plan":
        validate_private_source_cache_dir(Path(args.out).parent)
        validate_private_source_cache_dir(Path(args.report_out).parent)
        plan = build_local_parquet_token_target_tranche_plan(
            local_schedule=_load_json(args.local_schedule),
            source_metadata_manifest=_load_json(args.source_metadata),
            estimated_source_token_target=args.estimated_source_token_target,
            seed=args.seed,
        )
        report = _local_parquet_token_tranche_plan_report(plan)
        _write_json_atomic(args.out, plan)
        _write_json_atomic(args.report_out, report)
        print(json.dumps({"plan_hash": plan["schedule_hash"], "report_hash": report["report_hash"]}, sort_keys=True))
    elif args.command == "run-schedule":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        deterministic_ontology = None
        deterministic_policy_payload = None
        if args.deterministic_ontology:
            deterministic_ontology = load_ontology(args.deterministic_ontology)
            deterministic_policy_payload = build_deterministic_filter_policy_payload(deterministic_ontology)
            if (
                config_payload.get("deterministic_filter_policy_hash")
                != deterministic_policy_payload["deterministic_filter_policy_hash"]
            ):
                raise ValueError("run-schedule deterministic ontology hash does not match config")
        elif config_payload.get("deterministic_filter_policy_hash") is not None:
            raise ValueError("run-schedule config binds deterministic filter; pass --deterministic-ontology")
        validate_private_source_cache_dir(args.cache_dir)
        schedule = _load_json(args.schedule)
        if schedule.get("artifact_kind") == CHUNK_FILTER_SCHEDULE_KIND:
            if args.source_schedule is None:
                raise ValueError("run-schedule requires --source-schedule when --schedule is a public canary schedule")
            schedule = hydrate_chunk_filter_execution_schedule(schedule, _load_json(args.source_schedule))
        scorers, thresholds = _load_real_scorers(args, config_payload)
        provider = _build_schedule_row_provider(
            row_provider=args.row_provider,
            cache_dir=args.cache_dir,
            cache_delta_cap_bytes=args.cache_delta_cap_bytes,
            source_concurrency=args.source_concurrency,
            local_row_group_batch_size=args.local_row_group_batch_size,
        )
        report = run_chunk_filtering_schedule(
            schedule=schedule,
            config_payload=config_payload,
            row_provider=provider,
            scorers=scorers,
            thresholds=thresholds,
            run_dir=args.run_dir,
            private_clean_runs_jsonl=args.private_clean_runs_jsonl,
            public_report_path=args.public_report,
            max_rows=args.max_rows,
            deterministic_policy_payload=deterministic_policy_payload,
            deterministic_ontology=deterministic_ontology,
        )
        print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}, sort_keys=True))
    elif args.command == "prefilter-cpu":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        deterministic_ontology = load_ontology(args.deterministic_ontology)
        deterministic_policy_payload = build_deterministic_filter_policy_payload(deterministic_ontology)
        if config_payload.get("deterministic_filter_policy_hash") != deterministic_policy_payload["deterministic_filter_policy_hash"]:
            raise ValueError("prefilter-cpu deterministic ontology hash does not match config")
        schedule = _load_json(args.schedule)
        if schedule.get("artifact_kind") == CHUNK_FILTER_SCHEDULE_KIND:
            if args.source_schedule is None:
                raise ValueError("prefilter-cpu requires --source-schedule when --schedule is a public canary schedule")
            schedule = hydrate_chunk_filter_execution_schedule(schedule, _load_json(args.source_schedule))
        validate_private_source_cache_dir(args.cache_dir)
        provider = _build_schedule_row_provider(
            row_provider=args.row_provider,
            cache_dir=args.cache_dir,
            cache_delta_cap_bytes=args.cache_delta_cap_bytes,
            source_concurrency=args.source_concurrency,
            local_row_group_batch_size=args.local_row_group_batch_size,
        )
        report = run_prefilter_cpu_stage_for_schedule(
            schedule=schedule,
            config_payload=config_payload,
            row_provider=provider,
            sparse_scorer=_load_sparse_scorer(args, config_payload),
            run_dir=args.run_dir,
            deterministic_policy_payload=deterministic_policy_payload,
            deterministic_ontology=deterministic_ontology,
            max_rows=args.max_rows,
            cache_dir=args.cache_dir,
            min_free_bytes=args.min_free_bytes,
        )
        print(json.dumps({"status": report["status"], "stage_receipt_hash": report["stage_receipt_hash"]}, sort_keys=True))
    elif args.command == "prefilter-local-parquet-all-rows":
        context = _load_prefilter_local_parquet_cli_context(
            args,
            command_name="prefilter-local-parquet-all-rows",
        )
        report = run_prefilter_cpu_stage_for_local_parquet_all_rows(
            local_schedule=_load_json(args.local_schedule),
            source_metadata_manifest=_load_json(args.source_metadata),
            config_payload=context.config_payload,
            row_provider=LocalParquetRowProvider(
                columns=LOCAL_PARQUET_PREFILTER_COLUMNS,
                row_group_batch_size=args.row_group_batch_size,
            ),
            sparse_scorer=context.sparse_scorer,
            run_dir=args.run_dir,
            deterministic_policy_payload=context.deterministic_policy_payload,
            deterministic_ontology=context.deterministic_ontology,
            max_rows=args.max_rows,
            cache_dir=args.cache_dir,
            min_free_bytes=args.min_free_bytes,
            row_group_shard_count=args.row_group_shard_count,
            row_group_shard_index=args.row_group_shard_index,
            row_group_sharding_policy=args.row_group_sharding_policy,
            supervisor_manifest_hash=args.supervisor_manifest_hash,
            supervisor_command_hash=args.supervisor_command_hash,
            supervisor_run_dir_sha256=args.supervisor_run_dir_sha256,
            supervisor_manifest_path=args.supervisor_manifest,
            segmentation_validation_mode=args.segmentation_validation_mode,
            segmentation_validation_sample_modulus=args.segmentation_validation_sample_modulus,
        )
        print(json.dumps({"status": report["status"], "stage_receipt_hash": report["stage_receipt_hash"]}, sort_keys=True))
    elif args.command == "prefilter-local-parquet-plan":
        context = _load_prefilter_local_parquet_cli_context(
            args,
            command_name="prefilter-local-parquet-plan",
        )
        report = run_prefilter_cpu_stage_for_local_parquet_plan(
            prefilter_plan=_load_json(args.plan),
            config_payload=context.config_payload,
            row_provider=LocalParquetRowProvider(
                columns=LOCAL_PARQUET_PREFILTER_COLUMNS,
                row_group_batch_size=args.row_group_batch_size,
            ),
            sparse_scorer=context.sparse_scorer,
            run_dir=args.run_dir,
            deterministic_policy_payload=context.deterministic_policy_payload,
            deterministic_ontology=context.deterministic_ontology,
            max_rows=args.max_rows,
            cache_dir=args.cache_dir,
            min_free_bytes=args.min_free_bytes,
            row_group_shard_count=args.row_group_shard_count,
            row_group_shard_index=args.row_group_shard_index,
            row_group_sharding_policy=args.row_group_sharding_policy,
            supervisor_manifest_hash=args.supervisor_manifest_hash,
            supervisor_command_hash=args.supervisor_command_hash,
            supervisor_run_dir_sha256=args.supervisor_run_dir_sha256,
            supervisor_manifest_path=args.supervisor_manifest,
            segmentation_validation_mode=args.segmentation_validation_mode,
            segmentation_validation_sample_modulus=args.segmentation_validation_sample_modulus,
        )
        print(json.dumps({"status": report["status"], "stage_receipt_hash": report["stage_receipt_hash"]}, sort_keys=True))
    elif args.command == "build-prefilter-plan-supervisor-manifest":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        validate_private_filtering_run_dir(args.supervisor_run_dir)
        validate_private_filtering_run_dir(Path(args.manifest_out).parent)
        manifest = build_prefilter_local_parquet_plan_supervisor_manifest(
            prefilter_plan=_load_json(args.plan),
            config_payload=config_payload,
            supervisor_run_dir=args.supervisor_run_dir,
            shard_count=args.shard_count,
            worker_count=args.worker_count,
            row_group_batch_size=args.row_group_batch_size,
            manifest_mode=args.manifest_mode,
            max_rows=args.max_rows,
            min_free_bytes=args.min_free_bytes,
            row_group_sharding_policy=args.row_group_sharding_policy,
            segmentation_validation_mode=args.segmentation_validation_mode,
            segmentation_validation_sample_modulus=args.segmentation_validation_sample_modulus,
        )
        _write_json_atomic(args.manifest_out, manifest)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "manifest_hash": manifest["manifest_hash"],
                    "merge_allowed": manifest["merge_allowed"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "run-prefilter-plan-supervisor":
        summary = run_prefilter_local_parquet_plan_supervisor(
            config_path=args.config,
            selected_policy_report_path=args.selected_policy_report,
            plan_path=args.plan,
            supervisor_run_dir=args.supervisor_run_dir,
            cache_dir=args.cache_dir,
            sparse_report_path=args.sparse_report,
            sparse_model_path=args.sparse_model,
            deterministic_ontology_path=args.deterministic_ontology,
            shard_count=args.shard_count,
            worker_count=args.worker_count,
            row_group_batch_size=args.row_group_batch_size,
            manifest_mode=args.manifest_mode,
            max_rows=args.max_rows,
            min_free_bytes=args.min_free_bytes,
            row_group_sharding_policy=args.row_group_sharding_policy,
            merge_run_dir=args.merge_run_dir,
            python_executable=args.python_executable,
            poll_interval_seconds=args.poll_interval_seconds,
            segmentation_validation_mode=args.segmentation_validation_mode,
            segmentation_validation_sample_modulus=args.segmentation_validation_sample_modulus,
        )
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "summary_hash": summary["summary_hash"],
                    "manifest_hash": summary["manifest_hash"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "merge-prefilter-shards":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        report = merge_prefilter_local_parquet_all_row_shards(
            local_schedule=_load_json(args.local_schedule),
            source_metadata_manifest=_load_json(args.source_metadata),
            config_payload=config_payload,
            shard_run_dirs=args.shard_run_dirs,
            run_dir=args.run_dir,
            min_free_bytes=args.min_free_bytes,
        )
        print(json.dumps({"status": report["status"], "stage_receipt_hash": report["stage_receipt_hash"]}, sort_keys=True))
    elif args.command == "merge-prefilter-plan-shards":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        report = merge_prefilter_local_parquet_plan_shards(
            prefilter_plan=_load_json(args.plan),
            config_payload=config_payload,
            shard_run_dirs=args.shard_run_dirs,
            run_dir=args.run_dir,
            min_free_bytes=args.min_free_bytes,
            row_group_sharding_policy=args.row_group_sharding_policy,
        )
        print(json.dumps({"status": report["status"], "stage_receipt_hash": report["stage_receipt_hash"]}, sort_keys=True))
    elif args.command == "merge-prefilter-plan-supervisor-manifest":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        report = merge_prefilter_local_parquet_plan_supervisor_manifest(
            prefilter_plan=_load_json(args.plan),
            config_payload=config_payload,
            manifest=_load_json(args.manifest),
            supervisor_run_dir=args.supervisor_run_dir,
            run_dir=args.run_dir,
            min_free_bytes=args.min_free_bytes,
        )
        print(json.dumps({"status": report["status"], "stage_receipt_hash": report["stage_receipt_hash"]}, sort_keys=True))
    elif args.command == "build-sharded-w1-handoff":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        handoff = build_sharded_w1_handoff(
            prefilter_plan=_load_json(args.plan),
            config_payload=config_payload,
            manifest=_load_json(args.manifest),
            supervisor_run_dir=args.supervisor_run_dir,
            handoff_path=args.handoff_out,
            full_audit=bool(args.full_audit),
        )
        print(json.dumps({"status": handoff["status"], "handoff_hash": handoff["handoff_hash"]}, sort_keys=True))
    elif args.command == "build-partial-sharded-w1-diagnostic-handoff":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        handoff = build_partial_sharded_w1_diagnostic_handoff(
            prefilter_plan=_load_json(args.plan),
            config_payload=config_payload,
            manifest=_load_json(args.manifest),
            supervisor_run_dir=args.supervisor_run_dir,
            shard_indexes=args.shard_index,
            handoff_path=args.handoff_out,
            full_audit=bool(args.full_audit),
        )
        print(
            json.dumps(
                {"status": handoff["status"], "diagnostic_handoff_hash": handoff["diagnostic_handoff_hash"]},
                sort_keys=True,
            )
        )
    elif args.command == "build-w1-compact-start-state":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        payload = build_w1_compact_start_state(
            config_payload=config_payload,
            source_run_dir=args.source_run_dir,
            dest_run_dir=args.dest_run_dir,
            expected_prefilter_status=args.expected_prefilter_status,
            destination_supervisor_manifest=(
                _load_json(args.destination_supervisor_manifest)
                if args.destination_supervisor_manifest
                else None
            ),
        )
        _write_json_atomic(args.out, payload)
        print(
            json.dumps(
                {"status": payload["status"], "compact_start_state_hash": payload["compact_start_state_hash"]},
                sort_keys=True,
            )
        )
    elif args.command == "validate-w1-compact-start-state":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        payload = validate_w1_compact_start_state(
            config_payload=config_payload,
            run_dir=args.run_dir,
        )
        _write_json_atomic(args.out, payload)
        print(
            json.dumps(
                {"status": payload["status"], "compact_start_state_hash": payload["compact_start_state_hash"]},
                sort_keys=True,
            )
        )
    elif args.command == "archive-w1-private-score-state":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        payload = archive_w1_private_score_state(
            config_payload=config_payload,
            run_dir=args.run_dir,
            archive_dir=args.archive_dir,
            extra_state_files=args.extra_state_file,
        )
        _write_json_atomic(args.out, payload)
        print(
            json.dumps(
                {"status": payload["status"], "archive_manifest_hash": payload["archive_manifest_hash"]},
                sort_keys=True,
            )
        )
    elif args.command == "w1-score-sharded":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        summary = run_w1_score_sharded(
            config_payload=config_payload,
            handoff=_load_json(args.handoff),
            supervisor_run_dir=args.supervisor_run_dir,
            w1_scorer=_load_w1_scorer(args, config_payload),
            batch_size=args.batch_size,
            shard_index_modulus=args.shard_index_modulus,
            shard_index_remainder=args.shard_index_remainder,
            assigned_shard_indexes=args.assigned_shard_index,
            rank_assignment_hash=args.rank_assignment_hash,
            summary_path=args.summary_out,
            w1_prepare_pipeline=args.w1_prepare_pipeline,
            max_estimated_windows_per_prepare_batch=args.w1_max_estimated_windows_per_prepare_batch,
            w1_prepare_queue_depth=args.w1_prepare_queue_depth,
            w1_pin_memory=args.w1_pin_memory,
            w1_tokenizers_parallelism=args.w1_tokenizers_parallelism,
            w1_persist_commit_row_threshold=args.w1_persist_commit_row_threshold,
            accepted_token_ledger_path=args.accepted_token_ledger,
            enforce_accepted_token_target=bool(args.enforce_accepted_token_target),
        )
        print(json.dumps({"status": summary["status"], "summary_hash": summary["summary_hash"]}, sort_keys=True))
    elif args.command == "w1-score-partial-sharded-diagnostic":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        summary = run_w1_score_partial_sharded_diagnostic(
            config_payload=config_payload,
            handoff=_load_json(args.handoff),
            supervisor_run_dir=args.supervisor_run_dir,
            w1_scorer=_load_w1_scorer(args, config_payload),
            batch_size=args.batch_size,
            summary_path=args.summary_out,
            w1_prepare_pipeline=args.w1_prepare_pipeline,
            max_estimated_windows_per_prepare_batch=args.w1_max_estimated_windows_per_prepare_batch,
            w1_prepare_queue_depth=args.w1_prepare_queue_depth,
            w1_pin_memory=args.w1_pin_memory,
            w1_tokenizers_parallelism=args.w1_tokenizers_parallelism,
            w1_persist_commit_row_threshold=args.w1_persist_commit_row_threshold,
        )
        print(json.dumps({"status": summary["status"], "summary_hash": summary["summary_hash"]}, sort_keys=True))
    elif args.command == "compare-w1-runtime-cap-runs":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        report = compare_w1_runtime_cap_score_runs(
            config_payload=config_payload,
            baseline_run_dir=args.baseline_run_dir,
            comparison_run_dir=args.comparison_run_dir,
            report_path=args.report_out,
            threshold_band_epsilon=args.threshold_band_epsilon,
            max_allowed_score_delta=args.max_allowed_score_delta,
            require_synchronized_cuda=args.require_synchronized_cuda,
            require_cap_exercised=args.require_cap_exercised,
        )
        print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}, sort_keys=True))
    elif args.command == "compare-w1-score-runs":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        report = compare_w1_score_runs(
            config_payload=config_payload,
            baseline_run_dir=args.baseline_run_dir,
            comparison_run_dir=args.comparison_run_dir,
            report_path=args.report_out,
            threshold_band_epsilon=args.threshold_band_epsilon,
            max_allowed_score_delta=args.max_allowed_score_delta,
            require_synchronized_cuda=args.require_synchronized_cuda,
            comparison_backend_runtime_compatibility_manifest=(
                args.comparison_backend_runtime_compatibility_manifest
                if args.comparison_backend_runtime_compatibility_manifest
                else None
            ),
        )
        print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}, sort_keys=True))
    elif args.command == "build-w1-fa2-package-candidate-manifest":
        report = build_w1_fa2_package_candidate_manifest_from_files(
            code_root=args.code_root,
            code_commit=str(args.code_commit),
            flash_attn_wheel=args.flash_attn_wheel,
            flash_attn_version=str(args.flash_attn_version),
            install_command_file=args.install_command_file,
            install_log_summary=args.install_log_summary,
            python_version=args.python_version,
            platform_tag=args.platform_tag,
            torch_version_expected=str(args.torch_version_expected),
            torch_cuda_expected=str(args.torch_cuda_expected),
            torch_cxx11_abi_expected=args.torch_cxx11_abi_expected,
            report_path=args.out,
        )
        print(json.dumps({"status": report["status"], "manifest_hash": report["manifest_hash"]}, sort_keys=True))
    elif args.command == "build-w1-fa2-runtime-compatibility-manifest":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        report = build_w1_fa2_runtime_compatibility_manifest_from_scorer(
            config_payload=config_payload,
            package_candidate_manifest=_load_json(args.package_candidate_manifest),
            code_commit=str(args.code_commit),
            code_root=args.code_root,
            flash_attn_wheel=args.flash_attn_wheel,
            w1_scorer=_load_w1_scorer(
                SimpleNamespace(
                    window_mil_report=args.window_mil_report,
                    window_mil_model=args.window_mil_model,
                    device=args.device,
                    require_cuda=True,
                    max_windows_per_device_batch=None,
                    synchronize_cuda_telemetry=True,
                    w1_attn_implementation=W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2,
                ),
                config_payload,
            ),
            report_path=args.out,
        )
        print(json.dumps({"status": report["status"], "manifest_hash": report["manifest_hash"]}, sort_keys=True))
    elif args.command == "build-w1-scorer-local-microbench":
        from twc.classifiers.window_mil import build_window_mil_local_microbench_report

        report = build_window_mil_local_microbench_report(
            document_count=args.document_count,
            max_length=args.max_length,
            stride=args.stride,
            max_windows_per_device_batch=args.max_windows_per_device_batch,
            pooling_top_k=args.pooling_top_k,
            synchronize_cuda_timing=bool(args.synchronize_cuda_telemetry),
            device=args.device,
        )
        _write_json_atomic(args.out, report)
        print(json.dumps({"status": report["status"], "microbench_hash": report["microbench_hash"]}, sort_keys=True))
    elif args.command == "build-w1-survivor-length-distribution":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        report = build_w1_survivor_length_distribution_report(
            config_payload=config_payload,
            run_dir=args.run_dir,
            report_path=args.out,
        )
        print(
            json.dumps(
                {"status": report["status"], "length_distribution_hash": report["length_distribution_hash"]},
                sort_keys=True,
            )
        )
    elif args.command == "build-w1-scorer-performance-ledger":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        ledger = build_w1_scorer_performance_ledger(
            config_payload=config_payload,
            w1_reports=[_load_json(path) for path in args.w1_report],
            w1_run_dirs=args.w1_run_dir,
            microbench_reports=[_load_json(path) for path in args.microbench_report],
            length_distribution_report=(
                None if args.length_distribution_report is None else _load_json(args.length_distribution_report)
            ),
            target_docs_per_second=args.target_docs_per_second,
            require_synchronized_cuda=args.require_synchronized_cuda,
            report_path=args.out,
        )
        print(
            json.dumps(
                {"status": ledger["status"], "performance_ledger_hash": ledger["performance_ledger_hash"]},
                sort_keys=True,
            )
        )
    elif args.command == "assemble-sharded-report":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        report = assemble_sharded_filtering_stage_report(
            config_payload=config_payload,
            handoff=_load_json(args.handoff),
            supervisor_run_dir=args.supervisor_run_dir,
            report_path=args.report_out,
        )
        print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}, sort_keys=True))
    elif args.command == "w1-score":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        report = run_w1_score_stage(
            config_payload=config_payload,
            run_dir=args.run_dir,
            w1_scorer=_load_w1_scorer(args, config_payload),
            batch_size=args.batch_size,
            w1_prepare_pipeline=args.w1_prepare_pipeline,
            max_estimated_windows_per_prepare_batch=args.w1_max_estimated_windows_per_prepare_batch,
            w1_prepare_queue_depth=args.w1_prepare_queue_depth,
            w1_pin_memory=args.w1_pin_memory,
            w1_tokenizers_parallelism=args.w1_tokenizers_parallelism,
            w1_persist_commit_row_threshold=args.w1_persist_commit_row_threshold,
            enforce_accepted_token_target=bool(args.enforce_accepted_token_target),
        )
        print(json.dumps({"status": report["status"], "stage_receipt_hash": report["stage_receipt_hash"]}, sort_keys=True))
    elif args.command == "assemble-report":
        config_payload = build_chunk_filtering_config_payload(_load_yaml(args.config))
        validate_locked_chunk_filtering_config_payload(config_payload)
        _load_and_validate_selected_policy_report(args.selected_policy_report, config_payload, required=True)
        report = assemble_filtering_stage_report(config_payload=config_payload, run_dir=args.run_dir)
        print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
