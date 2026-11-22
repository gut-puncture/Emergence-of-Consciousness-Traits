"""Scheduled Phase 2A decision logging without public raw-text serialization."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit

from twc.data.decision_schema import (
    PHASE2A_DECISION_CAPABILITY,
    PHASE2A_ID,
    stable_json_hash,
)
from twc.data.document_filter import WINDOW_CONFIG, evaluate_deterministic_foundation
from twc.data.normalize import normalize_text
from twc.data.risk_ontology import RiskOntology
from twc.data.risk_ontology import load_ontology
from twc.data.source_schedule import (
    build_initial_resume_state,
    update_resume_after_entry,
    validate_resume_state,
    write_resume_state_atomic,
)


DECISION_LOG_SCHEMA_VERSION = 1
DECISION_LOG_REPORT_SCHEMA_VERSION = 1
DEFAULT_MAX_TEXT_BYTES = 2_000_000
DEFAULT_HF_ROW_READ_BLOCK_SIZE_BYTES = 16 * 1024 * 1024
DEFAULT_HF_ROW_READ_CACHE_DELTA_CAP_BYTES = 256 * 1024 * 1024
DEFAULT_HF_ROW_READ_MAX_CACHE_BLOCKS = 4
DEFAULT_HF_ROW_GROUP_BATCH_SIZE = 8
DEFAULT_HF_ROW_SOURCE_CONCURRENCY = 1
DEFAULT_LOCAL_PARQUET_FILE_CACHE_SIZE = 8
ALLOWED_MODES = {"diagnostic_audit", "corpus_freeze"}
HF_FSSPEC_DEFAULT_COLUMNS = (
    "text",
    "id",
    "dump",
    "url",
    "language",
    "language_score",
    "token_count",
    "score",
    "int_score",
    "title",
    "path",
    "domain",
)
RAW_TEXT_FORBIDDEN_KEYS = {
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
ALLOWED_PROVIDER_OBSERVATION_FIELDS = {
    "provider_class",
    "provider_version",
    "provider_config_hash",
    "row_read_transport",
    "row_read_block_size_bytes",
    "row_read_cache_type",
    "row_read_memory_cache_policy",
    "row_read_memory_cache_max_blocks",
    "row_read_memory_cache_cap_bytes",
    "row_read_hf_cache_env_managed",
    "row_read_disk_side_effect_dir_policy",
    "row_read_disk_side_effect_dir_sha256",
    "row_read_disk_side_effect_before_bytes",
    "row_read_disk_side_effect_after_bytes",
    "row_read_disk_side_effect_delta_bytes",
    "row_read_disk_side_effect_delta_cap_bytes",
    "row_read_opened_file_count",
    "row_read_row_group_count",
    "row_read_row_group_batch_size",
    "row_read_http_total_timeout_seconds",
    "row_read_columns",
}
PROVIDER_OBSERVATION_STRING_FIELDS = {
    "provider_class",
    "provider_config_hash",
    "row_read_transport",
    "row_read_cache_type",
    "row_read_memory_cache_policy",
    "row_read_disk_side_effect_dir_policy",
    "row_read_disk_side_effect_dir_sha256",
}
PROVIDER_OBSERVATION_INTEGER_FIELDS = {
    "provider_version",
    "row_read_block_size_bytes",
    "row_read_memory_cache_max_blocks",
    "row_read_memory_cache_cap_bytes",
    "row_read_disk_side_effect_before_bytes",
    "row_read_disk_side_effect_after_bytes",
    "row_read_disk_side_effect_delta_bytes",
    "row_read_disk_side_effect_delta_cap_bytes",
    "row_read_opened_file_count",
    "row_read_row_group_count",
    "row_read_row_group_batch_size",
    "row_read_http_total_timeout_seconds",
}
PROVIDER_OBSERVATION_BOOLEAN_FIELDS = {
    "row_read_hf_cache_env_managed",
}
_HEX_DIGITS = frozenset("0123456789abcdef")
SAFE_SOURCE_METADATA_FIELDS = (
    "domain",
    "dump",
    "language",
    "language_score",
    "token_count",
    "score",
    "int_score",
    "source_wrapper",
)
SOURCE_PRESENCE_FIELDS = (
    "id",
    "dump",
    "url",
    "title",
    "path",
    "domain",
    "language",
    "language_score",
    "token_count",
    "score",
    "int_score",
    "source_wrapper",
)


class ScheduledRowProvider(Protocol):
    """Offline-testable interface for resolving scheduled row-group offsets."""

    def fetch_rows(self, entry: dict[str, Any], offsets: list[int]) -> dict[int, Any]:
        """Return rows keyed by row-group-local offset."""


class ProviderFailClosedError(RuntimeError):
    """Provider guardrail violation that must abort the scheduled audit."""


class LocalParquetRowProvider:
    """Resolve scheduled rows from locally readable Parquet files."""

    provider_version = 2

    def __init__(
        self,
        *,
        columns: tuple[str, ...] | None = None,
        row_group_batch_size: int = DEFAULT_HF_ROW_GROUP_BATCH_SIZE,
        parquet_file_cache_size: int = DEFAULT_LOCAL_PARQUET_FILE_CACHE_SIZE,
    ) -> None:
        if row_group_batch_size < 1:
            raise ValueError("row_group_batch_size must be positive")
        if parquet_file_cache_size < 0:
            raise ValueError("parquet_file_cache_size must be nonnegative")
        self.columns = columns
        self.row_group_batch_size = int(row_group_batch_size)
        self.parquet_file_cache_size = int(parquet_file_cache_size)
        self._validated_file_sha256: dict[str, tuple[str, tuple[int, int, int, int]]] = {}
        self._parquet_file_cache: dict[str, tuple[tuple[int, int, int, int], Any]] = {}
        self._opened_file_count = 0
        self._row_group_read_count = 0

    def observations(self) -> dict[str, Any]:
        columns = list(self.columns) if self.columns else []
        return {
            "provider_class": type(self).__name__,
            "provider_version": self.provider_version,
            "provider_config_hash": _hash_payload(
                {
                    "columns": columns,
                    "row_group_batch_size": self.row_group_batch_size,
                    "parquet_file_cache_size": self.parquet_file_cache_size,
                }
            ),
            "row_read_transport": "local_parquet_file",
            "row_read_block_size_bytes": 0,
            "row_read_cache_type": "local_parquet_file_object",
            "row_read_memory_cache_policy": "signature_validated_per_provider_file_cache",
            "row_read_memory_cache_max_blocks": self.parquet_file_cache_size,
            "row_read_memory_cache_cap_bytes": 0,
            "row_read_hf_cache_env_managed": False,
            "row_read_disk_side_effect_dir_policy": "none",
            "row_read_disk_side_effect_dir_sha256": _sha256_text("none"),
            "row_read_disk_side_effect_before_bytes": 0,
            "row_read_disk_side_effect_after_bytes": 0,
            "row_read_disk_side_effect_delta_bytes": 0,
            "row_read_disk_side_effect_delta_cap_bytes": 0,
            "row_read_opened_file_count": self._opened_file_count,
            "row_read_row_group_count": self._row_group_read_count,
            "row_read_row_group_batch_size": self.row_group_batch_size,
            "row_read_http_total_timeout_seconds": 0,
            "row_read_columns": columns,
        }

    def fetch_rows(self, entry: dict[str, Any], offsets: list[int]) -> dict[int, Any]:
        key = _row_group_result_key(entry)
        return self.fetch_row_groups([(entry, offsets)])[key]

    def fetch_row_groups(
        self,
        entry_offsets: list[tuple[dict[str, Any], list[int]]],
    ) -> dict[tuple[str, int], dict[int, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install data extras first: uv sync --all-extras --group dev") from exc

        if not entry_offsets:
            return {}
        source_groups: dict[str, list[tuple[dict[str, Any], list[int]]]] = {}
        for entry, offsets in entry_offsets:
            source = entry.get("parquet_url_or_path")
            if not source:
                raise ValueError("schedule entry is missing parquet_url_or_path")
            source_groups.setdefault(str(source), []).append((entry, [int(offset) for offset in offsets]))

        resolved_by_group: dict[tuple[str, int], dict[int, Any]] = {}
        for source, grouped in source_groups.items():
            source_signature: tuple[int, int, int, int] | None = None
            for entry, _offsets in grouped:
                source_signature = self._validate_local_file_identity(source, entry)
            if source_signature is None:
                raise ValueError("local Parquet source has no grouped row requests")
            cached_file = self._parquet_file_cache.get(source)
            if cached_file is None or cached_file[0] != source_signature:
                if cached_file is not None:
                    self._close_parquet_file(cached_file[1])
                parquet_file = pq.ParquetFile(source)
                if self.parquet_file_cache_size > 0:
                    self._parquet_file_cache[source] = (source_signature, parquet_file)
                    self._evict_parquet_file_cache()
                self._opened_file_count += 1
            else:
                self._parquet_file_cache.pop(source)
                self._parquet_file_cache[source] = cached_file
                parquet_file = cached_file[1]
            available_columns = set(getattr(getattr(parquet_file, "schema", None), "names", []) or [])
            requested_columns = list(self.columns) if self.columns else []
            if requested_columns and available_columns:
                missing_columns = sorted(set(requested_columns) - available_columns)
                if missing_columns:
                    raise ValueError(f"local Parquet row group is missing required columns: {missing_columns}")
                selected_columns = requested_columns
            else:
                selected_columns = requested_columns or list(available_columns)
            row_group_requests: list[tuple[dict[str, Any], list[int], tuple[str, int], int, int]] = []
            for entry, requested_offsets in grouped:
                row_group_index = int(entry["row_group_index"])
                result_key = _row_group_result_key(entry)
                if result_key in resolved_by_group:
                    raise ValueError("duplicate row group requested from row provider")
                metadata_group = parquet_file.metadata.row_group(row_group_index)
                num_rows = int(metadata_group.num_rows)
                expected_num_rows = entry.get("row_group_num_rows")
                if expected_num_rows is not None and int(expected_num_rows) != num_rows:
                    raise ValueError("row_group_num_rows drift between schedule and Parquet metadata")
                row_group_requests.append((entry, requested_offsets, result_key, row_group_index, num_rows))
            for chunk_start in range(0, len(row_group_requests), self.row_group_batch_size):
                chunk = row_group_requests[chunk_start : chunk_start + self.row_group_batch_size]
                chunk_rows = self._read_row_group_chunk(
                    parquet_file=parquet_file,
                    selected_columns=selected_columns,
                    chunk=chunk,
                )
                self._row_group_read_count += len(chunk)
                for (_entry, requested_offsets, result_key, _row_group_index, num_rows), rows in zip(
                    chunk,
                    chunk_rows,
                    strict=True,
                ):
                    resolved: dict[int, Any] = {}
                    for offset in requested_offsets:
                        if 0 <= int(offset) < num_rows and int(offset) < len(rows):
                            resolved[int(offset)] = rows[int(offset)]
                    resolved_by_group[result_key] = resolved
        return resolved_by_group

    def _read_row_group_chunk(
        self,
        *,
        parquet_file: Any,
        selected_columns: list[str],
        chunk: list[tuple[dict[str, Any], list[int], tuple[str, int], int, int]],
    ) -> list[list[dict[str, Any]]]:
        if len(chunk) > 1 and hasattr(parquet_file, "read_row_groups"):
            row_group_indices = [row_group_index for _entry, _offsets, _key, row_group_index, _num_rows in chunk]
            table = _read_row_groups_no_threads(parquet_file, row_group_indices, selected_columns)
            all_rows = table.to_pylist()
            grouped_rows: list[list[dict[str, Any]]] = []
            cursor = 0
            for _entry, _offsets, _key, _row_group_index, num_rows in chunk:
                grouped_rows.append(all_rows[cursor : cursor + num_rows])
                cursor += num_rows
            return grouped_rows
        grouped_rows = []
        for _entry, _offsets, _key, row_group_index, _num_rows in chunk:
            table = _read_row_group_no_threads(parquet_file, row_group_index, selected_columns)
            grouped_rows.append(table.to_pylist())
        return grouped_rows

    def _validate_local_file_identity(self, source: str, entry: Mapping[str, Any]) -> tuple[int, int, int, int]:
        identity = entry.get("parquet_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("local Parquet mirror identity is missing local_mirror_file_sha256")
        expected_sha = identity.get("local_mirror_file_sha256")
        if expected_sha is None:
            raise ValueError("local Parquet mirror identity is missing local_mirror_file_sha256")
        if not _is_sha256(expected_sha):
            raise ValueError("local Parquet mirror identity has invalid local_mirror_file_sha256")
        signature = _local_file_signature(source)
        cached = self._validated_file_sha256.get(source)
        if cached is None or cached[1] != signature:
            cached_sha = _sha256_file(source)
            self._validated_file_sha256[source] = (cached_sha, signature)
        else:
            cached_sha = cached[0]
        if cached_sha != expected_sha:
            raise ValueError("local Parquet file SHA-256 drift")
        return signature

    def _evict_parquet_file_cache(self) -> None:
        while len(self._parquet_file_cache) > self.parquet_file_cache_size:
            first_source = next(iter(self._parquet_file_cache))
            _signature, parquet_file = self._parquet_file_cache.pop(first_source)
            self._close_parquet_file(parquet_file)

    @staticmethod
    def _close_parquet_file(parquet_file: Any) -> None:
        close = getattr(parquet_file, "close", None)
        if callable(close):
            close()


def _directory_size(path: Path) -> int:
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


def _row_group_result_key(entry: Mapping[str, Any]) -> tuple[str, int]:
    source_file_id = entry.get("source_file_id")
    if not isinstance(source_file_id, str) or not source_file_id:
        raise ValueError("schedule entry is missing source_file_id")
    return source_file_id, int(entry["row_group_index"])


def _merge_row_group_results(
    target: dict[tuple[str, int], dict[int, Any]],
    source: dict[tuple[str, int], dict[int, Any]],
) -> None:
    for key, value in source.items():
        if key in target:
            raise ValueError("duplicate row group requested from row provider")
        target[key] = value


def _read_row_groups_no_threads(parquet_file: Any, row_group_indices: list[int], columns: list[str]) -> Any:
    try:
        return parquet_file.read_row_groups(row_group_indices, columns=columns, use_threads=False)
    except TypeError:
        return parquet_file.read_row_groups(row_group_indices, columns=columns)


def _read_row_group_no_threads(parquet_file: Any, row_group_index: int, columns: list[str]) -> Any:
    try:
        return parquet_file.read_row_group(row_group_index, columns=columns, use_threads=False)
    except TypeError:
        return parquet_file.read_row_group(row_group_index, columns=columns)


@contextmanager
def _temporary_hf_cache_env(cache_path: Path):
    old_values = {
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
    }
    os.environ["HF_HOME"] = str(cache_path / "hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(cache_path / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class HFFsspecParquetRowProvider:
    """Resolve scheduled rows from Hugging Face HTTPS Parquet artifacts."""

    provider_version = 1

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        cache_delta_cap_bytes: int = DEFAULT_HF_ROW_READ_CACHE_DELTA_CAP_BYTES,
        block_size_bytes: int = DEFAULT_HF_ROW_READ_BLOCK_SIZE_BYTES,
        columns: tuple[str, ...] = HF_FSSPEC_DEFAULT_COLUMNS,
        cache_type: str = "blockcache",
        max_cache_blocks: int = DEFAULT_HF_ROW_READ_MAX_CACHE_BLOCKS,
        manage_hf_cache_env: bool = True,
        row_group_batch_size: int = DEFAULT_HF_ROW_GROUP_BATCH_SIZE,
        source_concurrency: int = DEFAULT_HF_ROW_SOURCE_CONCURRENCY,
        source_retry_count: int = 1,
        source_retry_backoff_seconds: float = 1.0,
        http_total_timeout_seconds: float = 120.0,
    ) -> None:
        if cache_delta_cap_bytes < 0:
            raise ValueError("cache_delta_cap_bytes must be non-negative")
        if block_size_bytes < 1:
            raise ValueError("block_size_bytes must be positive")
        if not columns:
            raise ValueError("columns must not be empty")
        if cache_type != "blockcache":
            raise ValueError("cache_type must be blockcache")
        if max_cache_blocks < 1:
            raise ValueError("max_cache_blocks must be positive")
        if row_group_batch_size < 1:
            raise ValueError("row_group_batch_size must be positive")
        if source_concurrency < 1:
            raise ValueError("source_concurrency must be positive")
        if manage_hf_cache_env and source_concurrency != 1:
            raise ValueError("source_concurrency > 1 requires manage_hf_cache_env=false")
        if source_retry_count < 1:
            raise ValueError("source_retry_count must be positive")
        if source_retry_backoff_seconds < 0:
            raise ValueError("source_retry_backoff_seconds must be non-negative")
        if http_total_timeout_seconds <= 0:
            raise ValueError("http_total_timeout_seconds must be positive")
        self.cache_dir = Path(cache_dir)
        self.cache_delta_cap_bytes = int(cache_delta_cap_bytes)
        self.block_size_bytes = int(block_size_bytes)
        self.columns = tuple(columns)
        self.cache_type = cache_type
        self.max_cache_blocks = int(max_cache_blocks)
        self.manage_hf_cache_env = bool(manage_hf_cache_env)
        self.row_group_batch_size = int(row_group_batch_size)
        self.source_concurrency = int(source_concurrency)
        self.source_retry_count = int(source_retry_count)
        self.source_retry_backoff_seconds = float(source_retry_backoff_seconds)
        self.http_total_timeout_seconds = float(http_total_timeout_seconds)
        self._cache_before = 0
        self._cache_after = 0
        self._opened_file_count = 0
        self._row_group_read_count = 0
        self._lock = threading.Lock()

    def _cache_delta(self) -> int:
        return self._cache_after - self._cache_before

    def _check_cache_delta(self) -> None:
        with self._lock:
            self._cache_after = _directory_size(self.cache_dir)
            if self._cache_delta() > self.cache_delta_cap_bytes:
                raise ProviderFailClosedError("row-read cache delta exceeded configured cap")

    def observations(self) -> dict[str, Any]:
        return {
            "provider_class": type(self).__name__,
            "provider_version": self.provider_version,
            "provider_config_hash": _hash_payload(
                {
                    "block_size_bytes": self.block_size_bytes,
                    "cache_type": self.cache_type,
                    "max_cache_blocks": self.max_cache_blocks,
                    "row_group_batch_size": self.row_group_batch_size,
                    "columns": list(self.columns),
                    "cache_delta_cap_bytes": self.cache_delta_cap_bytes,
                    "manage_hf_cache_env": self.manage_hf_cache_env,
                }
            ),
            "row_read_transport": f"fsspec_https_{self.cache_type}_file_like",
            "row_read_block_size_bytes": self.block_size_bytes,
            "row_read_cache_type": self.cache_type,
            "row_read_memory_cache_policy": "bounded_fsspec_cache",
            "row_read_memory_cache_max_blocks": self.max_cache_blocks,
            "row_read_memory_cache_cap_bytes": self.block_size_bytes * self.max_cache_blocks,
            "row_read_hf_cache_env_managed": self.manage_hf_cache_env,
            "row_read_disk_side_effect_dir_policy": "explicit_path_hash_only",
            "row_read_disk_side_effect_dir_sha256": _sha256_text(str(self.cache_dir)),
            "row_read_disk_side_effect_before_bytes": self._cache_before,
            "row_read_disk_side_effect_after_bytes": self._cache_after,
            "row_read_disk_side_effect_delta_bytes": self._cache_delta(),
            "row_read_disk_side_effect_delta_cap_bytes": self.cache_delta_cap_bytes,
            "row_read_opened_file_count": self._opened_file_count,
            "row_read_row_group_count": self._row_group_read_count,
            "row_read_row_group_batch_size": self.row_group_batch_size,
            "row_read_source_concurrency": self.source_concurrency,
            "row_read_source_retry_count": self.source_retry_count,
            "row_read_http_total_timeout_seconds": int(self.http_total_timeout_seconds),
            "row_read_columns": list(self.columns),
        }

    def fetch_rows(self, entry: dict[str, Any], offsets: list[int]) -> dict[int, Any]:
        key = _row_group_result_key(entry)
        return self.fetch_row_groups([(entry, offsets)])[key]

    def fetch_row_groups(
        self,
        entry_offsets: list[tuple[dict[str, Any], list[int]]],
    ) -> dict[tuple[str, int], dict[int, Any]]:
        """Return rows for multiple row groups, opening each source file once."""

        try:
            import fsspec
        except ImportError as exc:  # pragma: no cover - optional data dependency
            raise RuntimeError("fsspec is required for Hugging Face scheduled row reads") from exc
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - optional data dependency
            raise RuntimeError("aiohttp is required for bounded Hugging Face scheduled row reads") from exc
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional data dependency
            raise RuntimeError("pyarrow is required for Hugging Face scheduled row reads") from exc

        if not entry_offsets:
            return {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._opened_file_count == 0:
                self._cache_before = _directory_size(self.cache_dir)
                self._cache_after = self._cache_before

        source_groups: dict[str, list[tuple[dict[str, Any], list[int]]]] = {}
        for entry, offsets in entry_offsets:
            source = entry.get("parquet_url_or_path")
            if not isinstance(source, str) or not source:
                raise ValueError("schedule entry is missing parquet_url_or_path")
            if not source.startswith("https://"):
                raise ValueError("HFFsspecParquetRowProvider requires an HTTPS Parquet source")
            source_groups.setdefault(source, []).append((entry, [int(offset) for offset in offsets]))

        def fetch_source(source: str, grouped: list[tuple[dict[str, Any], list[int]]]) -> dict[tuple[str, int], dict[int, Any]]:
            last_error_type = "unknown"
            for attempt in range(1, self.source_retry_count + 1):
                source_results: dict[tuple[str, int], dict[int, Any]] = {}
                try:
                    cache_env = _temporary_hf_cache_env(self.cache_dir) if self.manage_hf_cache_env else nullcontext()
                    with cache_env:
                        self._fetch_source_row_groups(
                            fsspec=fsspec,
                            pq=pq,
                            aiohttp=aiohttp,
                            source=source,
                            grouped=grouped,
                            resolved_by_group=source_results,
                        )
                    self._check_cache_delta()
                    return source_results
                except (ProviderFailClosedError, ValueError):
                    raise
                except Exception as exc:
                    last_error_type = type(exc).__name__
                    if attempt >= self.source_retry_count:
                        break
                    if self.source_retry_backoff_seconds:
                        time.sleep(self.source_retry_backoff_seconds)
            raise ProviderFailClosedError(
                "scheduled source row-group fetch failed "
                f"after {self.source_retry_count} attempt(s): {last_error_type}"
            ) from None

        resolved_by_group: dict[tuple[str, int], dict[int, Any]] = {}
        source_items = list(source_groups.items())
        if self.source_concurrency == 1 or len(source_items) == 1:
            for source, grouped in source_items:
                _merge_row_group_results(resolved_by_group, fetch_source(source, grouped))
        else:
            worker_count = min(self.source_concurrency, len(source_items))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(fetch_source, source, grouped): source
                    for source, grouped in source_items
                }
                for future in as_completed(futures):
                    _merge_row_group_results(resolved_by_group, future.result())
        return resolved_by_group

    def _fetch_source_row_groups(
        self,
        *,
        fsspec: Any,
        pq: Any,
        aiohttp: Any,
        source: str,
        grouped: list[tuple[dict[str, Any], list[int]]],
        resolved_by_group: dict[tuple[str, int], dict[int, Any]],
    ) -> None:
        cache_options = {"maxblocks": self.max_cache_blocks}
        client_kwargs = {
            "timeout": aiohttp.ClientTimeout(total=self.http_total_timeout_seconds),
        }
        with fsspec.open(
            source,
            mode="rb",
            block_size=self.block_size_bytes,
            cache_type=self.cache_type,
            cache_options=cache_options,
            client_kwargs=client_kwargs,
        ) as file_like:
            with self._lock:
                self._opened_file_count += 1
            parquet_file = pq.ParquetFile(file_like)
            available_columns = set(getattr(getattr(parquet_file, "schema", None), "names", []) or [])
            if available_columns:
                selected_columns = [column for column in self.columns if column in available_columns]
                if "text" not in selected_columns:
                    raise ValueError("scheduled Parquet row group is missing required text column")
            else:
                selected_columns = list(self.columns)
            row_group_requests: list[tuple[dict[str, Any], list[int], tuple[str, int], int, int]] = []
            for entry, requested_offsets in grouped:
                row_group_index = int(entry["row_group_index"])
                result_key = _row_group_result_key(entry)
                if result_key in resolved_by_group:
                    raise ValueError("duplicate row group requested from row provider")
                metadata_group = parquet_file.metadata.row_group(row_group_index)
                num_rows = int(metadata_group.num_rows)
                expected_num_rows = entry.get("row_group_num_rows")
                if expected_num_rows is not None and int(expected_num_rows) != num_rows:
                    raise ValueError("row_group_num_rows drift between schedule and Parquet metadata")
                row_group_requests.append((entry, requested_offsets, result_key, row_group_index, num_rows))
            for chunk_start in range(0, len(row_group_requests), self.row_group_batch_size):
                chunk = row_group_requests[chunk_start : chunk_start + self.row_group_batch_size]
                chunk_rows = self._read_row_group_chunk(
                    parquet_file=parquet_file,
                    selected_columns=selected_columns,
                    chunk=chunk,
                )
                with self._lock:
                    self._row_group_read_count += len(chunk)
                for (_entry, requested_offsets, result_key, _row_group_index, num_rows), rows in zip(
                    chunk,
                    chunk_rows,
                    strict=True,
                ):
                    resolved: dict[int, Any] = {}
                    for offset in requested_offsets:
                        if offset < 0 or offset >= num_rows or offset >= len(rows):
                            resolved[offset] = RuntimeError(
                                f"scheduled row offset {offset} out of bounds for row group with {num_rows} rows"
                            )
                        else:
                            resolved[offset] = rows[offset]
                    resolved_by_group[result_key] = resolved

    def _read_row_group_chunk(
        self,
        *,
        parquet_file: Any,
        selected_columns: list[str],
        chunk: list[tuple[dict[str, Any], list[int], tuple[str, int], int, int]],
    ) -> list[list[dict[str, Any]]]:
        if len(chunk) > 1 and hasattr(parquet_file, "read_row_groups"):
            row_group_indices = [row_group_index for _entry, _offsets, _key, row_group_index, _num_rows in chunk]
            table = _read_row_groups_no_threads(parquet_file, row_group_indices, selected_columns)
            all_rows = table.to_pylist()
            grouped_rows: list[list[dict[str, Any]]] = []
            cursor = 0
            for _entry, _offsets, _key, _row_group_index, num_rows in chunk:
                grouped_rows.append(all_rows[cursor : cursor + num_rows])
                cursor += num_rows
            return grouped_rows
        grouped_rows = []
        for _entry, _offsets, _key, row_group_index, _num_rows in chunk:
            table = _read_row_group_no_threads(parquet_file, row_group_index, selected_columns)
            grouped_rows.append(table.to_pylist())
        return grouped_rows


@dataclass(frozen=True)
class DecisionLogConfig:
    run_id: str
    artifact_set_id: str
    mode: str = "diagnostic_audit"
    parent_run_id: str | None = None
    freeze_id: str | None = None
    max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES
    source_tree_hash: str | None = None
    config_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.artifact_set_id:
            raise ValueError("artifact_set_id is required")
        if self.mode not in ALLOWED_MODES:
            raise ValueError(f"unsupported decision log mode: {self.mode}")
        if self.max_text_bytes < 1:
            raise ValueError("max_text_bytes must be positive")
        if self.mode == "diagnostic_audit" and self.freeze_id is not None:
            raise ValueError("diagnostic Phase 2A decision logs must not set freeze_id")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_file_signature(path: str | Path) -> tuple[int, int, int, int]:
    stat = Path(path).stat()
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(getattr(stat, "st_dev", 0)),
        int(getattr(stat, "st_ino", 0)),
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX_DIGITS for char in value)


def _hash_payload(payload: Any) -> str:
    return stable_json_hash(payload)


def _json_dump_line(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _validate_no_public_raw_text(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in RAW_TEXT_FORBIDDEN_KEYS:
                raise ValueError(f"raw-text-like field is forbidden in decision log: {path}.{key}")
            _validate_no_public_raw_text(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_public_raw_text(child, path=f"{path}[{index}]")


def _looks_like_url_or_path(value: str) -> bool:
    lowered = value.casefold()
    if "://" in lowered or "token=" in lowered or "?" in lowered:
        return True
    return value.startswith(("/", "~", "./", "../")) or "\\" in value


def _sanitize_provider_observations(observations: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in observations.items():
        if key not in ALLOWED_PROVIDER_OBSERVATION_FIELDS:
            raise ValueError(f"unsupported provider observation field: {key}")
        if key in PROVIDER_OBSERVATION_INTEGER_FIELDS:
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(f"provider observation must be an integer or null: {key}")
            sanitized[key] = value
            continue
        if key in PROVIDER_OBSERVATION_BOOLEAN_FIELDS:
            if not isinstance(value, bool):
                raise ValueError(f"provider observation must be a boolean: {key}")
            sanitized[key] = value
            continue
        if key in PROVIDER_OBSERVATION_STRING_FIELDS:
            if not isinstance(value, str):
                raise ValueError(f"provider observation must be a string: {key}")
            if _looks_like_url_or_path(value):
                raise ValueError(f"provider observation string may not contain a URL or path: {key}")
            sanitized[key] = value
            continue
        if key == "row_read_columns":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError("row_read_columns must be a list of strings")
            if any(_looks_like_url_or_path(item) for item in value):
                raise ValueError("row_read_columns may not contain URLs or paths")
            sanitized[key] = list(value)
            continue
        raise ValueError(f"unsupported provider observation field: {key}")
    _validate_no_public_raw_text(sanitized)
    return sanitized


def _source_domain(url: Any) -> str | None:
    if not isinstance(url, str) or not url:
        return None
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    return host.rstrip(".").casefold() if host else None


def _source_path_hash(url: Any, path: Any = None) -> str | None:
    if isinstance(path, str) and path:
        return _sha256_text(path)
    if not isinstance(url, str) or not url:
        return None
    try:
        split = urlsplit(url)
    except ValueError:
        return None
    path = split.path or ""
    return _sha256_text(path) if path else None


def _url_hash(url: Any) -> str | None:
    return _sha256_text(url) if isinstance(url, str) and url else None


def _public_parquet_identity(identity: dict[str, Any]) -> dict[str, Any]:
    public = deepcopy(identity)
    raw_url = public.pop("url", None)
    raw_source_path = public.pop("source_path", None)
    if isinstance(raw_url, str) and raw_url:
        public["url_sha256"] = _sha256_text(raw_url)
    if isinstance(raw_source_path, str) and raw_source_path:
        public["source_path_sha256"] = _sha256_text(raw_source_path)
    return public


def _source_path(url: Any) -> str:
    if not isinstance(url, str) or not url:
        return ""
    try:
        return urlsplit(url).path or ""
    except ValueError:
        return ""


def _presence(row: dict[str, Any]) -> dict[str, bool]:
    return {
        field: row.get(field) is not None and row.get(field) != ""
        for field in SOURCE_PRESENCE_FIELDS
    }


def _public_source_metadata_presence(row: dict[str, Any]) -> dict[str, bool]:
    presence = _presence(row)
    filter_presence = _presence(_filter_metadata(row))
    for field in ("url", "title", "path", "domain"):
        presence[field] = filter_presence[field]
    return presence


def _safe_source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    domain = row.get("domain") or _source_domain(row.get("url"))
    metadata = {
        "domain": str(domain).casefold() if domain else None,
        "dump": row.get("dump"),
        "language": row.get("language"),
        "language_score": row.get("language_score"),
        "token_count": row.get("token_count"),
        "score": row.get("score"),
        "int_score": row.get("int_score"),
        "source_wrapper": row.get("source_wrapper"),
    }
    return {field: metadata.get(field) for field in SAFE_SOURCE_METADATA_FIELDS}


def _filter_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": row.get("url") or "",
        "title": row.get("title") or "",
        "path": row.get("path") or _source_path(row.get("url")),
        "domain": row.get("domain") or _source_domain(row.get("url")) or "",
    }


def _row_identity(
    *,
    config: DecisionLogConfig,
    entry: dict[str, Any],
    offset: int,
) -> str:
    return _hash_payload(
        [
            "twc_decision_log_row",
            config.run_id,
            entry["entry_hash"],
            entry["source_file_id"],
            int(entry["row_group_index"]),
            int(offset),
        ]
    )


def _window_count(text: str) -> int:
    words = text.split()
    if not words:
        return 1
    window_words = int(WINDOW_CONFIG["window_words"])
    overlap_words = int(WINDOW_CONFIG["window_overlap_words"])
    step = window_words - overlap_words
    if step <= 0:
        return 1
    count = 0
    start = 0
    while start < len(words):
        count += 1
        if start + window_words >= len(words):
            break
        start += step
    return count


def _sanitize_hits(hits: Any) -> list[dict[str, Any]]:
    sanitized = []
    for hit in hits:
        item = asdict(hit) if hasattr(hit, "__dataclass_fields__") else dict(hit)
        sanitized.append(
            {
                "category": item.get("category"),
                "rule_id": item.get("rule_id"),
                "severity": item.get("severity"),
                "reason": item.get("reason"),
                "source_field": item.get("source_field"),
                "match_view": item.get("match_view"),
                "span_normalized": item.get("span_normalized"),
                "raw_span": item.get("raw_span"),
                "span_projection_version": item.get("span_projection_version"),
                "pattern_type": item.get("pattern_type"),
                "hit_weight": item.get("hit_weight"),
                "scan_scope": item.get("scan_scope"),
                "window_index": item.get("window_index"),
                "window_start_normalized": item.get("window_start_normalized"),
                "window_end_normalized": item.get("window_end_normalized"),
                "window_word_start": item.get("window_word_start"),
                "window_word_end": item.get("window_word_end"),
            }
        )
    return sanitized


def _hit_counts(hits: list[dict[str, Any]], key: str) -> dict[str, int]:
    def safe_label(value: Any) -> str:
        label = str(value or "missing")
        if key == "source_field" and label == "text":
            return "document_body"
        return label

    return dict(sorted(Counter(safe_label(hit.get(key)) for hit in hits).items()))


def _rejection_reasons(hits: list[dict[str, Any]], status: str, quarantine_reason: str | None) -> list[str]:
    if status == "rejected":
        return sorted(
            {
                str(hit.get("reason") or hit.get("rule_id") or "deterministic_reject")
                for hit in hits
                if hit.get("severity") in {"hard_reject", "density_reject", "phrase_reject", "source_reject"}
            }
        )
    if status == "quarantined" and quarantine_reason:
        return [quarantine_reason]
    return []


def _risk_tags(hits: list[dict[str, Any]], status: str, quarantine_reason: str | None) -> list[str]:
    tags = {str(hit.get("category")) for hit in hits if hit.get("category")}
    tags.update(str(hit.get("rule_id")) for hit in hits if hit.get("rule_id"))
    if status == "quarantined" and quarantine_reason:
        tags.add(quarantine_reason)
    if status == "rejected":
        tags.add("deterministic_reject")
    return sorted(tags)


def _foundation_payload(
    *,
    status: str,
    raw_text_sha256: str | None,
    normalized_text_sha256: str | None,
    ontology: RiskOntology,
    hits: list[dict[str, Any]],
    source_metadata_presence: dict[str, bool],
    quarantine_reason_code: str | None = None,
    error_class: str | None = None,
    error_stage: str | None = None,
    error_message_hash: str | None = None,
    decision: Any | None = None,
) -> dict[str, Any]:
    return {
        "decision_status": status,
        "phase_id": PHASE2A_ID,
        "decision_capability": PHASE2A_DECISION_CAPABILITY,
        "all_required_gates_present": False,
        "executed_gates": list(getattr(decision, "executed_gates", ("source_channel_gate", "deterministic_regex_gate"))),
        "skipped_required_gates": list(
            getattr(
                decision,
                "skipped_required_gates",
                (
                    "density_gate",
                    "safe_topic_gate",
                    "local_classifier_ensemble",
                    "teacher_audit_failure_gate",
                    "post_tokenization_rescan",
                ),
            )
        ),
        "pending_gate_reasons": list(
            getattr(
                decision,
                "pending_gate_reasons",
                (
                    "safe_topic_gate_not_implemented",
                    "local_classifier_ensemble_not_implemented",
                    "teacher_audit_failure_gate_not_implemented",
                    "post_tokenization_rescan_not_implemented",
                ),
            )
        ),
        "raw_text_sha256": raw_text_sha256,
        "normalized_text_sha256": normalized_text_sha256,
        "ontology_semantic_hash": ontology.semantic_hash,
        "ontology_source_tree_hash": ontology.source_tree_hash,
        "config_hash": getattr(decision, "config_hash", None),
        "source_metadata_presence": source_metadata_presence,
        "hits": hits,
        "quarantine_reason_code": quarantine_reason_code,
        "error_class": error_class,
        "error_stage": error_stage,
        "error_message_hash": error_message_hash,
    }


def _preflight_decision(
    *,
    status: str,
    reason: str,
    row: dict[str, Any] | None,
    ontology: RiskOntology,
    raw_text_sha256: str | None = None,
    normalized_text_sha256: str | None = None,
) -> dict[str, Any]:
    return _foundation_payload(
        status=status,
        raw_text_sha256=raw_text_sha256,
        normalized_text_sha256=normalized_text_sha256,
        ontology=ontology,
        hits=[],
        source_metadata_presence=_presence(row or {}),
        quarantine_reason_code=reason if status == "quarantined" else None,
        error_class="RowResolutionError" if status == "error" else None,
        error_stage="row_resolution" if status == "error" else None,
        error_message_hash=_sha256_text(reason) if status == "error" else None,
    )


def _evaluate_row(
    row_record: Any,
    *,
    ontology: RiskOntology,
    max_text_bytes: int,
) -> tuple[dict[str, Any] | None, dict[str, Any], int, bool, bool]:
    if isinstance(row_record, BaseException):
        foundation = _preflight_decision(
            status="error",
            reason=f"{type(row_record).__name__}: {row_record}",
            row=None,
            ontology=ontology,
        )
        return None, foundation, 0, False, False
    if not isinstance(row_record, dict):
        foundation = _preflight_decision(
            status="quarantined",
            reason="non_dict_row",
            row={},
            ontology=ontology,
        )
        return {}, foundation, 0, False, False

    text = row_record.get("text")
    text_is_string = isinstance(text, str)
    text_is_empty = text == "" if text_is_string else False
    if not text_is_string:
        foundation = _preflight_decision(
            status="quarantined",
            reason="non_string_text",
            row=row_record,
            ontology=ontology,
        )
        return row_record, foundation, 0, False, False

    text_byte_length = len(text.encode("utf-8", errors="surrogatepass"))
    raw_hash = _sha256_text(text)
    if text_is_empty:
        normalized = normalize_text("")
        foundation = _preflight_decision(
            status="quarantined",
            reason="empty_text",
            row=row_record,
            ontology=ontology,
            raw_text_sha256=raw_hash,
            normalized_text_sha256=normalized.canonical_sha256,
        )
        return row_record, foundation, text_byte_length, True, True
    if text_byte_length > max_text_bytes:
        foundation = _preflight_decision(
            status="quarantined",
            reason="text_exceeds_max_bytes",
            row=row_record,
            ontology=ontology,
            raw_text_sha256=raw_hash,
        )
        return row_record, foundation, text_byte_length, True, False

    normalized = normalize_text(text)
    if not normalized.canonical_text:
        foundation = _preflight_decision(
            status="quarantined",
            reason="empty_after_normalization",
            row=row_record,
            ontology=ontology,
            raw_text_sha256=raw_hash,
            normalized_text_sha256=normalized.canonical_sha256,
        )
        return row_record, foundation, text_byte_length, True, False

    decision = evaluate_deterministic_foundation(
        text,
        metadata=_filter_metadata(row_record),
        ontology=ontology,
    )
    hits = _sanitize_hits(decision.hits)
    foundation = _foundation_payload(
        status=decision.decision_status,
        raw_text_sha256=decision.raw_text_sha256,
        normalized_text_sha256=decision.normalized_text_sha256,
        ontology=ontology,
        hits=hits,
        source_metadata_presence=decision.source_metadata_presence,
        quarantine_reason_code=decision.quarantine_reason_code,
        error_class=decision.error_class,
        error_stage=decision.error_stage,
        error_message_hash=decision.error_message_hash,
        decision=decision,
    )
    return row_record, foundation, text_byte_length, True, False


def build_public_decision_row(
    entry: dict[str, Any],
    row_group_local_row_index: int,
    row_record: Any,
    *,
    schedule_hash: str,
    source_metadata_hash: str,
    config: DecisionLogConfig,
    ontology: RiskOntology,
) -> dict[str, Any]:
    """Build one canonical public decision-log row for a scheduled document."""

    safe_row, foundation, text_byte_length, text_is_string, text_is_empty = _evaluate_row(
        row_record,
        ontology=ontology,
        max_text_bytes=config.max_text_bytes,
    )
    source_row = safe_row or {}
    hits = foundation["hits"]
    status = foundation["decision_status"]
    quarantine_reason = foundation["quarantine_reason_code"]
    failed_windows = {
        hit.get("window_index")
        for hit in hits
        if hit.get("scan_scope") == "window"
        and hit.get("severity") in {"hard_reject", "density_reject", "phrase_reject", "source_reject"}
        and hit.get("window_index") is not None
    }
    window_count = _window_count(source_row["text"]) if isinstance(source_row.get("text"), str) else 0
    safe_metadata = _safe_source_metadata(source_row)
    source_presence = _public_source_metadata_presence(source_row)
    row = {
        "decision_log_schema_version": DECISION_LOG_SCHEMA_VERSION,
        "decision_log_id": _row_identity(config=config, entry=entry, offset=row_group_local_row_index),
        "run_id": config.run_id,
        "parent_run_id": config.parent_run_id,
        "artifact_set_id": config.artifact_set_id,
        "freeze_id": config.freeze_id,
        "mode": config.mode,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "dataset_repo_id": entry["dataset_repo_id"],
        "config": entry["config"],
        "split": entry["split"],
        "resolved_repo_sha": entry["resolved_repo_sha"],
        "source_metadata_hash": source_metadata_hash,
        "source_schedule_hash": schedule_hash,
        "schedule_entry_hash": entry["entry_hash"],
        "source_file_id": entry["source_file_id"],
        "parquet_identity": _public_parquet_identity(entry["parquet_identity"]),
        "row_group_index": int(entry["row_group_index"]),
        "row_group_local_row_index": int(row_group_local_row_index),
        "global_schedule_index": int(entry["global_schedule_index"]),
        "schedule_pass_index": int(entry["schedule_pass_index"]),
        "source_doc_id": source_row.get("id"),
        "dump": source_row.get("dump"),
        "url_sha256": _url_hash(source_row.get("url")),
        "source_domain": safe_metadata["domain"],
        "source_path_sha256": _source_path_hash(source_row.get("url"), source_row.get("path")),
        "source_metadata_presence": source_presence,
        "source_metadata": safe_metadata,
        "raw_text_sha256": foundation["raw_text_sha256"],
        "normalized_text_sha256": foundation["normalized_text_sha256"],
        "text_byte_length": text_byte_length,
        "text_is_string": text_is_string,
        "text_is_empty": text_is_empty,
        "foundation_decision": foundation,
        "decision_status": status,
        "decision_stage": PHASE2A_ID,
        "decision_capability": PHASE2A_DECISION_CAPABILITY,
        "all_required_gates_present": False,
        "executed_gates": foundation["executed_gates"],
        "skipped_required_gates": foundation["skipped_required_gates"],
        "pending_gate_reasons": foundation["pending_gate_reasons"],
        "rejection_reasons": _rejection_reasons(hits, status, quarantine_reason),
        "risk_tags": _risk_tags(hits, status, quarantine_reason),
        "hit_count": len(hits),
        "hit_counts_by_category": _hit_counts(hits, "category"),
        "hit_counts_by_severity": _hit_counts(hits, "severity"),
        "hit_counts_by_source_field": _hit_counts(hits, "source_field"),
        "hit_counts_by_scan_scope": _hit_counts(hits, "scan_scope"),
        "hits": hits,
        "window_summary": {
            "window_count": window_count,
            "failed_window_count": len(failed_windows),
        },
        "quarantine_reason_code": quarantine_reason,
        "error_class": foundation["error_class"],
        "error_stage": foundation["error_stage"],
        "error_message_hash": foundation["error_message_hash"],
        "source_tree_hash": config.source_tree_hash,
        "decision_log_config_hash": config.config_hash,
        "ontology_semantic_hash": ontology.semantic_hash,
        "ontology_source_tree_hash": ontology.source_tree_hash,
    }
    if status == "accepted":
        raise ValueError("Phase 2A decision logs must not emit accepted")
    row["decision_row_hash"] = _hash_payload(
        {key: value for key, value in row.items() if key != "decision_row_hash"}
    )
    validate_decision_log_row(row)
    return row


def validate_decision_log_row(row: dict[str, Any]) -> None:
    if row.get("decision_log_schema_version") != DECISION_LOG_SCHEMA_VERSION:
        raise ValueError("unsupported decision log row schema")
    if row.get("decision_status") == "accepted":
        raise ValueError("Phase 2A decision logs must not emit accepted")
    if row.get("raw_text_serialized") is not False:
        raise ValueError("decision log row must set raw_text_serialized=false")
    if row.get("raw_url_serialized") is not False:
        raise ValueError("decision log row must set raw_url_serialized=false")
    if row.get("token_serialized") is not False:
        raise ValueError("decision log row must set token_serialized=false")
    if not row.get("decision_log_id"):
        raise ValueError("missing decision_log_id")
    if not row.get("decision_row_hash"):
        raise ValueError("missing decision_row_hash")
    expected = _hash_payload(
        {key: value for key, value in row.items() if key != "decision_row_hash"}
    )
    if row["decision_row_hash"] != expected:
        raise ValueError("decision row hash mismatch")
    _validate_no_public_raw_text(row)


def read_decision_log_rows(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows = []
    seen: dict[str, str] = {}
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corrupt jsonl at line {line_number}") from exc
        try:
            validate_decision_log_row(row)
        except ValueError as exc:
            raise ValueError(f"invalid decision log row at line {line_number}: {exc}") from exc
        row_id = row["decision_log_id"]
        row_hash = row["decision_row_hash"]
        if row_id in seen:
            if seen[row_id] != row_hash:
                raise ValueError(f"conflicting duplicate decision row at line {line_number}")
            continue
        seen[row_id] = row_hash
        rows.append(row)
    return rows


def append_decision_row_resume_safe(path: str | Path, row: dict[str, Any]) -> str:
    """Atomically append ``row`` unless the same id/hash already exists."""

    validate_decision_log_row(row)
    output = Path(path)
    existing = read_decision_log_rows(output)
    for previous in existing:
        if previous["decision_log_id"] == row["decision_log_id"]:
            if previous["decision_row_hash"] == row["decision_row_hash"]:
                return "duplicate"
            raise ValueError("conflicting duplicate decision row")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    lines = [_json_dump_line(previous) for previous in existing]
    lines.append(_json_dump_line(row))
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(output)
    return "appended"


def _load_resume_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _scheduled_offsets_to_visit(state: dict[str, Any], entry: dict[str, Any]) -> list[int]:
    completed = set(
        int(offset)
        for offset in state.get("entry_progress", {})
        .get(entry["entry_hash"], {})
        .get("completed_offsets", [])
    )
    offsets = []
    for offset in entry.get("row_group_local_offsets", []):
        int_offset = int(offset)
        if int_offset not in completed and int_offset not in offsets:
            offsets.append(int_offset)
    return offsets


def _progress_entry(entry: dict[str, Any]) -> dict[str, Any]:
    progress = deepcopy(entry)
    progress["row_group_local_offsets"] = sorted(
        {int(offset) for offset in entry.get("row_group_local_offsets", [])}
    )
    return progress


def _validate_schedule_contract(schedule: dict[str, Any]) -> None:
    if schedule.get("schema_version") != 1:
        raise ValueError("unsupported source schedule schema")
    if schedule.get("raw_text_serialized") is not False:
        raise ValueError("source schedule must set raw_text_serialized=false")
    if schedule.get("raw_url_serialized") is not False:
        raise ValueError("source schedule must set raw_url_serialized=false")
    if schedule.get("token_serialized") is not False:
        raise ValueError("source schedule must set token_serialized=false")
    if not schedule.get("schedule_hash"):
        raise ValueError("source schedule is missing schedule_hash")
    stable = {
        key: value
        for key, value in schedule.items()
        if key not in {"generated_at_utc", "schedule_hash"}
    }
    if schedule["schedule_hash"] != _hash_payload(stable):
        raise ValueError("source schedule hash mismatch")
    for entry in schedule.get("entries", []):
        offsets = [int(offset) for offset in entry.get("row_group_local_offsets", [])]
        if len(offsets) != len(set(offsets)):
            raise ValueError("duplicate scheduled row offset")


def _validate_schedule_mode(schedule: dict[str, Any], config: DecisionLogConfig) -> None:
    validation = schedule.get("validation") or {}
    if validation.get("diagnostic_only") and config.mode == "corpus_freeze":
        raise ValueError("diagnostic schedule cannot be used in corpus_freeze mode")


def visit_scheduled_entries(
    schedule: dict[str, Any],
    *,
    provider: ScheduledRowProvider,
    ontology: RiskOntology,
    decision_log_path: str | Path,
    resume_state_path: str | Path,
    config: DecisionLogConfig,
    max_entries: int | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Visit scheduled non-prefix offsets and write resume-safe public rows."""

    _validate_schedule_contract(schedule)
    _validate_schedule_mode(schedule, config)
    if max_entries is not None and max_entries < 1:
        raise ValueError("max_entries must be at least 1 when set")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be at least 1 when set")
    resume_path = Path(resume_state_path)
    state = _load_resume_state(resume_path)
    if state is None:
        state = build_initial_resume_state(
            run_id=config.run_id,
            schedule=schedule,
            source_tree_hash=config.source_tree_hash or "",
            config_hash=config.config_hash or "",
        )
    validate_resume_state(state, schedule)

    processed = 0
    duplicates = 0
    entries_seen = 0
    for entry in schedule.get("entries", []):
        if max_entries is not None and entries_seen >= max_entries:
            break
        offsets = _scheduled_offsets_to_visit(state, entry)
        if not offsets:
            continue
        if max_rows is not None:
            remaining_rows = max_rows - processed - duplicates
            if remaining_rows <= 0:
                break
            offsets = offsets[:remaining_rows]
        entries_seen += 1
        try:
            rows_by_offset = provider.fetch_rows(entry, offsets)
        except ProviderFailClosedError:
            raise
        except Exception as exc:
            rows_by_offset = {offset: exc for offset in offsets}
        completed_offsets = []
        for offset in offsets:
            row_record = rows_by_offset.get(offset)
            if row_record is None:
                row_record = RuntimeError(f"missing scheduled row offset {offset}")
            decision_row = build_public_decision_row(
                entry,
                offset,
                row_record,
                schedule_hash=schedule["schedule_hash"],
                source_metadata_hash=schedule["source_metadata_hash"],
                config=config,
                ontology=ontology,
            )
            outcome = append_decision_row_resume_safe(decision_log_path, decision_row)
            if outcome == "appended":
                processed += 1
            else:
                duplicates += 1
            completed_offsets.append(offset)
            state = update_resume_after_entry(
                state,
                _progress_entry(entry),
                completed_offsets=completed_offsets,
            )
            write_resume_state_atomic(state, resume_path)

    rows = read_decision_log_rows(decision_log_path)
    report = build_decision_log_report(rows, schedule=schedule, config=config)
    return {
        "decision_log_path": str(decision_log_path),
        "resume_state_path": str(resume_state_path),
        "processed_row_count": processed,
        "duplicate_row_count": duplicates,
        "total_row_count": len(rows),
        "diagnostic_only": report["diagnostic_only"],
        "corpus_freeze_ready": report["corpus_freeze_ready"],
        "status_counts": report["status_counts"],
    }


def build_decision_log_report(
    rows: list[dict[str, Any]],
    *,
    schedule: dict[str, Any],
    config: DecisionLogConfig,
    provider_observations: dict[str, Any] | None = None,
    run_caps: dict[str, int | None] | None = None,
) -> dict[str, Any]:
    """Build a no-raw-text manifest/report for a public decision log."""

    for row in rows:
        validate_decision_log_row(row)
    status_counts = dict(sorted(Counter(row["decision_status"] for row in rows).items()))
    diagnostic_only = bool(schedule.get("validation", {}).get("diagnostic_only")) or config.mode == "diagnostic_audit"
    report = {
        "schema_version": DECISION_LOG_REPORT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "run_id": config.run_id,
        "parent_run_id": config.parent_run_id,
        "artifact_set_id": config.artifact_set_id,
        "freeze_id": config.freeze_id,
        "mode": config.mode,
        "row_count": len(rows),
        "status_counts": status_counts,
        "source_schedule_hash": schedule.get("schedule_hash"),
        "source_metadata_hash": schedule.get("source_metadata_hash"),
        "provider": provider_observations or {},
        "run_caps": run_caps or {"max_entries": None, "max_rows": None},
        "source_tree_hash": config.source_tree_hash,
        "decision_log_config_hash": config.config_hash,
        "ontology_semantic_hashes": sorted(
            {row["ontology_semantic_hash"] for row in rows if row.get("ontology_semantic_hash")}
        ),
        "ontology_source_tree_hashes": sorted(
            {row["ontology_source_tree_hash"] for row in rows if row.get("ontology_source_tree_hash")}
        ),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "accepted_corpus_claim": False,
        "diagnostic_only": diagnostic_only,
        "corpus_freeze_ready": False,
        "validation": {
            "decision_log_rows_validated": True,
            "phase2a_reject_only": "accepted" not in status_counts,
            "raw_text_leakage_checked": True,
            "raw_url_leakage_checked": True,
        },
    }
    stable = {key: value for key, value in report.items() if key not in {"generated_at_utc", "manifest_hash"}}
    report["manifest_hash"] = _hash_payload(stable)
    _validate_no_public_raw_text(report)
    return report


def _provider_observations(provider: ScheduledRowProvider) -> dict[str, Any]:
    observations = getattr(provider, "observations", None)
    if callable(observations):
        payload = observations()
        if not isinstance(payload, dict):
            raise ValueError("provider observations must be a dict")
        return _sanitize_provider_observations(payload)
    return _sanitize_provider_observations({
        "provider_class": type(provider).__name__,
        "provider_version": None,
    })


def run_scheduled_audit(
    schedule: dict[str, Any],
    *,
    provider: ScheduledRowProvider,
    ontology: RiskOntology,
    decision_log_path: str | Path,
    resume_state_path: str | Path,
    report_path: str | Path,
    config: DecisionLogConfig,
    max_entries: int | None,
    max_rows: int | None,
) -> dict[str, Any]:
    """Run a capped scheduled audit and write its sanitized report."""

    if max_entries is None and max_rows is None:
        raise ValueError("scheduled audit requires --max-entries and/or --max-rows")
    result = visit_scheduled_entries(
        schedule,
        provider=provider,
        ontology=ontology,
        decision_log_path=decision_log_path,
        resume_state_path=resume_state_path,
        config=config,
        max_entries=max_entries,
        max_rows=max_rows,
    )
    report = build_decision_log_report(
        read_decision_log_rows(decision_log_path),
        schedule=schedule,
        config=config,
        provider_observations=_provider_observations(provider),
        run_caps={"max_entries": max_entries, "max_rows": max_rows},
    )
    _write_json(report, report_path)
    result["report_path"] = str(report_path)
    result["manifest_hash"] = report["manifest_hash"]
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _visit_local_parquet_from_args(args: argparse.Namespace) -> dict[str, Any]:
    schedule = _load_json(args.schedule)
    ontology = load_ontology(args.ontology)
    config = DecisionLogConfig(
        run_id=args.run_id,
        parent_run_id=args.parent_run_id,
        artifact_set_id=args.artifact_set_id,
        freeze_id=None,
        mode=args.mode,
        max_text_bytes=args.max_text_bytes,
        source_tree_hash=args.source_tree_hash,
        config_hash=args.config_hash,
    )
    result = run_scheduled_audit(
        schedule,
        provider=LocalParquetRowProvider(),
        ontology=ontology,
        decision_log_path=args.decision_log,
        resume_state_path=args.resume_state,
        report_path=args.report,
        config=config,
        max_entries=args.max_entries,
        max_rows=args.max_rows,
    )
    return result


def _visit_hf_parquet_from_args(args: argparse.Namespace) -> dict[str, Any]:
    schedule = _load_json(args.schedule)
    ontology = load_ontology(args.ontology)
    config = DecisionLogConfig(
        run_id=args.run_id,
        parent_run_id=args.parent_run_id,
        artifact_set_id=args.artifact_set_id,
        freeze_id=None,
        mode=args.mode,
        max_text_bytes=args.max_text_bytes,
        source_tree_hash=args.source_tree_hash,
        config_hash=args.config_hash,
    )
    provider = HFFsspecParquetRowProvider(
        cache_dir=args.cache_dir,
        cache_delta_cap_bytes=args.max_cache_delta_bytes,
        block_size_bytes=args.block_size_bytes,
        columns=tuple(args.columns.split(",")) if args.columns else HF_FSSPEC_DEFAULT_COLUMNS,
        cache_type=args.cache_type,
        max_cache_blocks=args.max_cache_blocks,
        source_concurrency=args.source_concurrency,
    )
    return run_scheduled_audit(
        schedule,
        provider=provider,
        ontology=ontology,
        decision_log_path=args.decision_log,
        resume_state_path=args.resume_state,
        report_path=args.report,
        config=config,
        max_entries=args.max_entries,
        max_rows=args.max_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    visit = sub.add_parser(
        "visit-local-parquet",
        help="Visit scheduled rows from locally readable Parquet files.",
    )
    visit.add_argument("--schedule", required=True)
    visit.add_argument("--ontology", default="risk_ontology")
    visit.add_argument("--decision-log", required=True)
    visit.add_argument("--resume-state", required=True)
    visit.add_argument("--report", required=True)
    visit.add_argument("--run-id", required=True)
    visit.add_argument("--artifact-set-id", required=True)
    visit.add_argument("--parent-run-id", default=None)
    visit.add_argument("--mode", choices=sorted(ALLOWED_MODES), default="diagnostic_audit")
    visit.add_argument("--max-text-bytes", type=int, default=DEFAULT_MAX_TEXT_BYTES)
    visit.add_argument("--source-tree-hash", default=None)
    visit.add_argument("--config-hash", default=None)
    visit.add_argument("--max-entries", type=int, default=None)
    visit.add_argument("--max-rows", type=int, default=None)

    hf = sub.add_parser(
        "visit-hf-parquet",
        help="Visit scheduled rows from Hugging Face HTTPS Parquet files through fsspec.",
    )
    hf.add_argument("--schedule", required=True)
    hf.add_argument("--ontology", default="risk_ontology")
    hf.add_argument("--decision-log", default="reports/decision_logs/phase2a_decisions.jsonl")
    hf.add_argument("--resume-state", default="reports/decision_logs/phase2a_resume.json")
    hf.add_argument("--report", default="reports/decision_logs/phase2a_decision_report.json")
    hf.add_argument("--run-id", required=True)
    hf.add_argument("--artifact-set-id", required=True)
    hf.add_argument("--parent-run-id", default=None)
    hf.add_argument("--mode", choices=sorted(ALLOWED_MODES), default="diagnostic_audit")
    hf.add_argument("--max-text-bytes", type=int, default=DEFAULT_MAX_TEXT_BYTES)
    hf.add_argument("--source-tree-hash", default=None)
    hf.add_argument("--config-hash", default=None)
    hf.add_argument("--cache-dir", default="reports/cache_probe/decision_logs")
    hf.add_argument("--max-cache-delta-bytes", type=int, required=True)
    hf.add_argument("--block-size-bytes", type=int, default=DEFAULT_HF_ROW_READ_BLOCK_SIZE_BYTES)
    hf.add_argument("--cache-type", choices=["blockcache"], default="blockcache")
    hf.add_argument("--max-cache-blocks", type=int, default=DEFAULT_HF_ROW_READ_MAX_CACHE_BLOCKS)
    hf.add_argument("--source-concurrency", type=int, default=DEFAULT_HF_ROW_SOURCE_CONCURRENCY)
    hf.add_argument("--columns", default=",".join(HF_FSSPEC_DEFAULT_COLUMNS))
    hf.add_argument("--max-entries", type=int, default=None)
    hf.add_argument("--max-rows", type=int, default=None)

    args = parser.parse_args()
    if args.command == "visit-local-parquet":
        result = _visit_local_parquet_from_args(args)
        print(json.dumps(result, sort_keys=True))
    elif args.command == "visit-hf-parquet":
        if args.max_entries is None and args.max_rows is None:
            parser.error("visit-hf-parquet requires --max-entries and/or --max-rows")
        result = _visit_hf_parquet_from_args(args)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
