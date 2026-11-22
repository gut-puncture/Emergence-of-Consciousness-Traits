"""Private local text cache for the final 35k sparse challenger.

This module intentionally separates private raw-text storage from public
classifier reports. Raw text is allowed only inside the configured private
SQLite cache; all returned reports and provider observations are no-raw.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import queue
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from twc.classifiers.feature_extraction import (
    LEGACY_PRIVATE_MINING_TEXT_HASH_ALGORITHM,
    TEXT_HASH_ALGORITHM_UTF8_SHA256_V1,
    WORD_RE,
)
from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.data.decision_schema import stable_json_hash


DEFAULT_CACHE_ROOT = Path("private_text_cache")   # was a folder on the author's machine
DEFAULT_CACHE_CAP_BYTES = 30_000_000_000
DEFAULT_MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
PRIVATE_CACHE_SCHEMA_VERSION = 1
PUBLIC_CACHE_REPORT_KIND = "final_35k_private_text_cache_validation_report"
PROGRESS_KIND = "final_35k_private_text_cache_progress"
TRANSIENT_ERROR_NAMES = {
    "TimeoutError",
    "ConnectionError",
    "ClientPayloadError",
    "ContentLengthError",
    "IncompleteRead",
    "ChunkedEncodingError",
    "ReadTimeout",
}


def prepare_private_cache_root(
    cache_root: str | Path,
    *,
    forbidden_roots: Sequence[str | Path] | None = None,
    cache_cap_bytes: int = DEFAULT_CACHE_CAP_BYTES,
) -> Path:
    """Create and permission a cache root after path-safety checks."""

    requested = Path(cache_root).expanduser()
    resolved = requested.resolve(strict=False)
    forbidden = list(forbidden_roots) if forbidden_roots is not None else _default_forbidden_roots()
    for root in forbidden:
        root_path = Path(root).expanduser().resolve(strict=False)
        if _is_relative_to(resolved, root_path):
            raise ValueError("private cache root is inside a forbidden project path")
    if any(part == ".git" for part in resolved.parts):
        raise ValueError("private cache root must not be inside .git")
    if cache_cap_bytes < 1:
        raise ValueError("cache_cap_bytes must be positive")

    resolved.mkdir(parents=True, exist_ok=True)
    os.chmod(resolved, 0o700)
    free_bytes = shutil.disk_usage(resolved).free
    if free_bytes < int(cache_cap_bytes):
        raise ValueError("not enough free disk for configured private cache cap")
    marker = resolved / ".metadata_never_index"
    marker.touch(exist_ok=True)
    os.chmod(marker, 0o600)
    return resolved


def build_final35k_private_text_cache(
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    public_report_path: str | Path | None = None,
    expected_fit_rows: int | None = 34_992,
    forbidden_roots: Sequence[str | Path] | None = None,
    cache_cap_bytes: int = DEFAULT_CACHE_CAP_BYTES,
    fetch_concurrency: int = 16,
    source_group_batch_size: int = 64,
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
    retry_attempts: int = 5,
    max_rows: int | None = None,
    row_provider_factory: Callable[[], Any] | None = None,
    fetch_execution: str = "thread",
    fetch_batch_timeout_seconds: float | None = None,
    fetch_process_start_method: str | None = None,
) -> dict[str, Any]:
    """Fetch and store exact final fit-eligible private text rows."""

    cache_path = prepare_private_cache_root(
        cache_root,
        forbidden_roots=forbidden_roots,
        cache_cap_bytes=cache_cap_bytes,
    )
    if fetch_concurrency < 1:
        raise ValueError("fetch_concurrency must be positive")
    if source_group_batch_size < 1:
        raise ValueError("source_group_batch_size must be positive")
    if max_document_bytes < 1:
        raise ValueError("max_document_bytes must be positive")
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be positive")
    if fetch_execution not in {"thread", "process"}:
        raise ValueError("fetch_execution must be 'thread' or 'process'")
    if fetch_batch_timeout_seconds is not None and float(fetch_batch_timeout_seconds) <= 0:
        raise ValueError("fetch_batch_timeout_seconds must be positive")
    if fetch_process_start_method is None:
        available_start_methods = multiprocessing.get_all_start_methods()
        fetch_process_start_method = "fork" if "fork" in available_start_methods else "spawn"
    elif fetch_process_start_method not in multiprocessing.get_all_start_methods():
        raise ValueError("unsupported fetch_process_start_method")
    if fetch_execution == "thread":
        if row_provider is None:
            if row_provider_factory is None:
                raise ValueError("row_provider is required for thread fetch execution")
            row_provider = row_provider_factory()
    elif row_provider_factory is None:
        raise ValueError("row_provider_factory is required for process fetch execution")

    lock_file = _acquire_cache_build_lock(cache_path)
    try:
        _validate_no_raw_input(training_manifest, "training_manifest")
        _validate_schedule_no_raw(schedule)
        selected = _selected_fit_rows(
            training_manifest=training_manifest,
            candidates=candidates,
            schedule=schedule,
            max_rows=max_rows,
        )
        if expected_fit_rows is not None and max_rows is None and len(selected.rows) != int(expected_fit_rows):
            raise ValueError(f"expected {expected_fit_rows} fit-eligible rows, found {len(selected.rows)}")

        db_path = cache_path / "cache.sqlite3"
        conn = _connect_cache(db_path)
        try:
            _init_schema(conn)
            _write_metadata(
                conn,
                training_manifest=training_manifest,
                candidates=candidates,
                schedule=schedule,
                selected=selected,
                cache_cap_bytes=cache_cap_bytes,
                max_document_bytes=max_document_bytes,
            )
            _chmod_private_cache_files(cache_path)
            groups = _fetch_groups_from_selected(selected.rows)
            _reconcile_cache_for_selected_rows(conn, selected_rows=selected.rows, groups=groups)
            pending_groups = [
                group for group in groups
                if not _fetch_unit_complete(conn, group)
            ]
            pending_groups.sort(key=lambda group: (_fetch_unit_retry_rank(conn, group), group.source_key, group.row_group_index))
            _write_progress(
                cache_path,
                status="building",
                cached_row_count=_cached_row_count(conn),
                expected_row_count=len(selected.rows),
                completed_fetch_unit_count=len(groups) - len(pending_groups),
                total_fetch_unit_count=len(groups),
                failed_fetch_unit_count=_failed_fetch_unit_count(conn),
                timeout_fetch_unit_count=_timeout_fetch_unit_count(conn),
                last_error_class=_last_error_class(conn),
            )
            source_group_batches = _iter_source_group_batches(
                pending_groups,
                max_groups_per_batch=source_group_batch_size,
            )
            if fetch_execution == "process":
                _fetch_source_group_batches_process_isolated(
                    conn,
                    cache_path=cache_path,
                    source_group_batches=source_group_batches,
                    row_provider_factory=row_provider_factory,
                    retry_attempts=retry_attempts,
                    max_document_bytes=max_document_bytes,
                    expected_row_count=len(selected.rows),
                    total_fetch_unit_count=len(groups),
                    fetch_concurrency=fetch_concurrency,
                    fetch_batch_timeout_seconds=fetch_batch_timeout_seconds or 180.0,
                    fetch_process_start_method=fetch_process_start_method,
                )
            elif fetch_concurrency == 1 or len(source_group_batches) <= 1:
                for source_group in source_group_batches:
                    result = _fetch_source_group_with_retries(
                        row_provider=row_provider,
                        source_group=source_group,
                        retry_attempts=retry_attempts,
                    )
                    _write_fetch_results(
                        conn,
                        cache_path=cache_path,
                        result=result,
                        max_document_bytes=max_document_bytes,
                        expected_row_count=len(selected.rows),
                        total_fetch_unit_count=len(groups),
                    )
            else:
                with ThreadPoolExecutor(max_workers=min(int(fetch_concurrency), len(source_group_batches))) as executor:
                    future_to_source_group = {
                        executor.submit(
                            _fetch_source_group_with_retries,
                            row_provider=row_provider,
                            source_group=source_group,
                            retry_attempts=retry_attempts,
                        ): source_group
                        for source_group in source_group_batches
                    }
                    for future in as_completed(future_to_source_group):
                        source_group = future_to_source_group[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            for group in source_group:
                                _write_failed_group(conn, group=group, error=exc)
                            raise
                        _write_fetch_results(
                            conn,
                            cache_path=cache_path,
                            result=result,
                            max_document_bytes=max_document_bytes,
                            expected_row_count=len(selected.rows),
                            total_fetch_unit_count=len(groups),
                        )
        finally:
            conn.close()
            _chmod_private_cache_files(cache_path)
    finally:
        _release_cache_build_lock(lock_file)

    return validate_final35k_private_text_cache(
        training_manifest=training_manifest,
        candidates=candidates,
        schedule=schedule,
        cache_root=cache_path,
        public_report_path=public_report_path,
        expected_fit_rows=expected_fit_rows if max_rows is None else len(selected.rows),
        forbidden_roots=forbidden_roots,
        cache_cap_bytes=cache_cap_bytes,
        max_rows=max_rows,
    )


def validate_final35k_private_text_cache(
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    public_report_path: str | Path | None = None,
    expected_fit_rows: int | None = 34_992,
    forbidden_roots: Sequence[str | Path] | None = None,
    cache_cap_bytes: int = DEFAULT_CACHE_CAP_BYTES,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Validate the private cache and optionally write a no-raw public report."""

    cache_path = prepare_private_cache_root(
        cache_root,
        forbidden_roots=forbidden_roots,
        cache_cap_bytes=1,
    )
    _validate_no_raw_input(training_manifest, "training_manifest")
    _validate_schedule_no_raw(schedule)
    selected = _selected_fit_rows(
        training_manifest=training_manifest,
        candidates=candidates,
        schedule=schedule,
        max_rows=max_rows,
    )
    if expected_fit_rows is not None and max_rows is None and len(selected.rows) != int(expected_fit_rows):
        raise ValueError(f"expected {expected_fit_rows} fit-eligible rows, found {len(selected.rows)}")

    db_path = cache_path / "cache.sqlite3"
    if not db_path.exists():
        raise ValueError("private text cache database is missing")
    conn = _connect_cache(db_path)
    try:
        _init_schema(conn)
        expected_ids = {row.candidate_id for row in selected.rows}
        actual_rows = list(conn.execute(
            "select candidate_id, split, classifier_label, fit_eligible, training_weight, "
            "source_file_id, row_group_index, row_group_local_row_index, text_sha256, "
            "text_hash_algorithm, byte_length, word_count from rows"
        ))
        actual_ids = {str(row[0]) for row in actual_rows}
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if missing:
            raise ValueError("private text cache missing expected candidates")
        if extra:
            raise ValueError("private text cache contains extra candidates")
        cached_row_count = len(actual_rows)
        if cached_row_count != len(selected.rows):
            raise ValueError("private text cache row count mismatch")

        split_counts = Counter(str(row[1]) for row in actual_rows)
        label_counts = Counter(str(row[2]) for row in actual_rows)
        byte_lengths = [int(row[10]) for row in actual_rows]
        total_bytes = sum(byte_lengths)
        duplicate_physical_count = _duplicate_count(conn, "source_file_id, row_group_index, row_group_local_row_index")
        duplicate_text_hash_count = _duplicate_count(conn, "text_sha256")
        if duplicate_physical_count:
            raise ValueError("private text cache duplicate physical locator")
        if duplicate_text_hash_count:
            raise ValueError("private text cache duplicate text hash")
    finally:
        conn.close()
        _chmod_private_cache_files(cache_path)

    report = {
        "artifact_kind": PUBLIC_CACHE_REPORT_KIND,
        "schema_version": 1,
        "status": "complete" if cached_row_count == len(selected.rows) else "incomplete",
        "private_cache_schema_version": PRIVATE_CACHE_SCHEMA_VERSION,
        "private_cache_root_hash": stable_json_hash({"private_cache_root": str(cache_path)}),
        "cache_db_file_name": db_path.name,
        "cache_db_sha256": _sha256_file(db_path),
        "training_data_manifest_hash": _training_manifest_hash(training_manifest),
        "candidate_manifest_hash": _candidate_manifest_hash(candidates),
        "schedule_hash": _schedule_hash(schedule),
        "expected_fit_row_count": len(selected.rows),
        "cached_row_count": cached_row_count,
        "audit_only_row_count": selected.audit_only_row_count,
        "input_missing_text_hash_count": selected.missing_text_hash_count,
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "total_text_bytes": int(total_bytes),
        "max_document_bytes": max(byte_lengths) if byte_lengths else 0,
        "cache_cap_bytes": int(cache_cap_bytes),
        "duplicate_physical_locator_count": duplicate_physical_count,
        "duplicate_text_hash_count": duplicate_text_hash_count,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    report["report_hash"] = stable_json_hash({key: value for key, value in report.items() if key != "report_hash"})
    validate_public_no_raw_payload(report)
    if public_report_path is not None:
        _write_public_report(Path(public_report_path), report)
    return report


def inspect_final35k_private_text_cache(
    *,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    forbidden_roots: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Return no-raw cache counts without needing source manifests."""

    cache_path = prepare_private_cache_root(cache_root, forbidden_roots=forbidden_roots, cache_cap_bytes=1)
    db_path = cache_path / "cache.sqlite3"
    if not db_path.exists():
        raise ValueError("private text cache database is missing")
    conn = _connect_cache(db_path)
    try:
        row_count = int(conn.execute("select count(*) from rows").fetchone()[0])
        total_bytes = int(conn.execute("select coalesce(sum(byte_length), 0) from rows").fetchone()[0])
        split_counts = dict(conn.execute("select split, count(*) from rows group by split").fetchall())
        label_counts = dict(conn.execute("select classifier_label, count(*) from rows group by classifier_label").fetchall())
    finally:
        conn.close()
    payload = {
        "artifact_kind": "final_35k_private_text_cache_inspection",
        "schema_version": 1,
        "cache_db_file_name": db_path.name,
        "cache_db_sha256": _sha256_file(db_path),
        "cached_row_count": row_count,
        "total_text_bytes": total_bytes,
        "split_counts": dict(sorted((str(k), int(v)) for k, v in split_counts.items())),
        "label_counts": dict(sorted((str(k), int(v)) for k, v in label_counts.items())),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    validate_public_no_raw_payload(payload)
    return payload


def audit_final35k_private_text_cache_duplicate_collisions(
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    expected_fit_rows: int | None = 34_992,
    forbidden_roots: Sequence[str | Path] | None = None,
    cache_cap_bytes: int = DEFAULT_CACHE_CAP_BYTES,
    fetch_concurrency: int = 16,
    source_group_batch_size: int = 64,
    retry_attempts: int = 5,
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Audit text-hash collisions without writing raw text to public output."""

    cache_path = prepare_private_cache_root(
        cache_root,
        forbidden_roots=forbidden_roots,
        cache_cap_bytes=cache_cap_bytes,
    )
    if fetch_concurrency < 1:
        raise ValueError("fetch_concurrency must be positive")
    if source_group_batch_size < 1:
        raise ValueError("source_group_batch_size must be positive")
    if retry_attempts < 1:
        raise ValueError("retry_attempts must be positive")
    _validate_no_raw_input(training_manifest, "training_manifest")
    _validate_schedule_no_raw(schedule)
    selected = _selected_fit_rows(
        training_manifest=training_manifest,
        candidates=candidates,
        schedule=schedule,
        max_rows=max_rows,
    )
    if expected_fit_rows is not None and max_rows is None and len(selected.rows) != int(expected_fit_rows):
        raise ValueError(f"expected {expected_fit_rows} fit-eligible rows, found {len(selected.rows)}")

    db_path = cache_path / "cache.sqlite3"
    if not db_path.exists():
        raise ValueError("private text cache database is missing")
    conn = _connect_cache(db_path)
    try:
        _init_schema(conn)
        audit_rows = _cached_duplicate_audit_rows(conn, selected.rows)
    finally:
        conn.close()
        _chmod_private_cache_files(cache_path)

    cached_ids = {str(row["candidate_id"]) for row in audit_rows}
    missing_rows = [row for row in selected.rows if row.candidate_id not in cached_ids]
    fetched_rows = _fetch_duplicate_audit_rows(
        missing_rows,
        row_provider=row_provider,
        fetch_concurrency=fetch_concurrency,
        source_group_batch_size=source_group_batch_size,
        retry_attempts=retry_attempts,
        max_document_bytes=max_document_bytes,
    )
    audit_rows.extend(fetched_rows)
    by_text_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        by_text_hash[str(row["text_sha256"])].append(row)
    collisions = [
        _collision_report(text_hash, rows)
        for text_hash, rows in sorted(by_text_hash.items())
        if len(rows) > 1
    ]
    duplicate_member_count = sum(int(collision["member_count"]) for collision in collisions)
    payload = {
        "artifact_kind": "final_35k_private_text_cache_duplicate_audit",
        "schema_version": 1,
        "status": "duplicates_found" if collisions else "no_duplicates_found",
        "private_cache_schema_version": PRIVATE_CACHE_SCHEMA_VERSION,
        "private_cache_root_hash": stable_json_hash({"private_cache_root": str(cache_path)}),
        "cache_db_file_name": db_path.name,
        "cache_db_sha256": _sha256_file(db_path),
        "training_data_manifest_hash": _training_manifest_hash(training_manifest),
        "candidate_manifest_hash": _candidate_manifest_hash(candidates),
        "schedule_hash": _schedule_hash(schedule),
        "expected_fit_row_count": len(selected.rows),
        "audited_row_count": len(audit_rows),
        "cached_seed_row_count": len(cached_ids),
        "refetched_audit_row_count": len(fetched_rows),
        "input_missing_text_hash_count": selected.missing_text_hash_count,
        "duplicate_text_hash_collision_count": len(collisions),
        "duplicate_text_hash_member_count": duplicate_member_count,
        "collisions": collisions,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    payload["report_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "report_hash"})
    validate_public_no_raw_payload(payload)
    return payload


class PrivateTextCacheRowProvider:
    """Resolve scheduled row offsets from the private SQLite text cache."""

    provider_version = 1
    manage_hf_cache_env = False

    def __init__(self, *, cache_root: str | Path = DEFAULT_CACHE_ROOT) -> None:
        self.cache_root = Path(cache_root).expanduser().resolve(strict=False)
        self.db_path = self.cache_root / "cache.sqlite3"
        if not self.db_path.exists():
            raise ValueError("private text cache database is missing")
        self._cache_hit_row_count = 0
        self._cache_miss_row_count = 0
        self._fetch_call_count = 0

    def observations(self) -> dict[str, Any]:
        payload = {
            "provider_class": type(self).__name__,
            "provider_version": self.provider_version,
            "private_cache_root_hash": stable_json_hash({"private_cache_root": str(self.cache_root)}),
            "cache_db_file_name": self.db_path.name,
            "cache_db_sha256": _sha256_file(self.db_path),
            "cache_hit_row_count": int(self._cache_hit_row_count),
            "cache_miss_row_count": int(self._cache_miss_row_count),
            "fetch_call_count": int(self._fetch_call_count),
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
            "accepted_manifest": False,
            "training_shards_created": False,
        }
        validate_public_no_raw_payload(payload)
        return payload

    def fetch_rows(self, entry: dict[str, Any], offsets: list[int]) -> dict[int, dict[str, str]]:
        key = _schedule_result_key(entry)
        return self.fetch_row_groups([(entry, offsets)])[key]

    def fetch_row_groups(
        self,
        entry_offsets: list[tuple[dict[str, Any], list[int]]],
    ) -> dict[tuple[str, int], dict[int, dict[str, str]]]:
        self._fetch_call_count += 1
        resolved: dict[tuple[str, int], dict[int, dict[str, str]]] = {}
        conn = _connect_cache(self.db_path)
        try:
            for entry, offsets in entry_offsets:
                source_file_id, row_group_index = _schedule_result_key(entry)
                row_group_count = conn.execute(
                    "select count(*) from rows where source_file_id = ? and row_group_index = ?",
                    (source_file_id, row_group_index),
                ).fetchone()[0]
                if int(row_group_count) == 0:
                    self._cache_miss_row_count += len(offsets)
                    raise ValueError("private text cache row-group mismatch")
                rows_by_offset: dict[int, dict[str, str]] = {}
                for offset in [int(value) for value in offsets]:
                    row = conn.execute(
                        "select raw_text, text_sha256, text_hash_algorithm from rows "
                        "where source_file_id = ? and row_group_index = ? and row_group_local_row_index = ?",
                        (source_file_id, row_group_index, offset),
                    ).fetchone()
                    if row is None:
                        self._cache_miss_row_count += 1
                        raise ValueError("private text cache miss")
                    raw_text, expected_hash, algorithm = str(row[0]), str(row[1]), str(row[2])
                    actual_hash = _text_hash_for_algorithm(raw_text, algorithm)
                    if actual_hash != expected_hash:
                        raise ValueError("private text cache hash drift")
                    rows_by_offset[offset] = {"text": raw_text}
                    self._cache_hit_row_count += 1
                resolved[(source_file_id, row_group_index)] = rows_by_offset
        finally:
            conn.close()
        return resolved


class _SelectedRows:
    def __init__(
        self,
        *,
        rows: list["_SelectedRow"],
        audit_only_row_count: int,
        missing_text_hash_count: int,
    ) -> None:
        self.rows = rows
        self.audit_only_row_count = audit_only_row_count
        self.missing_text_hash_count = missing_text_hash_count


class _SelectedRow:
    def __init__(
        self,
        *,
        manifest_row: Mapping[str, Any],
        candidate: Mapping[str, Any],
        schedule_entry: Mapping[str, Any],
    ) -> None:
        self.manifest_row = manifest_row
        self.candidate = candidate
        self.schedule_entry = schedule_entry
        self.candidate_id = _required_str(manifest_row, "candidate_id")
        self.source_file_id = _required_str(candidate, "source_file_id")
        self.row_group_index = int(candidate["row_group_index"])
        self.offset = int(candidate["row_group_local_row_index"])


class _FetchGroup:
    def __init__(self, *, entry: Mapping[str, Any], selected_rows: list[_SelectedRow]) -> None:
        self.entry = dict(entry)
        self.selected_rows = selected_rows
        self.source_file_id = str(entry["source_file_id"])
        self.row_group_index = int(entry["row_group_index"])
        self.offsets = sorted({row.offset for row in selected_rows})
        self.offsets_hash = stable_json_hash(
            {
                "source_file_id": self.source_file_id,
                "row_group_index": self.row_group_index,
                "offsets": self.offsets,
            }
        )

    @property
    def source_key(self) -> str:
        source = self.entry.get("parquet_url_or_path") or self.source_file_id
        return str(source)


def _selected_fit_rows(
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    max_rows: int | None,
) -> _SelectedRows:
    candidate_by_id = _index_candidates(candidates)
    schedule_by_key = _schedule_entries_by_key(schedule)
    rows = list(training_manifest.get("rows") or [])
    selected: list[_SelectedRow] = []
    audit_only_count = 0
    missing_text_hash_count = 0
    seen_physical: set[tuple[str, int, int]] = set()
    seen_text_hashes: set[str] = set()
    for manifest_row in rows:
        if manifest_row.get("fit_eligible") is False or str(manifest_row.get("split")) == "audit":
            audit_only_count += 1
            continue
        candidate_id = _required_str(manifest_row, "candidate_id")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError("candidate missing for training manifest row")
        if _required_str(manifest_row, "candidate_hash") != _required_str(candidate, "candidate_hash"):
            raise ValueError("candidate hash drift")
        entry = _schedule_entry_for_candidate(candidate, schedule_by_key)
        physical = (
            _required_str(candidate, "source_file_id"),
            int(candidate["row_group_index"]),
            int(candidate["row_group_local_row_index"]),
        )
        if physical in seen_physical:
            raise ValueError("duplicate physical locator")
        seen_physical.add(physical)
        text_hash = str(candidate.get("text_hash_sha256") or candidate.get("text_hash") or "")
        if text_hash:
            _require_sha256(text_hash, "text_hash")
            if text_hash in seen_text_hashes:
                raise ValueError("duplicate text hash")
            seen_text_hashes.add(text_hash)
            _required_str(candidate, "text_hash_algorithm")
        else:
            missing_text_hash_count += 1
        selected.append(_SelectedRow(manifest_row=manifest_row, candidate=candidate, schedule_entry=entry))
        if max_rows is not None and len(selected) >= int(max_rows):
            break
    return _SelectedRows(
        rows=selected,
        audit_only_row_count=audit_only_count,
        missing_text_hash_count=missing_text_hash_count,
    )


def _fetch_groups_from_selected(rows: Sequence[_SelectedRow]) -> list[_FetchGroup]:
    grouped: dict[tuple[str, int], list[_SelectedRow]] = defaultdict(list)
    entry_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (row.source_file_id, row.row_group_index)
        grouped[key].append(row)
        entry_by_key[key] = row.schedule_entry
    return [
        _FetchGroup(entry=entry_by_key[key], selected_rows=grouped[key])
        for key in sorted(grouped)
    ]


def _cached_duplicate_audit_rows(
    conn: sqlite3.Connection,
    selected_rows: Sequence[_SelectedRow],
) -> list[dict[str, Any]]:
    selected_by_id = {row.candidate_id: row for row in selected_rows}
    outputs: list[dict[str, Any]] = []
    for db_row in conn.execute(
        "select candidate_id, candidate_hash, split, classifier_label, fit_eligible, training_weight, "
        "source_file_id, row_group_index, row_group_local_row_index, text_sha256, text_hash_algorithm, "
        "byte_length, word_count from rows"
    ):
        candidate_id = str(db_row[0])
        selected = selected_by_id.get(candidate_id)
        if selected is None:
            continue
        outputs.append(
            _duplicate_audit_row_payload(
                selected=selected,
                candidate_hash=str(db_row[1]),
                split=str(db_row[2]),
                classifier_label=str(db_row[3]),
                fit_eligible=bool(int(db_row[4])),
                training_weight=float(db_row[5]),
                text_sha256=str(db_row[9]),
                text_hash_algorithm=str(db_row[10]),
                byte_length=int(db_row[11]),
                word_count=int(db_row[12]),
                audit_source="cache",
            )
        )
    return outputs


def _fetch_duplicate_audit_rows(
    rows: Sequence[_SelectedRow],
    *,
    row_provider: Any,
    fetch_concurrency: int,
    source_group_batch_size: int,
    retry_attempts: int,
    max_document_bytes: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    groups = _fetch_groups_from_selected(rows)
    source_group_batches = _iter_source_group_batches(groups, max_groups_per_batch=source_group_batch_size)
    outputs: list[dict[str, Any]] = []
    if fetch_concurrency == 1 or len(source_group_batches) <= 1:
        for source_group in source_group_batches:
            result = _fetch_source_group_with_retries(
                row_provider=row_provider,
                source_group=source_group,
                retry_attempts=retry_attempts,
            )
            outputs.extend(
                _duplicate_audit_rows_from_fetch_result(
                    result,
                    max_document_bytes=max_document_bytes,
                )
            )
        return outputs
    with ThreadPoolExecutor(max_workers=min(int(fetch_concurrency), len(source_group_batches))) as executor:
        futures = {
            executor.submit(
                _fetch_source_group_with_retries,
                row_provider=row_provider,
                source_group=source_group,
                retry_attempts=retry_attempts,
            ): source_group
            for source_group in source_group_batches
        }
        for future in as_completed(futures):
            result = future.result()
            outputs.extend(
                _duplicate_audit_rows_from_fetch_result(
                    result,
                    max_document_bytes=max_document_bytes,
                )
            )
    return outputs


def _duplicate_audit_rows_from_fetch_result(
    result: Sequence[tuple[_FetchGroup, Mapping[int, Any]]],
    *,
    max_document_bytes: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for group, rows_by_offset in result:
        for selected in group.selected_rows:
            private_row = rows_by_offset.get(selected.offset)
            if not isinstance(private_row, Mapping):
                raise ValueError("missing cached source row")
            private_text = private_row.get("text")
            if not isinstance(private_text, str) or not private_text:
                raise ValueError("missing or empty text")
            byte_length = len(private_text.encode("utf-8", errors="surrogatepass"))
            if byte_length > int(max_document_bytes):
                raise ValueError("document text exceeds configured byte cap")
            algorithm = str(selected.candidate.get("text_hash_algorithm") or TEXT_HASH_ALGORITHM_UTF8_SHA256_V1)
            expected = _expected_text_hash(selected.candidate, private_text, algorithm)
            actual = _text_hash_for_algorithm(private_text, algorithm)
            if expected != actual:
                raise ValueError("text hash drift")
            outputs.append(
                _duplicate_audit_row_payload(
                    selected=selected,
                    candidate_hash=_required_str(selected.candidate, "candidate_hash"),
                    split=_required_str(selected.manifest_row, "split"),
                    classifier_label=_required_str(selected.manifest_row, "classifier_label"),
                    fit_eligible=selected.manifest_row.get("fit_eligible") is not False,
                    training_weight=float(selected.manifest_row.get("training_weight", 1.0)),
                    text_sha256=actual,
                    text_hash_algorithm=algorithm,
                    byte_length=byte_length,
                    word_count=len(WORD_RE.findall(private_text)),
                    audit_source="refetch",
                )
            )
    return outputs


def _duplicate_audit_row_payload(
    *,
    selected: _SelectedRow,
    candidate_hash: str,
    split: str,
    classifier_label: str,
    fit_eligible: bool,
    training_weight: float,
    text_sha256: str,
    text_hash_algorithm: str,
    byte_length: int,
    word_count: int,
    audit_source: str,
) -> dict[str, Any]:
    _require_sha256(text_sha256, "text_sha256")
    _require_sha256(candidate_hash, "candidate_hash")
    physical_hash = str(selected.candidate.get("physical_source_row_key_hash") or "")
    if not physical_hash:
        physical_hash = stable_json_hash(
            {
                "source_file_id": selected.source_file_id,
                "row_group_index": selected.row_group_index,
                "row_group_local_row_index": selected.offset,
            }
        )
    _require_sha256(physical_hash, "physical_source_row_key_hash")
    payload = {
        "candidate_id": selected.candidate_id,
        "candidate_id_hash": stable_json_hash({"candidate_id": selected.candidate_id}),
        "candidate_hash": candidate_hash,
        "physical_source_row_key_hash": physical_hash,
        "source_identity_hash": stable_json_hash({"source_file_id": selected.source_file_id}),
        "text_sha256": text_sha256,
        "text_hash_algorithm": text_hash_algorithm,
        "split": split,
        "classifier_label": classifier_label,
        "fit_eligible": bool(fit_eligible),
        "training_weight": float(training_weight),
        "byte_length": int(byte_length),
        "word_count": int(word_count),
        "audit_source": audit_source,
    }
    payload["audit_row_hash"] = stable_json_hash(payload)
    validate_public_no_raw_payload({key: value for key, value in payload.items() if key != "candidate_id"})
    return payload


def _collision_report(text_hash: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require_sha256(text_hash, "text_sha256")
    members = [
        {
            key: row[key]
            for key in (
                "candidate_id_hash",
                "candidate_hash",
                "physical_source_row_key_hash",
                "source_identity_hash",
                "split",
                "classifier_label",
                "fit_eligible",
                "training_weight",
                "byte_length",
                "word_count",
                "audit_source",
                "audit_row_hash",
            )
        }
        for row in sorted(rows, key=lambda item: str(item["candidate_id_hash"]))
    ]
    split_values = {str(row["split"]) for row in rows}
    label_values = {str(row["classifier_label"]) for row in rows}
    payload = {
        "text_sha256": text_hash,
        "member_count": len(members),
        "same_split": len(split_values) == 1,
        "same_classifier_label": len(label_values) == 1,
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "label_counts": dict(sorted(Counter(str(row["classifier_label"]) for row in rows).items())),
        "members": members,
    }
    payload["collision_hash"] = stable_json_hash(payload)
    validate_public_no_raw_payload(payload)
    return payload


def _iter_source_group_batches(
    groups: Sequence[_FetchGroup],
    *,
    max_groups_per_batch: int,
) -> list[list[_FetchGroup]]:
    if max_groups_per_batch < 1:
        raise ValueError("max_groups_per_batch must be positive")
    by_source: dict[str, list[_FetchGroup]] = defaultdict(list)
    ordered_sources: list[str] = []
    seen_sources: set[str] = set()
    for group in groups:
        source_key = group.source_key
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            ordered_sources.append(source_key)
        by_source[source_key].append(group)
    batches: list[list[_FetchGroup]] = []
    for source in ordered_sources:
        source_groups = by_source[source]
        for start in range(0, len(source_groups), int(max_groups_per_batch)):
            batches.append(source_groups[start : start + int(max_groups_per_batch)])
    return batches


def _fetch_source_group_with_retries(
    *,
    row_provider: Any,
    source_group: list[_FetchGroup],
    retry_attempts: int,
) -> list[tuple[_FetchGroup, Mapping[int, Any]]]:
    attempt = 0
    while True:
        attempt += 1
        try:
            return _fetch_source_group(row_provider=row_provider, source_group=source_group)
        except Exception as exc:
            if attempt >= retry_attempts or not _is_transient_error(exc):
                raise
            time.sleep(min(30.0, 0.5 * (2 ** (attempt - 1))))


def _fetch_source_group(*, row_provider: Any, source_group: list[_FetchGroup]) -> list[tuple[_FetchGroup, Mapping[int, Any]]]:
    if hasattr(row_provider, "fetch_row_groups"):
        result = row_provider.fetch_row_groups([(dict(group.entry), list(group.offsets)) for group in source_group])
        outputs = []
        for group in source_group:
            key = (group.source_file_id, group.row_group_index)
            rows_by_offset = result.get(key)
            if not isinstance(rows_by_offset, Mapping):
                raise ValueError("missing cached source row")
            outputs.append((group, rows_by_offset))
        return outputs
    return [
        (group, row_provider.fetch_rows(dict(group.entry), list(group.offsets)))
        for group in source_group
    ]


def _fetch_process_worker(
    *,
    row_provider_factory: Callable[[], Any],
    source_group: list[_FetchGroup],
    retry_attempts: int,
    result_queue: Any,
) -> None:
    try:
        row_provider = row_provider_factory()
        result = _fetch_source_group_with_retries(
            row_provider=row_provider,
            source_group=source_group,
            retry_attempts=retry_attempts,
        )
        result_queue.put(("ok", result))
    except BaseException as exc:  # pragma: no cover - exercised through parent.
        result_queue.put(("error", type(exc).__name__))


class _ActiveFetchProcess:
    def __init__(
        self,
        *,
        process: multiprocessing.Process,
        result_queue: Any,
        source_group: list[_FetchGroup],
        started_at: float,
    ) -> None:
        self.process = process
        self.result_queue = result_queue
        self.source_group = source_group
        self.started_at = float(started_at)


def _fetch_source_group_batches_process_isolated(
    conn: sqlite3.Connection,
    *,
    cache_path: Path,
    source_group_batches: list[list[_FetchGroup]],
    row_provider_factory: Callable[[], Any],
    retry_attempts: int,
    max_document_bytes: int,
    expected_row_count: int,
    total_fetch_unit_count: int,
    fetch_concurrency: int,
    fetch_batch_timeout_seconds: float,
    fetch_process_start_method: str,
) -> None:
    if not source_group_batches:
        return
    ctx = multiprocessing.get_context(fetch_process_start_method)
    active: dict[int, _ActiveFetchProcess] = {}
    next_batch_index = 0
    first_error: BaseException | None = None

    def start_next() -> None:
        nonlocal next_batch_index
        source_group = source_group_batches[next_batch_index]
        next_batch_index += 1
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_fetch_process_worker,
            kwargs={
                "row_provider_factory": row_provider_factory,
                "source_group": source_group,
                "retry_attempts": retry_attempts,
                "result_queue": result_queue,
            },
        )
        process.start()
        active[int(process.pid or id(process))] = _ActiveFetchProcess(
            process=process,
            result_queue=result_queue,
            source_group=source_group,
            started_at=time.monotonic(),
        )

    while next_batch_index < len(source_group_batches) or active:
        while next_batch_index < len(source_group_batches) and len(active) < int(fetch_concurrency):
            start_next()
        progressed = False
        now = time.monotonic()
        for active_key, active_fetch in list(active.items()):
            process = active_fetch.process
            timed_out = (now - active_fetch.started_at) > float(fetch_batch_timeout_seconds)
            if timed_out:
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                _write_failed_source_group(
                    conn,
                    cache_path=cache_path,
                    source_group=active_fetch.source_group,
                    error_class="TimeoutError",
                    expected_row_count=expected_row_count,
                    total_fetch_unit_count=total_fetch_unit_count,
                )
                first_error = first_error or TimeoutError("source batch fetch timed out")
                active.pop(active_key, None)
                progressed = True
                continue
            if process.exitcode is None:
                continue
            process.join(timeout=1)
            try:
                status, payload = active_fetch.result_queue.get_nowait()
            except queue.Empty:
                status, payload = "error", "ChildProcessNoResult"
            if status == "ok":
                _write_fetch_results(
                    conn,
                    cache_path=cache_path,
                    result=payload,
                    max_document_bytes=max_document_bytes,
                    expected_row_count=expected_row_count,
                    total_fetch_unit_count=total_fetch_unit_count,
                )
            else:
                error_class = str(payload or "ChildProcessError")
                _write_failed_source_group(
                    conn,
                    cache_path=cache_path,
                    source_group=active_fetch.source_group,
                    error_class=error_class,
                    expected_row_count=expected_row_count,
                    total_fetch_unit_count=total_fetch_unit_count,
                )
                first_error = first_error or RuntimeError(f"source batch fetch failed: {error_class}")
            active.pop(active_key, None)
            progressed = True
        if not progressed and active:
            time.sleep(0.05)
    if first_error is not None:
        raise first_error


def _write_fetch_results(
    conn: sqlite3.Connection,
    *,
    cache_path: Path,
    result: list[tuple[_FetchGroup, Mapping[int, Any]]],
    max_document_bytes: int,
    expected_row_count: int,
    total_fetch_unit_count: int,
) -> None:
    for group, rows_by_offset in result:
        _write_completed_group(
            conn,
            group=group,
            rows_by_offset=rows_by_offset,
            max_document_bytes=max_document_bytes,
        )
        _chmod_private_cache_files(cache_path)
        failed_count = _failed_fetch_unit_count(conn)
        _write_progress(
            cache_path,
            status="blocked" if failed_count else "building",
            cached_row_count=_cached_row_count(conn),
            expected_row_count=expected_row_count,
            completed_fetch_unit_count=_completed_fetch_unit_count(conn),
            total_fetch_unit_count=total_fetch_unit_count,
            failed_fetch_unit_count=failed_count,
            timeout_fetch_unit_count=_timeout_fetch_unit_count(conn),
            last_error_class=_last_error_class(conn),
        )


def _write_failed_group(conn: sqlite3.Connection, *, group: _FetchGroup, error: BaseException) -> None:
    _write_failed_group_error_class(conn, group=group, error_class=type(error).__name__)


def _write_failed_group_error_class(conn: sqlite3.Connection, *, group: _FetchGroup, error_class: str) -> None:
    conn.execute(
        "insert or replace into fetch_units(source_file_id, row_group_index, offsets_hash, status, attempts, "
        "last_error_class, row_count, updated_at) values (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            group.source_file_id,
            group.row_group_index,
            group.offsets_hash,
            "failed",
            1,
            str(error_class),
            len(group.selected_rows),
            _utc_now(),
        ),
    )
    conn.commit()


def _write_failed_source_group(
    conn: sqlite3.Connection,
    *,
    cache_path: Path,
    source_group: list[_FetchGroup],
    error_class: str,
    expected_row_count: int,
    total_fetch_unit_count: int,
) -> None:
    for group in source_group:
        _write_failed_group_error_class(conn, group=group, error_class=error_class)
    _write_progress(
        cache_path,
        status="blocked",
        cached_row_count=_cached_row_count(conn),
        expected_row_count=expected_row_count,
        completed_fetch_unit_count=_completed_fetch_unit_count(conn),
        total_fetch_unit_count=total_fetch_unit_count,
        failed_fetch_unit_count=_failed_fetch_unit_count(conn),
        timeout_fetch_unit_count=_timeout_fetch_unit_count(conn),
        last_error_class=_last_error_class(conn),
    )


def _write_completed_group(
    conn: sqlite3.Connection,
    *,
    group: _FetchGroup,
    rows_by_offset: Mapping[int, Any],
    max_document_bytes: int,
) -> None:
    now = _utc_now()
    conn.execute("begin")
    try:
        conn.execute(
            "delete from rows where source_file_id = ? and row_group_index = ?",
            (group.source_file_id, group.row_group_index),
        )
        for selected in group.selected_rows:
            private_row = rows_by_offset.get(selected.offset)
            if not isinstance(private_row, Mapping):
                raise ValueError("missing cached source row")
            private_text = private_row.get("text")
            if not isinstance(private_text, str) or not private_text:
                raise ValueError("missing or empty text")
            byte_length = len(private_text.encode("utf-8", errors="surrogatepass"))
            if byte_length > int(max_document_bytes):
                raise ValueError("document text exceeds configured byte cap")
            algorithm = str(selected.candidate.get("text_hash_algorithm") or TEXT_HASH_ALGORITHM_UTF8_SHA256_V1)
            expected = _expected_text_hash(selected.candidate, private_text, algorithm)
            actual = _text_hash_for_algorithm(private_text, algorithm)
            if expected != actual:
                raise ValueError("text hash drift")
            row_payload = {
                "candidate_id": selected.candidate_id,
                "candidate_hash": _required_str(selected.candidate, "candidate_hash"),
                "split": _required_str(selected.manifest_row, "split"),
                "classifier_label": _required_str(selected.manifest_row, "classifier_label"),
                "fit_eligible": 1 if selected.manifest_row.get("fit_eligible") is not False else 0,
                "training_weight": float(selected.manifest_row.get("training_weight", 1.0)),
                "source_file_id": selected.source_file_id,
                "row_group_index": selected.row_group_index,
                "row_group_local_row_index": selected.offset,
                "text_sha256": actual,
                "text_hash_algorithm": algorithm,
                "byte_length": byte_length,
                "word_count": len(WORD_RE.findall(private_text)),
            }
            row_hash = stable_json_hash(row_payload)
            try:
                conn.execute(
                    "insert into rows(candidate_id, candidate_hash, split, classifier_label, fit_eligible, "
                    "training_weight, source_file_id, row_group_index, row_group_local_row_index, text_sha256, "
                    "text_hash_algorithm, byte_length, word_count, raw_text, row_hash) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row_payload["candidate_id"],
                        row_payload["candidate_hash"],
                        row_payload["split"],
                        row_payload["classifier_label"],
                        row_payload["fit_eligible"],
                        row_payload["training_weight"],
                        row_payload["source_file_id"],
                        row_payload["row_group_index"],
                        row_payload["row_group_local_row_index"],
                        row_payload["text_sha256"],
                        row_payload["text_hash_algorithm"],
                        row_payload["byte_length"],
                        row_payload["word_count"],
                        private_text,
                        row_hash,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                message = str(exc)
                if "source_file_id" in message:
                    raise ValueError("duplicate physical locator") from exc
                if "text_sha256" in message:
                    raise ValueError("duplicate text hash") from exc
                raise ValueError("duplicate candidate_id") from exc
        conn.execute(
            "insert or replace into fetch_units(source_file_id, row_group_index, offsets_hash, status, attempts, "
            "last_error_class, row_count, updated_at) values (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                group.source_file_id,
                group.row_group_index,
                group.offsets_hash,
                "complete",
                1,
                "",
                len(group.selected_rows),
                now,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("pragma foreign_keys=on")
    conn.execute(
        "create table if not exists metadata(key text primary key, value text not null)"
    )
    conn.execute(
        "create table if not exists rows("
        "candidate_id text primary key,"
        "candidate_hash text not null,"
        "split text not null,"
        "classifier_label text not null,"
        "fit_eligible integer not null,"
        "training_weight real not null,"
        "source_file_id text not null,"
        "row_group_index integer not null,"
        "row_group_local_row_index integer not null,"
        "text_sha256 text not null unique,"
        "text_hash_algorithm text not null,"
        "byte_length integer not null,"
        "word_count integer not null,"
        "raw_text text not null,"
        "row_hash text not null,"
        "unique(source_file_id, row_group_index, row_group_local_row_index)"
        ")"
    )
    conn.execute(
        "create table if not exists fetch_units("
        "source_file_id text not null,"
        "row_group_index integer not null,"
        "offsets_hash text not null,"
        "status text not null,"
        "attempts integer not null,"
        "last_error_class text not null,"
        "row_count integer not null,"
        "updated_at text not null,"
        "primary key(source_file_id, row_group_index, offsets_hash)"
        ")"
    )
    conn.commit()


def _write_metadata(
    conn: sqlite3.Connection,
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    selected: _SelectedRows,
    cache_cap_bytes: int,
    max_document_bytes: int,
) -> None:
    metadata = {
        "private_cache_schema_version": PRIVATE_CACHE_SCHEMA_VERSION,
        "training_data_manifest_hash": _training_manifest_hash(training_manifest),
        "candidate_manifest_hash": _candidate_manifest_hash(candidates),
        "schedule_hash": _schedule_hash(schedule),
        "expected_fit_row_count": len(selected.rows),
        "audit_only_row_count": selected.audit_only_row_count,
        "input_missing_text_hash_count": selected.missing_text_hash_count,
        "cache_cap_bytes": int(cache_cap_bytes),
        "max_document_bytes": int(max_document_bytes),
    }
    existing_metadata = {
        str(key): str(value)
        for key, value in conn.execute("select key, value from metadata")
    }
    immutable_keys = {
        "private_cache_schema_version",
        "candidate_manifest_hash",
        "schedule_hash",
        "cache_cap_bytes",
        "max_document_bytes",
    }
    for key in immutable_keys:
        encoded = json.dumps(metadata[key], sort_keys=True)
        if key in existing_metadata and existing_metadata[key] != encoded:
            raise ValueError("private text cache metadata drift")
    for key, value in metadata.items():
        encoded = json.dumps(value, sort_keys=True)
        conn.execute("insert or replace into metadata(key, value) values (?, ?)", (key, encoded))
    conn.commit()


def _reconcile_cache_for_selected_rows(
    conn: sqlite3.Connection,
    *,
    selected_rows: Sequence[_SelectedRow],
    groups: Sequence[_FetchGroup],
) -> None:
    """Prune and refresh cached metadata after a deterministic manifest repair."""

    selected_ids = {row.candidate_id for row in selected_rows}
    conn.execute("create temporary table if not exists selected_candidate_ids(candidate_id text primary key)")
    conn.execute("delete from selected_candidate_ids")
    conn.executemany(
        "insert into selected_candidate_ids(candidate_id) values (?)",
        [(candidate_id,) for candidate_id in sorted(selected_ids)],
    )
    conn.execute(
        "delete from rows where not exists ("
        "select 1 from selected_candidate_ids s where s.candidate_id = rows.candidate_id)"
    )

    for selected in selected_rows:
        cached = conn.execute(
            "select candidate_hash, source_file_id, row_group_index, row_group_local_row_index, "
            "text_sha256, text_hash_algorithm, byte_length, word_count from rows where candidate_id = ?",
            (selected.candidate_id,),
        ).fetchone()
        if cached is None:
            continue
        cached_candidate_hash = str(cached[0])
        cached_source_file_id = str(cached[1])
        cached_row_group_index = int(cached[2])
        cached_offset = int(cached[3])
        text_sha256 = str(cached[4])
        text_hash_algorithm = str(cached[5])
        byte_length = int(cached[6])
        word_count = int(cached[7])
        if cached_candidate_hash != _required_str(selected.candidate, "candidate_hash"):
            raise ValueError("private text cache candidate hash drift")
        if (
            cached_source_file_id != selected.source_file_id
            or cached_row_group_index != selected.row_group_index
            or cached_offset != selected.offset
        ):
            raise ValueError("private text cache physical locator drift")
        declared_hash = str(selected.candidate.get("text_hash_sha256") or selected.candidate.get("text_hash") or "")
        if declared_hash and declared_hash != text_sha256:
            raise ValueError("private text cache text hash drift")
        row_payload = {
            "candidate_id": selected.candidate_id,
            "candidate_hash": _required_str(selected.candidate, "candidate_hash"),
            "split": _required_str(selected.manifest_row, "split"),
            "classifier_label": _required_str(selected.manifest_row, "classifier_label"),
            "fit_eligible": 1 if selected.manifest_row.get("fit_eligible") is not False else 0,
            "training_weight": float(selected.manifest_row.get("training_weight", 1.0)),
            "source_file_id": selected.source_file_id,
            "row_group_index": selected.row_group_index,
            "row_group_local_row_index": selected.offset,
            "text_sha256": text_sha256,
            "text_hash_algorithm": text_hash_algorithm,
            "byte_length": byte_length,
            "word_count": word_count,
        }
        conn.execute(
            "update rows set split = ?, classifier_label = ?, fit_eligible = ?, training_weight = ?, row_hash = ? "
            "where candidate_id = ?",
            (
                row_payload["split"],
                row_payload["classifier_label"],
                row_payload["fit_eligible"],
                row_payload["training_weight"],
                stable_json_hash(row_payload),
                selected.candidate_id,
            ),
        )

    conn.execute(
        "create temporary table if not exists expected_fetch_units("
        "source_file_id text not null, row_group_index integer not null, offsets_hash text not null, "
        "primary key(source_file_id, row_group_index, offsets_hash))"
    )
    conn.execute("delete from expected_fetch_units")
    conn.executemany(
        "insert into expected_fetch_units(source_file_id, row_group_index, offsets_hash) values (?, ?, ?)",
        [(group.source_file_id, group.row_group_index, group.offsets_hash) for group in groups],
    )
    conn.execute(
        "delete from fetch_units where not exists ("
        "select 1 from expected_fetch_units e "
        "where e.source_file_id = fetch_units.source_file_id "
        "and e.row_group_index = fetch_units.row_group_index "
        "and e.offsets_hash = fetch_units.offsets_hash)"
    )
    now = _utc_now()
    for group in groups:
        present_rows = conn.execute(
            "select candidate_id, row_group_local_row_index from rows where source_file_id = ? and row_group_index = ?",
            (group.source_file_id, group.row_group_index),
        ).fetchall()
        present = {(str(candidate_id), int(offset)) for candidate_id, offset in present_rows}
        expected = {(selected.candidate_id, selected.offset) for selected in group.selected_rows}
        if present != expected:
            continue
        conn.execute(
            "insert or replace into fetch_units(source_file_id, row_group_index, offsets_hash, status, attempts, "
            "last_error_class, row_count, updated_at) values (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                group.source_file_id,
                group.row_group_index,
                group.offsets_hash,
                "complete",
                1,
                "",
                len(group.selected_rows),
                now,
            ),
        )
    conn.commit()


def _connect_cache(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("pragma busy_timeout=30000")
    return conn


def _fetch_unit_complete(conn: sqlite3.Connection, group: _FetchGroup) -> bool:
    row = conn.execute(
        "select status, row_count from fetch_units where source_file_id = ? and row_group_index = ? and offsets_hash = ?",
        (group.source_file_id, group.row_group_index, group.offsets_hash),
    ).fetchone()
    if row is None or str(row[0]) != "complete" or int(row[1]) != len(group.selected_rows):
        return False
    present_rows = conn.execute(
        "select candidate_id, row_group_local_row_index from rows where source_file_id = ? and row_group_index = ?",
        (group.source_file_id, group.row_group_index),
    ).fetchall()
    present = {(str(candidate_id), int(offset)) for candidate_id, offset in present_rows}
    expected = {(selected.candidate_id, selected.offset) for selected in group.selected_rows}
    return present == expected


def _fetch_unit_retry_rank(conn: sqlite3.Connection, group: _FetchGroup) -> int:
    row = conn.execute(
        "select status from fetch_units where source_file_id = ? and row_group_index = ? and offsets_hash = ?",
        (group.source_file_id, group.row_group_index, group.offsets_hash),
    ).fetchone()
    if row is not None and str(row[0]) == "failed":
        return 1
    return 0


def _cached_row_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("select count(*) from rows").fetchone()[0])


def _completed_fetch_unit_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("select count(*) from fetch_units where status = 'complete'").fetchone()[0])


def _failed_fetch_unit_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("select count(*) from fetch_units where status = 'failed'").fetchone()[0])


def _timeout_fetch_unit_count(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "select count(*) from fetch_units where status = 'failed' and last_error_class = 'TimeoutError'"
        ).fetchone()[0]
    )


def _last_error_class(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "select last_error_class from fetch_units where status = 'failed' order by updated_at desc limit 1"
    ).fetchone()
    return "" if row is None else str(row[0])


def _duplicate_count(conn: sqlite3.Connection, columns: str) -> int:
    query = f"select count(*) from (select {columns}, count(*) c from rows group by {columns} having c > 1)"
    return int(conn.execute(query).fetchone()[0])


def _write_progress(
    cache_root: Path,
    *,
    status: str,
    cached_row_count: int,
    expected_row_count: int,
    completed_fetch_unit_count: int,
    total_fetch_unit_count: int,
    failed_fetch_unit_count: int = 0,
    timeout_fetch_unit_count: int = 0,
    last_error_class: str = "",
) -> None:
    payload = {
        "artifact_kind": PROGRESS_KIND,
        "schema_version": 1,
        "status": status,
        "cached_row_count": int(cached_row_count),
        "expected_row_count": int(expected_row_count),
        "completed_fetch_unit_count": int(completed_fetch_unit_count),
        "failed_fetch_unit_count": int(failed_fetch_unit_count),
        "timeout_fetch_unit_count": int(timeout_fetch_unit_count),
        "total_fetch_unit_count": int(total_fetch_unit_count),
        "last_error_class": str(last_error_class),
        "updated_at_utc": _utc_now(),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    payload["progress_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "progress_hash"})
    validate_public_no_raw_payload(payload)
    path = cache_root / "progress.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _write_public_report(path: Path, report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _chmod_private_cache_files(cache_root: Path) -> None:
    if cache_root.exists():
        os.chmod(cache_root, 0o700)
    for name in ("cache.sqlite3", "cache.sqlite3-wal", "cache.sqlite3-shm", "cache.sqlite3-journal", "progress.json", "build.lock", ".metadata_never_index"):
        path = cache_root / name
        if path.exists():
            os.chmod(path, 0o600)


def _acquire_cache_build_lock(cache_root: Path) -> Any:
    lock_path = cache_root / "build.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    os.chmod(lock_path, 0o600)
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise RuntimeError("private text cache build already running") from exc
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(json.dumps({"pid": os.getpid(), "started_at_utc": _utc_now()}, sort_keys=True) + "\n")
    lock_file.flush()
    os.fsync(lock_file.fileno())
    return lock_file


def _release_cache_build_lock(lock_file: Any) -> None:
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def _index_candidates(candidates: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        candidate_id = _required_str(candidate, "candidate_id")
        if candidate_id in indexed:
            raise ValueError("duplicate candidate_id in candidates")
        indexed[candidate_id] = candidate
    return indexed


def _schedule_entries_by_key(schedule: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in schedule.get("entries") or []:
        key = (_required_str(entry, "source_file_id"), int(entry["row_group_index"]))
        if key in indexed:
            raise ValueError("duplicate schedule row group")
        indexed[key] = entry
    return indexed


def _schedule_entry_for_candidate(
    candidate: Mapping[str, Any],
    schedule_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
) -> Mapping[str, Any]:
    source_file_id = _required_str(candidate, "source_file_id")
    row_group_index = int(candidate["row_group_index"])
    offset = int(candidate["row_group_local_row_index"])
    entry = schedule_by_key.get((source_file_id, row_group_index))
    if entry is None:
        raise ValueError("candidate physical row is not scheduled")
    offsets = {int(value) for value in entry.get("row_group_local_offsets") or []}
    if offset not in offsets:
        raise ValueError("candidate offset is not scheduled")
    if offset < 0 or offset >= int(entry["row_group_num_rows"]):
        raise ValueError("candidate offset is out of row-group bounds")
    return entry


def _schedule_result_key(entry: Mapping[str, Any]) -> tuple[str, int]:
    return _required_str(entry, "source_file_id"), int(entry["row_group_index"])


def _validate_no_raw_input(payload: Mapping[str, Any], label: str) -> None:
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if payload.get(field) is not False:
            raise ValueError(f"{label} must set {field}=false")


def _validate_schedule_no_raw(schedule: Mapping[str, Any]) -> None:
    _validate_no_raw_input(schedule, "schedule")
    for entry in schedule.get("entries") or []:
        if entry.get("raw_text_serialized") is True or entry.get("raw_url_serialized") is True or entry.get("token_serialized") is True:
            raise ValueError("schedule entry must not serialize raw text, raw urls, or tokens")


def _expected_text_hash(candidate: Mapping[str, Any], private_text: str, algorithm: str) -> str:
    expected = candidate.get("text_hash_sha256") or candidate.get("text_hash")
    if expected:
        _require_sha256(str(expected), "text_hash")
        return str(expected)
    return _text_hash_for_algorithm(private_text, algorithm)


def _text_hash_for_algorithm(private_text: str, algorithm: str) -> str:
    if algorithm == TEXT_HASH_ALGORITHM_UTF8_SHA256_V1:
        return hashlib.sha256(private_text.encode("utf-8", errors="surrogatepass")).hexdigest()
    if algorithm == LEGACY_PRIVATE_MINING_TEXT_HASH_ALGORITHM:
        return stable_json_hash(["raw_text", private_text])
    raise ValueError("unsupported candidate text_hash_algorithm")


def _training_manifest_hash(training_manifest: Mapping[str, Any]) -> str:
    value = training_manifest.get("training_manifest_hash") or training_manifest.get("manifest_hash")
    if not value:
        value = stable_json_hash(training_manifest.get("rows") or [])
    return _require_sha256(str(value), "training_manifest_hash")


def _candidate_manifest_hash(candidates: list[Mapping[str, Any]]) -> str:
    return stable_json_hash(
        [
            {
                "candidate_id": candidate.get("candidate_id"),
                "candidate_hash": candidate.get("candidate_hash"),
                "source_file_id": candidate.get("source_file_id"),
                "row_group_index": candidate.get("row_group_index"),
                "row_group_local_row_index": candidate.get("row_group_local_row_index"),
                "text_hash": candidate.get("text_hash_sha256") or candidate.get("text_hash"),
            }
            for candidate in candidates
        ]
    )


def _schedule_hash(schedule: Mapping[str, Any]) -> str:
    value = schedule.get("schedule_hash") or schedule.get("final_refetch_schedule_hash")
    if value:
        return _require_sha256(str(value), "schedule_hash")
    return stable_json_hash(
        [
            {
                "source_file_id": entry.get("source_file_id"),
                "row_group_index": entry.get("row_group_index"),
                "row_group_local_offsets": entry.get("row_group_local_offsets"),
                "entry_hash": entry.get("entry_hash"),
            }
            for entry in schedule.get("entries") or []
        ]
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return value


def _required_str(row: Mapping[str, Any], key: str) -> str:
    if key not in row or row[key] in (None, ""):
        raise ValueError(f"missing required field {key}")
    return str(row[key])


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return type(exc).__name__ in TRANSIENT_ERROR_NAMES


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _default_forbidden_roots() -> list[Path]:
    code_root = Path(__file__).resolve().parents[2]
    project_root = code_root.parent
    return [project_root, code_root, code_root / ".git"]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_candidates(path: str | Path) -> list[Mapping[str, Any]]:
    payload = _read_json(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate manifest must contain candidates list")
    return candidates


class _HFFsspecRowProviderFactory:
    def __init__(
        self,
        *,
        hf_cache_dir: str,
        cache_delta_cap_bytes: int,
        block_size_bytes: int,
        max_cache_blocks: int,
        row_group_batch_size: int,
        http_total_timeout_seconds: float,
    ) -> None:
        self.hf_cache_dir = hf_cache_dir
        self.cache_delta_cap_bytes = int(cache_delta_cap_bytes)
        self.block_size_bytes = int(block_size_bytes)
        self.max_cache_blocks = int(max_cache_blocks)
        self.row_group_batch_size = int(row_group_batch_size)
        self.http_total_timeout_seconds = float(http_total_timeout_seconds)

    def __call__(self) -> Any:
        from twc.data.decision_log import HFFsspecParquetRowProvider

        hf_cache_dir = Path(self.hf_cache_dir)
        os.environ["HF_HOME"] = str(hf_cache_dir / "hf_home")
        os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")
        os.environ["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")
        return HFFsspecParquetRowProvider(
            cache_dir=hf_cache_dir,
            cache_delta_cap_bytes=self.cache_delta_cap_bytes,
            block_size_bytes=self.block_size_bytes,
            columns=("text",),
            max_cache_blocks=self.max_cache_blocks,
            manage_hf_cache_env=False,
            row_group_batch_size=self.row_group_batch_size,
            http_total_timeout_seconds=self.http_total_timeout_seconds,
        )


def _build_hf_row_provider_factory(args: argparse.Namespace, *, cache_root: Path) -> _HFFsspecRowProviderFactory:
    hf_cache_dir = Path(args.hf_cache_dir) if args.hf_cache_dir else cache_root / "hf_transport_cache"
    return _HFFsspecRowProviderFactory(
        hf_cache_dir=str(hf_cache_dir),
        cache_delta_cap_bytes=args.max_cache_delta_bytes,
        block_size_bytes=args.block_size_bytes,
        max_cache_blocks=args.max_cache_blocks,
        row_group_batch_size=args.row_group_batch_size,
        http_total_timeout_seconds=args.http_total_timeout_seconds,
    )


def _build_hf_row_provider(args: argparse.Namespace, *, cache_root: Path) -> Any:
    return _build_hf_row_provider_factory(args, cache_root=cache_root)()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--training-manifest", required=True)
        subparser.add_argument("--candidates", required=True)
        subparser.add_argument("--schedule", required=True)
        subparser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
        subparser.add_argument("--public-report", default=None)
        subparser.add_argument("--expected-fit-rows", type=int, default=34_992)
        subparser.add_argument("--max-rows", type=int, default=None)

    build = subparsers.add_parser("build-final35k")
    add_common(build)
    build.add_argument("--row-provider", choices=["hf-fsspec"], default="hf-fsspec")
    build.add_argument("--hf-cache-dir", default=None)
    build.add_argument("--fetch-concurrency", type=int, default=16)
    build.add_argument("--row-group-batch-size", type=int, default=64)
    build.add_argument("--block-size-bytes", type=int, default=8 * 1024 * 1024)
    build.add_argument("--max-cache-blocks", type=int, default=1024)
    build.add_argument("--max-cache-delta-bytes", type=int, default=DEFAULT_CACHE_CAP_BYTES)
    build.add_argument("--max-document-bytes", type=int, default=DEFAULT_MAX_DOCUMENT_BYTES)
    build.add_argument("--retry-attempts", type=int, default=5)
    build.add_argument("--http-total-timeout-seconds", type=float, default=120.0)
    build.add_argument("--fetch-execution", choices=["thread", "process"], default="process")
    build.add_argument("--fetch-batch-timeout-seconds", type=float, default=180.0)
    build.add_argument("--fetch-process-start-method", choices=multiprocessing.get_all_start_methods(), default=None)

    validate = subparsers.add_parser("validate-final35k")
    add_common(validate)

    audit = subparsers.add_parser("audit-duplicate-collisions")
    add_common(audit)
    audit.add_argument("--row-provider", choices=["hf-fsspec"], default="hf-fsspec")
    audit.add_argument("--hf-cache-dir", default=None)
    audit.add_argument("--fetch-concurrency", type=int, default=16)
    audit.add_argument("--row-group-batch-size", type=int, default=64)
    audit.add_argument("--block-size-bytes", type=int, default=8 * 1024 * 1024)
    audit.add_argument("--max-cache-blocks", type=int, default=1024)
    audit.add_argument("--max-cache-delta-bytes", type=int, default=DEFAULT_CACHE_CAP_BYTES)
    audit.add_argument("--max-document-bytes", type=int, default=DEFAULT_MAX_DOCUMENT_BYTES)
    audit.add_argument("--retry-attempts", type=int, default=5)
    audit.add_argument("--http-total-timeout-seconds", type=float, default=120.0)

    inspect = subparsers.add_parser("inspect-final35k")
    inspect.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))

    args = parser.parse_args(argv)
    if args.command == "inspect-final35k":
        print(json.dumps(inspect_final35k_private_text_cache(cache_root=args.cache_root), sort_keys=True))
        return 0

    training_manifest = _read_json(args.training_manifest)
    candidates = _load_candidates(args.candidates)
    schedule = _read_json(args.schedule)
    if args.command == "build-final35k":
        cache_root = prepare_private_cache_root(args.cache_root, cache_cap_bytes=args.max_cache_delta_bytes)
        provider_factory = _build_hf_row_provider_factory(args, cache_root=cache_root)
        provider = provider_factory() if args.fetch_execution == "thread" else None
        report = build_final35k_private_text_cache(
            training_manifest=training_manifest,
            candidates=candidates,
            schedule=schedule,
            row_provider=provider,
            row_provider_factory=provider_factory,
            cache_root=cache_root,
            public_report_path=args.public_report,
            expected_fit_rows=args.expected_fit_rows,
            cache_cap_bytes=args.max_cache_delta_bytes,
            fetch_concurrency=args.fetch_concurrency,
            source_group_batch_size=args.row_group_batch_size,
            max_document_bytes=args.max_document_bytes,
            retry_attempts=args.retry_attempts,
            max_rows=args.max_rows,
            fetch_execution=args.fetch_execution,
            fetch_batch_timeout_seconds=args.fetch_batch_timeout_seconds,
            fetch_process_start_method=args.fetch_process_start_method,
        )
    elif args.command == "validate-final35k":
        report = validate_final35k_private_text_cache(
            training_manifest=training_manifest,
            candidates=candidates,
            schedule=schedule,
            cache_root=args.cache_root,
            public_report_path=args.public_report,
            expected_fit_rows=args.expected_fit_rows,
            max_rows=args.max_rows,
        )
    elif args.command == "audit-duplicate-collisions":
        cache_root = prepare_private_cache_root(args.cache_root, cache_cap_bytes=args.max_cache_delta_bytes)
        provider = _build_hf_row_provider(args, cache_root=cache_root)
        report = audit_final35k_private_text_cache_duplicate_collisions(
            training_manifest=training_manifest,
            candidates=candidates,
            schedule=schedule,
            row_provider=provider,
            cache_root=cache_root,
            expected_fit_rows=args.expected_fit_rows,
            cache_cap_bytes=args.max_cache_delta_bytes,
            fetch_concurrency=args.fetch_concurrency,
            source_group_batch_size=args.row_group_batch_size,
            max_document_bytes=args.max_document_bytes,
            retry_attempts=args.retry_attempts,
            max_rows=args.max_rows,
        )
        if args.public_report:
            _write_public_report(Path(args.public_report), report)
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(args.command)
    print(json.dumps({"status": report["status"], "cached_row_count": report.get("cached_row_count"), "audited_row_count": report.get("audited_row_count"), "report_hash": report["report_hash"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
