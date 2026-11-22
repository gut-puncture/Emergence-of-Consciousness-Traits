"""Deterministic wide-coverage source schedules over validated Parquet metadata."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from math import gcd
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import yaml

from twc.repro.hashing import stable_json_hash
from twc.data.hf_source_metadata import validate_source_metadata_manifest
from twc.data.hf_source_metadata import write_live_metadata_manifest
from twc.repro.filesystem import directory_size as _directory_size


SOURCE_SCHEDULE_SCHEMA_VERSION = 1
SCHEDULE_ALGORITHM_VERSION = "wide_file_rowgroup_stride_v1"
SINGLE_FILE_ALL_ROWS_ALGORITHM_VERSION = "single_file_all_row_groups_v1"
SINGLE_FILE_ALL_ROWS_SCHEDULE_KIND = "twc_private_single_source_file_all_row_groups_schedule_v1"
SINGLE_FILE_SOURCE_METADATA_SUBSET_KIND = "twc_private_single_source_file_metadata_subset_v1"
SINGLE_FILE_ALL_ROWS_REPORT_KIND = "twc_single_source_file_all_rows_schedule_report_v1"
DEFAULT_ROWS_PER_ENTRY = 4
DEFAULT_MAX_ROW_GROUP_COMPRESSED_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_ROW_GROUP_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
METADATA_SCHEDULE_ALLOWED_COLUMNS = (
    "url",
    "path",
    "domain",
    "language",
    "language_score",
    "token_count",
    "score",
    "int_score",
)
METADATA_SCHEDULE_FORBIDDEN_COLUMNS = {
    "text",
    "title",
    "content",
    "raw_text",
    "document_text",
    "tokens",
    "token_ids",
}
WEIGHTED_METADATA_BUCKET_WEIGHTS = (
    ("high_token_stem_education_preferred", 0.35),
    ("high_token_general_education", 0.20),
    ("non_stem_non_education_varied", 0.25),
    ("random_wide_coverage", 0.20),
)
WEIGHTED_METADATA_BUCKET_NAMES = tuple(name for name, _weight in WEIGHTED_METADATA_BUCKET_WEIGHTS)
STEM_METADATA_KEYWORDS = (
    "math",
    "mathematics",
    "physics",
    "chemistry",
    "biology",
    "science",
    "engineering",
    "computer-science",
    "computer_science",
    "statistics",
    "economics",
)
EDUCATION_METADATA_KEYWORDS = (
    "education",
    "edu",
    "lesson",
    "course",
    "textbook",
    "lecture",
    "curriculum",
    "school",
    "university",
    "college",
    "academic",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash(payload: Any) -> str:
    return stable_json_hash(payload)


def _rng(seed: int, *parts: Any) -> random.Random:
    digest = _hash([seed, *parts])
    return random.Random(int(digest[:16], 16))


def _permutation(items: list[Any], seed: int, *parts: Any) -> list[Any]:
    copied = list(items)
    _rng(seed, *parts).shuffle(copied)
    return copied


def _stride_offsets(num_rows: int, *, seed: int, entry_key: str, count: int) -> list[int]:
    if num_rows < 1:
        return []
    count = min(count, num_rows)
    rng = _rng(seed, "offsets", entry_key, num_rows, count)
    start = rng.randrange(num_rows)
    stride = rng.randrange(1, num_rows + 1)
    while gcd(stride, num_rows) != 1:
        stride = (stride % num_rows) + 1
    offsets = []
    current = start
    while len(offsets) < count:
        if current not in offsets:
            offsets.append(current)
        current = (current + stride) % num_rows
    if count > 2 and offsets == list(range(offsets[0], offsets[0] + count)):
        offsets = list(reversed(offsets))
    return offsets


def _footer_by_file_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {footer["file_id"]: footer for footer in manifest.get("parquet_footers", [])}


def _eligible_groups(
    manifest: dict[str, Any],
    *,
    max_row_group_compressed_bytes: int,
    max_row_group_total_bytes: int,
) -> tuple[dict[str, list[dict[str, Any]]], int, int, list[str]]:
    footers = _footer_by_file_id(manifest)
    groups_by_file: dict[str, list[dict[str, Any]]] = {}
    zero_count = 0
    oversized_count = 0
    warnings: list[str] = []
    for file_entry in manifest.get("files", []):
        footer = footers.get(file_entry["file_id"], {})
        groups = []
        for group in footer.get("row_groups", []):
            if int(group.get("num_rows") or 0) <= 0:
                zero_count += 1
                continue
            if int(group.get("compressed_byte_size") or 0) > max_row_group_compressed_bytes:
                oversized_count += 1
                continue
            if int(group.get("total_byte_size") or 0) > max_row_group_total_bytes:
                oversized_count += 1
                continue
            groups.append(group)
        if groups:
            groups_by_file[file_entry["file_id"]] = groups
    if oversized_count:
        warnings.append("oversized_row_group_skipped")
    return groups_by_file, zero_count, oversized_count, warnings


def _entry(
    *,
    manifest: dict[str, Any],
    file_entry: dict[str, Any],
    group: dict[str, Any],
    global_index: int,
    pass_index: int,
    seed: int,
    rows_per_entry: int,
) -> dict[str, Any]:
    entry_key = f"{file_entry['file_id']}:{group['row_group_index']}:{global_index}"
    offsets = _stride_offsets(
        int(group["num_rows"]),
        seed=seed,
        entry_key=entry_key,
        count=rows_per_entry,
    )
    recipe = {
        "mode": "explicit_offsets",
        "start": offsets[0] if offsets else None,
        "stride": None,
        "count": len(offsets),
        "modulus": int(group["num_rows"]),
    }
    base = {
        "schedule_entry_schema_version": 1,
        "global_schedule_index": global_index,
        "schedule_pass_index": pass_index,
        "dataset_repo_id": manifest["dataset"]["repo_id"],
        "config": manifest["dataset"]["config"],
        "split": manifest["dataset"]["split"],
        "resolved_repo_sha": manifest["dataset"]["resolved_repo_sha"],
        "source_file_id": file_entry["file_id"],
        "parquet_url_or_path": file_entry["parquet_url"],
        "parquet_identity": file_entry["parquet_identity"],
        "row_group_index": int(group["row_group_index"]),
        "row_group_num_rows": int(group["num_rows"]),
        "row_group_size_estimate_bytes": {
            "compressed": int(group.get("compressed_byte_size") or 0),
            "total": int(group.get("total_byte_size") or 0),
        },
        "row_selection_mode": "explicit_offsets",
        "row_group_local_offsets": offsets,
        "offset_recipe": recipe,
        "freezable_required": True,
    }
    base["entry_hash"] = _hash(base)
    return base


def build_source_schedule(
    source_metadata_manifest: dict[str, Any],
    *,
    seed: int,
    max_entries: int,
    rows_per_entry: int = DEFAULT_ROWS_PER_ENTRY,
    max_row_group_compressed_bytes: int = DEFAULT_MAX_ROW_GROUP_COMPRESSED_BYTES,
    max_row_group_total_bytes: int = DEFAULT_MAX_ROW_GROUP_TOTAL_BYTES,
) -> dict[str, Any]:
    """Build a deterministic, non-prefix schedule from validated source metadata."""

    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")
    if rows_per_entry < 1:
        raise ValueError("rows_per_entry must be at least 1")
    validate_source_metadata_manifest(source_metadata_manifest, require_freezable=False)

    groups_by_file, zero_count, oversized_count, warnings = _eligible_groups(
        source_metadata_manifest,
        max_row_group_compressed_bytes=max_row_group_compressed_bytes,
        max_row_group_total_bytes=max_row_group_total_bytes,
    )
    files = [
        file
        for file in sorted(source_metadata_manifest.get("files", []), key=lambda item: item["parquet_url"])
        if file["file_id"] in groups_by_file
    ]
    if not files:
        raise ValueError("no eligible source files with nonzero row groups")

    entries = []
    pass_index = 0
    while len(entries) < max_entries:
        file_cycle = _permutation(files, seed, "files", source_metadata_manifest["manifest_hash"], pass_index)
        for file_entry in file_cycle:
            if len(entries) >= max_entries:
                break
            group_cycle = _permutation(
                groups_by_file[file_entry["file_id"]],
                seed,
                "row_groups",
                file_entry["file_id"],
                pass_index,
            )
            group = group_cycle[0]
            entries.append(
                _entry(
                    manifest=source_metadata_manifest,
                    file_entry=file_entry,
                    group=group,
                    global_index=len(entries),
                    pass_index=pass_index,
                    seed=seed,
                    rows_per_entry=rows_per_entry,
                )
            )
        pass_index += 1

    first_cycle_file_ids = [entry["source_file_id"] for entry in entries[: len(files)]]
    total_groups = sum(len(footer.get("row_groups", [])) for footer in source_metadata_manifest.get("parquet_footers", []))
    scheduled_group_keys = {
        (entry["source_file_id"], entry["row_group_index"])
        for entry in entries
    }
    coverage_metrics = {
        "source_file_count": len(source_metadata_manifest.get("files", [])),
        "eligible_source_file_count": len(files),
        "scheduled_source_file_count": len({entry["source_file_id"] for entry in entries}),
        "scheduled_source_file_fraction": len({entry["source_file_id"] for entry in entries}) / len(files),
        "total_row_group_count": total_groups,
        "scheduled_row_group_count": len(scheduled_group_keys),
        "zero_row_group_count": zero_count,
        "oversized_row_group_count": oversized_count,
        "all_files_visited_before_file_repetition": len(set(first_cycle_file_ids)) == len(first_cycle_file_ids),
        "contiguous_prefix_traversal": False,
        "max_consecutive_same_file": _max_consecutive_same_file(entries),
    }
    validation = {
        "freezable": bool(source_metadata_manifest.get("validation", {}).get("freezable")) and oversized_count == 0,
        "diagnostic_only": not bool(source_metadata_manifest.get("validation", {}).get("freezable")) or oversized_count > 0,
        "errors": [] if source_metadata_manifest.get("validation", {}).get("freezable") else ["source_metadata_not_freezable"],
        "warnings": sorted(set(warnings)),
    }
    schedule = {
        "schema_version": SOURCE_SCHEDULE_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "algorithm_version": SCHEDULE_ALGORITHM_VERSION,
        "seed": seed,
        "source_metadata_hash": source_metadata_manifest["manifest_hash"],
        "dataset": source_metadata_manifest["dataset"],
        "max_entries": max_entries,
        "rows_per_entry": rows_per_entry,
        "max_row_group_compressed_bytes": max_row_group_compressed_bytes,
        "max_row_group_total_bytes": max_row_group_total_bytes,
        "coverage_metrics": coverage_metrics,
        "validation": validation,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "entries": entries,
    }
    stable = {key: value for key, value in schedule.items() if key != "generated_at_utc"}
    schedule["schedule_hash"] = _hash(stable)
    return schedule


def _max_consecutive_same_file(entries: list[dict[str, Any]]) -> int:
    if not entries:
        return 0
    best = 1
    current = 1
    previous = entries[0]["source_file_id"]
    for entry in entries[1:]:
        if entry["source_file_id"] == previous:
            current += 1
        else:
            current = 1
            previous = entry["source_file_id"]
        best = max(best, current)
    return best


def validate_schedule_for_visit(schedule: dict[str, Any], source_metadata_manifest: dict[str, Any]) -> None:
    if schedule.get("schema_version") != SOURCE_SCHEDULE_SCHEMA_VERSION:
        raise ValueError("unsupported source schedule schema")
    if schedule.get("raw_text_serialized") is not False or schedule.get("token_serialized") is not False:
        raise ValueError("source schedule must set raw_text_serialized=false and token_serialized=false")
    if schedule.get("raw_url_serialized") is not False:
        raise ValueError("source schedule must set raw_url_serialized=false")
    if schedule.get("source_metadata_hash") != source_metadata_manifest.get("manifest_hash"):
        raise ValueError("source metadata drift")
    footers = _footer_by_file_id(source_metadata_manifest)
    files = {file["file_id"]: file for file in source_metadata_manifest.get("files", [])}
    for entry in schedule.get("entries", []):
        file_entry = files.get(entry["source_file_id"])
        if file_entry is None:
            raise ValueError("scheduled file missing from source metadata")
        if file_entry["parquet_identity"] != entry["parquet_identity"]:
            raise ValueError("file identity drift")
        footer = footers.get(entry["source_file_id"])
        if footer is None:
            raise ValueError("row group metadata drift")
        matching = [
            group
            for group in footer.get("row_groups", [])
            if int(group["row_group_index"]) == int(entry["row_group_index"])
        ]
        if not matching:
            raise ValueError("row group metadata drift")
        group = matching[0]
        if int(group["num_rows"]) != int(entry["row_group_num_rows"]):
            raise ValueError("row group metadata drift")
        for offset in entry.get("row_group_local_offsets", []):
            if int(offset) < 0 or int(offset) >= int(group["num_rows"]):
                raise ValueError("scheduled row offset out of bounds")
    serialized = json.dumps(schedule, sort_keys=True)
    if "snippet" in serialized.casefold():
        raise ValueError("schedule must not serialize snippets")


def build_initial_resume_state(
    *,
    run_id: str,
    schedule: dict[str, Any],
    source_tree_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "parent_run_id": None,
        "schedule_hash": schedule["schedule_hash"],
        "source_metadata_hash": schedule["source_metadata_hash"],
        "source_tree_hash": source_tree_hash,
        "config_hash": config_hash,
        "next_global_schedule_index": 0,
        "completed_entry_hashes": [],
        "entry_progress": {},
    }


def validate_resume_state(state: dict[str, Any], schedule: dict[str, Any]) -> None:
    if state.get("schedule_hash") != schedule.get("schedule_hash"):
        raise ValueError("resume schedule drift")
    if state.get("source_metadata_hash") != schedule.get("source_metadata_hash"):
        raise ValueError("resume source metadata drift")


def update_resume_after_entry(
    state: dict[str, Any],
    entry: dict[str, Any],
    *,
    completed_offsets: list[int],
) -> dict[str, Any]:
    updated = deepcopy(state)
    progress = updated.setdefault("entry_progress", {})
    entry_state = progress.setdefault(entry["entry_hash"], {"completed_offsets": []})
    merged = sorted(set(entry_state["completed_offsets"]) | set(int(offset) for offset in completed_offsets))
    entry_state["completed_offsets"] = merged
    required = sorted(int(offset) for offset in entry.get("row_group_local_offsets", []))
    if merged == required and entry["entry_hash"] not in updated["completed_entry_hashes"]:
        updated["completed_entry_hashes"].append(entry["entry_hash"])
        updated["next_global_schedule_index"] = max(
            int(updated["next_global_schedule_index"]),
            int(entry["global_schedule_index"]) + 1,
        )
    return updated


def collect_schedule_row_metadata(
    source_schedule: dict[str, Any],
    *,
    columns: tuple[str, ...] = METADATA_SCHEDULE_ALLOWED_COLUMNS,
    cache_dir: str | Path,
    cache_delta_cap_bytes: int,
    block_size_bytes: int = 8 * 1024 * 1024,
    row_group_batch_size: int = 16,
    progress_callback: Any | None = None,
    progress_every_files: int = 10,
    candidate_requested_row_count: int | None = None,
    continue_on_read_error: bool = False,
    max_read_error_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested_columns = tuple(str(column) for column in columns)
    forbidden = sorted(set(requested_columns) & METADATA_SCHEDULE_FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"forbidden metadata columns requested: {forbidden}")
    if not requested_columns:
        raise ValueError("metadata columns must not be empty")
    if cache_delta_cap_bytes < 0:
        raise ValueError("cache_delta_cap_bytes must be non-negative")
    if max_read_error_count < 0:
        raise ValueError("max_read_error_count must be non-negative")
    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("fsspec and pyarrow are required for metadata-only schedule reads") from exc

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    before = _directory_size(cache_path)
    old_env = {
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
    }
    os.environ["HF_HOME"] = str(cache_path / "hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(cache_path / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
    rows: list[dict[str, Any]] = []
    opened_file_count = 0
    row_group_read_count = 0
    missing_column_count = 0
    read_error_count = 0
    read_error_classes: Counter[str] = Counter()
    read_error_source_file_id_hashes: set[str] = set()
    entries_by_source: dict[str, list[dict[str, Any]]] = {}
    for entry in source_schedule.get("entries", []):
        entries_by_source.setdefault(str(entry.get("parquet_url_or_path")), []).append(entry)
    started_at = time.monotonic()

    def emit_progress(phase: str, *, force: bool = False, extra: dict[str, Any] | None = None) -> None:
        if progress_callback is None:
            return
        every = max(1, int(progress_every_files))
        if not force and opened_file_count % every != 0:
            return
        event = {
            "phase": phase,
            "opened_file_count": int(opened_file_count),
            "source_file_total": int(len(entries_by_source)),
            "row_group_read_count": int(row_group_read_count),
            "row_metadata_count": int(len(rows)),
            "row_metadata_read_error_count": int(read_error_count),
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
            "candidate_requested_row_count": candidate_requested_row_count,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        if extra:
            event.update(extra)
        progress_callback(event)

    emit_progress("metadata_fetch_start", force=True)
    try:
        for parquet_url, entries in entries_by_source.items():
            if not parquet_url.startswith("https://"):
                raise ValueError("metadata schedule reader requires HTTPS Parquet sources")
            try:
                with fsspec.open(parquet_url, mode="rb", block_size=block_size_bytes, cache_type="blockcache") as file_like:
                    opened_file_count += 1
                    parquet_file = pq.ParquetFile(file_like)
                    available_columns = set(getattr(getattr(parquet_file, "schema", None), "names", []) or [])
                    selected_columns = [column for column in requested_columns if column in available_columns]
                    missing_column_count += len([column for column in requested_columns if column not in available_columns])
                    if not selected_columns:
                        raise ValueError("no requested metadata columns are available in scheduled Parquet file")
                    for entry in sorted(entries, key=lambda item: int(item["row_group_index"])):
                        row_group_index = int(entry["row_group_index"])
                        metadata_group = parquet_file.metadata.row_group(row_group_index)
                        if int(metadata_group.num_rows) != int(entry.get("row_group_num_rows")):
                            raise ValueError("row_group_num_rows drift during metadata schedule read")
                        table = parquet_file.read_row_group(row_group_index, columns=selected_columns)
                        row_group_rows = table.to_pylist()
                        row_group_read_count += 1
                        for raw_offset in entry.get("row_group_local_offsets", []):
                            offset = int(raw_offset)
                            if offset < 0 or offset >= len(row_group_rows):
                                raise ValueError("scheduled row offset out of bounds during metadata schedule read")
                            source_row = row_group_rows[offset]
                            payload = {
                                "source_file_id": entry.get("source_file_id"),
                                "row_group_index": row_group_index,
                                "row_group_local_row_index": offset,
                            }
                            for column in selected_columns:
                                payload[column] = source_row.get(column)
                            rows.append(payload)
                after_each = _directory_size(cache_path)
                if after_each - before > cache_delta_cap_bytes:
                    raise RuntimeError("metadata schedule cache delta exceeded configured cap")
                emit_progress("metadata_fetch")
            except Exception as exc:
                read_error_count += 1
                read_error_classes[type(exc).__name__] += 1
                read_error_source_file_id_hashes.update(
                    _hash(str(entry.get("source_file_id") or "")) for entry in entries
                )
                emit_progress(
                    "metadata_fetch_read_error",
                    force=True,
                    extra={
                        "error_class": type(exc).__name__,
                        "error_source_file_id_hashes": sorted(read_error_source_file_id_hashes)[:20],
                    },
                )
                if not continue_on_read_error or read_error_count > int(max_read_error_count):
                    raise RuntimeError("metadata schedule read error cap exceeded") from None
                continue
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    after = _directory_size(cache_path)
    emit_progress("metadata_fetch_done", force=True)
    return rows, {
        "row_metadata_count": len(rows),
        "row_metadata_columns": list(requested_columns),
        "row_metadata_selected_columns_policy": "metadata_only_no_text",
        "row_metadata_opened_file_count": opened_file_count,
        "row_metadata_row_group_read_count": row_group_read_count,
        "row_metadata_missing_requested_column_count": missing_column_count,
        "row_metadata_read_error_count": read_error_count,
        "row_metadata_read_error_classes": dict(sorted(read_error_classes.items())),
        "row_metadata_error_source_file_id_hashes": sorted(read_error_source_file_id_hashes),
        "row_metadata_cache_delta_bytes": after - before,
        "row_metadata_cache_delta_cap_bytes": cache_delta_cap_bytes,
        "row_metadata_cache_dir_sha256": _hash(str(cache_path)),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }


def build_weighted_metadata_schedule_from_rows(
    *,
    source_schedule: dict[str, Any],
    row_metadata: list[dict[str, Any]],
    target_rows: int,
    seed: int,
    strict_bucket_fill: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if target_rows < 1:
        raise ValueError("target_rows must be positive")
    rng = random.Random(int(seed))
    targets = _weighted_bucket_targets(target_rows)
    entry_by_group = _source_entries_by_group(source_schedule)
    candidates = []
    seen_physical = set()
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
        physical_key = _metadata_physical_key(entry, offset)
        if physical_key in seen_physical:
            duplicate_physical_rows_skipped += 1
            continue
        seen_physical.add(physical_key)
        token_count = _metadata_token_count(row)
        if token_count is None:
            invalid_or_missing_token_count += 1
        signal = _metadata_signal(row)
        is_stem = _signal_contains(signal, STEM_METADATA_KEYWORDS)
        is_education = _signal_contains(signal, EDUCATION_METADATA_KEYWORDS)
        candidates.append(
            {
                "entry": entry,
                "offset": offset,
                "physical_key": physical_key,
                "token_count": token_count,
                "token_sort_value": token_count if token_count is not None else -1,
                "is_stem_education": is_stem or is_education,
                "is_education_metadata": is_education,
                "metadata_signal_hash": _hash(signal),
            }
        )
    if len(candidates) < target_rows:
        raise ValueError("metadata schedule does not contain enough unique source rows")

    selected = []
    selected_keys = set()
    bucket_counts: Counter[str] = Counter()
    bucket_underfills: Counter[str] = Counter()

    def take(bucket_name: str, pool: list[dict[str, Any]], target: int) -> None:
        ordered = _metadata_diverse_order(
            [item for item in pool if item["physical_key"] not in selected_keys],
            rng=rng,
            token_desc=bucket_name.startswith("high_token"),
        )
        taken = 0
        for item in ordered:
            if taken >= target:
                break
            selected.append(item | {"bucket": bucket_name})
            selected_keys.add(item["physical_key"])
            bucket_counts[bucket_name] += 1
            taken += 1
        if taken < target:
            bucket_underfills[bucket_name] += target - taken

    high_token = [item for item in candidates if item["token_count"] is not None]
    take("high_token_general_education", [item for item in high_token if item["is_education_metadata"]], targets["high_token_general_education"])
    take("high_token_stem_education_preferred", [item for item in high_token if item["is_stem_education"]], targets["high_token_stem_education_preferred"])
    take("non_stem_non_education_varied", [item for item in candidates if not item["is_stem_education"]], targets["non_stem_non_education_varied"])
    take("random_wide_coverage", candidates, targets["random_wide_coverage"])

    if strict_bucket_fill and any(bucket_underfills.values()):
        raise ValueError(f"strict bucket fill failed: {dict(bucket_underfills)}")
    if len(selected) < target_rows:
        refill = _metadata_diverse_order(
            [item for item in candidates if item["physical_key"] not in selected_keys],
            rng=rng,
            token_desc=True,
        )
        for item in refill:
            if len(selected) >= target_rows:
                break
            bucket_name = _first_underfilled_bucket(bucket_underfills) or "random_wide_coverage"
            selected.append(item | {"bucket": bucket_name})
            selected_keys.add(item["physical_key"])
            bucket_counts[bucket_name] += 1
    if len(selected) != target_rows:
        raise ValueError("metadata schedule unique source rows exhausted before target")

    entries = _schedule_entries_from_selected(selected)
    token_values = [int(item["token_count"]) for item in selected if item["token_count"] is not None]
    schedule = {
        "schema_version": SOURCE_SCHEDULE_SCHEMA_VERSION,
        "dataset": source_schedule.get("dataset"),
        "source_metadata_hash": source_schedule.get("source_metadata_hash"),
        "entries": entries,
        "sampling_policy": {
            "mode": "twc_revised_v2_weighted_metadata_no_duplicate_physical_rows",
            "seed": int(seed),
            "target_row_count": int(target_rows),
            "selected_row_count": sum(len(entry["row_group_local_offsets"]) for entry in entries),
            "schedule_bucket_weights": dict(WEIGHTED_METADATA_BUCKET_WEIGHTS),
            "bucket_target_counts": targets,
            "bucket_counts": {name: int(bucket_counts.get(name, 0)) for name in WEIGHTED_METADATA_BUCKET_NAMES},
            "bucket_underfill_counts": {name: int(count) for name, count in sorted(bucket_underfills.items()) if count},
            "strict_bucket_fill": bool(strict_bucket_fill),
            "duplicate_physical_rows_skipped": int(duplicate_physical_rows_skipped),
            "invalid_or_missing_token_count": int(invalid_or_missing_token_count),
            "source_file_count_selected": len({item["entry"]["source_file_id"] for item in selected}),
            "row_group_count_selected": len({(item["entry"]["source_file_id"], item["entry"]["row_group_index"]) for item in selected}),
            "token_count_summary": _token_count_summary(token_values),
        },
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    schedule["schedule_hash"] = _hash({key: value for key, value in schedule.items() if key != "schedule_hash"})
    report = {
        "artifact_kind": "weighted_metadata_source_schedule_report",
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "schedule_hash": schedule["schedule_hash"],
        "source_metadata_hash": schedule["source_metadata_hash"],
        "target_row_count": int(target_rows),
        "candidate_metadata_row_count": len(row_metadata),
        "unique_candidate_row_count": len(candidates),
        "bucket_target_counts": targets,
        "bucket_counts": schedule["sampling_policy"]["bucket_counts"],
        "bucket_underfill_counts": schedule["sampling_policy"]["bucket_underfill_counts"],
        "strict_bucket_fill": bool(strict_bucket_fill),
        "duplicate_physical_rows_skipped": int(duplicate_physical_rows_skipped),
        "invalid_or_missing_token_count": int(invalid_or_missing_token_count),
        "token_count_summary": schedule["sampling_policy"]["token_count_summary"],
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
    }
    report["report_hash"] = _hash({key: value for key, value in report.items() if key != "report_hash"})
    return schedule, report


def build_weighted_metadata_from_files(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_config(args.config)
    manifest = _load_json(args.metadata)
    validate_source_metadata_manifest(manifest, require_freezable=True)
    target_rows = int(args.target_rows)
    candidate_rows = int(args.candidate_rows)
    max_candidate_rows = int(args.max_candidate_rows)
    rows_per_entry = int(args.rows_per_entry or config.get("source_schedule", {}).get("rows_per_entry", 20))
    seed = int(args.seed if args.seed is not None else config.get("source_schedule", {}).get("seed", 20260609))
    if target_rows < 1 or candidate_rows < target_rows or max_candidate_rows < candidate_rows:
        raise ValueError("candidate row settings must satisfy 1 <= target_rows <= candidate_rows <= max_candidate_rows")
    expansion_count = 0
    last_error: Exception | None = None
    current_candidate_rows = candidate_rows
    progress_every_files = int(getattr(args, "progress_every_files", 10) or 10)
    while current_candidate_rows <= max_candidate_rows:
        _emit_operator_progress(
            {
                "phase": "candidate_schedule_build",
                "candidate_requested_row_count": int(current_candidate_rows),
                "target_row_count": int(target_rows),
                "candidate_expansion_count": int(expansion_count),
            }
        )
        candidate_schedule = build_source_schedule(
            manifest,
            seed=seed,
            max_entries=_ceil_div(current_candidate_rows, rows_per_entry),
            rows_per_entry=rows_per_entry,
        )
        row_metadata, observations = collect_schedule_row_metadata(
            candidate_schedule,
            cache_dir=args.cache_dir,
            cache_delta_cap_bytes=int(args.cache_delta_cap_bytes),
            progress_callback=_emit_operator_progress,
            progress_every_files=progress_every_files,
            candidate_requested_row_count=int(current_candidate_rows),
        )
        try:
            _emit_operator_progress(
                {
                    "phase": "bucket_selection",
                    "candidate_requested_row_count": int(current_candidate_rows),
                    "row_metadata_count": int(len(row_metadata)),
                    "target_row_count": int(target_rows),
                }
            )
            schedule, report = build_weighted_metadata_schedule_from_rows(
                source_schedule=candidate_schedule,
                row_metadata=row_metadata,
                target_rows=target_rows,
                seed=seed,
                strict_bucket_fill=True,
            )
        except ValueError as exc:
            if "strict bucket fill failed" not in str(exc) or current_candidate_rows >= max_candidate_rows:
                raise
            last_error = exc
            expansion_count += 1
            _emit_operator_progress(
                {
                    "phase": "candidate_pool_expansion",
                    "candidate_requested_row_count": int(current_candidate_rows),
                    "next_candidate_requested_row_count": int(
                        min(max_candidate_rows, max(current_candidate_rows + 1, current_candidate_rows * 2))
                    ),
                    "candidate_expansion_count": int(expansion_count),
                    "error_class": type(exc).__name__,
                    "error_summary": "strict_bucket_fill_failed",
                }
            )
            current_candidate_rows = min(max_candidate_rows, max(current_candidate_rows + 1, current_candidate_rows * 2))
            continue
        report.update(
            {
                "candidate_requested_row_count": current_candidate_rows,
                "candidate_expansion_count": expansion_count,
                "metadata_observations": observations,
                "last_expansion_error": str(last_error) if last_error else None,
            }
        )
        report["report_hash"] = _hash({key: value for key, value in report.items() if key != "report_hash"})
        _write_json(schedule, args.out)
        _write_json(report, args.report_out)
        _emit_operator_progress(
            {
                "phase": "schedule_written",
                "candidate_requested_row_count": int(current_candidate_rows),
                "target_row_count": int(target_rows),
                "schedule_hash": schedule["schedule_hash"],
                "report_hash": report["report_hash"],
                "raw_text_serialized": False,
                "raw_url_serialized": False,
                "token_serialized": False,
            }
        )
        return schedule, report
    raise ValueError("metadata schedule candidate pool exhausted before strict bucket fill")


def _emit_operator_progress(event: dict[str, Any]) -> None:
    safe_event = {
        key: value
        for key, value in dict(event).items()
        if key
        not in {
            "url",
            "file_path",
            "path",
            "domain",
            "text",
            "raw_text",
            "content",
            "document_text",
            "tokens",
            "token_ids",
        }
    }
    safe_event.setdefault("raw_text_serialized", False)
    safe_event.setdefault("raw_url_serialized", False)
    safe_event.setdefault("token_serialized", False)
    print("metadata_schedule_progress " + json.dumps(safe_event, sort_keys=True), file=sys.stderr, flush=True)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def _source_entries_by_group(source_schedule: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(entry.get("source_file_id")), int(entry.get("row_group_index"))): {
            key: value for key, value in dict(entry).items() if key != "entry_hash"
        }
        for entry in source_schedule.get("entries", [])
    }


def _metadata_physical_key(entry: dict[str, Any], offset: int) -> str:
    return _hash(
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


def _metadata_token_count(row: dict[str, Any]) -> int | None:
    try:
        value = int(row.get("token_count"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _metadata_signal(row: dict[str, Any]) -> str:
    return "\n".join(
        str(row.get(field) or "").casefold().replace("_", "-")
        for field in ("url", "path", "domain", "language", "score", "int_score")
    )


def _signal_contains(signal: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.casefold().replace("_", "-") in signal for keyword in keywords)


def _weighted_bucket_targets(target_rows: int) -> dict[str, int]:
    weights = dict(WEIGHTED_METADATA_BUCKET_WEIGHTS)
    counts = {name: int(target_rows * weights[name]) for name in WEIGHTED_METADATA_BUCKET_NAMES}
    remainder = int(target_rows) - sum(counts.values())
    fractional = sorted(
        WEIGHTED_METADATA_BUCKET_NAMES,
        key=lambda name: (target_rows * weights[name]) - counts[name],
        reverse=True,
    )
    for name in fractional[:remainder]:
        counts[name] += 1
    return counts


def _metadata_diverse_order(candidates: list[dict[str, Any]], *, rng: random.Random, token_desc: bool) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        buckets.setdefault(str(item["entry"].get("source_file_id")), []).append(item)
    for values in buckets.values():
        rng.shuffle(values)
        values.sort(key=lambda item: int(item["token_sort_value"]), reverse=token_desc)
    source_ids = list(buckets)
    rng.shuffle(source_ids)
    ordered = []
    while True:
        progressed = False
        for source_id in list(source_ids):
            bucket = buckets[source_id]
            if bucket:
                ordered.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    return ordered


def _first_underfilled_bucket(bucket_underfills: Counter[str]) -> str | None:
    for name in WEIGHTED_METADATA_BUCKET_NAMES:
        if int(bucket_underfills.get(name, 0)) > 0:
            return name
    return None


def _schedule_entries_from_selected(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, tuple[dict[str, Any], list[int]]] = {}
    for item in selected:
        entry_base = dict(item["entry"])
        offset = int(item["offset"])
        group_key = _hash(
            {
                "dataset_repo_id": entry_base.get("dataset_repo_id"),
                "config": entry_base.get("config"),
                "split": entry_base.get("split"),
                "resolved_repo_sha": entry_base.get("resolved_repo_sha"),
                "source_file_id": entry_base.get("source_file_id"),
                "row_group_index": int(entry_base.get("row_group_index")),
            }
        )
        if group_key not in grouped:
            grouped[group_key] = (entry_base, [])
        grouped[group_key][1].append(offset)
    entries = []
    for index, (_group_key, (entry_base, offsets)) in enumerate(grouped.items()):
        entry = dict(entry_base)
        entry["global_schedule_index"] = index
        entry["row_selection_mode"] = "explicit_offsets"
        entry["row_group_local_offsets"] = sorted(set(int(offset) for offset in offsets))
        entry["entry_hash"] = _hash(entry)
        entries.append(entry)
    return entries


def _token_count_summary(values: list[int]) -> dict[str, int | None]:
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


def write_resume_state_atomic(state: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_from_files(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_yaml(args.config)
    schedule_config = config.get("source_schedule", {})
    manifest = _load_json(args.metadata)
    schedule = build_source_schedule(
        manifest,
        seed=int(args.seed if args.seed is not None else schedule_config.get("seed", 20260603)),
        max_entries=int(args.max_entries if args.max_entries is not None else schedule_config.get("max_entries", 1000)),
        rows_per_entry=int(
            args.rows_per_entry
            if getattr(args, "rows_per_entry", None) is not None
            else schedule_config.get("rows_per_entry", DEFAULT_ROWS_PER_ENTRY)
        ),
        max_row_group_compressed_bytes=int(
            schedule_config.get("max_row_group_compressed_bytes", DEFAULT_MAX_ROW_GROUP_COMPRESSED_BYTES)
        ),
        max_row_group_total_bytes=int(
            schedule_config.get("max_row_group_total_bytes", DEFAULT_MAX_ROW_GROUP_TOTAL_BYTES)
        ),
    )
    _write_json(schedule, args.out)
    return schedule


def validate_from_files(args: argparse.Namespace) -> None:
    schedule = _load_json(args.schedule)
    manifest = _load_json(args.metadata)
    validate_schedule_for_visit(schedule, manifest)


def inspect_schedule(args: argparse.Namespace) -> dict[str, Any]:
    schedule = _load_json(args.schedule)
    preview = {
        "schedule_hash": schedule["schedule_hash"],
        "coverage_metrics": schedule["coverage_metrics"],
        "validation": schedule["validation"],
        "entries": schedule["entries"][: args.limit],
    }
    print(json.dumps(preview, indent=2, sort_keys=True))
    return preview


def _manifest_source_metadata_hash(manifest: dict[str, Any]) -> str:
    value = manifest.get("source_metadata_hash") or manifest.get("manifest_hash")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("source metadata manifest missing source_metadata_hash")
    return value


def _eligible_single_file_candidates(
    manifest: dict[str, Any],
    *,
    min_size_bytes: int | None,
    max_size_bytes: int | None,
) -> list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    footers = _footer_by_file_id(manifest)
    candidates: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for file_entry in manifest.get("files", []):
        if not isinstance(file_entry, dict):
            continue
        source_file_id = str(file_entry.get("file_id") or "")
        footer = footers.get(source_file_id)
        if not isinstance(footer, dict):
            continue
        size = int(file_entry.get("size") or 0)
        if min_size_bytes is not None and size < int(min_size_bytes):
            continue
        if max_size_bytes is not None and size > int(max_size_bytes):
            continue
        row_groups = [
            dict(group)
            for group in footer.get("row_groups", [])
            if isinstance(group, dict) and int(group.get("num_rows") or 0) > 0
        ]
        if row_groups:
            candidates.append((dict(file_entry), dict(footer), row_groups))
    if not candidates:
        raise ValueError("no eligible source file matched the one-file selection constraints")
    return candidates


def _select_single_file_candidate(
    manifest: dict[str, Any],
    *,
    seed: int,
    source_file_id: str | None,
    min_size_bytes: int | None,
    max_size_bytes: int | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    candidates = _eligible_single_file_candidates(
        manifest,
        min_size_bytes=min_size_bytes,
        max_size_bytes=max_size_bytes,
    )
    if source_file_id:
        matches = [candidate for candidate in candidates if candidate[0].get("file_id") == source_file_id]
        if not matches:
            raise ValueError("requested source_file_id is not eligible for one-file scheduling")
        return matches[0]
    source_metadata_hash = _manifest_source_metadata_hash(manifest)
    return sorted(
        candidates,
        key=lambda candidate: _hash(
            {
                "algorithm_version": SINGLE_FILE_ALL_ROWS_ALGORITHM_VERSION,
                "seed": int(seed),
                "source_metadata_hash": source_metadata_hash,
                "source_file_id": candidate[0]["file_id"],
            }
        ),
    )[0]


def _single_file_entry(
    *,
    manifest: dict[str, Any],
    file_entry: dict[str, Any],
    group: dict[str, Any],
    global_index: int,
) -> dict[str, Any]:
    dataset = manifest.get("dataset") or {}
    num_rows = int(group.get("num_rows") or 0)
    if num_rows <= 0:
        raise ValueError("single-file schedule row group must have positive rows")
    entry = {
        "schedule_entry_schema_version": 1,
        "global_schedule_index": int(global_index),
        "schedule_pass_index": 0,
        "dataset_repo_id": dataset.get("repo_id") or file_entry.get("dataset_repo_id"),
        "config": dataset.get("config") or file_entry.get("config"),
        "split": dataset.get("split") or file_entry.get("split"),
        "resolved_repo_sha": dataset.get("resolved_repo_sha"),
        "source_file_id": file_entry["file_id"],
        "parquet_url_or_path": file_entry["parquet_url"],
        "parquet_identity": file_entry["parquet_identity"],
        "row_group_index": int(group["row_group_index"]),
        "row_group_num_rows": num_rows,
        "row_group_size_estimate_bytes": {
            "compressed": int(group.get("compressed_byte_size") or 0),
            "total": int(group.get("total_byte_size") or 0),
        },
        "row_selection_mode": "all_rows_in_row_group",
        "offset_recipe": {
            "mode": "all_rows_in_row_group",
            "count": num_rows,
            "modulus": num_rows,
        },
        "freezable_required": True,
    }
    entry["entry_hash"] = _hash(entry)
    return entry


def _single_file_source_metadata_subset(
    *,
    manifest: dict[str, Any],
    file_entry: dict[str, Any],
    footer: dict[str, Any],
) -> dict[str, Any]:
    source_metadata_hash = _manifest_source_metadata_hash(manifest)
    subset = {
        "artifact_kind": SINGLE_FILE_SOURCE_METADATA_SUBSET_KIND,
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "source_metadata_hash": source_metadata_hash,
        "parent_source_metadata_hash": source_metadata_hash,
        "parent_manifest_hash": manifest.get("manifest_hash"),
        "dataset": deepcopy(manifest.get("dataset") or {}),
        "files": [deepcopy(file_entry)],
        "parquet_footers": [deepcopy(footer)],
        "source_file_count": 1,
        "row_group_count": len(footer.get("row_groups") or []),
        "expected_physical_rows": sum(
            int(group.get("num_rows") or 0)
            for group in footer.get("row_groups", [])
            if isinstance(group, dict) and int(group.get("num_rows") or 0) > 0
        ),
        "private_source_locator_serialized": True,
        "raw_text_serialized": False,
        "raw_url_serialized": True,
        "token_serialized": False,
    }
    subset["subset_hash"] = _hash({key: value for key, value in subset.items() if key not in {"generated_at_utc", "subset_hash"}})
    return subset


def build_single_file_all_rows_schedule(
    source_metadata_manifest: dict[str, Any],
    *,
    seed: int,
    source_file_id: str | None = None,
    min_size_bytes: int | None = None,
    max_size_bytes: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build private inputs for mirroring and filtering one complete source file.

    The schedule and metadata subset contain private source locators and must not
    be committed. The report is no-raw and is safe to use for operator handoffs.
    """

    if min_size_bytes is not None and int(min_size_bytes) < 0:
        raise ValueError("min_size_bytes must be non-negative")
    if max_size_bytes is not None and int(max_size_bytes) < 0:
        raise ValueError("max_size_bytes must be non-negative")
    if min_size_bytes is not None and max_size_bytes is not None and int(min_size_bytes) > int(max_size_bytes):
        raise ValueError("min_size_bytes must not exceed max_size_bytes")
    validate_source_metadata_manifest(source_metadata_manifest, require_freezable=True)
    source_metadata_hash = _manifest_source_metadata_hash(source_metadata_manifest)
    file_entry, footer, row_groups = _select_single_file_candidate(
        source_metadata_manifest,
        seed=int(seed),
        source_file_id=source_file_id,
        min_size_bytes=min_size_bytes,
        max_size_bytes=max_size_bytes,
    )
    ordered_groups = sorted(row_groups, key=lambda group: int(group["row_group_index"]))
    entries = [
        _single_file_entry(
            manifest=source_metadata_manifest,
            file_entry=file_entry,
            group=group,
            global_index=index,
        )
        for index, group in enumerate(ordered_groups)
    ]
    expected_rows = sum(int(entry["row_group_num_rows"]) for entry in entries)
    selection_policy = {
        "mode": SINGLE_FILE_ALL_ROWS_ALGORITHM_VERSION,
        "seed_hash": _hash(str(int(seed))),
        "source_file_id_requested_hash": _hash(source_file_id) if source_file_id else None,
        "min_size_bytes": None if min_size_bytes is None else int(min_size_bytes),
        "max_size_bytes": None if max_size_bytes is None else int(max_size_bytes),
        "row_selection_mode": "all_rows_in_row_group",
    }
    schedule = {
        "artifact_kind": SINGLE_FILE_ALL_ROWS_SCHEDULE_KIND,
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "algorithm_version": SINGLE_FILE_ALL_ROWS_ALGORITHM_VERSION,
        "source_metadata_hash": source_metadata_hash,
        "dataset": deepcopy(source_metadata_manifest.get("dataset") or {}),
        "selection_policy": selection_policy,
        "source_file_count": 1,
        "row_group_count": len(entries),
        "expected_physical_rows": int(expected_rows),
        "row_selection_mode": "all_rows_in_row_group",
        "entries": entries,
        "private_source_locator_serialized": True,
        "raw_text_serialized": False,
        "raw_url_serialized": True,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    schedule["schedule_hash"] = _hash({key: value for key, value in schedule.items() if key not in {"generated_at_utc", "schedule_hash"}})
    source_subset = _single_file_source_metadata_subset(
        manifest=source_metadata_manifest,
        file_entry=file_entry,
        footer=footer,
    )
    report = {
        "artifact_kind": SINGLE_FILE_ALL_ROWS_REPORT_KIND,
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "status": "ready",
        "schedule_hash": schedule["schedule_hash"],
        "source_metadata_subset_hash": source_subset["subset_hash"],
        "source_metadata_hash": source_metadata_hash,
        "source_file_id_hash": _hash(str(file_entry["file_id"])),
        "selected_file_size_bytes": int(file_entry.get("size") or 0),
        "row_group_count": len(entries),
        "expected_physical_rows": int(expected_rows),
        "selection_policy_hash": _hash(selection_policy),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    report["report_hash"] = _hash({key: value for key, value in report.items() if key != "report_hash"})
    serialized_report = json.dumps(report, sort_keys=True)
    if "http://" in serialized_report.casefold() or "https://" in serialized_report.casefold():
        raise ValueError("single-file all-rows report contains raw URL-like content")
    return schedule, source_subset, report


def build_single_file_all_rows_from_files(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _load_json(args.metadata)
    schedule, source_subset, report = build_single_file_all_rows_schedule(
        manifest,
        seed=int(args.seed),
        source_file_id=args.source_file_id,
        min_size_bytes=args.min_size_bytes,
        max_size_bytes=args.max_size_bytes,
    )
    _write_json(schedule, args.out)
    _write_json(source_subset, args.source_metadata_out)
    _write_json(report, args.report_out)
    return schedule, source_subset, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    metadata = sub.add_parser("metadata", help="Collect live source metadata.")
    metadata.add_argument("--config", required=True)
    metadata.add_argument("--split", default="train")
    metadata.add_argument("--out", required=True)
    metadata.add_argument("--probe-footers", action="store_true")
    metadata.add_argument("--max-footer-files", type=int, default=None)
    metadata.add_argument("--cache-dir", default="reports/cache_probe/source_metadata")

    build = sub.add_parser("build", help="Build a deterministic source schedule.")
    build.add_argument("--config", required=True)
    build.add_argument("--metadata", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--seed", type=int, default=None)
    build.add_argument("--max-entries", type=int, default=None)
    build.add_argument("--rows-per-entry", type=int, default=None)

    weighted = sub.add_parser("build-weighted-metadata", help="Build revised-v2 metadata-weighted no-text schedule.")
    weighted.add_argument("--config", required=True)
    weighted.add_argument("--metadata", required=True)
    weighted.add_argument("--out", required=True)
    weighted.add_argument("--report-out", required=True)
    weighted.add_argument("--seed", type=int, default=None)
    weighted.add_argument("--target-rows", type=int, default=10_000)
    weighted.add_argument("--candidate-rows", type=int, default=50_000)
    weighted.add_argument("--max-candidate-rows", type=int, default=100_000)
    weighted.add_argument("--rows-per-entry", type=int, default=20)
    weighted.add_argument("--cache-dir", default="reports/cache_probe/weighted_metadata_schedule")
    weighted.add_argument("--cache-delta-cap-bytes", type=int, default=10 * 1024 * 1024 * 1024)
    weighted.add_argument("--progress-every-files", type=int, default=10)

    single_file = sub.add_parser(
        "build-single-file-all-rows",
        help="Build private one-file all-row-group schedule inputs for local-Parquet filtering.",
    )
    single_file.add_argument("--metadata", required=True)
    single_file.add_argument("--out", required=True)
    single_file.add_argument("--source-metadata-out", required=True)
    single_file.add_argument("--report-out", required=True)
    single_file.add_argument("--seed", type=int, required=True)
    single_file.add_argument("--source-file-id", default=None)
    single_file.add_argument("--min-size-bytes", type=int, default=None)
    single_file.add_argument("--max-size-bytes", type=int, default=None)

    validate = sub.add_parser("validate", help="Validate a schedule against source metadata.")
    validate.add_argument("--schedule", required=True)
    validate.add_argument("--metadata", required=True)

    inspect = sub.add_parser("inspect", help="Print a compact schedule preview.")
    inspect.add_argument("--schedule", required=True)
    inspect.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    if args.command == "metadata":
        manifest = write_live_metadata_manifest(args)
        print(
            json.dumps(
                {
                    "out": args.out,
                    "file_count": len(manifest["files"]),
                    "freezable": manifest["validation"]["freezable"],
                    "errors": manifest["validation"]["errors"],
                    "source_metadata_hash": manifest["source_metadata_hash"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "build":
        schedule = build_from_files(args)
        print(
            json.dumps(
                {
                    "out": args.out,
                    "schedule_hash": schedule["schedule_hash"],
                    "entry_count": len(schedule["entries"]),
                    "freezable": schedule["validation"]["freezable"],
                    "coverage_metrics": schedule["coverage_metrics"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "build-weighted-metadata":
        schedule, report = build_weighted_metadata_from_files(args)
        print(
            json.dumps(
                {
                    "out": args.out,
                    "report_out": args.report_out,
                    "schedule_hash": schedule["schedule_hash"],
                    "report_hash": report["report_hash"],
                    "selected_row_count": schedule["sampling_policy"]["selected_row_count"],
                    "bucket_counts": schedule["sampling_policy"]["bucket_counts"],
                    "candidate_expansion_count": report["candidate_expansion_count"],
                    "raw_text_serialized": report["raw_text_serialized"],
                    "raw_url_serialized": report["raw_url_serialized"],
                    "token_serialized": report["token_serialized"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "build-single-file-all-rows":
        schedule, source_subset, report = build_single_file_all_rows_from_files(args)
        print(
            json.dumps(
                {
                    "out": args.out,
                    "source_metadata_out": args.source_metadata_out,
                    "report_out": args.report_out,
                    "schedule_hash": schedule["schedule_hash"],
                    "source_metadata_subset_hash": source_subset["subset_hash"],
                    "report_hash": report["report_hash"],
                    "source_file_id_hash": report["source_file_id_hash"],
                    "selected_file_size_bytes": report["selected_file_size_bytes"],
                    "row_group_count": report["row_group_count"],
                    "expected_physical_rows": report["expected_physical_rows"],
                    "raw_text_serialized": report["raw_text_serialized"],
                    "raw_url_serialized": report["raw_url_serialized"],
                    "token_serialized": report["token_serialized"],
                },
                sort_keys=True,
            )
        )
    elif args.command == "validate":
        validate_from_files(args)
        print(json.dumps({"schedule": args.schedule, "valid": True}, sort_keys=True))
    elif args.command == "inspect":
        inspect_schedule(args)


if __name__ == "__main__":
    main()
