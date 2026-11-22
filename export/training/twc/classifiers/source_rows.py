"""Shared source-row lookup rules for classifier readers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def required_str(row: Mapping[str, Any], key: str) -> str:
    if key not in row or row[key] in (None, ""):
        raise ValueError(f"missing required field {key}")
    return str(row[key])


def index_candidates(candidates: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        candidate_id = required_str(candidate, "candidate_id")
        if candidate_id in indexed:
            raise ValueError(f"duplicate candidate_id in candidates: {candidate_id}")
        indexed[candidate_id] = candidate
    return indexed


def schedule_entries_by_key(schedule: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in schedule.get("entries") or []:
        source_file_id = required_str(entry, "source_file_id")
        row_group_index = int(entry["row_group_index"])
        key = (source_file_id, row_group_index)
        if key in indexed:
            raise ValueError("schedule contains duplicate source_file_id/row_group_index entries")
        indexed[key] = entry
    return indexed


def schedule_entry_for_candidate(
    candidate: Mapping[str, Any],
    schedule_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
) -> Mapping[str, Any]:
    source_file_id = required_str(candidate, "source_file_id")
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
