"""Private active-learning refetch-map validation for chunk classifier paths."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from twc.classifiers.chunk_dataset import (
    ACTIVE_LEARNING_EVIDENCE_TIER,
    ACTIVE_LEARNING_PRIVATE_CACHE_ROUTE,
)
from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.data.decision_schema import stable_json_hash


def load_active_learning_private_index(
    active_learning_private_maps: Iterable[str | Path],
    *,
    classifier_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    paths = [Path(path) for path in active_learning_private_maps]
    manifest_rows = {
        str(row["candidate_id"]): row
        for row in classifier_manifest.get("rows") or []
        if str(row.get("evidence_tier") or "") == ACTIVE_LEARNING_EVIDENCE_TIER
    }
    if not paths:
        return {}
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=lambda item: str(item)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_public_no_raw_payload(payload)
        if payload.get("artifact_kind") != "twc_active_learning_private_refetch_map":
            raise ValueError("unsupported active-learning private map kind")
        if payload.get("private_map_route") != ACTIVE_LEARNING_PRIVATE_CACHE_ROUTE:
            raise ValueError("active-learning private map route mismatch")
        if payload.get("private_map_hash") is not None and _required_sha256(
            str(payload.get("private_map_hash") or ""),
            "active_learning_private_map_hash",
        ) != stable_json_hash({key: value for key, value in payload.items() if key != "private_map_hash"}):
            raise ValueError("active-learning private map hash mismatch")
        chunks = list(payload.get("chunks") or [])
        if int(payload.get("chunk_count", -1)) != len(chunks):
            raise ValueError("active-learning private map chunk_count mismatch")
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise ValueError("active-learning private map chunk must be an object")
            candidate_id = _required_sha256(str(chunk.get("candidate_id") or ""), "candidate_id")
            manifest_row = manifest_rows.get(candidate_id)
            if manifest_row is None:
                raise ValueError("active-learning private map contains candidate outside manifest")
            if candidate_id in index:
                raise ValueError("duplicate active-learning private candidate")
            selected_vote_hashes = [str(value) for value in (chunk.get("selected_vote_row_hashes") or [])]
            if selected_vote_hashes != [str(value) for value in (manifest_row.get("selected_vote_row_hashes") or [])]:
                raise ValueError("active-learning private map selected vote hash drift")
            private_chunk = {
                "candidate_id": candidate_id,
                "candidate_schedule_row_hash": _required_sha256(
                    str(chunk.get("candidate_schedule_row_hash") or ""),
                    "candidate_schedule_row_hash",
                ),
                "selected_vote_row_hashes": selected_vote_hashes,
                "chunk_text_sha256": _required_sha256(str(chunk.get("core_text_sha256") or ""), "core_text_sha256"),
                "evidence_text_sha256": _required_sha256(
                    str(chunk.get("evidence_text_sha256") or ""),
                    "evidence_text_sha256",
                ),
                "teacher_input_text_hash_sha256": _required_sha256(
                    str(chunk.get("teacher_input_text_hash_sha256") or ""),
                    "teacher_input_text_hash_sha256",
                ),
                "parent_text_hash_sha256": str(chunk.get("parent_text_hash_sha256") or ""),
                "parent_text_hash_algorithm": str(chunk.get("parent_text_hash_algorithm") or "utf8_sha256_v1"),
                "source_file_id": str(chunk.get("source_file_id") or ""),
                "source_schedule_hash": _required_sha256(str(chunk.get("source_schedule_hash") or ""), "source_schedule_hash"),
                "schedule_entry_hash": _required_sha256(str(chunk.get("schedule_entry_hash") or ""), "schedule_entry_hash"),
                "row_group_index": _nonnegative_int(chunk.get("row_group_index"), "row_group_index"),
                "row_group_local_row_index": _nonnegative_int(chunk.get("row_group_local_row_index"), "row_group_local_row_index"),
                "raw_char_start": _nonnegative_int(chunk.get("raw_char_start"), "raw_char_start"),
                "raw_char_end": _positive_int(chunk.get("raw_char_end"), "raw_char_end"),
                "evidence_char_start": _nonnegative_int(chunk.get("evidence_char_start"), "evidence_char_start"),
                "evidence_char_end": _positive_int(chunk.get("evidence_char_end"), "evidence_char_end"),
            }
            if not private_chunk["source_file_id"]:
                raise ValueError("active-learning private map missing source_file_id")
            if private_chunk["raw_char_end"] <= private_chunk["raw_char_start"]:
                raise ValueError("active-learning private map core span is empty")
            if private_chunk["evidence_char_end"] <= private_chunk["evidence_char_start"]:
                raise ValueError("active-learning private map evidence span is empty")
            if private_chunk["candidate_schedule_row_hash"] != manifest_row.get("candidate_schedule_row_hash"):
                raise ValueError("active-learning private map schedule row hash drift")
            if private_chunk["chunk_text_sha256"] != manifest_row.get("text_hash_sha256"):
                raise ValueError("active-learning private map core hash drift")
            if private_chunk["teacher_input_text_hash_sha256"] != manifest_row.get("teacher_input_text_hash_sha256"):
                raise ValueError("active-learning private map teacher-input hash drift")
            if private_chunk["evidence_text_sha256"] != manifest_row.get("teacher_input_text_hash_sha256"):
                raise ValueError("active-learning private map evidence hash drift")
            if private_chunk["source_schedule_hash"] != manifest_row.get("source_schedule_hash"):
                raise ValueError("active-learning private map source schedule hash drift")
            if private_chunk["schedule_entry_hash"] != manifest_row.get("schedule_entry_hash"):
                raise ValueError("active-learning private map schedule entry hash drift")
            if stable_json_hash(private_chunk["source_file_id"]) != manifest_row.get("source_file_ref_hash"):
                raise ValueError("active-learning private map source file hash drift")
            if (
                private_chunk["parent_text_hash_sha256"]
                and manifest_row.get("parent_text_hash_sha256")
                and private_chunk["parent_text_hash_sha256"] != manifest_row.get("parent_text_hash_sha256")
            ):
                raise ValueError("active-learning private map parent hash drift")
            candidate_for_fetch = {
                **dict(manifest_row),
                "source_file_id": private_chunk["source_file_id"],
                "source_schedule_hash": private_chunk["source_schedule_hash"],
                "schedule_entry_hash": private_chunk["schedule_entry_hash"],
                "row_group_index": private_chunk["row_group_index"],
                "row_group_local_row_index": private_chunk["row_group_local_row_index"],
            }
            index[candidate_id] = {
                "candidate": candidate_for_fetch,
                "private_chunk": private_chunk,
                "private_map_hash": payload.get("private_map_hash"),
            }
    return index


def _required_sha256(value: str, field_name: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return candidate


def _nonnegative_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return result


def _positive_int(value: Any, field_name: str) -> int:
    result = _nonnegative_int(value, field_name)
    if result < 1:
        raise ValueError(f"{field_name} must be positive")
    return result
