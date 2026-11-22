"""Dependency-light R4 -> R6 artifact contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from twc.repro.hashing import stable_json_hash


@dataclass(frozen=True)
class TokenizedCleanRun:
    """One exact private token sequence; never serialized in public JSON."""

    split: str
    clean_run_id_hash: str
    source_document_id_hash: str
    token_ids: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256 hex digest") from exc
    return value


def require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def hash_without(payload: Mapping[str, Any], field: str) -> str:
    return stable_json_hash({key: value for key, value in payload.items() if key != field})
