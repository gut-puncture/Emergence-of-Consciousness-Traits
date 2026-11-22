"""Canonical dependency-light hashing for TWC JSON contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    """Hash file bytes with bounded memory; propagate filesystem errors."""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("hash block size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    """Return the historical TWC SHA-256 over canonical JSON bytes."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def self_hash_mismatches(payload: Mapping[str, Any], hash_field: str) -> bool:
    return payload.get(hash_field) != stable_json_hash(
        {key: value for key, value in payload.items() if key != hash_field}
    )
