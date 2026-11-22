"""Primitive value validation and coercion shared by artifact contracts."""

from __future__ import annotations

import re
from typing import Any


_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWERCASE_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def require_lowercase_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _LOWERCASE_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def require_lowercase_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _LOWERCASE_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def coerce_lowercase_sha256(value: Any, field_name: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(
        char not in "0123456789abcdef" for char in candidate
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return candidate


def coerce_nonnegative_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return result


def coerce_positive_int(value: Any, field_name: str) -> int:
    result = coerce_nonnegative_int(value, field_name)
    if result < 1:
        raise ValueError(f"{field_name} must be positive")
    return result
