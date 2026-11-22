"""No-raw validation for public pretraining state."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


RAW_FIELD_NAMES = {
    "completion",
    "document_text",
    "input_ids",
    "labels",
    "matched_text",
    "normalized_text",
    "prompt",
    "raw_context_snippet",
    "raw_text",
    "raw_url",
    "source_url",
    "snippet",
    "text",
    "token_ids",
    "tokens",
    "url",
}
PRIVATE_SOURCE_LOCATOR_FIELD_NAMES = {
    "local_source_path",
    "parquet_url_or_path",
    "private_locator",
    "private_source_locator",
    "raw_source_locator",
    "source_file_path",
    "source_locator",
    "source_path",
}

ALLOWED_NO_RAW_FLAG_NAMES = {
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
}
ALLOWED_NO_RAW_METADATA_NAMES = {
    "floor_300m_train_tokens",
    "min_train_tokens",
    "min_validation_tokens",
    "target_300m_train_tokens",
    "target_train_tokens",
    "target_validation_tokens",
    "clean_run_token_count",
    "loss_token_count",
    "dropped_tail_token_count",
    "packed_example_count",
    "train_example_count",
    "validation_example_count",
    "train_token_count",
    "validation_token_count",
    "warmup_tokens",
    "stable_tokens",
    "decay_tokens",
    "tokens_per_second",
    "tokens_seen",
}
RAW_URL_VALUE_RE = re.compile(r"(?i)\b(?:https?://|ftp://|s3://|hf://|www\.)\S+")
SECRET_VALUE_RE = re.compile(r"(?i)\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|api[_-]?key\s*[:=])")


def validate_no_raw_public_payload(payload: Any, *, path: str = "$") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            key_lower = key_text.casefold()
            if key_lower in ALLOWED_NO_RAW_FLAG_NAMES:
                if value is not False:
                    raise ValueError(f"public no-raw flag must be false: {path}.{key_text}")
            elif key_lower not in ALLOWED_NO_RAW_METADATA_NAMES and (
                key_lower in RAW_FIELD_NAMES
                or key_lower in PRIVATE_SOURCE_LOCATOR_FIELD_NAMES
                or _looks_like_raw_key(key_lower)
            ):
                raise ValueError(f"raw field not allowed in public payload: {path}.{key_text}")
            validate_no_raw_public_payload(value, path=f"{path}.{key_text}")
        return
    if isinstance(payload, str):
        if RAW_URL_VALUE_RE.search(payload):
            raise ValueError(f"raw-url-like value not allowed in public payload: {path}")
        if SECRET_VALUE_RE.search(payload):
            raise ValueError(f"secret-like value not allowed in public payload: {path}")
        return
    if isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray):
        for index, item in enumerate(payload):
            validate_no_raw_public_payload(item, path=f"{path}[{index}]")


def _looks_like_raw_key(key: str) -> bool:
    lowered = key.casefold()
    return (
        "raw_text" in lowered
        or "raw_url" in lowered
        or "matched_text" in lowered
        or "context_snippet" in lowered
        or "token_ids" in lowered
        or lowered == "token_id"
        or lowered.endswith("_token_id")
        or lowered.endswith("_tokens")
        or lowered.endswith("_input_ids")
    )
