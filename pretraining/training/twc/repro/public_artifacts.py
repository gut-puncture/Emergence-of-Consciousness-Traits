"""No-raw validation for public classifier and filtering artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


SERIALIZATION_FALSE_FLAGS = {
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "accepted_manifest",
    "labels_are_corpus_approval",
    "training_shards_created",
    "classifier_training_performed",
}
FORBIDDEN_PUBLIC_KEYS = {
    "document_text",
    "matched_text",
    "messages",
    "model_output",
    "normalized_text",
    "parquet_url_or_path",
    "prompt",
    "prompt_text",
    "raw_context",
    "raw_context_snippet",
    "raw_response",
    "local_source_path",
    "baseline_run_dir",
    "comparison_run_dir",
    "run_dir",
    "raw_text",
    "raw_url",
    "request_body",
    "response_body",
    "rationale",
    "rationale_short",
    "risk_spans",
    "source_doc_id",
    "source_domain",
    "source_locator",
    "private_locator",
    "private_source_locator",
    "raw_source_locator",
    "source_file_path",
    "source_path",
    "source_url",
    "snippet",
    "text",
}
URL_RE = re.compile(r"https?://", re.IGNORECASE)
RAW_URL_VALUE_RE = re.compile(r"(?i)\b(?:https?://|ftp://|s3://|hf://|www\.)\S+")
SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|api[_-]?key\s*[:=])"
)


def validate_public_no_raw_payload(payload: Any, *, path: str = "$") -> None:
    """Fail if a public classifier/filtering payload contains raw/private fields."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            key_lower = key_text.casefold()
            child_path = f"{path}.{key_text}"
            if key_lower in SERIALIZATION_FALSE_FLAGS:
                if value is not False:
                    raise ValueError(f"{child_path} must be false")
                continue
            if key_lower in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(
                    f"public payload contains forbidden key {key_text} at {child_path}"
                )
            if "url" in key_lower:
                raise ValueError(
                    f"public payload contains url-like key {key_text} at {child_path}"
                )
            if _looks_like_token_array_key(key_lower):
                raise ValueError(
                    f"public payload contains token-array key {key_text} at {child_path}"
                )
            validate_public_no_raw_payload(value, path=child_path)
    elif isinstance(payload, list | tuple):
        for index, item in enumerate(payload):
            validate_public_no_raw_payload(item, path=f"{path}[{index}]")
    elif isinstance(payload, str):
        if RAW_URL_VALUE_RE.search(payload):
            raise ValueError(f"public payload contains raw url-like string at {path}")
        if SECRET_VALUE_RE.search(payload):
            raise ValueError(f"public payload contains secret-like string at {path}")


def _looks_like_token_array_key(key: str) -> bool:
    return (
        key in {"input_ids", "labels", "token_id", "token_ids", "tokens"}
        or key.endswith("_input_ids")
        or key.endswith("_token_ids")
    )
