"""Capped Hugging Face streaming helpers for corpus integration smoke tests."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlsplit

from twc.data.decision_schema import PHASE2A_DECISION_CAPABILITY
from twc.data.document_filter import evaluate_deterministic_foundation
from twc.data.normalize import normalize_text
from twc.data.risk_ontology import RiskOntology


EXPECTED_FINEWEB_EDU_FIELDS = (
    "text",
    "id",
    "dump",
    "url",
    "language",
    "language_score",
    "token_count",
    "score",
    "int_score",
)
DEFAULT_DATASET_PATH = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG_NAME = "sample-10BT"
DEFAULT_SPLIT = "train"
DEFAULT_MAX_TEXT_BYTES = 2_000_000
MAX_SMOKE_DOCS = 1_000


@dataclass(frozen=True)
class StreamRecord:
    row_index: int
    dataset_path: str
    config_name: str
    split: str
    revision: str | None
    observed_fields: tuple[str, ...]
    field_presence: dict[str, bool]
    missing_expected_fields: tuple[str, ...]
    source_id: Any
    dump: Any
    url: Any
    url_sha256: str | None
    source_domain: str | None
    language: Any
    language_score: Any
    token_count: Any
    score: Any
    int_score: Any
    text_is_string: bool
    text_is_empty: bool
    text_byte_length: int
    raw_text_sha256: str | None
    normalized_text_sha256: str | None
    phase2a_decision_status: str
    phase2a_decision_capability: str
    phase2a_hit_count: int
    phase2a_quarantine_reason_code: str | None
    smoke_record_schema_version: int = 1

    def to_report_dict(self) -> dict[str, Any]:
        return asdict(self)

    def training_guard_probe_row(self) -> dict[str, Any]:
        return {
            "decision_status": self.phase2a_decision_status,
            "decision_capability": self.phase2a_decision_capability,
            "all_required_gates_present": False,
            "freeze_id": None,
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _url_sha256(url: Any) -> str | None:
    if not isinstance(url, str) or not url:
        return None
    return _sha256_text(url)


def _source_domain(url: Any) -> str | None:
    if not isinstance(url, str) or not url:
        return None
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    return host.rstrip(".").casefold() if host else None


def _source_path(url: Any) -> str:
    if not isinstance(url, str) or not url:
        return ""
    try:
        split = urlsplit(url)
    except ValueError:
        return ""
    path = split.path or ""
    if split.query:
        path = f"{path}?{split.query}"
    return path


def _load_dataset(
    dataset_path: str,
    *,
    config_name: str,
    split: str,
    revision: str | None,
    token: str | bool | None,
    load_dataset_fn: Callable[..., Iterable[dict[str, Any]]] | None,
) -> Iterable[dict[str, Any]]:
    if load_dataset_fn is None:
        try:
            from datasets import load_dataset as imported_load_dataset
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Install data dependencies first: uv sync --all-extras --group dev"
            ) from exc
        load_dataset_fn = imported_load_dataset

    kwargs: dict[str, Any] = {
        "name": config_name,
        "split": split,
        "streaming": True,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if token is not None:
        kwargs["token"] = token

    return load_dataset_fn(dataset_path, **kwargs)


def collect_stream_records(
    *,
    dataset_path: str = DEFAULT_DATASET_PATH,
    config_name: str = DEFAULT_CONFIG_NAME,
    split: str = DEFAULT_SPLIT,
    max_docs: int,
    ontology: RiskOntology,
    revision: str | None = None,
    token: str | bool | None = None,
    max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES,
    load_dataset_fn: Callable[..., Iterable[dict[str, Any]]] | None = None,
) -> Iterator[StreamRecord]:
    """Stream at most ``max_docs`` rows and emit no raw document text."""

    if max_docs < 1:
        raise ValueError("max_docs must be at least 1")
    if max_docs > MAX_SMOKE_DOCS:
        raise ValueError(f"max_docs exceeds smoke hard cap: {MAX_SMOKE_DOCS}")
    if max_text_bytes < 1:
        raise ValueError("max_text_bytes must be at least 1")

    dataset = _load_dataset(
        dataset_path,
        config_name=config_name,
        split=split,
        revision=revision,
        token=token,
        load_dataset_fn=load_dataset_fn,
    )
    iterator = iter(dataset)

    try:
        for row_index in range(max_docs):
            try:
                row = next(iterator)
            except StopIteration:
                break
            if not isinstance(row, dict):
                row = {"text": None, "_non_dict_row_type": type(row).__name__}
            yield _record_from_row(
                row,
                row_index=row_index,
                dataset_path=dataset_path,
                config_name=config_name,
                split=split,
                revision=revision,
                ontology=ontology,
                max_text_bytes=max_text_bytes,
            )
    finally:
        _close_streaming_resource(iterator)
        if dataset is not iterator:
            _close_streaming_resource(dataset)


def _close_streaming_resource(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def _record_from_row(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset_path: str,
    config_name: str,
    split: str,
    revision: str | None,
    ontology: RiskOntology,
    max_text_bytes: int,
) -> StreamRecord:
    observed_fields = tuple(sorted(str(key) for key in row.keys()))
    field_presence = {
        field: row.get(field) is not None and row.get(field) != ""
        for field in EXPECTED_FINEWEB_EDU_FIELDS
    }
    missing_expected_fields = tuple(
        field for field in EXPECTED_FINEWEB_EDU_FIELDS if field not in row
    )

    text_value = row.get("text")
    text_is_string = isinstance(text_value, str)
    text_is_empty = text_value == "" if text_is_string else False
    text_byte_length = (
        len(text_value.encode("utf-8", errors="surrogatepass")) if text_is_string else 0
    )
    raw_text_sha256 = _sha256_text(text_value) if text_is_string else None
    normalized_text_sha256 = None
    status = "quarantined"
    hit_count = 0
    quarantine_reason = None

    if not text_is_string:
        quarantine_reason = "non_string_text"
    elif text_is_empty:
        quarantine_reason = "empty_text"
        normalized_text_sha256 = normalize_text("").canonical_sha256
    elif text_byte_length > max_text_bytes:
        quarantine_reason = "text_exceeds_smoke_max_bytes"
    else:
        normalized = normalize_text(text_value)
        normalized_text_sha256 = normalized.canonical_sha256
        if not normalized.canonical_text:
            return StreamRecord(
                row_index=row_index,
                dataset_path=dataset_path,
                config_name=config_name,
                split=split,
                revision=revision,
                observed_fields=observed_fields,
                field_presence=field_presence,
                missing_expected_fields=missing_expected_fields,
                source_id=row.get("id"),
                dump=row.get("dump"),
                url=row.get("url"),
                url_sha256=_url_sha256(row.get("url")),
                source_domain=_source_domain(row.get("url")),
                language=row.get("language"),
                language_score=row.get("language_score"),
                token_count=row.get("token_count"),
                score=row.get("score"),
                int_score=row.get("int_score"),
                text_is_string=text_is_string,
                text_is_empty=text_is_empty,
                text_byte_length=text_byte_length,
                raw_text_sha256=raw_text_sha256,
                normalized_text_sha256=normalized_text_sha256,
                phase2a_decision_status="quarantined",
                phase2a_decision_capability=PHASE2A_DECISION_CAPABILITY,
                phase2a_hit_count=0,
                phase2a_quarantine_reason_code="empty_after_normalization",
            )
        metadata = {
            "url": row.get("url") or "",
            "title": "",
            "path": _source_path(row.get("url")),
            "domain": _source_domain(row.get("url")) or "",
        }
        decision = evaluate_deterministic_foundation(
            text_value,
            metadata=metadata,
            ontology=ontology,
        )
        status = decision.decision_status
        hit_count = len(decision.hits)
        quarantine_reason = decision.quarantine_reason_code
        normalized_text_sha256 = decision.normalized_text_sha256

    return StreamRecord(
        row_index=row_index,
        dataset_path=dataset_path,
        config_name=config_name,
        split=split,
        revision=revision,
        observed_fields=observed_fields,
        field_presence=field_presence,
        missing_expected_fields=missing_expected_fields,
        source_id=row.get("id"),
        dump=row.get("dump"),
        url=row.get("url"),
        url_sha256=_url_sha256(row.get("url")),
        source_domain=_source_domain(row.get("url")),
        language=row.get("language"),
        language_score=row.get("language_score"),
        token_count=row.get("token_count"),
        score=row.get("score"),
        int_score=row.get("int_score"),
        text_is_string=text_is_string,
        text_is_empty=text_is_empty,
        text_byte_length=text_byte_length,
        raw_text_sha256=raw_text_sha256,
        normalized_text_sha256=normalized_text_sha256,
        phase2a_decision_status=status,
        phase2a_decision_capability=PHASE2A_DECISION_CAPABILITY,
        phase2a_hit_count=hit_count,
        phase2a_quarantine_reason_code=quarantine_reason,
    )


def build_smoke_report(
    records: Iterable[StreamRecord],
    *,
    dataset_path: str,
    config_name: str,
    split: str,
    elapsed_seconds: float,
    auth_state: dict[str, Any],
    revision: str | None = None,
    max_docs: int | None = None,
) -> dict[str, Any]:
    record_list = list(records)
    status_counts = Counter(record.phase2a_decision_status for record in record_list)
    observed_fields = sorted(
        {field for record in record_list for field in record.observed_fields}
    )
    field_presence_counts = {
        field: sum(1 for record in record_list if record.field_presence.get(field))
        for field in EXPECTED_FINEWEB_EDU_FIELDS
    }
    missing_expected_counts = {
        field: sum(1 for record in record_list if not record.field_presence.get(field))
        for field in EXPECTED_FINEWEB_EDU_FIELDS
    }

    return {
        "schema_version": 1,
        "generated_at_utc": _utc_now_iso(),
        "dataset_path": dataset_path,
        "config_name": config_name,
        "split": split,
        "revision": revision,
        "max_docs": max_docs,
        "records_seen": len(record_list),
        "first_row_index": record_list[0].row_index if record_list else None,
        "last_row_index": record_list[-1].row_index if record_list else None,
        "expected_fields": list(EXPECTED_FINEWEB_EDU_FIELDS),
        "observed_fields": observed_fields,
        "field_presence_counts": field_presence_counts,
        "missing_expected_field_counts": missing_expected_counts,
        "phase2a_status_counts": dict(sorted(status_counts.items())),
        "raw_text_serialized": False,
        "token_serialized": False,
        "traversal_mode": "contiguous_prefix_smoke_only",
        "auth_state": auth_state,
        "datasets_version": _package_version("datasets"),
        "huggingface_hub_version": _package_version("huggingface_hub"),
        "elapsed_seconds": elapsed_seconds,
        "hash_samples": [
            {
                "row_index": record.row_index,
                "source_id": record.source_id,
                "dump": record.dump,
                "url": record.url,
                "url_sha256": record.url_sha256,
                "raw_text_sha256": record.raw_text_sha256,
                "normalized_text_sha256": record.normalized_text_sha256,
                "phase2a_decision_status": record.phase2a_decision_status,
            }
            for record in record_list[:5]
        ],
        "records": [record.to_report_dict() for record in record_list],
    }


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
