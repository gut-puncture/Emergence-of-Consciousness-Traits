"""Private text refetch to redacted classifier feature shards."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.data.decision_schema import stable_json_hash


FEATURE_SHARD_KIND = "final_35k_redacted_hashed_ngram_feature_shard"
TEXT_HASH_ALGORITHM_UTF8_SHA256_V1 = "utf8_sha256_v1"
LEGACY_PRIVATE_MINING_TEXT_HASH_ALGORITHM = "legacy_private_mining_stable_json_raw_text_v1"
WORD_RE = re.compile(r"\w+", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")


def hashed_ngram_feature_counts(
    text: str,
    *,
    n_features: int = 2**18,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_ngram_range: tuple[int, int] = (3, 5),
) -> dict[str, int]:
    """Return stable hashed word/char n-gram counts without exposing n-grams."""

    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    normalized = WHITESPACE_RE.sub(" ", text.casefold()).strip()
    words = WORD_RE.findall(normalized)
    counts: Counter[str] = Counter()

    min_word, max_word = word_ngram_range
    if min_word < 1 or max_word < min_word:
        raise ValueError("invalid word_ngram_range")
    for ngram_size in range(min_word, max_word + 1):
        if len(words) < ngram_size:
            continue
        for index in range(0, len(words) - ngram_size + 1):
            ngram = " ".join(words[index : index + ngram_size])
            counts[_feature_id(f"w{ngram_size}", ngram, n_features=n_features)] += 1

    min_char, max_char = char_ngram_range
    if min_char < 1 or max_char < min_char:
        raise ValueError("invalid char_ngram_range")
    char_view = f" {normalized} "
    for ngram_size in range(min_char, max_char + 1):
        if len(char_view) < ngram_size:
            continue
        for index in range(0, len(char_view) - ngram_size + 1):
            ngram = char_view[index : index + ngram_size]
            counts[_feature_id(f"c{ngram_size}", ngram, n_features=n_features)] += 1
    return dict(sorted(counts.items()))


def build_final_text_feature_shard(
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    output_json: str | Path,
    shard_index: int,
    start_row_index: int,
    max_rows: int,
    include_audit: bool = False,
    include_feature_counts: bool = False,
    n_features: int = 2**18,
    fetch_concurrency: int = 1,
) -> dict[str, Any]:
    """Build one resumable no-raw hashed-feature shard from final labels."""

    output_path = Path(output_json)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        validate_feature_shard(existing)
        skipped = dict(existing)
        skipped["status"] = "skipped_existing_valid"
        return skipped

    collected = collect_final_text_feature_rows(
        training_manifest=training_manifest,
        candidates=candidates,
        schedule=schedule,
        row_provider=row_provider,
        start_row_index=start_row_index,
        max_rows=max_rows,
        include_audit=include_audit,
        include_feature_counts=include_feature_counts,
        n_features=n_features,
        fetch_concurrency=fetch_concurrency,
    )
    output_rows = list(collected["rows"])

    shard = {
        "artifact_kind": FEATURE_SHARD_KIND,
        "schema_version": 1,
        "status": "written",
        "shard_index": int(shard_index),
        "start_row_index": int(start_row_index),
        "requested_row_count": int(collected["requested_row_count"]),
        "written_row_count": len(output_rows),
        "skipped_audit_row_count": int(collected["skipped_audit_row_count"]),
        "fetch_group_count": int(collected["fetch_group_count"]),
        "feature_config": dict(collected["feature_config"]),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "rows": output_rows,
    }
    shard["feature_shard_hash"] = _feature_shard_hash(shard)
    validate_feature_shard(shard)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_path, shard)
    return shard


def collect_final_text_feature_rows(
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    start_row_index: int,
    max_rows: int,
    include_audit: bool = False,
    include_feature_counts: bool = False,
    n_features: int = 2**18,
    fetch_concurrency: int = 1,
) -> dict[str, Any]:
    """Refetch private text and return redacted hashed-feature rows in memory."""

    _validate_no_raw_source_manifest(training_manifest, "training_manifest")
    _validate_schedule_no_raw(schedule)
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if start_row_index < 0:
        raise ValueError("start_row_index must be non-negative")
    if fetch_concurrency < 1:
        raise ValueError("fetch_concurrency must be positive")
    if fetch_concurrency > 1 and bool(getattr(row_provider, "manage_hf_cache_env", False)):
        raise ValueError("parallel feature collection requires row_provider.manage_hf_cache_env=false")

    rows = list(training_manifest.get("rows") or [])
    requested_rows = rows[start_row_index : start_row_index + max_rows]
    candidate_by_id = _index_candidates(candidates)
    schedule_by_key = _schedule_entries_by_key(schedule)

    selected: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    skipped_audit = 0
    for manifest_row in requested_rows:
        if manifest_row.get("fit_eligible") is False and include_audit is not True:
            skipped_audit += 1
            continue
        candidate_id = _required_str(manifest_row, "candidate_id")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"candidate missing for training manifest row: {candidate_id}")
        entry = _schedule_entry_for_candidate(candidate, schedule_by_key)
        selected.append((manifest_row, candidate, entry))

    grouped_offsets: dict[tuple[str, int], set[int]] = defaultdict(set)
    entry_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for _manifest_row, candidate, entry in selected:
        key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
        entry_by_key[key] = entry
        grouped_offsets[key].add(int(candidate["row_group_local_row_index"]))

    fetched: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    fetch_inputs = [
        (key, entry_by_key[key], sorted(grouped_offsets[key]))
        for key in sorted(grouped_offsets)
    ]
    if hasattr(row_provider, "fetch_row_groups"):
        fetch_outputs = _fetch_groups_with_batch_provider(
            row_provider,
            fetch_inputs,
            fetch_concurrency=fetch_concurrency,
        )
    elif fetch_concurrency == 1 or len(fetch_inputs) <= 1:
        fetch_outputs = [
            _fetch_group_rows(row_provider, key, entry, offsets)
            for key, entry, offsets in fetch_inputs
        ]
    else:
        fetch_outputs = []
        with ThreadPoolExecutor(max_workers=int(fetch_concurrency)) as executor:
            futures = [
                executor.submit(_fetch_group_rows, row_provider, key, entry, offsets)
                for key, entry, offsets in fetch_inputs
            ]
            for future in as_completed(futures):
                fetch_outputs.append(future.result())

    for key, offsets, rows_by_offset in fetch_outputs:
        for offset in offsets:
            row = rows_by_offset.get(offset)
            if isinstance(row, BaseException):
                raise ValueError(f"private row refetch failed: {type(row).__name__}")
            if not isinstance(row, Mapping):
                raise ValueError("private row provider returned a non-object row")
            fetched[(key[0], key[1], offset)] = row

    output_rows: list[dict[str, Any]] = []
    for manifest_row, candidate, entry in selected:
        key = (str(entry["source_file_id"]), int(entry["row_group_index"]), int(candidate["row_group_local_row_index"]))
        private_row = fetched[key]
        private_text = private_row.get("text")
        if not isinstance(private_text, str) or not private_text:
            raise ValueError("private row provider returned missing or empty text")
        text_sha256, text_hash_algorithm = _validate_or_compute_text_hash(candidate, private_text)
        features = hashed_ngram_feature_counts(private_text, n_features=n_features)
        feature_payload = {
            "feature_family": "hashed_word_char_ngrams",
            "n_features": int(n_features),
            "feature_hash_counts": features,
        }
        row_payload = {
            "row_schema_version": 1,
            "candidate_id": _required_str(manifest_row, "candidate_id"),
            "candidate_hash": _required_str(candidate, "candidate_hash"),
            "classifier_label": _required_str(manifest_row, "classifier_label"),
            "split": _required_str(manifest_row, "split"),
            "fit_eligible": bool(manifest_row.get("fit_eligible", True)),
            "training_weight": float(manifest_row.get("training_weight", 1.0)),
            "label_unit": _required_str(manifest_row, "label_unit"),
            "bucket_id_hash": stable_json_hash({"bucket_id": str(manifest_row.get("bucket_id") or "")}),
            "source_file_id_hash": stable_json_hash({"source_file_id": str(entry.get("source_file_id") or "")}),
            "row_group_index": int(entry["row_group_index"]),
            "row_group_local_row_index": int(candidate["row_group_local_row_index"]),
            "text_sha256": text_sha256,
            "text_hash_algorithm": text_hash_algorithm,
            "document_byte_length": len(private_text.encode("utf-8", errors="surrogatepass")),
            "document_word_count": len(WORD_RE.findall(private_text)),
            "nonzero_feature_count": len(features),
            "feature_vector_hash": stable_json_hash(feature_payload),
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        if include_feature_counts:
            row_payload["feature_hash_counts"] = features
        row_payload["row_hash"] = stable_json_hash(row_payload)
        validate_public_no_raw_payload(row_payload)
        output_rows.append(row_payload)

    collected = {
        "artifact_kind": "final_35k_redacted_hashed_ngram_feature_rows",
        "schema_version": 1,
        "start_row_index": int(start_row_index),
        "requested_row_count": len(requested_rows),
        "written_row_count": len(output_rows),
        "skipped_audit_row_count": skipped_audit,
        "fetch_group_count": len(grouped_offsets),
        "feature_config": {
            "family": "hashed_word_char_ngrams",
            "n_features": int(n_features),
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 5],
        },
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "rows": output_rows,
    }
    validate_public_no_raw_payload(collected)
    return collected


def _fetch_group_rows(
    row_provider: Any,
    key: tuple[str, int],
    entry: Mapping[str, Any],
    offsets: list[int],
) -> tuple[tuple[str, int], list[int], Mapping[int, Any]]:
    return key, offsets, row_provider.fetch_rows(dict(entry), offsets)


def _fetch_groups_with_batch_provider(
    row_provider: Any,
    fetch_inputs: list[tuple[tuple[str, int], Mapping[str, Any], list[int]]],
    *,
    fetch_concurrency: int,
) -> list[tuple[tuple[str, int], list[int], Mapping[int, Any]]]:
    source_groups: dict[str, list[tuple[tuple[str, int], Mapping[str, Any], list[int]]]] = defaultdict(list)
    for key, entry, offsets in fetch_inputs:
        source_groups[_entry_source_group_key(entry)].append((key, entry, offsets))
    grouped_inputs = [source_groups[source_key] for source_key in sorted(source_groups)]
    if fetch_concurrency == 1 or len(grouped_inputs) <= 1:
        outputs: list[tuple[tuple[str, int], list[int], Mapping[int, Any]]] = []
        for source_group in grouped_inputs:
            outputs.extend(_fetch_source_group_rows(row_provider, source_group))
        return outputs

    outputs = []
    with ThreadPoolExecutor(max_workers=int(fetch_concurrency)) as executor:
        futures = [
            executor.submit(_fetch_source_group_rows, row_provider, source_group)
            for source_group in grouped_inputs
        ]
        for future in as_completed(futures):
            outputs.extend(future.result())
    return outputs


def _fetch_source_group_rows(
    row_provider: Any,
    source_group: list[tuple[tuple[str, int], Mapping[str, Any], list[int]]],
) -> list[tuple[tuple[str, int], list[int], Mapping[int, Any]]]:
    batch_result = row_provider.fetch_row_groups(
        [(dict(entry), offsets) for _key, entry, offsets in source_group]
    )
    outputs: list[tuple[tuple[str, int], list[int], Mapping[int, Any]]] = []
    for key, entry, offsets in source_group:
        result_key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
        rows_by_offset = batch_result.get(result_key)
        if not isinstance(rows_by_offset, Mapping):
            raise ValueError("batch row provider did not return requested row group")
        outputs.append((key, offsets, rows_by_offset))
    return outputs


def _entry_source_group_key(entry: Mapping[str, Any]) -> str:
    source = entry.get("parquet_url_or_path") or entry.get("source_file_id")
    if not isinstance(source, str) or not source:
        raise ValueError("schedule entry is missing source identity")
    return source


def validate_feature_shard(shard: Mapping[str, Any]) -> None:
    if shard.get("artifact_kind") != FEATURE_SHARD_KIND:
        raise ValueError("unsupported feature shard kind")
    validate_public_no_raw_payload(shard)
    if shard.get("feature_shard_hash") != _feature_shard_hash(shard):
        raise ValueError("feature shard hash mismatch")
    if int(shard.get("written_row_count") or 0) != len(shard.get("rows") or []):
        raise ValueError("feature shard written_row_count mismatch")


def _feature_id(prefix: str, value: str, *, n_features: int) -> str:
    digest = hashlib.blake2b(f"{prefix}:{value}".encode("utf-8", errors="surrogatepass"), digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="big", signed=False) % int(n_features)
    return f"h{bucket:06x}"


def _validate_no_raw_source_manifest(payload: Mapping[str, Any], label: str) -> None:
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if payload.get(field) is not False:
            raise ValueError(f"{label} must set {field}=false")


def _validate_schedule_no_raw(schedule: Mapping[str, Any]) -> None:
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if schedule.get(field) is not False:
            raise ValueError(f"schedule must set {field}=false")
    for entry in schedule.get("entries") or []:
        if entry.get("raw_text_serialized") is True or entry.get("token_serialized") is True:
            raise ValueError("schedule entry must not serialize raw text or tokens")


def _index_candidates(candidates: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        candidate_id = _required_str(candidate, "candidate_id")
        if candidate_id in indexed:
            raise ValueError(f"duplicate candidate_id in candidates: {candidate_id}")
        indexed[candidate_id] = candidate
    return indexed


def _schedule_entries_by_key(schedule: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in schedule.get("entries") or []:
        source_file_id = _required_str(entry, "source_file_id")
        row_group_index = int(entry["row_group_index"])
        key = (source_file_id, row_group_index)
        if key in indexed:
            raise ValueError("schedule contains duplicate source_file_id/row_group_index entries")
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


def _validate_or_compute_text_hash(candidate: Mapping[str, Any], private_text: str) -> tuple[str, str]:
    computed = _sha256_text(private_text)
    expected = candidate.get("text_hash_sha256") or candidate.get("text_hash")
    if not expected:
        return computed, TEXT_HASH_ALGORITHM_UTF8_SHA256_V1
    algorithm = _required_str(candidate, "text_hash_algorithm")
    if algorithm == TEXT_HASH_ALGORITHM_UTF8_SHA256_V1:
        expected_computed = computed
    elif algorithm == LEGACY_PRIVATE_MINING_TEXT_HASH_ALGORITHM:
        expected_computed = stable_json_hash(["raw_text", private_text])
    else:
        raise ValueError(f"unsupported candidate text_hash_algorithm: {algorithm}")
    if str(expected) != expected_computed:
        raise ValueError("private refetch text hash drift")
    return expected_computed, algorithm


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _required_str(row: Mapping[str, Any], key: str) -> str:
    if key not in row or row[key] in (None, ""):
        raise ValueError(f"missing required field {key}")
    return str(row[key])


def _feature_shard_hash(shard: Mapping[str, Any]) -> str:
    return stable_json_hash(
        {
            key: value
            for key, value in shard.items()
            if key not in {"status", "feature_shard_hash"}
        }
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    validate_public_no_raw_payload(payload)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(serialized + "\n", encoding="utf-8")
    tmp_path.replace(path)
