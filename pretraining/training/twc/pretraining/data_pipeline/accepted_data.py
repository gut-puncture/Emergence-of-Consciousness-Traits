"""Disk-backed accepted clean-run materialization.

Selection truth comes only from a private, R4-root-bound MaterializationInput.
Caller-supplied source blocks carry content, never selection booleans.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import os
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twc.pretraining.materialization_handoff import (
    MaterializationInput,
    SelectedMaterializationFact,
    iter_validated_materialization_facts,
    validate_selected_materialization_handoff,
)
from twc.repro.hashing import stable_json_hash
from twc.repro.public_artifacts import validate_public_no_raw_payload
from twc.pretraining.chunked_clean_run import (
    BLOCK_KINDS,
    SourceDocumentRecord,
    segment_source_document,
)

from .contracts import hash_without, require_int, require_sha256

MANIFEST_KIND = "twc_accepted_clean_run_manifest_v1"
CORE_RUN_MANIFEST_KIND = "twc_accepted_candidate_core_run_manifest_v1"
CORE_EXPORT_KIND = "twc_w1_private_accepted_core_export_v1"
_CANONICAL_PART_RECEIPT_FIELDS = (
    "artifact_kind",
    "schema_version",
    "part_ordinal",
    "part_name",
    "record_count",
    "byte_count",
    "sha256",
    "split_counts",
    "record_metadata_sha256",
    "last_text_hash",
    "selection_handoff_hash",
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
)


def _validate_core_export_manifest_header(
    manifest: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    validate_public_no_raw_payload(manifest)
    if (
        manifest.get("artifact_kind") != CORE_EXPORT_KIND
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("content_mode") != "verified-core"
        or manifest.get("manifest_hash") != hash_without(manifest, "manifest_hash")
    ):
        raise ValueError("verified core export manifest is invalid")
    require_sha256(manifest.get("handoff_hash"), "core export handoff_hash")
    require_int(manifest.get("accepted_row_count"), "accepted_row_count", minimum=1)
    require_int(
        manifest.get("accepted_token_count_lower_bound"),
        "accepted_token_count_lower_bound",
    )
    require_int(manifest.get("last_global_ordinal"), "last_global_ordinal")
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts or len(parts) != manifest.get("part_count"):
        raise ValueError("verified core export part set mismatch")
    return parts


def _canonical_part_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {field: receipt[field] for field in _CANONICAL_PART_RECEIPT_FIELDS}


def _strict_nonnegative_int(value: Any, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"accepted part receipt {field} must be a {qualifier} integer")
    return value


def _validate_part_receipt_schema(
    receipt: Mapping[str, Any], *, local: bool
) -> dict[str, Any]:
    """Strictly validate local or canonical accepted-part receipt shape."""

    canonical_keys = set(_CANONICAL_PART_RECEIPT_FIELDS)
    expected_keys = canonical_keys | ({"file_identity", "receipt_hash"} if local else set())
    if set(receipt) != expected_keys:
        raise ValueError("accepted part receipt keys mismatch")
    if receipt.get("artifact_kind") != "twc_accepted_clean_run_part_receipt_v1":
        raise ValueError("accepted part receipt artifact kind mismatch")
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        raise ValueError("accepted part receipt schema version mismatch")
    ordinal = _strict_nonnegative_int(receipt.get("part_ordinal"), "part_ordinal")
    record_count = _strict_nonnegative_int(
        receipt.get("record_count"), "record_count", positive=True
    )
    byte_count = _strict_nonnegative_int(
        receipt.get("byte_count"), "byte_count", positive=True
    )
    expected_name = f"accepted_clean_runs-{ordinal:06d}.jsonl"
    if receipt.get("part_name") != expected_name:
        raise ValueError("accepted part receipt part name mismatch")
    for field in (
        "sha256",
        "record_metadata_sha256",
        "last_text_hash",
        "selection_handoff_hash",
    ):
        require_sha256(receipt.get(field), field)
    split_counts = receipt.get("split_counts")
    if not isinstance(split_counts, Mapping) or set(split_counts) != {"train", "validation"}:
        raise ValueError("accepted part receipt split counts mismatch")
    for split in ("train", "validation"):
        _strict_nonnegative_int(split_counts[split], f"split_counts.{split}")
    if sum(split_counts.values()) != record_count:
        raise ValueError("accepted part receipt split counts do not sum to record count")
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if receipt.get(field) is not False:
            raise ValueError(f"accepted part receipt {field} must be false")
    if local:
        identity = receipt.get("file_identity")
        if not isinstance(identity, Mapping) or set(identity) != {
            "device",
            "inode",
            "byte_count",
            "mtime_ns",
        }:
            raise ValueError("accepted part receipt file identity mismatch")
        for field in ("device", "inode", "byte_count", "mtime_ns"):
            _strict_nonnegative_int(identity[field], f"file_identity.{field}")
        if identity["byte_count"] != byte_count:
            raise ValueError("accepted part receipt file byte counts disagree")
        require_sha256(receipt.get("receipt_hash"), "receipt_hash")
        if receipt["receipt_hash"] != stable_json_hash(
            {key: value for key, value in receipt.items() if key != "receipt_hash"}
        ):
            raise ValueError("accepted part receipt hash mismatch")
    return _canonical_part_receipt(receipt)


@dataclass(frozen=True)
class SourceDocument:
    source_document_id_hash: str
    split: str
    raw_text: str
    source_lineage_hash: str | None = None
    source_token_count_hint: int | None = None


@dataclass(frozen=True)
class AcceptedCleanRunBuildResult:
    output_root: Path
    manifest: dict[str, Any]
    public_summary: dict[str, Any]
    private_part_paths: tuple[Path, ...]
    manifest_path: Path

    @property
    def private_part_path(self) -> Path:
        return self.private_part_paths[0]


@dataclass(frozen=True)
class SourceDocumentUnit:
    next_cursor: int
    document: SourceDocument


class SequenceSourceDocumentSchedule:
    """O(1)-seek schedule used by tests and bounded local adapters."""

    def __init__(self, documents: Sequence[SourceDocument], identity_hash: str) -> None:
        require_sha256(identity_hash, "source_schedule_identity_hash")
        self._documents = documents
        self.identity_hash = identity_hash

    def iter_from(self, cursor: int) -> Iterable[SourceDocumentUnit]:
        if not 0 <= cursor <= len(self._documents):
            raise ValueError("source schedule cursor is outside the sequence")
        for index in range(cursor, len(self._documents)):
            yield SourceDocumentUnit(index + 1, self._documents[index])


def _atomic_write(path: Path, data: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(partial, 0o600)
    os.replace(partial, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _split(component_root: str, validation_basis_points: int) -> str:
    return "validation" if int(component_root[:8], 16) % 10_000 < validation_basis_points else "train"


def _open_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    os.chmod(path, 0o600)
    connection.executescript(
        """
        pragma journal_mode=wal;
        pragma synchronous=full;
        create table if not exists selected_facts(
          candidate_id_hash text primary key,shard_number integer not null,
          source_document_id_hash text not null,
          first_block_index integer not null, last_block_index integer not null,
          core_text_sha256 text not null,core_char_count integer not null,
          core_byte_count integer not null,core_char_start integer not null,
          core_char_end integer not null,segmentation_receipt_hash text not null,
          matched integer not null default 0
        );
        create index if not exists selected_by_source on selected_facts(source_document_id_hash,first_block_index);
        create table if not exists parents(source_document_id_hash text primary key,parent_hash text not null);
        create table if not exists duplicate_first(text_hash text primary key,source_document_id_hash text not null);
        create table if not exists runs(
          run_id_hash text primary key,source_document_id_hash text not null,text_hash text not null,
          text_value text not null,utf8_byte_count integer not null,selected_block_count integer not null,
          selected_block_root text not null
        );
        create index if not exists runs_by_text on runs(text_hash,run_id_hash);
        create table if not exists build_identity(key text primary key,value text not null);
        create table if not exists build_progress(
          id integer primary key check(id=1),facts_ingested integer not null,
          next_fact_shard integer not null,next_source_cursor integer not null,
          emission_next_part integer not null,emission_last_text_hash text
        );
        insert or ignore into build_progress values (1,0,0,0,0,null);
        create trigger if not exists selected_facts_no_delete
          before delete on selected_facts begin
          select raise(abort,'selected facts are append-only'); end;
        create trigger if not exists selected_facts_no_semantic_update
          before update of candidate_id_hash,shard_number,source_document_id_hash,
            first_block_index,last_block_index,core_text_sha256,core_char_count,
            core_byte_count,core_char_start,core_char_end,segmentation_receipt_hash
          on selected_facts begin
          select raise(abort,'selected fact semantics are immutable'); end;
        create trigger if not exists selected_facts_no_insert_after_freeze
          before insert on selected_facts
          when (select facts_ingested from build_progress where id=1)=1 begin
          select raise(abort,'selected facts are frozen'); end;
        create table if not exists processed_documents(
          source_document_id_hash text primary key,raw_text_sha256 text not null,
          segmentation_receipt_hash text not null
        );
        """
    )
    return connection


def _find(connection: sqlite3.Connection, item: str) -> str:
    trail: list[str] = []
    while True:
        row = connection.execute(
            "select parent_hash from parents where source_document_id_hash=?", (item,)
        ).fetchone()
        if row is None:
            connection.execute("insert into parents values (?,?)", (item, item))
            root = item
            break
        parent = str(row[0])
        if parent == item:
            root = item
            break
        trail.append(item)
        item = parent
    for member in trail:
        connection.execute(
            "update parents set parent_hash=? where source_document_id_hash=?", (root, member)
        )
    return root


def _union(connection: sqlite3.Connection, left: str, right: str) -> None:
    left_root, right_root = _find(connection, left), _find(connection, right)
    if left_root != right_root:
        lower, upper = sorted((left_root, right_root))
        connection.execute(
            "update parents set parent_hash=? where source_document_id_hash=?", (lower, upper)
        )


def _segment_document(document: SourceDocument) -> Any:
    if not isinstance(document, SourceDocument):
        raise ValueError("source documents must be SourceDocument")
    require_sha256(document.source_document_id_hash, "source_document_id_hash")
    if not isinstance(document.raw_text, str) or not document.raw_text:
        raise ValueError("source document raw text must be non-empty")
    return segment_source_document(
        SourceDocumentRecord(
            source_document_id_hash=document.source_document_id_hash,
            split=document.split,
            raw_text=document.raw_text,
            source_lineage_hash=document.source_lineage_hash,
            source_token_count_hint=document.source_token_count_hint,
        )
    )


def _insert_run(
    connection: sqlite3.Connection,
    document: SourceDocument,
    selected_blocks: list[Any],
    *,
    text_start: int,
    text_end: int,
) -> None:
    if not selected_blocks:
        return
    text = document.raw_text[text_start:text_end]
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    run_id = stable_json_hash(
        {
            "source_document_id_hash": document.source_document_id_hash,
            "first_block_index": selected_blocks[0].block_index,
            "last_block_index": selected_blocks[-1].block_index,
            "clean_run_text_hash": text_hash,
        }
    )
    connection.execute(
        "insert into runs values (?,?,?,?,?,?,?)",
        (
            run_id,
            document.source_document_id_hash,
            text_hash,
            text,
            len(text.encode()),
            len(selected_blocks),
            stable_json_hash([block.block_id_hash for block in selected_blocks]),
        ),
    )
    prior = connection.execute(
        "select source_document_id_hash from duplicate_first where text_hash=?", (text_hash,)
    ).fetchone()
    if prior is None:
        connection.execute(
            "insert into duplicate_first values (?,?)", (text_hash, document.source_document_id_hash)
        )
    else:
        _union(connection, str(prior[0]), document.source_document_id_hash)


def _materialize_document(connection: sqlite3.Connection, document: SourceDocument) -> int:
    segmentation = _segment_document(document)
    source_hash = hashlib.sha256(document.raw_text.encode()).hexdigest()
    processed = connection.execute(
        "select raw_text_sha256,segmentation_receipt_hash from processed_documents "
        "where source_document_id_hash=?",
        (document.source_document_id_hash,),
    ).fetchone()
    if processed is not None:
        if processed != (source_hash, segmentation.receipt.receipt_hash):
            raise ValueError("resumed source document content changed")
        return 0
    _find(connection, document.source_document_id_hash)
    fact_rows = connection.execute(
        "select candidate_id_hash,first_block_index,last_block_index,core_text_sha256,"
        "core_char_count,core_byte_count,core_char_start,core_char_end,segmentation_receipt_hash "
        "from selected_facts where source_document_id_hash=? order by first_block_index",
        (document.source_document_id_hash,),
    ).fetchall()
    by_index = {block.block_index: block for block in segmentation.blocks}
    selected_indexes: set[int] = set()
    selected_groups: list[list[tuple[Any, ...]]] = []
    for row in fact_rows:
        (
            candidate_id, first, last, core_hash, char_count, byte_count,
            char_start, char_end, segmentation_receipt_hash,
        ) = row
        if segmentation_receipt_hash != segmentation.receipt.receipt_hash:
            raise ValueError("selected candidate segmentation receipt mismatch")
        indexes = list(range(int(first), int(last) + 1))
        if any(index not in by_index for index in indexes):
            raise ValueError("selected candidate references a missing natural block")
        if selected_indexes.intersection(indexes):
            raise ValueError("selected candidate block spans overlap")
        if not 0 <= int(char_start) < int(char_end) <= len(document.raw_text):
            raise ValueError("selected candidate character span is outside source text")
        candidate_text = document.raw_text[int(char_start) : int(char_end)]
        if hashlib.sha256(candidate_text.encode()).hexdigest() != core_hash:
            raise ValueError("selected candidate core hash does not match rehydrated blocks")
        if len(candidate_text) != int(char_count) or len(candidate_text.encode()) != int(byte_count):
            raise ValueError("selected candidate core length does not match rehydrated span")
        if int(char_start) != by_index[int(first)].content_char_start:
            raise ValueError("selected candidate start does not match first natural block")
        if int(char_end) != by_index[int(last)].content_char_end:
            raise ValueError("selected candidate end does not match last natural block")
        selected_indexes.update(indexes)
        connection.execute(
            "update selected_facts set matched=1 where candidate_id_hash=?", (candidate_id,)
        )
        normalized_row = tuple(row)
        block_adjacent = bool(
            selected_groups and int(first) == int(selected_groups[-1][-1][2]) + 1
        )
        if block_adjacent:
            previous = selected_groups[-1][-1]
            previous_block = by_index[int(previous[2])]
            next_block = by_index[int(first)]
            gap_start, gap_end = int(previous[7]), int(char_start)
            coverage = [
                (previous_block.content_char_end, previous_block.owned_char_end),
                (next_block.owned_char_start, next_block.content_char_start),
            ]
            coverage.extend(
                (trivia.char_start, trivia.char_end)
                for trivia in segmentation.trivia_spans
                if trivia.left_block_index == previous_block.block_index
                and trivia.right_block_index == next_block.block_index
            )
            cursor = gap_start
            for start, end in sorted((start, end) for start, end in coverage if end > start):
                if start != cursor:
                    raise ValueError("adjacent selected blocks have non-canonical separator bytes")
                cursor = end
            if cursor != gap_end:
                raise ValueError("adjacent selected blocks separator coverage mismatch")
        if not block_adjacent:
            selected_groups.append([normalized_row])
        else:
            selected_groups[-1].append(normalized_row)
    for group in selected_groups:
        first_index = int(group[0][1])
        last_index = int(group[-1][2])
        blocks = [by_index[index] for index in range(first_index, last_index + 1)]
        _insert_run(
            connection,
            document,
            blocks,
            text_start=int(group[0][6]),
            text_end=int(group[-1][7]),
        )
    connection.execute(
        "insert into processed_documents values (?,?,?)",
        (
            document.source_document_id_hash,
            source_hash,
            segmentation.receipt.receipt_hash,
        ),
    )
    return len(fact_rows)


def _ingest_verified_core_export(
    connection: sqlite3.Connection,
    manifest: Mapping[str, Any],
    root: Path,
    *,
    interrupt_after_parts: int | None = None,
) -> tuple[int, str, dict[str, Any]]:
    parts = _validate_core_export_manifest_header(manifest)
    seen: set[str] = set()
    previous = -1
    total_rows = total_tokens = 0
    part_roots: list[str] = []
    chars: Counter[str] = Counter()
    bytes_: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    bucket_names = ("<=255", "<=511", "<=1023", "<=2047", "<=4095", "4096+")
    required = {"global_ordinal", "shard_index", "shard_row_index", "candidate_id_hash",
        "private_evidence_row_id", "evidence_text_sha256", "core_text_sha256",
        "source_document_id_hash", "core_proxy_token_count", "w1_route", "text",
        "core_char_count", "core_byte_count", "core_block_indexes", "block_id_hashes", "block_kinds"}

    def bucket(value: int) -> str:
        for upper in (255, 511, 1023, 2047, 4095):
            if value <= upper:
                return f"<={upper}"
        return "4096+"

    for part_index, descriptor in enumerate(parts):
        if not isinstance(descriptor, Mapping) or descriptor.get("part_index") != part_index:
            raise ValueError("verified core export part order mismatch")
        relative = descriptor.get("relative_path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise ValueError("verified core export part path is unsafe")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("verified core export part must be a regular file")
        compressed_digest = hashlib.sha256()
        with path.open("rb") as compressed:
            while chunk := compressed.read(8 * 1024 * 1024):
                compressed_digest.update(chunk)
        if path.stat().st_size != descriptor.get("bytes") or compressed_digest.hexdigest() != descriptor.get("sha256"):
            raise ValueError("verified core export part hash mismatch")
        raw_digest = hashlib.sha256()
        part_rows = part_tokens = 0
        first: int | None = None
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                raw_digest.update(line.encode())
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != required:
                    raise ValueError("verified core export row schema mismatch")
                ordinal = require_int(row["global_ordinal"], "global_ordinal")
                require_int(row["shard_index"], "shard_index")
                require_int(row["shard_row_index"], "shard_row_index")
                if ordinal <= previous:
                    raise ValueError("verified core export rows are out of order")
                first = ordinal if first is None else first
                previous = ordinal
                candidate = require_sha256(row["candidate_id_hash"], "candidate_id_hash")
                source = require_sha256(row["source_document_id_hash"], "source_document_id_hash")
                core_hash = require_sha256(row["core_text_sha256"], "core_text_sha256")
                require_sha256(row["evidence_text_sha256"], "evidence_text_sha256")
                require_sha256(row["private_evidence_row_id"], "private_evidence_row_id")
                indexes, block_ids, block_kinds = row["core_block_indexes"], row["block_id_hashes"], row["block_kinds"]
                if (candidate in seen or row["w1_route"] != "w1_pass" or not isinstance(indexes, list)
                        or not indexes or any(type(index) is not int or index < 0 for index in indexes)
                        or indexes != list(range(indexes[0], indexes[-1] + 1))
                        or not isinstance(block_ids, list) or len(block_ids) != len(indexes)
                        or not isinstance(block_kinds, list) or len(block_kinds) != len(indexes)
                        or any(kind not in BLOCK_KINDS for kind in block_kinds)):
                    raise ValueError("verified core export candidate/block binding mismatch")
                for block_id in block_ids:
                    require_sha256(block_id, "block_id_hash")
                text = row["text"]
                char_count = require_int(row["core_char_count"], "core_char_count", minimum=1)
                byte_count = require_int(row["core_byte_count"], "core_byte_count", minimum=1)
                if (not isinstance(text, str) or not text or hashlib.sha256(text.encode()).hexdigest() != core_hash
                        or len(text) != char_count or len(text.encode()) != byte_count):
                    raise ValueError("verified core export text hash or length mismatch")
                tokens = require_int(row["core_proxy_token_count"], "core_proxy_token_count")
                run_id = stable_json_hash({"candidate_id_hash": candidate, "core_text_sha256": core_hash})
                values = (run_id, source, core_hash, text, len(text.encode()), len(indexes),
                          stable_json_hash(list(zip(indexes, block_ids, block_kinds))))
                existing = connection.execute("select * from runs where run_id_hash=?", (run_id,)).fetchone()
                if existing is None:
                    connection.execute("insert into runs values (?,?,?,?,?,?,?)", values)
                elif tuple(existing) != values:
                    raise ValueError("verified core export changed during resume")
                _find(connection, source)
                prior = connection.execute("select source_document_id_hash from duplicate_first where text_hash=?", (core_hash,)).fetchone()
                if prior is None:
                    connection.execute("insert into duplicate_first values (?,?)", (core_hash, source))
                else:
                    _union(connection, str(prior[0]), source)
                seen.add(candidate); sources[source] += 1; kinds.update(block_kinds)
                chars[bucket(len(text))] += 1; bytes_[bucket(len(text.encode()))] += 1
                part_rows += 1; part_tokens += tokens
        if (raw_digest.hexdigest() != descriptor.get("raw_jsonl_sha256")
                or part_rows != descriptor.get("row_count") or part_tokens != descriptor.get("accepted_token_count_lower_bound")
                or first != descriptor.get("first_global_ordinal") or previous != descriptor.get("last_global_ordinal")):
            raise ValueError("verified core export part facts mismatch")
        total_rows += part_rows; total_tokens += part_tokens
        part_roots.append(stable_json_hash(dict(descriptor)))
        connection.commit()
        if interrupt_after_parts is not None and part_index + 1 >= interrupt_after_parts:
            raise InterruptedError("synthetic interruption after verified core export part")
    if (total_rows != manifest.get("accepted_row_count") or total_tokens != manifest.get("accepted_token_count_lower_bound")
            or previous != manifest.get("last_global_ordinal")):
        raise ValueError("verified core export totals mismatch")
    concentration = Counter(sources.values())
    aggregate = {"core_char_length_bucket_counts": {name: chars[name] for name in bucket_names},
        "core_byte_length_bucket_counts": {name: bytes_[name] for name in bucket_names}, "block_kind_counts": dict(sorted(kinds.items())),
        "source_document_count": len(sources), "source_run_count_histogram": {str(k): v for k, v in sorted(concentration.items())},
        "max_source_run_share_basis_points": max(sources.values(), default=0) * 10_000 // max(total_rows, 1),
        "top_source_run_shares_basis_points": [count * 10_000 // max(total_rows, 1) for count in sorted(sources.values(), reverse=True)[:10]]}
    aggregate["aggregate_hash"] = stable_json_hash(aggregate)
    return total_rows, stable_json_hash(part_roots), aggregate


def build_accepted_clean_runs(
    *,
    handoff: Mapping[str, Any] | None = None,
    materialization_input: MaterializationInput | None = None,
    source_documents: Iterable[SourceDocument] | SequenceSourceDocumentSchedule = (),
    core_export_manifest: Mapping[str, Any] | None = None,
    core_export_root: Path | None = None,
    output_root: Path,
    validation_basis_points: int = 500,
    max_records_per_part: int = 10_000,
    max_part_bytes: int = 64 * 1024 * 1024,
    interrupt_after_documents: int | None = None,
    interrupt_after_fact_shards: int | None = None,
    interrupt_after_emitted_parts: int | None = None,
    fail_after_receipt_publication_parts: int | None = None,
) -> AcceptedCleanRunBuildResult:
    core_mode = core_export_manifest is not None
    if core_mode == (handoff is not None):
        raise ValueError("exactly one accepted input mode is required")
    if core_mode:
        _validate_core_export_manifest_header(core_export_manifest)
        export_hash = require_sha256(core_export_manifest.get("manifest_hash"), "core export manifest_hash")
        export_count = require_int(core_export_manifest.get("accepted_row_count"), "accepted_row_count", minimum=1)
        normalized_handoff = {
            "handoff_hash": export_hash,
            "selected_private_fact_index_root": export_hash,
            "source_content_root": export_hash,
            "selected_candidate_count": export_count,
        }
    else:
        normalized_handoff = validate_selected_materialization_handoff(handoff or {})
    require_int(validation_basis_points, "validation_basis_points")
    require_int(max_records_per_part, "max_records_per_part", minimum=1)
    require_int(max_part_bytes, "max_part_bytes", minimum=1)
    if not 0 < validation_basis_points < 10_000:
        raise ValueError("validation_basis_points must be between 1 and 9999")
    output_root = Path(output_root)
    manifest_path = output_root / "accepted_clean_run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        validate_accepted_clean_run_manifest(manifest, root=output_root)
        if core_mode and manifest.get("core_export_manifest_hash") != normalized_handoff["handoff_hash"]:
            raise ValueError("completed candidate-core manifest uses a different export")
        parts = tuple(output_root / item["part_name"] for item in manifest["part_receipts"])
        summary = {
            "artifact_kind": "twc_accepted_clean_run_public_summary_v1",
            "schema_version": 1,
            "manifest_hash": manifest["manifest_hash"],
            "selected_candidate_count": manifest["selected_candidate_count"],
            "accepted_clean_run_count": manifest["accepted_clean_run_count"],
            "duplicate_clean_run_count": manifest["duplicate_clean_run_count"],
            "split_counts": manifest["split_counts"],
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        return AcceptedCleanRunBuildResult(output_root, manifest, summary, parts, manifest_path)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_root, 0o700)
    state_path = output_root / ".materialization.sqlite"
    connection = _open_state(state_path)
    completed = False
    try:
        schedule = (
            source_documents
            if hasattr(source_documents, "iter_from")
            and hasattr(source_documents, "identity_hash")
            else SequenceSourceDocumentSchedule(
                source_documents,
                normalized_handoff["source_content_root"],
            )
            if isinstance(source_documents, Sequence)
            else None
        )
        if schedule is None:
            progress_cursor = int(
                connection.execute(
                    "select next_source_cursor from build_progress where id=1"
                ).fetchone()[0]
            )
            if progress_cursor:
                raise ValueError(
                    "resuming accepted materialization requires a seekable source schedule"
                )
            schedule_identity = normalized_handoff["source_content_root"]
        else:
            schedule_identity = schedule.identity_hash
        build_identity = {
            "handoff_hash": normalized_handoff["handoff_hash"],
            "selected_index_root": normalized_handoff["selected_private_fact_index_root"],
            "validation_basis_points": str(validation_basis_points),
            "max_records_per_part": str(max_records_per_part),
            "max_part_bytes": str(max_part_bytes),
            "source_schedule_identity": schedule_identity,
        }
        for key, value in build_identity.items():
            row = connection.execute("select value from build_identity where key=?", (key,)).fetchone()
            if row is None:
                connection.execute("insert into build_identity values (?,?)", (key, str(value)))
            elif row[0] != str(value):
                raise ValueError("accepted materialization resume identity mismatch")
        progress_row = connection.execute(
            "select facts_ingested,next_fact_shard,next_source_cursor,emission_next_part,"
            "emission_last_text_hash from build_progress where id=1"
        ).fetchone()
        if int(progress_row[2]) == 0 and hasattr(schedule, "verify_content"):
            schedule.verify_content()
        core_part_root: str | None = None
        core_aggregates: dict[str, Any] | None = None
        if core_mode:
            if core_export_root is None:
                raise ValueError("verified core export root is required")
            selected_count, core_part_root, core_aggregates = _ingest_verified_core_export(
                connection, core_export_manifest, Path(core_export_root),
                interrupt_after_parts=interrupt_after_fact_shards,
            )
            connection.execute("update build_progress set facts_ingested=1 where id=1")
        if int(progress_row[0]) == 0 and not core_mode:
            assert handoff is not None and materialization_input is not None
            start_fact_shard = int(progress_row[1])
            active_shard: int | None = None
            committed_this_call = 0
            for fact in iter_validated_materialization_facts(
                handoff,
                materialization_input,
                start_shard=start_fact_shard,
            ):
                if active_shard is None:
                    active_shard = fact.shard_number
                elif fact.shard_number != active_shard:
                    connection.execute(
                        "update build_progress set next_fact_shard=? where id=1",
                        (fact.shard_number,),
                    )
                    connection.commit()
                    committed_this_call += 1
                    if (
                        interrupt_after_fact_shards is not None
                        and committed_this_call >= interrupt_after_fact_shards
                    ):
                        raise InterruptedError(
                            "synthetic interruption after durable selected-fact shard"
                        )
                    active_shard = fact.shard_number
                values = (
                    fact.candidate_id_hash,
                    fact.shard_number,
                    fact.source_document_id_hash,
                    fact.first_block_index,
                    fact.last_block_index,
                    fact.core_text_sha256,
                    fact.core_char_count,
                    fact.core_byte_count,
                    fact.core_char_start,
                    fact.core_char_end,
                    fact.segmentation_receipt_hash,
                )
                existing = connection.execute(
                    "select candidate_id_hash,shard_number,source_document_id_hash,first_block_index,"
                    "last_block_index,core_text_sha256,core_char_count,core_byte_count,"
                    "core_char_start,core_char_end,segmentation_receipt_hash "
                    "from selected_facts where candidate_id_hash=?",
                    (fact.candidate_id_hash,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "insert into selected_facts(candidate_id_hash,shard_number,source_document_id_hash,"
                        "first_block_index,last_block_index,core_text_sha256,core_char_count,"
                        "core_byte_count,core_char_start,core_char_end,segmentation_receipt_hash) "
                        "values (?,?,?,?,?,?,?,?,?,?,?)",
                        values,
                    )
                elif tuple(existing) != values:
                    raise ValueError("resumed selected materialization fact changed")
            connection.execute(
                "update build_progress set facts_ingested=1,next_fact_shard=? where id=1",
                (max((int(item["shard_number"]) for item in materialization_input.selected_index_descriptor["shards"]), default=-1) + 1,),
            )
        connection.commit()
        document_count = matched_count = 0
        next_cursor = int(progress_row[2])
        units: Iterable[SourceDocumentUnit]
        if schedule is not None:
            units = schedule.iter_from(next_cursor)
        else:
            units = (
                SourceDocumentUnit(index + 1, document)
                for index, document in enumerate(source_documents)
            )
        for unit in units:
            document = unit.document
            matched_count += _materialize_document(connection, document)
            document_count += 1
            next_cursor = unit.next_cursor
            connection.execute(
                "update build_progress set next_source_cursor=? where id=1", (next_cursor,)
            )
            if document_count % 1_000 == 0:
                connection.commit()
            if interrupt_after_documents is not None and document_count >= interrupt_after_documents:
                connection.commit()
                raise InterruptedError("synthetic interruption after durable source document")
        connection.commit()
        if not core_mode:
            selected_count, unmatched = connection.execute(
                "select count(*),sum(case when matched=0 then 1 else 0 end) from selected_facts"
            ).fetchone()
            if int(selected_count) != normalized_handoff["selected_candidate_count"] or int(unmatched or 0):
                raise ValueError("not every selected candidate was rehydrated exactly once")

        before_count = int(connection.execute("select count(*) from runs").fetchone()[0])
        accepted_count = int(connection.execute("select count(distinct text_hash) from runs").fetchone()[0])
        emission_progress = connection.execute(
            "select emission_next_part,emission_last_text_hash from build_progress where id=1"
        ).fetchone()
        emission_next_part = int(emission_progress[0])
        emission_last_text_hash = emission_progress[1]
        try:
            receipts = [
                json.loads(path.read_text())
                for path in sorted(output_root.glob("accepted_clean_runs-*.receipt.json"))
            ]
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("accepted emission receipt JSON is invalid") from exc
        for receipt in receipts:
            _validate_part_receipt_schema(receipt, local=True)
        if len(receipts) < emission_next_part or len(receipts) > emission_next_part + 1:
            raise ValueError("accepted emission receipt cursor mismatch")

        def encoded_run(row: tuple[Any, ...]) -> tuple[bytes, bytes, str, str]:
            run_id, source_id, text_hash, text, byte_count, block_count, block_root = row
            split = _split(_find(connection, str(source_id)), validation_basis_points)
            private_row = {
                "clean_run_id_hash": run_id,
                "source_document_id_hash": source_id,
                "clean_run_text_hash": text_hash,
                "split": split,
                "utf8_byte_count": byte_count,
                "selected_block_count": block_count,
                "selected_block_root": block_root,
                "text": text,
            }
            metadata = {key: value for key, value in private_row.items() if key != "text"}
            encoded = (
                json.dumps(private_row, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            metadata_encoded = (
                json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            return encoded, metadata_encoded, split, str(text_hash)

        if len(receipts) == emission_next_part + 1:
            orphan = receipts[-1]
            expected_orphan_name = f"accepted_clean_runs-{emission_next_part:06d}.jsonl"
            if (
                orphan.get("part_ordinal") != emission_next_part
                or orphan.get("part_name") != expected_orphan_name
            ):
                raise ValueError("accepted orphan part ordinal mismatch")
            if orphan.get("selection_handoff_hash") != normalized_handoff["handoff_hash"]:
                raise ValueError("accepted orphan receipt binding mismatch")
            orphan_path = output_root / orphan["part_name"]
            if (
                not orphan_path.is_file()
                or orphan_path.is_symlink()
                or orphan_path.stat().st_nlink != 1
            ):
                raise ValueError("accepted orphan part must be a single-link regular file")
            file_metadata = orphan_path.stat()
            if orphan.get("file_identity") != {
                "device": file_metadata.st_dev,
                "inode": file_metadata.st_ino,
                "byte_count": file_metadata.st_size,
                "mtime_ns": file_metadata.st_mtime_ns,
            }:
                raise ValueError("accepted orphan file identity mismatch")
            orphan_data = orphan_path.read_bytes()
            if hashlib.sha256(orphan_data).hexdigest() != orphan.get("sha256"):
                raise ValueError("accepted orphan content hash mismatch")
            expected_rows = connection.execute(
                "select r.run_id_hash,r.source_document_id_hash,r.text_hash,r.text_value,"
                "r.utf8_byte_count,r.selected_block_count,r.selected_block_root "
                "from runs r where r.run_id_hash=(select min(r2.run_id_hash) from runs r2 "
                "where r2.text_hash=r.text_hash) and (? is null or r.text_hash>?) "
                "order by r.text_hash limit ?",
                (
                    emission_last_text_hash,
                    emission_last_text_hash,
                    orphan["record_count"] + 1,
                ),
            ).fetchall()
            orphan_count = orphan["record_count"]
            if orphan_count < 1 or len(expected_rows) < orphan_count:
                raise ValueError("accepted orphan record coverage mismatch")
            encoded_rows = [encoded_run(tuple(row)) for row in expected_rows]
            expected_data = b"".join(item[0] for item in encoded_rows[:orphan_count])
            expected_metadata = hashlib.sha256(
                b"".join(item[1] for item in encoded_rows[:orphan_count])
            ).hexdigest()
            expected_splits = {
                split: sum(item[2] == split for item in encoded_rows[:orphan_count])
                for split in ("train", "validation")
            }
            expected_last = encoded_rows[orphan_count - 1][3]
            if (
                orphan_data != expected_data
                or orphan.get("byte_count") != len(expected_data)
                or orphan.get("record_metadata_sha256") != expected_metadata
                or orphan.get("split_counts") != expected_splits
                or orphan.get("last_text_hash") != expected_last
            ):
                raise ValueError("accepted orphan does not match the next deterministic part")
            if orphan_count > max_records_per_part or (
                orphan_count > 1 and len(expected_data) > max_part_bytes
            ):
                raise ValueError("accepted orphan exceeds configured part bounds")
            if orphan_count < len(expected_rows):
                next_size = len(encoded_rows[orphan_count][0])
                if orphan_count < max_records_per_part and len(expected_data) + next_size <= max_part_bytes:
                    raise ValueError("accepted orphan ended before its deterministic boundary")
            connection.execute(
                "update build_progress set emission_next_part=?,emission_last_text_hash=? "
                "where id=1",
                (emission_next_part + 1, expected_last),
            )
            connection.commit()
            emission_next_part += 1
            emission_last_text_hash = expected_last
        parts: list[Path] = []
        for ordinal, receipt in enumerate(receipts):
            if (
                receipt.get("part_ordinal") != ordinal
                or receipt.get("receipt_hash")
                != stable_json_hash(
                    {key: value for key, value in receipt.items() if key != "receipt_hash"}
                )
            ):
                raise ValueError("accepted emission receipt identity mismatch")
            path = output_root / receipt["part_name"]
            metadata = path.stat()
            if receipt.get("file_identity") != {
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "byte_count": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
            }:
                raise ValueError("committed accepted part file identity changed")
            parts.append(path)
        if receipts and receipts[-1].get("last_text_hash") != emission_last_text_hash:
            raise ValueError("accepted emission keyset cursor mismatch")
        rows = connection.execute(
            "select r.run_id_hash,r.source_document_id_hash,r.text_hash,r.text_value,"
            "r.utf8_byte_count,r.selected_block_count,r.selected_block_root "
            "from runs r where r.run_id_hash=(select min(r2.run_id_hash) from runs r2 "
            "where r2.text_hash=r.text_hash) and (? is null or r.text_hash>?) "
            "order by r.text_hash",
            (emission_last_text_hash, emission_last_text_hash),
        )
        split_counts = {
            split: sum(int(receipt["split_counts"][split]) for receipt in receipts)
            for split in ("train", "validation")
        }
        metadata_unit_roots = [str(receipt["record_metadata_sha256"]) for receipt in receipts]
        part_rows: list[bytes] = []
        part_bytes = 0
        part_split_counts = {"train": 0, "validation": 0}
        part_metadata_digest = hashlib.sha256()
        part_last_text_hash: str | None = None
        emitted_this_call = 0
        published_before_commit = 0

        def flush_part() -> None:
            nonlocal part_bytes, part_split_counts, part_metadata_digest
            nonlocal part_last_text_hash, emitted_this_call
            nonlocal published_before_commit
            if not part_rows:
                return
            ordinal = len(parts)
            path = output_root / f"accepted_clean_runs-{ordinal:06d}.jsonl"
            data = b"".join(part_rows)
            _atomic_write(path, data)
            file_metadata = path.stat()
            parts.append(path)
            receipt = {
                    "artifact_kind": "twc_accepted_clean_run_part_receipt_v1",
                    "schema_version": 1,
                    "part_ordinal": ordinal,
                    "part_name": path.name,
                    "record_count": len(part_rows),
                    "byte_count": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "split_counts": part_split_counts,
                    "record_metadata_sha256": part_metadata_digest.hexdigest(),
                    "last_text_hash": part_last_text_hash,
                    "selection_handoff_hash": normalized_handoff["handoff_hash"],
                    "raw_text_serialized": False,
                    "raw_url_serialized": False,
                    "token_serialized": False,
                    "file_identity": {
                        "device": file_metadata.st_dev,
                        "inode": file_metadata.st_ino,
                        "byte_count": file_metadata.st_size,
                        "mtime_ns": file_metadata.st_mtime_ns,
                    },
            }
            receipt["receipt_hash"] = stable_json_hash(receipt)
            _validate_part_receipt_schema(receipt, local=True)
            _atomic_write(
                output_root / f"accepted_clean_runs-{ordinal:06d}.receipt.json",
                json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(),
            )
            published_before_commit += 1
            if (
                fail_after_receipt_publication_parts is not None
                and published_before_commit >= fail_after_receipt_publication_parts
            ):
                raise RuntimeError(
                    "synthetic crash after accepted receipt publication before cursor commit"
                )
            receipts.append(receipt)
            metadata_unit_roots.append(part_metadata_digest.hexdigest())
            connection.execute(
                "update build_progress set emission_next_part=?,emission_last_text_hash=? "
                "where id=1",
                (ordinal + 1, part_last_text_hash),
            )
            connection.commit()
            part_rows.clear()
            part_bytes = 0
            part_split_counts = {"train": 0, "validation": 0}
            part_metadata_digest = hashlib.sha256()
            part_last_text_hash = None
            emitted_this_call += 1
            if (
                interrupt_after_emitted_parts is not None
                and emitted_this_call >= interrupt_after_emitted_parts
            ):
                raise InterruptedError("synthetic interruption after durable accepted part")

        for run_id, source_id, text_hash, text, byte_count, block_count, block_root in rows:
            component_root = _find(connection, str(source_id))
            split = _split(component_root, validation_basis_points)
            private_row = {
                "clean_run_id_hash": run_id,
                "source_document_id_hash": source_id,
                "clean_run_text_hash": text_hash,
                "split": split,
                "utf8_byte_count": byte_count,
                "selected_block_count": block_count,
                "selected_block_root": block_root,
                "text": text,
            }
            encoded = (json.dumps(private_row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            if part_rows and part_bytes + len(encoded) > max_part_bytes:
                flush_part()
            split_counts[split] += 1
            part_split_counts[split] += 1
            part_last_text_hash = str(text_hash)
            part_rows.append(encoded)
            part_bytes += len(encoded)
            metadata = {
                "clean_run_id_hash": run_id,
                "source_document_id_hash": source_id,
                "clean_run_text_hash": text_hash,
                "split": split,
                "utf8_byte_count": byte_count,
                "selected_block_count": block_count,
                "selected_block_root": block_root,
            }
            part_metadata_digest.update(
                (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
            if len(part_rows) >= max_records_per_part:
                flush_part()
        flush_part()
        manifest: dict[str, Any] = {
            "artifact_kind": CORE_RUN_MANIFEST_KIND if core_mode else MANIFEST_KIND,
            "schema_version": 1,
            "selection_handoff_hash": normalized_handoff["handoff_hash"],
            "selected_private_fact_index_root": normalized_handoff["selected_private_fact_index_root"],
            "source_snapshot_root": normalized_handoff["source_content_root"],
            "selected_candidate_count": int(selected_count),
            "clean_run_count_before_dedup": before_count,
            "accepted_clean_run_count": accepted_count,
            "duplicate_clean_run_count": before_count - accepted_count,
            "split_counts": split_counts,
            "record_metadata_sha256": stable_json_hash(metadata_unit_roots),
            "part_receipts": [_canonical_part_receipt(receipt) for receipt in receipts],
            "inline_records_present": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        if core_mode:
            manifest.pop("selection_handoff_hash")
            manifest.pop("selected_private_fact_index_root")
            manifest.pop("source_snapshot_root")
            manifest.update({"core_export_manifest_hash": normalized_handoff["handoff_hash"],
                             "core_export_part_root": core_part_root, "core_run_aggregates": core_aggregates})
        manifest["manifest_hash"] = stable_json_hash(manifest)
        validate_accepted_clean_run_manifest(manifest, root=output_root)
        _atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
        summary = {
            "artifact_kind": "twc_accepted_clean_run_public_summary_v1",
            "schema_version": 1,
            "manifest_hash": manifest["manifest_hash"],
            "selected_candidate_count": int(selected_count),
            "accepted_clean_run_count": accepted_count,
            "duplicate_clean_run_count": before_count - accepted_count,
            "split_counts": split_counts,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        validate_public_no_raw_payload(summary)
        completed = True
        return AcceptedCleanRunBuildResult(output_root, manifest, summary, tuple(parts), manifest_path)
    except BaseException:
        for partial in output_root.glob("*.partial"):
            partial.unlink(missing_ok=True)
        raise
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass
        if completed:
            for suffix in ("", "-wal", "-shm"):
                (Path(str(state_path) + suffix)).unlink(missing_ok=True)


def validate_accepted_clean_run_manifest(
    payload: Mapping[str, Any], *, root: Path | None = None
) -> None:
    validate_public_no_raw_payload(payload)
    kind = payload.get("artifact_kind")
    if kind not in {MANIFEST_KIND, CORE_RUN_MANIFEST_KIND} or payload.get("schema_version") != 1:
        raise ValueError("unsupported accepted clean-run manifest")
    if payload.get("inline_records_present") is not False or "records" in payload:
        raise ValueError("accepted clean-run manifest forbids inline records")
    hash_fields = ["record_metadata_sha256", "manifest_hash"]
    hash_fields += (["selection_handoff_hash", "selected_private_fact_index_root", "source_snapshot_root"] if kind == MANIFEST_KIND
                    else ["core_export_manifest_hash", "core_export_part_root"])
    for field in hash_fields:
        require_sha256(payload.get(field), field)
    if kind == CORE_RUN_MANIFEST_KIND:
        aggregate = payload.get("core_run_aggregates")
        if not isinstance(aggregate, Mapping) or aggregate.get("aggregate_hash") != hash_without(aggregate, "aggregate_hash"):
            raise ValueError("accepted candidate-core aggregates mismatch")
    if payload["manifest_hash"] != hash_without(payload, "manifest_hash"):
        raise ValueError("accepted clean-run manifest hash mismatch")
    receipts = payload.get("part_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("accepted clean-run parts must be non-empty")
    for item in receipts:
        _validate_part_receipt_schema(item, local=False)
    if kind == CORE_RUN_MANIFEST_KIND and any(
        item["selection_handoff_hash"] != payload["core_export_manifest_hash"]
        for item in receipts
    ):
        raise ValueError("accepted candidate-core core export receipt binding mismatch")
    if sum(item["record_count"] for item in receipts) != payload.get(
        "accepted_clean_run_count"
    ):
        raise ValueError("accepted clean-run part count mismatch")
    if root is not None:
        root = Path(root)
        validation_path = root / ".accepted-validation.sqlite"
        validation = sqlite3.connect(validation_path)
        os.chmod(validation_path, 0o600)
        validation.executescript(
            "create table texts(hash text primary key);"
            "create table source_splits(source text primary key,split text not null);"
        )
        metadata_unit_roots: list[str] = []
        count = 0
        split_counts = {"train": 0, "validation": 0}
        try:
            for item in receipts:
                local_receipt_path = root / str(item["part_name"]).replace(
                    ".jsonl", ".receipt.json"
                )
                try:
                    local_item = json.loads(local_receipt_path.read_text())
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError("accepted clean-run receipt JSON is invalid") from exc
                if _validate_part_receipt_schema(local_item, local=True) != item:
                    raise ValueError("accepted clean-run part receipt hash mismatch")
                name = item.get("part_name")
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError("invalid accepted clean-run part name")
                path = root / name
                if not path.is_file() or path.stat().st_size != item.get("byte_count"):
                    raise ValueError("accepted clean-run part size mismatch")
                digest = hashlib.sha256()
                part_metadata_digest = hashlib.sha256()
                part_count = 0
                with path.open("rb") as handle:
                    for line in handle:
                        digest.update(line)
                        row = json.loads(line)
                        required = {
                            "clean_run_id_hash",
                            "source_document_id_hash",
                            "clean_run_text_hash",
                            "split",
                            "utf8_byte_count",
                            "selected_block_count",
                            "selected_block_root",
                            "text",
                        }
                        if set(row) != required:
                            raise ValueError("accepted clean-run private row schema mismatch")
                        for field in (
                            "clean_run_id_hash",
                            "source_document_id_hash",
                            "clean_run_text_hash",
                            "selected_block_root",
                        ):
                            require_sha256(row[field], field)
                        text = row["text"]
                        if not isinstance(text, str) or not text:
                            raise ValueError("accepted clean-run text is invalid")
                        if hashlib.sha256(text.encode()).hexdigest() != row["clean_run_text_hash"]:
                            raise ValueError("accepted clean-run text hash mismatch")
                        if len(text.encode()) != row["utf8_byte_count"]:
                            raise ValueError("accepted clean-run byte count mismatch")
                        split = row["split"]
                        if split not in split_counts:
                            raise ValueError("accepted clean-run split is invalid")
                        try:
                            validation.execute("insert into texts values (?)", (row["clean_run_text_hash"],))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError("accepted clean-run dedup uniqueness mismatch") from exc
                        prior = validation.execute(
                            "select split from source_splits where source=?",
                            (row["source_document_id_hash"],),
                        ).fetchone()
                        if prior is None:
                            validation.execute(
                                "insert into source_splits values (?,?)",
                                (row["source_document_id_hash"], split),
                            )
                        elif prior[0] != split:
                            raise ValueError("accepted source-document split leakage")
                        metadata = {key: value for key, value in row.items() if key != "text"}
                        part_metadata_digest.update(
                            (json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n").encode()
                        )
                        count += 1
                        part_count += 1
                        split_counts[split] += 1
                if digest.hexdigest() != item.get("sha256"):
                    raise ValueError("accepted clean-run part hash mismatch")
                if part_count != item.get("record_count"):
                    raise ValueError("accepted clean-run part record count mismatch")
                if part_metadata_digest.hexdigest() != item.get("record_metadata_sha256"):
                    raise ValueError("accepted clean-run part metadata root mismatch")
                metadata_unit_roots.append(part_metadata_digest.hexdigest())
            if count != payload.get("accepted_clean_run_count"):
                raise ValueError("accepted clean-run aggregate count mismatch")
            if split_counts != payload.get("split_counts"):
                raise ValueError("accepted clean-run split counts mismatch")
            if stable_json_hash(metadata_unit_roots) != payload.get("record_metadata_sha256"):
                raise ValueError("accepted clean-run record metadata root mismatch")
        finally:
            validation.close()
            validation_path.unlink(missing_ok=True)
