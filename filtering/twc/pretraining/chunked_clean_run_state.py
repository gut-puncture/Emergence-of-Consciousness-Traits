"""SQLite state for chunked clean-run segmentation.

SQLite is the source of truth for this chunk. Public JSON summaries are derived
from committed rows and contain only aggregate, no-raw evidence.
"""

from __future__ import annotations

import json
import fcntl
import os
import sqlite3
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from twc.pretraining.chunked_clean_run import (
    BLOCK_KINDS,
    CHUNKED_CLEAN_RUN_SCHEMA_VERSION,
    NORMALIZER_VERSION,
    PUBLIC_REPORT_STATUS,
    PUBLIC_REPORT_SCHEMA_VERSION,
    PROXY_TOKEN_ESTIMATOR_VERSION,
    RAW_LINE_POLICY_VERSION,
    SEGMENTATION_POLICY_VERSION,
    TERMINATORS,
    NaturalBlockRecord,
    SegmentationReceipt,
    SegmentationResult,
    _validate_public_chunked_segmentation_report_payload,
    block_id_policy_hash,
    build_public_chunked_segmentation_report,
    natural_block_id_hash,
    public_report_schema_hash,
    raw_line_policy_hash,
    segmentation_policy_hash,
    validate_segmentation_result,
)
from twc.data.decision_schema import stable_json_hash


STATE_DB_FILENAME = "chunked_clean_run_state.sqlite"
STATE_LOCK_FILENAME = ".chunked_clean_run_state.lock"
_ACTIVE_LOCKS: set[Path] = set()
FILTERING_LANE_TABLES = (
    "classifier_candidates",
    "chunk_decisions",
    "gap_events",
    "clean_runs",
)
FUTURE_LANE_TABLES = (
    *FILTERING_LANE_TABLES,
    "tokenizer_counts",
    "packed_examples",
)
_RECEIPT_PAYLOAD_KEYS = set(SegmentationReceipt.__dataclass_fields__)
_RECEIPT_TRUE_FIELDS = {"raw_coverage_validation", "reconstruction_validation", "block_order_validation"}
_RAW_LINE_KINDS = {
    "blank",
    "fence",
    "html_pre",
    "heading",
    "table_like",
    "list",
    "stack_trace",
    "log",
    "indented_code",
    "html",
    "ui_chrome",
    "prose",
}


@dataclass(frozen=True)
class ChunkedCleanRunStateConfig:
    run_dir: Path
    source_manifest_hash: str | None = None
    source_schedule_hash: str | None = None


@dataclass
class ChunkedCleanRunState:
    config: ChunkedCleanRunStateConfig
    conn: sqlite3.Connection
    lock_path: Path
    _lock_key: Path
    _lock_fd: int | None

    def close(self) -> None:
        try:
            self.conn.close()
        finally:
            if self._lock_fd is not None:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(self._lock_fd)
                self._lock_fd = None
            _ACTIVE_LOCKS.discard(self._lock_key)

    def __enter__(self) -> "ChunkedCleanRunState":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_chunked_clean_run_state(
    config: ChunkedCleanRunStateConfig,
    *,
    resume: bool = False,
) -> ChunkedCleanRunState:
    del resume  # Existing compatible state is always resumable/idempotent.
    normalized = _validate_config(config)
    normalized.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(normalized.run_dir, 0o700)
    except OSError:
        pass
    lock_path = normalized.run_dir / STATE_LOCK_FILENAME
    lock_key = lock_path.resolve(strict=False)
    lock_fd: int | None = None
    if lock_key in _ACTIVE_LOCKS:
        raise ValueError("chunked clean-run state is already locked")
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        _ACTIVE_LOCKS.add(lock_key)
    except BlockingIOError as exc:
        if lock_fd is not None:
            os.close(lock_fd)
        raise ValueError("chunked clean-run state is already locked") from exc

    try:
        conn = sqlite3.connect(normalized.run_dir / STATE_DB_FILENAME, timeout=1.0)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma synchronous=normal")
        _create_state_schema(conn)
        _bind_state_metadata(conn, normalized)
        return ChunkedCleanRunState(
            config=normalized,
            conn=conn,
            lock_path=lock_path,
            _lock_key=lock_key,
            _lock_fd=lock_fd,
        )
    except sqlite3.DatabaseError as exc:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_fd)
        _ACTIVE_LOCKS.discard(lock_key)
        raise ValueError("chunked clean-run state is corrupt or unreadable") from exc
    except Exception:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_fd)
        _ACTIVE_LOCKS.discard(lock_key)
        raise


def store_segmentation_result(state: ChunkedCleanRunState, result: SegmentationResult) -> bool:
    validate_segmentation_result(result)
    expected_policy_hash = _metadata_dict(state.conn)["segmentation_policy_hash"]
    if result.receipt.segmentation_policy_hash != expected_policy_hash:
        raise ValueError("segmentation policy drift between result and chunked state")
    source_id = result.source.source_document_id_hash
    existing = state.conn.execute(
        "select split, status, raw_text_sha256, normalized_text_sha256, "
        "source_lineage_hash, source_token_count_hint, segmentation_receipt_hash "
        "from source_documents where source_document_id_hash = ?",
        (source_id,),
    ).fetchone()
    expected_receipt_hash = result.receipt.receipt_hash
    source_token_count_hint = result.source.source_token_count_hint
    if existing is not None:
        if (
            str(existing["split"]) == result.source.split
            and str(existing["status"]) == result.receipt.document_status
            and _nullable(existing["raw_text_sha256"]) == result.receipt.raw_text_sha256
            and _nullable(existing["normalized_text_sha256"]) == result.receipt.normalized_text_sha256
            and _nullable(existing["source_lineage_hash"]) == result.source.source_lineage_hash
            and _nullable_int(existing["source_token_count_hint"]) == source_token_count_hint
            and str(existing["segmentation_receipt_hash"]) == expected_receipt_hash
        ):
            _validate_existing_document_matches_result(state.conn, result)
            return False
        raise ValueError("conflicting source document segmentation already exists")

    with state.conn:
        _insert_source_document(state.conn, result)
        _insert_raw_lines(state.conn, result)
        _insert_natural_blocks(state.conn, result)
        _insert_trivia_spans(state.conn, result)
        _insert_segmentation_receipt(state.conn, result)
        state.conn.execute(
            "insert into phase_receipts(phase_name, source_document_id_hash, receipt_hash, created_at) values (?, ?, ?, ?)",
            ("natural_block_segmentation", source_id, expected_receipt_hash, time.time()),
        )
    return True


def validate_chunked_clean_run_state(state: ChunkedCleanRunState, *, lane_mode: str = "segmentation_only") -> None:
    conn = state.conn
    if lane_mode == "segmentation_only":
        _validate_future_lane_tables_empty(conn)
    elif lane_mode == "filtering":
        _validate_future_lane_tables_empty(conn, tables=("tokenizer_counts", "packed_examples"))
    else:
        raise ValueError("unsupported chunked clean-run lane_mode")
    _validate_no_orphan_rows(conn)
    metadata = _metadata_dict(conn)
    for row in conn.execute("select * from source_documents order by source_document_id_hash"):
        source_id = str(row["source_document_id_hash"])
        _require_sha256(source_id, "source_document_id_hash")
        if str(row["split"]) not in {"train", "validation"}:
            raise ValueError("source document split is unsupported in chunked state")
        for field in ("raw_text_sha256", "normalized_text_sha256", "source_lineage_hash", "segmentation_receipt_hash"):
            value = row[field]
            if value is not None:
                _require_sha256(str(value), field)
        status = str(row["status"])
        raw_char_count = _require_nonnegative_int(row["raw_char_count"], "raw_char_count")
        raw_line_count = _require_nonnegative_int(row["raw_line_count"], "raw_line_count")
        total_proxy_token_count = _require_nonnegative_int(
            row["total_proxy_token_count"],
            "total_proxy_token_count",
        )
        line_rows = list(
            conn.execute(
                "select * from raw_lines where source_document_id_hash = ? order by line_index",
                (source_id,),
            )
        )
        block_rows = list(
            conn.execute(
                "select * from natural_blocks where source_document_id_hash = ? order by block_index",
                (source_id,),
            )
        )
        trivia_rows = list(
            conn.execute(
                "select * from natural_trivia_spans where source_document_id_hash = ? order by char_start",
                (source_id,),
            )
        )
        receipt = conn.execute(
            "select * from segmentation_receipts where source_document_id_hash = ?",
            (source_id,),
        ).fetchone()
        if receipt is None:
            raise ValueError("segmentation receipt missing from chunked state")

        if status in {"missing_text", "non_string_text", "empty_or_whitespace"}:
            _validate_terminal_source_row(row, block_rows, trivia_rows, line_rows)
        elif status == "segmented":
            if raw_char_count == 0 or raw_line_count == 0 or not block_rows:
                raise ValueError("segmented source document must have text, lines, and blocks")
            if row["raw_text_sha256"] is None or row["normalized_text_sha256"] is None:
                raise ValueError("segmented source document must have text hashes")
            _validate_raw_line_rows(line_rows, raw_char_count)
            if len(line_rows) != raw_line_count:
                raise ValueError("source document raw line count mismatch")
            indexes = [int(block["block_index"]) for block in block_rows]
            if indexes != list(range(len(block_rows))):
                raise ValueError("natural block indexes must be contiguous in state")
            _validate_state_block_order(block_rows)
            expected_policy_hash = _metadata_dict(conn)["segmentation_policy_hash"]
            for block in block_rows:
                _validate_block_row(
                    block,
                    expected_policy_hash=expected_policy_hash,
                    line_rows=line_rows,
                )
            for trivia in trivia_rows:
                _validate_trivia_row(
                    trivia,
                    line_rows=line_rows,
                )
            coverage = [
                (int(block["owned_char_start"]), int(block["owned_char_end"]), "block")
                for block in block_rows
            ]
            coverage.extend(
                (int(trivia["char_start"]), int(trivia["char_end"]), "trivia")
                for trivia in trivia_rows
            )
            _validate_span_coverage(coverage, raw_char_count)
            _validate_state_trivia_adjacency(block_rows, trivia_rows, raw_char_count)
        else:
            raise ValueError("unsupported source document status in chunked state")

        if int(row["block_count"]) != len(block_rows):
            raise ValueError("source document block count mismatch")
        if int(row["trivia_span_count"]) != len(trivia_rows):
            raise ValueError("source document trivia span count mismatch")
        if int(row["raw_line_count"]) != len(line_rows):
            raise ValueError("source document raw line count mismatch")
        if int(row["total_proxy_token_count"]) != sum(int(block["proxy_token_count"]) for block in block_rows):
            raise ValueError("source document proxy token count mismatch")
        if int(receipt["block_count"]) != len(block_rows):
            raise ValueError("segmentation receipt block count mismatch")
        if int(receipt["trivia_span_count"]) != len(trivia_rows):
            raise ValueError("segmentation receipt trivia span count mismatch")
        if int(receipt["raw_line_count"]) != len(line_rows):
            raise ValueError("segmentation receipt raw line count mismatch")
        if int(receipt["total_proxy_token_count"]) != total_proxy_token_count:
            raise ValueError("segmentation receipt proxy token count mismatch")
        receipt_payload = _validate_receipt_row(receipt)
        _validate_source_receipt_consistency(row, receipt, receipt_payload, metadata)
        _validate_block_receipt_consistency(block_rows, receipt_payload)
        _validate_phase_receipts(conn, row)
    if lane_mode == "filtering":
        _validate_filtering_lane_rows(conn)


def build_public_chunked_segmentation_report_from_state(state: ChunkedCleanRunState) -> dict[str, Any]:
    report = _build_expected_public_chunked_segmentation_report_from_state(state)
    _validate_public_chunked_segmentation_report_payload(report, allow_validated_report=True)
    return report


def validate_public_chunked_segmentation_report_from_state(
    state: ChunkedCleanRunState,
    payload: dict[str, Any],
) -> None:
    expected = _build_expected_public_chunked_segmentation_report_from_state(state)
    _validate_public_chunked_segmentation_report_payload(expected, allow_validated_report=True)
    _validate_public_chunked_segmentation_report_payload(payload, allow_validated_report=True)
    if payload != expected:
        raise ValueError("chunked segmentation report does not match SQLite source of truth")


def _build_expected_public_chunked_segmentation_report_from_state(state: ChunkedCleanRunState) -> dict[str, Any]:
    validate_chunked_clean_run_state(state)
    conn = state.conn
    document_count = _sql_count(conn, "source_documents")
    block_count = _sql_count(conn, "natural_blocks")
    trivia_count = _sql_count(conn, "natural_trivia_spans")

    status_counts = _sql_count_by(conn, "source_documents", "status")
    split_counts = _sql_count_by(conn, "source_documents", "split")
    block_kind_counts = _sql_count_by(conn, "natural_blocks", "block_kind")
    warning_counts: Counter[str] = Counter()
    proxy_histogram: Counter[str] = Counter()
    for block in conn.execute("select warning_codes_json, proxy_token_count from natural_blocks"):
        warning_counts.update(json.loads(str(block["warning_codes_json"])))
        proxy_histogram[_proxy_bucket(int(block["proxy_token_count"]))] += 1

    metadata = _metadata_dict(conn)
    sqlite_aggregate_hash = _sqlite_aggregate_hash(conn)
    report = build_public_chunked_segmentation_report(
        document_count=document_count,
        status_counts=dict(status_counts),
        split_counts=dict(split_counts),
        total_block_count=block_count,
        total_trivia_span_count=int(trivia_count),
        block_kind_counts=dict(block_kind_counts),
        warning_code_counts=dict(warning_counts),
        proxy_token_histogram=dict(proxy_histogram),
        source_manifest_hash=_none_to_hash(metadata.get("source_manifest_hash")),
        source_schedule_hash=_none_to_hash(metadata.get("source_schedule_hash")),
    )
    report = {
        **report,
        "artifact_status": PUBLIC_REPORT_STATUS,
        "sqlite_source_of_truth": True,
        "sqlite_aggregate_hash": sqlite_aggregate_hash,
    }
    report["report_hash"] = stable_json_hash({key: value for key, value in report.items() if key != "report_hash"})
    _validate_public_chunked_segmentation_report_payload(report, allow_validated_report=True)
    return report


def write_public_chunked_segmentation_report(state: ChunkedCleanRunState, path: str | Path) -> dict[str, Any]:
    report = build_public_chunked_segmentation_report_from_state(state)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        Path(tmp_name).replace(output)
    except Exception:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass
        raise
    return report


def _create_state_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create table if not exists metadata(
            key text primary key,
            value text not null
        );
        create table if not exists source_documents(
            source_document_id_hash text primary key,
            split text not null,
            status text not null,
            raw_text_sha256 text,
            normalized_text_sha256 text,
            source_lineage_hash text,
            source_token_count_hint integer,
            raw_char_count integer not null,
            raw_line_count integer not null,
            block_count integer not null,
            trivia_span_count integer not null,
            total_proxy_token_count integer not null,
            segmentation_receipt_hash text not null,
            warning_codes_json text not null,
            created_at real not null,
            updated_at real not null
        );
        create table if not exists raw_lines(
            source_document_id_hash text not null,
            line_index integer not null,
            char_start integer not null,
            char_end integer not null,
            content_end integer not null,
            terminator text not null,
            line_kind text not null,
            unique(source_document_id_hash, line_index),
            unique(source_document_id_hash, char_start, char_end)
        );
        create table if not exists natural_blocks(
            block_id_hash text primary key,
            source_document_id_hash text not null,
            segmentation_policy_hash text not null default '',
            block_index integer not null,
            block_kind text not null,
            content_char_start integer not null,
            content_char_end integer not null,
            owned_char_start integer not null,
            owned_char_end integer not null,
            start_line_index integer not null,
            end_line_index_exclusive integer not null,
            raw_text_sha256 text not null,
            normalized_text_sha256 text not null,
            proxy_token_count integer not null,
            warning_codes_json text not null,
            unique(source_document_id_hash, block_index),
            unique(source_document_id_hash, content_char_start, content_char_end)
        );
        create table if not exists natural_trivia_spans(
            source_document_id_hash text not null,
            char_start integer not null,
            char_end integer not null,
            start_line_index integer not null,
            end_line_index_exclusive integer not null,
            trivia_kind text not null,
            left_block_index integer,
            right_block_index integer,
            unique(source_document_id_hash, char_start, char_end)
        );
        create table if not exists segmentation_receipts(
            source_document_id_hash text primary key,
            receipt_hash text not null,
            document_status text not null,
            raw_char_count integer not null,
            raw_line_count integer not null,
            block_count integer not null,
            trivia_span_count integer not null,
            total_proxy_token_count integer not null,
            receipt_json text not null
        );
        create table if not exists phase_receipts(
            id integer primary key autoincrement,
            phase_name text not null,
            source_document_id_hash text not null,
            receipt_hash text not null,
            created_at real not null,
            unique(phase_name, source_document_id_hash)
        );
        create table if not exists classifier_candidates(
            candidate_id_hash text primary key,
            status text not null
        );
        create table if not exists chunk_decisions(
            decision_id_hash text primary key,
            status text not null
        );
        create table if not exists gap_events(
            gap_event_id_hash text primary key,
            status text not null
        );
        create table if not exists clean_runs(
            clean_run_id_hash text primary key,
            status text not null
        );
        create table if not exists tokenizer_counts(
            tokenizer_count_id_hash text primary key,
            status text not null
        );
        create table if not exists packed_examples(
            packed_example_id_hash text primary key,
            status text not null
        );
        """
    )


def _bind_state_metadata(conn: sqlite3.Connection, config: ChunkedCleanRunStateConfig) -> None:
    expected = {
        "chunked_clean_run_schema_version": str(CHUNKED_CLEAN_RUN_SCHEMA_VERSION),
        "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
        "segmentation_policy_hash": segmentation_policy_hash(),
        "raw_line_policy_version": RAW_LINE_POLICY_VERSION,
        "raw_line_policy_hash": raw_line_policy_hash(),
        "normalizer_version": NORMALIZER_VERSION,
        "proxy_token_estimator_version": PROXY_TOKEN_ESTIMATOR_VERSION,
        "block_id_policy_hash": block_id_policy_hash(),
        "public_report_schema_version": str(PUBLIC_REPORT_SCHEMA_VERSION),
        "public_report_schema_hash": public_report_schema_hash(),
        "source_manifest_hash": config.source_manifest_hash or "<none>",
        "source_schedule_hash": config.source_schedule_hash or "<none>",
    }
    with conn:
        for key, value in expected.items():
            row = conn.execute("select value from metadata where key = ?", (key,)).fetchone()
            if row is not None and str(row["value"] if isinstance(row, sqlite3.Row) else row[0]) != value:
                raise ValueError(f"chunked clean-run state metadata drift: {key}")
            conn.execute("insert or ignore into metadata(key, value) values (?, ?)", (key, value))


def _insert_source_document(conn: sqlite3.Connection, result: SegmentationResult) -> None:
    receipt = result.receipt
    now = time.time()
    conn.execute(
        """
        insert into source_documents(
            source_document_id_hash, split, status, raw_text_sha256, normalized_text_sha256,
            source_lineage_hash, source_token_count_hint, raw_char_count, raw_line_count,
            block_count, trivia_span_count, total_proxy_token_count, segmentation_receipt_hash,
            warning_codes_json, created_at, updated_at
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.source.source_document_id_hash,
            result.source.split,
            receipt.document_status,
            receipt.raw_text_sha256,
            receipt.normalized_text_sha256,
            result.source.source_lineage_hash,
            result.source.source_token_count_hint,
            receipt.raw_char_count,
            receipt.raw_line_count,
            receipt.block_count,
            receipt.trivia_span_count,
            receipt.total_proxy_token_count,
            receipt.receipt_hash,
            json.dumps(receipt.warning_code_counts, sort_keys=True),
            now,
            now,
        ),
    )


def _insert_raw_lines(conn: sqlite3.Connection, result: SegmentationResult) -> None:
    for line in result.lines:
        conn.execute(
            """
            insert into raw_lines(
                source_document_id_hash, line_index, char_start, char_end,
                content_end, terminator, line_kind
            ) values (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.source.source_document_id_hash,
                line.line_index,
                line.char_start,
                line.char_end,
                line.content_end,
                line.terminator,
                line.line_kind,
            ),
        )


def _insert_natural_blocks(conn: sqlite3.Connection, result: SegmentationResult) -> None:
    for block in result.blocks:
        conn.execute(
            """
            insert into natural_blocks(
                block_id_hash, source_document_id_hash, block_index, block_kind,
                segmentation_policy_hash, content_char_start, content_char_end,
                owned_char_start, owned_char_end, start_line_index, end_line_index_exclusive,
                raw_text_sha256, normalized_text_sha256, proxy_token_count, warning_codes_json
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                block.block_id_hash,
                block.source_document_id_hash,
                block.block_index,
                block.block_kind,
                block.segmentation_policy_hash,
                block.content_char_start,
                block.content_char_end,
                block.owned_char_start,
                block.owned_char_end,
                block.start_line_index,
                block.end_line_index_exclusive,
                block.raw_text_sha256,
                block.normalized_text_sha256,
                block.proxy_token_count,
                json.dumps(list(block.warning_codes), sort_keys=True),
            ),
        )


def _insert_trivia_spans(conn: sqlite3.Connection, result: SegmentationResult) -> None:
    for trivia in result.trivia_spans:
        conn.execute(
            """
            insert into natural_trivia_spans(
                source_document_id_hash, char_start, char_end, start_line_index,
                end_line_index_exclusive, trivia_kind, left_block_index, right_block_index
            ) values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trivia.source_document_id_hash,
                trivia.char_start,
                trivia.char_end,
                trivia.start_line_index,
                trivia.end_line_index_exclusive,
                trivia.trivia_kind,
                trivia.left_block_index,
                trivia.right_block_index,
            ),
        )


def _insert_segmentation_receipt(conn: sqlite3.Connection, result: SegmentationResult) -> None:
    receipt = result.receipt
    conn.execute(
        """
        insert into segmentation_receipts(
            source_document_id_hash, receipt_hash, document_status, raw_char_count,
            raw_line_count, block_count, trivia_span_count, total_proxy_token_count,
            receipt_json
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.source.source_document_id_hash,
            receipt.receipt_hash,
            receipt.document_status,
            receipt.raw_char_count,
            receipt.raw_line_count,
            receipt.block_count,
            receipt.trivia_span_count,
            receipt.total_proxy_token_count,
            json.dumps(result.receipt.__dict__, sort_keys=True),
        ),
    )


def _validate_future_lane_tables_empty(
    conn: sqlite3.Connection,
    *,
    tables: tuple[str, ...] = FUTURE_LANE_TABLES,
) -> None:
    for table in tables:
        count = int(conn.execute(f"select count(*) from {table}").fetchone()[0])
        if count:
            raise ValueError(f"future lane table must be empty before readiness: {table}")


def _validate_filtering_lane_rows(conn: sqlite3.Connection) -> None:
    required_columns = {
        "classifier_candidates": {
            "candidate_id_hash",
            "source_document_id_hash",
            "segmentation_receipt_hash",
            "candidate_policy_hash",
            "core_block_indexes_json",
            "candidate_json",
            "status",
        },
        "chunk_decisions": {
            "decision_id_hash",
            "candidate_id_hash",
            "source_document_id_hash",
            "decision_json",
            "status",
        },
        "gap_events": {
            "gap_event_id_hash",
            "candidate_id_hash",
            "source_document_id_hash",
            "block_indexes_json",
            "gap_json",
            "status",
        },
        "clean_runs": {
            "clean_run_id_hash",
            "source_document_id_hash",
            "block_indexes_json",
            "clean_run_json",
            "status",
        },
    }
    for table, columns in required_columns.items():
        missing = columns - _sqlite_column_names(conn, table)
        if missing:
            raise ValueError(f"filtering lane table schema is missing columns in {table}: {sorted(missing)}")

    candidate_rows = list(conn.execute("select * from classifier_candidates order by candidate_id_hash"))
    decision_rows = list(conn.execute("select * from chunk_decisions order by decision_id_hash"))
    gap_rows = list(conn.execute("select * from gap_events order by gap_event_id_hash"))
    clean_run_rows = list(conn.execute("select * from clean_runs order by clean_run_id_hash"))
    if not candidate_rows and not decision_rows and not gap_rows and not clean_run_rows:
        return
    if not candidate_rows:
        raise ValueError("filtering lane decisions require classifier candidates")

    candidates_by_id: dict[str, sqlite3.Row] = {}
    candidate_blocks_by_source: dict[str, list[int]] = {}
    candidate_receipts_by_source: dict[str, set[str]] = {}
    for row in candidate_rows:
        candidate_id = str(row["candidate_id_hash"])
        _require_sha256(candidate_id, "candidate_id_hash")
        if candidate_id in candidates_by_id:
            raise ValueError("duplicate classifier candidate id")
        source_id = str(row["source_document_id_hash"])
        _require_sha256(source_id, "candidate source_document_id_hash")
        receipt_hash = str(row["segmentation_receipt_hash"])
        _require_sha256(receipt_hash, "candidate segmentation_receipt_hash")
        _require_sha256(str(row["candidate_policy_hash"]), "candidate_policy_hash")
        source_receipt = conn.execute(
            "select segmentation_receipt_hash from source_documents where source_document_id_hash = ?",
            (source_id,),
        ).fetchone()
        if source_receipt is None:
            raise ValueError("classifier candidate source document missing")
        if str(source_receipt["segmentation_receipt_hash"]) != receipt_hash:
            raise ValueError("classifier candidate segmentation receipt drift")
        blocks = _json_int_list(str(row["core_block_indexes_json"]), "candidate core blocks")
        if not blocks:
            raise ValueError("classifier candidate must cover at least one core block")
        payload = _json_object(str(row["candidate_json"]), "candidate_json")
        if payload.get("candidate_id_hash") != candidate_id:
            raise ValueError("classifier candidate JSON id mismatch")
        candidates_by_id[candidate_id] = row
        candidate_blocks_by_source.setdefault(source_id, []).extend(blocks)
        candidate_receipts_by_source.setdefault(source_id, set()).add(receipt_hash)

    decisions_by_candidate: dict[str, sqlite3.Row] = {}
    for row in decision_rows:
        decision_id = str(row["decision_id_hash"])
        _require_sha256(decision_id, "decision_id_hash")
        candidate_id = str(row["candidate_id_hash"])
        _require_sha256(candidate_id, "decision candidate_id_hash")
        if candidate_id not in candidates_by_id:
            raise ValueError("chunk decision references missing candidate")
        if candidate_id in decisions_by_candidate:
            raise ValueError("duplicate chunk decision for candidate")
        status = str(row["status"])
        if status not in {"safe", "reject", "quarantine", "error"}:
            raise ValueError("unsupported chunk decision status")
        payload = _json_object(str(row["decision_json"]), "decision_json")
        if payload.get("decision_id_hash") != decision_id or payload.get("candidate_id_hash") != candidate_id:
            raise ValueError("chunk decision JSON identity mismatch")
        if payload.get("classifier_decision") != status:
            raise ValueError("chunk decision JSON status mismatch")
        decisions_by_candidate[candidate_id] = row
    if set(decisions_by_candidate) != set(candidates_by_id):
        raise ValueError("filtering lane candidate/decision coverage mismatch")

    gap_blocks_by_source: dict[str, set[int]] = {}
    for row in gap_rows:
        gap_id = str(row["gap_event_id_hash"])
        _require_sha256(gap_id, "gap_event_id_hash")
        candidate_id = str(row["candidate_id_hash"])
        if candidate_id not in candidates_by_id:
            raise ValueError("gap event references missing candidate")
        status = str(row["status"])
        if status not in {"reject", "quarantine", "error"}:
            raise ValueError("gap event status must be reject, quarantine, or error")
        blocks = _json_int_list(str(row["block_indexes_json"]), "gap blocks")
        payload = _json_object(str(row["gap_json"]), "gap_json")
        if payload.get("gap_event_id_hash") != gap_id or payload.get("candidate_id_hash") != candidate_id:
            raise ValueError("gap event JSON identity mismatch")
        source_id = str(row["source_document_id_hash"])
        gap_blocks_by_source.setdefault(source_id, set()).update(blocks)

    clean_blocks_by_source: dict[str, set[int]] = {}
    for row in clean_run_rows:
        clean_run_id = str(row["clean_run_id_hash"])
        _require_sha256(clean_run_id, "clean_run_id_hash")
        if str(row["status"]) != "safe":
            raise ValueError("clean run status must be safe")
        blocks = _json_int_list(str(row["block_indexes_json"]), "clean run blocks")
        if not blocks:
            raise ValueError("clean run must cover at least one block")
        if blocks != list(range(blocks[0], blocks[-1] + 1)):
            raise ValueError("clean run blocks must be contiguous")
        payload = _json_object(str(row["clean_run_json"]), "clean_run_json")
        if payload.get("clean_run_id_hash") != clean_run_id:
            raise ValueError("clean run JSON identity mismatch")
        source_id = str(row["source_document_id_hash"])
        existing = clean_blocks_by_source.setdefault(source_id, set())
        if existing & set(blocks):
            raise ValueError("duplicate clean run block coverage")
        existing.update(blocks)

    for source_id, candidate_blocks in candidate_blocks_by_source.items():
        block_rows = list(
            conn.execute(
                "select block_index from natural_blocks where source_document_id_hash = ? order by block_index",
                (source_id,),
            )
        )
        expected = [int(row["block_index"]) for row in block_rows]
        if sorted(candidate_blocks) != expected:
            raise ValueError("filtering lane candidate core block coverage mismatch")
        if len(candidate_blocks) != len(set(candidate_blocks)):
            raise ValueError("filtering lane candidate core blocks overlap")
        if len(candidate_receipts_by_source[source_id]) != 1:
            raise ValueError("filtering lane candidates mix segmentation receipts")
        clean_blocks = clean_blocks_by_source.get(source_id, set())
        gap_blocks = gap_blocks_by_source.get(source_id, set())
        if clean_blocks & gap_blocks:
            raise ValueError("clean runs cross classifier gap blocks")
        if clean_blocks | gap_blocks != set(expected):
            raise ValueError("filtering lane clean run/gap block coverage mismatch")


def _sqlite_column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"pragma table_info({table})")}


def _json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _json_int_list(raw: str, label: str) -> list[int]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON list")
    values: list[int] = []
    for value in payload:
        integer = _require_nonnegative_int(value, label)
        values.append(integer)
    if values != sorted(values):
        raise ValueError(f"{label} must be sorted")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicates")
    return values


def _validate_no_orphan_rows(conn: sqlite3.Connection) -> None:
    checks = {
        "raw_lines": "source_document_id_hash",
        "natural_blocks": "source_document_id_hash",
        "natural_trivia_spans": "source_document_id_hash",
        "segmentation_receipts": "source_document_id_hash",
        "phase_receipts": "source_document_id_hash",
    }
    for table, field in checks.items():
        count = int(
            conn.execute(
                f"""
                select count(*)
                from {table} child
                left join source_documents parent
                    on child.{field} = parent.source_document_id_hash
                where parent.source_document_id_hash is null
                """
            ).fetchone()[0]
        )
        if count:
            raise ValueError(f"chunked state contains orphan rows in {table}")


def _validate_span_coverage(spans: list[tuple[int, int, str]], expected_end: int) -> None:
    cursor = 0
    for start, end, label in sorted(spans):
        if start != cursor:
            raise ValueError(f"state coverage gap before {label}")
        if end <= start:
            raise ValueError(f"state coverage empty span for {label}")
        cursor = end
    if cursor != expected_end:
        raise ValueError("state coverage does not reach raw char count")


def _validate_terminal_source_row(
    row: sqlite3.Row,
    block_rows: list[sqlite3.Row],
    trivia_rows: list[sqlite3.Row],
    line_rows: list[sqlite3.Row],
) -> None:
    if block_rows or trivia_rows or line_rows:
        raise ValueError("terminal source document has block, trivia, or raw-line rows")
    for field in ("raw_line_count", "block_count", "trivia_span_count", "total_proxy_token_count"):
        if _require_nonnegative_int(row[field], field) != 0:
            raise ValueError(f"terminal source document {field} must be zero")
    try:
        warning_counts = json.loads(str(row["warning_codes_json"]))
    except json.JSONDecodeError as exc:
        raise ValueError("terminal source warning JSON is invalid") from exc
    if warning_counts != {}:
        raise ValueError("terminal source document warning counts must be empty")
    status = str(row["status"])
    if status in {"missing_text", "non_string_text"}:
        if row["raw_text_sha256"] is not None or row["normalized_text_sha256"] is not None:
            raise ValueError("missing/non-string terminal source must not carry text hashes")
        if _require_nonnegative_int(row["raw_char_count"], "raw_char_count") != 0:
            raise ValueError("missing/non-string terminal source raw_char_count must be zero")
    elif status == "empty_or_whitespace":
        _require_sha256(str(row["raw_text_sha256"]), "raw_text_sha256")
        _require_sha256(str(row["normalized_text_sha256"]), "normalized_text_sha256")
        _require_nonnegative_int(row["raw_char_count"], "raw_char_count")
    else:
        raise ValueError("unsupported terminal source document status")


def _validate_raw_line_rows(line_rows: list[sqlite3.Row], raw_char_count: int) -> None:
    cursor = 0
    indexes: list[int] = []
    for row in line_rows:
        index = _require_nonnegative_int(row["line_index"], "line_index")
        indexes.append(index)
        char_start = _require_nonnegative_int(row["char_start"], "line_char_start")
        char_end = _require_nonnegative_int(row["char_end"], "line_char_end")
        content_end = _require_nonnegative_int(row["content_end"], "line_content_end")
        if char_start != cursor:
            raise ValueError("raw line rows must cover raw text contiguously")
        if not char_start < char_end <= raw_char_count:
            raise ValueError("raw line row span is invalid")
        if not char_start <= content_end <= char_end:
            raise ValueError("raw line content_end is outside line span")
        terminator = str(row["terminator"])
        if terminator and terminator not in (TERMINATORS | {"\r\n"}):
            raise ValueError("raw line terminator is unsupported")
        if char_end - content_end != len(terminator):
            raise ValueError("raw line terminator/content_end mismatch")
        if not terminator and content_end != char_end:
            raise ValueError("raw line missing terminator with shortened content")
        if str(row["line_kind"]) not in _RAW_LINE_KINDS:
            raise ValueError("raw line kind must be non-empty")
        cursor = char_end
    if indexes != list(range(len(line_rows))):
        raise ValueError("raw line indexes must be contiguous")
    if cursor != raw_char_count:
        raise ValueError("raw line rows do not reach raw char count")


def _validate_state_block_order(block_rows: list[sqlite3.Row]) -> None:
    ordered_by_span = sorted(block_rows, key=lambda row: (int(row["owned_char_start"]), int(row["owned_char_end"])))
    if [int(row["block_index"]) for row in ordered_by_span] != list(range(len(block_rows))):
        raise ValueError("state natural block indexes must match span order")


def _validate_block_row(
    row: sqlite3.Row,
    *,
    expected_policy_hash: str,
    line_rows: list[sqlite3.Row],
) -> None:
    _require_sha256(str(row["block_id_hash"]), "block_id_hash")
    _require_sha256(str(row["source_document_id_hash"]), "source_document_id_hash")
    if str(row["block_kind"]) not in BLOCK_KINDS:
        raise ValueError("state contains unsupported natural block kind")
    if str(row["segmentation_policy_hash"]) != expected_policy_hash:
        raise ValueError("state natural block segmentation policy mismatch")
    _require_sha256(str(row["segmentation_policy_hash"]), "segmentation_policy_hash")
    owned_start = _require_nonnegative_int(row["owned_char_start"], "owned_char_start")
    owned_end = _require_nonnegative_int(row["owned_char_end"], "owned_char_end")
    content_start = _require_nonnegative_int(row["content_char_start"], "content_char_start")
    content_end = _require_nonnegative_int(row["content_char_end"], "content_char_end")
    if not owned_start < owned_end:
        raise ValueError("state natural block owned span is empty")
    if not owned_start <= content_start < content_end <= owned_end:
        raise ValueError("state natural block content span must be non-empty and inside owned span")
    start_line_index, end_line_index_exclusive = _validate_state_line_range(
        row,
        line_rows,
        span_start=owned_start,
        span_end=owned_end,
        field_prefix="state natural block",
        require_exact_boundaries=str(row["block_kind"]) != "oversized_soft_wrapped_prose",
    )
    try:
        warning_codes = json.loads(str(row["warning_codes_json"]))
    except json.JSONDecodeError as exc:
        raise ValueError("state natural block warning JSON is invalid") from exc
    if not isinstance(warning_codes, list) or not all(isinstance(code, str) for code in warning_codes):
        raise ValueError("state natural block warning codes must be a string list")
    block = NaturalBlockRecord(
        block_id_hash=str(row["block_id_hash"]),
        source_document_id_hash=str(row["source_document_id_hash"]),
        segmentation_policy_hash=str(row["segmentation_policy_hash"]),
        block_index=_require_nonnegative_int(row["block_index"], "block_index"),
        block_kind=str(row["block_kind"]),
        content_char_start=content_start,
        content_char_end=content_end,
        owned_char_start=owned_start,
        owned_char_end=owned_end,
        start_line_index=start_line_index,
        end_line_index_exclusive=end_line_index_exclusive,
        raw_text_sha256=_require_sha256(str(row["raw_text_sha256"]), "raw_text_sha256"),
        normalized_text_sha256=_require_sha256(str(row["normalized_text_sha256"]), "normalized_text_sha256"),
        proxy_token_count=_require_nonnegative_int(row["proxy_token_count"], "proxy_token_count"),
        warning_codes=tuple(sorted(set(warning_codes))),
    )
    if natural_block_id_hash(block) != str(row["block_id_hash"]):
        raise ValueError("state natural block id hash mismatch")


def _validate_trivia_row(row: sqlite3.Row, *, line_rows: list[sqlite3.Row]) -> None:
    start = _require_nonnegative_int(row["char_start"], "char_start")
    end = _require_nonnegative_int(row["char_end"], "char_end")
    if not start < end:
        raise ValueError("state trivia span is empty")
    _validate_state_line_range(
        row,
        line_rows,
        span_start=start,
        span_end=end,
        field_prefix="state trivia",
        require_exact_boundaries=True,
    )
    if str(row["trivia_kind"]) != "blank_separator":
        raise ValueError("state contains unsupported trivia kind")


def _validate_state_line_range(
    row: sqlite3.Row,
    line_rows: list[sqlite3.Row],
    *,
    span_start: int,
    span_end: int,
    field_prefix: str,
    require_exact_boundaries: bool,
) -> tuple[int, int]:
    start_line_index = _require_nonnegative_int(row["start_line_index"], "start_line_index")
    end_line_index_exclusive = _require_nonnegative_int(
        row["end_line_index_exclusive"],
        "end_line_index_exclusive",
    )
    if not start_line_index < end_line_index_exclusive <= len(line_rows):
        raise ValueError(f"{field_prefix} line range is invalid")
    line_start = int(line_rows[start_line_index]["char_start"])
    line_end = int(line_rows[end_line_index_exclusive - 1]["char_end"])
    if not line_start <= span_start < span_end <= line_end:
        raise ValueError(f"{field_prefix} line range does not contain span")
    minimal_start = next(
        index
        for index, line in enumerate(line_rows)
        if int(line["char_start"]) <= span_start < int(line["char_end"])
    )
    minimal_end = next(
        index + 1
        for index, line in enumerate(line_rows)
        if int(line["char_start"]) < span_end <= int(line["char_end"])
    )
    if start_line_index != minimal_start or end_line_index_exclusive != minimal_end:
        raise ValueError(f"{field_prefix} line range is not minimal")
    if require_exact_boundaries and (span_start != line_start or span_end != line_end):
        raise ValueError(f"{field_prefix} line range does not match line boundaries")
    return start_line_index, end_line_index_exclusive


def _validate_state_trivia_adjacency(
    block_rows: list[sqlite3.Row],
    trivia_rows: list[sqlite3.Row],
    raw_char_count: int,
) -> None:
    blocks_by_start = {int(row["owned_char_start"]): row for row in block_rows}
    blocks_by_end = {int(row["owned_char_end"]): row for row in block_rows}
    for trivia in trivia_rows:
        char_start = int(trivia["char_start"])
        char_end = int(trivia["char_end"])
        left_block = blocks_by_end.get(char_start)
        right_block = blocks_by_start.get(char_end)
        expected_left = int(left_block["block_index"]) if left_block is not None else None
        expected_right = int(right_block["block_index"]) if right_block is not None else None
        if expected_left is None and char_start != 0:
            raise ValueError("state trivia span is not adjacent to a left block")
        if expected_right is None and char_end != raw_char_count:
            raise ValueError("state trivia span is not adjacent to a right block")
        if _nullable_int(trivia["left_block_index"]) != expected_left:
            raise ValueError("state trivia left block index mismatch")
        if _nullable_int(trivia["right_block_index"]) != expected_right:
            raise ValueError("state trivia right block index mismatch")
        if expected_left is not None and expected_right is not None and expected_right != expected_left + 1:
            raise ValueError("state trivia adjacency must bridge consecutive blocks")


def _validate_receipt_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        payload = json.loads(str(row["receipt_json"]))
    except json.JSONDecodeError as exc:
        raise ValueError("state segmentation receipt JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("state segmentation receipt JSON must be an object")
    if set(payload) != _RECEIPT_PAYLOAD_KEYS:
        raise ValueError("state segmentation receipt JSON has unsupported fields")
    if payload.get("receipt_hash") != str(row["receipt_hash"]):
        raise ValueError("state segmentation receipt hash mismatch")
    if payload.get("source_document_id_hash") != str(row["source_document_id_hash"]):
        raise ValueError("state segmentation receipt source id mismatch")
    if payload.get("document_status") != str(row["document_status"]):
        raise ValueError("state segmentation receipt status mismatch")
    for field in (
        "raw_char_count",
        "raw_line_count",
        "block_count",
        "trivia_span_count",
        "total_proxy_token_count",
    ):
        if _require_nonnegative_int(payload.get(field), field) != _require_nonnegative_int(row[field], field):
            raise ValueError(f"state segmentation receipt {field} mismatch")
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if payload.get(field) is not False:
            raise ValueError(f"state segmentation receipt must keep {field}=false")
    for field in _RECEIPT_TRUE_FIELDS:
        if payload.get(field) is not True:
            raise ValueError(f"state segmentation receipt must keep {field}=true")
    stable = dict(payload)
    receipt_hash = stable.pop("receipt_hash", None)
    if receipt_hash != stable_json_hash(stable):
        raise ValueError("state segmentation receipt hash mismatch")
    return payload


def _validate_source_receipt_consistency(
    source_row: sqlite3.Row,
    receipt_row: sqlite3.Row,
    receipt_payload: dict[str, Any],
    metadata: dict[str, str],
) -> None:
    if str(source_row["segmentation_receipt_hash"]) != str(receipt_row["receipt_hash"]):
        raise ValueError("source document segmentation receipt hash mismatch")
    comparisons = {
        "source_document_id_hash": str(source_row["source_document_id_hash"]),
        "split": str(source_row["split"]),
        "document_status": str(source_row["status"]),
        "raw_text_sha256": _nullable(source_row["raw_text_sha256"]),
        "normalized_text_sha256": _nullable(source_row["normalized_text_sha256"]),
        "source_lineage_hash": _nullable(source_row["source_lineage_hash"]),
        "source_token_count_hint": _nullable_int(source_row["source_token_count_hint"]),
        "raw_char_count": _require_nonnegative_int(source_row["raw_char_count"], "raw_char_count"),
        "raw_line_count": _require_nonnegative_int(source_row["raw_line_count"], "raw_line_count"),
        "block_count": _require_nonnegative_int(source_row["block_count"], "block_count"),
        "trivia_span_count": _require_nonnegative_int(source_row["trivia_span_count"], "trivia_span_count"),
        "total_proxy_token_count": _require_nonnegative_int(
            source_row["total_proxy_token_count"],
            "total_proxy_token_count",
        ),
        "segmentation_policy_version": metadata["segmentation_policy_version"],
        "segmentation_policy_hash": metadata["segmentation_policy_hash"],
        "raw_line_policy_hash": metadata["raw_line_policy_hash"],
        "normalizer_version": metadata["normalizer_version"],
        "proxy_token_estimator_version": metadata["proxy_token_estimator_version"],
        "block_id_policy_hash": metadata["block_id_policy_hash"],
    }
    for field, expected in comparisons.items():
        if receipt_payload.get(field) != expected:
            raise ValueError(f"source document and receipt disagree on {field}")
    try:
        source_warning_counts = json.loads(str(source_row["warning_codes_json"]))
    except json.JSONDecodeError as exc:
        raise ValueError("source document warning JSON is invalid") from exc
    if source_warning_counts != receipt_payload.get("warning_code_counts"):
        raise ValueError("source document warning counts do not match receipt")


def _validate_block_receipt_consistency(block_rows: list[sqlite3.Row], receipt_payload: dict[str, Any]) -> None:
    block_kind_counts = Counter(str(row["block_kind"]) for row in block_rows)
    warning_counts: Counter[str] = Counter()
    proxy_total = 0
    for row in block_rows:
        warning_counts.update(json.loads(str(row["warning_codes_json"])))
        proxy_total += _require_nonnegative_int(row["proxy_token_count"], "proxy_token_count")
    if dict(sorted(block_kind_counts.items())) != receipt_payload.get("block_kind_counts"):
        raise ValueError("state natural block kind counts do not match receipt")
    if dict(sorted(warning_counts.items())) != receipt_payload.get("warning_code_counts"):
        raise ValueError("state natural block warning counts do not match receipt")
    if proxy_total != receipt_payload.get("total_proxy_token_count"):
        raise ValueError("state natural block proxy total does not match receipt")


def _validate_phase_receipts(conn: sqlite3.Connection, source_row: sqlite3.Row) -> None:
    source_id = str(source_row["source_document_id_hash"])
    rows = list(
        conn.execute(
            "select phase_name, receipt_hash from phase_receipts where source_document_id_hash = ?",
            (source_id,),
        )
    )
    if len(rows) != 1:
        raise ValueError("source document must have exactly one phase receipt")
    row = rows[0]
    if str(row["phase_name"]) != "natural_block_segmentation":
        raise ValueError("source document has unknown phase receipt")
    if str(row["receipt_hash"]) != str(source_row["segmentation_receipt_hash"]):
        raise ValueError("source document phase receipt hash mismatch")


def _validate_existing_document_matches_result(conn: sqlite3.Connection, result: SegmentationResult) -> None:
    source_id = result.source.source_document_id_hash
    block_rows = list(
        conn.execute(
            "select * from natural_blocks where source_document_id_hash = ? order by block_index",
            (source_id,),
        )
    )
    line_rows = list(
        conn.execute(
            "select * from raw_lines where source_document_id_hash = ? order by line_index",
            (source_id,),
        )
    )
    trivia_rows = list(
        conn.execute(
            "select * from natural_trivia_spans where source_document_id_hash = ? order by char_start",
            (source_id,),
        )
    )
    receipt = conn.execute(
        "select * from segmentation_receipts where source_document_id_hash = ?",
        (source_id,),
    ).fetchone()
    source_row = conn.execute(
        "select * from source_documents where source_document_id_hash = ?",
        (source_id,),
    ).fetchone()
    if source_row is None or receipt is None:
        raise ValueError("existing segmentation source or receipt is missing")
    if len(line_rows) != len(result.lines):
        raise ValueError("existing segmentation raw-line rows drifted")
    if len(block_rows) != len(result.blocks) or len(trivia_rows) != len(result.trivia_spans):
        raise ValueError("existing segmentation child rows drifted")
    for row, line in zip(line_rows, result.lines, strict=True):
        stored = {
            "line_index": _require_nonnegative_int(row["line_index"], "line_index"),
            "char_start": _require_nonnegative_int(row["char_start"], "line_char_start"),
            "char_end": _require_nonnegative_int(row["char_end"], "line_char_end"),
            "content_end": _require_nonnegative_int(row["content_end"], "line_content_end"),
            "terminator": str(row["terminator"]),
            "line_kind": str(row["line_kind"]),
        }
        expected = {
            "line_index": line.line_index,
            "char_start": line.char_start,
            "char_end": line.char_end,
            "content_end": line.content_end,
            "terminator": line.terminator,
            "line_kind": line.line_kind,
        }
        if stored != expected:
            raise ValueError("existing segmentation raw-line rows drifted")
    for row, block in zip(block_rows, result.blocks, strict=True):
        stored = {
            "block_id_hash": str(row["block_id_hash"]),
            "segmentation_policy_hash": str(row["segmentation_policy_hash"]),
            "block_index": _require_nonnegative_int(row["block_index"], "block_index"),
            "block_kind": str(row["block_kind"]),
            "content_char_start": _require_nonnegative_int(row["content_char_start"], "content_char_start"),
            "content_char_end": _require_nonnegative_int(row["content_char_end"], "content_char_end"),
            "owned_char_start": _require_nonnegative_int(row["owned_char_start"], "owned_char_start"),
            "owned_char_end": _require_nonnegative_int(row["owned_char_end"], "owned_char_end"),
            "start_line_index": _require_nonnegative_int(row["start_line_index"], "start_line_index"),
            "end_line_index_exclusive": _require_nonnegative_int(
                row["end_line_index_exclusive"],
                "end_line_index_exclusive",
            ),
            "raw_text_sha256": str(row["raw_text_sha256"]),
            "normalized_text_sha256": str(row["normalized_text_sha256"]),
            "proxy_token_count": _require_nonnegative_int(row["proxy_token_count"], "proxy_token_count"),
            "warning_codes": tuple(json.loads(str(row["warning_codes_json"]))),
        }
        expected = {
            "block_id_hash": block.block_id_hash,
            "segmentation_policy_hash": block.segmentation_policy_hash,
            "block_index": block.block_index,
            "block_kind": block.block_kind,
            "content_char_start": block.content_char_start,
            "content_char_end": block.content_char_end,
            "owned_char_start": block.owned_char_start,
            "owned_char_end": block.owned_char_end,
            "start_line_index": block.start_line_index,
            "end_line_index_exclusive": block.end_line_index_exclusive,
            "raw_text_sha256": block.raw_text_sha256,
            "normalized_text_sha256": block.normalized_text_sha256,
            "proxy_token_count": block.proxy_token_count,
            "warning_codes": block.warning_codes,
        }
        if stored != expected:
            raise ValueError("existing segmentation child rows drifted")
    for row, trivia in zip(trivia_rows, result.trivia_spans, strict=True):
        stored = {
            "char_start": _require_nonnegative_int(row["char_start"], "char_start"),
            "char_end": _require_nonnegative_int(row["char_end"], "char_end"),
            "start_line_index": _require_nonnegative_int(row["start_line_index"], "start_line_index"),
            "end_line_index_exclusive": _require_nonnegative_int(
                row["end_line_index_exclusive"],
                "end_line_index_exclusive",
            ),
            "trivia_kind": str(row["trivia_kind"]),
            "left_block_index": _nullable_int(row["left_block_index"]),
            "right_block_index": _nullable_int(row["right_block_index"]),
        }
        expected = {
            "char_start": trivia.char_start,
            "char_end": trivia.char_end,
            "start_line_index": trivia.start_line_index,
            "end_line_index_exclusive": trivia.end_line_index_exclusive,
            "trivia_kind": trivia.trivia_kind,
            "left_block_index": trivia.left_block_index,
            "right_block_index": trivia.right_block_index,
        }
        if stored != expected:
            raise ValueError("existing segmentation child rows drifted")
    _validate_receipt_row(receipt)
    _validate_phase_receipts(conn, source_row)


def _sqlite_aggregate_hash(conn: sqlite3.Connection) -> str:
    digest = sha256()
    _hash_stream_item(digest, "metadata", _metadata_dict(conn))
    for row in conn.execute("select * from source_documents order by source_document_id_hash"):
        _hash_stream_item(
            digest,
            "source_documents",
            {key: row[key] for key in row.keys() if key not in {"created_at", "updated_at"}},
        )
    for row in conn.execute("select * from raw_lines order by source_document_id_hash, line_index"):
        _hash_stream_item(digest, "raw_lines", dict(row))
    for row in conn.execute("select * from natural_blocks order by source_document_id_hash, block_index"):
        _hash_stream_item(digest, "natural_blocks", dict(row))
    for row in conn.execute("select * from natural_trivia_spans order by source_document_id_hash, char_start"):
        _hash_stream_item(digest, "natural_trivia_spans", dict(row))
    for row in conn.execute("select * from segmentation_receipts order by source_document_id_hash"):
        _hash_stream_item(
            digest,
            "segmentation_receipts",
            {
                **{key: row[key] for key in row.keys() if key != "receipt_json"},
                "receipt_payload": json.loads(str(row["receipt_json"])),
            },
        )
    for row in conn.execute(
        "select phase_name, source_document_id_hash, receipt_hash "
        "from phase_receipts order by source_document_id_hash, phase_name"
    ):
        _hash_stream_item(digest, "phase_receipts", dict(row))
    return digest.hexdigest()


def _hash_stream_item(digest: Any, table: str, payload: dict[str, Any]) -> None:
    digest.update(table.encode("utf-8"))
    digest.update(b"\0")
    digest.update(stable_json_hash(payload).encode("ascii"))
    digest.update(b"\n")


def _sql_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"select count(*) from {table}").fetchone()[0])


def _sql_count_by(conn: sqlite3.Connection, table: str, field: str) -> Counter[str]:
    return Counter(
        {
            str(row["value"]): int(row["count"])
            for row in conn.execute(
                f"select {field} as value, count(*) as count from {table} group by {field}"
            )
        }
    )


def _metadata_dict(conn: sqlite3.Connection) -> dict[str, str]:
    return {str(row["key"]): str(row["value"]) for row in conn.execute("select key, value from metadata")}


def _proxy_bucket(value: int) -> str:
    _require_nonnegative_int(value, "proxy_token_count")
    if value == 0:
        return "0"
    if value <= 128:
        return "1_128"
    if value <= 512:
        return "129_512"
    if value <= 1024:
        return "513_1024"
    return "1025_plus"


def _validate_config(config: ChunkedCleanRunStateConfig) -> ChunkedCleanRunStateConfig:
    run_dir = Path(config.run_dir)
    for value, field in (
        (config.source_manifest_hash, "source_manifest_hash"),
        (config.source_schedule_hash, "source_schedule_hash"),
    ):
        if value is not None:
            _require_sha256(value, field)
    return ChunkedCleanRunStateConfig(
        run_dir=run_dir,
        source_manifest_hash=config.source_manifest_hash,
        source_schedule_hash=config.source_schedule_hash,
    )


def _nullable(value: Any) -> str | None:
    return None if value is None else str(value)


def _nullable_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _none_to_hash(value: str | None) -> str | None:
    return None if value in {None, "<none>"} else value


def _require_sha256(value: str | None, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase sha256")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)
