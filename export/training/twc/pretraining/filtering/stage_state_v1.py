"""Exact SQLite ownership for the filtering stage-state v1 database.

The handle remains connection-like during the staged migration, while stable
prefilter and hot W1 operations live here so callers do not learn schema,
locking, permission, PRAGMA, metadata-binding, or replay rules. This module
owns no filtering policy and no other project database.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
from typing import Any

from twc.repro.hashing import stable_json_hash
from twc.pretraining.filtering import artifacts as filtering_artifacts


STAGE_STATE_FILENAME = "filtering_stage_state.sqlite"
_PRIVATE_FILE_MODE = 0o600
PREFILTER_TERMINAL_ROW_STATUSES = frozenset(
    {"prefiltered", "segmentation_error", "fetch_error", "non_mapping_row", "missing_text"}
)
PREFILTER_ROUTES = frozenset(
    {
        "deterministic_reject",
        "s1_reject",
        "prefilter_survivor",
        "prefilter_error",
        "prefilter_quarantine",
    }
)
_SOURCE_ROW_IMMUTABLE_FIELDS = (
    "source_schedule_hash",
    "schedule_entry_hash",
    "source_file_id_hash",
    "row_group_index",
    "row_group_local_offset",
    "source_document_id_hash",
)


@dataclass(frozen=True)
class PrefilterSurvivorInput:
    candidate_id_hash: str
    candidate: dict[str, Any]
    deterministic_result: dict[str, Any]
    s1_score: float | None
    private_evidence_row_id: str
    evidence_text_sha256: str
    private_evidence_text: str


@dataclass(frozen=True)
class W1InputRecord:
    candidate_json: str | None
    evidence_candidate_id_hash: str | None
    evidence_text_sha256: str | None
    private_evidence_text: str | None


@dataclass(frozen=True)
class CloseoutSurvivorFact:
    candidate_id_hash: str
    source_document_id_hash: str
    segmentation_receipt_hash: str
    core_text_sha256: str
    core_char_count: int
    core_byte_count: int
    core_proxy_token_count: int
    first_block_index: int
    last_block_index: int
    core_char_start: int
    core_char_end: int
    evidence_text_sha256: str
    materialized_score_payload: dict[str, Any] | None

_TABLE_COLUMNS: dict[str, tuple[tuple[str, str, int, str | None, int], ...]] = {
    "stage_metadata": (
        ("key", "TEXT", 0, None, 1),
        ("value_json", "TEXT", 1, None, 0),
    ),
    "prefilter_source_rows": (
        ("physical_row_key_hash", "TEXT", 0, None, 1),
        ("source_schedule_hash", "TEXT", 0, None, 0),
        ("schedule_entry_hash", "TEXT", 1, None, 0),
        ("source_file_id_hash", "TEXT", 1, None, 0),
        ("row_group_index", "INTEGER", 1, None, 0),
        ("row_group_local_offset", "INTEGER", 1, None, 0),
        ("source_document_id_hash", "TEXT", 1, None, 0),
        ("row_status", "TEXT", 1, None, 0),
        ("error_class", "TEXT", 0, None, 0),
        ("segmentation_receipt_hash", "TEXT", 0, None, 0),
        ("candidate_count", "INTEGER", 1, "0", 0),
        ("survivor_count", "INTEGER", 1, "0", 0),
        ("row_receipt_hash", "TEXT", 1, None, 0),
        ("created_at", "REAL", 1, None, 0),
        ("updated_at", "REAL", 1, None, 0),
    ),
    "prefilter_candidates": (
        ("candidate_id_hash", "TEXT", 0, None, 1),
        ("source_document_id_hash", "TEXT", 1, None, 0),
        ("segmentation_receipt_hash", "TEXT", 1, None, 0),
        ("candidate_json", "TEXT", 1, None, 0),
        ("deterministic_result_json", "TEXT", 1, None, 0),
        ("prefilter_route", "TEXT", 1, None, 0),
        ("s1_score", "REAL", 0, None, 0),
        ("created_at", "REAL", 1, None, 0),
    ),
    "private_w1_evidence": (
        ("private_evidence_row_id", "TEXT", 0, None, 1),
        ("candidate_id_hash", "TEXT", 1, None, 0),
        ("evidence_text_sha256", "TEXT", 1, None, 0),
        ("private_evidence_text", "TEXT", 1, None, 0),
    ),
    "w1_scores": (
        ("candidate_id_hash", "TEXT", 0, None, 1),
        ("score_json", "TEXT", 1, None, 0),
        ("created_at", "REAL", 1, None, 0),
    ),
}

_UNIQUE_COLUMNS = {
    "stage_metadata": {("key",)},
    "prefilter_source_rows": {("physical_row_key_hash",)},
    "prefilter_candidates": {("candidate_id_hash",)},
    "private_w1_evidence": {("private_evidence_row_id",), ("candidate_id_hash",)},
    "w1_scores": {("candidate_id_hash",)},
}

_SCHEMA_SQL = """
create table stage_metadata(
    key text primary key,
    value_json text not null
);
create table prefilter_source_rows(
    physical_row_key_hash text primary key,
    source_schedule_hash text,
    schedule_entry_hash text not null,
    source_file_id_hash text not null,
    row_group_index integer not null,
    row_group_local_offset integer not null,
    source_document_id_hash text not null,
    row_status text not null,
    error_class text,
    segmentation_receipt_hash text,
    candidate_count integer not null default 0,
    survivor_count integer not null default 0,
    row_receipt_hash text not null,
    created_at real not null,
    updated_at real not null
);
create table prefilter_candidates(
    candidate_id_hash text primary key,
    source_document_id_hash text not null,
    segmentation_receipt_hash text not null,
    candidate_json text not null,
    deterministic_result_json text not null,
    prefilter_route text not null,
    s1_score real,
    created_at real not null
);
create table private_w1_evidence(
    private_evidence_row_id text primary key,
    candidate_id_hash text not null unique,
    evidence_text_sha256 text not null,
    private_evidence_text text not null
);
create table w1_scores(
    candidate_id_hash text primary key,
    score_json text not null,
    created_at real not null
);
"""

_STORED_CANDIDATE_SELECT = (
    "select c.candidate_id_hash, "
    "c.source_document_id_hash as candidate_source_document_id_hash, "
    "c.segmentation_receipt_hash as candidate_segmentation_receipt_hash, "
    "c.candidate_json, c.deterministic_result_json, c.prefilter_route, c.s1_score, "
    "e.private_evidence_row_id, e.candidate_id_hash as evidence_candidate_id, "
    "e.evidence_text_sha256, e.private_evidence_text "
    "from prefilter_candidates c left join private_w1_evidence e "
    "on e.candidate_id_hash=c.candidate_id_hash"
)


class FilteringStageStateV1:
    """Connection-like owner of one validated filtering-stage database."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        path: Path,
        *,
        writer_lock_fd: int | None,
        immutable_frozen: bool,
    ) -> None:
        self._connection = connection
        self.path = path
        self._writer_lock_fd = writer_lock_fd
        self._immutable_frozen = bool(immutable_frozen)
        self._closed = False

    @property
    def writer_lock_held(self) -> bool:
        return self._writer_lock_fd is not None and not self._closed

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._connection.close()
        finally:
            if self._writer_lock_fd is not None:
                lock_fd = self._writer_lock_fd
                self._writer_lock_fd = None
                try:
                    filtering_artifacts._chmod_private_sqlite_files(self.path)
                finally:
                    os.close(lock_fd)
            self._closed = True

    def __enter__(self) -> FilteringStageStateV1:
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool | None:
        try:
            return self._connection.__exit__(exc_type, exc, traceback)
        finally:
            self.close()

    def bind_prefilter_schedule(
        self,
        input_schedule_hash: str,
        expected_physical_row_count: int,
        expected_physical_row_key_set_hash: str | None,
    ) -> None:
        expected = {
            "input_schedule_hash": filtering_artifacts._require_sha256(
                input_schedule_hash, "input_schedule_hash"
            ),
            "expected_physical_row_count": filtering_artifacts._positive_int(
                expected_physical_row_count, "expected_physical_row_count"
            ),
            "expected_physical_row_key_set_hash": (
                None
                if expected_physical_row_key_set_hash is None
                else filtering_artifacts._require_sha256(
                    expected_physical_row_key_set_hash,
                    "expected_physical_row_key_set_hash",
                )
            ),
        }
        existing = {
            str(row["key"]): json.loads(str(row["value_json"]))
            for row in self._connection.execute(
                "select key, value_json from stage_metadata "
                "where key in ('input_schedule_hash', 'expected_physical_row_count', "
                "'expected_physical_row_key_set_hash')"
            )
        }
        for key, value in expected.items():
            if key in existing and existing[key] != value:
                raise ValueError(f"prefilter-cpu {key} drift")
        pending = [
            (key, json.dumps(value, sort_keys=True))
            for key, value in expected.items()
            if key not in existing
        ]
        if pending:
            with self._connection:
                self._connection.executemany(
                    "insert into stage_metadata(key, value_json) values (?, ?)", pending
                )

    def source_row_status(self, physical_row_key_hash: str) -> str | None:
        physical_key = filtering_artifacts._require_sha256(
            physical_row_key_hash, "physical_row_key_hash"
        )
        row = self._connection.execute(
            "select row_status from prefilter_source_rows where physical_row_key_hash=?",
            (physical_key,),
        ).fetchone()
        return None if row is None else str(row["row_status"])

    def terminal_offsets(self, key_to_offset: Mapping[str, int]) -> set[int]:
        if not key_to_offset:
            return set()
        normalized = {
            filtering_artifacts._require_sha256(key, "physical_row_key_hash"): int(offset)
            for key, offset in key_to_offset.items()
        }
        terminal: set[int] = set()
        keys = list(normalized)
        for start in range(0, len(keys), 500):
            chunk = keys[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            terminal_rows = [
                row
                for row in self._connection.execute(
                f"select physical_row_key_hash, source_document_id_hash, row_status, "
                f"segmentation_receipt_hash, candidate_count, survivor_count "
                f"from prefilter_source_rows "
                f"where physical_row_key_hash in ({placeholders})",
                chunk,
                )
                if str(row["row_status"]) in PREFILTER_TERMINAL_ROW_STATUSES
            ]
            _validate_terminal_source_candidates(self._connection, terminal_rows)
            terminal.update(
                normalized[str(row["physical_row_key_hash"])] for row in terminal_rows
            )
        return terminal

    def store_source_row(self, payload: Mapping[str, Any]) -> None:
        with self._connection:
            _store_source_row_in_transaction(self._connection, payload, now=time.time())

    def candidate_json(self, candidate_id_hash: str) -> dict[str, Any] | None:
        candidate_id = filtering_artifacts._require_sha256(
            candidate_id_hash, "candidate_id_hash"
        )
        row = self._connection.execute(
            _STORED_CANDIDATE_SELECT + " where c.candidate_id_hash=?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            return None
        candidate, _deterministic, _evidence = _validated_stored_candidate(row)
        return candidate

    def store_candidate_with_evidence(
        self,
        candidate_row: tuple[Any, ...],
        evidence_row: tuple[Any, ...] | None,
    ) -> None:
        _validate_candidate_record(candidate_row, evidence_row)
        candidate_id = str(candidate_row[0])
        existing = self._connection.execute(
            "select candidate_id_hash, source_document_id_hash, segmentation_receipt_hash, "
            "candidate_json, deterministic_result_json, prefilter_route, s1_score "
            "from prefilter_candidates where candidate_id_hash=?",
            (candidate_id,),
        ).fetchone()
        if existing is not None:
            _validate_candidate_replay(
                self._connection,
                existing,
                candidate_row=candidate_row,
                evidence_row=evidence_row,
            )
            return
        with self._connection:
            _insert_candidate_records(self._connection, [(candidate_row, evidence_row)])

    def commit_prefilter_batch(
        self,
        source_row_payloads: Sequence[Mapping[str, Any]],
        candidate_records: Sequence[tuple[tuple[Any, ...], tuple[Any, ...] | None]],
    ) -> None:
        now = time.time()
        for record in candidate_records:
            _validate_candidate_record(*record)
        try:
            with self._connection:
                _insert_source_rows_strict(
                    self._connection, source_row_payloads, now=now
                )
                _insert_candidate_records(self._connection, candidate_records)
        except sqlite3.IntegrityError:
            with self._connection:
                for payload in source_row_payloads:
                    _store_source_row_in_transaction(
                        self._connection, payload, now=now
                    )
                _insert_candidate_records(self._connection, candidate_records)
        filtering_artifacts._chmod_private_sqlite_files(self.path)

    def candidate_counts(self) -> dict[str, int]:
        row = self._connection.execute(
            "select count(*) as candidate_count, "
            "sum(case when prefilter_route='prefilter_survivor' then 1 else 0 end) "
            "as survivor_count from prefilter_candidates"
        ).fetchone()
        return {
            "candidate_count": int(row["candidate_count"] or 0),
            "survivor_count": int(row["survivor_count"] or 0),
        }

    def candidate_route_counts(self) -> dict[str, int]:
        return dict(
            sorted(
                (str(row["prefilter_route"]), int(row["route_count"]))
                for row in self._connection.execute(
                    "select prefilter_route, count(*) as route_count "
                    "from prefilter_candidates group by prefilter_route"
                )
            )
        )

    def prefilter_summary_facts(self) -> dict[str, Any]:
        metadata = {
            str(row["key"]): json.loads(str(row["value_json"]))
            for row in self._connection.execute(
                "select key, value_json from stage_metadata "
                "where key in ('input_schedule_hash', 'expected_physical_row_count', "
                "'expected_physical_row_key_set_hash')"
            )
        }
        counts = Counter(
            {
                str(row["row_status"]): int(row["row_count"])
                for row in self._connection.execute(
                    "select row_status, count(*) as row_count "
                    "from prefilter_source_rows group by row_status"
                )
            }
        )
        error_counts = {
            key: int(counts.get(key, 0))
            for key in ("fetch_error", "missing_text", "non_mapping_row", "segmentation_error")
            if counts.get(key, 0)
        }
        unaccounted = sum(
            value for key, value in counts.items() if key not in PREFILTER_TERMINAL_ROW_STATUSES
        )
        terminal_rows = sum(
            value for key, value in counts.items() if key in PREFILTER_TERMINAL_ROW_STATUSES
        )
        pending_rows = sum(
            value for key, value in counts.items() if key not in PREFILTER_TERMINAL_ROW_STATUSES
        )
        scheduled_key_hash = terminal_key_hash = pending_key_hash = None
        if metadata.get("expected_physical_row_key_set_hash") is not None:
            rows = list(
                self._connection.execute(
                    "select physical_row_key_hash, row_status from prefilter_source_rows"
                )
            )
            scheduled_key_hash = stable_json_hash(
                sorted(str(row["physical_row_key_hash"]) for row in rows)
            )
            terminal_key_hash = stable_json_hash(
                sorted(
                    str(row["physical_row_key_hash"])
                    for row in rows
                    if str(row["row_status"]) in PREFILTER_TERMINAL_ROW_STATUSES
                )
            )
            pending_key_hash = stable_json_hash(
                sorted(
                    str(row["physical_row_key_hash"])
                    for row in rows
                    if str(row["row_status"]) not in PREFILTER_TERMINAL_ROW_STATUSES
                )
            )
        expected_rows = metadata.get("expected_physical_row_count")
        if expected_rows is None:
            expected_rows = int(sum(counts.values()))
        return {
            "input_schedule_hash": metadata.get("input_schedule_hash"),
            "expected_physical_row_key_set_hash": metadata.get(
                "expected_physical_row_key_set_hash"
            ),
            "expected_physical_rows": int(expected_rows),
            "scheduled_rows": int(sum(counts.values())),
            "scheduled_physical_row_key_set_hash": scheduled_key_hash,
            "terminal_rows": int(terminal_rows),
            "terminal_physical_row_key_set_hash": terminal_key_hash,
            "pending_rows": int(pending_rows),
            "pending_physical_row_key_set_hash": pending_key_hash,
            "fetched_rows": int(
                counts.get("prefiltered", 0)
                + counts.get("missing_text", 0)
                + counts.get("segmentation_error", 0)
            ),
            "segmented_rows": int(counts.get("prefiltered", 0)),
            "row_error_counts": dict(sorted(error_counts.items())),
            "unaccounted_row_count": int(unaccounted),
        }

    def iter_survivor_inputs(self) -> Iterator[PrefilterSurvivorInput]:
        for row in self._connection.execute(
            _STORED_CANDIDATE_SELECT
            + " where c.prefilter_route='prefilter_survivor' order by c.candidate_id_hash"
        ):
            candidate, deterministic, evidence = _validated_stored_candidate(row)
            if evidence is None:  # pragma: no cover - route validation above is fail-closed
                raise ValueError("prefilter survivor missing private evidence")
            evidence_id, evidence_hash, evidence_text = evidence
            yield PrefilterSurvivorInput(
                candidate_id_hash=str(row["candidate_id_hash"]),
                candidate=candidate,
                deterministic_result=deterministic,
                s1_score=None if row["s1_score"] is None else float(row["s1_score"]),
                private_evidence_row_id=evidence_id,
                evidence_text_sha256=evidence_hash,
                private_evidence_text=evidence_text,
            )

    def fetch_w1_input_records(
        self,
        requested_pairs: Sequence[tuple[str, str]],
    ) -> dict[tuple[str, str], W1InputRecord]:
        """Fetch bounded W1 inputs while preserving first-request order.

        The left joins are intentional: callers distinguish missing candidate
        metadata from missing private evidence and apply the existing fail-closed
        scientific behavior themselves.
        """

        normalized = list(
            dict.fromkeys(
                (
                    filtering_artifacts._require_sha256(candidate_id, "candidate_id_hash"),
                    filtering_artifacts._require_sha256(
                        evidence_id, "private_evidence_row_id"
                    ),
                )
                for candidate_id, evidence_id in requested_pairs
            )
        )
        fetched: dict[tuple[str, str], W1InputRecord] = {}
        for start in range(0, len(normalized), 450):
            chunk = normalized[start : start + 450]
            placeholders = ",".join("(?, ?)" for _ in chunk)
            parameters = [value for pair in chunk for value in pair]
            for row in self._connection.execute(
                f"""
                with requested(candidate_id_hash, private_evidence_row_id) as (
                    values {placeholders}
                )
                select r.candidate_id_hash as requested_candidate_id_hash,
                       r.private_evidence_row_id as requested_private_evidence_row_id,
                       c.candidate_json as candidate_json,
                       e.candidate_id_hash as evidence_candidate_id_hash,
                       e.evidence_text_sha256 as evidence_text_sha256,
                       e.private_evidence_text as private_evidence_text
                from requested r
                left join prefilter_candidates c
                  on c.candidate_id_hash = r.candidate_id_hash
                left join private_w1_evidence e
                  on e.private_evidence_row_id = r.private_evidence_row_id
                """,
                parameters,
            ):
                key = (
                    filtering_artifacts._require_sha256(
                        row["requested_candidate_id_hash"], "candidate_id_hash"
                    ),
                    filtering_artifacts._require_sha256(
                        row["requested_private_evidence_row_id"],
                        "private_evidence_row_id",
                    ),
                )
                fetched[key] = W1InputRecord(
                    candidate_json=(
                        None if row["candidate_json"] is None else str(row["candidate_json"])
                    ),
                    evidence_candidate_id_hash=(
                        None
                        if row["evidence_candidate_id_hash"] is None
                        else str(row["evidence_candidate_id_hash"])
                    ),
                    evidence_text_sha256=(
                        None
                        if row["evidence_text_sha256"] is None
                        else str(row["evidence_text_sha256"])
                    ),
                    private_evidence_text=(
                        None
                        if row["private_evidence_text"] is None
                        else str(row["private_evidence_text"])
                    ),
                )
        return {pair: fetched[pair] for pair in normalized}

    def w1_scores(self) -> list[dict[str, Any]]:
        scores: list[dict[str, Any]] = []
        for row in self._connection.execute(
            "select candidate_id_hash, score_json from w1_scores order by candidate_id_hash"
        ):
            candidate_id = filtering_artifacts._require_sha256(
                row["candidate_id_hash"], "candidate_id_hash"
            )
            payload = json.loads(str(row["score_json"]))
            if not isinstance(payload, dict):
                raise ValueError("W1 score JSON must be an object")
            if payload.get("candidate_id_hash") != candidate_id:
                raise ValueError("W1 score JSON id mismatch")
            scores.append(payload)
        return scores

    def iter_closeout_survivor_facts(self) -> Iterator[CloseoutSurvivorFact]:
        """Validate compact survivor/evidence rows and emit no-text audit facts."""

        query = (
            _STORED_CANDIDATE_SELECT
            + " left join w1_scores s on s.candidate_id_hash=c.candidate_id_hash "
            "where c.prefilter_route='prefilter_survivor' order by c.candidate_id_hash"
        )
        # _STORED_CANDIDATE_SELECT already has a complete FROM/JOIN clause; add
        # the score projection explicitly to retain stage SQL ownership here.
        query = query.replace(
            "e.evidence_text_sha256, e.private_evidence_text ",
            "e.evidence_text_sha256, e.private_evidence_text, s.score_json ",
        )
        for row in self._connection.execute(query):
            candidate_id = filtering_artifacts._require_sha256(
                row["candidate_id_hash"], "closeout candidate_id_hash"
            )
            if row["candidate_json"] is None or row["private_evidence_text"] is None:
                raise ValueError("closeout compact survivor is missing candidate or evidence")
            candidate = json.loads(str(row["candidate_json"]))
            if not isinstance(candidate, dict) or candidate.get("candidate_id_hash") != candidate_id:
                raise ValueError("closeout compact survivor candidate binding mismatch")
            source_id = filtering_artifacts._require_sha256(
                candidate.get("source_document_id_hash"), "closeout source_document_id_hash"
            )
            segmentation_hash = filtering_artifacts._require_sha256(
                candidate.get("segmentation_receipt_hash"), "closeout segmentation_receipt_hash"
            )
            if (
                source_id != row["candidate_source_document_id_hash"]
                or segmentation_hash != row["candidate_segmentation_receipt_hash"]
                or row["evidence_candidate_id"] != candidate_id
            ):
                raise ValueError("closeout compact survivor lineage mismatch")
            evidence_text = str(row["private_evidence_text"])
            evidence_hash = filtering_artifacts._require_sha256(
                row["evidence_text_sha256"], "closeout evidence_text_sha256"
            )
            if (
                candidate.get("evidence_text_sha256") != evidence_hash
                or filtering_artifacts._sha256_text(evidence_text) != evidence_hash
            ):
                raise ValueError("closeout compact survivor evidence hash mismatch")
            core_start = filtering_artifacts._nonnegative_int(
                candidate.get("core_char_start"), "closeout core_char_start"
            )
            core_end = filtering_artifacts._nonnegative_int(
                candidate.get("core_char_end"), "closeout core_char_end"
            )
            evidence_start = filtering_artifacts._nonnegative_int(
                candidate.get("evidence_char_start"), "closeout evidence_char_start"
            )
            evidence_end = filtering_artifacts._nonnegative_int(
                candidate.get("evidence_char_end"), "closeout evidence_char_end"
            )
            if not evidence_start <= core_start < core_end <= evidence_end:
                raise ValueError("closeout compact survivor span ordering mismatch")
            if len(evidence_text) != evidence_end - evidence_start:
                raise ValueError("closeout compact survivor evidence length mismatch")
            core_text = evidence_text[core_start - evidence_start : core_end - evidence_start]
            core_hash = filtering_artifacts._require_sha256(
                candidate.get("core_text_sha256"), "closeout core_text_sha256"
            )
            if filtering_artifacts._sha256_text(core_text) != core_hash:
                raise ValueError("closeout compact survivor core hash mismatch")
            block_indexes = candidate.get("core_block_indexes")
            if (
                not isinstance(block_indexes, list)
                or not block_indexes
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in block_indexes)
                or block_indexes != list(range(block_indexes[0], block_indexes[-1] + 1))
            ):
                raise ValueError("closeout compact survivor block indexes mismatch")
            score_payload: dict[str, Any] | None = None
            if row["score_json"] is not None:
                parsed = json.loads(str(row["score_json"]))
                if not isinstance(parsed, dict) or parsed.get("candidate_id_hash") != candidate_id:
                    raise ValueError("closeout materialized score candidate mismatch")
                score_hash = parsed.get("stage_receipt_hash")
                if score_hash != stable_json_hash(
                    {key: value for key, value in parsed.items() if key != "stage_receipt_hash"}
                ):
                    raise ValueError("closeout materialized score hash mismatch")
                filtering_artifacts.validate_public_no_raw_payload(parsed)
                score_payload = parsed
            yield CloseoutSurvivorFact(
                candidate_id_hash=candidate_id,
                source_document_id_hash=source_id,
                segmentation_receipt_hash=segmentation_hash,
                core_text_sha256=core_hash,
                core_char_count=len(core_text),
                core_byte_count=len(core_text.encode("utf-8", errors="surrogatepass")),
                core_proxy_token_count=filtering_artifacts._nonnegative_int(
                    candidate.get("core_proxy_token_count"), "closeout core_proxy_token_count"
                ),
                first_block_index=int(block_indexes[0]),
                last_block_index=int(block_indexes[-1]),
                core_char_start=core_start,
                core_char_end=core_end,
                evidence_text_sha256=evidence_hash,
                materialized_score_payload=score_payload,
            )

    def w1_score_count(self) -> int:
        row = self._connection.execute("select count(*) from w1_scores").fetchone()
        return int(row[0])

    def insert_w1_score_payloads(
        self,
        payloads: Sequence[Mapping[str, Any]],
    ) -> int:
        """Atomically persist new scores after all replay conflicts are checked."""

        serialized_by_id: dict[str, str] = {}
        for payload in payloads:
            candidate_id = filtering_artifacts._require_sha256(
                payload.get("candidate_id_hash"), "candidate_id_hash"
            )
            score_json = json.dumps(dict(payload), sort_keys=True)
            previous = serialized_by_id.get(candidate_id)
            if previous is not None and previous != score_json:
                raise ValueError("conflicting W1 score payloads for candidate")
            serialized_by_id.setdefault(candidate_id, score_json)
        if not serialized_by_id:
            return 0

        candidate_ids = list(serialized_by_id)
        existing_by_id: dict[str, str] = {}
        for start in range(0, len(candidate_ids), 900):
            chunk = candidate_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            for row in self._connection.execute(
                f"select candidate_id_hash, score_json from w1_scores "
                f"where candidate_id_hash in ({placeholders})",
                chunk,
            ):
                existing_by_id[str(row["candidate_id_hash"])] = str(row["score_json"])

        for candidate_id, existing in existing_by_id.items():
            if existing != serialized_by_id[candidate_id]:
                raise ValueError("conflicting W1 score row for existing candidate")

        created_at = time.time()
        rows_to_insert = [
            (candidate_id, score_json, created_at)
            for candidate_id, score_json in serialized_by_id.items()
            if candidate_id not in existing_by_id
        ]
        if rows_to_insert:
            with self._connection:
                self._connection.executemany(
                    "insert into w1_scores(candidate_id_hash, score_json, created_at) "
                    "values (?, ?, ?)",
                    rows_to_insert,
                )
            filtering_artifacts._chmod_private_sqlite_files(self.path)
        return len(rows_to_insert)

    def runtime_storage_evidence(self) -> dict[str, Any]:
        pragma_names = (
            "journal_mode",
            "synchronous",
            "wal_autocheckpoint",
            "temp_store",
            "cache_size",
            "locking_mode",
        )
        pragmas = {
            name: self._connection.execute(f"pragma {name}").fetchone()[0]
            for name in pragma_names
        }
        return {
            "database_path_hash": stable_json_hash(
                str(self.path.resolve(strict=False))
            ),
            "sqlite_pragmas": pragmas,
        }

    def merge_prefilter_shards(
        self,
        source_states: Sequence[FilteringStageStateV1],
        *,
        batch_size: int = 10_000,
    ) -> None:
        """Stream immutable shard state into one all-or-nothing target merge."""

        if not self.writer_lock_held:
            raise RuntimeError("prefilter shard merge requires a writable target")
        size = int(batch_size)
        if size < 1:
            raise ValueError("prefilter merge copy batch_size must be positive")
        for source in source_states:
            source._require_immutable_query_source()
        table_columns = (
            ("prefilter_source_rows", tuple(name for name, *_rest in _TABLE_COLUMNS["prefilter_source_rows"])),
            ("prefilter_candidates", tuple(name for name, *_rest in _TABLE_COLUMNS["prefilter_candidates"])),
            ("private_w1_evidence", tuple(name for name, *_rest in _TABLE_COLUMNS["private_w1_evidence"])),
        )
        try:
            with self._connection:
                for table, columns in table_columns:
                    for source in source_states:
                        cursor = source._connection.execute(
                            f"select {', '.join(columns)} from {table} order by {columns[0]}"
                        )
                        placeholders = ", ".join("?" for _ in columns)
                        while True:
                            rows = cursor.fetchmany(size)
                            if not rows:
                                break
                            self._connection.executemany(
                                f"insert into {table}({', '.join(columns)}) values ({placeholders})",
                                [tuple(row[column] for column in columns) for row in rows],
                            )
        except sqlite3.IntegrityError as exc:
            message = str(exc).casefold()
            table = next(
                (name for name, _columns in table_columns if name.casefold() in message),
                "stage-state",
            )
            if table == "stage-state":
                # SQLite constraint messages name columns, not always tables.
                table = table_columns[0][0] if "physical_row_key_hash" in message else table
            raise ValueError(
                f"merged prefilter shard contains duplicate or conflicting {table} rows"
            ) from exc
        filtering_artifacts._chmod_private_sqlite_files(self.path)

    def copy_compact_survivors_from(
        self,
        source: FilteringStageStateV1,
        candidate_ids: Sequence[str],
        evidence_ids: Sequence[str],
    ) -> None:
        """Copy a survivor-only compact subset, rolling back on incomplete input."""

        if not self.writer_lock_held:
            raise RuntimeError("compact copy requires a writable destination")
        source._require_immutable_query_source()
        normalized_candidates = [
            filtering_artifacts._require_sha256(value, "compact candidate_id_hash")
            for value in candidate_ids
        ]
        normalized_evidence = [
            filtering_artifacts._require_sha256(value, "compact private_evidence_row_id")
            for value in evidence_ids
        ]
        if len(set(normalized_candidates)) != len(normalized_candidates):
            raise ValueError("compact start state contains duplicate candidate ids")
        if len(set(normalized_evidence)) != len(normalized_evidence):
            raise ValueError("compact start state contains duplicate private evidence ids")

        candidate_columns = tuple(name for name, *_rest in _TABLE_COLUMNS["prefilter_candidates"])
        evidence_columns = tuple(name for name, *_rest in _TABLE_COLUMNS["private_w1_evidence"])
        score_columns = tuple(name for name, *_rest in _TABLE_COLUMNS["w1_scores"])

        try:
            with self._connection:
                metadata_rows = list(
                    source._connection.execute(
                        "select key, value_json from stage_metadata order by key"
                    )
                )
                nonempty_tables = [
                    table
                    for table in (
                        "prefilter_source_rows",
                        "prefilter_candidates",
                        "private_w1_evidence",
                        "w1_scores",
                    )
                    if int(
                        self._connection.execute(
                            f"select count(*) from {table}"
                        ).fetchone()[0]
                    )
                    != 0
                ]
                if nonempty_tables:
                    raise ValueError("compact start destination must have empty data tables")
                destination_metadata = {
                    str(row["key"]): str(row["value_json"])
                    for row in self._connection.execute(
                        "select key, value_json from stage_metadata"
                    )
                }
                source_metadata = {
                    str(row["key"]): str(row["value_json"])
                    for row in metadata_rows
                }
                for key in destination_metadata.keys() & source_metadata.keys():
                    if destination_metadata[key] != source_metadata[key]:
                        raise ValueError("compact start destination metadata conflict")
                candidate_set = set(normalized_candidates)
                missing_metadata = [
                    (key, value)
                    for key, value in source_metadata.items()
                    if key not in destination_metadata
                ]
                self._connection.executemany(
                    "insert into stage_metadata(key, value_json) values (?, ?)",
                    missing_metadata,
                )

                observed_candidates: set[str] = set()
                observed_scores: set[str] = set()
                candidate_placeholders = ", ".join("?" for _ in candidate_columns)
                score_placeholders = ", ".join("?" for _ in score_columns)
                for start in range(0, len(normalized_candidates), 900):
                    chunk = normalized_candidates[start : start + 900]
                    placeholders = ",".join("?" for _ in chunk)
                    candidates = list(
                        source._connection.execute(
                            f"select {', '.join(candidate_columns)} from prefilter_candidates "
                            f"where candidate_id_hash in ({placeholders}) order by candidate_id_hash",
                            chunk,
                        )
                    )
                    chunk_set = set(chunk)
                    observed_chunk = {str(row["candidate_id_hash"]) for row in candidates}
                    if observed_chunk != chunk_set:
                        raise ValueError("compact start state is missing survivor candidate rows")
                    if any(
                        str(row["prefilter_route"]) != "prefilter_survivor"
                        for row in candidates
                    ):
                        raise ValueError("compact start state may contain only survivor candidate rows")
                    self._connection.executemany(
                        f"insert into prefilter_candidates({', '.join(candidate_columns)}) "
                        f"values ({candidate_placeholders})",
                        [
                            tuple(row[column] for column in candidate_columns)
                            for row in candidates
                        ],
                    )
                    observed_candidates.update(observed_chunk)

                    scores = list(
                        source._connection.execute(
                            f"select {', '.join(score_columns)} from w1_scores "
                            f"where candidate_id_hash in ({placeholders}) order by candidate_id_hash",
                            chunk,
                        )
                    )
                    for row in scores:
                        candidate_id = str(row["candidate_id_hash"])
                        score_payload = json.loads(str(row["score_json"]))
                        if (
                            candidate_id not in chunk_set
                            or not isinstance(score_payload, dict)
                            or score_payload.get("candidate_id_hash") != candidate_id
                        ):
                            raise ValueError("compact start state W1 score candidate binding mismatch")
                        observed_scores.add(candidate_id)
                    self._connection.executemany(
                        f"insert into w1_scores({', '.join(score_columns)}) "
                        f"values ({score_placeholders})",
                        [tuple(row[column] for column in score_columns) for row in scores],
                    )

                observed_evidence: set[str] = set()
                evidence_candidates: set[str] = set()
                evidence_placeholders = ", ".join("?" for _ in evidence_columns)
                for start in range(0, len(normalized_evidence), 900):
                    chunk = normalized_evidence[start : start + 900]
                    placeholders = ",".join("?" for _ in chunk)
                    evidence = list(
                        source._connection.execute(
                            "select e.private_evidence_row_id, e.candidate_id_hash, "
                            "e.evidence_text_sha256, e.private_evidence_text, "
                            "c.candidate_json, c.prefilter_route "
                            "from private_w1_evidence e join prefilter_candidates c "
                            "on c.candidate_id_hash=e.candidate_id_hash "
                            f"where private_evidence_row_id in ({placeholders}) "
                            "order by private_evidence_row_id",
                            chunk,
                        )
                    )
                    chunk_set = set(chunk)
                    observed_chunk = {
                        str(row["private_evidence_row_id"]) for row in evidence
                    }
                    if observed_chunk != chunk_set:
                        raise ValueError("compact start state is missing private evidence rows")
                    for row in evidence:
                        candidate_id = str(row["candidate_id_hash"])
                        evidence_hash = filtering_artifacts._require_sha256(
                            row["evidence_text_sha256"], "evidence_text_sha256"
                        )
                        candidate = json.loads(str(row["candidate_json"]))
                        if (
                            candidate_id not in candidate_set
                            or str(row["prefilter_route"]) != "prefilter_survivor"
                            or not isinstance(candidate, dict)
                            or candidate.get("candidate_id_hash") != candidate_id
                            or candidate.get("evidence_text_sha256") != evidence_hash
                            or filtering_artifacts._sha256_text(
                                str(row["private_evidence_text"])
                            )
                            != evidence_hash
                        ):
                            raise ValueError(
                                "compact start state private evidence candidate binding mismatch"
                            )
                        evidence_candidates.add(candidate_id)
                    self._connection.executemany(
                        f"insert into private_w1_evidence({', '.join(evidence_columns)}) "
                        f"values ({evidence_placeholders})",
                        [tuple(row[column] for column in evidence_columns) for row in evidence],
                    )
                    observed_evidence.update(observed_chunk)

                if observed_candidates != candidate_set:
                    raise ValueError("compact start state is missing survivor candidate rows")
                if observed_evidence != set(normalized_evidence):
                    raise ValueError("compact start state is missing private evidence rows")
                if evidence_candidates != candidate_set:
                    raise ValueError(
                        "compact start state private evidence candidate binding mismatch"
                    )
                if not observed_scores.issubset(candidate_set):
                    raise ValueError("compact start state W1 score candidate binding mismatch")
        except sqlite3.IntegrityError as exc:
            raise ValueError("compact start destination contains conflicting state") from exc
        filtering_artifacts._chmod_private_sqlite_files(self.path)

    def checkpoint_truncate_and_sha256(self) -> str:
        if not self.writer_lock_held:
            raise RuntimeError("checkpoint requires a writable stage state")
        row = self._connection.execute("pragma wal_checkpoint(TRUNCATE)").fetchone()
        if row is None or len(row) < 3:
            raise RuntimeError("SQLite checkpoint returned no complete proof")
        busy, log_frames, checkpointed_frames = (int(row[0]), int(row[1]), int(row[2]))
        if busy != 0 or log_frames != checkpointed_frames:
            raise RuntimeError("SQLite checkpoint was busy or incomplete")
        return filtering_artifacts._sha256_file(self.path)

    def compact_inventory(self, evidence_ids: Sequence[str]) -> dict[str, int]:
        normalized = [
            filtering_artifacts._require_sha256(value, "private_evidence_row_id")
            for value in evidence_ids
        ]
        if len(set(normalized)) != len(normalized):
            raise ValueError("compact inventory contains duplicate private evidence ids")
        rows: list[sqlite3.Row] = []
        for start in range(0, len(normalized), 900):
            chunk = normalized[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                self._connection.execute(
                    "select private_evidence_row_id, candidate_id_hash, evidence_text_sha256, "
                    f"private_evidence_text from private_w1_evidence where private_evidence_row_id in ({placeholders})",
                    chunk,
                )
            )
        if {str(row["private_evidence_row_id"]) for row in rows} != set(normalized):
            raise ValueError("compact start state private evidence id set mismatch")
        for row in rows:
            if filtering_artifacts._sha256_text(str(row["private_evidence_text"])) != str(
                row["evidence_text_sha256"]
            ):
                raise ValueError("compact start state private evidence text hash mismatch")
        counts = self.candidate_counts()
        evidence_count = int(
            self._connection.execute("select count(*) from private_w1_evidence").fetchone()[0]
        )
        if (
            counts["candidate_count"] != len(normalized)
            or counts["survivor_count"] != len(normalized)
            or evidence_count != len(normalized)
        ):
            raise ValueError("compact start state must be survivor-only with exact counts")
        candidate_ids = {str(row["candidate_id_hash"]) for row in rows}
        stored_ids = {
            str(row["candidate_id_hash"])
            for row in self._connection.execute(
                "select candidate_id_hash from prefilter_candidates where prefilter_route='prefilter_survivor'"
            )
        }
        if candidate_ids != stored_ids:
            raise ValueError("compact start state private evidence candidate mismatch")
        return {
            "candidate_count": counts["candidate_count"],
            "survivor_count": counts["survivor_count"],
            "private_evidence_count": evidence_count,
            "w1_score_count": self.w1_score_count(),
        }

    def iter_candidate_routes(self) -> Iterator[tuple[str, str]]:
        for row in self._connection.execute(
            "select candidate_id_hash, prefilter_route from prefilter_candidates order by candidate_id_hash"
        ):
            yield str(row["candidate_id_hash"]), str(row["prefilter_route"])

    def candidate_ids(self, *, route: str | None = None) -> set[str]:
        if route is None:
            rows = self._connection.execute("select candidate_id_hash from prefilter_candidates")
        else:
            rows = self._connection.execute(
                "select candidate_id_hash from prefilter_candidates where prefilter_route=?",
                (str(route),),
            )
        return {str(row["candidate_id_hash"]) for row in rows}

    def w1_score_candidate_ids(self) -> set[str]:
        return {
            str(row["candidate_id_hash"])
            for row in self._connection.execute("select candidate_id_hash from w1_scores")
        }

    def _require_immutable_query_source(self) -> None:
        if self.writer_lock_held or not self._immutable_frozen:
            raise ValueError("stage-state copy source must be immutable/query-only")
        query_only = int(self._connection.execute("pragma query_only").fetchone()[0])
        if query_only != 1:
            raise ValueError("stage-state copy source must be immutable/query-only")

    def open_active_reader(
        self,
        prefilter_config_payload: Mapping[str, Any],
        *,
        check_same_thread: bool = False,
    ) -> FilteringStageStateV1:
        """Open a WAL-aware reader owned by this active NORMAL-locking writer."""

        if not self.writer_lock_held:
            raise RuntimeError("active stage-state reader requires its writer handle")
        locking_mode = str(self._connection.execute("pragma locking_mode").fetchone()[0]).casefold()
        if locking_mode != "normal":
            raise RuntimeError("active stage-state reader requires NORMAL writer locking")
        connection = _readonly_connection(
            self.path,
            check_same_thread=check_same_thread,
            immutable=False,
        )
        desired = _desired_metadata(
            producer_config_payload=prefilter_config_payload,
            w1_config_payload=None,
        )
        try:
            _validate_schema(connection)
            _pending_metadata(connection, desired, allow_missing=False)
            return FilteringStageStateV1(
                connection,
                self.path,
                writer_lock_fd=None,
                immutable_frozen=False,
            )
        except BaseException:
            connection.close()
            raise


def _validated_stored_candidate(
    row: sqlite3.Row,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, str, str] | None]:
    candidate_id = filtering_artifacts._require_sha256(
        row["candidate_id_hash"], "candidate_id_hash"
    )
    candidate = json.loads(str(row["candidate_json"]))
    deterministic = json.loads(str(row["deterministic_result_json"]))
    if not isinstance(candidate, dict) or not isinstance(deterministic, dict):
        raise ValueError("prefilter candidate JSON must contain objects")
    if candidate.get("candidate_id_hash") != candidate_id:
        raise ValueError("prefilter candidate JSON id mismatch")
    source_id = filtering_artifacts._require_sha256(
        row["candidate_source_document_id_hash"], "candidate source_document_id_hash"
    )
    receipt_hash = filtering_artifacts._require_sha256(
        row["candidate_segmentation_receipt_hash"], "candidate segmentation_receipt_hash"
    )
    if candidate.get("source_document_id_hash") != source_id:
        raise ValueError("prefilter candidate source document mismatch")
    if candidate.get("segmentation_receipt_hash") != receipt_hash:
        raise ValueError("prefilter candidate segmentation receipt mismatch")
    route = str(row["prefilter_route"])
    if route not in PREFILTER_ROUTES:
        raise ValueError("unsupported prefilter route")
    evidence_id = row["private_evidence_row_id"]
    if route != "prefilter_survivor":
        if evidence_id is not None:
            raise ValueError("non-survivor candidate has unexpected private evidence")
        return candidate, deterministic, None
    if evidence_id is None:
        raise ValueError("prefilter survivor missing private evidence")
    normalized_evidence_id = filtering_artifacts._require_sha256(
        evidence_id, "private_evidence_row_id"
    )
    if str(row["evidence_candidate_id"]) != candidate_id:
        raise ValueError("prefilter survivor private evidence candidate binding mismatch")
    evidence_hash = filtering_artifacts._require_sha256(
        row["evidence_text_sha256"], "evidence_text_sha256"
    )
    evidence_text = str(row["private_evidence_text"])
    if filtering_artifacts._sha256_text(evidence_text) != evidence_hash:
        raise ValueError("prefilter survivor private evidence text hash mismatch")
    if candidate.get("evidence_text_sha256") != evidence_hash:
        raise ValueError("prefilter survivor candidate evidence hash mismatch")
    return candidate, deterministic, (normalized_evidence_id, evidence_hash, evidence_text)


def _validate_terminal_source_candidates(
    connection: sqlite3.Connection,
    source_rows: Sequence[sqlite3.Row],
) -> None:
    if not source_rows:
        return
    expected: dict[str, sqlite3.Row] = {}
    for source_row in source_rows:
        source_id = filtering_artifacts._require_sha256(
            source_row["source_document_id_hash"], "source_document_id_hash"
        )
        if source_id in expected:
            raise ValueError("terminal source rows contain duplicate source document ids")
        expected[source_id] = source_row
    candidate_counts: Counter[str] = Counter()
    survivor_counts: Counter[str] = Counter()
    source_ids = list(expected)
    for start in range(0, len(source_ids), 500):
        chunk = source_ids[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        for candidate_row in connection.execute(
            _STORED_CANDIDATE_SELECT
            + f" where c.source_document_id_hash in ({placeholders}) "
            + "order by c.source_document_id_hash, c.candidate_id_hash",
            chunk,
        ):
            _candidate, _deterministic, _evidence = _validated_stored_candidate(
                candidate_row
            )
            source_id = str(candidate_row["candidate_source_document_id_hash"])
            source_row = expected[source_id]
            source_receipt = source_row["segmentation_receipt_hash"]
            if (
                source_receipt is None
                or str(candidate_row["candidate_segmentation_receipt_hash"])
                != str(source_receipt)
            ):
                raise ValueError("terminal source candidate segmentation receipt mismatch")
            candidate_counts[source_id] += 1
            if str(candidate_row["prefilter_route"]) == "prefilter_survivor":
                survivor_counts[source_id] += 1
    for source_id, source_row in expected.items():
        if candidate_counts[source_id] != int(source_row["candidate_count"]):
            raise ValueError("terminal source candidate count mismatch")
        if survivor_counts[source_id] != int(source_row["survivor_count"]):
            raise ValueError("terminal source survivor count mismatch")


def _normalized_source_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "physical_row_key_hash": filtering_artifacts._require_sha256(
            payload.get("physical_row_key_hash"), "physical_row_key_hash"
        ),
        "source_schedule_hash": (
            None
            if payload.get("source_schedule_hash") is None
            else filtering_artifacts._require_sha256(
                payload.get("source_schedule_hash"), "source_schedule_hash"
            )
        ),
        "schedule_entry_hash": filtering_artifacts._require_sha256(
            payload.get("schedule_entry_hash"), "schedule_entry_hash"
        ),
        "source_file_id_hash": filtering_artifacts._require_sha256(
            payload.get("source_file_id_hash"), "source_file_id_hash"
        ),
        "row_group_index": filtering_artifacts._nonnegative_int(
            payload.get("row_group_index"), "row_group_index"
        ),
        "row_group_local_offset": filtering_artifacts._nonnegative_int(
            payload.get("row_group_local_offset"), "row_group_local_offset"
        ),
        "source_document_id_hash": filtering_artifacts._require_sha256(
            payload.get("source_document_id_hash"), "source_document_id_hash"
        ),
        "row_status": str(payload.get("row_status") or ""),
        "error_class": payload.get("error_class"),
        "segmentation_receipt_hash": payload.get("segmentation_receipt_hash"),
        "candidate_count": filtering_artifacts._nonnegative_int(
            payload.get("candidate_count"), "candidate_count"
        ),
        "survivor_count": filtering_artifacts._nonnegative_int(
            payload.get("survivor_count"), "survivor_count"
        ),
        "row_receipt_hash": filtering_artifacts._require_sha256(
            payload.get("row_receipt_hash"), "row_receipt_hash"
        ),
    }
    if row["row_status"] not in {"scheduled", *PREFILTER_TERMINAL_ROW_STATUSES}:
        raise ValueError("unsupported prefilter source-row status")
    if row["segmentation_receipt_hash"] is not None:
        row["segmentation_receipt_hash"] = filtering_artifacts._require_sha256(
            row["segmentation_receipt_hash"], "segmentation_receipt_hash"
        )
    return row


def _source_row_values(row: Mapping[str, Any], *, created_at: float, updated_at: float) -> tuple[Any, ...]:
    return (
        row["physical_row_key_hash"],
        row["source_schedule_hash"],
        row["schedule_entry_hash"],
        row["source_file_id_hash"],
        int(row["row_group_index"]),
        int(row["row_group_local_offset"]),
        row["source_document_id_hash"],
        row["row_status"],
        row["error_class"],
        row["segmentation_receipt_hash"],
        int(row["candidate_count"]),
        int(row["survivor_count"]),
        row["row_receipt_hash"],
        created_at,
        updated_at,
    )


def _insert_source_rows_strict(
    connection: sqlite3.Connection,
    payloads: Sequence[Mapping[str, Any]],
    *,
    now: float,
) -> None:
    rows = [_normalized_source_row(payload) for payload in payloads]
    if not rows:
        return
    connection.executemany(
        "insert into prefilter_source_rows(physical_row_key_hash, source_schedule_hash, "
        "schedule_entry_hash, source_file_id_hash, row_group_index, row_group_local_offset, "
        "source_document_id_hash, row_status, error_class, segmentation_receipt_hash, "
        "candidate_count, survivor_count, row_receipt_hash, created_at, updated_at) "
        "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [_source_row_values(row, created_at=now, updated_at=now) for row in rows],
    )


def _store_source_row_in_transaction(
    connection: sqlite3.Connection,
    payload: Mapping[str, Any],
    *,
    now: float,
) -> None:
    row = _normalized_source_row(payload)
    existing = connection.execute(
        "select * from prefilter_source_rows where physical_row_key_hash=?",
        (row["physical_row_key_hash"],),
    ).fetchone()
    if existing is None:
        connection.execute(
            "insert into prefilter_source_rows(physical_row_key_hash, source_schedule_hash, "
            "schedule_entry_hash, source_file_id_hash, row_group_index, row_group_local_offset, "
            "source_document_id_hash, row_status, error_class, segmentation_receipt_hash, "
            "candidate_count, survivor_count, row_receipt_hash, created_at, updated_at) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _source_row_values(row, created_at=now, updated_at=now),
        )
        return
    for field in _SOURCE_ROW_IMMUTABLE_FIELDS:
        expected = int(row[field]) if field in {"row_group_index", "row_group_local_offset"} else row[field]
        observed = int(existing[field]) if field in {"row_group_index", "row_group_local_offset"} else existing[field]
        if observed != expected:
            raise ValueError("prefilter source-row immutable identity drift")
    if str(existing["row_receipt_hash"]) == row["row_receipt_hash"]:
        return
    previous_status = str(existing["row_status"])
    next_status = str(row["row_status"])
    if previous_status != next_status and not (
        previous_status == "scheduled" and next_status in PREFILTER_TERMINAL_ROW_STATUSES
    ):
        raise ValueError("illegal source-row status transition")
    connection.execute(
        "update prefilter_source_rows set row_status=?, error_class=?, "
        "segmentation_receipt_hash=?, candidate_count=?, survivor_count=?, "
        "row_receipt_hash=?, updated_at=? where physical_row_key_hash=?",
        (
            next_status,
            row["error_class"],
            row["segmentation_receipt_hash"],
            int(row["candidate_count"]),
            int(row["survivor_count"]),
            row["row_receipt_hash"],
            now,
            row["physical_row_key_hash"],
        ),
    )


def _validate_candidate_record(
    candidate_row: tuple[Any, ...],
    evidence_row: tuple[Any, ...] | None,
) -> None:
    if len(candidate_row) != 8:
        raise ValueError("prefilter candidate row shape mismatch")
    candidate_id = filtering_artifacts._require_sha256(candidate_row[0], "candidate_id_hash")
    candidate = json.loads(str(candidate_row[3]))
    deterministic = json.loads(str(candidate_row[4]))
    if not isinstance(candidate, dict) or not isinstance(deterministic, dict):
        raise ValueError("prefilter candidate record JSON must contain objects")
    if candidate.get("candidate_id_hash") != candidate_id:
        raise ValueError("prefilter candidate record id mismatch")
    survivor = str(candidate_row[5]) == "prefilter_survivor"
    if survivor != (evidence_row is not None):
        raise ValueError("prefilter survivor evidence presence mismatch")
    if evidence_row is not None:
        if len(evidence_row) != 4 or str(evidence_row[1]) != candidate_id:
            raise ValueError("prefilter private evidence candidate binding mismatch")
        evidence_hash = filtering_artifacts._require_sha256(
            evidence_row[2], "evidence_text_sha256"
        )
        if filtering_artifacts._sha256_text(str(evidence_row[3])) != evidence_hash:
            raise ValueError("prefilter private evidence text hash mismatch")
        if candidate.get("evidence_text_sha256") != evidence_hash:
            raise ValueError("prefilter candidate evidence hash mismatch")


def _insert_candidate_records(
    connection: sqlite3.Connection,
    records: Sequence[tuple[tuple[Any, ...], tuple[Any, ...] | None]],
) -> None:
    if not records:
        return
    connection.executemany(
        "insert into prefilter_candidates(candidate_id_hash, source_document_id_hash, "
        "segmentation_receipt_hash, candidate_json, deterministic_result_json, "
        "prefilter_route, s1_score, created_at) values (?, ?, ?, ?, ?, ?, ?, ?)",
        [record[0] for record in records],
    )
    evidence_rows = [record[1] for record in records if record[1] is not None]
    if evidence_rows:
        connection.executemany(
            "insert into private_w1_evidence(private_evidence_row_id, candidate_id_hash, "
            "evidence_text_sha256, private_evidence_text) values (?, ?, ?, ?)",
            evidence_rows,
        )


def _validate_candidate_replay(
    connection: sqlite3.Connection,
    existing: sqlite3.Row,
    *,
    candidate_row: tuple[Any, ...],
    evidence_row: tuple[Any, ...] | None,
) -> None:
    expected = candidate_row[:6]
    observed = tuple(existing[index] for index in range(6))
    if observed != expected:
        raise ValueError("conflicting prefilter candidate row for existing id")
    observed_score = existing["s1_score"]
    expected_score = candidate_row[6]
    if not (
        (observed_score is None and expected_score is None)
        or (
            observed_score is not None
            and expected_score is not None
            and float(observed_score) == float(expected_score)
        )
    ):
        raise ValueError("conflicting prefilter candidate score for existing id")
    evidence = connection.execute(
        "select private_evidence_row_id, candidate_id_hash, evidence_text_sha256, "
        "private_evidence_text from private_w1_evidence where candidate_id_hash=?",
        (candidate_row[0],),
    ).fetchone()
    if evidence_row is None:
        if evidence is not None:
            raise ValueError("unexpected private evidence for non-survivor candidate")
        return
    if evidence is None:
        raise ValueError("prefilter candidate replay missing private evidence")
    if tuple(evidence) != evidence_row:
        raise ValueError("prefilter candidate replay private evidence mismatch")
    if filtering_artifacts._sha256_text(str(evidence["private_evidence_text"])) != str(
        evidence["evidence_text_sha256"]
    ):
        raise ValueError("prefilter candidate replay private evidence hash mismatch")


def _state_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / STAGE_STATE_FILENAME


def _validate_run_dir(run_dir: str | Path, *, create: bool) -> Path:
    path = Path(run_dir)
    filtering_artifacts.validate_private_filtering_run_dir(path)
    if path.is_symlink():
        raise ValueError("private filtering run dir must not be a symlink")
    if create:
        filtering_artifacts._ensure_private_dir(path)
    elif not path.is_dir():
        raise ValueError("private filtering run dir is missing")
    if os.name == "posix" and path.stat().st_mode & 0o777 != 0o700:
        if create:
            os.chmod(path, 0o700)
        else:
            raise ValueError("private filtering run dir must use mode 0700")
    return path


def _validate_existing_state_path(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise ValueError("filtering stage state is missing") from None
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("filtering stage state must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
        raise ValueError("filtering stage state must be a single-link regular file")
    if os.name == "posix":
        if int(metadata.st_uid) != int(os.geteuid()):
            raise ValueError("filtering stage state must belong to the current user")
        if stat.S_IMODE(metadata.st_mode) != _PRIVATE_FILE_MODE:
            raise ValueError("filtering stage state must use mode 0600")
    return metadata


def _acquire_writer_lock(state_path: Path) -> int:
    if os.name != "posix":  # pragma: no cover - production hosts are POSIX
        raise RuntimeError("filtering stage state writers require POSIX flock")
    lock_path = state_path.with_suffix(state_path.suffix + ".writer.lock")
    create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        create_flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, create_flags, _PRIVATE_FILE_MODE)
        expected = os.fstat(fd)
        filtering_artifacts._validate_private_partial_metadata(expected)
    except FileExistsError:
        try:
            expected = filtering_artifacts._validate_owned_private_partial(lock_path)
            flags = os.O_RDWR | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
            fd = os.open(lock_path, flags)
        except (OSError, ValueError):
            raise RuntimeError("filtering stage state writer lock is unsafe") from None
    except (OSError, ValueError):
        raise RuntimeError("filtering stage state writer lock is unsafe") from None
    try:
        opened = os.fstat(fd)
        filtering_artifacts._validate_private_partial_metadata(opened)
        current = filtering_artifacts._validate_owned_private_partial(lock_path)
        if (
            (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise RuntimeError("filtering stage state writer lock changed while opening")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("filtering stage state writer is already active") from None
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def stage_state_snapshot_lock(state_path: str | Path) -> Iterator[None]:
    """Exclude the canonical writer while an immutable DB snapshot is observed."""

    path = Path(state_path)
    _validate_existing_state_path(path)
    fd = _acquire_writer_lock(path)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _readonly_connection(
    path: Path,
    *,
    check_same_thread: bool,
    immutable: bool = False,
) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    if immutable:
        uri += "&immutable=1"
    connection = sqlite3.connect(
        uri,
        uri=True,
        timeout=60.0,
        check_same_thread=bool(check_same_thread),
    )
    connection.row_factory = sqlite3.Row
    connection.execute("pragma query_only=ON")
    connection.execute("pragma busy_timeout=60000")
    connection.execute("pragma temp_store=MEMORY")
    connection.execute("pragma cache_size=-262144")
    return connection


def _writable_connection(
    path: Path,
    *,
    exclusive_locking: bool,
    check_same_thread: bool,
) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path,
        timeout=60.0,
        check_same_thread=bool(check_same_thread),
    )
    connection.row_factory = sqlite3.Row
    connection.execute("pragma busy_timeout=60000")
    connection.execute("pragma journal_mode=WAL")
    connection.execute("pragma synchronous=NORMAL")
    connection.execute("pragma wal_autocheckpoint=16384")
    connection.execute("pragma temp_store=MEMORY")
    connection.execute("pragma cache_size=-262144")
    connection.execute(
        "pragma locking_mode=EXCLUSIVE" if exclusive_locking else "pragma locking_mode=NORMAL"
    )
    filtering_artifacts._chmod_private_sqlite_files(path)
    return connection


def _column_shape(connection: sqlite3.Connection, table: str) -> tuple[tuple[str, str, int, str | None, int], ...]:
    return tuple(
        (
            str(row["name"]),
            str(row["type"]).upper(),
            int(row["notnull"]),
            None if row["dflt_value"] is None else str(row["dflt_value"]),
            int(row["pk"]),
        )
        for row in connection.execute(f"pragma table_info('{table}')")
    )


def _unique_column_sets(connection: sqlite3.Connection, table: str) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()
    for index in connection.execute(f"pragma index_list('{table}')"):
        if int(index["unique"]) != 1:
            raise ValueError("filtering stage state schema mismatch: non-unique index present")
        name = str(index["name"])
        result.add(
            tuple(
                str(row["name"])
                for row in connection.execute(f"pragma index_info('{name}')")
            )
        )
    return result


def _validate_schema(connection: sqlite3.Connection) -> None:
    table_rows = list(
        connection.execute(
            "select name, sql from sqlite_schema "
            "where type='table' and name not like 'sqlite_%' order by name"
        )
    )
    if {str(row["name"]) for row in table_rows} != set(_TABLE_COLUMNS):
        raise ValueError("filtering stage state schema mismatch: table set")
    if connection.execute("pragma user_version").fetchone()[0] != 0:
        raise ValueError("filtering stage state schema mismatch: user_version")
    extras = connection.execute(
        "select count(*) from sqlite_schema where type in ('view', 'trigger')"
    ).fetchone()[0]
    explicit_indexes = connection.execute(
        "select count(*) from sqlite_schema where type='index' and sql is not null"
    ).fetchone()[0]
    if int(extras) != 0 or int(explicit_indexes) != 0:
        raise ValueError("filtering stage state schema mismatch: extra schema objects")
    for row in table_rows:
        table = str(row["name"])
        sql = str(row["sql"] or "").casefold()
        if "without rowid" in sql or " strict" in sql:
            raise ValueError("filtering stage state schema mismatch: table options")
        if _column_shape(connection, table) != _TABLE_COLUMNS[table]:
            raise ValueError(f"filtering stage state schema mismatch: {table} columns")
        if list(connection.execute(f"pragma foreign_key_list('{table}')")):
            raise ValueError("filtering stage state schema mismatch: foreign keys")
        if _unique_column_sets(connection, table) != _UNIQUE_COLUMNS[table]:
            raise ValueError(f"filtering stage state schema mismatch: {table} uniqueness")


def _desired_metadata(
    *,
    producer_config_payload: Mapping[str, Any] | None,
    producer_retained_config_payload: Mapping[str, Any] | None = None,
    w1_config_payload: Mapping[str, Any] | None,
) -> tuple[tuple[str, Any, str], ...]:
    desired: list[tuple[str, Any, str]] = []
    if producer_config_payload is not None:
        retained_payload = dict(producer_config_payload)
        if producer_retained_config_payload is not None:
            retained_payload = dict(producer_retained_config_payload)
            producer_hash = producer_config_payload.get("config_hash")
            retained_hash = retained_payload.get("config_hash")
            if (
                not isinstance(producer_hash, str)
                or not isinstance(retained_hash, str)
                or producer_hash == retained_hash
            ):
                raise ValueError("filtering stage state legacy retained producer config metadata mismatch")
        desired.extend(
            (
                ("config_hash", producer_config_payload["config_hash"], "producer"),
                ("config_payload", retained_payload, "producer"),
            )
        )
    if w1_config_payload is not None:
        desired.extend(
            (
                ("w1_config_hash", w1_config_payload["config_hash"], "W1"),
                ("w1_config_payload", dict(w1_config_payload), "W1"),
            )
        )
    return tuple(desired)


def _pending_metadata(
    connection: sqlite3.Connection,
    desired: tuple[tuple[str, Any, str], ...],
    *,
    allow_missing: bool,
    legacy_retained_config_payload: Mapping[str, Any] | None = None,
) -> list[tuple[str, str]]:
    existing = {
        str(row["key"]): str(row["value_json"])
        for row in connection.execute("select key, value_json from stage_metadata")
    }
    desired_by_key = {key: expected for key, expected, _label in desired}
    pending: list[tuple[str, str]] = []
    for key, expected, label in desired:
        raw = existing.get(key)
        if raw is None:
            if not allow_missing:
                raise ValueError(f"filtering stage state {label} config metadata missing")
            pending.append((key, json.dumps(expected, sort_keys=key.endswith("_payload"))))
            continue
        try:
            stored = json.loads(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"filtering stage state {label} config metadata mismatch"
            ) from None
        if stored != expected and not _is_historical_producer_payload_for_w1(
            key=key,
            stored=stored,
            existing=existing,
            desired=desired_by_key,
            retained_config_payload=legacy_retained_config_payload,
        ):
            raise ValueError(f"filtering stage state {label} config metadata mismatch")
    return pending


def _is_historical_producer_payload_for_w1(
    *,
    key: str,
    stored: Any,
    existing: Mapping[str, str],
    desired: Mapping[str, Any],
    retained_config_payload: Mapping[str, Any] | None,
) -> bool:
    """Recognize the bounded legacy state shape without rewriting its history.

    Historical prefilter fixtures changed the immutable producer ``config_hash``
    to the legacy identity but retained the then-current full config payload.
    W1 may consume that state with the retained payload only when the producer
    hash matches exactly and that payload is exactly the requested W1 consumer
    config.  Every other producer-payload mismatch remains fail-closed.
    """

    if key != "config_payload":
        return False
    if not {"config_hash", "config_payload"}.issubset(desired):
        return False
    retained = retained_config_payload
    if retained is None and {"w1_config_hash", "w1_config_payload"}.issubset(desired):
        retained = desired["w1_config_payload"]
    if retained is None:
        try:
            stored_w1_hash = json.loads(existing["w1_config_hash"])
            stored_w1_payload = json.loads(existing["w1_config_payload"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if (
                isinstance(stored_w1_payload, Mapping)
                and stored_w1_payload.get("config_hash") == stored_w1_hash
            ):
                retained = stored_w1_payload
    if not isinstance(retained, Mapping):
        return False
    retained = dict(retained)
    retained_hash = retained.get("config_hash")
    if not isinstance(retained_hash, str) or desired["config_hash"] == retained_hash:
        return False
    try:
        stored_producer_hash = json.loads(existing["config_hash"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        stored_producer_hash == desired["config_hash"]
        and stored == retained
        and desired["config_payload"] != stored
    )


def _validate_existing_readonly(
    path: Path,
    desired: tuple[tuple[str, Any, str], ...],
    *,
    allow_missing_metadata: bool,
    check_same_thread: bool,
    legacy_retained_config_payload: Mapping[str, Any] | None = None,
) -> tuple[os.stat_result, list[tuple[str, str]]]:
    expected = _validate_existing_state_path(path)
    if any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError(
            "filtering stage state must be frozen and checkpointed before immutable validation"
        )
    connection = _readonly_connection(
        path,
        check_same_thread=check_same_thread,
        immutable=True,
    )
    try:
        _validate_schema(connection)
        pending = _pending_metadata(
            connection,
            desired,
            allow_missing=allow_missing_metadata,
            legacy_retained_config_payload=legacy_retained_config_payload,
        )
    finally:
        connection.close()
    current = _validate_existing_state_path(path)
    if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("filtering stage state changed during validation")
    return current, pending


def _create_fresh_state(path: Path, *, check_same_thread: bool) -> sqlite3.Connection:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, _PRIVATE_FILE_MODE)
    try:
        if os.name == "posix":
            os.fchmod(fd, _PRIVATE_FILE_MODE)
    finally:
        os.close(fd)
    connection = _writable_connection(
        path,
        exclusive_locking=True,
        check_same_thread=check_same_thread,
    )
    try:
        with connection:
            connection.executescript(_SCHEMA_SQL)
        _validate_schema(connection)
        return connection
    except BaseException:
        connection.close()
        for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        raise


def _open_writer(
    run_dir: str | Path,
    *,
    producer_config_payload: Mapping[str, Any],
    producer_retained_config_payload: Mapping[str, Any] | None = None,
    w1_config_payload: Mapping[str, Any] | None,
    exclusive_locking: bool,
    check_same_thread: bool,
) -> FilteringStageStateV1:
    run_path = _validate_run_dir(run_dir, create=True)
    path = _state_path(run_path)
    lock_fd = _acquire_writer_lock(path)
    connection: sqlite3.Connection | None = None
    try:
        desired = _desired_metadata(
            producer_config_payload=producer_config_payload,
            producer_retained_config_payload=producer_retained_config_payload,
            w1_config_payload=w1_config_payload,
        )
        if path.exists() or path.is_symlink():
            sidecars_present = any(
                Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm")
            )
            if sidecars_present:
                expected = _validate_existing_state_path(path)
                connection = _writable_connection(
                    path,
                    exclusive_locking=exclusive_locking,
                    check_same_thread=check_same_thread,
                )
                opened = path.lstat()
                if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
                    raise ValueError("filtering stage state changed before writer open")
                _validate_schema(connection)
                pending = _pending_metadata(
                    connection,
                    desired,
                    allow_missing=True,
                    legacy_retained_config_payload=w1_config_payload,
                )
            else:
                expected, pending = _validate_existing_readonly(
                    path,
                    desired,
                    allow_missing_metadata=True,
                    check_same_thread=check_same_thread,
                    legacy_retained_config_payload=w1_config_payload,
                )
                connection = _writable_connection(
                    path,
                    exclusive_locking=exclusive_locking,
                    check_same_thread=check_same_thread,
                )
                opened = path.lstat()
                if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
                    raise ValueError("filtering stage state changed before writer open")
                _validate_schema(connection)
        else:
            connection = _create_fresh_state(path, check_same_thread=check_same_thread)
            if not exclusive_locking:
                connection.execute("pragma locking_mode=NORMAL")
            pending = _pending_metadata(
                connection,
                desired,
                allow_missing=True,
                legacy_retained_config_payload=w1_config_payload,
            )
        if pending:
            with connection:
                connection.executemany(
                    "insert into stage_metadata(key, value_json) values (?, ?)",
                    pending,
                )
        filtering_artifacts._chmod_private_sqlite_files(path)
        return FilteringStageStateV1(
            connection,
            path,
            writer_lock_fd=lock_fd,
            immutable_frozen=False,
        )
    except BaseException:
        if connection is not None:
            connection.close()
        os.close(lock_fd)
        raise


def create_or_open_prefilter_state(
    run_dir: str | Path,
    producer_config_payload: Mapping[str, Any],
    *,
    check_same_thread: bool = True,
    legacy_retained_config_payload: Mapping[str, Any] | None = None,
) -> FilteringStageStateV1:
    """Create a fresh v1 state or validate and open an exact existing v1 state."""

    return _open_writer(
        run_dir,
        producer_config_payload=producer_config_payload,
        producer_retained_config_payload=legacy_retained_config_payload,
        w1_config_payload=None,
        exclusive_locking=True,
        check_same_thread=check_same_thread,
    )


def open_stage_state_readonly(
    run_dir: str | Path,
    prefilter_config_payload: Mapping[str, Any],
    *,
    check_same_thread: bool = True,
    legacy_retained_config_payload: Mapping[str, Any] | None = None,
) -> FilteringStageStateV1:
    """Open a frozen/stopped exact v1 state without creating WAL sidecars."""

    run_path = _validate_run_dir(run_dir, create=False)
    path = _state_path(run_path)
    desired = _desired_metadata(
        producer_config_payload=prefilter_config_payload,
        w1_config_payload=None,
    )
    _validate_existing_readonly(
        path,
        desired,
        allow_missing_metadata=False,
        check_same_thread=check_same_thread,
        legacy_retained_config_payload=legacy_retained_config_payload,
    )
    connection = _readonly_connection(
        path,
        check_same_thread=check_same_thread,
        immutable=True,
    )
    try:
        _validate_schema(connection)
        _pending_metadata(
            connection,
            desired,
            allow_missing=False,
            legacy_retained_config_payload=legacy_retained_config_payload,
        )
        return FilteringStageStateV1(
            connection,
            path,
            writer_lock_fd=None,
            immutable_frozen=True,
        )
    except BaseException:
        connection.close()
        raise


def open_closeout_stage_state_readonly(
    path: str | Path,
    *,
    check_same_thread: bool = True,
) -> FilteringStageStateV1:
    """Open a WAL-merged immutable v1 audit DB without inventing config input."""

    state_path = Path(path).expanduser().resolve(strict=False)
    filtering_artifacts.validate_private_filtering_run_dir(state_path.parent)
    _validate_existing_state_path(state_path)
    if any(Path(str(state_path) + suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("closeout stage state must be WAL-merged before immutable audit")
    connection = _readonly_connection(
        state_path,
        check_same_thread=check_same_thread,
        immutable=True,
    )
    try:
        _validate_schema(connection)
        metadata = {
            str(row["key"]): json.loads(str(row["value_json"]))
            for row in connection.execute(
                "select key, value_json from stage_metadata order by key"
            )
        }
        filtering_artifacts.validate_public_no_raw_payload(metadata)
        return FilteringStageStateV1(
            connection,
            state_path,
            writer_lock_fd=None,
            immutable_frozen=True,
        )
    except BaseException:
        connection.close()
        raise


def open_stage_state_for_w1(
    run_dir: str | Path,
    *,
    config_payload: Mapping[str, Any],
    prefilter_config_payload: Mapping[str, Any],
    exclusive_locking: bool = True,
    check_same_thread: bool = True,
) -> FilteringStageStateV1:
    """Open exact v1 stage state for W1 and bind immutable W1 metadata once."""

    run_path = _validate_run_dir(run_dir, create=False)
    if not _state_path(run_path).exists():
        raise ValueError("filtering stage state is missing")
    return _open_writer(
        run_path,
        producer_config_payload=prefilter_config_payload,
        producer_retained_config_payload=None,
        w1_config_payload=config_payload,
        exclusive_locking=exclusive_locking,
        check_same_thread=check_same_thread,
    )
