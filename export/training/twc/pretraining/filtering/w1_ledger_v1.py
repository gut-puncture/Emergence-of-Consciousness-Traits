"""Canonical storage and readback contract for the legacy W1 token ledger.

The scorer owns score-payload construction.  This module owns the ledger's
SQLite schema, binding metadata, global-ordinal commit decisions, and bounded
read surfaces used by scoring, controller status, and preserved-state audits.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from twc.repro.hashing import stable_json_hash
from twc.pretraining.filtering import artifacts as filtering_artifacts


ScoreRecord = tuple[Mapping[str, Any], float | None, str, str | None]
ScorePayloadBuilder = Callable[..., dict[str, Any]]


def _iter_sized_chunks(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    chunk_size = max(1, int(size))
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def read_ledger_state(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        """
        select next_commit_ordinal, accepted_token_count_lower_bound,
               accepted_token_target_reached, stop_ordinal, total_expected_rows,
               accepted_token_target_lower_bound, updated_at
        from ledger_state where id = 1
        """
    ).fetchone()
    if row is None:
        raise ValueError("accepted-token ledger state is missing")
    return {
        "next_commit_ordinal": int(row["next_commit_ordinal"]),
        "accepted_token_count_lower_bound": int(row["accepted_token_count_lower_bound"]),
        "accepted_token_target_reached": bool(row["accepted_token_target_reached"]),
        "stop_ordinal": None if row["stop_ordinal"] is None else int(row["stop_ordinal"]),
        "total_expected_rows": int(row["total_expected_rows"]),
        "accepted_token_target_lower_bound": int(row["accepted_token_target_lower_bound"]),
        "updated_at": float(row["updated_at"]),
    }


def _setup(conn: sqlite3.Connection) -> None:
    conn.execute("pragma busy_timeout=60000")
    try:
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma synchronous=normal")
    except sqlite3.DatabaseError:
        pass
    conn.execute(
        """
        create table if not exists ledger_metadata(
            key text primary key,
            value_json text not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists ledger_state(
            id integer primary key check (id = 1),
            next_commit_ordinal integer not null,
            accepted_token_count_lower_bound integer not null,
            accepted_token_target_reached integer not null,
            stop_ordinal integer,
            total_expected_rows integer not null,
            accepted_token_target_lower_bound integer not null,
            updated_at real not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists ledger_rows(
            global_ordinal integer primary key,
            shard_index integer not null,
            shard_row_index integer not null,
            candidate_id_hash text not null unique,
            w1_route text not null,
            accepted_token_count integer not null,
            score_json text not null,
            score_json_hash text not null,
            commit_decision text not null,
            producer_rank_assignment_hash text,
            created_at real not null,
            updated_at real not null
        )
        """
    )
    conn.execute("create index if not exists ledger_rows_shard_decision_idx on ledger_rows(shard_index, commit_decision)")
    conn.execute(
        "create index if not exists ledger_rows_shard_decision_row_idx "
        "on ledger_rows(shard_index, commit_decision, shard_row_index)"
    )
    conn.execute("create index if not exists ledger_rows_shard_route_idx on ledger_rows(shard_index, w1_route)")
    conn.execute("create index if not exists ledger_rows_decision_ordinal_idx on ledger_rows(commit_decision, global_ordinal)")


def connect_accepted_token_ledger(
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> sqlite3.Connection:
    ledger_path = Path(path).expanduser().resolve(strict=False)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ledger_path), timeout=60.0)
    conn.row_factory = sqlite3.Row
    _setup(conn)
    target = filtering_artifacts._positive_int(
        metadata.get("accepted_token_target_lower_bound"),
        "accepted_token_target_lower_bound",
    )
    total_expected_rows = filtering_artifacts._nonnegative_int(
        metadata.get("survivor_count", 0),
        "accepted-token ledger survivor_count",
    )
    metadata_payload = dict(metadata)
    metadata_payload["artifact_kind"] = filtering_artifacts.W1_ACCEPTED_TOKEN_LEDGER_KIND
    metadata_payload["schema_version"] = 1
    metadata_payload["accepted_token_target_lower_bound"] = target
    metadata_payload["survivor_count"] = total_expected_rows
    metadata_payload["raw_text_serialized"] = False
    metadata_payload["raw_url_serialized"] = False
    metadata_payload["token_serialized"] = False
    metadata_payload["ledger_binding_hash"] = stable_json_hash(
        {key: value for key, value in metadata_payload.items() if key != "ledger_binding_hash"}
    )
    filtering_artifacts.validate_public_no_raw_payload(metadata_payload)
    expected = {key: json.dumps(value, sort_keys=True) for key, value in sorted(metadata_payload.items())}
    now = time.time()
    with conn:
        for key, value_json in expected.items():
            conn.execute(
                "insert or ignore into ledger_metadata(key, value_json) values (?, ?)",
                (key, value_json),
            )
        existing = {
            str(row["key"]): str(row["value_json"])
            for row in conn.execute("select key, value_json from ledger_metadata")
        }
        for key, value_json in expected.items():
            if existing.get(key) != value_json:
                raise ValueError(f"accepted-token ledger metadata mismatch for {key}")
        unexpected = sorted(set(existing).difference(expected))
        if unexpected:
            raise ValueError(f"accepted-token ledger has unexpected metadata keys: {unexpected[:5]}")
        conn.execute(
            """
            insert or ignore into ledger_state(
                id, next_commit_ordinal, accepted_token_count_lower_bound,
                accepted_token_target_reached, stop_ordinal, total_expected_rows,
                accepted_token_target_lower_bound, updated_at
            ) values (1, 0, 0, 0, null, ?, ?, ?)
            """,
            (total_expected_rows, target, now),
        )
        state = read_ledger_state(conn)
        if int(state["total_expected_rows"]) != total_expected_rows:
            raise ValueError("accepted-token ledger survivor_count mismatch")
        if int(state["accepted_token_target_lower_bound"]) != target:
            raise ValueError("accepted-token ledger target mismatch")
    filtering_artifacts._chmod_private_sqlite_files(str(ledger_path))
    return conn


def _advance_locked(conn: sqlite3.Connection) -> dict[str, Any]:
    state = read_ledger_state(conn)
    target = int(state["accepted_token_target_lower_bound"])
    total_expected_rows = int(state["total_expected_rows"])
    next_ordinal = int(state["next_commit_ordinal"])
    accepted = int(state["accepted_token_count_lower_bound"])
    target_reached = bool(state["accepted_token_target_reached"])
    stop_ordinal = state["stop_ordinal"]
    if target_reached:
        if stop_ordinal is not None:
            conn.execute(
                """
                update ledger_rows
                set commit_decision = 'after_stop', updated_at = ?
                where global_ordinal > ? and commit_decision = 'pending'
                """,
                (time.time(), int(stop_ordinal)),
            )
        return read_ledger_state(conn)
    while next_ordinal < total_expected_rows:
        rows = list(
            conn.execute(
                """
                select global_ordinal, w1_route, accepted_token_count, commit_decision
                from ledger_rows
                where global_ordinal >= ?
                order by global_ordinal
                limit 4096
                """,
                (next_ordinal,),
            )
        )
        if not rows:
            break
        update_now = time.time()
        range_start = next_ordinal
        range_end: int | None = None
        consumed_rows = 0
        for row in rows:
            row_ordinal = int(row["global_ordinal"])
            if row_ordinal != next_ordinal:
                break
            if str(row["commit_decision"]) == "after_stop":
                raise ValueError("accepted-token ledger encountered after-stop row before target")
            if str(row["commit_decision"]) != "persist" and str(row["w1_route"]) == "w1_pass":
                accepted += int(row["accepted_token_count"])
            range_end = next_ordinal
            consumed_rows += 1
            if accepted >= target:
                target_reached = True
                stop_ordinal = next_ordinal
                next_ordinal += 1
                break
            next_ordinal += 1
        if range_end is not None:
            conn.execute(
                """
                update ledger_rows
                set commit_decision = 'persist', updated_at = ?
                where global_ordinal between ? and ? and commit_decision != 'persist'
                """,
                (update_now, int(range_start), int(range_end)),
            )
        if target_reached or consumed_rows < len(rows):
            break
    now = time.time()
    conn.execute(
        """
        update ledger_state
        set next_commit_ordinal = ?, accepted_token_count_lower_bound = ?,
            accepted_token_target_reached = ?, stop_ordinal = ?, updated_at = ?
        where id = 1
        """,
        (next_ordinal, accepted, 1 if target_reached else 0, stop_ordinal, now),
    )
    if target_reached and stop_ordinal is not None:
        conn.execute(
            """
            update ledger_rows
            set commit_decision = 'after_stop', updated_at = ?
            where global_ordinal > ? and commit_decision = 'pending'
            """,
            (now, int(stop_ordinal)),
        )
    return read_ledger_state(conn)


def submit_score_records(
    conn: sqlite3.Connection,
    *,
    score_records: Sequence[ScoreRecord],
    config_payload: Mapping[str, Any],
    w1_runtime_envelope_hash: str,
    global_ordinal_by_candidate_id: Mapping[str, int],
    global_ordinal_base: int,
    shard_index: int,
    rank_assignment_hash: str | None,
    build_score_payload: ScorePayloadBuilder,
) -> dict[str, Any]:
    if not score_records:
        return read_ledger_state(conn)
    now = time.time()
    ordinal_base = filtering_artifacts._nonnegative_int(
        global_ordinal_base,
        "accepted-token ledger global ordinal base",
    )
    with conn:
        state = read_ledger_state(conn)
        target_reached = bool(state["accepted_token_target_reached"])
        stop_ordinal = state["stop_ordinal"]
        prepared_rows: list[tuple[int, str, str, int, str, str, str, int]] = []
        for survivor_row, score, route, error_class in score_records:
            payload = build_score_payload(
                survivor_row,
                config_payload=config_payload,
                score=score,
                route=route,
                error_class=error_class,
                w1_runtime_envelope_hash=w1_runtime_envelope_hash,
            )
            candidate_id = str(payload["candidate_id_hash"])
            if candidate_id not in global_ordinal_by_candidate_id:
                raise ValueError("accepted-token ledger missing candidate ordinal")
            global_ordinal = int(global_ordinal_by_candidate_id[candidate_id])
            accepted_tokens = int(survivor_row.get("core_proxy_token_count", 0) or 0) if route == "w1_pass" else 0
            score_json = json.dumps(payload, sort_keys=True)
            score_json_hash = stable_json_hash(payload)
            decision = "pending"
            if target_reached and stop_ordinal is not None and global_ordinal > int(stop_ordinal):
                decision = "after_stop"
            shard_row_index = global_ordinal - ordinal_base
            if shard_row_index < 0:
                raise ValueError("accepted-token ledger shard row index is negative")
            prepared_rows.append(
                (global_ordinal, candidate_id, str(route), int(accepted_tokens), score_json,
                 score_json_hash, decision, int(shard_row_index))
            )
        existing_by_ordinal: dict[int, tuple[str, str]] = {}
        ordinals = [row[0] for row in prepared_rows]
        for chunk in _iter_sized_chunks(list(dict.fromkeys(ordinals)), 900):
            placeholders = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"select global_ordinal, candidate_id_hash, score_json from ledger_rows "
                f"where global_ordinal in ({placeholders})",
                tuple(chunk),
            ):
                existing_by_ordinal[int(row["global_ordinal"])] = (
                    str(row["candidate_id_hash"]), str(row["score_json"])
                )
        existing_by_candidate_id: dict[str, tuple[int, str]] = {}
        candidate_ids = [row[1] for row in prepared_rows]
        for chunk in _iter_sized_chunks(list(dict.fromkeys(candidate_ids)), 900):
            placeholders = ",".join("?" for _ in chunk)
            for row in conn.execute(
                f"select candidate_id_hash, global_ordinal, score_json from ledger_rows "
                f"where candidate_id_hash in ({placeholders})",
                tuple(chunk),
            ):
                existing_by_candidate_id[str(row["candidate_id_hash"])] = (
                    int(row["global_ordinal"]), str(row["score_json"])
                )
        rows_to_insert: list[tuple[int, int, int, str, str, int, str, str, str, str | None, float, float]] = []
        seen_ordinals: dict[int, tuple[str, str]] = {}
        seen_candidate_ids: dict[str, tuple[int, str]] = {}
        for global_ordinal, candidate_id, route, accepted_tokens, score_json, score_json_hash, decision, shard_row_index in prepared_rows:
            batch_ordinal = seen_ordinals.get(global_ordinal)
            if batch_ordinal is not None:
                if batch_ordinal != (candidate_id, score_json):
                    raise ValueError("conflicting accepted-token ledger row for global ordinal")
                continue
            batch_candidate = seen_candidate_ids.get(candidate_id)
            if batch_candidate is not None:
                if batch_candidate != (global_ordinal, score_json):
                    raise ValueError("conflicting accepted-token ledger row for candidate")
                continue
            existing = existing_by_ordinal.get(global_ordinal)
            if existing is not None:
                if existing != (candidate_id, score_json):
                    raise ValueError("conflicting accepted-token ledger row for global ordinal")
                seen_ordinals[global_ordinal] = (candidate_id, score_json)
                seen_candidate_ids[candidate_id] = (global_ordinal, score_json)
                continue
            existing_by_id = existing_by_candidate_id.get(candidate_id)
            if existing_by_id is not None:
                if existing_by_id != (global_ordinal, score_json):
                    raise ValueError("conflicting accepted-token ledger row for candidate")
                seen_ordinals[global_ordinal] = (candidate_id, score_json)
                seen_candidate_ids[candidate_id] = (global_ordinal, score_json)
                continue
            rows_to_insert.append(
                (global_ordinal, int(shard_index), int(shard_row_index), candidate_id, route,
                 accepted_tokens, score_json, score_json_hash, decision, rank_assignment_hash, now, now)
            )
            seen_ordinals[global_ordinal] = (candidate_id, score_json)
            seen_candidate_ids[candidate_id] = (global_ordinal, score_json)
        if rows_to_insert:
            conn.executemany(
                """
                insert into ledger_rows(
                    global_ordinal, shard_index, shard_row_index, candidate_id_hash,
                    w1_route, accepted_token_count, score_json, score_json_hash,
                    commit_decision, producer_rank_assignment_hash, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )
        return _advance_locked(conn)


def persisted_score_payloads_for_shard_delta(
    conn: sqlite3.Connection,
    *,
    shard_index: int,
    after_shard_row_index: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    where = "shard_index = ? and commit_decision = 'persist'"
    params: list[Any] = [int(shard_index)]
    if after_shard_row_index is not None:
        where += " and shard_row_index > ?"
        params.append(filtering_artifacts._nonnegative_int(after_shard_row_index, "after_shard_row_index"))
    return [
        (int(row["shard_row_index"]), json.loads(str(row["score_json"])))
        for row in conn.execute(
            f"select shard_row_index, score_json from ledger_rows where {where} order by shard_row_index",
            tuple(params),
        )
    ]


def row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts = {
        str(row["commit_decision"]): int(row["row_count"])
        for row in conn.execute(
            "select commit_decision, count(*) as row_count from ledger_rows group by commit_decision"
        )
    }
    return {name: int(counts.get(name, 0)) for name in ("persist", "pending", "after_stop")}


def candidate_ids_for_shard(conn: sqlite3.Connection, shard_index: int) -> set[str]:
    return {
        str(row["candidate_id_hash"])
        for row in conn.execute(
            "select candidate_id_hash from ledger_rows where shard_index = ?",
            (int(shard_index),),
        )
    }


def shard_summary(conn: sqlite3.Connection, shard_index: int) -> dict[str, Any]:
    row = conn.execute(
        "select count(*) as row_count, coalesce(sum(accepted_token_count), 0) as accepted_token_count "
        "from ledger_rows where shard_index = ?",
        (int(shard_index),),
    ).fetchone()
    route_counts = {
        str(item["w1_route"]): int(item["row_count"])
        for item in conn.execute(
            "select w1_route, count(*) as row_count from ledger_rows "
            "where shard_index = ? group by w1_route",
            (int(shard_index),),
        )
    }
    decision_counts = {
        str(item["commit_decision"]): int(item["row_count"])
        for item in conn.execute(
            "select commit_decision, count(*) as row_count from ledger_rows "
            "where shard_index = ? group by commit_decision",
            (int(shard_index),),
        )
    }
    return {
        "row_count": 0 if row is None else int(row["row_count"]),
        "accepted_token_count_lower_bound": 0 if row is None else int(row["accepted_token_count"]),
        "route_counts": filtering_artifacts._sorted_counter(Counter(route_counts)),
        "commit_decision_counts": filtering_artifacts._sorted_counter(Counter(decision_counts)),
    }


def read_compact_state_for_run_spec(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    paths: set[Path] = set()
    for command in run_spec.get("score_commands", []):
        argv = list(command.get("argv", []))
        if "--accepted-token-ledger" not in argv:
            continue
        index = argv.index("--accepted-token-ledger")
        if index + 1 >= len(argv):
            return {"status": "invalid_binding"}
        paths.add(Path(argv[index + 1]).expanduser().resolve(strict=False))
    if not paths:
        return {"status": "not_configured"}
    if len(paths) != 1:
        return {"status": "invalid_binding"}
    ledger_path = paths.pop()
    if not ledger_path.is_file():
        return {"status": "missing"}
    try:
        connection = sqlite3.connect(ledger_path.as_uri() + "?mode=ro", uri=True, timeout=0.05)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("pragma query_only=on")
            connection.execute("pragma busy_timeout=50")
            try:
                state = read_ledger_state(connection)
            except ValueError:
                return {"status": "invalid_binding"}
        finally:
            connection.close()
    except sqlite3.Error:
        return {"status": "busy_or_unavailable"}
    return {"status": "available", **state}


def _open_closeout_reader(path: str | Path) -> sqlite3.Connection:
    ledger_path = Path(path).expanduser().resolve(strict=True)
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise ValueError("closeout accepted-token ledger must be a regular file")
    connection = sqlite3.connect(
        ledger_path.as_uri() + "?mode=ro&immutable=1",
        uri=True,
        timeout=5.0,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("pragma query_only=on")
    connection.execute("pragma busy_timeout=5000")
    tables = {
        str(row[0])
        for row in connection.execute(
            "select name from sqlite_master where type='table' and name like 'ledger_%'"
        )
    }
    if tables != {"ledger_metadata", "ledger_state", "ledger_rows"}:
        connection.close()
        raise ValueError("closeout accepted-token ledger schema mismatch")
    return connection


def read_closeout_ledger_snapshot(path: str | Path) -> dict[str, Any]:
    """Read compact prefix invariants from an immutable derived ledger view."""

    connection = _open_closeout_reader(path)
    try:
        state = read_ledger_state(connection)
        decision_counts = {
            str(row["commit_decision"]): int(row["row_count"])
            for row in connection.execute(
                "select commit_decision, count(*) as row_count from ledger_rows group by commit_decision"
            )
        }
        unexpected = set(decision_counts).difference({"persist", "pending", "after_stop"})
        if unexpected:
            raise ValueError("closeout accepted-token ledger commit decision mismatch")
        next_ordinal = int(state["next_commit_ordinal"])
        persisted_count = int(decision_counts.get("persist", 0))
        persisted_bounds = connection.execute(
            "select min(global_ordinal), max(global_ordinal) from ledger_rows where commit_decision='persist'"
        ).fetchone()
        if next_ordinal == 0:
            prefix_shape_valid = persisted_count == 0
        else:
            prefix_shape_valid = (
                persisted_count == next_ordinal
                and persisted_bounds is not None
                and int(persisted_bounds[0]) == 0
                and int(persisted_bounds[1]) == next_ordinal - 1
            )
        if not prefix_shape_valid:
            raise ValueError("closeout accepted-token ledger persisted prefix is not exact")
        persisted_accepted = connection.execute(
            "select coalesce(sum(accepted_token_count), 0) from ledger_rows where commit_decision='persist'"
        ).fetchone()
        if int(persisted_accepted[0]) != int(state["accepted_token_count_lower_bound"]):
            raise ValueError("closeout accepted-token ledger accepted subtotal mismatch")
        metadata = {
            str(row["key"]): json.loads(str(row["value_json"]))
            for row in connection.execute(
                "select key, value_json from ledger_metadata order by key"
            )
        }
        if metadata.get("artifact_kind") != filtering_artifacts.W1_ACCEPTED_TOKEN_LEDGER_KIND:
            raise ValueError("closeout accepted-token ledger metadata kind mismatch")
        filtering_artifacts.validate_public_no_raw_payload(metadata)
        return {
            **state,
            "persisted_row_count": persisted_count,
            "pending_row_count": int(decision_counts.get("pending", 0)),
            "after_stop_row_count": int(decision_counts.get("after_stop", 0)),
            "metadata_hash": stable_json_hash(metadata),
            "binding_metadata": {
                key: metadata.get(key)
                for key in (
                    "sharded_w1_handoff_hash",
                    "config_hash",
                    "prefilter_producer_config_hash",
                    "selected_policy_hash",
                    "selected_policy_report_hash",
                    "source_metadata_hash",
                    "w1_runtime_envelope_hash",
                    "survivor_count",
                    "shard_count",
                    "shard_survivor_counts_hash",
                    "rank_assignment_hash",
                )
            },
        }
    finally:
        connection.close()


def iter_closeout_persisted_rows(
    path: str | Path,
    *,
    start_ordinal: int,
    end_ordinal_exclusive: int,
) -> Iterator[dict[str, Any]]:
    """Yield normalized scientific facts for one bounded ordinal range."""

    start = filtering_artifacts._nonnegative_int(start_ordinal, "closeout ledger start ordinal")
    end = filtering_artifacts._nonnegative_int(
        end_ordinal_exclusive, "closeout ledger end ordinal"
    )
    if end < start:
        raise ValueError("closeout ledger ordinal range is reversed")
    connection = _open_closeout_reader(path)
    try:
        expected = start
        for row in connection.execute(
            "select global_ordinal, candidate_id_hash, w1_route, accepted_token_count, "
            "score_json, score_json_hash, commit_decision "
            "from ledger_rows where global_ordinal >= ? and global_ordinal < ? "
            "order by global_ordinal",
            (start, end),
        ):
            ordinal = int(row["global_ordinal"])
            if ordinal != expected or str(row["commit_decision"]) != "persist":
                raise ValueError("closeout accepted-token ledger persisted prefix has a gap")
            candidate_id = filtering_artifacts._require_sha256(
                row["candidate_id_hash"], "closeout ledger candidate_id_hash"
            )
            route = str(row["w1_route"])
            if route not in {"w1_pass", "w1_reject", "w1_error"}:
                raise ValueError("closeout accepted-token ledger route mismatch")
            accepted = filtering_artifacts._nonnegative_int(
                row["accepted_token_count"], "closeout ledger accepted token count"
            )
            if route != "w1_pass" and accepted != 0:
                raise ValueError("closeout rejected ledger row has accepted tokens")
            score_payload = json.loads(str(row["score_json"]))
            if not isinstance(score_payload, dict):
                raise ValueError("closeout ledger score payload must be an object")
            score_hash = filtering_artifacts._require_sha256(
                row["score_json_hash"], "closeout ledger score hash"
            )
            if (
                score_payload.get("candidate_id_hash") != candidate_id
                or score_payload.get("w1_route") != route
                or stable_json_hash(score_payload) != score_hash
            ):
                raise ValueError("closeout ledger score payload binding mismatch")
            filtering_artifacts.validate_public_no_raw_payload(score_payload)
            yield {
                "global_ordinal": ordinal,
                "candidate_id_hash": candidate_id,
                "score_hash": score_hash,
                "score_payload_hash": stable_json_hash(score_payload),
                "w1_route": route,
                "core_proxy_token_count": accepted if route == "w1_pass" else 0,
                "accepted_proxy_token_count": accepted,
                "singleton_accepted_proxy_token_subtotal": accepted,
            }
            expected += 1
        if expected != end:
            raise ValueError("closeout accepted-token ledger persisted prefix has a gap")
    finally:
        connection.close()


def closeout_rows_for_candidate_ids(
    path: str | Path,
    candidate_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Read bounded ledger corroboration for one staged compact shard."""

    normalized = list(
        dict.fromkeys(
            filtering_artifacts._require_sha256(value, "closeout candidate_id_hash")
            for value in candidate_ids
        )
    )
    connection = _open_closeout_reader(path)
    found: dict[str, dict[str, Any]] = {}
    try:
        for chunk in _iter_sized_chunks(normalized, 900):
            placeholders = ",".join("?" for _ in chunk)
            for row in connection.execute(
                "select global_ordinal, candidate_id_hash, w1_route, accepted_token_count, "
                "score_json, score_json_hash, commit_decision from ledger_rows "
                f"where candidate_id_hash in ({placeholders})",
                tuple(chunk),
            ):
                candidate_id = str(row["candidate_id_hash"])
                payload = json.loads(str(row["score_json"]))
                score_hash = filtering_artifacts._require_sha256(
                    row["score_json_hash"], "closeout ledger score hash"
                )
                if (
                    not isinstance(payload, dict)
                    or payload.get("candidate_id_hash") != candidate_id
                    or payload.get("w1_route") != row["w1_route"]
                    or stable_json_hash(payload) != score_hash
                ):
                    raise ValueError("closeout ledger candidate lookup binding mismatch")
                found[candidate_id] = {
                    "global_ordinal": int(row["global_ordinal"]),
                    "candidate_id_hash": candidate_id,
                    "w1_route": str(row["w1_route"]),
                    "accepted_proxy_token_count": int(row["accepted_token_count"]),
                    "score_hash": score_hash,
                    "commit_decision": str(row["commit_decision"]),
                }
        return found
    finally:
        connection.close()
