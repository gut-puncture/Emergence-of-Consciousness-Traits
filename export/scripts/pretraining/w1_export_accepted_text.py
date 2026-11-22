"""Export the W1-accepted text that becomes the pretraining corpus.

This is the step between the W1 filtering run and ``twc-data``. It reads the
W1 accepted-token ledger, the sharded prefilter handoff and each shard's
``filtering_stage_state.sqlite`` evidence, and writes private raw-text JSONL
parts plus a no-raw ``manifest_no_raw.json``. The A0 corpus used
``--content-mode verified-core``, which keeps only each candidate's verified
core span. Its manifest is the input of
``twc-data build accepted --input-mode verified-selected-core``.

The output contains raw web text. Write it to a private directory outside the
repository; only the no-raw manifest and progress files may be shared.

This file is the exporter that produced the A0 corpus on 2026-07-20, kept as a
script with its export logic unchanged (it reads the stage and ledger SQLite
tables directly, which package code may not do). It was moved into the
repository on 2026-10-10 from a private tools folder; only the import header,
this docstring and the removal of the built-in self-test (now
``tests/test_filtering_accepted_text_export.py``) changed. Run it with the
locked environment: ``uv run --locked python scripts/pretraining/w1_export_accepted_text.py``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from twc.pretraining.filtering.offline_v1 import (
    validate_closeout_handoff_payload,
    validate_closeout_shard_support_files,
)
from twc.pretraining.filtering.w1_ledger_v1 import read_closeout_ledger_snapshot


DEFAULT_TARGET = 3_300_000_000
DEFAULT_PART_TOKENS = 100_000_000
DEFAULT_BATCH_SIZE = 1_000
CORE_EXPORT_KIND = "twc_w1_private_accepted_core_export_v1"
BLOCK_KINDS = {"heading", "prose", "list", "table", "fenced_code", "indented_code", "html_pre", "html_block", "log_run", "stack_trace", "ui_chrome", "oversized_soft_wrapped_prose", "oversized_unsegmentable"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _json_hash(payload: Any) -> str:
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if os.name == "posix":
        os.chmod(tmp, 0o600)
    tmp.replace(path)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _load_handoff(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("handoff must be an object")
    return validate_closeout_handoff_payload(payload)


def _ledger_state(ledger: Path) -> dict[str, int | None]:
    with _connect_ro(ledger) as conn:
        row = conn.execute(
            """
            select next_commit_ordinal, accepted_token_count_lower_bound,
                   accepted_token_target_reached, stop_ordinal,
                   total_expected_rows, accepted_token_target_lower_bound
            from ledger_state
            """
        ).fetchone()
    if row is None:
        raise ValueError("ledger_state row missing")
    return {
        "next_commit_ordinal": int(row["next_commit_ordinal"]),
        "accepted_token_count_lower_bound": int(row["accepted_token_count_lower_bound"]),
        "accepted_token_target_reached": int(row["accepted_token_target_reached"]),
        "stop_ordinal": None if row["stop_ordinal"] is None else int(row["stop_ordinal"]),
        "total_expected_rows": int(row["total_expected_rows"]),
        "accepted_token_target_lower_bound": int(row["accepted_token_target_lower_bound"]),
    }


def _iter_accepted_ledger_rows(
    ledger: Path,
    *,
    shard_index: int,
    next_commit_ordinal: int,
    resume_after_global_ordinal: int,
) -> Iterable[sqlite3.Row]:
    with _connect_ro(ledger) as conn:
        cursor = conn.execute(
            """
            select global_ordinal, shard_index, shard_row_index, candidate_id_hash,
                   w1_route, accepted_token_count, score_json_hash, commit_decision
            from ledger_rows
            where shard_index = ?
              and global_ordinal < ?
              and global_ordinal > ?
              and commit_decision = 'persist'
              and w1_route = 'w1_pass'
            order by global_ordinal
            """,
            (int(shard_index), int(next_commit_ordinal), int(resume_after_global_ordinal)),
        )
        for row in cursor:
            yield row


def _load_queue_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("raw_text_serialized") is not False:
                    raise ValueError(f"queue row is not no-raw in {path}")
                rows.append(row)
    return rows


def _fetch_evidence_batch(conn: sqlite3.Connection, evidence_ids: list[str], *, include_candidate: bool = False) -> dict[str, sqlite3.Row]:
    if not evidence_ids:
        return {}
    placeholders = ",".join("?" for _ in evidence_ids)
    candidate_projection = ", c.candidate_json" if include_candidate else ""
    candidate_join = " join prefilter_candidates c on c.candidate_id_hash=e.candidate_id_hash" if include_candidate else ""
    rows = conn.execute(
        f"select e.private_evidence_row_id,e.candidate_id_hash,e.evidence_text_sha256,"
        f"e.private_evidence_text{candidate_projection} from private_w1_evidence e{candidate_join} "
        f"where e.private_evidence_row_id in ({placeholders})", evidence_ids,
    ).fetchall()
    return {str(row["private_evidence_row_id"]): row for row in rows}


def _verified_core_fields(candidate_json: str, queue: dict[str, Any], evidence: sqlite3.Row) -> dict[str, Any]:
    candidate = json.loads(candidate_json)
    candidate_id = str(queue["candidate_id_hash"])
    if not isinstance(candidate, dict) or candidate.get("candidate_id_hash") != candidate_id:
        raise ValueError("candidate JSON identity mismatch")
    identity_fields = ("source_document_id_hash", "segmentation_receipt_hash", "candidate_policy_hash",
                       "core_block_indexes", "core_char_start", "core_char_end", "evidence_char_start",
                       "evidence_char_end", "core_text_sha256", "evidence_text_sha256")
    if _json_hash({field: candidate.get(field) for field in identity_fields}) != candidate_id:
        raise ValueError("candidate ID hash mismatch")
    for field in ("source_document_id_hash", "evidence_text_sha256", "core_text_sha256", "core_proxy_token_count"):
        if candidate.get(field) != queue.get(field):
            raise ValueError(f"candidate/queue {field} mismatch")
    evidence_start, evidence_end = int(candidate["evidence_char_start"]), int(candidate["evidence_char_end"])
    core_start, core_end = int(candidate["core_char_start"]), int(candidate["core_char_end"])
    text = str(evidence["private_evidence_text"])
    if not evidence_start <= core_start < core_end <= evidence_end or len(text) != evidence_end - evidence_start:
        raise ValueError("candidate evidence/core offsets mismatch")
    core = text[core_start - evidence_start : core_end - evidence_start]
    indexes, block_ids, block_kinds = (candidate.get(name) for name in ("core_block_indexes", "block_id_hashes", "block_kinds"))
    if (not isinstance(indexes, list) or not indexes or indexes != list(range(indexes[0], indexes[-1] + 1))
            or not isinstance(block_ids, list) or len(block_ids) != len(indexes)
            or not all(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower()) for value in block_ids)
            or not isinstance(block_kinds, list) or len(block_kinds) != len(indexes)
            or any(kind not in BLOCK_KINDS for kind in block_kinds)):
        raise ValueError("candidate contiguous block identities/kinds mismatch")
    if _sha256_text(text) != candidate["evidence_text_sha256"] or _sha256_text(core) != candidate["core_text_sha256"]:
        raise ValueError("candidate evidence/core content hash mismatch")
    return {"text": core, "core_char_count": len(core), "core_byte_count": len(core.encode("utf-8")),
            "core_block_indexes": indexes, "block_id_hashes": block_ids, "block_kinds": block_kinds}


class PartWriter:
    def __init__(self, output_dir: Path, part_index: int) -> None:
        self.output_dir = output_dir
        self.part_index = int(part_index)
        self.final_path = output_dir / f"accepted_text_part_{self.part_index:05d}.jsonl.gz"
        self.tmp_path = output_dir / f"accepted_text_part_{self.part_index:05d}.jsonl.gz.part"
        if self.final_path.exists():
            raise FileExistsError(f"part already exists: {self.final_path}")
        if self.tmp_path.exists():
            raise FileExistsError(f"partial part exists; inspect before resuming: {self.tmp_path}")
        self.raw_jsonl_sha256 = hashlib.sha256()
        self.row_count = 0
        self.token_count = 0
        self.first_global_ordinal: int | None = None
        self.last_global_ordinal: int | None = None
        self.handle = gzip.open(self.tmp_path, "wt", encoding="utf-8", compresslevel=1)
        if os.name == "posix":
            os.chmod(self.tmp_path, 0o600)

    def write_row(self, payload: dict[str, Any]) -> None:
        global_ordinal = int(payload["global_ordinal"])
        token_count = int(payload["core_proxy_token_count"])
        encoded = (json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self.raw_jsonl_sha256.update(encoded)
        self.handle.write(encoded.decode("utf-8"))
        self.row_count += 1
        self.token_count += token_count
        if self.first_global_ordinal is None:
            self.first_global_ordinal = global_ordinal
        self.last_global_ordinal = global_ordinal

    def close(self) -> dict[str, Any] | None:
        self.handle.close()
        if self.row_count == 0:
            self.tmp_path.unlink(missing_ok=True)
            return None
        self.tmp_path.replace(self.final_path)
        if os.name == "posix":
            os.chmod(self.final_path, 0o600)
        return {
            "part_index": self.part_index,
            "relative_path": self.final_path.name,
            "bytes": int(self.final_path.stat().st_size),
            "sha256": _file_sha256(self.final_path),
            "raw_jsonl_sha256": self.raw_jsonl_sha256.hexdigest(),
            "row_count": int(self.row_count),
            "accepted_token_count_lower_bound": int(self.token_count),
            "first_global_ordinal": int(self.first_global_ordinal),
            "last_global_ordinal": int(self.last_global_ordinal),
        }


def _load_resume(
    manifest_path: Path,
    artifact_kind: str,
    *,
    export_identity_hash: str,
    output_dir: Path,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    if not manifest_path.exists():
        return -1, 0, 0, []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != artifact_kind:
        raise ValueError("existing manifest artifact_kind mismatch")
    if manifest.get("manifest_hash") != _json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    ):
        raise ValueError("existing manifest self-hash mismatch")
    if manifest.get("export_identity_hash") != export_identity_hash:
        raise ValueError("existing manifest export identity mismatch")
    parts = list(manifest.get("parts") or [])
    rows = 0
    tokens = 0
    last_global = -1
    for expected_index, part in enumerate(parts):
        if not isinstance(part, dict) or int(part.get("part_index", -1)) != expected_index:
            raise ValueError("existing manifest part ordering mismatch")
        relative = str(part.get("relative_path") or "")
        expected_name = f"accepted_text_part_{expected_index:05d}.jsonl.gz"
        if relative != expected_name:
            raise ValueError("existing manifest part path mismatch")
        path = output_dir / relative
        if not path.is_file() or int(path.stat().st_size) != int(part.get("bytes", -1)):
            raise ValueError("existing manifest part file mismatch")
        if _file_sha256(path) != part.get("sha256"):
            raise ValueError("existing manifest part hash mismatch")
        first = int(part.get("first_global_ordinal", -1))
        current_last = int(part.get("last_global_ordinal", -1))
        if first <= last_global or current_last < first:
            raise ValueError("existing manifest ordinal coverage mismatch")
        last_global = current_last
        rows += int(part.get("row_count", 0))
        tokens += int(part.get("accepted_token_count_lower_bound", 0))
    if (
        int(manifest.get("part_count", -1)) != len(parts)
        or int(manifest.get("last_global_ordinal", -1)) != last_global
        or int(manifest.get("accepted_row_count", -1)) != rows
        or int(manifest.get("accepted_token_count_lower_bound", -1)) != tokens
    ):
        raise ValueError("existing manifest aggregate mismatch")
    return last_global, rows, tokens, parts


def _validate_authority(
    *, handoff_path: Path, supervisor_dir: Path, ledger_path: Path
) -> tuple[dict[str, Any], dict[str, Any], str]:
    handoff = _load_handoff(handoff_path)
    support_facts: list[dict[str, Any]] = []
    for shard in handoff["shards"]:
        shard_dir = supervisor_dir / str(shard["shard_run_dir_name"])
        support_facts.append(
            validate_closeout_shard_support_files(
                files={
                    name: shard_dir / name
                    for name in (
                        "prefilter_cpu_receipt.json",
                        "w1_survivor_queue.jsonl.ready",
                        "w1_survivor_queue.ids.tsv.ready",
                    )
                },
                shard=shard,
            )
        )
    ledger = read_closeout_ledger_snapshot(ledger_path)
    binding = ledger["binding_metadata"]
    if binding.get("sharded_w1_handoff_hash") != handoff["handoff_hash"]:
        raise ValueError("accepted-token ledger handoff binding mismatch")
    if int(binding.get("survivor_count", -1)) != int(handoff["survivor_count"]):
        raise ValueError("accepted-token ledger survivor count mismatch")
    if int(binding.get("shard_count", -1)) != int(handoff["shard_count"]):
        raise ValueError("accepted-token ledger shard count mismatch")
    if int(ledger["total_expected_rows"]) != int(handoff["survivor_count"]):
        raise ValueError("accepted-token ledger expected-row count mismatch")
    if binding.get("prefilter_producer_config_hash") != handoff.get("config_hash"):
        raise ValueError("accepted-token ledger prefilter producer config binding mismatch")
    if binding.get("source_metadata_hash") != handoff.get("source_metadata_hash"):
        raise ValueError("accepted-token ledger source metadata binding mismatch")
    return handoff, ledger, _json_hash(
        [
            {
                "shard_index": fact["shard_index"],
                "survivor_count": fact["survivor_count"],
                "support_fact_root": fact["support_fact_root"],
            }
            for fact in support_facts
        ]
    )


def export(args: argparse.Namespace) -> dict[str, Any]:
    handoff_path = Path(args.handoff)
    supervisor_dir = Path(args.supervisor_dir)
    ledger_path = Path(args.ledger)
    output_dir = Path(args.output_dir)
    target_tokens = int(args.target_tokens)
    part_token_target = int(args.part_token_target)
    batch_size = int(args.batch_size)
    content_mode = getattr(args, "content_mode", "evidence")
    if content_mode not in {"evidence", "verified-core"}:
        raise ValueError("unsupported content mode")
    export_kind = CORE_EXPORT_KIND if content_mode == "verified-core" else "twc_w1_private_accepted_text_export_v1"
    if target_tokens <= 0 or part_token_target <= 0 or batch_size <= 0:
        raise ValueError("target, part token target, and batch size must be positive")
    handoff, ledger_snapshot, shard_support_root = _validate_authority(
        handoff_path=handoff_path,
        supervisor_dir=supervisor_dir,
        ledger_path=ledger_path,
    )
    state = {
        key: ledger_snapshot[key]
        for key in (
            "next_commit_ordinal",
            "accepted_token_count_lower_bound",
            "accepted_token_target_reached",
            "stop_ordinal",
            "total_expected_rows",
            "accepted_token_target_lower_bound",
        )
    }
    if int(state["accepted_token_count_lower_bound"]) < target_tokens:
        raise ValueError("ledger accepted-token lower bound is below requested export target")
    if int(state["next_commit_ordinal"]) <= 0:
        raise ValueError("ledger next_commit_ordinal is empty")
    export_identity_hash = _json_hash(
        {
            "artifact_kind": export_kind,
            "content_mode": content_mode,
            "handoff_hash": handoff["handoff_hash"],
            "ledger_metadata_hash": ledger_snapshot["metadata_hash"],
            "ledger_state": state,
            "shard_support_root": shard_support_root,
            "target_token_count_lower_bound": target_tokens,
            "part_token_target": part_token_target,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(output_dir, 0o700)

    manifest_path = output_dir / "manifest_no_raw.json"
    progress_path = output_dir / "progress_no_raw.json"
    resume_after_global_ordinal, accepted_rows, accepted_tokens, parts = (
        _load_resume(
            manifest_path,
            export_kind,
            export_identity_hash=export_identity_hash,
            output_dir=output_dir,
        )
        if args.resume
        else (-1, 0, 0, [])
    )
    part_index = len(parts)
    writer: PartWriter | None = None
    start_time = time.time()
    new_parts_this_call = 0
    interrupt_after_parts = getattr(args, "interrupt_after_parts", None)

    def current_manifest(status: str) -> dict[str, Any]:
        payload = {
            "artifact_kind": export_kind,
            "schema_version": 1,
            "status": status,
            "created_at": start_time,
            "output_dir": str(output_dir),
            "handoff_path": str(handoff_path),
            "supervisor_dir": str(supervisor_dir),
            "ledger_path": str(ledger_path),
            "handoff_hash": handoff.get("handoff_hash"),
            "handoff_shard_count": int(handoff.get("shard_count")),
            "handoff_survivor_count": int(handoff.get("survivor_count")),
            "ledger_metadata_hash": ledger_snapshot["metadata_hash"],
            "shard_support_root": shard_support_root,
            "export_identity_hash": export_identity_hash,
            "ledger_state": state,
            "target_token_count_lower_bound": target_tokens,
            "accepted_token_count_lower_bound": accepted_tokens,
            "accepted_row_count": accepted_rows,
            "last_global_ordinal": resume_after_global_ordinal,
            "part_count": len(parts),
            "parts": parts,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
            "private_raw_text_files_created": True,
            "private_raw_text_files_are_not_public_artifacts": True,
            "text_field_name_in_private_jsonl": "text",
            "validation": {
                "ledger_queue_candidate_binding_checked": True,
                "queue_private_evidence_id_checked": True,
                "private_evidence_candidate_binding_checked": True,
                "private_evidence_text_sha256_checked": True,
                "global_ordinal_order": "ascending",
                "stopped_after_target_breached": accepted_tokens >= target_tokens,
            },
        }
        if content_mode == "verified-core":
            payload["content_mode"] = content_mode
        payload["manifest_hash"] = _json_hash(payload)
        return payload

    def preserve_part_progress() -> None:
        nonlocal new_parts_this_call
        _atomic_write_json(manifest_path, current_manifest("running"))
        new_parts_this_call += 1
        if interrupt_after_parts is not None and new_parts_this_call >= interrupt_after_parts:
            raise InterruptedError("synthetic interruption after durable export part")

    def write_progress(status: str, *, current_shard: int | None = None) -> None:
        payload = {
            "artifact_kind": ("twc_w1_private_accepted_core_export_progress_v1" if content_mode == "verified-core" else "twc_w1_private_accepted_text_export_progress_v1"),
            "status": status,
            "updated_at": time.time(),
            "elapsed_seconds": time.time() - start_time,
            "output_dir": str(output_dir),
            "target_token_count_lower_bound": target_tokens,
            "accepted_token_count_lower_bound": accepted_tokens,
            "accepted_row_count": accepted_rows,
            "last_global_ordinal": resume_after_global_ordinal,
            "part_count": len(parts),
            "current_part_index": None if writer is None else writer.part_index,
            "current_shard": current_shard,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        payload["progress_hash"] = _json_hash(payload)
        _atomic_write_json(progress_path, payload)

    write_progress("running")
    next_commit_ordinal = int(state["next_commit_ordinal"])
    shards = sorted(handoff["shards"], key=lambda item: int(item["shard_index"]))

    for shard in shards:
        if accepted_tokens >= target_tokens:
            break
        shard_index = int(shard["shard_index"])
        shard_dir = supervisor_dir / str(shard["shard_run_dir_name"])
        db_path = shard_dir / "filtering_stage_state.sqlite"
        queue_path = shard_dir / "w1_survivor_queue.jsonl.ready"
        if not db_path.exists() or not queue_path.exists():
            raise FileNotFoundError(f"missing shard DB or queue for shard {shard_index}")

        ledger_rows = list(
            _iter_accepted_ledger_rows(
                ledger_path,
                shard_index=shard_index,
                next_commit_ordinal=next_commit_ordinal,
                resume_after_global_ordinal=resume_after_global_ordinal,
            )
        )
        if not ledger_rows:
            continue
        queue_rows = _load_queue_rows(queue_path)
        with _connect_ro(db_path) as db:
            for start in range(0, len(ledger_rows), batch_size):
                batch = ledger_rows[start : start + batch_size]
                queue_by_evidence: dict[str, dict[str, Any]] = {}
                ordered_queue_rows: list[dict[str, Any]] = []
                for ledger_row in batch:
                    row_index = int(ledger_row["shard_row_index"])
                    if row_index < 0 or row_index >= len(queue_rows):
                        raise ValueError(f"ledger shard row index out of range for shard {shard_index}: {row_index}")
                    queue_row = queue_rows[row_index]
                    if str(queue_row["candidate_id_hash"]) != str(ledger_row["candidate_id_hash"]):
                        raise ValueError("ledger/queue candidate_id_hash mismatch")
                    token_count = int(queue_row["core_proxy_token_count"])
                    if token_count != int(ledger_row["accepted_token_count"]):
                        raise ValueError("ledger accepted token count mismatch")
                    evidence_id = str(queue_row["private_evidence_row_id"])
                    queue_by_evidence[evidence_id] = queue_row
                    ordered_queue_rows.append(queue_row)
                evidence_rows = _fetch_evidence_batch(
                    db, list(queue_by_evidence), include_candidate=content_mode == "verified-core"
                )
                for ledger_row, queue_row in zip(batch, ordered_queue_rows, strict=True):
                    if accepted_tokens >= target_tokens:
                        break
                    evidence_id = str(queue_row["private_evidence_row_id"])
                    evidence = evidence_rows.get(evidence_id)
                    if evidence is None:
                        raise ValueError("missing private evidence row")
                    text = str(evidence["private_evidence_text"])
                    if str(evidence["candidate_id_hash"]) != str(queue_row["candidate_id_hash"]):
                        raise ValueError("private evidence candidate binding mismatch")
                    if str(evidence["evidence_text_sha256"]) != str(queue_row["evidence_text_sha256"]):
                        raise ValueError("private evidence hash field mismatch")
                    if _sha256_text(text) != str(queue_row["evidence_text_sha256"]):
                        raise ValueError("private evidence text hash mismatch")
                    if writer is None:
                        writer = PartWriter(output_dir, part_index)
                    payload = {
                        "global_ordinal": int(ledger_row["global_ordinal"]),
                        "shard_index": shard_index,
                        "shard_row_index": int(ledger_row["shard_row_index"]),
                        "candidate_id_hash": str(queue_row["candidate_id_hash"]),
                        "private_evidence_row_id": evidence_id,
                        "evidence_text_sha256": str(queue_row["evidence_text_sha256"]),
                        "core_text_sha256": str(queue_row.get("core_text_sha256") or queue_row["evidence_text_sha256"]),
                        "source_document_id_hash": str(queue_row["source_document_id_hash"]),
                        "core_proxy_token_count": int(queue_row["core_proxy_token_count"]),
                        "w1_route": "w1_pass",
                        "text": text,
                    }
                    if content_mode == "verified-core":
                        payload.update(_verified_core_fields(str(evidence["candidate_json"]), queue_row, evidence))
                    writer.write_row(payload)
                    accepted_rows += 1
                    accepted_tokens += int(payload["core_proxy_token_count"])
                    resume_after_global_ordinal = int(payload["global_ordinal"])
                    if writer.token_count >= part_token_target:
                        part = writer.close()
                        if part is not None:
                            parts.append(part)
                            preserve_part_progress()
                        writer = None
                        part_index += 1
                        write_progress("running", current_shard=shard_index)
                write_progress("running", current_shard=shard_index)

    if writer is not None:
        part = writer.close()
        if part is not None:
            parts.append(part)
            preserve_part_progress()
    status = "complete" if accepted_tokens >= target_tokens else "incomplete"
    manifest = current_manifest(status)
    _atomic_write_json(manifest_path, manifest)
    write_progress(status)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff")
    parser.add_argument("--supervisor-dir")
    parser.add_argument("--ledger")
    parser.add_argument("--output-dir")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--part-token-target", type=int, default=DEFAULT_PART_TOKENS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--content-mode", choices=("evidence", "verified-core"), default="evidence")
    args = parser.parse_args(argv)
    required = ("handoff", "supervisor_dir", "ledger", "output_dir")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")
    manifest = export(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "accepted_row_count": manifest["accepted_row_count"],
                "accepted_token_count_lower_bound": manifest["accepted_token_count_lower_bound"],
                "part_count": manifest["part_count"],
                "manifest_hash": manifest["manifest_hash"],
                "raw_text_printed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
