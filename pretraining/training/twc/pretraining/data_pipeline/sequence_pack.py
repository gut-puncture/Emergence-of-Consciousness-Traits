"""R7 isolated-sequence pack, finite-view, and semantic-step artifacts.

The module deliberately keeps semantic identity independent of physical roots.
It stores tokens in immutable little-endian uint16 shards and keeps every
next-token target exactly once without allowing attention across clean runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from twc.repro.hashing import stable_json_hash
from twc.repro.public_artifacts import validate_public_no_raw_payload

from .contracts import TokenizedCleanRun, hash_without, require_int, require_sha256

INDEX_RECORD = struct.Struct("<QQIIBBH")
_SPLIT_CODE = {"train": 0, "validation": 1}
_CODE_SPLIT = {value: key for key, value in _SPLIT_CODE.items()}
DEFAULT_TARGET_SHARD_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class IsolatedSequenceYield:
    clean_run_token_count: int
    segment_count: int
    loss_token_count: int
    context_only_token_count: int
    dropped_token_count: int
    storage_token_count: int


@dataclass(frozen=True)
class PackedSegment:
    split: str
    split_segment_index: int
    clean_run_ordinal: int
    component_ordinal: int
    tokens: tuple[int, ...]
    loss_token_count: int


@dataclass(frozen=True)
class IsolatedPackResult:
    output_root: Path
    manifest: dict[str, Any]
    manifest_path: Path


class _FusedAcceptedSchedule:
    def __init__(
        self,
        root: Path,
        manifest: Mapping[str, Any],
        encoder: Callable[[str], Iterable[int]],
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.encoder = encoder
        self.total_count = int(manifest["accepted_clean_run_count"])

    def __iter__(self) -> Iterator[TokenizedCleanRun]:
        return self.iter_from(0)

    def iter_from(self, cursor: int) -> Iterator[TokenizedCleanRun]:
        if cursor < 0 or cursor > self.total_count:
            raise ValueError("fused accepted input cursor is out of range")
        covered = 0
        for receipt in self.manifest["part_receipts"]:
            part_count = int(receipt["record_count"])
            if covered + part_count <= cursor:
                covered += part_count
                continue
            with (self.root / receipt["part_name"]).open(encoding="utf-8") as handle:
                for local_index, line in enumerate(handle):
                    if covered + local_index < cursor:
                        continue
                    row = json.loads(line)
                    token_ids = tuple(self.encoder(row["text"]))
                    if not token_ids:
                        raise ValueError("tokenizer produced an empty clean run")
                    yield TokenizedCleanRun(
                        row["split"],
                        row["clean_run_id_hash"],
                        row["source_document_id_hash"],
                        token_ids,
                    )
            covered += part_count


def _validate_accepted_part_chain(
    accepted_manifest: Mapping[str, Any], accepted_root: Path
) -> None:
    """Verify a relocatable accepted input using manifest-bound immutable parts."""

    from .accepted_data import validate_accepted_clean_run_manifest

    validate_accepted_clean_run_manifest(accepted_manifest)
    for receipt in accepted_manifest["part_receipts"]:
        part = _relative_file(accepted_root, receipt.get("part_name"), "accepted part")
        if part.stat().st_size != receipt.get("byte_count"):
            raise ValueError("accepted input part byte count mismatch")
        local_path = accepted_root / str(receipt["part_name"]).replace(
            ".jsonl", ".receipt.json"
        )
        try:
            local = json.loads(local_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("accepted input local receipt is missing or invalid") from exc
        canonical_local = {
            key: value
            for key, value in local.items()
            if key not in {"file_identity", "receipt_hash"}
        }
        if (
            canonical_local != receipt
            or local.get("receipt_hash") != hash_without(local, "receipt_hash")
        ):
            raise ValueError("accepted input local receipt binding mismatch")
        metadata = part.stat()
        unchanged_local_file = local.get("file_identity") == {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "byte_count": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
        }
        if not unchanged_local_file and _sha256_file(part) != receipt.get("sha256"):
            raise ValueError("accepted input part hash mismatch")


def isolated_sequence_yield(
    clean_run_token_count: int, context_length: int
) -> IsolatedSequenceYield:
    count = require_int(clean_run_token_count, "clean_run_token_count")
    context = require_int(context_length, "context_length", minimum=1)
    loss = max(0, count - 1)
    segments = (loss + context - 1) // context
    return IsolatedSequenceYield(
        clean_run_token_count=count,
        segment_count=segments,
        loss_token_count=loss,
        context_only_token_count=int(count > 0),
        dropped_token_count=0,
        storage_token_count=loss + segments,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(partial, 0o600)
    os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _open_private_exclusive(path: Path):
    """Create a binary artifact as 0600 before any content becomes visible."""

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "wb")


def _relative_file(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"{field} must be a safe relative path")
    path = root.joinpath(*pure.parts)
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{field} may not traverse a symlink")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes the artifact root") from exc
    if not path.is_file():
        raise ValueError(f"{field} is missing or is not a regular file")
    return path


def _validate_resume_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        raise ValueError("isolated-pack receipt resume snapshot is invalid")
    required = {
        "run_cursor",
        "target_start",
        "current_run_accounted",
        "clean_run_count",
        "component_count",
        "clean_run_token_count",
        "loss_token_count",
        "context_only_token_count",
        "storage_token_count",
        "split_run_counts",
        "split_segment_counts",
        "split_loss_token_counts",
        "input_prefix_root",
    }
    if set(snapshot) != required or type(snapshot.get("current_run_accounted")) is not bool:
        raise ValueError("isolated-pack receipt resume snapshot keys mismatch")
    for field in required - {
        "current_run_accounted",
        "split_run_counts",
        "split_segment_counts",
        "split_loss_token_counts",
        "input_prefix_root",
    }:
        require_int(snapshot[field], field)
    require_sha256(snapshot["input_prefix_root"], "input_prefix_root")
    for field in ("split_run_counts", "split_segment_counts", "split_loss_token_counts"):
        counts = snapshot[field]
        if not isinstance(counts, Mapping) or set(counts) != set(_SPLIT_CODE):
            raise ValueError("isolated-pack receipt split counters are invalid")
        for split in _SPLIT_CODE:
            require_int(counts[split], f"{field}.{split}")
    return dict(snapshot)


def build_isolated_sequence_pack(
    *,
    output_root: Path,
    tokenized_runs: Iterable[TokenizedCleanRun],
    tokenizer_manifest_hash: str,
    tokenized_manifest_hash: str | None = None,
    accepted_clean_run_manifest_hash: str | None = None,
    accepted_input_artifact_kind: str = "twc_accepted_clean_run_manifest_v1",
    configuration_proxy_manifest_hash: str | None = None,
    configuration_composite_manifest_hash: str | None = None,
    context_length: int,
    vocab_size: int,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    allow_unbound_test_input: bool = False,
    _validated_fused_input: bool = False,
    _validated_configuration_input: bool = False,
    _validated_configuration_composite_input: bool = False,
    interrupt_after_shards: int | None = None,
) -> IsolatedPackResult:
    """Build a content-bound pack from a validated canonical run stream.

    Direct iterables are intentionally marked as test-only until the fused
    accepted-text/tokenizer operator supplies the validated production stream.
    """

    root = Path(output_root)
    require_sha256(tokenizer_manifest_hash, "tokenizer_manifest_hash")
    bindings = (
        tokenized_manifest_hash,
        accepted_clean_run_manifest_hash,
        configuration_proxy_manifest_hash,
        configuration_composite_manifest_hash,
    )
    if sum(value is not None for value in bindings) != 1:
        raise ValueError(
            "exactly one tokenized, accepted, or configuration input hash is required"
        )
    input_artifact_kind = (
        accepted_input_artifact_kind
        if accepted_clean_run_manifest_hash is not None
        else "twc_configuration_composite_corpus_manifest_v1"
        if configuration_composite_manifest_hash is not None
        else "twc_configuration_proxy_corpus_manifest_v1"
        if configuration_proxy_manifest_hash is not None
        else "twc_unbound_test_token_stream_v1"
    )
    input_manifest_hash = require_sha256(
        accepted_clean_run_manifest_hash
        or configuration_composite_manifest_hash
        or configuration_proxy_manifest_hash
        or tokenized_manifest_hash,
        "input_manifest_hash",
    )
    context = require_int(context_length, "context_length", minimum=1)
    vocabulary = require_int(vocab_size, "vocab_size", minimum=1)
    shard_limit = require_int(target_shard_bytes, "target_shard_bytes", minimum=2)
    if vocabulary > 65_536:
        raise ValueError("uint16 pack requires vocab_size <= 65536")
    if tokenized_manifest_hash is not None and not allow_unbound_test_input:
        raise ValueError("unbound token streams are test-only")
    if accepted_clean_run_manifest_hash is not None and not _validated_fused_input:
        raise ValueError("production packing requires the validated fused input operator")
    if accepted_clean_run_manifest_hash is not None and accepted_input_artifact_kind not in {
        "twc_accepted_clean_run_manifest_v1", "twc_accepted_candidate_core_run_manifest_v1"
    }:
        raise ValueError("unsupported accepted input artifact kind")
    if configuration_proxy_manifest_hash is not None and not _validated_configuration_input:
        raise ValueError("configuration packing requires the validated proxy operator")
    if (
        configuration_composite_manifest_hash is not None
        and not _validated_configuration_composite_input
    ):
        raise ValueError(
            "configuration packing requires the validated composite operator"
        )
    if interrupt_after_shards is not None:
        require_int(interrupt_after_shards, "interrupt_after_shards", minimum=1)
    build_contract = {
        "artifact_kind": "twc_isolated_sequence_pack_build_contract_v1",
        "tokenizer_manifest_hash": tokenizer_manifest_hash,
        "input_artifact_kind": input_artifact_kind,
        "input_manifest_hash": input_manifest_hash,
        "context_length": context,
        "vocab_size": vocabulary,
        "dtype": "uint16-le",
        "index_record": "<QQIIBBH",
        "component_identity_kind": "source_document_id_hash",
        "target_shard_bytes": shard_limit,
        "segmentation_policy": "isolated_overlap_one_v1",
        "order_policy": (
            "parent_train_then_extension_train_then_frozen_parent_validation_v1"
            if configuration_composite_manifest_hash is not None
            else "upstream_canonical_content_hash_order"
        ),
    }
    build_contract_hash = stable_json_hash(build_contract)
    manifest_path = root / "isolated_sequence_pack_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_isolated_sequence_pack(manifest, root)
        if manifest.get("build_contract_hash") != build_contract_hash:
            raise ValueError("completed isolated pack build contract mismatch")
        return IsolatedPackResult(root, manifest, manifest_path)
    (root / "shards").mkdir(parents=True, exist_ok=True, mode=0o700)
    (root / "receipts").mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    os.chmod(root / "shards", 0o700)
    os.chmod(root / "receipts", 0o700)

    receipt_paths = sorted((root / "receipts").glob("*.json"))
    receipts: list[dict[str, Any]] = []
    previous_receipt_hash: str | None = None
    for ordinal, receipt_path in enumerate(receipt_paths):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("isolated-pack shard receipt JSON is invalid") from exc
        if (
            receipt.get("artifact_kind") != "twc_isolated_sequence_pack_shard_receipt_v1"
            or receipt.get("ordinal") != ordinal
            or receipt.get("build_contract_hash") != build_contract_hash
            or receipt.get("previous_receipt_hash") != previous_receipt_hash
            or receipt.get("receipt_hash") != hash_without(receipt, "receipt_hash")
        ):
            raise ValueError("isolated-pack shard receipt chain mismatch")
        shard = receipt.get("shard")
        if not isinstance(shard, Mapping):
            raise ValueError("isolated-pack shard receipt is missing shard metadata")
        if shard.get("ordinal") != ordinal:
            raise ValueError("isolated-pack shard receipt ordinal mismatch")
        _validate_resume_snapshot(receipt.get("resume"))
        token_path = _relative_file(root, shard.get("token_path"), "token_path")
        index_path = _relative_file(root, shard.get("index_path"), "index_path")
        if (
            token_path.stat().st_size != shard.get("token_byte_count")
            or index_path.stat().st_size != shard.get("index_byte_count")
            or _sha256_file(token_path) != shard.get("token_sha256")
            or _sha256_file(index_path) != shard.get("index_sha256")
        ):
            raise ValueError("isolated-pack completed shard content changed")
        receipts.append(receipt)
        previous_receipt_hash = receipt["receipt_hash"]

    progress_path = root / "isolated_sequence_pack_progress.json"
    progress: dict[str, Any] | None = None
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("isolated-pack progress JSON is invalid") from exc
        receipt_hashes = {receipt["receipt_hash"] for receipt in receipts}
        if (
            progress.get("artifact_kind") != "twc_isolated_sequence_pack_progress_v1"
            or progress.get("build_contract_hash") != build_contract_hash
            or progress.get("progress_hash") != hash_without(progress, "progress_hash")
            or (
                progress.get("latest_receipt_hash") is not None
                and progress.get("latest_receipt_hash") not in receipt_hashes
            )
        ):
            raise ValueError("isolated-pack progress identity mismatch")
        if progress.get("latest_receipt_hash") is None:
            if receipts or progress.get("completed_shard_count") != 0:
                raise ValueError("isolated-pack progress head mismatch")
        else:
            referenced_index = next(
                index
                for index, receipt in enumerate(receipts)
                if receipt["receipt_hash"] == progress["latest_receipt_hash"]
            )
            if (
                progress.get("completed_shard_count") != referenced_index + 1
                or progress.get("resume") != receipts[referenced_index]["resume"]
            ):
                raise ValueError("isolated-pack progress receipt binding mismatch")
    resume = (
        receipts[-1]["resume"]
        if receipts
        else {
            "run_cursor": 0,
            "target_start": 0,
            "current_run_accounted": False,
            "clean_run_count": 0,
            "component_count": 0,
            "clean_run_token_count": 0,
            "loss_token_count": 0,
            "context_only_token_count": 0,
            "storage_token_count": 0,
            "split_run_counts": {split: 0 for split in _SPLIT_CODE},
            "split_segment_counts": {split: 0 for split in _SPLIT_CODE},
            "split_loss_token_counts": {split: 0 for split in _SPLIT_CODE},
            "input_prefix_root": stable_json_hash([]),
        }
    )
    if receipts and (
        progress is None
        or progress.get("latest_receipt_hash") != receipts[-1]["receipt_hash"]
    ):
        progress = {
            "artifact_kind": "twc_isolated_sequence_pack_progress_v1",
            "schema_version": 1,
            "build_contract_hash": build_contract_hash,
            "input_manifest_hash": input_manifest_hash,
            "completed_shard_count": len(receipts),
            "latest_receipt_hash": receipts[-1]["receipt_hash"],
            "resume": resume,
        }
        progress["progress_hash"] = hash_without(progress, "progress_hash")
        _atomic_write(
            progress_path,
            json.dumps(progress, sort_keys=True, separators=(",", ":")).encode(),
        )

    shards = [dict(receipt["shard"]) for receipt in receipts]
    split_segment_counts = dict(resume["split_segment_counts"])
    split_loss_counts = dict(resume["split_loss_token_counts"])
    split_run_counts = dict(resume["split_run_counts"])
    clean_tokens = int(resume["clean_run_token_count"])
    loss_tokens = int(resume["loss_token_count"])
    context_only = int(resume["context_only_token_count"])
    storage_tokens = int(resume["storage_token_count"])
    resume_run_cursor = int(resume["run_cursor"])
    resume_target_start = int(resume["target_start"])
    resume_run_accounted = resume.get("current_run_accounted") is True
    committed_row_cursor = resume_run_cursor + int(resume_run_accounted)
    identity_path = root / ".isolated-pack-identities.sqlite"
    if not identity_path.exists():
        descriptor = os.open(identity_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
    identity = sqlite3.connect(identity_path)
    os.chmod(identity_path, 0o600)
    identity.executescript(
        "create table if not exists clean_runs("
        "ordinal integer primary key,id text not null unique,chain_root text not null);"
        "create table if not exists components("
        "id text primary key,ordinal integer not null unique,split text not null,"
        "first_run_ordinal integer not null);"
    )
    identity.execute("delete from clean_runs where ordinal>=?", (committed_row_cursor,))
    identity.execute(
        "delete from components where first_run_ordinal>=?", (committed_row_cursor,)
    )
    identity.commit()
    identity_count = int(identity.execute("select count(*) from clean_runs").fetchone()[0])
    identity_root_row = identity.execute(
        "select chain_root from clean_runs order by ordinal desc limit 1"
    ).fetchone()
    observed_identity_root = (
        str(identity_root_row[0]) if identity_root_row is not None else stable_json_hash([])
    )
    if identity_count != committed_row_cursor or observed_identity_root != resume["input_prefix_root"]:
        identity.close()
        raise ValueError("isolated-pack durable input-prefix index mismatch")
    open_shard: dict[str, Any] | None = None
    completed_this_call = 0

    expected_files = {
        relative
        for shard in shards
        for relative in (shard["token_path"], shard["index_path"])
    }
    next_ordinal = len(receipts)
    allowed_uncommitted = {
        f"{split}-{next_ordinal:06d}.{suffix}"
        for split in _SPLIT_CODE
        for suffix in ("tok", "idx")
    }
    for existing in (root / "shards").iterdir():
        relative = existing.relative_to(root).as_posix().removesuffix(".partial")
        if relative in expected_files:
            continue
        if existing.name.removesuffix(".partial") in allowed_uncommitted:
            existing.unlink()
        else:
            raise ValueError("unexpected uncommitted isolated-pack shard file")

    def close_shard(run_cursor: int, target_start: int, current_run_accounted: bool) -> None:
        nonlocal open_shard, previous_receipt_hash, completed_this_call
        if open_shard is None:
            return
        for handle in (open_shard["tokens"], open_shard["index"]):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        identity.commit()
        token_partial: Path = open_shard["token_partial"]
        index_partial: Path = open_shard["index_partial"]
        token_path: Path = open_shard["token_path"]
        index_path: Path = open_shard["index_path"]
        os.chmod(token_partial, 0o600)
        os.chmod(index_partial, 0o600)
        os.replace(token_partial, token_path)
        os.replace(index_partial, index_path)
        directory = os.open(token_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        shard = {
                "ordinal": len(shards),
                "split": open_shard["split"],
                "token_path": token_path.relative_to(root).as_posix(),
                "index_path": index_path.relative_to(root).as_posix(),
                "first_split_segment_index": open_shard["first_index"],
                "segment_count": open_shard["segments"],
                "storage_token_count": open_shard["storage_tokens"],
                "loss_token_count": open_shard["loss_tokens"],
                "token_byte_count": token_path.stat().st_size,
                "index_byte_count": index_path.stat().st_size,
                "token_sha256": _sha256_file(token_path),
                "index_sha256": _sha256_file(index_path),
            }
        shards.append(shard)
        snapshot = {
            "run_cursor": run_cursor,
            "target_start": target_start,
            "current_run_accounted": current_run_accounted,
            "clean_run_count": int(
                identity.execute("select count(*) from clean_runs").fetchone()[0]
            ),
            "component_count": int(
                identity.execute("select count(*) from components").fetchone()[0]
            ),
            "clean_run_token_count": clean_tokens,
            "loss_token_count": loss_tokens,
            "context_only_token_count": context_only,
            "storage_token_count": storage_tokens,
            "split_run_counts": dict(split_run_counts),
            "split_segment_counts": dict(split_segment_counts),
            "split_loss_token_counts": dict(split_loss_counts),
            "input_prefix_root": input_prefix_root,
        }
        receipt = {
            "artifact_kind": "twc_isolated_sequence_pack_shard_receipt_v1",
            "schema_version": 1,
            "ordinal": shard["ordinal"],
            "build_contract_hash": build_contract_hash,
            "previous_receipt_hash": previous_receipt_hash,
            "shard": shard,
            "resume": snapshot,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        receipt["receipt_hash"] = hash_without(receipt, "receipt_hash")
        validate_public_no_raw_payload(receipt)
        _atomic_write(
            root / "receipts" / f"shard-{shard['ordinal']:06d}.json",
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(),
        )
        receipts.append(receipt)
        previous_receipt_hash = receipt["receipt_hash"]
        progress_payload = {
            "artifact_kind": "twc_isolated_sequence_pack_progress_v1",
            "schema_version": 1,
            "build_contract_hash": build_contract_hash,
            "input_manifest_hash": input_manifest_hash,
            "completed_shard_count": len(shards),
            "latest_receipt_hash": previous_receipt_hash,
            "resume": snapshot,
        }
        progress_payload["progress_hash"] = hash_without(progress_payload, "progress_hash")
        validate_public_no_raw_payload(progress_payload)
        _atomic_write(
            progress_path,
            json.dumps(progress_payload, sort_keys=True, separators=(",", ":")).encode(),
        )
        open_shard = None
        completed_this_call += 1
        if interrupt_after_shards is not None and completed_this_call >= interrupt_after_shards:
            raise InterruptedError("synthetic interruption after durable isolated-pack shard")

    def ensure_shard(
        split: str,
        required_bytes: int,
        run_cursor: int,
        target_start: int,
    ) -> dict[str, Any]:
        nonlocal open_shard
        if open_shard is not None and (
            open_shard["split"] != split
            or (
                open_shard["token_bytes"] > 0
                and open_shard["token_bytes"] + required_bytes > shard_limit
            )
        ):
            close_shard(run_cursor, target_start, True)
        if open_shard is None:
            ordinal = len(shards)
            stem = f"{split}-{ordinal:06d}"
            token_path = root / "shards" / f"{stem}.tok"
            index_path = root / "shards" / f"{stem}.idx"
            token_partial = token_path.with_name(token_path.name + ".partial")
            index_partial = index_path.with_name(index_path.name + ".partial")
            open_shard = {
                "split": split,
                "token_path": token_path,
                "index_path": index_path,
                "token_partial": token_partial,
                "index_partial": index_partial,
                "tokens": _open_private_exclusive(token_partial),
                "index": _open_private_exclusive(index_partial),
                "first_index": split_segment_counts[split],
                "segments": 0,
                "storage_tokens": 0,
                "loss_tokens": 0,
                "token_bytes": 0,
            }
        return open_shard

    try:
        run_count = resume_run_cursor
        input_prefix_root = str(resume["input_prefix_root"])
        if receipts:
            if not hasattr(tokenized_runs, "iter_from"):
                raise ValueError("resuming isolated packing requires a seekable canonical input")
            run_stream = tokenized_runs.iter_from(resume_run_cursor)  # type: ignore[attr-defined]
            start_ordinal = resume_run_cursor
        else:
            run_stream = tokenized_runs
            start_ordinal = 0
        for run_ordinal, run in enumerate(run_stream, start=start_ordinal):
            run_count = run_ordinal + 1
            if not isinstance(run, TokenizedCleanRun) or run.split not in _SPLIT_CODE:
                raise ValueError("tokenized stream contains an invalid clean run")
            require_sha256(run.clean_run_id_hash, "clean_run_id_hash")
            require_sha256(run.source_document_id_hash, "source_document_id_hash")
            if not run.token_ids or any(
                type(token) is not int or token < 0 or token >= vocabulary
                for token in run.token_ids
            ):
                raise ValueError("tokenized clean run contains an invalid token")
            run_yield = isolated_sequence_yield(len(run.token_ids), context)
            already_accounted = run_ordinal == resume_run_cursor and resume_run_accounted
            if already_accounted:
                committed = identity.execute(
                    "select id,chain_root from clean_runs where ordinal=?", (run_ordinal,)
                ).fetchone()
                if committed != (run.clean_run_id_hash, input_prefix_root):
                    raise ValueError("isolated-pack committed input row identity changed")
            else:
                input_prefix_root = stable_json_hash(
                    {
                        "previous_root": input_prefix_root,
                        "split": run.split,
                        "clean_run_id_hash": run.clean_run_id_hash,
                        "source_document_id_hash": run.source_document_id_hash,
                        "token_sequence_hash": stable_json_hash(list(run.token_ids)),
                    }
                )
                try:
                    identity.execute(
                        "insert into clean_runs values (?,?,?)",
                        (run_ordinal, run.clean_run_id_hash, input_prefix_root),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError("duplicate clean-run identity") from exc
            component_row = identity.execute(
                "select ordinal,split from components where id=?",
                (run.source_document_id_hash,),
            ).fetchone()
            if component_row is None:
                component_ordinal = int(
                    identity.execute("select count(*) from components").fetchone()[0]
                )
                identity.execute(
                    "insert into components values (?,?,?,?)",
                    (
                        run.source_document_id_hash,
                        component_ordinal,
                        run.split,
                        run_ordinal,
                    ),
                )
            else:
                component_ordinal = int(component_row[0])
                if component_row[1] != run.split:
                    raise ValueError("tokenized component split leakage")
            target_start = resume_target_start if run_ordinal == resume_run_cursor else 0
            if not already_accounted:
                clean_tokens += len(run.token_ids)
                context_only += run_yield.context_only_token_count
                split_run_counts[run.split] += 1
            while target_start < run_yield.loss_token_count:
                segment_loss = min(context, run_yield.loss_token_count - target_start)
                tokens = run.token_ids[target_start : target_start + segment_loss + 1]
                payload = struct.pack(f"<{len(tokens)}H", *tokens)
                shard = ensure_shard(run.split, len(payload), run_ordinal, target_start)
                token_offset = shard["token_bytes"]
                shard["tokens"].write(payload)
                shard["index"].write(
                    INDEX_RECORD.pack(
                        token_offset,
                        run_ordinal,
                        len(tokens),
                        component_ordinal,
                        _SPLIT_CODE[run.split],
                        0,
                        0,
                    )
                )
                shard["segments"] += 1
                shard["storage_tokens"] += len(tokens)
                shard["loss_tokens"] += segment_loss
                shard["token_bytes"] += len(payload)
                split_segment_counts[run.split] += 1
                split_loss_counts[run.split] += segment_loss
                loss_tokens += segment_loss
                storage_tokens += len(tokens)
                target_start += segment_loss
        expected_total = getattr(tokenized_runs, "total_count", None)
        if expected_total is not None and run_count != expected_total:
            raise ValueError("isolated-pack canonical input ended before its declared count")
        close_shard(run_count, 0, False)
    except BaseException:
        if open_shard is not None:
            for key in ("tokens", "index"):
                try:
                    open_shard[key].close()
                except OSError:
                    pass
        identity.rollback()
        identity.close()
        raise
    if progress_path.exists():
        progress_path.unlink()
    if loss_tokens == 0:
        identity.close()
        raise ValueError("isolated sequence pack contains no loss tokens")
    identity.commit()
    final_clean_run_count = int(
        identity.execute("select count(*) from clean_runs").fetchone()[0]
    )
    final_component_count = int(
        identity.execute("select count(*) from components").fetchone()[0]
    )
    identity.close()
    manifest: dict[str, Any] = {
        "artifact_kind": "twc_isolated_sequence_pack_manifest_v1",
        "schema_version": 1,
        "build_contract": build_contract,
        "build_contract_hash": build_contract_hash,
        "tokenizer_manifest_hash": tokenizer_manifest_hash,
        "input_artifact_kind": input_artifact_kind,
        "input_manifest_hash": input_manifest_hash,
        "context_length": context,
        "vocab_size": vocabulary,
        "dtype": "uint16-le",
        "index_record": "<QQIIBBH",
        "component_identity_kind": "source_document_id_hash",
        "clean_run_count": final_clean_run_count,
        "component_count": final_component_count,
        "clean_run_token_count": clean_tokens,
        "segment_count": sum(split_segment_counts.values()),
        "loss_token_count": loss_tokens,
        "context_only_token_count": context_only,
        "dropped_token_count": 0,
        "storage_token_count": storage_tokens,
        "split_run_counts": split_run_counts,
        "split_segment_counts": split_segment_counts,
        "split_loss_token_counts": split_loss_counts,
        "shard_receipt_count": len(receipts),
        "shard_receipt_root": stable_json_hash(
            [receipt["receipt_hash"] for receipt in receipts]
        ),
        "shards": shards,
        "inline_tokens_present": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    manifest["manifest_hash"] = hash_without(manifest, "manifest_hash")
    validate_public_no_raw_payload(manifest)
    _atomic_write(
        manifest_path,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
    )
    validate_isolated_sequence_pack(manifest, root)
    identity_path.unlink()
    return IsolatedPackResult(root, manifest, manifest_path)


def build_fused_isolated_sequence_pack(
    *,
    output_root: Path,
    accepted_root: Path,
    accepted_manifest: Mapping[str, Any],
    tokenizer_root: Path,
    tokenizer_manifest: Mapping[str, Any],
    context_length: int,
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    encoder: Callable[[str], Iterable[int]] | None = None,
    interrupt_after_shards: int | None = None,
) -> IsolatedPackResult:
    """Validate, tokenize, and pack accepted runs without a token JSON spool."""

    from .accepted_data import validate_accepted_clean_run_manifest

    accepted_root = Path(accepted_root)
    tokenizer_root = Path(tokenizer_root)
    if list((Path(output_root) / "receipts").glob("*.json")):
        _validate_accepted_part_chain(accepted_manifest, accepted_root)
    else:
        validate_accepted_clean_run_manifest(accepted_manifest, root=accepted_root)
    accepted_hash = require_sha256(
        accepted_manifest.get("manifest_hash"), "accepted_clean_run_manifest_hash"
    )
    if (
        tokenizer_manifest.get("artifact_kind") != "twc_tokenizer_manifest_v1"
        or tokenizer_manifest.get("schema_version") != 1
        or tokenizer_manifest.get("manifest_hash")
        != hash_without(tokenizer_manifest, "manifest_hash")
    ):
        raise ValueError("tokenizer manifest is invalid")
    tokenizer_hash = require_sha256(
        tokenizer_manifest.get("manifest_hash"), "tokenizer_manifest_hash"
    )
    if tokenizer_manifest.get("corpus_manifest_hash") != accepted_hash:
        raise ValueError("tokenizer is not bound to the accepted clean-run manifest")
    diagnostics = tokenizer_manifest.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("ready") is not True:
        raise ValueError("tokenizer diagnostics are not ready")
    vocabulary = require_int(
        tokenizer_manifest.get("actual_vocab_size"), "actual_vocab_size", minimum=1
    )
    if vocabulary > 65_536:
        raise ValueError("tokenizer vocabulary does not fit uint16")
    artifact = tokenizer_root / "tokenizer.bin"
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("tokenizer artifact must be a regular non-symlink file")
    if _sha256_file(artifact) != tokenizer_manifest.get("artifact_sha256"):
        raise ValueError("tokenizer artifact hash mismatch")
    if encoder is None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - environment-dependent boundary
            raise RuntimeError("install the project data extra to build the fused pack") from exc
        tokenizer = Tokenizer.from_str(artifact.read_text(encoding="utf-8"))
        encode: Callable[[str], Iterable[int]] = lambda text: tokenizer.encode(text).ids
    else:
        encode = encoder

    schedule = _FusedAcceptedSchedule(accepted_root, accepted_manifest, encode)

    return build_isolated_sequence_pack(
        output_root=output_root,
        tokenized_runs=schedule,
        tokenizer_manifest_hash=tokenizer_hash,
        accepted_clean_run_manifest_hash=accepted_hash,
        accepted_input_artifact_kind=str(accepted_manifest["artifact_kind"]),
        context_length=context_length,
        vocab_size=vocabulary,
        target_shard_bytes=target_shard_bytes,
        _validated_fused_input=True,
        interrupt_after_shards=interrupt_after_shards,
    )


def validate_isolated_sequence_pack(manifest: Mapping[str, Any], root: Path) -> None:
    if manifest.get("artifact_kind") != "twc_isolated_sequence_pack_manifest_v1":
        raise ValueError("unsupported isolated sequence pack manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported isolated sequence pack schema")
    if manifest.get("manifest_hash") != hash_without(manifest, "manifest_hash"):
        raise ValueError("isolated sequence pack manifest hash mismatch")
    require_sha256(manifest.get("build_contract_hash"), "build_contract_hash")
    require_sha256(manifest.get("tokenizer_manifest_hash"), "tokenizer_manifest_hash")
    if manifest.get("input_artifact_kind") not in {
        "twc_accepted_clean_run_manifest_v1",
        "twc_accepted_candidate_core_run_manifest_v1",
        "twc_configuration_proxy_corpus_manifest_v1",
        "twc_configuration_composite_corpus_manifest_v1",
        "twc_unbound_test_token_stream_v1",
    }:
        raise ValueError("unsupported isolated sequence input artifact")
    require_sha256(manifest.get("input_manifest_hash"), "input_manifest_hash")
    build_contract = manifest.get("build_contract")
    if (
        not isinstance(build_contract, Mapping)
        or stable_json_hash(build_contract) != manifest.get("build_contract_hash")
    ):
        raise ValueError("isolated sequence build contract hash mismatch")
    if (
        build_contract.get("artifact_kind")
        != "twc_isolated_sequence_pack_build_contract_v1"
        or build_contract.get("segmentation_policy") != "isolated_overlap_one_v1"
    ):
        raise ValueError("isolated sequence build contract policy mismatch")
    require_int(
        build_contract.get("target_shard_bytes"),
        "target_shard_bytes",
        minimum=2,
    )
    for field in (
        "tokenizer_manifest_hash",
        "input_artifact_kind",
        "input_manifest_hash",
        "context_length",
        "vocab_size",
        "dtype",
        "index_record",
        "component_identity_kind",
    ):
        if manifest.get(field) != build_contract.get(field):
            raise ValueError("isolated sequence build contract field mismatch")
    expected_order_policy = (
        "parent_train_then_extension_train_then_frozen_parent_validation_v1"
        if manifest.get("input_artifact_kind")
        == "twc_configuration_composite_corpus_manifest_v1"
        else "upstream_canonical_content_hash_order"
    )
    if build_contract.get("order_policy") != expected_order_policy:
        raise ValueError("isolated sequence build contract order policy mismatch")
    if manifest.get("dtype") != "uint16-le" or manifest.get("index_record") != "<QQIIBBH":
        raise ValueError("unsupported isolated sequence binary encoding")
    if manifest.get("shard_receipt_count") != len(manifest.get("shards", ())):
        raise ValueError("isolated sequence shard receipt count mismatch")
    require_sha256(manifest.get("shard_receipt_root"), "shard_receipt_root")
    root = Path(root)
    observed_segments = observed_loss = observed_storage = 0
    observed_split_loss = {split: 0 for split in _SPLIT_CODE}
    split_next = {split: 0 for split in _SPLIT_CODE}
    run_metadata: dict[int, tuple[str, int]] = {}
    component_splits: dict[int, str] = {}
    previous_run_ordinal = -1
    previous_run_tail: int | None = None
    for expected_ordinal, shard in enumerate(manifest.get("shards", ())):
        if not isinstance(shard, Mapping) or shard.get("ordinal") != expected_ordinal:
            raise ValueError("pack shard ordinal mismatch")
        split = shard.get("split")
        if split not in _SPLIT_CODE or shard.get("first_split_segment_index") != split_next[split]:
            raise ValueError("pack shard split coverage mismatch")
        token_path = _relative_file(root, shard.get("token_path"), "token_path")
        index_path = _relative_file(root, shard.get("index_path"), "index_path")
        token_bytes = require_int(shard.get("token_byte_count"), "token_byte_count")
        index_bytes = require_int(shard.get("index_byte_count"), "index_byte_count")
        if token_path.stat().st_size != token_bytes or index_path.stat().st_size != index_bytes:
            raise ValueError("pack shard byte count mismatch")
        if _sha256_file(token_path) != shard.get("token_sha256"):
            raise ValueError("pack token shard hash mismatch")
        if _sha256_file(index_path) != shard.get("index_sha256"):
            raise ValueError("pack index shard hash mismatch")
        segments = require_int(shard.get("segment_count"), "segment_count", minimum=1)
        if index_bytes != segments * INDEX_RECORD.size:
            raise ValueError("pack index byte count mismatch")
        shard_loss = shard_storage = 0
        expected_offset = 0
        with index_path.open("rb") as index_handle, token_path.open("rb") as token_handle:
            for record in struct.iter_unpack(INDEX_RECORD.format, index_handle.read()):
                offset, run_ordinal, token_count, component_ordinal, split_code, flags, reserved = record
                if offset != expected_offset or token_count < 2:
                    raise ValueError("pack index token coverage mismatch")
                if split_code != _SPLIT_CODE[split] or flags != 0 or reserved != 0:
                    raise ValueError("pack index metadata mismatch")
                if run_ordinal >= manifest.get("clean_run_count") or component_ordinal >= manifest.get(
                    "component_count"
                ):
                    raise ValueError("pack index ordinal is out of range")
                if run_ordinal < previous_run_ordinal:
                    raise ValueError("pack index clean-run order is not canonical")
                metadata = (split, component_ordinal)
                if run_metadata.setdefault(run_ordinal, metadata) != metadata:
                    raise ValueError("pack clean-run metadata changed between segments")
                if component_splits.setdefault(component_ordinal, split) != split:
                    raise ValueError("pack component split leakage")
                token_handle.seek(offset)
                raw = token_handle.read(token_count * 2)
                if len(raw) != token_count * 2:
                    raise ValueError("pack token shard is truncated")
                values = struct.unpack(f"<{token_count}H", raw)
                if any(token >= manifest.get("vocab_size") for token in values):
                    raise ValueError("pack token id exceeds the frozen vocabulary")
                if run_ordinal == previous_run_ordinal and previous_run_tail != values[0]:
                    raise ValueError("pack clean-run segment overlap mismatch")
                previous_run_ordinal = run_ordinal
                previous_run_tail = values[-1]
                expected_offset += token_count * 2
                shard_storage += token_count
                shard_loss += token_count - 1
        if expected_offset != token_bytes:
            raise ValueError("pack token shard coverage mismatch")
        if shard_storage != shard.get("storage_token_count") or shard_loss != shard.get("loss_token_count"):
            raise ValueError("pack shard accounting mismatch")
        split_next[split] += segments
        observed_segments += segments
        observed_storage += shard_storage
        observed_loss += shard_loss
        observed_split_loss[split] += shard_loss
    if observed_segments != manifest.get("segment_count"):
        raise ValueError("pack segment count mismatch")
    if observed_storage != manifest.get("storage_token_count"):
        raise ValueError("pack storage token count mismatch")
    if observed_loss != manifest.get("loss_token_count"):
        raise ValueError("pack loss token count mismatch")
    if split_next != manifest.get("split_segment_counts"):
        raise ValueError("pack split segment count mismatch")
    if observed_split_loss != manifest.get("split_loss_token_counts"):
        raise ValueError("pack split loss token count mismatch")
    if manifest.get("storage_token_count") != manifest.get("loss_token_count") + manifest.get(
        "segment_count"
    ):
        raise ValueError("pack overlap storage accounting mismatch")
    if manifest.get("clean_run_token_count") != manifest.get("loss_token_count") + manifest.get(
        "context_only_token_count"
    ):
        raise ValueError("pack clean-run accounting mismatch")
    if manifest.get("context_only_token_count") != manifest.get("clean_run_count"):
        raise ValueError("pack context-only accounting mismatch")
    if manifest.get("dropped_token_count") != 0:
        raise ValueError("isolated sequence pack may not drop tokens")
    validate_public_no_raw_payload(manifest)


def iter_view_segments(
    pack_manifest: Mapping[str, Any],
    pack_root: Path,
    *,
    split: str | None = None,
    view: Mapping[str, Any] | None = None,
) -> Iterator[PackedSegment]:
    validate_isolated_sequence_pack(pack_manifest, pack_root)
    if view is not None:
        if view.get("artifact_kind") != "twc_finite_sequence_view_v1":
            raise ValueError("unsupported finite sequence view")
        if view.get("view_hash") != hash_without(view, "view_hash"):
            raise ValueError("finite sequence view hash mismatch")
        if view.get("pack_manifest_hash") != pack_manifest.get("manifest_hash"):
            raise ValueError("finite sequence view pack binding mismatch")
        split = view.get("split")
        remaining = require_int(view.get("loss_token_count"), "loss_token_count", minimum=1)
    else:
        if split not in _SPLIT_CODE:
            raise ValueError("split must be train or validation")
        remaining = None
    split_index = 0
    for shard in pack_manifest["shards"]:
        if shard["split"] != split:
            continue
        token_path = Path(pack_root) / shard["token_path"]
        index_path = Path(pack_root) / shard["index_path"]
        with token_path.open("rb") as tokens, index_path.open("rb") as index:
            for record in struct.iter_unpack(INDEX_RECORD.format, index.read()):
                offset, run_ordinal, token_count, component_ordinal, split_code, _, _ = record
                tokens.seek(offset)
                values = struct.unpack(f"<{token_count}H", tokens.read(token_count * 2))
                available_loss = token_count - 1
                use_loss = available_loss if remaining is None else min(available_loss, remaining)
                if use_loss <= 0:
                    return
                yield PackedSegment(
                    _CODE_SPLIT[split_code],
                    split_index,
                    run_ordinal,
                    component_ordinal,
                    values[: use_loss + 1],
                    use_loss,
                )
                split_index += 1
                if remaining is not None:
                    remaining -= use_loss
                    if remaining == 0:
                        return
    if remaining is not None:
        raise ValueError("finite sequence view exceeds available pack tokens")


def build_finite_sequence_view(
    *,
    pack_manifest: Mapping[str, Any],
    pack_root: Path,
    split: str,
    requested_loss_tokens: int,
    output_path: Path,
) -> dict[str, Any]:
    if split not in _SPLIT_CODE:
        raise ValueError("split must be train or validation")
    requested = require_int(requested_loss_tokens, "requested_loss_tokens", minimum=1)
    available = require_int(
        pack_manifest.get("split_loss_token_counts", {}).get(split),
        "available_loss_tokens",
    )
    if requested > available:
        raise ValueError("requested finite view exceeds available unique loss tokens")
    full = 0
    final = 0
    selected = 0
    for segment in iter_view_segments(pack_manifest, pack_root, split=split):
        if selected + segment.loss_token_count < requested:
            full += 1
            selected += segment.loss_token_count
        elif selected + segment.loss_token_count == requested:
            final = segment.loss_token_count
            selected = requested
            break
        else:
            final = requested - selected
            selected = requested
            break
    view: dict[str, Any] = {
        "artifact_kind": "twc_finite_sequence_view_v1",
        "schema_version": 1,
        "pack_manifest_hash": pack_manifest["manifest_hash"],
        "split": split,
        "loss_token_count": requested,
        "full_segment_count": full,
        "final_segment_loss_tokens": final,
        "finite": True,
        "repeat_allowed": False,
        "order_policy": "pack_canonical_order",
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    view["view_hash"] = hash_without(view, "view_hash")
    _atomic_write(Path(output_path), json.dumps(view, sort_keys=True, separators=(",", ":")).encode())
    return view


def build_global_step_plan(
    *,
    view: Mapping[str, Any],
    pack_manifest: Mapping[str, Any],
    pack_root: Path,
    output_path: Path,
    max_loss_tokens: int,
    max_attention_cells: int,
    max_sequences: int,
) -> dict[str, Any]:
    limits = (
        require_int(max_loss_tokens, "max_loss_tokens", minimum=1),
        require_int(max_attention_cells, "max_attention_cells", minimum=1),
        require_int(max_sequences, "max_sequences", minimum=1),
    )
    steps: list[dict[str, int]] = []
    current: list[PackedSegment] = []

    def publish() -> None:
        if not current:
            return
        first = current[0].split_segment_index
        loss = sum(item.loss_token_count for item in current)
        attention = sum(item.loss_token_count * item.loss_token_count for item in current)
        steps.append(
            {
                "ordinal": len(steps),
                "first_segment_index": first,
                "sequence_count": len(current),
                "loss_token_count": loss,
                "attention_cells": attention,
            }
        )
        current.clear()

    for segment in iter_view_segments(pack_manifest, pack_root, view=view):
        cells = segment.loss_token_count * segment.loss_token_count
        if segment.loss_token_count > limits[0] or cells > limits[1]:
            raise ValueError("one sequence exceeds the semantic global-step limits")
        proposed_loss = sum(item.loss_token_count for item in current) + segment.loss_token_count
        proposed_cells = sum(item.loss_token_count**2 for item in current) + cells
        if current and (
            proposed_loss > limits[0]
            or proposed_cells > limits[1]
            or len(current) + 1 > limits[2]
        ):
            publish()
        current.append(segment)
    publish()
    plan: dict[str, Any] = {
        "artifact_kind": "twc_global_step_plan_v1",
        "schema_version": 1,
        "view_hash": view["view_hash"],
        "pack_manifest_hash": pack_manifest["manifest_hash"],
        "max_loss_tokens": limits[0],
        "max_attention_cells": limits[1],
        "max_sequences": limits[2],
        "step_count": len(steps),
        "loss_token_count": sum(step["loss_token_count"] for step in steps),
        "steps": steps,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    plan["plan_hash"] = hash_without(plan, "plan_hash")
    _atomic_write(Path(output_path), json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())
    return plan


def validate_global_step_plan(
    plan: Mapping[str, Any], *, view: Mapping[str, Any], pack_manifest: Mapping[str, Any], pack_root: Path
) -> None:
    if plan.get("artifact_kind") != "twc_global_step_plan_v1" or plan.get("schema_version") != 1:
        raise ValueError("unsupported global step plan")
    if plan.get("plan_hash") != hash_without(plan, "plan_hash"):
        raise ValueError("global step plan hash mismatch")
    if plan.get("view_hash") != view.get("view_hash") or plan.get("pack_manifest_hash") != pack_manifest.get("manifest_hash"):
        raise ValueError("global step plan binding mismatch")
    max_loss = require_int(plan.get("max_loss_tokens"), "max_loss_tokens", minimum=1)
    max_cells = require_int(
        plan.get("max_attention_cells"), "max_attention_cells", minimum=1
    )
    max_sequences = require_int(plan.get("max_sequences"), "max_sequences", minimum=1)
    segments = iter(iter_view_segments(pack_manifest, pack_root, view=view))
    pending: PackedSegment | None = None
    cursor = 0
    total_loss = 0
    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) != plan.get("step_count"):
        raise ValueError("global step plan step count mismatch")
    for ordinal, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("ordinal") != ordinal:
            raise ValueError("global step plan ordinal mismatch")
        count = require_int(step.get("sequence_count"), "sequence_count", minimum=1)
        if step.get("first_segment_index") != cursor:
            raise ValueError("global step plan segment coverage mismatch")
        loss = cells = 0
        for _ in range(count):
            try:
                segment = pending if pending is not None else next(segments)
            except StopIteration as error:
                raise ValueError("global step plan segment coverage mismatch") from error
            pending = None
            loss += segment.loss_token_count
            cells += segment.loss_token_count**2
        if loss != step.get("loss_token_count") or cells != step.get("attention_cells"):
            raise ValueError("global step plan accounting mismatch")
        if loss > max_loss or cells > max_cells or count > max_sequences:
            raise ValueError("global step plan limit exceeded")
        try:
            pending = next(segments)
        except StopIteration:
            pending = None
        if pending is not None:
            if (
                loss + pending.loss_token_count <= max_loss
                and cells + pending.loss_token_count**2 <= max_cells
                and count + 1 <= max_sequences
            ):
                raise ValueError("global step plan is not deterministic sequential next-fit")
        cursor += count
        total_loss += loss
    if (
        pending is not None
        or next(segments, None) is not None
        or total_loss != view.get("loss_token_count")
        or total_loss != plan.get("loss_token_count")
    ):
        raise ValueError("global step plan semantic coverage mismatch")


def relocate_and_verify_pack(
    *, pack_manifest: Mapping[str, Any], source_root: Path, destination_root: Path
) -> dict[str, Any]:
    validate_isolated_sequence_pack(pack_manifest, source_root)
    destination = Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    expected_names = {
        PurePosixPath(shard[field]).as_posix()
        for shard in pack_manifest["shards"]
        for field in ("token_path", "index_path")
    } | {"isolated_sequence_pack_manifest.json", "placement_receipt.json"}
    for existing in destination.rglob("*"):
        if existing.is_dir():
            continue
        relative = existing.relative_to(destination).as_posix()
        normalized = relative.removesuffix(".partial")
        if normalized not in expected_names:
            raise ValueError("destination contains an unexpected file")
    for shard in pack_manifest["shards"]:
        for field, hash_field, size_field in (
            ("token_path", "token_sha256", "token_byte_count"),
            ("index_path", "index_sha256", "index_byte_count"),
        ):
            relative = PurePosixPath(shard[field])
            source = Path(source_root).joinpath(*relative.parts)
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.exists():
                if (
                    target.is_symlink()
                    or target.stat().st_size != shard[size_field]
                    or _sha256_file(target) != shard[hash_field]
                ):
                    raise ValueError("existing destination file does not match the pack")
                continue
            partial = target.with_name(target.name + ".partial")
            partial.unlink(missing_ok=True)
            with source.open("rb") as reader, partial.open("wb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            os.chmod(partial, 0o600)
            os.replace(partial, target)
    _atomic_write(
        destination / "isolated_sequence_pack_manifest.json",
        json.dumps(pack_manifest, sort_keys=True, separators=(",", ":")).encode(),
    )
    validate_isolated_sequence_pack(pack_manifest, destination)
    destination_resolved = destination.resolve(strict=True)
    file_identities = []
    for shard in pack_manifest["shards"]:
        for field, hash_field in (("token_path", "token_sha256"), ("index_path", "index_sha256")):
            relative = PurePosixPath(shard[field]).as_posix()
            stat = destination_resolved.joinpath(*PurePosixPath(relative).parts).stat()
            file_identities.append({
                "relative_path": relative,
                "device_id": stat.st_dev,
                "inode": stat.st_ino,
                "byte_count": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": shard[hash_field],
            })
    receipt: dict[str, Any] = {
        "artifact_kind": "twc_data_placement_receipt_v1",
        "schema_version": 1,
        "pack_manifest_hash": pack_manifest["manifest_hash"],
        "verified_file_count": len(pack_manifest["shards"]) * 2,
        "full_destination_readback": True,
        "destination_root": str(destination_resolved),
        "destination_device_id": destination_resolved.stat().st_dev,
        "file_identities": file_identities,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    receipt["receipt_hash"] = hash_without(receipt, "receipt_hash")
    _atomic_write(
        destination / "placement_receipt.json",
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(),
    )
    return receipt
