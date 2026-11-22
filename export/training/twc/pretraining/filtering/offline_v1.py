"""CPU-only W1 archive and report assembly operations.

This module deliberately has no scorer or GPU dependency.  Archive publication
is transactional: a verified private partial is promoted first and the no-raw
manifest is written last.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
from typing import Any, BinaryIO
import zipfile

from twc.repro.hashing import stable_json_hash, self_hash_mismatches
from twc.pretraining.filtering import artifacts as filtering_artifacts
from twc.pretraining.filtering import policy as filtering_policy
from twc.pretraining.filtering import prefilter
from twc.pretraining.filtering import stage_state_v1


_ARCHIVE_NAME = "w1_private_score_state.zip"
_MANIFEST_NAME = "w1_private_score_state_manifest.json"
_ALLOWED_RUN_FILENAMES = frozenset(
    {
        "filtering_stage_state.sqlite",
        "filtering_stage_state.sqlite-wal",
        "filtering_stage_state.sqlite-shm",
        "prefilter_cpu_receipt.json",
        "w1_survivor_queue.jsonl.ready",
        "w1_survivor_queue.ids.tsv.ready",
        "w1_scores.jsonl.ready",
        "w1_scores.jsonl.part",
        "w1_score_receipt.json",
        "assembled_filtering_report.json",
        "sharded_w1_handoff.json",
        "prefilter_supervisor_manifest.json",
        "prefilter_supervisor_summary.json",
        "w1_sharded_summary.json",
        "w1_assembled_report.json",
        "launch_plan.json",
        "launcher_receipt.json",
        "commands.jsonl",
    }
)


@dataclass(frozen=True)
class StagedCloseoutArchiveMembers:
    """Attempt-local private paths plus a compact, publishable no-raw receipt."""

    files: dict[str, Path]
    receipt: dict[str, Any]


def _safe_archive_member_name(value: Any) -> str:
    name = str(value or "")
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in name
    ):
        raise ValueError("W1 closeout archive contains an unsafe member path")
    return path.as_posix()


def _closeout_manifest_files(
    archive_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    filtering_artifacts.validate_public_no_raw_payload(archive_manifest)
    if archive_manifest.get("artifact_kind") != filtering_artifacts.W1_PRIVATE_SCORE_STATE_ARCHIVE_KIND:
        raise ValueError("W1 closeout archive manifest kind mismatch")
    if int(archive_manifest.get("schema_version", 0)) != 1:
        raise ValueError("W1 closeout archive manifest schema mismatch")
    files = archive_manifest.get("files")
    if not isinstance(files, list) or int(archive_manifest.get("file_count", -1)) != len(files):
        raise ValueError("W1 closeout archive manifest file count mismatch")
    normalized: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("W1 closeout archive manifest file must be an object")
        name = _safe_archive_member_name(item.get("relative_path"))
        if name in normalized:
            raise ValueError("W1 closeout archive manifest contains duplicate members")
        size = int(item.get("size_bytes", -1))
        if size < 0:
            raise ValueError("W1 closeout archive member size must be nonnegative")
        normalized[name] = {
            **dict(item),
            "relative_path": name,
            "sha256": filtering_artifacts._require_sha256(
                item.get("sha256"), "W1 closeout archive member sha256"
            ),
            "size_bytes": size,
        }
    return normalized


def _closeout_zip_inventory(archive_path: Path) -> dict[str, zipfile.ZipInfo]:
    _validate_archive_source(archive_path)
    observed: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(archive_path, "r") as archive:
        for info in archive.infolist():
            name = _safe_archive_member_name(info.filename)
            unix_mode = (int(info.external_attr) >> 16) & 0o170000
            if info.is_dir() or unix_mode == stat.S_IFLNK:
                raise ValueError("W1 closeout archive members must be regular files")
            if name in observed:
                raise ValueError("W1 closeout archive contains duplicate members")
            observed[name] = info
    return observed


def preflight_closeout_archive(
    *,
    archive_path: str | Path,
    archive_manifest: Mapping[str, Any],
    required_members: set[str] | Sequence[str],
) -> dict[str, Any]:
    """Validate the manifest and ZIP directory without reading member payloads."""

    source = Path(archive_path)
    declared = _closeout_manifest_files(archive_manifest)
    observed = _closeout_zip_inventory(source)
    if set(observed) != set(declared):
        raise ValueError("W1 closeout archive central directory differs from its manifest")
    for name, info in observed.items():
        if int(info.file_size) != int(declared[name]["size_bytes"]):
            raise ValueError("W1 closeout archive member size differs from its manifest")
    required = {_safe_archive_member_name(name) for name in required_members}
    if not required.issubset(declared):
        raise ValueError("W1 closeout archive is missing a required member")
    result = {
        "artifact_kind": "twc_w1_closeout_archive_preflight_v1",
        "schema_version": 1,
        "status": "ready",
        "member_count": len(declared),
        "required_member_count": len(required),
        "declared_total_uncompressed_bytes": sum(
            int(item["size_bytes"]) for item in declared.values()
        ),
        "required_max_uncompressed_bytes": max(
            (int(declared[name]["size_bytes"]) for name in required), default=0
        ),
        "outer_archive_hash_verified": False,
        "member_payloads_read": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    filtering_artifacts.validate_public_no_raw_payload(result)
    return result


def stage_closeout_archive_members(
    *,
    archive_path: str | Path,
    archive_manifest: Mapping[str, Any],
    relative_paths: Sequence[str],
    scratch_root: str | Path,
) -> StagedCloseoutArchiveMembers:
    """Extract and verify one bounded group; never expand the whole archive."""

    source = Path(archive_path)
    declared = _closeout_manifest_files(archive_manifest)
    observed = _closeout_zip_inventory(source)
    requested = [_safe_archive_member_name(name) for name in relative_paths]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("W1 closeout archive member group must be nonempty and unique")
    if any(name not in declared or name not in observed for name in requested):
        raise ValueError("W1 closeout archive member group is missing")
    target_root = Path(scratch_root)
    filtering_artifacts.validate_private_filtering_run_dir(target_root)
    if target_root.exists() or target_root.is_symlink():
        raise FileExistsError("W1 closeout scratch group already exists")
    target_root.mkdir(parents=True, mode=0o700)
    if os.name == "posix":
        os.chmod(target_root, 0o700)
    staged: dict[str, Path] = {}
    facts: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(source, "r") as archive:
            for name in requested:
                target = target_root / PurePosixPath(name)
                target.resolve(strict=False).relative_to(target_root.resolve(strict=True))
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if os.name == "posix":
                    os.chmod(target.parent, 0o700)
                digest = hashlib.sha256()
                size = 0
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with archive.open(observed[name], "r") as incoming, os.fdopen(
                        descriptor, "wb", closefd=False
                    ) as outgoing:
                        while True:
                            block = incoming.read(1024 * 1024)
                            if not block:
                                break
                            digest.update(block)
                            size += len(block)
                            outgoing.write(block)
                        outgoing.flush()
                        os.fsync(outgoing.fileno())
                finally:
                    os.close(descriptor)
                if size != int(declared[name]["size_bytes"]):
                    raise ValueError("W1 closeout archive staged member size mismatch")
                member_hash = digest.hexdigest()
                if member_hash != declared[name]["sha256"]:
                    raise ValueError("W1 closeout archive staged member hash mismatch")
                staged[name] = target
                facts.append({"relative_path_hash": stable_json_hash(name), "sha256": member_hash, "size_bytes": size})
        receipt = {
            "artifact_kind": "twc_w1_closeout_archive_member_group_receipt_v1",
            "schema_version": 1,
            "member_count": len(facts),
            "total_bytes": sum(int(item["size_bytes"]) for item in facts),
            "fact_root": stable_json_hash(facts),
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        filtering_artifacts.validate_public_no_raw_payload(receipt)
        return StagedCloseoutArchiveMembers(files=staged, receipt=receipt)
    except BaseException:
        shutil.rmtree(target_root, ignore_errors=True)
        raise


def derive_closeout_sqlite_audit_view(
    *, source_db: str | Path, destination_db: str | Path
) -> dict[str, Any]:
    """Merge a staged SQLite+WAL view into one immutable, integrity-checked DB."""

    source, destination = Path(source_db), Path(destination_db)
    _validate_archive_source(source)
    filtering_artifacts.validate_private_filtering_run_dir(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("W1 closeout derived SQLite view already exists")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_uri = f"file:{source}?mode=ro"
    source_connection = sqlite3.connect(source_uri, uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection, pages=1024)
        destination_connection.commit()
        check = destination_connection.execute("pragma integrity_check").fetchone()
        if check is None or str(check[0]) != "ok":
            raise ValueError("W1 closeout derived SQLite integrity check failed")
    except BaseException:
        destination_connection.close()
        source_connection.close()
        destination.unlink(missing_ok=True)
        raise
    else:
        destination_connection.close()
        source_connection.close()
    filtering_artifacts._chmod_private_sqlite_files(destination)
    receipt = {
        "artifact_kind": "twc_w1_closeout_sqlite_audit_view_receipt_v1",
        "schema_version": 1,
        "integrity_check": "ok",
        "size_bytes": int(destination.stat().st_size),
        "sha256": filtering_artifacts._sha256_file(destination),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    filtering_artifacts.validate_public_no_raw_payload(receipt)
    return receipt


def validate_closeout_handoff_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate compact immutable handoff facts without requiring live source paths."""

    filtering_artifacts.validate_public_no_raw_payload(payload)
    if payload.get("artifact_kind") != filtering_artifacts.SHARDED_W1_HANDOFF_KIND:
        raise ValueError("closeout W1 handoff kind mismatch")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("closeout W1 handoff schema mismatch")
    if payload.get("status") != "sharded_prefilter_ready_for_w1":
        raise ValueError("closeout W1 handoff is not ready")
    if self_hash_mismatches(payload, "handoff_hash"):
        raise ValueError("closeout W1 handoff hash mismatch")
    shards = payload.get("shards")
    if not isinstance(shards, list) or len(shards) != int(payload.get("shard_count", -1)):
        raise ValueError("closeout W1 handoff shard count mismatch")
    normalized: list[dict[str, Any]] = []
    survivor_total = 0
    seen_names: set[str] = set()
    for expected_index, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or int(shard.get("shard_index", -1)) != expected_index:
            raise ValueError("closeout W1 handoff shard ordering mismatch")
        name = str(shard.get("shard_run_dir_name") or "")
        if not name or "/" in name or "\\" in name or name in seen_names:
            raise ValueError("closeout W1 handoff shard directory name mismatch")
        seen_names.add(name)
        survivor_count = filtering_artifacts._nonnegative_int(
            shard.get("survivor_count"), "closeout handoff survivor_count"
        )
        survivor_total += survivor_count
        normalized.append(
            {
                **dict(shard),
                "shard_index": expected_index,
                "shard_run_dir_name": name,
                "survivor_count": survivor_count,
                "prefilter_stage_receipt_hash": filtering_artifacts._require_sha256(
                    shard.get("prefilter_stage_receipt_hash"),
                    "closeout handoff prefilter_stage_receipt_hash",
                ),
                "survivor_queue_hash": filtering_artifacts._require_sha256(
                    shard.get("survivor_queue_hash"), "closeout handoff survivor_queue_hash"
                ),
                "survivor_queue_id_index_hash": filtering_artifacts._require_sha256(
                    shard.get("survivor_queue_id_index_hash"),
                    "closeout handoff survivor_queue_id_index_hash",
                ),
            }
        )
    if survivor_total != filtering_artifacts._nonnegative_int(
        payload.get("survivor_count"), "closeout handoff total survivor_count"
    ):
        raise ValueError("closeout W1 handoff survivor total mismatch")
    return {**dict(payload), "shards": normalized}


def validate_closeout_shard_support_files(
    *,
    files: Mapping[str, Path],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "prefilter_cpu_receipt.json",
        "w1_survivor_queue.jsonl.ready",
        "w1_survivor_queue.ids.tsv.ready",
    }
    by_name = {Path(relative).name: path for relative, path in files.items()}
    if not required.issubset(by_name):
        raise ValueError("closeout compact shard support files are incomplete")
    receipt = filtering_artifacts._load_json(by_name["prefilter_cpu_receipt.json"])
    if not isinstance(receipt, Mapping):
        raise ValueError("closeout compact shard receipt must be an object")
    filtering_artifacts.validate_public_no_raw_payload(receipt)
    if self_hash_mismatches(receipt, "stage_receipt_hash"):
        raise ValueError("closeout compact shard receipt self-hash mismatch")
    if receipt.get("stage_receipt_hash") != shard.get("prefilter_stage_receipt_hash"):
        raise ValueError("closeout compact shard receipt hash mismatch")
    if int(receipt.get("survivor_count", -1)) != int(shard.get("survivor_count", -2)):
        raise ValueError("closeout compact shard survivor count mismatch")
    if (
        filtering_artifacts._sha256_file(by_name["w1_survivor_queue.jsonl.ready"])
        != shard.get("survivor_queue_hash")
    ):
        raise ValueError("closeout compact shard survivor queue hash mismatch")
    if (
        filtering_artifacts._sha256_file(by_name["w1_survivor_queue.ids.tsv.ready"])
        != shard.get("survivor_queue_id_index_hash")
    ):
        raise ValueError("closeout compact shard survivor id-index hash mismatch")
    queue_ids: list[str] = []
    with by_name["w1_survivor_queue.jsonl.ready"].open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError("closeout compact shard queue row must be an object")
            queue_ids.append(
                filtering_artifacts._require_sha256(
                    row.get("candidate_id_hash"),
                    "closeout compact shard queue candidate_id_hash",
                )
            )
    index_ids = [
        filtering_artifacts._require_sha256(
            line.split("\t", 1)[0],
            "closeout compact shard id-index candidate_id_hash",
        )
        for line in by_name["w1_survivor_queue.ids.tsv.ready"]
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    expected_count = int(shard.get("survivor_count", -1))
    if (
        len(queue_ids) != expected_count
        or len(index_ids) != expected_count
        or len(set(queue_ids)) != expected_count
        or queue_ids != index_ids
    ):
        raise ValueError("closeout compact shard queue and id-index membership mismatch")
    return {
        "shard_index": int(shard["shard_index"]),
        "survivor_count": int(shard["survivor_count"]),
        "support_fact_root": stable_json_hash(
            {
                "prefilter_stage_receipt_hash": shard["prefilter_stage_receipt_hash"],
                "survivor_queue_hash": shard["survivor_queue_hash"],
                "survivor_queue_id_index_hash": shard["survivor_queue_id_index_hash"],
            }
        ),
        "candidate_ids": frozenset(queue_ids),
    }


def _validate_archive_source(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise ValueError("W1 private score archive source file missing") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("W1 private score archive source must be a regular non-symlink file")
    if int(metadata.st_nlink) != 1:
        raise ValueError("W1 private score archive source must have a single link")
    if os.name == "posix" and int(metadata.st_uid) != int(os.geteuid()):
        raise ValueError("W1 private score archive source must belong to the current user")
    return metadata


def _iter_run_files(run_path: Path) -> Iterator[tuple[Path, str, str]]:
    for path in sorted(run_path.rglob("*")):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            raise ValueError("W1 private score archive source changed during inventory") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("W1 private score archive run tree must not contain symlinks")
        if not stat.S_ISREG(metadata.st_mode):
            continue
        if path.name in _ALLOWED_RUN_FILENAMES or re.fullmatch(r"rank_\d+_w1_summary\.json", path.name):
            relative = path.relative_to(run_path).as_posix()
            yield path, relative, "run_dir"


def _iter_extra_files(paths: Sequence[str | Path] | None) -> Iterator[tuple[Path, str, str]]:
    names: set[str] = set()
    for item in list(paths or []):
        base = Path(item)
        for path in (base, Path(str(base) + "-wal"), Path(str(base) + "-shm")):
            if path != base and not path.exists() and not path.is_symlink():
                continue
            _validate_archive_source(path)
            name = f"extra_state/{path.name}"
            if name in names:
                raise ValueError("W1 private score archive extra state member collision")
            names.add(name)
            yield path, name, "extra_state"


def _hash_source(path: Path) -> tuple[str, int]:
    expected = _validate_archive_source(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("W1 private score archive source changed while opening")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        closed_shape = os.fstat(descriptor)
        if (
            (closed_shape.st_dev, closed_shape.st_ino, closed_shape.st_size, closed_shape.st_mtime_ns)
            != (expected.st_dev, expected.st_ino, expected.st_size, expected.st_mtime_ns)
        ):
            raise ValueError("W1 private score archive source changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _inventory(
    run_path: Path,
    extra_state_files: Sequence[str | Path] | None,
) -> tuple[list[dict[str, Any]], list[tuple[Path, str, str]]]:
    sources = [*_iter_run_files(run_path), *_iter_extra_files(extra_state_files)]
    if not sources:
        raise ValueError("W1 private score archive found no allowlisted files")
    names = [name for _path, name, _kind in sources]
    if len(names) != len(set(names)):
        raise ValueError("W1 private score archive member collision")
    contains_ledger = any(
        name == "extra_state/w1_accepted_token_ledger.sqlite" for name in names
    )
    contains_handoff = any(name.endswith("sharded_w1_handoff.json") for name in names)
    if contains_ledger and not contains_handoff:
        raise ValueError(
            "W1 private score archive ledger requires the sharded W1 handoff"
        )
    if not any(name.endswith("filtering_stage_state.sqlite") for name in names):
        raise ValueError("W1 private score archive requires SQLite state")
    if not any(name.endswith("w1_score_receipt.json") for name in names):
        raise ValueError("W1 private score archive requires W1 score receipt")
    payloads: list[dict[str, Any]] = []
    for path, name, kind in sources:
        digest, size = _hash_source(path)
        payloads.append(
            {"relative_path": name, "source_kind": kind, "sha256": digest, "size_bytes": size}
        )
    return payloads, sources


def _copy_source_into_zip(archive: zipfile.ZipFile, path: Path, name: str, expected: Mapping[str, Any]) -> None:
    metadata = _validate_archive_source(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError("W1 private score archive source changed while opening")
        info = zipfile.ZipInfo(name)
        info.compress_type = zipfile.ZIP_DEFLATED
        info._compresslevel = 1
        info.external_attr = 0o600 << 16
        with archive.open(info, "w", force_zip64=True) as output:
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                size += len(block)
        final = os.fstat(descriptor)
        if (
            (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
            != (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
            or size != int(expected["size_bytes"])
            or digest.hexdigest() != expected["sha256"]
        ):
            raise ValueError("W1 private score archive source changed during archive")
    finally:
        os.close(descriptor)


def _validate_zip(path: Path, files: Sequence[Mapping[str, Any]]) -> None:
    expected = {str(item["relative_path"]): item for item in files}
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("W1 private score archive member set mismatch")
        for info in infos:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts or info.is_dir():
                raise ValueError("W1 private score archive member path is unsafe")
            proof = expected[info.filename]
            if int(info.file_size) != int(proof["size_bytes"]):
                raise ValueError("W1 private score archive member size mismatch")
            digest = hashlib.sha256()
            size = 0
            with archive.open(info, "r") as source:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    size += len(block)
            if size != int(proof["size_bytes"]) or digest.hexdigest() != proof["sha256"]:
                raise ValueError("W1 private score archive member hash mismatch")
        if archive.testzip() is not None:
            raise ValueError("W1 private score archive CRC validation failed")


def _score_receipt_facts(
    zip_path: Path,
    files: Sequence[Mapping[str, Any]],
    config_hash: str,
) -> tuple[int, list[str], int]:
    scored = 0
    statuses: set[str] = set()
    count = 0
    receipt_names = [
        str(item["relative_path"])
        for item in files
        if str(item["relative_path"]).endswith("w1_score_receipt.json")
    ]
    with zipfile.ZipFile(zip_path, "r") as archive:
        receipts = [(name, archive.read(name)) for name in receipt_names]
    for name, encoded in receipts:
        if not name.endswith("w1_score_receipt.json"):
            continue
        try:
            receipt = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("W1 private score archive receipt must be valid JSON") from None
        if not isinstance(receipt, Mapping):
            raise ValueError("W1 private score archive receipt must be an object")
        if receipt.get("artifact_kind") != filtering_artifacts.W1_SCORE_STAGE_KIND:
            raise ValueError("W1 private score archive receipt kind mismatch")
        if receipt.get("config_hash") != config_hash:
            raise ValueError("W1 private score archive config hash mismatch")
        scored += int(receipt.get("scored_count", 0))
        statuses.add(str(receipt.get("status")))
        count += 1
    return scored, sorted(statuses), count


def _archive_manifest(
    *, config_hash: str, run_path: Path, zip_path: Path, files: list[dict[str, Any]],
) -> dict[str, Any]:
    scored, statuses, receipt_count = _score_receipt_facts(zip_path, files, config_hash)
    payload = {
        "artifact_kind": filtering_artifacts.W1_PRIVATE_SCORE_STATE_ARCHIVE_KIND,
        "schema_version": 1,
        "status": "w1_private_score_state_archived",
        "config_hash": config_hash,
        "run_dir_sha256": stable_json_hash(str(run_path.resolve(strict=False))),
        "archive_file": _ARCHIVE_NAME,
        "archive_sha256": filtering_artifacts._sha256_file(zip_path),
        "archive_size_bytes": int(zip_path.stat().st_size),
        "file_count": len(files),
        "files": files,
        "extra_state_file_count": sum(1 for item in files if item["source_kind"] == "extra_state"),
        "private_archive_contains_accepted_token_ledger_state": any(
            item["relative_path"] == "extra_state/w1_accepted_token_ledger.sqlite" for item in files
        ),
        "w1_score_receipt_count": receipt_count,
        "w1_statuses": statuses,
        "w1_scored_count": scored,
        "private_archive_contains_sqlite_state": True,
        "private_archive_may_contain_private_evidence_text": True,
        "accepted_manifest_created": False, "accepted_manifest": False,
        "tokenizer_shards_created": False, "packed_shards_created": False,
        "training_shards_created": False, "base_pretraining_started": False,
        "pretraining_started": False, "raw_text_serialized": False,
        "raw_url_serialized": False, "token_serialized": False,
    }
    payload["archive_manifest_hash"] = stable_json_hash(payload)
    filtering_artifacts.validate_public_no_raw_payload(payload)
    return payload


def _validate_archive_output_file(path: Path) -> None:
    metadata = _validate_archive_source(path)
    if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError("W1 private score archive output must use mode 0600")


def _validate_existing_manifest(
    manifest: Mapping[str, Any],
    zip_path: Path,
    files: Sequence[Mapping[str, Any]],
    *,
    config_hash: str,
    run_path: Path,
) -> bool:
    filtering_artifacts.validate_public_no_raw_payload(manifest)
    if manifest.get("artifact_kind") != filtering_artifacts.W1_PRIVATE_SCORE_STATE_ARCHIVE_KIND:
        raise ValueError("W1 private score archive manifest kind mismatch")
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("W1 private score archive manifest schema mismatch")
    if manifest.get("status") != "w1_private_score_state_archived":
        raise ValueError("W1 private score archive manifest status mismatch")
    if manifest.get("archive_file") != _ARCHIVE_NAME:
        raise ValueError("W1 private score archive manifest filename mismatch")
    if manifest.get("config_hash") != config_hash:
        raise ValueError("W1 private score archive manifest config mismatch")
    if manifest.get("run_dir_sha256") != stable_json_hash(str(run_path.resolve(strict=False))):
        raise ValueError("W1 private score archive manifest run directory mismatch")
    if self_hash_mismatches(manifest, "archive_manifest_hash"):
        raise ValueError("W1 private score archive manifest hash mismatch")
    manifest_files = list(manifest.get("files") or [])
    if int(manifest.get("file_count", -1)) != len(manifest_files):
        raise ValueError("W1 private score archive manifest file count mismatch")
    if int(manifest.get("extra_state_file_count", -1)) != sum(
        1 for item in manifest_files if item.get("source_kind") == "extra_state"
    ):
        raise ValueError("W1 private score archive manifest extra state count mismatch")
    _validate_archive_output_file(zip_path)
    if manifest.get("archive_sha256") != filtering_artifacts._sha256_file(zip_path):
        raise ValueError("W1 private score archive file hash mismatch")
    if int(manifest.get("archive_size_bytes", -1)) != int(zip_path.stat().st_size):
        raise ValueError("W1 private score archive file size mismatch")
    _validate_zip(zip_path, manifest_files)
    return manifest_files == list(files)


def archive_w1_private_score_state(
    *, config_payload: Mapping[str, Any], run_dir: str | Path, archive_dir: str | Path,
    extra_state_files: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    filtering_policy.validate_chunk_filtering_config_payload(config_payload)
    run_path, archive_path = Path(run_dir), Path(archive_dir)
    filtering_artifacts.validate_private_filtering_run_dir(run_path)
    filtering_artifacts.validate_private_filtering_run_dir(archive_path)
    filtering_artifacts._ensure_private_dir(archive_path)
    files, sources = _inventory(run_path, extra_state_files)
    zip_path = archive_path / _ARCHIVE_NAME
    partial_path = archive_path / f"{_ARCHIVE_NAME}.partial"
    manifest_path = archive_path / _MANIFEST_NAME
    next_manifest_path = archive_path / f"{_MANIFEST_NAME}.next"

    if next_manifest_path.exists() or next_manifest_path.is_symlink():
        _validate_archive_output_file(next_manifest_path)
        candidate_manifest = filtering_artifacts._load_json(next_manifest_path)
        if not isinstance(candidate_manifest, Mapping):
            raise ValueError("W1 private score archive next manifest must be an object")
        candidate_files = list(candidate_manifest.get("files") or [])
        try:
            candidate_on_final = _validate_existing_manifest(
                candidate_manifest,
                zip_path,
                candidate_files,
                config_hash=str(config_payload["config_hash"]),
                run_path=run_path,
            )
        except (OSError, ValueError, zipfile.BadZipFile):
            candidate_on_final = False
        if candidate_on_final:
            filtering_artifacts._promote_private_partial(next_manifest_path, manifest_path)
            return dict(candidate_manifest)
        if partial_path.exists() or partial_path.is_symlink():
            try:
                candidate_on_partial = _validate_existing_manifest(
                    candidate_manifest,
                    partial_path,
                    candidate_files,
                    config_hash=str(config_payload["config_hash"]),
                    run_path=run_path,
                )
            except (OSError, ValueError, zipfile.BadZipFile):
                candidate_on_partial = False
            if candidate_on_partial:
                filtering_artifacts._promote_private_partial(partial_path, zip_path)
                filtering_artifacts._promote_private_partial(next_manifest_path, manifest_path)
                return dict(candidate_manifest)
        raise ValueError("W1 private score archive interrupted candidate is not recoverable")

    if manifest_path.exists() or manifest_path.is_symlink():
        _validate_archive_output_file(manifest_path)
        manifest = filtering_artifacts._load_json(manifest_path)
        if not isinstance(manifest, Mapping):
            raise ValueError("W1 private score archive manifest must be an object")
        if _validate_existing_manifest(
            manifest,
            zip_path,
            files,
            config_hash=str(config_payload["config_hash"]),
            run_path=run_path,
        ):
            expected_manifest = _archive_manifest(
                config_hash=str(config_payload["config_hash"]),
                run_path=run_path,
                zip_path=zip_path,
                files=files,
            )
            if dict(manifest) != expected_manifest:
                raise ValueError("W1 private score archive manifest semantic mismatch")
            return dict(manifest)
    elif zip_path.exists() or zip_path.is_symlink():
        _validate_archive_output_file(zip_path)
        _validate_zip(zip_path, files)
        payload = _archive_manifest(
            config_hash=str(config_payload["config_hash"]), run_path=run_path,
            zip_path=zip_path, files=files,
        )
        filtering_artifacts._write_json_atomic(manifest_path, payload)
        return payload
    descriptor = filtering_artifacts._create_private_partial(partial_path)
    try:
        with os.fdopen(descriptor, "w+b") as handle:
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                by_name = {item["relative_path"]: item for item in files}
                for path, name, _kind in sources:
                    _copy_source_into_zip(archive, path, name, by_name[name])
            filtering_artifacts._flush_and_fsync(handle)
        _validate_zip(partial_path, files)
        payload = _archive_manifest(
            config_hash=str(config_payload["config_hash"]), run_path=run_path,
            zip_path=partial_path, files=files,
        )
        filtering_artifacts._write_json_atomic(next_manifest_path, payload)
        filtering_artifacts._promote_private_partial(partial_path, zip_path)
        filtering_artifacts._promote_private_partial(next_manifest_path, manifest_path)
        return payload
    except BaseException:
        if not next_manifest_path.exists() and not next_manifest_path.is_symlink():
            try:
                partial_path.unlink()
            except FileNotFoundError:
                pass
        raise


def _expected_prefilter_status(run_dir: str | Path) -> str:
    receipt = filtering_artifacts._load_json(Path(run_dir) / "prefilter_cpu_receipt.json")
    if not isinstance(receipt, Mapping):
        raise ValueError("W1 start prefilter receipt must be an object")
    return prefilter._w1_start_prefilter_expected_status(receipt)


def assemble_filtering_stage_report(
    *, config_payload: Mapping[str, Any], run_dir: str | Path,
    expected_prefilter_status: str = "prefilter_ready_for_w1",
) -> dict[str, Any]:
    filtering_policy.validate_chunk_filtering_config_payload(config_payload)
    run_path = Path(run_dir)
    filtering_artifacts.validate_private_filtering_run_dir(run_path)
    receipt_path = run_path / "w1_score_receipt.json"
    if receipt_path.exists():
        early = filtering_artifacts._load_json(receipt_path)
        if isinstance(early, Mapping) and early.get("status") in {
            "accepted_token_target_reached_partial", "queue_exhausted_before_token_target",
        }:
            raise ValueError("assembled report cannot consume partial token-target W1 state before accepted token target review")
    queue_rows = filtering_artifacts._load_ready_jsonl(run_path / "w1_survivor_queue.jsonl", "survivor queue")
    score_rows = filtering_artifacts._load_ready_jsonl(run_path / "w1_scores.jsonl", "W1 scores")
    observed = queue_rows[0].get("config_hash") if queue_rows else config_payload["config_hash"]
    producer = filtering_policy._prefilter_input_config_payload_for_hash(config_payload, observed, label="survivor queue")
    prefilter._validate_survivor_queue_rows(queue_rows, producer, allow_empty=True)
    survivor_by_id = {str(row["candidate_id_hash"]): row for row in queue_rows}
    filtering_artifacts._validate_w1_score_rows(score_rows, config_payload, survivor_by_id=survivor_by_id, allow_empty=True)
    survivor_ids, scored_ids = set(survivor_by_id), {str(row["candidate_id_hash"]) for row in score_rows}
    if survivor_ids != scored_ids:
        raise ValueError("W1 scored set must equal survivor set")
    score_by_id = {str(row["candidate_id_hash"]): row for row in score_rows}
    conn = stage_state_v1.open_stage_state_readonly(
        run_path, producer, legacy_retained_config_payload=config_payload,
    )
    try:
        prefilter._validate_prefilter_ready_receipt(
            conn, run_path, producer, queue_rows=queue_rows,
            expected_status=expected_prefilter_status, require_compact_w1_scores_empty=False,
        )
        if not prefilter._survivor_queue_rows_match_live_db(
            queue_rows, prefilter._prefilter_survivor_queue_rows(conn, producer)
        ):
            raise ValueError("survivor queue export does not match SQLite state")
        if score_rows != conn.w1_scores():
            raise ValueError("W1 score export does not match SQLite state")
        if not receipt_path.exists():
            raise ValueError("assembled report requires W1 score receipt")
        receipt = filtering_artifacts._load_json(receipt_path)
        if not isinstance(receipt, Mapping):
            raise ValueError("W1 score receipt must be an object")
        if self_hash_mismatches(receipt, "stage_receipt_hash"):
            raise ValueError("W1 score receipt hash mismatch")
        envelope, envelope_hash = filtering_artifacts._validated_w1_runtime_envelope_from_receipt(receipt)
        if score_rows and filtering_artifacts._w1_score_row_runtime_hashes(score_rows) != {envelope_hash}:
            raise ValueError("W1 score row runtime envelope mismatch")
        if conn.candidate_ids(route="prefilter_survivor") != survivor_ids:
            raise ValueError("DB survivor set must equal survivor queue set")
        if conn.w1_score_candidate_ids() != scored_ids:
            raise ValueError("DB W1 score set must equal W1 score export set")
        routes: Counter[str] = Counter()
        candidate_count = 0
        for candidate_id, route in conn.iter_candidate_routes():
            candidate_count += 1
            if route != "prefilter_survivor":
                routes["final_error" if route.endswith("error") else "final_quarantine" if route.endswith("quarantine") else "final_reject"] += 1
            else:
                w1_route = score_by_id[candidate_id]["w1_route"]
                routes["final_safe" if w1_route == "w1_pass" else "final_error" if w1_route == "w1_error" else "final_reject"] += 1
        if candidate_count < len(survivor_ids):
            raise ValueError("candidate state is missing survivor rows")
        payload = {
            "artifact_kind": filtering_artifacts.ASSEMBLED_STAGE_REPORT_KIND, "schema_version": 1,
            "status": "assembled_ready_for_review", "stop_reason": "assembled_complete",
            "next_required_phase": "coordinator_review_before_any_accepted_manifest_or_rental_scale",
            "config_hash": config_payload["config_hash"], "prefilter_producer_config_hash": producer["config_hash"],
            "w1_consumer_config_hash": config_payload["config_hash"],
            "selected_policy_hash": config_payload["selected_policy_hash"],
            "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
            "source_metadata_hash": config_payload["source_metadata_hash"],
            "deterministic_filter_policy_hash": config_payload.get("deterministic_filter_policy_hash"),
            "candidate_count": candidate_count, "survivor_count": len(survivor_ids),
            "w1_scored_count": len(scored_ids), "survivor_w1_set_equality_validated": True,
            "w1_runtime_envelope": dict(envelope), "w1_runtime_envelope_hash": envelope_hash,
            "route_counts": filtering_artifacts._sorted_counter(routes),
            "accepted_manifest_created": False, "accepted_manifest": False,
            "tokenizer_shards_created": False, "packed_shards_created": False,
            "training_shards_created": False, "base_pretraining_started": False,
            "pretraining_started": False, "raw_text_serialized": False,
            "raw_url_serialized": False, "token_serialized": False,
        }
        payload["report_hash"] = stable_json_hash(payload)
        filtering_artifacts.validate_public_no_raw_payload(payload)
        filtering_artifacts._write_json_atomic(run_path / "assembled_filtering_report.json", payload)
        return payload
    finally:
        conn.close()


def assemble_sharded_filtering_stage_report(
    *, config_payload: Mapping[str, Any], handoff: Mapping[str, Any],
    supervisor_run_dir: str | Path, report_path: str | Path | None = None,
) -> dict[str, Any]:
    filtering_policy.validate_chunk_filtering_config_payload(config_payload)
    prefilter._validate_sharded_w1_handoff_payload(
        handoff, config_payload=config_payload, supervisor_run_dir=supervisor_run_dir,
    )
    routes: Counter[str] = Counter()
    hashes: list[str] = []
    envelope_hashes: set[str] = set()
    envelope: dict[str, Any] | None = None
    candidate_count = survivor_count = scored_count = 0
    for shard in sorted(handoff["shards"], key=lambda item: int(item["shard_index"])):
        shard_dir = prefilter._prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
        report = assemble_filtering_stage_report(
            config_payload=config_payload, run_dir=shard_dir,
            expected_prefilter_status=_expected_prefilter_status(shard_dir),
        )
        routes.update({str(key): int(value) for key, value in dict(report["route_counts"]).items()})
        hashes.append(str(report["report_hash"]))
        envelope_hashes.add(filtering_artifacts._require_sha256(report.get("w1_runtime_envelope_hash"), "w1_runtime_envelope_hash"))
        if envelope is None:
            value = report.get("w1_runtime_envelope")
            if not isinstance(value, Mapping):
                raise ValueError("sharded assembled report missing W1 runtime envelope")
            envelope = dict(value)
        candidate_count += int(report["candidate_count"])
        survivor_count += int(report["survivor_count"])
        scored_count += int(report["w1_scored_count"])
    if len(envelope_hashes) != 1:
        raise ValueError("sharded assembled report W1 runtime envelope mismatch")
    if candidate_count != int(handoff["candidate_count"]):
        raise ValueError("sharded assembled report candidate count mismatch")
    if survivor_count != int(handoff["survivor_count"]) or scored_count != int(handoff["survivor_count"]):
        raise ValueError("sharded assembled report survivor/W1 score count mismatch")
    payload = {
        "artifact_kind": filtering_artifacts.ASSEMBLED_STAGE_REPORT_KIND, "schema_version": 1,
        "status": "assembled_ready_for_review", "stop_reason": "assembled_sharded_complete",
        "next_required_phase": "coordinator_review_before_any_accepted_manifest_or_rental_scale",
        "assembly_input_mode": "sharded_w1_handoff", "sharded_w1_handoff_hash": handoff["handoff_hash"],
        "config_hash": config_payload["config_hash"], "prefilter_producer_config_hash": handoff["config_hash"],
        "w1_consumer_config_hash": config_payload["config_hash"],
        "selected_policy_hash": config_payload["selected_policy_hash"],
        "selected_policy_report_hash": config_payload["selected_policy_report_hash"],
        "source_metadata_hash": config_payload["source_metadata_hash"],
        "deterministic_filter_policy_hash": config_payload.get("deterministic_filter_policy_hash"),
        "candidate_count": candidate_count, "survivor_count": survivor_count,
        "w1_scored_count": scored_count, "survivor_w1_set_equality_validated": True,
        "w1_runtime_envelope": envelope, "w1_runtime_envelope_hash": next(iter(envelope_hashes)),
        "route_counts": filtering_artifacts._sorted_counter(routes), "shard_count": int(handoff["shard_count"]),
        "shard_report_hashes": hashes, "accepted_manifest_created": False, "accepted_manifest": False,
        "tokenizer_shards_created": False, "packed_shards_created": False,
        "training_shards_created": False, "base_pretraining_started": False,
        "pretraining_started": False, "raw_text_serialized": False,
        "raw_url_serialized": False, "token_serialized": False,
    }
    payload["report_hash"] = stable_json_hash(payload)
    filtering_artifacts.validate_public_no_raw_payload(payload)
    if report_path is not None:
        filtering_artifacts._write_json_atomic(report_path, payload)
    return payload


__all__ = [
    "archive_w1_private_score_state",
    "assemble_filtering_stage_report",
    "assemble_sharded_filtering_stage_report",
]
