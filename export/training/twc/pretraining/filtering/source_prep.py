"""Private source mirroring and deterministic local-Parquet planning.

This module owns preparation before CPU prefilter execution. It never scores
rows, writes filtering stage state, supervises workers, or publishes raw source
locators in reports and errors.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import math
import os
from pathlib import Path
from typing import Any

from twc.repro.hashing import stable_json_hash, self_hash_mismatches
from twc.pretraining.filtering import artifacts as filtering_artifacts


TOKEN_TARGET_TRANCHE_MODE = "token_target_stratified_tranche_v1"
ROW_GROUP_SHARDING_POLICY_ORDINAL = "ordinal_modulo_v1"
ROW_GROUP_SHARDING_POLICY_WEIGHTED_LPT = "weighted_lpt_v1"
ROW_GROUP_SHARDING_POLICY_SOURCE_LOCAL_WEIGHTED_LPT = "source_local_weighted_lpt_v1"
ROW_GROUP_SHARDING_POLICIES = {
    ROW_GROUP_SHARDING_POLICY_ORDINAL,
    ROW_GROUP_SHARDING_POLICY_WEIGHTED_LPT,
    ROW_GROUP_SHARDING_POLICY_SOURCE_LOCAL_WEIGHTED_LPT,
}

_COPY_CHUNK_BYTES = 16 * 1024 * 1024
_MIN_MIRROR_RESERVE_BYTES = 50 * 1024**3


class _MirrorDiskStop(Exception):
    """Internal control flow for a durable low-disk stop."""


def _local_parquet_mirror_path(mirror_dir: str | Path, source_file_id: str) -> Path:
    if not source_file_id:
        raise ValueError("local mirror source_file_id is required")
    return Path(mirror_dir) / f"{stable_json_hash(source_file_id)}.parquet"


def _schedule_hash_payload(schedule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in schedule.items()
        if key not in {"schedule_hash", "generated_at_utc"}
    }


def _validate_private_source_schedule_hash(schedule: Mapping[str, Any]) -> str:
    schedule_hash = filtering_artifacts._require_sha256(
        schedule.get("schedule_hash"), "schedule_hash"
    )
    current_payload_hash = stable_json_hash(
        {key: value for key, value in schedule.items() if key != "schedule_hash"}
    )
    source_schedule_payload_hash = stable_json_hash(_schedule_hash_payload(schedule))
    if schedule_hash not in {current_payload_hash, source_schedule_payload_hash}:
        raise ValueError("private source schedule hash mismatch")
    return schedule_hash


def _local_parquet_metadata(path: Path) -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - optional production dependency
        raise RuntimeError("pyarrow is required to validate local Parquet mirror files") from None
    try:
        return pq.ParquetFile(path).metadata
    except Exception:
        raise ValueError("local mirror Parquet validation failed") from None


def _validate_local_parquet_entry(
    path: Path,
    entry: Mapping[str, Any],
    metadata_cache: dict[Path, Any],
) -> None:
    if not path.exists():
        source_file_id_hash = stable_json_hash(str(entry.get("source_file_id") or ""))
        raise FileNotFoundError(
            f"local mirror file missing for source_file_id_hash {source_file_id_hash}"
        )
    if path.is_symlink():
        raise ValueError("local mirror file must not be a symlink")
    metadata = path.lstat()
    if not path.is_file() or int(metadata.st_nlink) != 1:
        raise ValueError("local mirror path must be a single-link regular file")
    if os.name == "posix":
        if int(metadata.st_uid) != int(os.geteuid()):
            raise ValueError("local mirror file must belong to the current user")
        if metadata.st_mode & 0o777 != 0o600:
            raise ValueError("local mirror file must use mode 0600")
    parquet_metadata = metadata_cache.get(path)
    if parquet_metadata is None:
        parquet_metadata = _local_parquet_metadata(path)
        metadata_cache[path] = parquet_metadata
    row_group_index = int(entry["row_group_index"])
    if row_group_index < 0 or row_group_index >= int(parquet_metadata.num_row_groups):
        raise ValueError("local mirror row_group_index out of bounds")
    expected_num_rows = entry.get("row_group_num_rows")
    if expected_num_rows is not None and int(expected_num_rows) != int(
        parquet_metadata.row_group(row_group_index).num_rows
    ):
        raise ValueError("local mirror row_group_num_rows drift")


def _local_parquet_identity(
    entry: Mapping[str, Any],
    path: Path,
    *,
    file_sha256: str,
) -> dict[str, Any]:
    identity = dict(entry.get("parquet_identity") or {})
    for field in ("url", "source_path", "filename"):
        identity.pop(field, None)
    identity.update(
        {
            "source_file_id_hash": stable_json_hash(str(entry["source_file_id"])),
            "local_mirror_file_name_hash": stable_json_hash(path.name),
            "local_mirror_file_size_bytes": int(path.stat().st_size),
            "local_mirror_file_sha256": file_sha256,
            "raw_url_serialized": False,
        }
    )
    return identity


def _local_mirror_file_ready_payload(
    *,
    source_file_id: str,
    parent_schedule_hash: str,
    source_metadata_hash: str,
    local_file: Path,
    entries: Sequence[Mapping[str, Any]],
    file_sha256: str,
) -> dict[str, Any]:
    row_group_indexes = sorted({int(entry["row_group_index"]) for entry in entries})
    offset_count = sum(len(entry.get("row_group_local_offsets") or []) for entry in entries)
    return {
        "artifact_kind": "twc_private_local_parquet_mirror_file_ready_v1",
        "parent_schedule_hash": parent_schedule_hash,
        "source_metadata_hash": source_metadata_hash,
        "source_file_id_hash": stable_json_hash(source_file_id),
        "source_entry_hashes": sorted(str(entry.get("entry_hash") or "") for entry in entries),
        "row_group_indexes": row_group_indexes,
        "requested_offset_count": int(offset_count),
        "local_mirror_file_name_hash": stable_json_hash(local_file.name),
        "local_mirror_file_size_bytes": int(local_file.stat().st_size),
        "local_mirror_file_sha256": file_sha256,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }


def _write_local_mirror_ready(path: Path, payload: Mapping[str, Any]) -> None:
    ready_payload = dict(payload)
    ready_payload["ready_hash"] = stable_json_hash(ready_payload)
    ready_path = path.with_suffix(path.suffix + ".ready.json")
    filtering_artifacts._write_json_atomic(ready_path, ready_payload)


def _validate_local_mirror_ready(path: Path, expected_payload: Mapping[str, Any]) -> None:
    ready_path = path.with_suffix(path.suffix + ".ready.json")
    if not ready_path.exists():
        raise ValueError("local mirror ready file missing")
    try:
        metadata = ready_path.lstat()
    except OSError:
        raise ValueError("local mirror ready file is unavailable") from None
    try:
        filtering_artifacts._validate_private_partial_metadata(metadata)
    except ValueError:
        raise ValueError("local mirror ready file is unsafe") from None
    ready = filtering_artifacts._load_json(ready_path)
    if not isinstance(ready, Mapping):
        raise ValueError("local mirror ready file must be a JSON object")
    filtering_artifacts.validate_public_no_raw_payload(ready)
    if ready.get("artifact_kind") != "twc_private_local_parquet_mirror_file_ready_v1":
        raise ValueError("local mirror ready file kind mismatch")
    if self_hash_mismatches(ready, "ready_hash"):
        raise ValueError("local mirror ready file hash mismatch")
    for key, value in expected_payload.items():
        if ready.get(key) != value:
            raise ValueError("local mirror ready file hash mismatch")


def _declared_source_size(
    entry: Mapping[str, Any],
    *,
    allow_local_sources: bool,
    proven_local_size: int | None = None,
) -> int:
    identity = entry.get("parquet_identity")
    values: list[int] = []
    if isinstance(identity, Mapping):
        for field in ("declared_size", "content_length", "x_linked_size"):
            if identity.get(field) is not None:
                values.append(int(identity[field]))
    if entry.get("source_file_size_bytes") is not None:
        values.append(int(entry["source_file_size_bytes"]))
    if any(value < 0 for value in values) or len(set(values)) > 1:
        raise ValueError("source identity or declared-size drift")
    if values:
        return values[0]
    if proven_local_size is not None:
        if proven_local_size < 0:
            raise ValueError("local mirror size must be nonnegative")
        return proven_local_size
    source = str(entry.get("parquet_url_or_path") or "")
    if allow_local_sources:
        try:
            return int(Path(source).stat().st_size)
        except OSError:
            raise FileNotFoundError("local test source file missing") from None
    raise ValueError("source identity is missing a declared size")


def _frozen_source_identity(entry: Mapping[str, Any]) -> str:
    identity = entry.get("parquet_identity")
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("source entry is missing frozen identity")
    return stable_json_hash(dict(identity))


def _group_source_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    allow_local_sources: bool,
    proven_local_sizes: Mapping[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("private source schedule entries must be mappings")
        source_file_id = str(entry.get("source_file_id") or "")
        if not source_file_id:
            raise ValueError("private source schedule entry missing source_file_id")
        source = str(entry.get("parquet_url_or_path") or "")
        if not source:
            raise ValueError("private source schedule entry missing parquet_url_or_path")
        declared_size = _declared_source_size(
            entry,
            allow_local_sources=allow_local_sources,
            proven_local_size=(
                proven_local_sizes.get(source_file_id)
                if proven_local_sizes is not None
                else None
            ),
        )
        identity_hash = _frozen_source_identity(entry)
        group = grouped.setdefault(
            source_file_id,
            {
                "source": source,
                "declared_size": declared_size,
                "identity_hash": identity_hash,
                "entries": [],
            },
        )
        if (
            group["source"] != source
            or int(group["declared_size"]) != declared_size
            or group["identity_hash"] != identity_hash
        ):
            raise ValueError("source identity or declared-size drift")
        group["entries"].append(entry)
    return grouped


@contextmanager
def _mirror_root_lock(mirror_path: Path) -> Iterator[None]:
    if os.name != "posix":  # pragma: no cover - production hosts are POSIX
        raise RuntimeError("source mirroring requires POSIX file locking")
    lock_path = mirror_path / ".mirror.lock"
    create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        create_flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, create_flags, 0o600)
        expected = os.fstat(fd)
        filtering_artifacts._validate_private_partial_metadata(expected)
    except FileExistsError:
        try:
            expected = filtering_artifacts._validate_owned_private_partial(lock_path)
            open_flags = os.O_RDWR | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
            fd = os.open(lock_path, open_flags)
        except (OSError, ValueError):
            raise RuntimeError("source mirror lock is unsafe or unavailable") from None
    except OSError:
        raise RuntimeError("source mirror lock is unsafe or unavailable") from None
    try:
        opened = os.fstat(fd)
        try:
            filtering_artifacts._validate_private_partial_metadata(opened)
            current = filtering_artifacts._validate_owned_private_partial(lock_path)
        except ValueError:
            raise RuntimeError("source mirror lock is unsafe or unavailable")
        if (
            (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino)
            or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise RuntimeError("source mirror lock changed while opening")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("source mirror writer is already active") from None
        yield
    finally:
        os.close(fd)


def _open_private_source(source: str, *, allow_local_sources: bool) -> Any:
    if source.startswith("https://"):
        try:
            import fsspec
        except ImportError:  # pragma: no cover - optional production dependency
            raise RuntimeError("fsspec is required to mirror HTTPS Parquet sources") from None
        return fsspec.open(source, mode="rb")
    if allow_local_sources:
        local_source = Path(source)
        if local_source.is_symlink() or not local_source.is_file():
            raise FileNotFoundError("local test source file missing")
        return local_source.open("rb")
    raise ValueError("source mirroring requires HTTPS sources")


def _copy_private_schedule_source(
    source: str,
    part_path: Path,
    *,
    allow_local_sources: bool,
    stop_path: Path | None,
    reserve_bytes: int,
    declared_size: int,
) -> tuple[str, int, bool]:
    digest = hashlib.sha256()
    bytes_written = 0
    fd = filtering_artifacts._create_private_partial(part_path)
    try:
        with os.fdopen(fd, "wb") as destination:
            with _open_private_source(source, allow_local_sources=allow_local_sources) as source_handle:
                for chunk in iter(lambda: source_handle.read(_COPY_CHUNK_BYTES), b""):
                    if not chunk:
                        break
                    if filtering_artifacts._path_free_bytes(part_path) < reserve_bytes + len(chunk):
                        filtering_artifacts._flush_and_fsync(destination)
                        raise _MirrorDiskStop
                    if bytes_written + len(chunk) > int(declared_size):
                        raise ValueError("source exceeded its declared size")
                    digest.update(chunk)
                    destination.write(chunk)
                    bytes_written += len(chunk)
                    destination.flush()
                    if filtering_artifacts._path_free_bytes(part_path) < reserve_bytes + len(chunk):
                        filtering_artifacts._flush_and_fsync(destination)
                        raise _MirrorDiskStop
                    if stop_path is not None and stop_path.exists():
                        filtering_artifacts._flush_and_fsync(destination)
                        return digest.hexdigest(), bytes_written, False
                filtering_artifacts._flush_and_fsync(destination)
    except BaseException:
        raise
    return digest.hexdigest(), bytes_written, True


def _source_failure(source_file_id: str, reason: str) -> RuntimeError:
    return RuntimeError(
        f"source mirror {reason} for source_file_id_hash {stable_json_hash(source_file_id)}"
    )


def _validated_ready_file(
    *,
    local_file: Path,
    source_file_id: str,
    entries: Sequence[Mapping[str, Any]],
    declared_size: int,
    parent_schedule_hash: str,
    source_metadata_hash: str,
    metadata_cache: dict[Path, Any],
) -> tuple[str, dict[str, Any]]:
    if int(local_file.stat().st_size) != int(declared_size):
        raise ValueError("local mirror declared size drift")
    file_sha256 = filtering_artifacts._sha256_file(local_file)
    for entry in entries:
        _validate_local_parquet_entry(local_file, entry, metadata_cache)
    ready_payload = _local_mirror_file_ready_payload(
        source_file_id=source_file_id,
        parent_schedule_hash=parent_schedule_hash,
        source_metadata_hash=source_metadata_hash,
        local_file=local_file,
        entries=entries,
        file_sha256=file_sha256,
    )
    _validate_local_mirror_ready(local_file, ready_payload)
    return file_sha256, ready_payload


def build_local_parquet_execution_schedule(
    private_source_schedule: Mapping[str, Any],
    mirror_dir: str | Path,
) -> dict[str, Any]:
    """Build a private local schedule only from files with exact ready proof."""

    parent_schedule_hash = _validate_private_source_schedule_hash(private_source_schedule)
    source_metadata_hash = filtering_artifacts._require_sha256(
        private_source_schedule.get("source_metadata_hash"), "source_metadata_hash"
    )
    entries = private_source_schedule.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("private source schedule entries must be a non-empty list")
    mirror_path = Path(mirror_dir)
    filtering_artifacts.validate_private_source_cache_dir(mirror_path)
    proven_local_sizes: dict[str, int] = {}
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"private source schedule entry {entry_index} is not an object")
        source_file_id = str(entry.get("source_file_id") or "")
        if not source_file_id:
            raise ValueError("private source schedule entry missing source_file_id")
        local_file = _local_parquet_mirror_path(mirror_path, source_file_id)
        try:
            local_stat = local_file.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"local mirror file missing for source_file_id_hash {stable_json_hash(source_file_id)}"
            ) from None
        proven_local_sizes[source_file_id] = int(local_stat.st_size)
    grouped = _group_source_entries(
        entries,
        allow_local_sources=True,
        proven_local_sizes=proven_local_sizes,
    )
    metadata_cache: dict[Path, Any] = {}
    file_sha_cache: dict[Path, str] = {}
    for source_file_id, group in grouped.items():
        local_file = _local_parquet_mirror_path(mirror_path, source_file_id)
        file_sha256, _ = _validated_ready_file(
            local_file=local_file,
            source_file_id=source_file_id,
            entries=group["entries"],
            declared_size=int(group["declared_size"]),
            parent_schedule_hash=parent_schedule_hash,
            source_metadata_hash=source_metadata_hash,
            metadata_cache=metadata_cache,
        )
        file_sha_cache[local_file] = file_sha256
    local_entries: list[dict[str, Any]] = []
    local_files: set[Path] = set()
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"private source schedule entry {entry_index} is not an object")
        source_file_id = str(entry.get("source_file_id") or "")
        local_file = _local_parquet_mirror_path(mirror_path, source_file_id)
        local_files.add(local_file)
        local_entry = dict(entry)
        local_entry["parquet_url_or_path"] = str(local_file)
        local_entry["parquet_identity"] = _local_parquet_identity(
            entry,
            local_file,
            file_sha256=file_sha_cache[local_file],
        )
        local_entry["entry_hash"] = stable_json_hash(
            {key: value for key, value in local_entry.items() if key != "entry_hash"}
        )
        local_entries.append(local_entry)
    local_schedule = dict(private_source_schedule)
    local_schedule.update(
        {
            "artifact_kind": filtering_artifacts.LOCAL_PARQUET_EXECUTION_SCHEDULE_KIND,
            "parent_schedule_hash": parent_schedule_hash,
            "source_metadata_hash": source_metadata_hash,
            "local_mirror_dir_sha256": filtering_artifacts._sha256_text(str(mirror_path)),
            "local_mirror_file_count": len(local_files),
            "entries": local_entries,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
    )
    local_schedule["schedule_hash"] = stable_json_hash(_schedule_hash_payload(local_schedule))
    return local_schedule


def mirror_private_schedule_sources(
    private_source_schedule: Mapping[str, Any],
    mirror_dir: str | Path,
    *,
    allow_local_sources: bool = False,
    stop_file: str | Path | None = None,
    min_free_bytes: int = 0,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Mirror source files with resumable stop boundaries and no-raw reporting."""

    parent_schedule_hash = _validate_private_source_schedule_hash(private_source_schedule)
    source_metadata_hash = filtering_artifacts._require_sha256(
        private_source_schedule.get("source_metadata_hash"), "source_metadata_hash"
    )
    entries = private_source_schedule.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("private source schedule entries must be a non-empty list")
    mirror_path = Path(mirror_dir)
    filtering_artifacts._ensure_private_cache_dir(mirror_path)
    stop_path = Path(stop_file) if stop_file is not None else None
    grouped = _group_source_entries(entries, allow_local_sources=allow_local_sources)
    unique_total = sum(int(group["declared_size"]) for group in grouped.values())
    reserve_bytes = max(
        filtering_artifacts._nonnegative_int(min_free_bytes, "min_free_bytes"),
        _MIN_MIRROR_RESERVE_BYTES,
        math.ceil(unique_total / 10),
    )
    remaining_bytes = 0
    for source_file_id, group in grouped.items():
        local_file = _local_parquet_mirror_path(mirror_path, source_file_id)
        ready_path = local_file.with_suffix(local_file.suffix + ".ready.json")
        if ready_path.exists() and not local_file.exists():
            raise ValueError("local mirror ready proof exists without its final file")
        if not local_file.exists():
            remaining_bytes += int(group["declared_size"])
    preflight_low_disk = (
        filtering_artifacts._path_free_bytes(mirror_path) < remaining_bytes + reserve_bytes
    )

    file_reports: list[dict[str, Any]] = []
    completed = not preflight_low_disk
    stop_reason = "low_disk" if preflight_low_disk else None
    metadata_cache: dict[Path, Any] = {}
    with _mirror_root_lock(mirror_path) if completed else nullcontext():
        for source_file_id, group in (sorted(grouped.items()) if completed else []):
            if stop_path is not None and stop_path.exists():
                completed = False
                break
            local_file = _local_parquet_mirror_path(mirror_path, source_file_id)
            ready_path = local_file.with_suffix(local_file.suffix + ".ready.json")
            part_path = local_file.with_suffix(local_file.suffix + ".part")
            if local_file.exists() and ready_path.exists():
                status = "reused"
            elif local_file.exists():
                status = "recovered"
            else:
                status = "mirrored"
            try:
                if status == "reused":
                    if part_path.exists() or part_path.is_symlink():
                        raise ValueError("private partial conflicts with ready mirror")
                    file_sha256, _ = _validated_ready_file(
                        local_file=local_file,
                        source_file_id=source_file_id,
                        entries=group["entries"],
                        declared_size=int(group["declared_size"]),
                        parent_schedule_hash=parent_schedule_hash,
                        source_metadata_hash=source_metadata_hash,
                        metadata_cache=metadata_cache,
                    )
                elif status == "recovered":
                    if part_path.exists() or part_path.is_symlink():
                        raise ValueError("private partial conflicts with unproven final mirror")
                    file_sha256 = filtering_artifacts._sha256_file(local_file)
                    if int(local_file.stat().st_size) != int(group["declared_size"]):
                        raise ValueError("local mirror declared size drift")
                    for entry in group["entries"]:
                        _validate_local_parquet_entry(local_file, entry, metadata_cache)
                    ready_payload = _local_mirror_file_ready_payload(
                        source_file_id=source_file_id,
                        parent_schedule_hash=parent_schedule_hash,
                        source_metadata_hash=source_metadata_hash,
                        local_file=local_file,
                        entries=group["entries"],
                        file_sha256=file_sha256,
                    )
                    _write_local_mirror_ready(local_file, ready_payload)
                else:
                    file_sha256, copied_bytes, copy_complete = _copy_private_schedule_source(
                        str(group["source"]),
                        part_path,
                        allow_local_sources=allow_local_sources,
                        stop_path=stop_path,
                        reserve_bytes=reserve_bytes,
                        declared_size=int(group["declared_size"]),
                    )
                    if not copy_complete:
                        completed = False
                        break
                    if copied_bytes != int(group["declared_size"]):
                        raise ValueError("local mirror copied byte count drift")
                    if int(part_path.stat().st_size) != copied_bytes:
                        raise ValueError("local mirror copied byte count drift")
                    for entry in group["entries"]:
                        _validate_local_parquet_entry(part_path, entry, metadata_cache)
                    filtering_artifacts._promote_private_partial(part_path, local_file)
                    ready_payload = _local_mirror_file_ready_payload(
                        source_file_id=source_file_id,
                        parent_schedule_hash=parent_schedule_hash,
                        source_metadata_hash=source_metadata_hash,
                        local_file=local_file,
                        entries=group["entries"],
                        file_sha256=file_sha256,
                    )
                    _write_local_mirror_ready(local_file, ready_payload)
                file_reports.append(
                    {
                        "source_file_id_hash": stable_json_hash(source_file_id),
                        "local_mirror_file_name_hash": stable_json_hash(local_file.name),
                        "local_mirror_file_size_bytes": int(local_file.stat().st_size),
                        "local_mirror_file_sha256": file_sha256,
                        "entry_count": len(group["entries"]),
                        "status": status,
                    }
                )
            except _MirrorDiskStop:
                completed = False
                stop_reason = "low_disk"
                break
            except (ValueError, RuntimeError, OSError):
                raise _source_failure(source_file_id, "failed") from None
            if stop_path is not None and stop_path.exists():
                completed = False
                stop_reason = "requested"
                break
    local_schedule = (
        build_local_parquet_execution_schedule(private_source_schedule, mirror_path)
        if completed
        else None
    )
    report = {
        "artifact_kind": filtering_artifacts.LOCAL_PARQUET_MIRROR_REPORT_KIND,
        "schema_version": 1,
        "status": "ready" if completed else "stopped",
        "parent_schedule_hash": parent_schedule_hash,
        "local_schedule_hash": local_schedule.get("schedule_hash") if local_schedule is not None else None,
        "source_metadata_hash": source_metadata_hash,
        "local_mirror_dir_sha256": filtering_artifacts._sha256_text(str(mirror_path)),
        "source_file_count": len(grouped),
        "mirrored_or_reused_file_count": len(file_reports),
        "entry_count": len(entries),
        "total_mirrored_bytes": sum(
            int(item["local_mirror_file_size_bytes"]) for item in file_reports
        ),
        "file_reports": file_reports,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    if not completed:
        report["stop_reason"] = stop_reason or "requested"
    report["report_hash"] = stable_json_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    filtering_artifacts.validate_public_no_raw_payload(report)
    return local_schedule, report


def _source_metadata_hash(source_metadata_manifest: Mapping[str, Any]) -> str:
    value = source_metadata_manifest.get("source_metadata_hash") or source_metadata_manifest.get(
        "manifest_hash"
    )
    return filtering_artifacts._require_sha256(value, "source_metadata_hash")


def _source_metadata_files_by_id(
    source_metadata_manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    files = source_metadata_manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source metadata files must be a non-empty list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("source metadata file entries must be mappings")
        file_id = str(item.get("file_id") or "")
        if not file_id:
            raise ValueError("source metadata file entry missing file_id")
        if file_id in by_id:
            raise ValueError("source metadata contains duplicate file_id")
        by_id[file_id] = item
    return by_id


def _source_metadata_footers_by_id(
    source_metadata_manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    footers = source_metadata_manifest.get("parquet_footers")
    if not isinstance(footers, list) or not footers:
        raise ValueError("source metadata parquet_footers must be a non-empty list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in footers:
        if not isinstance(item, Mapping):
            raise ValueError("source metadata footer entries must be mappings")
        file_id = str(item.get("file_id") or "")
        if not file_id:
            raise ValueError("source metadata footer entry missing file_id")
        if file_id in by_id:
            raise ValueError("source metadata contains duplicate footer file_id")
        by_id[file_id] = item
    return by_id


def _validate_local_parquet_execution_schedule(
    local_schedule: Mapping[str, Any],
) -> None:
    if local_schedule.get("artifact_kind") != filtering_artifacts.LOCAL_PARQUET_EXECUTION_SCHEDULE_KIND:
        raise ValueError("local Parquet execution schedule kind mismatch")
    schedule_hash = filtering_artifacts._require_sha256(
        local_schedule.get("schedule_hash"), "local_schedule_hash"
    )
    accepted_hashes = {
        stable_json_hash(
            {key: value for key, value in local_schedule.items() if key != "schedule_hash"}
        ),
        stable_json_hash(_schedule_hash_payload(local_schedule)),
    }
    if schedule_hash not in accepted_hashes:
        raise ValueError("local Parquet execution schedule hash mismatch")
    filtering_artifacts._require_sha256(
        local_schedule.get("parent_schedule_hash"), "parent_schedule_hash"
    )
    filtering_artifacts._require_sha256(
        local_schedule.get("source_metadata_hash"), "source_metadata_hash"
    )
    entries = local_schedule.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("local schedule entries must be a non-empty list")
    seen: set[tuple[str, int]] = set()
    source_bindings: dict[str, tuple[str, Mapping[str, Any]]] = {}
    metadata_cache: dict[Path, Any] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            raise ValueError("local schedule entries must be mappings")
        source_file_id = str(item.get("source_file_id") or "")
        try:
            row_group_index = int(item.get("row_group_index"))
        except (TypeError, ValueError):
            raise ValueError("local schedule row_group_index must be an integer") from None
        key = (source_file_id, row_group_index)
        if not source_file_id or row_group_index < 0 or key in seen:
            raise ValueError("local schedule contains an invalid duplicate source entry")
        seen.add(key)
        if self_hash_mismatches(item, "entry_hash"):
            raise ValueError("local schedule entry_hash mismatch")
        row_group_num_rows = filtering_artifacts._positive_int(
            item.get("row_group_num_rows"), "row_group_num_rows"
        )
        selection_mode = str(item.get("row_selection_mode") or "")
        if selection_mode == "explicit_offsets":
            offsets = item.get("row_group_local_offsets")
            if not isinstance(offsets, list) or not offsets:
                raise ValueError("local schedule explicit offsets must be a non-empty list")
            try:
                parsed_offsets = [int(value) for value in offsets]
            except (TypeError, ValueError):
                raise ValueError("local schedule offsets must be integers") from None
            if (
                len(set(parsed_offsets)) != len(parsed_offsets)
                or any(value < 0 or value >= row_group_num_rows for value in parsed_offsets)
            ):
                raise ValueError("local schedule offsets are duplicate or out of bounds")
        elif selection_mode == "all_rows_in_row_group":
            if "row_group_local_offsets" in item:
                raise ValueError("local schedule all-row entry must not contain offsets")
        else:
            raise ValueError("local schedule row_selection_mode mismatch")
        local_path = Path(str(item.get("parquet_url_or_path") or ""))
        identity = item.get("parquet_identity")
        if not str(local_path) or not isinstance(identity, Mapping):
            raise ValueError("local schedule source binding is missing")
        binding = (str(local_path), identity)
        prior = source_bindings.setdefault(source_file_id, binding)
        if prior != binding:
            raise ValueError("local schedule source entries disagree on file identity")
        _validate_local_parquet_entry(local_path, item, metadata_cache)
    if int(local_schedule.get("local_mirror_file_count") or -1) != len(source_bindings):
        raise ValueError("local schedule local_mirror_file_count mismatch")


def _local_schedule_entries_by_source_file(
    local_schedule: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    _validate_local_parquet_execution_schedule(local_schedule)
    entries = local_schedule["entries"]
    by_file: dict[str, Mapping[str, Any]] = {}
    for item in entries:
        file_id = str(item["source_file_id"])
        by_file.setdefault(file_id, item)
    return by_file


def build_local_parquet_all_rows_prefilter_plan(
    *,
    local_schedule: Mapping[str, Any],
    source_metadata_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_local_parquet_execution_schedule(local_schedule)
    local_schedule_hash = str(local_schedule["schedule_hash"])
    source_metadata_hash = _source_metadata_hash(source_metadata_manifest)
    if local_schedule.get("source_metadata_hash") != source_metadata_hash:
        raise ValueError("local schedule source metadata hash mismatch")
    files_by_id = _source_metadata_files_by_id(source_metadata_manifest)
    footers_by_id = _source_metadata_footers_by_id(source_metadata_manifest)
    local_by_file = _local_schedule_entries_by_source_file(local_schedule)
    metadata_file_ids = set(files_by_id)
    if set(local_by_file) != metadata_file_ids:
        raise ValueError("local mirror coverage does not match source metadata files")
    if set(footers_by_id) != metadata_file_ids:
        raise ValueError("source metadata footer coverage does not match files")
    entries: list[dict[str, Any]] = []
    metadata_cache: dict[Path, Any] = {}
    expected_rows = 0
    for source_file_id in sorted(metadata_file_ids):
        local_prototype = dict(local_by_file[source_file_id])
        row_groups = footers_by_id[source_file_id].get("row_groups")
        if not isinstance(row_groups, list):
            raise ValueError("source metadata footer row_groups must be a list")
        for group in sorted(row_groups, key=lambda item: int(item.get("row_group_index", -1))):
            if not isinstance(group, Mapping):
                raise ValueError("source metadata row group entries must be mappings")
            num_rows = int(group.get("num_rows") or 0)
            if num_rows <= 0:
                continue
            entry = dict(local_prototype)
            entry["row_group_index"] = int(group["row_group_index"])
            entry["row_group_num_rows"] = num_rows
            entry["row_group_size_estimate_bytes"] = {
                "compressed": int(group.get("compressed_byte_size") or 0),
                "total": int(group.get("total_byte_size") or 0),
            }
            entry["row_selection_mode"] = "all_rows_in_row_group"
            entry.pop("row_group_local_offsets", None)
            entry["offset_recipe"] = {
                "mode": "all_rows_in_row_group",
                "count": num_rows,
                "modulus": num_rows,
            }
            entry["entry_hash"] = stable_json_hash(
                {key: value for key, value in entry.items() if key != "entry_hash"}
            )
            _validate_local_parquet_entry(
                Path(str(entry["parquet_url_or_path"])), entry, metadata_cache
            )
            entries.append(entry)
            expected_rows += num_rows
    if not entries or expected_rows <= 0:
        raise ValueError("local all-row prefilter plan has no rows")
    plan = {
        "artifact_kind": filtering_artifacts.LOCAL_PARQUET_ALL_ROWS_PREFILTER_PLAN_KIND,
        "schema_version": 1,
        "source_metadata_hash": source_metadata_hash,
        "local_schedule_hash": local_schedule_hash,
        "parent_schedule_hash": local_schedule.get("parent_schedule_hash"),
        "local_mirror_file_count": int(local_schedule.get("local_mirror_file_count") or 0),
        "source_file_count": len(metadata_file_ids),
        "row_group_count": len(entries),
        "expected_physical_rows": int(expected_rows),
        "row_selection_mode": "all_rows_in_row_group",
        "entries": entries,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    plan["schedule_hash"] = stable_json_hash(
        {key: value for key, value in plan.items() if key != "schedule_hash"}
    )
    filtering_artifacts.validate_public_no_raw_payload(
        {key: value for key, value in plan.items() if key != "entries"}
    )
    return plan


def _validate_local_parquet_prefilter_plan(
    plan: Mapping[str, Any],
    *,
    allow_shard: bool = False,
) -> None:
    kind = plan.get("artifact_kind")
    if kind == filtering_artifacts.LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND:
        validate_local_parquet_token_target_tranche_plan(plan, allow_shard=allow_shard)
        return
    if kind != filtering_artifacts.LOCAL_PARQUET_ALL_ROWS_PREFILTER_PLAN_KIND:
        raise ValueError("local Parquet prefilter plan kind mismatch")
    schedule_hash = filtering_artifacts._require_sha256(
        plan.get("schedule_hash"), "local_parquet_prefilter_plan_hash"
    )
    if schedule_hash != stable_json_hash(
        {key: value for key, value in plan.items() if key != "schedule_hash"}
    ):
        raise ValueError("local Parquet all-row prefilter plan hash mismatch")
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("local Parquet prefilter plan entries must be a non-empty list")
    if int(plan.get("row_group_count") or -1) != len(entries):
        raise ValueError("local Parquet prefilter plan row_group_count mismatch")
    expected_rows = sum(int(entry.get("row_group_num_rows") or 0) for entry in entries)
    if int(plan.get("expected_physical_rows") or -1) != expected_rows:
        raise ValueError("local Parquet prefilter plan expected_physical_rows mismatch")
    if plan.get("row_selection_mode") != "all_rows_in_row_group":
        raise ValueError("local Parquet all-row prefilter plan row_selection_mode mismatch")
    seen: set[tuple[str, int]] = set()
    source_file_ids: set[str] = set()
    metadata_cache: dict[Path, Any] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("local Parquet prefilter plan entries must be mappings")
        key = (str(entry.get("source_file_id") or ""), int(entry.get("row_group_index")))
        if not key[0] or key in seen:
            raise ValueError("local Parquet prefilter plan contains invalid duplicate row groups")
        seen.add(key)
        source_file_ids.add(key[0])
        expected_mode = (
            "all_rows_in_row_group"
            if kind == filtering_artifacts.LOCAL_PARQUET_ALL_ROWS_PREFILTER_PLAN_KIND
            else "all_rows_in_selected_row_groups"
        )
        if entry.get("row_selection_mode") != expected_mode:
            raise ValueError("local Parquet prefilter plan row_selection_mode mismatch")
        row_group_num_rows = filtering_artifacts._positive_int(
            entry.get("row_group_num_rows"), "row_group_num_rows"
        )
        offset_recipe = entry.get("offset_recipe")
        expected_recipe = {
            "mode": "all_rows_in_row_group",
            "count": row_group_num_rows,
            "modulus": row_group_num_rows,
        }
        if offset_recipe != expected_recipe:
            raise ValueError("local Parquet prefilter plan offset_recipe mismatch")
        if "row_group_local_offsets" in entry:
            raise ValueError("local Parquet prefilter plan offset_recipe conflicts with explicit offsets")
        if self_hash_mismatches(entry, "entry_hash"):
            raise ValueError("local Parquet prefilter plan entry_hash mismatch")
        _validate_local_parquet_entry(
            Path(str(entry.get("parquet_url_or_path") or "")),
            entry,
            metadata_cache,
        )
    declared_source_file_count = int(plan.get("source_file_count") or -1)
    if (
        declared_source_file_count < len(source_file_ids)
        or (not allow_shard and declared_source_file_count != len(source_file_ids))
    ):
        raise ValueError("local Parquet prefilter plan source_file_count mismatch")
    declared_mirror_file_count = int(plan.get("local_mirror_file_count") or -1)
    if (
        declared_mirror_file_count < len(source_file_ids)
        or (not allow_shard and declared_mirror_file_count != len(source_file_ids))
    ):
        raise ValueError("local Parquet prefilter plan local_mirror_file_count mismatch")
    filtering_artifacts.validate_public_no_raw_payload(
        {key: value for key, value in plan.items() if key != "entries"}
    )


def _validate_row_group_sharding_policy(policy: str | None) -> str:
    value = ROW_GROUP_SHARDING_POLICY_ORDINAL if policy is None else str(policy)
    if value not in ROW_GROUP_SHARDING_POLICIES:
        raise ValueError(
            "row_group_sharding_policy must be ordinal_modulo_v1, weighted_lpt_v1, "
            "or source_local_weighted_lpt_v1"
        )
    return value


def _row_group_sharding_weight(entry: Mapping[str, Any]) -> int:
    value = entry.get("row_group_source_token_sum", entry.get("row_group_num_rows"))
    try:
        weight = int(value)
    except (TypeError, ValueError):
        raise ValueError("row group sharding weight must be an integer") from None
    if weight < 0:
        raise ValueError("row group sharding weight must be nonnegative")
    return max(1, weight)


def _row_group_sharding_source_hash(entry: Mapping[str, Any]) -> str:
    existing = entry.get("source_file_id_hash")
    if isinstance(existing, str) and len(existing) == 64:
        return existing
    return stable_json_hash(str(entry.get("source_file_id") or ""))


def _source_local_lpt_units(
    entries: Sequence[Mapping[str, Any]], *, shard_count: int
) -> list[list[int]]:
    total_weight = sum(_row_group_sharding_weight(entry) for entry in entries)
    target_weight = max(1, math.ceil(total_weight / max(1, int(shard_count))))
    groups: dict[str, list[int]] = {}
    for ordinal, entry in enumerate(entries):
        groups.setdefault(_row_group_sharding_source_hash(entry), []).append(ordinal)
    units: list[list[int]] = []
    for source_hash in sorted(groups):
        ordinals = sorted(
            groups[source_hash],
            key=lambda ordinal: (int(entries[ordinal].get("row_group_index", 0)), ordinal),
        )
        current: list[int] = []
        current_weight = 0
        for ordinal in ordinals:
            weight = _row_group_sharding_weight(entries[ordinal])
            if current and current_weight + weight > target_weight:
                units.append(current)
                current = []
                current_weight = 0
            current.append(ordinal)
            current_weight += weight
        if current:
            units.append(current)
    while len(units) < int(shard_count):
        splittable = [index for index, unit in enumerate(units) if len(unit) > 1]
        if not splittable:
            break
        split_index = max(
            splittable,
            key=lambda index: (
                sum(_row_group_sharding_weight(entries[o]) for o in units[index]),
                len(units[index]),
                -min(units[index]),
            ),
        )
        unit = units.pop(split_index)
        half_weight = sum(_row_group_sharding_weight(entries[o]) for o in unit) / 2.0
        left: list[int] = []
        right: list[int] = []
        running_weight = 0
        for ordinal in unit:
            if left and running_weight >= half_weight:
                right.append(ordinal)
            else:
                left.append(ordinal)
                running_weight += _row_group_sharding_weight(entries[ordinal])
        if not left or not right:
            midpoint = max(1, len(unit) // 2)
            left, right = unit[:midpoint], unit[midpoint:]
        units.extend([left, right])
    return units


def _assigned_row_group_ordinals(
    entries: Sequence[Mapping[str, Any]],
    *,
    shard_count: int,
    row_group_sharding_policy: str,
) -> list[list[int]]:
    count = filtering_artifacts._positive_int(shard_count, "row_group_shard_count")
    policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    if count > len(entries):
        raise ValueError("row_group_shard_count cannot exceed row_group_count")
    if policy == ROW_GROUP_SHARDING_POLICY_ORDINAL:
        return [
            [ordinal for ordinal in range(len(entries)) if ordinal % count == index]
            for index in range(count)
        ]
    shard_ordinals: list[list[int]] = [[] for _ in range(count)]
    shard_weights = [0 for _ in range(count)]
    units = (
        _source_local_lpt_units(entries, shard_count=count)
        if policy == ROW_GROUP_SHARDING_POLICY_SOURCE_LOCAL_WEIGHTED_LPT
        else [[ordinal] for ordinal in range(len(entries))]
    )
    ordered_units = sorted(
        units,
        key=lambda unit: (
            -sum(_row_group_sharding_weight(entries[o]) for o in unit),
            _row_group_sharding_source_hash(entries[unit[0]]),
            int(entries[unit[0]].get("row_group_index", 0)),
            unit[0],
        ),
    )
    for unit in ordered_units:
        shard_index = min(range(count), key=lambda index: (shard_weights[index], index))
        shard_ordinals[shard_index].extend(unit)
        shard_weights[shard_index] += sum(
            _row_group_sharding_weight(entries[ordinal]) for ordinal in unit
        )
    return [sorted(ordinals) for ordinals in shard_ordinals]


def _shard_local_parquet_prefilter_plan(
    plan: Mapping[str, Any],
    *,
    shard_count: int,
    shard_index: int,
    row_group_sharding_policy: str = ROW_GROUP_SHARDING_POLICY_ORDINAL,
) -> dict[str, Any]:
    _validate_local_parquet_prefilter_plan(plan)
    count = filtering_artifacts._positive_int(shard_count, "row_group_shard_count")
    index = int(shard_index)
    if index < 0 or index >= count:
        raise ValueError("row_group_shard_index must be within row_group_shard_count")
    policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("local all-row prefilter plan has no entries to shard")
    assigned_ordinals = _assigned_row_group_ordinals(
        entries,
        shard_count=count,
        row_group_sharding_policy=policy,
    )
    return _shard_local_parquet_prefilter_plan_from_assigned_ordinals(
        plan,
        entries=entries,
        assigned_ordinals=assigned_ordinals,
        shard_count=count,
        shard_index=index,
        row_group_sharding_policy=policy,
    )


def _shard_local_parquet_prefilter_plan_from_assigned_ordinals(
    plan: Mapping[str, Any],
    *,
    entries: Sequence[Mapping[str, Any]],
    assigned_ordinals: Sequence[Sequence[int]],
    shard_count: int,
    shard_index: int,
    row_group_sharding_policy: str,
) -> dict[str, Any]:
    count = filtering_artifacts._positive_int(shard_count, "row_group_shard_count")
    index = int(shard_index)
    if index < 0 or index >= count:
        raise ValueError("row_group_shard_index must be within row_group_shard_count")
    if len(assigned_ordinals) != count:
        raise ValueError("row group shard assignment count mismatch")
    policy = _validate_row_group_sharding_policy(row_group_sharding_policy)
    shard_entries = [dict(entries[ordinal]) for ordinal in assigned_ordinals[index]]
    if not shard_entries:
        raise ValueError("row group shard has no entries")
    plan_hash = filtering_artifacts._require_sha256(
        plan.get("schedule_hash"), "local_parquet_prefilter_plan_hash"
    )
    shard_plan = dict(plan)
    shard_plan.update(
        {
            "entries": shard_entries,
            "row_group_count": len(shard_entries),
            "expected_physical_rows": sum(
                int(entry["row_group_num_rows"]) for entry in shard_entries
            ),
            "row_group_sharding_enabled": count > 1,
            "row_group_shard_count": count,
            "row_group_shard_index": index,
            "local_parquet_prefilter_plan_hash": plan_hash,
            "local_parquet_prefilter_plan_kind": plan["artifact_kind"],
            "source_plan_row_group_count": int(plan["row_group_count"]),
            "source_plan_expected_physical_rows": int(plan["expected_physical_rows"]),
        }
    )
    if plan["artifact_kind"] == filtering_artifacts.LOCAL_PARQUET_ALL_ROWS_PREFILTER_PLAN_KIND:
        shard_plan.update(
            {
                "all_rows_prefilter_plan_hash": plan_hash,
                "all_rows_row_group_count": int(plan["row_group_count"]),
                "all_rows_expected_physical_rows": int(plan["expected_physical_rows"]),
            }
        )
    else:
        shard_plan.update(
            {
                "token_target_tranche_plan_hash": plan_hash,
                "token_target_tranche_row_group_count": int(plan["row_group_count"]),
                "token_target_tranche_expected_physical_rows": int(plan["expected_physical_rows"]),
            }
        )
    if policy != ROW_GROUP_SHARDING_POLICY_ORDINAL:
        shard_plan["row_group_sharding_policy"] = policy
    shard_plan["schedule_hash"] = stable_json_hash(
        {key: value for key, value in shard_plan.items() if key != "schedule_hash"}
    )
    filtering_artifacts.validate_public_no_raw_payload(
        {key: value for key, value in shard_plan.items() if key != "entries"}
    )
    _validate_local_parquet_prefilter_plan(shard_plan, allow_shard=True)
    return shard_plan


def _local_parquet_token_count_stats(
    path: Path, row_group_index: int, *, expected_rows: int
) -> dict[str, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - optional production dependency
        raise RuntimeError("pyarrow is required to build local Parquet token tranche plans") from None
    try:
        table = pq.ParquetFile(path).read_row_group(
            int(row_group_index), columns=["token_count"]
        )
    except Exception:
        raise ValueError("token_count column is required for token tranche planning") from None
    if "token_count" not in table.column_names:
        raise ValueError("token_count column is required for token tranche planning")
    values = table.column("token_count").to_pylist()
    if len(values) != int(expected_rows):
        raise ValueError("token_count row count drift in local Parquet row group")
    counts: list[int] = []
    for value in values:
        if value is None:
            raise ValueError("token_count contains null values")
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError("token_count values must be integers") from None
        if parsed < 0:
            raise ValueError("token_count values must be nonnegative")
        counts.append(parsed)
    if not counts:
        raise ValueError("token_count row group is empty")
    return {
        "row_group_source_token_sum": int(sum(counts)),
        "row_group_token_count_min": int(min(counts)),
        "row_group_token_count_max": int(max(counts)),
    }


def _median_int(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("median requires non-empty values")
    ordered = sorted(int(value) for value in values)
    return ordered[len(ordered) // 2]


def _seeded_round_robin_by_file(
    entries: Sequence[Mapping[str, Any]], *, seed: str, band: str
) -> list[dict[str, Any]]:
    by_file: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_file.setdefault(str(entry["source_file_id"]), []).append(dict(entry))
    file_order = sorted(
        by_file,
        key=lambda source_file_id: stable_json_hash(
            {
                "mode": TOKEN_TARGET_TRANCHE_MODE,
                "seed": seed,
                "band": band,
                "source_file_id_hash": stable_json_hash(source_file_id),
            }
        ),
    )
    for source_file_id in file_order:
        by_file[source_file_id].sort(
            key=lambda entry: stable_json_hash(
                {
                    "mode": TOKEN_TARGET_TRANCHE_MODE,
                    "seed": seed,
                    "band": band,
                    "source_file_id_hash": stable_json_hash(source_file_id),
                    "row_group_index": int(entry["row_group_index"]),
                }
            )
        )
    ordered: list[dict[str, Any]] = []
    while file_order:
        next_file_order: list[str] = []
        for source_file_id in file_order:
            pending = by_file[source_file_id]
            if pending:
                ordered.append(pending.pop(0))
            if pending:
                next_file_order.append(source_file_id)
        file_order = next_file_order
    return ordered


def build_local_parquet_token_target_tranche_plan(
    *,
    local_schedule: Mapping[str, Any],
    source_metadata_manifest: Mapping[str, Any],
    estimated_source_token_target: int,
    seed: str,
) -> dict[str, Any]:
    target = filtering_artifacts._positive_int(
        estimated_source_token_target, "estimated_source_token_target"
    )
    seed_value = str(seed or "")
    if not seed_value:
        raise ValueError("token tranche seed is required")
    population = build_local_parquet_all_rows_prefilter_plan(
        local_schedule=local_schedule,
        source_metadata_manifest=source_metadata_manifest,
    )
    entries: list[dict[str, Any]] = []
    for entry in population["entries"]:
        local_file = Path(str(entry["parquet_url_or_path"]))
        token_stats = _local_parquet_token_count_stats(
            local_file,
            int(entry["row_group_index"]),
            expected_rows=int(entry["row_group_num_rows"]),
        )
        enriched = dict(entry)
        enriched.update(token_stats)
        entries.append(enriched)
    population_token_sum = sum(int(entry["row_group_source_token_sum"]) for entry in entries)
    if target >= population_token_sum:
        raise ValueError("tranche target would select the full population; use the all-row plan")
    median_tokens = _median_int([int(entry["row_group_source_token_sum"]) for entry in entries])
    bands: dict[str, list[dict[str, Any]]] = {"high_token": [], "wide_coverage": []}
    for entry in entries:
        band = (
            "high_token"
            if int(entry["row_group_source_token_sum"]) >= median_tokens
            else "wide_coverage"
        )
        enriched = dict(entry)
        enriched["token_target_tranche_band"] = band
        enriched["row_selection_mode"] = "all_rows_in_selected_row_groups"
        enriched["offset_recipe"] = {
            "mode": "all_rows_in_selected_row_groups",
            "count": int(enriched["row_group_num_rows"]),
            "modulus": int(enriched["row_group_num_rows"]),
        }
        enriched["entry_hash"] = stable_json_hash(
            {key: value for key, value in enriched.items() if key != "entry_hash"}
        )
        bands[band].append(enriched)
    ordered_by_band = {
        band: _seeded_round_robin_by_file(items, seed=seed_value, band=band)
        for band, items in bands.items()
    }
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()
    selected_token_sum = 0
    while selected_token_sum < target and any(ordered_by_band.values()):
        progressed = False
        for band in ("high_token", "wide_coverage"):
            pending = ordered_by_band[band]
            if not pending:
                continue
            entry = pending.pop(0)
            key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
            if key in selected_keys:
                raise ValueError("token tranche selected duplicate row group")
            selected.append(entry)
            selected_keys.add(key)
            selected_token_sum += int(entry["row_group_source_token_sum"])
            progressed = True
            if selected_token_sum >= target:
                break
        if not progressed:
            break
    if not selected:
        raise ValueError("token tranche selected no row groups")
    if len(selected) == len(entries):
        raise ValueError("tranche target would select the full population; use the all-row plan")
    selected_band_counts = filtering_artifacts._sorted_counter(
        Counter(str(entry["token_target_tranche_band"]) for entry in selected)
    )
    sampling_policy = {
        "mode": TOKEN_TARGET_TRANCHE_MODE,
        "seed": seed_value,
        "estimated_source_token_target": target,
        "population_estimated_source_tokens_lower_bound": int(population_token_sum),
        "population_estimated_source_tokens_midpoint": int(population_token_sum),
        "population_estimated_source_tokens_upper_bound": int(population_token_sum),
        "selected_estimated_source_tokens_lower_bound": int(selected_token_sum),
        "selected_estimated_source_tokens_midpoint": int(selected_token_sum),
        "selected_estimated_source_tokens_upper_bound": int(selected_token_sum),
        "population_row_group_count": int(population["row_group_count"]),
        "population_physical_row_count": int(population["expected_physical_rows"]),
        "selected_row_group_count": len(selected),
        "selected_physical_row_count": sum(
            int(entry["row_group_num_rows"]) for entry in selected
        ),
        "selected_row_group_counts_by_band": selected_band_counts,
        "token_target_tranche_high_token_minimum": int(median_tokens),
        "row_group_selection_unit": "whole_row_group",
        "row_group_ordering": "seeded_file_round_robin_by_band_v1",
    }
    plan = {
        "artifact_kind": filtering_artifacts.LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND,
        "schema_version": 1,
        "source_metadata_hash": population["source_metadata_hash"],
        "local_schedule_hash": population["local_schedule_hash"],
        "parent_schedule_hash": population.get("parent_schedule_hash"),
        "local_mirror_file_count": int(population.get("local_mirror_file_count") or 0),
        "source_file_count": int(population["source_file_count"]),
        "row_group_count": len(selected),
        "expected_physical_rows": int(sampling_policy["selected_physical_row_count"]),
        "row_selection_mode": "all_rows_in_selected_row_groups",
        "population_row_group_count": int(population["row_group_count"]),
        "population_expected_physical_rows": int(population["expected_physical_rows"]),
        "sampling_policy": sampling_policy,
        "sampling_policy_hash": stable_json_hash(sampling_policy),
        "full_corpus_census": False,
        "selected_equals_population": False,
        "claims_scope": "tranche_only",
        "entries": selected,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    plan["schedule_hash"] = stable_json_hash(
        {key: value for key, value in plan.items() if key != "schedule_hash"}
    )
    validate_local_parquet_token_target_tranche_plan(plan)
    return plan


def _local_parquet_token_tranche_plan_report(plan: Mapping[str, Any]) -> dict[str, Any]:
    validate_local_parquet_token_target_tranche_plan(plan)
    sampling_policy = dict(plan["sampling_policy"])
    report = {
        "artifact_kind": "twc_local_parquet_token_target_tranche_plan_report_v1",
        "schema_version": 1,
        "status": "ready",
        "plan_hash": plan["schedule_hash"],
        "source_metadata_hash": plan["source_metadata_hash"],
        "local_schedule_hash": plan["local_schedule_hash"],
        "parent_schedule_hash": plan.get("parent_schedule_hash"),
        "claims_scope": "tranche_only",
        "full_corpus_census": False,
        "selected_equals_population": False,
        "row_group_selection_unit": "whole_row_group",
        "row_selection_mode": plan["row_selection_mode"],
        "source_file_count": int(plan["source_file_count"]),
        "selected_row_group_count": int(plan["row_group_count"]),
        "selected_physical_row_count": int(plan["expected_physical_rows"]),
        "population_row_group_count": int(plan["population_row_group_count"]),
        "population_physical_row_count": int(plan["population_expected_physical_rows"]),
        "sampling_policy_hash": plan["sampling_policy_hash"],
        "sampling_mode": sampling_policy["mode"],
        "seed_hash": stable_json_hash(str(sampling_policy["seed"])),
        "estimated_source_token_target": int(
            sampling_policy["estimated_source_token_target"]
        ),
        "selected_estimated_source_tokens": int(
            sampling_policy["selected_estimated_source_tokens_midpoint"]
        ),
        "population_estimated_source_tokens": int(
            sampling_policy["population_estimated_source_tokens_midpoint"]
        ),
        "selected_row_group_counts_by_band": dict(
            sampling_policy["selected_row_group_counts_by_band"]
        ),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "pretraining_started": False,
    }
    report["report_hash"] = stable_json_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    filtering_artifacts.validate_public_no_raw_payload(report)
    return report


def validate_local_parquet_token_target_tranche_plan(
    plan: Mapping[str, Any], *, allow_shard: bool = False
) -> None:
    """Validate token-tranche plans; construction is appended below."""

    if plan.get("artifact_kind") != filtering_artifacts.LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND:
        raise ValueError("token tranche plan kind mismatch")
    if plan.get("full_corpus_census") is not False or plan.get("selected_equals_population") is not False:
        raise ValueError("token tranche plan must remain a strict non-census subset")
    if plan.get("claims_scope") != "tranche_only":
        raise ValueError("token tranche plan claims_scope must be tranche_only")
    if plan.get("row_selection_mode") != "all_rows_in_selected_row_groups":
        raise ValueError("token tranche plan row_selection_mode mismatch")
    sampling_policy = plan.get("sampling_policy")
    if not isinstance(sampling_policy, Mapping) or sampling_policy.get("mode") != TOKEN_TARGET_TRANCHE_MODE:
        raise ValueError("token tranche plan sampling mode mismatch")
    if plan.get("sampling_policy_hash") != stable_json_hash(sampling_policy):
        raise ValueError("token tranche plan sampling_policy_hash mismatch")
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("token tranche plan entries must be a non-empty list")
    if int(plan.get("row_group_count") or 0) != len(entries):
        raise ValueError("token tranche plan row_group_count mismatch")
    if not allow_shard and int(plan.get("population_row_group_count") or 0) <= len(entries):
        raise ValueError("token tranche plan must not cover the full row-group population")
    expected_rows = sum(
        filtering_artifacts._positive_int(
            entry.get("row_group_num_rows") if isinstance(entry, Mapping) else None,
            "row_group_num_rows",
        )
        for entry in entries
    )
    if int(plan.get("expected_physical_rows") or -1) != expected_rows:
        raise ValueError("token tranche plan expected_physical_rows mismatch")
    selected_token_sum = 0
    seen: set[tuple[str, int]] = set()
    selected_bands: Counter[str] = Counter()
    selected_source_file_ids: set[str] = set()
    metadata_cache: dict[Path, Any] = {}
    band_threshold_raw = sampling_policy.get("token_target_tranche_high_token_minimum")
    band_threshold = (
        None
        if band_threshold_raw is None
        else filtering_artifacts._nonnegative_int(
            band_threshold_raw, "token_target_tranche_high_token_minimum"
        )
    )
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("token tranche plan entries must be mappings")
        key = (str(entry.get("source_file_id") or ""), int(entry.get("row_group_index")))
        if not key[0] or key in seen:
            raise ValueError("token tranche plan contains duplicate row groups")
        seen.add(key)
        selected_source_file_ids.add(key[0])
        if entry.get("row_selection_mode") != "all_rows_in_selected_row_groups":
            raise ValueError("token tranche plan entry row_selection_mode mismatch")
        row_group_num_rows = filtering_artifacts._positive_int(
            entry.get("row_group_num_rows"), "row_group_num_rows"
        )
        if entry.get("offset_recipe") != {
            "mode": "all_rows_in_selected_row_groups",
            "count": row_group_num_rows,
            "modulus": row_group_num_rows,
        }:
            raise ValueError("token tranche plan entry offset_recipe mismatch")
        if "row_group_local_offsets" in entry:
            raise ValueError("token tranche plan entry conflicts with explicit offsets")
        if self_hash_mismatches(entry, "entry_hash"):
            raise ValueError("token tranche plan entry_hash mismatch")
        actual_stats = _local_parquet_token_count_stats(
            Path(str(entry.get("parquet_url_or_path") or "")),
            int(entry["row_group_index"]),
            expected_rows=row_group_num_rows,
        )
        declared_stats = {
            field: filtering_artifacts._nonnegative_int(entry.get(field), field)
            for field in (
                "row_group_source_token_sum",
                "row_group_token_count_min",
                "row_group_token_count_max",
            )
        }
        if declared_stats != actual_stats:
            raise ValueError("token tranche plan token stats do not match local Parquet")
        selected_token_sum += int(declared_stats["row_group_source_token_sum"])
        band = str(entry.get("token_target_tranche_band") or "")
        if band not in {"high_token", "wide_coverage"}:
            raise ValueError("token tranche plan entry band mismatch")
        if band_threshold is not None:
            expected_band = (
                "high_token"
                if int(declared_stats["row_group_source_token_sum"]) >= band_threshold
                else "wide_coverage"
            )
            if band != expected_band:
                raise ValueError("token tranche plan entry band provenance mismatch")
        selected_bands[band] += 1
    declared_source_file_count = filtering_artifacts._positive_int(
        plan.get("source_file_count"), "source_file_count"
    )
    if declared_source_file_count < len(selected_source_file_ids):
        raise ValueError("token tranche plan source_file_count mismatch")
    declared_mirror_file_count = filtering_artifacts._positive_int(
        plan.get("local_mirror_file_count"), "local_mirror_file_count"
    )
    if declared_mirror_file_count < len(selected_source_file_ids):
        raise ValueError("token tranche plan local_mirror_file_count mismatch")
    if not allow_shard:
        if int(sampling_policy.get("selected_row_group_count") or -1) != len(entries):
            raise ValueError("token tranche plan selected_row_group_count mismatch")
        if int(sampling_policy.get("selected_physical_row_count") or -1) != expected_rows:
            raise ValueError("token tranche plan selected_physical_row_count mismatch")
        for field in (
            "selected_estimated_source_tokens_lower_bound",
            "selected_estimated_source_tokens_midpoint",
            "selected_estimated_source_tokens_upper_bound",
        ):
            if int(sampling_policy.get(field) or -1) != selected_token_sum:
                raise ValueError("token tranche plan selected token count mismatch")
        if selected_token_sum < filtering_artifacts._positive_int(
            sampling_policy.get("estimated_source_token_target"),
            "estimated_source_token_target",
        ):
            raise ValueError("token tranche plan does not satisfy token target")
        if dict(sampling_policy.get("selected_row_group_counts_by_band") or {}) != filtering_artifacts._sorted_counter(
            selected_bands
        ):
            raise ValueError("token tranche plan selected band counts mismatch")
    if sampling_policy.get("row_group_selection_unit") != "whole_row_group":
        raise ValueError("token tranche plan selection unit mismatch")
    if sampling_policy.get("row_group_ordering") != "seeded_file_round_robin_by_band_v1":
        raise ValueError("token tranche plan ordering mismatch")
    if not str(sampling_policy.get("seed") or ""):
        raise ValueError("token tranche plan seed is required")
    population_row_groups = filtering_artifacts._positive_int(
        sampling_policy.get("population_row_group_count"), "population_row_group_count"
    )
    population_rows = filtering_artifacts._positive_int(
        sampling_policy.get("population_physical_row_count"), "population_physical_row_count"
    )
    if not allow_shard and (
        population_row_groups <= len(entries) or population_rows <= expected_rows
    ):
        raise ValueError("token tranche plan population metadata mismatch")
    population_token_values = [
        filtering_artifacts._positive_int(sampling_policy.get(field), field)
        for field in (
            "population_estimated_source_tokens_lower_bound",
            "population_estimated_source_tokens_midpoint",
            "population_estimated_source_tokens_upper_bound",
        )
    ]
    if len(set(population_token_values)) != 1 or population_token_values[0] < selected_token_sum:
        raise ValueError("token tranche plan population token metadata mismatch")
    schedule_hash = filtering_artifacts._require_sha256(
        plan.get("schedule_hash"), "token_tranche_plan_hash"
    )
    if schedule_hash != stable_json_hash(
        {key: value for key, value in plan.items() if key != "schedule_hash"}
    ):
        raise ValueError("token tranche plan hash mismatch")
