"""Strict launch-scoped READY publication and activation classification."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO, Literal, Mapping
import uuid

from twc.repro.hashing import stable_json_hash


Role = Literal["worker", "watchdog"]
_ID = re.compile(r"[0-9a-f]{32}")
_HASH = re.compile(r"[0-9a-f]{64}")
_MAX = 64 * 1024
_FILES = {
    "worker": ("worker-ready.json", "worker.log"),
    "watchdog": ("watchdog-ready.json", "watchdog.log"),
}
_PRIVATE = {
    "raw_text_serialized": False,
    "raw_url_serialized": False,
    "tokens_serialized": False,
    "secrets_serialized": False,
}


class ReadyPublicationInProgress(ValueError):
    """The final READY link is visible before its private staging link is removed."""


def _role(value: str) -> Role:
    if value not in _FILES:
        raise ValueError("startup role must be worker or watchdog")
    return value  # type: ignore[return-value]


def _launch(value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError("launch ID must be exactly 32 lowercase hexadecimal characters")
    return value


def _hash(value: object, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")
    return value


def _positive(value: object, name: str, *, integer: bool = False) -> int | float:
    valid_type = type(value) is int if integer else type(value) in {int, float}
    if not valid_type or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return int(value) if integer else float(value)


def _open_tree(path: Path) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory(fd: int, name: str, exact_mode: int | None) -> None:
    observed = os.fstat(fd)
    mode = stat.S_IMODE(observed.st_mode)
    if not stat.S_ISDIR(observed.st_mode) or observed.st_uid != os.geteuid():
        raise ValueError(f"{name} must be an owned real directory")
    if (exact_mode is not None and mode != exact_mode) or (
        exact_mode is None and mode & 0o022
    ):
        raise ValueError(f"{name} has unsafe permissions")


def _child(parent: int, name: str, *, create: bool, exclusive: bool = False) -> int:
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
        except FileExistsError:
            if exclusive:
                raise
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    _directory(fd, name, 0o700)
    return fd


class StartupNamespace:
    """Opaque trusted directory descriptors for one launch."""

    __slots__ = ("_fds", "launch_id")

    def __init__(self, fds: tuple[int, int, int], launch_id: str) -> None:
        self._fds = list(fds)
        self.launch_id = launch_id

    @property
    def _launch_fd(self) -> int:
        if not self._fds or self._fds[-1] < 0:
            raise ValueError("startup namespace is closed")
        return self._fds[-1]

    def open_log_once(self, role: Role) -> BinaryIO:
        basename = _FILES[_role(role)][1]
        fd = os.open(
            basename, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600, dir_fd=self._launch_fd,
        )
        observed = os.fstat(fd)
        if not _safe_file(observed, expected_size=None):
            os.close(fd)
            raise ValueError("startup log must be an owned regular 0600 file")
        return os.fdopen(fd, "wb", buffering=0)

    def close(self) -> None:
        while self._fds:
            fd = self._fds.pop()
            if fd >= 0:
                os.close(fd)


def _namespace(root: Path, launch_id: str, *, create: bool) -> StartupNamespace:
    launch_id = _launch(launch_id)
    root_fd = _open_tree(root)
    fds = [root_fd]
    try:
        _directory(root_fd, "operator root", None)
        fds.append(_child(root_fd, "startup", create=create))
        fds.append(_child(fds[-1], launch_id, create=create, exclusive=create))
        return StartupNamespace(tuple(fds), launch_id)
    except BaseException:
        for fd in reversed(fds):
            os.close(fd)
        raise


def create_startup_namespace(root: Path, launch_id: str) -> StartupNamespace:
    return _namespace(root, launch_id, create=True)


def open_startup_namespace(root: Path, launch_id: str) -> StartupNamespace:
    return _namespace(root, launch_id, create=False)


def _safe_file(observed: os.stat_result, expected_size: int | None) -> bool:
    return (
        stat.S_ISREG(observed.st_mode)
        and observed.st_uid == os.geteuid()
        and stat.S_IMODE(observed.st_mode) == 0o600
        and observed.st_nlink == 1
        and (expected_size is None or 0 < observed.st_size <= expected_size)
    )


def _ready_publication_in_progress(
    observed: os.stat_result, expected_size: int
) -> bool:
    return (
        stat.S_ISREG(observed.st_mode)
        and observed.st_uid == os.geteuid()
        and stat.S_IMODE(observed.st_mode) == 0o600
        and observed.st_nlink == 2
        and 0 < observed.st_size <= expected_size
    )


def publish_ready_once(
    namespace: StartupNamespace, role: Role, payload: Mapping[str, Any]
) -> None:
    role = _role(role)
    value = load_worker_ready(payload) if role == "worker" else load_watchdog_ready(payload)
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if not 0 < len(encoded) <= _MAX:
        raise ValueError("READY payload size is outside the allowed bound")
    final = _FILES[role][0]
    partial = f".{role}-{uuid.uuid4().hex}.partial"
    directory = namespace._launch_fd
    fd = os.open(
        partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
        0o600, dir_fd=directory,
    )
    try:
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("READY partial write made no progress")
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(
            partial, final, src_dir_fd=directory, dst_dir_fd=directory,
            follow_symlinks=False,
        )
        os.unlink(partial, dir_fd=directory)
        os.fsync(directory)
    except BaseException:
        try:
            os.unlink(partial, dir_fd=directory)
            os.fsync(directory)
        except FileNotFoundError:
            pass
        raise


def read_ready_once(namespace: StartupNamespace, role: Role) -> dict[str, Any]:
    fd = os.open(
        _FILES[_role(role)][0], os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=namespace._launch_fd,
    )
    try:
        observed = os.fstat(fd)
        if _ready_publication_in_progress(observed, _MAX):
            raise ReadyPublicationInProgress("READY publication is in progress")
        if not _safe_file(observed, _MAX):
            raise ValueError("READY file type, owner, links, mode, or size is invalid")
        chunks: list[bytes] = []
        remaining = _MAX + 1
        while remaining:
            chunk = os.read(fd, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        if len(encoded) > _MAX:
            raise ValueError("READY file size exceeds the allowed bound")
    finally:
        os.close(fd)

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in rows:
            if key in value:
                raise ValueError(f"READY JSON contains duplicate key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            encoded.decode(), object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"READY JSON contains non-finite constant: {item}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError("READY file is not valid UTF-8") from error
    if not isinstance(value, dict):
        raise ValueError("READY JSON must be an object")
    return value


def _base(
    *, artifact_kind: str, schema_version: int, launch_id: str,
    run_identity: str, attempt_hash: str, fencing_epoch: int,
    release_hash: str, performance_policy_hash: str | None, pid: int,
    process_start_token: str, ready_at_epoch: float,
) -> dict[str, Any]:
    return {
        "artifact_kind": artifact_kind,
        "schema_version": schema_version,
        "launch_id": _launch(launch_id),
        "run_identity": _hash(run_identity, "run identity"),
        "attempt_hash": _hash(attempt_hash, "attempt hash"),
        "fencing_epoch": _positive(fencing_epoch, "fencing epoch", integer=True),
        "release_hash": _hash(release_hash, "release hash"),
        "performance_policy_hash": _hash(
            performance_policy_hash, "performance policy hash", optional=True
        ),
        "process_pid": _positive(pid, "process PID", integer=True),
        "process_start_token": _hash(process_start_token, "process start token"),
        "ready_at_epoch": _positive(ready_at_epoch, "READY timestamp"),
        **_PRIVATE,
    }


def build_worker_ready(
    *, launch_id: str, run_identity: str, attempt_hash: str, fencing_epoch: int,
    release_hash: str, environment_hash: str, data_placement_hash: str,
    storage_layout_hash: str, performance_policy_hash: str | None, pid: int,
    process_start_token: str, ready_at_epoch: float,
    configuration_physical_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = _base(
        artifact_kind="twc_pretraining_worker_ready_v3", schema_version=3,
        launch_id=launch_id, run_identity=run_identity, attempt_hash=attempt_hash,
        fencing_epoch=fencing_epoch, release_hash=release_hash,
        performance_policy_hash=performance_policy_hash, pid=pid,
        process_start_token=process_start_token, ready_at_epoch=ready_at_epoch,
    )
    value.update(
        environment_hash=_hash(environment_hash, "environment hash"),
        data_placement_hash=_hash(data_placement_hash, "data placement hash"),
        storage_layout_hash=_hash(storage_layout_hash, "storage layout hash"),
    )
    if configuration_physical_evidence is not None:
        value["configuration_physical_evidence"] = dict(configuration_physical_evidence)
    value["receipt_hash"] = stable_json_hash(value)
    return value


def build_watchdog_ready(
    *, launch_id: str, run_identity: str, attempt_hash: str, fencing_epoch: int,
    release_hash: str, performance_policy_hash: str | None, pid: int,
    process_start_token: str, ready_at_epoch: float,
) -> dict[str, Any]:
    value = _base(
        artifact_kind="twc_pretraining_watchdog_ready_v1", schema_version=1,
        launch_id=launch_id, run_identity=run_identity, attempt_hash=attempt_hash,
        fencing_epoch=fencing_epoch, release_hash=release_hash,
        performance_policy_hash=performance_policy_hash, pid=pid,
        process_start_token=process_start_token, ready_at_epoch=ready_at_epoch,
    )
    value["receipt_hash"] = stable_json_hash(value)
    return value


def _load(payload: Mapping[str, Any], *, worker: bool) -> dict[str, Any]:
    common = {
        "artifact_kind", "schema_version", "launch_id", "run_identity",
        "attempt_hash", "fencing_epoch", "release_hash", "performance_policy_hash",
        "process_pid", "process_start_token", "ready_at_epoch", *_PRIVATE,
        "receipt_hash",
    }
    required = common | (
        {"environment_hash", "data_placement_hash", "storage_layout_hash"}
        if worker else set()
    )
    allowed = required | ({"configuration_physical_evidence"} if worker else set())
    if set(payload) not in (required, allowed):
        raise ValueError("READY receipt fields are invalid")
    value = dict(payload)
    if (
        value["artifact_kind"]
        != ("twc_pretraining_worker_ready_v3" if worker else "twc_pretraining_watchdog_ready_v1")
        or value["schema_version"] != (3 if worker else 1)
    ):
        raise ValueError("READY receipt schema is invalid")
    _launch(value["launch_id"])
    for field in ("run_identity", "attempt_hash", "release_hash", "process_start_token"):
        _hash(value[field], field)
    _hash(value["performance_policy_hash"], "performance policy hash", optional=True)
    if worker:
        for field in ("environment_hash", "data_placement_hash", "storage_layout_hash"):
            _hash(value[field], field)
        if "configuration_physical_evidence" in value and not isinstance(
            value["configuration_physical_evidence"], dict
        ):
            raise ValueError("worker READY physical evidence must be an object")
    _positive(value["fencing_epoch"], "fencing epoch", integer=True)
    _positive(value["process_pid"], "process PID", integer=True)
    _positive(value["ready_at_epoch"], "READY timestamp")
    if any(value[field] is not False for field in _PRIVATE):
        raise ValueError("READY privacy flags must be false")
    unsigned = {key: item for key, item in value.items() if key != "receipt_hash"}
    if value["receipt_hash"] != stable_json_hash(unsigned):
        raise ValueError("READY receipt hash mismatch")
    return value


def load_worker_ready(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _load(payload, worker=True)


def load_watchdog_ready(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _load(payload, worker=False)


def _activation(
    state: Mapping[str, Any], launch_id: str,
    worker: Mapping[str, Any], watchdog: Mapping[str, Any],
) -> dict[str, Any] | None:
    history = state.get("launch_history")
    if not isinstance(history, list):
        return None
    rows = [row for row in history if isinstance(row, dict) and row.get("launch_id") == launch_id]
    if len(rows) > 1:
        raise ValueError("launch history contains duplicate activation")
    if not rows:
        return None
    row = rows[0]
    fields = {
        "schema_version", "launch_id", "command_kind", "run_identity",
        "attempt_hash", "fencing_epoch", "activated_at_epoch",
        "worker_process_identity", "watchdog_process_identity",
        "worker_ready_receipt", "watchdog_ready_receipt", "activation_hash",
    }
    unsigned = {key: item for key, item in row.items() if key != "activation_hash"}
    expected = {
        "run_identity": worker["run_identity"],
        "attempt_hash": worker["attempt_hash"],
        "fencing_epoch": worker["fencing_epoch"],
        "worker_ready_receipt": worker,
        "watchdog_ready_receipt": watchdog,
    }
    if (
        set(row) != fields
        or row.get("schema_version") != 1
        or row.get("command_kind") not in {"start", "resume"}
        or row.get("activation_hash") != stable_json_hash(unsigned)
        or any(
        row.get(key) != item for key, item in expected.items()
        )
    ):
        raise ValueError("activation history binding mismatch")
    return row


def activation_state(
    state: Mapping[str, Any], *, role: Role, launch_id: str,
    receipt: Mapping[str, Any], pid: int, start_token: str,
) -> Literal["wait", "run", "quiesce", "terminal"]:
    role = _role(role)
    launch_id = _launch(launch_id)
    own = load_worker_ready(receipt) if role == "worker" else load_watchdog_ready(receipt)
    if any((
        own["launch_id"] != launch_id,
        own["process_pid"] != pid,
        own["process_start_token"] != start_token,
        state.get("run_identity") != own["run_identity"],
        state.get("attempt_hash") != own["attempt_hash"],
        state.get("fencing_epoch") != own["fencing_epoch"],
    )):
        raise ValueError("startup self or operator binding mismatch")
    lifecycle = state.get("state")
    if lifecycle == "STARTING":
        command = state.get("start_command")
        if not isinstance(command, dict) or command.get("command_id") != launch_id:
            raise ValueError("STARTING launch binding mismatch")
        return "wait"

    worker = state.get("worker_ready_receipt")
    watchdog = state.get("watchdog_ready_receipt")
    if lifecycle not in {"RUNNING", "QUIESCING"}:
        history = state.get("launch_history")
        match = next(
            (row for row in history or [] if isinstance(row, dict) and row.get("launch_id") == launch_id),
            None,
        )
        if match is not None:
            worker, watchdog = match.get("worker_ready_receipt"), match.get("watchdog_ready_receipt")
    if not isinstance(worker, dict) or not isinstance(watchdog, dict):
        raise ValueError("activation receipts are missing")
    worker, watchdog = load_worker_ready(worker), load_watchdog_ready(watchdog)
    if own != (worker if role == "worker" else watchdog):
        raise ValueError("activation role receipt mismatch")
    entry = _activation(state, launch_id, worker, watchdog)
    if entry is None:
        raise ValueError("state has no exact activation")
    if lifecycle in {"RUNNING", "QUIESCING"}:
        if state.get("active_launch_id") != launch_id:
            raise ValueError("active launch binding mismatch")
        identities = {
            "worker_process_identity": state.get("process_identity"),
            "watchdog_process_identity": state.get("watchdog_process_identity"),
        }
        if any(
            not isinstance(identity, dict) or identity != entry.get(key)
            for key, identity in identities.items()
        ):
            raise ValueError("active process pair does not match activation history")
        key = "process_identity" if role == "worker" else "watchdog_process_identity"
        entry_key = "worker_process_identity" if role == "worker" else "watchdog_process_identity"
        identity = state.get(key)
        if (
            not isinstance(identity, dict) or identity != entry.get(entry_key)
            or identity.get("pid") != pid
            or identity.get("process_start_token") != start_token
        ):
            raise ValueError("active process identity mismatch")
        return "run" if lifecycle == "RUNNING" else "quiesce"
    return "terminal"
