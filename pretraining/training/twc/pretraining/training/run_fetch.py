"""Immutable local copyback for one stopped or completed generic training run."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from typing import Any, Mapping

from twc.pretraining.training.checkpoints import CheckpointManager, CheckpointReference
from twc.pretraining.training.configuration import load_run_root
from twc.pretraining.training.contracts import load_execution_attempt, training_config_contract
from twc.repro.hashing import stable_json_hash
from twc.repro.public_artifacts import validate_public_no_raw_payload


_RECEIPT_NAME = "fetch-receipt.json"
_LOCK_NAME = ".fetch.lock"
_RECEIPT_FIELDS = {
    "artifact_kind",
    "schema_version",
    "destination_root",
    "configuration_hash",
    "run_identity",
    "attempt_hash",
    "state",
    "checkpoint_generation",
    "checkpoint_durability",
    "checkpoint_manifest_sha256",
    "checkpoint_state_sha256",
    "raw_text_serialized",
    "raw_url_serialized",
    "tokens_serialized",
    "secrets_serialized",
    "status",
    "files",
    "receipt_hash",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise ValueError(f"{label} is missing") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if int(metadata.st_nlink) != 1:
        raise ValueError(f"{label} must have exactly one link")
    if os.name == "posix" and int(metadata.st_uid) != int(os.geteuid()):
        raise ValueError(f"{label} must belong to the current user")
    return metadata


def _safe_destination(destination: Path) -> Path:
    requested = destination.expanduser().absolute()
    for ancestor in (requested, *requested.parents):
        if ancestor.exists() or ancestor.is_symlink():
            if ancestor.is_symlink():
                raise ValueError("fetch destination must not contain a symlink ancestor")
    requested.mkdir(parents=True, exist_ok=True, mode=0o700)
    if requested.is_symlink() or not requested.is_dir():
        raise ValueError("fetch destination must be a real directory")
    if os.name == "posix":
        os.chmod(requested, 0o700)
    return requested.resolve(strict=True)


def _safe_parent(root: Path, target: Path) -> None:
    relative = target.parent.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise ValueError("fetch destination parent is unsafe")
        else:
            current.mkdir(mode=0o700)
        if os.name == "posix":
            os.chmod(current, 0o700)


def _json_or_jsonl_is_no_raw(path: Path) -> None:
    if path.suffix == ".json":
        payloads = [json.loads(path.read_text(encoding="utf-8"))]
    elif path.suffix == ".jsonl":
        payloads = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    else:
        return
    for payload in payloads:
        validate_public_no_raw_payload(payload)


def _attempt_chain(path: Path) -> list[Any]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("fetch attempt ledger is invalid") from error
    if not rows:
        raise ValueError("fetch attempt ledger is empty")
    attempts = [load_execution_attempt(row) for row in rows]
    for previous, current in zip(attempts, attempts[1:]):
        current.validate_successor(previous)
    return attempts


def _source_files(
    run_root: Path,
    state: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Path]:
    attempt_hash = str(state["attempt_hash"])
    files = {
        "run.json": run_root / "run.json",
        "runtime/global-plan.json": run_root / "runtime" / "global-plan.json",
        "runtime/release.json": run_root / "runtime" / "release.json",
        "operator/configuration-binding.json": run_root / "operator" / "configuration-binding.json",
        "operator/run_contract.json": run_root / "operator" / "run_contract.json",
        "operator/execution.json": run_root / "operator" / "execution.json",
        "operator/attempts.jsonl": run_root / "operator" / "attempts.jsonl",
        "operator/state.json": run_root / "operator" / "state.json",
    }
    generation_name = checkpoint_path.name
    for name in ("READY.json", "manifest.json", "state.pt"):
        files[f"checkpoints/{generation_name}/{name}"] = checkpoint_path / name
    if state["state"] == "COMPLETED":
        files.update(
            {
                "result.json": run_root / "result.json",
                f"operator/attempt-outcomes/{attempt_hash}.json": (
                    run_root / "operator" / "attempt-outcomes" / f"{attempt_hash}.json"
                ),
                f"operator/worker-evaluations-{attempt_hash}.jsonl": (
                    run_root / "operator" / f"worker-evaluations-{attempt_hash}.jsonl"
                ),
            }
        )
    return files


def _validate_source(
    run_root: Path,
    state: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    try:
        source_state = json.loads(
            (run_root / "operator" / "state.json").read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError("fetch source state is invalid") from error
    if not isinstance(source_state, dict) or source_state != dict(state):
        raise ValueError("fetch status snapshot differs from source state")
    if state.get("state") not in {"STOPPED", "COMPLETED"}:
        raise ValueError("fetch requires a stopped or completed run")
    if state.get("checkpoint_durability") != "persistent_verified":
        raise ValueError("fetch requires a persistent verified checkpoint")
    generation = state.get("checkpoint_generation")
    if type(generation) is not int or generation < 0:
        raise ValueError("fetch state does not select an exact checkpoint")
    config = load_run_root(run_root)
    contract = training_config_contract(config)
    if contract.identity != state.get("run_identity"):
        raise ValueError("fetch state does not bind the training configuration")
    attempts = _attempt_chain(run_root / "operator" / "attempts.jsonl")
    attempt = attempts[-1]
    if attempt.identity != state.get("attempt_hash") or attempt.run_identity != contract.identity:
        raise ValueError("fetch state does not bind the latest immutable attempt")
    if attempt.checkpoint_root is None or attempt.checkpoint_topology is None:
        raise ValueError("fetch attempt has no checkpoint execution snapshot")
    checkpoint_root = Path(attempt.checkpoint_root).expanduser().resolve(strict=True)
    persistent_root = (
        Path(attempt.persistent_checkpoint_root).expanduser().resolve(strict=True)
        if attempt.persistent_checkpoint_root is not None
        else None
    )
    manager = CheckpointManager(
        checkpoint_root,
        persistent_root=persistent_root,
        topology=attempt.checkpoint_topology,
        create=False,
    )
    loaded = manager.validate_reference(
        CheckpointReference(
            generation,
            checkpoint_root / f"generation-{generation:08d}",
            "persistent_verified",
        )
    )
    if (
        loaded.run_identity != state["run_identity"]
        or loaded.attempt_hash != state["attempt_hash"]
        or loaded.step != state.get("step")
        or loaded.next_global_ordinal != state.get("next_global_ordinal")
    ):
        raise ValueError("fetch checkpoint does not match selected run progress")
    checkpoint_path = checkpoint_root / f"generation-{generation:08d}"
    files = _source_files(run_root, state, checkpoint_path)
    for path in files.values():
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(run_root) and not resolved.is_relative_to(checkpoint_root):
            raise ValueError("fetch source escapes its managed roots")
        _regular_file(path, "fetch source")
        _json_or_jsonl_is_no_raw(path)
    receipt_base = {
        "artifact_kind": "twc_pretraining_fetch_receipt_v1",
        "schema_version": 1,
        "destination_root": None,
        "configuration_hash": config["configuration_hash"],
        "run_identity": state["run_identity"],
        "attempt_hash": state["attempt_hash"],
        "state": state["state"],
        "checkpoint_generation": generation,
        "checkpoint_durability": "persistent_verified",
        "checkpoint_manifest_sha256": loaded.manifest_sha256,
        "checkpoint_state_sha256": loaded.state_sha256,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "tokens_serialized": False,
        "secrets_serialized": False,
    }
    return files, receipt_base


def _copy_one(root: Path, source: Path, target: Path, expected_hash: str) -> None:
    _safe_parent(root, target)
    if target.is_symlink():
        raise ValueError("fetch destination file must not be a symlink")
    if target.exists():
        _regular_file(target, "fetch destination")
        if _sha256(target) != expected_hash:
            raise ValueError("fetch destination conflicts with immutable source")
        return
    partial = target.with_name(target.name + ".partial")
    if partial.exists() or partial.is_symlink():
        metadata = _regular_file(partial, "fetch partial")
        if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("fetch partial has unsafe permissions")
        partial.unlink()
    before = _regular_file(source, "fetch source")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(partial, flags, 0o600)
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb", closefd=False) as outgoing:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    finally:
        os.close(descriptor)
    after = _regular_file(source, "fetch source")
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or _sha256(partial) != expected_hash:
        partial.unlink(missing_ok=True)
        raise ValueError("fetch source changed during copy")
    os.replace(partial, target)
    _fsync_directory(target.parent)


def fetch_run(run: Path, destination: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    run_root = run.expanduser().resolve(strict=True)
    files, receipt_base = _validate_source(run_root, state)
    root = _safe_destination(destination)
    if root == run_root or root.is_relative_to(run_root):
        raise ValueError("fetch destination must be outside the source run")
    lock_path = root / _LOCK_NAME
    lock_flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        lock_flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, lock_flags, 0o600)
    try:
        lock_metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(lock_metadata.st_mode)
            or int(lock_metadata.st_nlink) != 1
            or (
                os.name == "posix"
                and int(lock_metadata.st_uid) != int(os.geteuid())
            )
        ):
            raise ValueError("fetch destination lock is unsafe")
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        allowed = {_LOCK_NAME, _RECEIPT_NAME, f"{_RECEIPT_NAME}.partial"}
        allowed.update(files)
        allowed.update(f"{relative}.partial" for relative in files)
        existing = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        unexpected = sorted(existing - allowed)
        if unexpected:
            raise ValueError(f"fetch destination contains unrelated files: {unexpected}")
        identities = {
            relative: {
                "relative_path": relative,
                "sha256": _sha256(source),
                "size_bytes": int(source.stat().st_size),
            }
            for relative, source in files.items()
        }
        required = sum(item["size_bytes"] for item in identities.values())
        if shutil.disk_usage(root).free < required:
            raise OSError("fetch destination has insufficient free space")
        for relative, source in sorted(files.items()):
            target = root / PurePosixPath(relative)
            target.resolve(strict=False).relative_to(root)
            _copy_one(root, source, target, identities[relative]["sha256"])
        # A terminal source must remain byte-identical until destination validation finishes.
        for relative, source in files.items():
            if _sha256(source) != identities[relative]["sha256"]:
                raise ValueError("fetch source changed during copy")
        receipt = {
            **receipt_base,
            "destination_root": str(root),
            "status": "complete",
            "files": [identities[key] for key in sorted(identities)],
        }
        receipt["receipt_hash"] = stable_json_hash(receipt)
        validate_public_no_raw_payload(receipt)
        receipt_path = root / _RECEIPT_NAME
        if receipt_path.exists() or receipt_path.is_symlink():
            _regular_file(receipt_path, "fetch receipt")
            existing_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if existing_receipt != receipt:
                raise ValueError("fetch receipt conflicts with immutable retry")
        else:
            _copy_receipt(receipt_path, receipt)
        return load_fetch_receipt(root)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _copy_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or partial.is_symlink():
        metadata = _regular_file(partial, "fetch receipt partial")
        if os.name == "posix" and stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("fetch receipt partial has unsafe permissions")
        partial.unlink()
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        encoded = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("fetch receipt write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(partial, path)
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def load_fetch_receipt(run: Path) -> dict[str, Any]:
    root = run.expanduser().resolve(strict=True)
    receipt_path = root / _RECEIPT_NAME
    _regular_file(receipt_path, "fetch receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if (
        set(receipt) != _RECEIPT_FIELDS
        or receipt.get("artifact_kind") != "twc_pretraining_fetch_receipt_v1"
        or receipt.get("schema_version") != 1
        or receipt.get("status") != "complete"
        or receipt.get("destination_root") != str(root)
        or receipt.get("receipt_hash") != stable_json_hash(expected)
        or any(
            receipt.get(field) is not False
            for field in (
                "raw_text_serialized",
                "raw_url_serialized",
                "tokens_serialized",
                "secrets_serialized",
            )
        )
    ):
        raise ValueError("fetch receipt is invalid")
    validate_public_no_raw_payload(receipt)
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("fetch receipt file inventory is invalid")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"relative_path", "sha256", "size_bytes"}:
            raise ValueError("fetch receipt file entry is invalid")
        relative = str(item["relative_path"])
        path = PurePosixPath(relative)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or relative in seen:
            raise ValueError("fetch receipt relative path is invalid")
        seen.add(relative)
        target = root / path
        _regular_file(target, "fetched artifact")
        if target.stat().st_size != item["size_bytes"] or _sha256(target) != item["sha256"]:
            raise ValueError("fetched artifact identity mismatch")
        _json_or_jsonl_is_no_raw(target)
    state = json.loads((root / "operator" / "state.json").read_text(encoding="utf-8"))
    if state.get("state_hash") != stable_json_hash(
        {key: value for key, value in state.items() if key != "state_hash"}
    ):
        raise ValueError("fetched operator state hash mismatch")
    generation = receipt.get("checkpoint_generation")
    if type(generation) is not int or generation < 0:
        raise ValueError("fetch receipt checkpoint generation is invalid")
    expected_paths = {
        "run.json",
        "runtime/global-plan.json",
        "runtime/release.json",
        "operator/configuration-binding.json",
        "operator/run_contract.json",
        "operator/execution.json",
        "operator/attempts.jsonl",
        "operator/state.json",
        f"checkpoints/generation-{generation:08d}/READY.json",
        f"checkpoints/generation-{generation:08d}/manifest.json",
        f"checkpoints/generation-{generation:08d}/state.pt",
    }
    if receipt.get("state") == "COMPLETED":
        attempt_hash = receipt.get("attempt_hash")
        expected_paths.update(
            {
                "result.json",
                f"operator/attempt-outcomes/{attempt_hash}.json",
                f"operator/worker-evaluations-{attempt_hash}.jsonl",
            }
        )
    if seen != expected_paths:
        raise ValueError("fetch receipt does not contain the exact artifact set")
    config = load_run_root(root)
    contract = training_config_contract(config)
    attempts = _attempt_chain(root / "operator" / "attempts.jsonl")
    attempt = attempts[-1]
    if (
        receipt.get("configuration_hash") != config.get("configuration_hash")
        or receipt.get("run_identity") != contract.identity
        or attempt.identity != receipt.get("attempt_hash")
        or attempt.run_identity != receipt.get("run_identity")
        or state.get("run_identity") != receipt.get("run_identity")
        or state.get("attempt_hash") != receipt.get("attempt_hash")
        or state.get("checkpoint_generation") != receipt.get("checkpoint_generation")
        or state.get("checkpoint_durability") != "persistent_verified"
        or state.get("state") != receipt.get("state")
    ):
        raise ValueError("fetch receipt does not bind copied operator state")
    loaded = CheckpointManager(
        root / "checkpoints", topology="direct_persistent", create=False
    ).validate_reference(
        CheckpointReference(
            generation,
            root / "checkpoints" / f"generation-{generation:08d}",
            "persistent_verified",
            str(receipt["checkpoint_manifest_sha256"]),
            str(receipt["checkpoint_state_sha256"]),
        )
    )
    if (
        loaded.run_identity != receipt["run_identity"]
        or loaded.attempt_hash != receipt["attempt_hash"]
        or loaded.step != state.get("step")
        or loaded.next_global_ordinal != state.get("next_global_ordinal")
    ):
        raise ValueError("fetched checkpoint does not bind copied run progress")
    return receipt
