"""Small immutable code releases and offline-bootstrap verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import platform
import sys
from typing import Any, Mapping

from twc.repro.hashing import stable_json_hash


def build_environment_receipt(
    *,
    device: str,
    precision: str,
    attention_backend: str,
    optimizer_backend: str,
    compile_mode: str,
    allow_tf32: bool,
) -> dict[str, Any]:
    import torch

    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("environment device must be cpu or cuda")
    cuda_requested = device.startswith("cuda")
    cuda_available = bool(torch.cuda.is_available())
    status = "ready" if not cuda_requested or cuda_available else "unavailable"
    payload: dict[str, Any] = {
        "artifact_kind": "twc_execution_environment_receipt_v1",
        "schema_version": 1,
        "status": status,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": device,
        "precision": precision,
        "attention_backend": attention_backend,
        "optimizer_backend": optimizer_backend,
        "compile_mode": compile_mode,
        "allow_tf32": allow_tf32,
    }
    payload["environment_hash"] = stable_json_hash(payload)
    return payload


def validate_environment_receipt(
    payload: Mapping[str, Any], *, require_cuda_ready: bool = False
) -> dict[str, Any]:
    """Validate copied environment evidence without probing the original host."""

    expected_fields = {
        "artifact_kind", "schema_version", "status", "python_version",
        "python_executable", "platform", "torch_version", "cuda_runtime_version",
        "cuda_available", "device", "precision", "attention_backend",
        "optimizer_backend", "compile_mode", "allow_tf32", "environment_hash",
    }
    if (
        set(payload) != expected_fields
        or payload.get("artifact_kind") != "twc_execution_environment_receipt_v1"
        or payload.get("schema_version") != 1
    ):
        raise ValueError("environment receipt schema fields are invalid")
    expected_hash = stable_json_hash(
        {key: value for key, value in payload.items() if key != "environment_hash"}
    )
    if payload.get("environment_hash") != expected_hash:
        raise ValueError("environment receipt hash mismatch")
    for field in (
        "python_version", "python_executable", "platform", "torch_version",
        "device", "precision", "attention_backend", "optimizer_backend",
        "compile_mode",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not value or any(char in value for char in "\r\n\0"):
            raise ValueError(f"environment receipt {field} is invalid")
    if not Path(str(payload["python_executable"])).is_absolute():
        raise ValueError("environment receipt Python executable must be absolute")
    cuda_runtime = payload.get("cuda_runtime_version")
    if cuda_runtime is not None and (
        not isinstance(cuda_runtime, str)
        or not cuda_runtime
        or any(char in cuda_runtime for char in "\r\n\0")
    ):
        raise ValueError("environment CUDA runtime version is invalid")
    if type(payload.get("cuda_available")) is not bool or type(payload.get("allow_tf32")) is not bool:
        raise ValueError("environment capability flags must be booleans")
    device = str(payload["device"])
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("environment device must be CPU or CUDA")
    expected_status = (
        "ready" if device == "cpu" or payload["cuda_available"] is True else "unavailable"
    )
    if payload.get("status") != expected_status:
        raise ValueError("environment status contradicts the captured device capability")
    if device.startswith("cuda") and payload["cuda_available"] is True and cuda_runtime is None:
        raise ValueError("ready CUDA environment requires a CUDA runtime version")
    if require_cuda_ready and (
        not device.startswith("cuda")
        or payload["cuda_available"] is not True
        or payload.get("status") != "ready"
    ):
        raise ValueError("environment receipt does not prove a ready CUDA path")
    return dict(payload)


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("release source must not contain symlinks")
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    return rows


def build_release(source_root: Path, release_root: Path) -> dict[str, Any]:
    source = source_root.resolve(strict=True)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("release source must be a real directory")
    target_root = release_root.expanduser().resolve()
    if target_root == source or source in target_root.parents or target_root in source.parents:
        raise ValueError("release root and source root must be non-nested")
    inventory = _inventory(source)
    release_hash = stable_json_hash(inventory)
    target_root.mkdir(parents=True, exist_ok=True)
    final = target_root / f"release-{release_hash}"
    partial = target_root / f"release-{release_hash}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    if not final.exists():
        shutil.copytree(source, partial)
        if _inventory(partial) != inventory:
            shutil.rmtree(partial)
            raise ValueError("release copy readback mismatch")
        os.replace(partial, final)
    elif _inventory(final) != inventory:
        raise ValueError("existing release content does not match its identity")
    payload = {
        "artifact_kind": "twc_code_release_receipt_v1",
        "schema_version": 1,
        "release_hash": release_hash,
        "code_hash": release_hash,
        "path": str(final.resolve()),
        "file_count": len(inventory),
        "bytes": sum(int(row["bytes"]) for row in inventory),
        "inventory": inventory,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload


def verify_release_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    expected = stable_json_hash({key: value for key, value in receipt.items() if key != "receipt_hash"})
    if receipt.get("artifact_kind") != "twc_code_release_receipt_v1" or receipt.get("receipt_hash") != expected:
        raise ValueError("release receipt hash or schema mismatch")
    root = Path(str(receipt.get("path"))).resolve(strict=True)
    inventory = _inventory(root)
    if inventory != receipt.get("inventory") or stable_json_hash(inventory) != receipt.get("release_hash"):
        raise ValueError("live release inventory/content mismatch")
    if receipt.get("code_hash") != receipt.get("release_hash"):
        raise ValueError("release code hash does not equal verified content identity")
    return {"path": str(root), "release_hash": receipt["release_hash"], "verified": True}


def verify_offline_bootstrap(
    *,
    release_receipt: dict[str, Any],
    python_executable: Path,
    expected_python: tuple[int, int],
) -> dict[str, Any]:
    expected = stable_json_hash(
        {key: value for key, value in release_receipt.items() if key != "receipt_hash"}
    )
    if release_receipt.get("receipt_hash") != expected:
        raise ValueError("release receipt hash mismatch")
    root = Path(str(release_receipt["path"])).resolve(strict=True)
    inventory = _inventory(root)
    if stable_json_hash(inventory) != release_receipt.get("release_hash"):
        raise ValueError("release content hash mismatch")
    executable = python_executable.resolve(strict=True)
    probe = subprocess.run(
        [str(executable), "-I", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", "")},
    )
    observed = tuple(int(value) for value in probe.stdout.strip().split("."))
    if observed != expected_python:
        raise ValueError("bootstrap Python version mismatch")
    payload = {
        "artifact_kind": "twc_offline_bootstrap_receipt_v1",
        "schema_version": 1,
        "release_hash": release_receipt["release_hash"],
        "python_executable": str(executable),
        "python_version": list(observed),
        "offline_ready": True,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload
