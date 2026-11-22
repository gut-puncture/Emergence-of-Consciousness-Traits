"""Measured storage-role validation for constrained and removable machines."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from twc.repro.hashing import stable_json_hash


@dataclass(frozen=True, slots=True)
class StorageRole:
    root: Path
    require_mount: bool
    sentinel: str | None = None


@dataclass(frozen=True, slots=True)
class StorageLayout:
    control: StorageRole
    data: StorageRole
    durable_checkpoints: StorageRole | None
    local_checkpoints: StorageRole | None
    copyback: StorageRole | None

    def validate(
        self, *, required_checkpoint_bytes: int, reserve_bytes: int = 1024**3
    ) -> dict[str, Any]:
        if type(required_checkpoint_bytes) is not int or required_checkpoint_bytes < 0:
            raise ValueError("required checkpoint bytes must be a nonnegative integer")
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("storage reserve bytes must be a nonnegative integer")
        named = {"control": self.control, "data": self.data}
        if self.durable_checkpoints is not None:
            named["durable_checkpoints"] = self.durable_checkpoints
        if self.local_checkpoints is not None:
            named["local_checkpoints"] = self.local_checkpoints
        if self.copyback is not None:
            named["copyback"] = self.copyback

        requirements = {
            "control": reserve_bytes,
            "data": 0,
            "durable_checkpoints": 3 * required_checkpoint_bytes + reserve_bytes,
            "local_checkpoints": 3 * required_checkpoint_bytes + reserve_bytes,
            "copyback": 2 * required_checkpoint_bytes + reserve_bytes,
        }
        roles: list[dict[str, Any]] = []
        filesystems: dict[int, dict[str, Any]] = {}
        for name, role in named.items():
            row = _validate_role(name, role)
            roles.append(row)
            device = int(row["device_id"])
            fs = filesystems.setdefault(
                device,
                {
                    "device_id": device,
                    "roles": [],
                    "required_bytes": 0,
                    "free_bytes": int(row["free_bytes"]),
                    "free_inodes": int(row["free_inodes"]),
                },
            )
            fs["roles"].append(name)
            fs["required_bytes"] += requirements[name]
            fs["free_bytes"] = min(fs["free_bytes"], int(row["free_bytes"]))
            fs["free_inodes"] = min(fs["free_inodes"], int(row["free_inodes"]))
        for fs in filesystems.values():
            if fs["free_bytes"] < fs["required_bytes"]:
                raise ValueError(
                    f"storage filesystem for {sorted(fs['roles'])} lacks required capacity"
                )
            if fs["free_inodes"] < 32:
                raise ValueError(
                    f"storage filesystem for {sorted(fs['roles'])} lacks inode reserve"
                )
            fs["roles"].sort()
        payload = {
            "artifact_kind": "twc_storage_layout_receipt_v1",
            "schema_version": 1,
            "roles": sorted(roles, key=lambda row: str(row["role"])),
            "filesystems": sorted(filesystems.values(), key=lambda row: int(row["device_id"])),
            "required_checkpoint_bytes": required_checkpoint_bytes,
            "reserve_bytes": reserve_bytes,
        }
        payload["receipt_hash"] = stable_json_hash(payload)
        return payload


_ROLE_FIELDS = {
    "role",
    "resolved_root",
    "device_id",
    "sentinel",
    "sentinel_hash",
    "free_bytes",
    "free_inodes",
    "mount_required",
    "probe",
}
_CHECKPOINT_ROLES = frozenset({"local_checkpoints", "durable_checkpoints"})


def checkpoint_filesystem_requirements(
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate a canonical layout receipt and return checkpoint filesystems."""

    expected_fields = {
        "artifact_kind",
        "schema_version",
        "roles",
        "filesystems",
        "required_checkpoint_bytes",
        "reserve_bytes",
        "receipt_hash",
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("artifact_kind") != "twc_storage_layout_receipt_v1"
        or receipt.get("schema_version") != 1
        or receipt.get("receipt_hash")
        != stable_json_hash(
            {key: value for key, value in receipt.items() if key != "receipt_hash"}
        )
    ):
        raise ValueError("storage layout receipt is not canonical")
    required_checkpoint_bytes = receipt.get("required_checkpoint_bytes")
    reserve_bytes = receipt.get("reserve_bytes")
    if (
        type(required_checkpoint_bytes) is not int
        or required_checkpoint_bytes < 0
        or type(reserve_bytes) is not int
        or reserve_bytes < 0
    ):
        raise ValueError("storage layout byte requirements are invalid")
    roles = receipt.get("roles")
    filesystems = receipt.get("filesystems")
    if not isinstance(roles, list) or not isinstance(filesystems, list):
        raise ValueError("storage layout role or filesystem rows are invalid")

    role_rows: dict[str, Mapping[str, Any]] = {}
    allowed_roles = {
        "control",
        "data",
        "durable_checkpoints",
        "local_checkpoints",
        "copyback",
    }
    for row in roles:
        if not isinstance(row, Mapping) or set(row) != _ROLE_FIELDS:
            raise ValueError("storage layout role row is not canonical")
        role = row.get("role")
        if role not in allowed_roles or role in role_rows:
            raise ValueError("storage layout role name is invalid or duplicated")
        for field in ("device_id", "free_bytes", "free_inodes"):
            if type(row.get(field)) is not int or int(row[field]) < 0:
                raise ValueError(f"storage layout role {field} is invalid")
        if (
            not isinstance(row.get("resolved_root"), str)
            or not Path(str(row["resolved_root"])).is_absolute()
            or type(row.get("mount_required")) is not bool
            or row.get("probe") != "write_fsync_rename_dir_fsync_readback"
        ):
            raise ValueError("storage layout role binding is invalid")
        sentinel = row.get("sentinel")
        sentinel_hash = row.get("sentinel_hash")
        if (sentinel is None) != (sentinel_hash is None) or (
            sentinel is not None
            and (
                not isinstance(sentinel, str)
                or not sentinel
                or not isinstance(sentinel_hash, str)
                or len(sentinel_hash) != 64
                or any(character not in "0123456789abcdef" for character in sentinel_hash)
            )
        ):
            raise ValueError("storage layout sentinel binding is invalid")
        role_rows[str(role)] = row
    if "control" not in role_rows or "data" not in role_rows:
        raise ValueError("storage layout is missing control or data role")

    role_requirement = {
        "control": reserve_bytes,
        "data": 0,
        "durable_checkpoints": 3 * required_checkpoint_bytes + reserve_bytes,
        "local_checkpoints": 3 * required_checkpoint_bytes + reserve_bytes,
        "copyback": 2 * required_checkpoint_bytes + reserve_bytes,
    }
    expected_by_device: dict[int, dict[str, Any]] = {}
    for role, row in role_rows.items():
        device = int(row["device_id"])
        aggregate = expected_by_device.setdefault(
            device,
            {
                "device_id": device,
                "roles": [],
                "required_bytes": 0,
                "free_bytes": int(row["free_bytes"]),
                "free_inodes": int(row["free_inodes"]),
            },
        )
        aggregate["roles"].append(role)
        aggregate["required_bytes"] += role_requirement[role]
        aggregate["free_bytes"] = min(
            aggregate["free_bytes"], int(row["free_bytes"])
        )
        aggregate["free_inodes"] = min(
            aggregate["free_inodes"], int(row["free_inodes"])
        )
    expected_rows = []
    for row in expected_by_device.values():
        row["roles"].sort()
        expected_rows.append(row)
    expected_rows.sort(key=lambda row: int(row["device_id"]))
    if filesystems != expected_rows:
        raise ValueError("storage layout filesystem aggregation is invalid")
    if any(
        int(row["free_bytes"]) < int(row["required_bytes"])
        or int(row["free_inodes"]) < 32
        for row in expected_rows
    ):
        raise ValueError("storage layout receipt did not pass initial capacity checks")

    checkpoint_rows = tuple(
        dict(row)
        for row in expected_rows
        if set(row["roles"]) & _CHECKPOINT_ROLES
    )
    if not checkpoint_rows:
        raise ValueError("storage layout has no checkpoint-writing filesystem")
    return checkpoint_rows


def checkpoint_filesystem_headroom(
    receipt: Mapping[str, Any], observations: object
) -> tuple[dict[str, Any], ...]:
    """Bind live free-space telemetry to every checkpoint-writing filesystem."""

    requirements = checkpoint_filesystem_requirements(receipt)
    expected_devices = {int(row["device_id"]) for row in requirements}
    observed: dict[int, int] = {}
    if not isinstance(observations, (list, tuple)):
        raise ValueError("checkpoint filesystem observations must be a sequence")
    for row in observations:
        if not isinstance(row, Mapping) or set(row) != {"device_id", "free_bytes"}:
            raise ValueError("checkpoint filesystem observation is not canonical")
        device = row.get("device_id")
        free_bytes = row.get("free_bytes")
        if (
            type(device) is not int
            or device in observed
            or type(free_bytes) is not int
            or free_bytes < 0
        ):
            raise ValueError("checkpoint filesystem observation values are invalid")
        observed[device] = free_bytes
    if set(observed) != expected_devices:
        raise ValueError("checkpoint filesystem observations are incomplete or unexpected")
    return tuple(
        {
            "device_id": int(requirement["device_id"]),
            "roles": list(requirement["roles"]),
            "required_bytes": int(requirement["required_bytes"]),
            "free_bytes": observed[int(requirement["device_id"])],
            "has_headroom": observed[int(requirement["device_id"])]
            >= int(requirement["required_bytes"]),
        }
        for requirement in requirements
    )


def _validate_role(name: str, role: StorageRole) -> dict[str, Any]:
    root = role.root.expanduser()
    if not root.exists():
        raise ValueError(f"declared storage root does not exist: {root}")
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"declared storage root is not a real directory: {root}")
    resolved = root.resolve(strict=True)
    if role.require_mount and not os.path.ismount(resolved):
        raise ValueError(f"declared storage root is not a mounted filesystem: {resolved}")
    sentinel_path = resolved / ".twc-storage.json"
    sentinel_payload: dict[str, Any] | None = None
    if role.sentinel is not None:
        try:
            sentinel_payload = json.loads(sentinel_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError(f"storage sentinel is missing or malformed for {name}") from error
        if sentinel_payload.get("storage_id") != role.sentinel:
            raise ValueError(f"storage sentinel identity mismatch for {name}")
        declared_role = sentinel_payload.get("role")
        if declared_role is not None and declared_role != name:
            raise ValueError(f"storage sentinel role mismatch for {name}")

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".twc-probe-", dir=resolved, delete=False
        ) as handle:
            probe = Path(handle.name)
            handle.write(b"twc-storage-probe")
            handle.flush()
            os.fsync(handle.fileno())
        renamed = probe.with_suffix(".renamed")
        os.replace(probe, renamed)
        with renamed.open("rb") as handle:
            if handle.read() != b"twc-storage-probe":
                raise OSError("storage probe readback mismatch")
        renamed.unlink()
        descriptor = os.open(resolved, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        try:
            probe.unlink(missing_ok=True)
        except UnboundLocalError:
            pass
        raise ValueError(f"storage root does not satisfy write/fsync/rename/readback: {name}") from error

    stat = resolved.stat()
    vfs = os.statvfs(resolved)
    return {
        "role": name,
        "resolved_root": str(resolved),
        "device_id": stat.st_dev,
        "sentinel": role.sentinel,
        "sentinel_hash": stable_json_hash(sentinel_payload) if sentinel_payload else None,
        "free_bytes": vfs.f_bavail * vfs.f_frsize,
        "free_inodes": vfs.f_favail,
        "mount_required": role.require_mount,
        "probe": "write_fsync_rename_dir_fsync_readback",
    }
