"""Typed, no-raw destination-readback evidence for shape-probe copyback."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import PurePosixPath
import re
from typing import Any, Mapping, Sequence

from twc.repro.hashing import stable_json_hash


_HASH = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase identifier")
    return value


def _role_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "artifact_kind",
        "schema_version",
        "files",
        "file_count",
        "bytes",
        "raw_text_serialized",
        "secrets_serialized",
        "manifest_hash",
    }
    unsigned = {key: value for key, value in payload.items() if key != "manifest_hash"}
    rows = payload.get("files")
    if (
        set(payload) != expected
        or payload.get("artifact_kind") != "twc_gpu_copyback_manifest_v1"
        or payload.get("schema_version") != 1
        or payload.get("manifest_hash") != stable_json_hash(unsigned)
        or payload.get("raw_text_serialized") is not False
        or payload.get("secrets_serialized") is not False
        or not isinstance(rows, list)
        or len(rows) != 4
    ):
        raise ValueError("shape copyback embedded manifest is invalid")
    paths: list[str] = []
    byte_total = 0
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "bytes", "sha256"}
            or type(row.get("bytes")) is not int
            or int(row["bytes"]) < 0
        ):
            raise ValueError("shape copyback manifest file row is invalid")
        paths.append(_role_path(row["path"], f"manifest files[{index}].path"))
        byte_total += int(row["bytes"])
        _require_hash(row["sha256"], f"manifest files[{index}].sha256")
    if (
        paths != sorted(set(paths))
        or payload.get("file_count") != len(rows)
        or payload.get("bytes") != byte_total
    ):
        raise ValueError("shape copyback manifest accounting is invalid")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class ConfigurationShapeCopybackReadbackV1:
    """Portable proof that one exact destination was independently read back."""

    campaign_hash: str
    campaign_wave_id: str
    campaign_attempt_id: str
    destination_root_digest: str
    destination_device_id: int
    manifest_hash: str
    file_count: int
    bytes: int
    bundle_path: str
    result_paths: tuple[str, ...]
    bundle_hash: str
    result_hashes: tuple[str, ...]
    artifact_hashes: tuple[str, ...]
    _manifest_json: str = field(repr=False, compare=True)
    _destination_file_readbacks_json: str = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        for name in (
            "campaign_hash",
            "destination_root_digest",
            "manifest_hash",
            "bundle_hash",
        ):
            _require_hash(getattr(self, name), name)
        _require_identifier(self.campaign_wave_id, "campaign_wave_id")
        _require_identifier(self.campaign_attempt_id, "campaign_attempt_id")
        if type(self.destination_device_id) is not int or self.destination_device_id < 0:
            raise ValueError("shape copyback destination device identity is invalid")
        if self.file_count != 4 or type(self.bytes) is not int or self.bytes <= 0:
            raise ValueError("shape copyback requires four nonempty destination files")
        bundle_path = _role_path(self.bundle_path, "bundle_path")
        if len(self.result_paths) != 3 or len(set(self.result_paths)) != 3:
            raise ValueError("shape copyback requires three unique result role paths")
        result_paths = tuple(
            _role_path(value, f"result_paths[{index}]")
            for index, value in enumerate(self.result_paths)
        )
        if bundle_path in result_paths:
            raise ValueError("shape copyback bundle and result role paths overlap")
        if len(self.result_hashes) != 3 or len(set(self.result_hashes)) != 3:
            raise ValueError("shape copyback requires three unique result identities")
        for index, value in enumerate(self.result_hashes):
            _require_hash(value, f"result_hashes[{index}]")
        if (
            len(self.artifact_hashes) != len(set(self.artifact_hashes))
            or tuple(sorted(self.artifact_hashes)) != self.artifact_hashes
        ):
            raise ValueError("shape copyback artifact identities must be unique and sorted")
        for index, value in enumerate(self.artifact_hashes):
            _require_hash(value, f"artifact_hashes[{index}]")
        if not {self.bundle_hash, *self.result_hashes}.issubset(
            set(self.artifact_hashes)
        ):
            raise ValueError("shape copyback artifact lineage is incomplete")

        try:
            manifest_payload = json.loads(self._manifest_json)
            readback_rows = json.loads(self._destination_file_readbacks_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("shape copyback embedded readback evidence is invalid") from error
        if not isinstance(manifest_payload, Mapping):
            raise ValueError("shape copyback embedded manifest is invalid")
        manifest = _validate_manifest(manifest_payload)
        if (
            manifest["manifest_hash"] != self.manifest_hash
            or manifest["file_count"] != self.file_count
            or manifest["bytes"] != self.bytes
        ):
            raise ValueError("shape copyback manifest identity or accounting mismatch")
        expected_paths = tuple(sorted((bundle_path, *result_paths)))
        manifest_rows = manifest["files"]
        if tuple(row["path"] for row in manifest_rows) != expected_paths:
            raise ValueError("shape copyback manifest does not bind the exact role paths")
        if not isinstance(readback_rows, list) or len(readback_rows) != 4:
            raise ValueError("shape copyback destination readback rows are incomplete")
        readback_fields = {
            "path",
            "bytes",
            "sha256",
            "device_id",
            "inode",
            "mtime_ns",
        }
        for manifest_row, readback in zip(manifest_rows, readback_rows, strict=True):
            if (
                not isinstance(readback, Mapping)
                or set(readback) != readback_fields
                or {
                    key: readback[key] for key in ("path", "bytes", "sha256")
                }
                != manifest_row
                or type(readback.get("device_id")) is not int
                or readback["device_id"] != self.destination_device_id
                or type(readback.get("inode")) is not int
                or readback["inode"] < 0
                or type(readback.get("mtime_ns")) is not int
                or readback["mtime_ns"] <= 0
            ):
                raise ValueError("shape copyback destination file identity is invalid")

    @property
    def identity(self) -> str:
        return stable_json_hash(self._payload())

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_json)

    @property
    def destination_file_readbacks(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self._destination_file_readbacks_json))

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_configuration_shape_copyback_readback_v1",
            "schema_version": 1,
            "verified": True,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "tokens_serialized": False,
            "secrets_serialized": False,
            **{
                key: value
                for key, value in asdict(self).items()
                if not key.startswith("_")
            },
            "manifest": self.manifest,
            "destination_file_readbacks": list(self.destination_file_readbacks),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_hash"] = stable_json_hash(payload)
        return payload


def build_configuration_shape_copyback_readback(
    *,
    campaign_hash: str,
    campaign_wave_id: str,
    campaign_attempt_id: str,
    destination_root_digest: str,
    destination_device_id: int,
    manifest: Mapping[str, Any],
    bundle_path: str,
    result_paths: Sequence[str],
    bundle_hash: str,
    result_hashes: Sequence[str],
    artifact_hashes: Sequence[str],
    destination_file_readbacks: Sequence[Mapping[str, Any]],
) -> ConfigurationShapeCopybackReadbackV1:
    validated_manifest = _validate_manifest(manifest)
    return ConfigurationShapeCopybackReadbackV1(
        campaign_hash=campaign_hash,
        campaign_wave_id=campaign_wave_id,
        campaign_attempt_id=campaign_attempt_id,
        destination_root_digest=destination_root_digest,
        destination_device_id=destination_device_id,
        manifest_hash=str(validated_manifest["manifest_hash"]),
        file_count=int(validated_manifest["file_count"]),
        bytes=int(validated_manifest["bytes"]),
        bundle_path=bundle_path,
        result_paths=tuple(result_paths),
        bundle_hash=bundle_hash,
        result_hashes=tuple(result_hashes),
        artifact_hashes=tuple(artifact_hashes),
        _manifest_json=_canonical_json(validated_manifest),
        _destination_file_readbacks_json=_canonical_json(
            list(destination_file_readbacks)
        ),
    )


def load_configuration_shape_copyback_readback(
    payload: Mapping[str, Any],
) -> ConfigurationShapeCopybackReadbackV1:
    public_fields = {
        name
        for name in ConfigurationShapeCopybackReadbackV1.__dataclass_fields__
        if not name.startswith("_")
    }
    embedded = {"manifest", "destination_file_readbacks"}
    metadata = {
        "artifact_kind",
        "schema_version",
        "verified",
        "raw_text_serialized",
        "raw_url_serialized",
        "tokens_serialized",
        "secrets_serialized",
        "receipt_hash",
    }
    if set(payload) != public_fields | embedded | metadata:
        raise ValueError("shape copyback readback receipt fields are invalid")
    if (
        payload.get("artifact_kind")
        != "twc_configuration_shape_copyback_readback_v1"
        or payload.get("schema_version") != 1
        or payload.get("verified") is not True
        or any(
            payload.get(name) is not False
            for name in (
                "raw_text_serialized",
                "raw_url_serialized",
                "tokens_serialized",
                "secrets_serialized",
            )
        )
        or payload.get("receipt_hash")
        != stable_json_hash(
            {key: value for key, value in payload.items() if key != "receipt_hash"}
        )
    ):
        raise ValueError("shape copyback readback receipt boundary is invalid")
    values = {name: payload[name] for name in public_fields}
    for name in ("result_paths", "result_hashes", "artifact_hashes"):
        if not isinstance(values[name], (list, tuple)):
            raise ValueError(f"shape copyback {name} is invalid")
        values[name] = tuple(values[name])
    return ConfigurationShapeCopybackReadbackV1(
        **values,
        _manifest_json=_canonical_json(payload["manifest"]),
        _destination_file_readbacks_json=_canonical_json(
            payload["destination_file_readbacks"]
        ),
    )
