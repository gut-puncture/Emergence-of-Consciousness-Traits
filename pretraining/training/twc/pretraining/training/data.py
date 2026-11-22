"""Finite external-index packed dataset and deterministic rank planning."""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import hashlib
import json
import mmap
from pathlib import Path, PurePosixPath
import os
from queue import Empty, Full, Queue
import struct
from threading import Event, Thread
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset

from twc.repro.hashing import stable_json_hash
from twc.pretraining.training.model import PackedCausalBatch


_DTYPES = {"uint16": ("<H", 2), "uint32": ("<I", 4)}
_SEQUENCE_INDEX = struct.Struct("<QQIIBBH")


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    """Physical location of one isolated sequence of ``loss_tokens + 1`` IDs."""

    relative_path: str
    token_offset_bytes: int
    token_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or not self.relative_path or Path(self.relative_path).is_absolute():
            raise ValueError("relative_path must be a nonempty relative path")
        if type(self.token_offset_bytes) is not int:
            raise ValueError("token_offset_bytes must be an integer")
        if self.token_offset_bytes < 0 or self.token_offset_bytes % 2:
            raise ValueError("token_offset_bytes must be a nonnegative uint16 boundary")
        if type(self.token_count) is not int:
            raise ValueError("token_count must be an integer")
        if not 2 <= self.token_count <= 4097:
            raise ValueError("token_count must be between 2 and 4097")


class BatchSource:
    """One bounded reader for isolated-sequence training batches.

    The source deliberately has only two I/O modes. Stable local files may be
    memory mapped; removable/network files use ``pread`` so an unavailable
    mount becomes a normal read failure rather than a process-level SIGBUS.
    """

    _SENTINEL = object()

    def __init__(
        self,
        *,
        root: Path,
        records: tuple[SequenceRecord, ...],
        read_mode: str,
        removable: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError("BatchSource root must be an existing directory")
        if read_mode not in {"local_mmap", "buffered_pread"}:
            raise ValueError("read_mode must be local_mmap or buffered_pread")
        if read_mode == "local_mmap" and removable:
            raise ValueError("local_mmap is forbidden for removable storage")
        if not records:
            raise ValueError("BatchSource requires at least one record")
        self.records = records
        self.read_mode = read_mode
        self._paths: dict[str, Path] = {}
        self._fds: dict[str, int] = {}
        self._mappings: dict[str, mmap.mmap] = {}
        self._closed = False
        self._stop = Event()
        for record in records:
            unresolved = self.root / record.relative_path
            if unresolved.is_symlink():
                raise ValueError("sequence path cannot be a symlink")
            path = unresolved.resolve()
            if path != self.root and self.root not in path.parents:
                raise ValueError("sequence path escapes BatchSource root")
            if not path.is_file():
                raise ValueError("sequence token shard is missing")
            self._paths[record.relative_path] = path

    def __enter__(self) -> "BatchSource":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _read(self, record: SequenceRecord) -> bytes:
        if self._closed:
            raise RuntimeError("BatchSource is closed")
        expected = record.token_count * 2
        path = self._paths[record.relative_path]
        if self.read_mode == "buffered_pread":
            descriptor = self._fds.get(record.relative_path)
            if descriptor is None:
                descriptor = os.open(path, os.O_RDONLY)
                self._fds[record.relative_path] = descriptor
            payload = os.pread(descriptor, expected, record.token_offset_bytes)
        else:
            mapping = self._mappings.get(record.relative_path)
            if mapping is None:
                descriptor = os.open(path, os.O_RDONLY)
                self._fds[record.relative_path] = descriptor
                mapping = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
                self._mappings[record.relative_path] = mapping
            start = record.token_offset_bytes
            payload = mapping[start : start + expected]
        if len(payload) != expected:
            raise RuntimeError("sequence token shard was truncated after validation")
        return payload

    def read_batch(self, ordinals: tuple[int, ...]) -> PackedCausalBatch:
        if not ordinals:
            raise ValueError("batch ordinals cannot be empty")
        if len(set(ordinals)) != len(ordinals):
            raise ValueError("batch ordinals cannot repeat")
        inputs: list[int] = []
        targets: list[int] = []
        offsets = [0]
        for ordinal in ordinals:
            if type(ordinal) is not int or not 0 <= ordinal < len(self.records):
                raise IndexError(ordinal)
            record = self.records[ordinal]
            values = [value[0] for value in struct.iter_unpack("<H", self._read(record))]
            inputs.extend(values[:-1])
            targets.extend(values[1:])
            offsets.append(len(inputs))
        return PackedCausalBatch.create(
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            offsets,
        )

    def prefetch(self, batches: Iterator[tuple[int, ...]] | tuple[tuple[int, ...], ...]):
        """Read one batch ahead and surface producer failures to the consumer."""

        queue: Queue[object] = Queue(maxsize=1)
        iterator_stop = Event()

        def publish(value: object) -> bool:
            while not self._stop.is_set() and not iterator_stop.is_set():
                try:
                    queue.put(value, timeout=0.05)
                    return True
                except Full:
                    continue
            return False

        def produce() -> None:
            try:
                for ordinals in batches:
                    if self._stop.is_set() or iterator_stop.is_set() or not publish(self.read_batch(ordinals)):
                        return
            except BaseException as error:  # propagated on the consumer thread
                publish(error)
            finally:
                publish(self._SENTINEL)

        worker = Thread(target=produce, name="twc-batch-source", daemon=True)
        worker.start()
        try:
            while True:
                try:
                    value = queue.get(timeout=0.1)
                except Empty:
                    if not worker.is_alive():
                        return
                    continue
                if value is self._SENTINEL:
                    return
                if isinstance(value, BaseException):
                    raise value
                yield value
        finally:
            iterator_stop.set()
            worker.join(timeout=1.0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        for mapping in self._mappings.values():
            mapping.close()
        for descriptor in self._fds.values():
            os.close(descriptor)
        self._mappings.clear()
        self._fds.clear()


@dataclass(frozen=True, slots=True)
class SequenceTrainingArtifacts:
    """Validated R7 scientific data identity plus physical sequence records."""

    root: Path
    records: tuple[SequenceRecord, ...]
    loss_token_lengths: tuple[int, ...]
    global_steps: tuple[tuple[int, int], ...]
    global_plan_max_loss_tokens: int
    global_plan_max_attention_cells: int
    global_plan_max_sequences: int
    manifest_hash: str
    view_hash: str
    global_plan_hash: str
    placement_hash: str
    split: str = "train"

    def open_source(self, *, read_mode: str, removable: bool = False) -> BatchSource:
        return BatchSource(
            root=self.root,
            records=self.records,
            read_mode=read_mode,
            removable=removable,
        )

    def records_artifact(self, *, tokenizer_hash: str) -> dict[str, Any]:
        if not isinstance(tokenizer_hash, str) or len(tokenizer_hash) != 64:
            raise ValueError("tokenizer_hash must be a SHA-256 hash")
        payload: dict[str, Any] = {
            "artifact_kind": "twc_batch_source_records_v1",
            "schema_version": 1,
            "data_manifest_hash": self.manifest_hash,
            "view_hash": self.view_hash,
            "split": self.split,
            "tokenizer_hash": tokenizer_hash,
            "placement_hash": self.placement_hash,
            "records": [
                {
                    "relative_path": row.relative_path,
                    "token_offset_bytes": row.token_offset_bytes,
                    "token_count": row.token_count,
                }
                for row in self.records
            ],
        }
        payload["records_hash"] = stable_json_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class SequenceEvaluationArtifacts:
    """Validated finite held-out records, intentionally separate from train."""

    root: Path
    records: tuple[SequenceRecord, ...]
    loss_token_lengths: tuple[int, ...]
    manifest_hash: str
    view_hash: str
    placement_hash: str
    split: str = "validation"

    def records_artifact(self, *, tokenizer_hash: str) -> dict[str, Any]:
        if not isinstance(tokenizer_hash, str) or len(tokenizer_hash) != 64:
            raise ValueError("tokenizer_hash must be a SHA-256 hash")
        payload: dict[str, Any] = {
            "artifact_kind": "twc_batch_source_records_v1",
            "schema_version": 1,
            "data_manifest_hash": self.manifest_hash,
            "view_hash": self.view_hash,
            "split": self.split,
            "tokenizer_hash": tokenizer_hash,
            "placement_hash": self.placement_hash,
            "records": [
                {"relative_path": row.relative_path, "token_offset_bytes": row.token_offset_bytes, "token_count": row.token_count}
                for row in self.records
            ],
        }
        payload["records_hash"] = stable_json_hash(payload)
        return payload


def _hash_without(payload: dict[str, Any], field: str) -> str:
    return stable_json_hash({key: value for key, value in payload.items() if key != field})


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or malformed") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def build_verified_placement_receipt(
    *, pack_root: Path, manifest_path: Path
) -> dict[str, Any]:
    """Fully read back an already-uploaded pack without CPU pipeline imports."""
    root = Path(pack_root).resolve(strict=True)
    manifest = _json_object(manifest_path, "isolated sequence manifest")
    if (
        manifest.get("artifact_kind") != "twc_isolated_sequence_pack_manifest_v1"
        or manifest.get("schema_version") != 1
        or manifest.get("manifest_hash") != _hash_without(manifest, "manifest_hash")
    ):
        raise ValueError("uploaded pack manifest is invalid")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("uploaded pack manifest has no shards")
    identities: list[dict[str, Any]] = []
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("uploaded pack shard descriptor is invalid")
        for path_field, hash_field, size_field in (
            ("token_path", "token_sha256", "token_byte_count"),
            ("index_path", "index_sha256", "index_byte_count"),
        ):
            relative = shard.get(path_field)
            expected_hash = shard.get(hash_field)
            expected_size = shard.get(size_field)
            relative_path = PurePosixPath(relative) if isinstance(relative, str) else None
            if (
                relative_path is None
                or relative_path.is_absolute()
                or relative_path.as_posix() != relative
                or any(part in {"", ".", ".."} for part in relative_path.parts)
            ):
                raise ValueError("uploaded pack shard path is invalid")
            path = root.joinpath(*relative_path.parts)
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError("uploaded pack shard path is missing or unsafe")
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
            if size != expected_size or digest.hexdigest() != expected_hash:
                raise ValueError("uploaded pack shard size or hash mismatch")
            stat = path.stat()
            identities.append(
                {
                    "relative_path": relative,
                    "device_id": stat.st_dev,
                    "inode": stat.st_ino,
                    "byte_count": size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": expected_hash,
                }
            )
    receipt: dict[str, Any] = {
        "artifact_kind": "twc_data_placement_receipt_v1",
        "schema_version": 1,
        "pack_manifest_hash": manifest["manifest_hash"],
        "verified_file_count": len(identities),
        "full_destination_readback": True,
        "destination_root": str(root),
        "destination_device_id": root.stat().st_dev,
        "file_identities": identities,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    receipt["receipt_hash"] = _hash_without(receipt, "receipt_hash")
    return receipt


def _load_finite_view_records(
    *, root: Path, manifest: dict[str, Any], view: dict[str, Any], split: str
) -> tuple[list[SequenceRecord], list[int]]:
    requested = view.get("loss_token_count")
    if type(requested) is not int or requested <= 0:
        raise ValueError("finite sequence view loss token count is invalid")
    remaining = requested
    records: list[SequenceRecord] = []
    lengths: list[int] = []
    expected_split_code = {"train": 0, "validation": 1}[split]
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("isolated sequence manifest requires shards")
    for shard in shards:
        if remaining == 0:
            break
        if not isinstance(shard, dict) or shard.get("split") != split:
            continue
        token_relative, index_relative = shard.get("token_path"), shard.get("index_path")
        if not isinstance(token_relative, str) or not isinstance(index_relative, str):
            raise ValueError("pack shard paths are invalid")
        token_path, index_path = root / token_relative, root / index_relative
        if token_path.is_symlink() or index_path.is_symlink():
            raise ValueError("pack shard paths cannot be symlinks")
        if not token_path.is_file() or not index_path.is_file():
            raise ValueError("pack shard file is missing")
        if token_path.stat().st_size != shard.get("token_byte_count"):
            raise ValueError("pack token shard byte count mismatch")
        index_payload = index_path.read_bytes()
        if len(index_payload) != shard.get("index_byte_count") or hashlib.sha256(index_payload).hexdigest() != shard.get("index_sha256"):
            raise ValueError("pack index shard readback mismatch")
        for record in struct.iter_unpack(_SEQUENCE_INDEX.format, index_payload):
            offset, _run, token_count, _component, split_code, flags, reserved = record
            if split_code != expected_split_code or flags != 0 or reserved != 0 or token_count < 2:
                raise ValueError("pack index contains unsupported metadata")
            use_loss = min(token_count - 1, remaining)
            records.append(SequenceRecord(token_relative, offset, use_loss + 1))
            lengths.append(use_loss)
            remaining -= use_loss
            if remaining == 0:
                break
    if remaining:
        raise ValueError(f"finite sequence view exceeds placed {split} data")
    return records, lengths


def load_sequence_evaluation_artifacts(
    *, pack_root: Path, manifest_path: Path, view_path: Path, placement_receipt: dict[str, Any]
) -> SequenceEvaluationArtifacts:
    if not isinstance(placement_receipt, dict) or placement_receipt.get("artifact_kind") != "twc_data_placement_receipt_v1" or placement_receipt.get("receipt_hash") != _hash_without(placement_receipt, "receipt_hash") or placement_receipt.get("full_destination_readback") is not True:
        raise ValueError("a hash-valid full-readback R7 data placement receipt is required")
    root = Path(pack_root).resolve()
    manifest = _json_object(manifest_path, "isolated sequence manifest")
    view = _json_object(view_path, "finite validation view")
    if manifest.get("artifact_kind") != "twc_isolated_sequence_pack_manifest_v1" or manifest.get("schema_version") != 1 or manifest.get("manifest_hash") != _hash_without(manifest, "manifest_hash") or placement_receipt.get("pack_manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError("validation manifest/placement binding mismatch")
    if view.get("artifact_kind") != "twc_finite_sequence_view_v1" or view.get("schema_version") != 1 or view.get("view_hash") != _hash_without(view, "view_hash") or view.get("pack_manifest_hash") != manifest.get("manifest_hash") or view.get("split") != "validation" or view.get("finite") is not True or view.get("repeat_allowed") is not False:
        raise ValueError("finite validation view is invalid or not pack-bound")
    records, lengths = _load_finite_view_records(root=root, manifest=manifest, view=view, split="validation")
    return SequenceEvaluationArtifacts(
        root=root, records=tuple(records), loss_token_lengths=tuple(lengths),
        manifest_hash=str(manifest["manifest_hash"]), view_hash=str(view["view_hash"]),
        placement_hash=str(placement_receipt["receipt_hash"]),
    )


def load_sequence_training_artifacts(
    *,
    pack_root: Path,
    manifest_path: Path,
    view_path: Path,
    global_plan_path: Path,
    placement_receipt: dict[str, Any],
) -> SequenceTrainingArtifacts:
    """Load the R7 sequence contract without importing CPU-pipeline writers.

    The CPU placement operation performs the expensive token-shard readback.
    Startup revalidates the small index files, byte bounds, identities, and
    semantic plan so a shape-valid but forged artifact cannot reach the model.
    """

    if not isinstance(placement_receipt, dict) or placement_receipt.get(
        "artifact_kind"
    ) != "twc_data_placement_receipt_v1":
        raise ValueError("a verified R7 data placement receipt is required")
    if placement_receipt.get("receipt_hash") != _hash_without(
        placement_receipt, "receipt_hash"
    ):
        raise ValueError("data placement receipt hash mismatch")
    if placement_receipt.get("full_destination_readback") is not True:
        raise ValueError("data placement receipt lacks full destination readback")

    root = Path(pack_root).resolve()
    manifest = _json_object(manifest_path, "isolated sequence manifest")
    view = _json_object(view_path, "finite sequence view")
    plan = _json_object(global_plan_path, "global step plan")
    if (
        manifest.get("artifact_kind") != "twc_isolated_sequence_pack_manifest_v1"
        or manifest.get("schema_version") != 1
        or manifest.get("manifest_hash") != _hash_without(manifest, "manifest_hash")
    ):
        raise ValueError("unsupported or hash-invalid isolated sequence manifest")
    manifest_hash = str(manifest["manifest_hash"])
    if placement_receipt.get("pack_manifest_hash") != manifest_hash:
        raise ValueError("data placement receipt does not bind the pack manifest")
    if (
        view.get("artifact_kind") != "twc_finite_sequence_view_v1"
        or view.get("schema_version") != 1
        or view.get("view_hash") != _hash_without(view, "view_hash")
        or view.get("pack_manifest_hash") != manifest_hash
        or view.get("split") != "train"
        or view.get("finite") is not True
        or view.get("repeat_allowed") is not False
    ):
        raise ValueError("finite sequence view is invalid or not pack-bound")
    if (
        plan.get("artifact_kind") != "twc_global_step_plan_v1"
        or plan.get("schema_version") != 1
        or plan.get("plan_hash") != _hash_without(plan, "plan_hash")
        or plan.get("pack_manifest_hash") != manifest_hash
        or plan.get("view_hash") != view.get("view_hash")
    ):
        raise ValueError("global step plan is invalid or not view-bound")

    requested = view.get("loss_token_count")
    records, lengths = _load_finite_view_records(root=root, manifest=manifest, view=view, split="train")

    steps = plan.get("steps")
    if not isinstance(steps, list) or len(steps) != plan.get("step_count"):
        raise ValueError("global step plan step count mismatch")
    max_loss_tokens = plan.get("max_loss_tokens")
    max_attention_cells = plan.get("max_attention_cells")
    max_sequences = plan.get("max_sequences")
    if any(
        type(value) is not int or value <= 0
        for value in (max_loss_tokens, max_attention_cells, max_sequences)
    ):
        raise ValueError("global step plan caps are invalid")
    global_steps: list[tuple[int, int]] = []
    cursor = 0
    total = 0
    for ordinal, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("ordinal") != ordinal:
            raise ValueError("global step plan ordinal mismatch")
        count = step.get("sequence_count")
        if type(count) is not int or count <= 0:
            raise ValueError("global step plan sequence count is invalid")
        stop = cursor + count
        selected = lengths[cursor:stop]
        loss = sum(selected)
        cells = sum(value * value for value in selected)
        if (
            len(selected) != count
            or step.get("first_segment_index") != cursor
            or step.get("loss_token_count") != loss
            or step.get("attention_cells") != cells
            or loss > max_loss_tokens
            or cells > max_attention_cells
            or count > max_sequences
        ):
            raise ValueError("global step plan semantic accounting mismatch")
        global_steps.append((cursor, stop))
        cursor = stop
        total += loss
    if cursor != len(records) or total != requested or total != plan.get("loss_token_count"):
        raise ValueError("global step plan does not cover the finite view exactly")
    return SequenceTrainingArtifacts(
        root=root,
        records=tuple(records),
        loss_token_lengths=tuple(lengths),
        global_steps=tuple(global_steps),
        global_plan_max_loss_tokens=max_loss_tokens,
        global_plan_max_attention_cells=max_attention_cells,
        global_plan_max_sequences=max_sequences,
        manifest_hash=manifest_hash,
        view_hash=str(view["view_hash"]),
        global_plan_hash=str(plan["plan_hash"]),
        placement_hash=str(placement_receipt["receipt_hash"]),
    )
