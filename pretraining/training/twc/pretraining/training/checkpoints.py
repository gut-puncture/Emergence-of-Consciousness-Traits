"""Immutable generational checkpoints with verified fallback heads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class CheckpointPayload:
    run_identity: str
    attempt_hash: str
    step: int
    consumed_loss_tokens: int
    next_global_ordinal: int
    state: dict[str, Any]
    cpu_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class CheckpointReference:
    generation: int
    path: Path
    durability: str
    manifest_sha256: str | None = None
    state_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    generation: int
    durability: str
    run_identity: str
    attempt_hash: str
    step: int
    consumed_loss_tokens: int
    next_global_ordinal: int
    state: dict[str, Any]
    cpu_rng_state: torch.Tensor
    cuda_rng_states: tuple[torch.Tensor, ...]
    manifest_sha256: str
    state_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    _fsync_file(partial)
    os.replace(partial, path)
    _fsync_directory(path.parent)


class CheckpointManager:
    def __init__(
        self,
        root: Path,
        *,
        persistent_root: Path | None = None,
        topology: str | None = None,
        create: bool = True,
    ) -> None:
        if topology is None:
            topology = "local_then_replicate" if persistent_root is not None else "local_only"
        if topology not in {"local_only", "direct_persistent", "local_then_replicate"}:
            raise ValueError("unsupported checkpoint topology")
        if topology == "local_then_replicate" and persistent_root is None:
            raise ValueError("local_then_replicate requires persistent_root")
        if topology == "direct_persistent" and persistent_root is not None:
            raise ValueError("direct_persistent uses root as the durable storage root")
        self.root = root
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("checkpoint root must already exist as a real directory")
        self.topology = topology
        self.persistent_root = persistent_root
        if persistent_root is not None:
            if create:
                persistent_root.mkdir(parents=True, exist_ok=True)
            elif not persistent_root.is_dir() or persistent_root.is_symlink():
                raise ValueError("persistent checkpoint root must already exist as a real directory")

    def publish(self, payload: CheckpointPayload, *, durability: str) -> CheckpointReference:
        if durability not in {"local_verified", "persistent_verified"}:
            raise ValueError("unsupported checkpoint durability")
        if self.topology == "local_only" and durability == "persistent_verified":
            raise ValueError("persistent_verified requires a persistent checkpoint topology")
        if self.topology == "direct_persistent":
            durability = "persistent_verified"
        if min(payload.step, payload.consumed_loss_tokens, payload.next_global_ordinal) < 0:
            raise ValueError("checkpoint counters must be nonnegative")
        generation = self._next_generation()
        name = f"generation-{generation:08d}"
        partial = self.root / f"{name}.partial"
        final = self.root / name
        if partial.exists() or final.exists():
            raise FileExistsError(f"checkpoint generation already exists: {name}")
        partial.mkdir(mode=0o700)
        state_path = partial / "state.pt"
        torch.save(
            {
                "state": payload.state,
                "cpu_rng_state": payload.cpu_rng_state,
                "cuda_rng_states": payload.cuda_rng_states,
            },
            state_path,
        )
        _fsync_file(state_path)
        manifest = {
            "artifact_kind": "twc_pretraining_checkpoint_v1",
            "schema_version": 1,
            "generation": generation,
            "run_identity": payload.run_identity,
            "attempt_hash": payload.attempt_hash,
            "step": payload.step,
            "consumed_loss_tokens": payload.consumed_loss_tokens,
            "next_global_ordinal": payload.next_global_ordinal,
            "durability": "local_verified",
            "state_sha256": _sha256(state_path),
            "state_bytes": state_path.stat().st_size,
        }
        _atomic_json(partial / "manifest.json", manifest)
        _atomic_json(partial / "READY.json", {"generation": generation, "manifest_sha256": _sha256(partial / "manifest.json")})
        _fsync_directory(partial)
        os.replace(partial, final)
        _fsync_directory(self.root)
        if durability == "persistent_verified":
            if self.topology == "local_then_replicate":
                self._replicate_and_verify(final, generation)
            else:
                receipt = {
                    "artifact_kind": "twc_persistent_checkpoint_receipt_v1",
                    "generation": generation,
                    "relative_path": final.name,
                    "manifest_sha256": _sha256(final / "manifest.json"),
                    "state_sha256": _sha256(final / "state.pt"),
                    "topology": "direct_persistent",
                }
                _atomic_json(final / "PERSISTENT.json", receipt)
        head_slot = "head-a.json" if generation % 2 == 0 else "head-b.json"
        _atomic_json(self.root / head_slot, {"generation": generation, "relative_path": name})
        self._prune_unreferenced()
        return CheckpointReference(generation=generation, path=final, durability=durability)

    def load_latest(self) -> LoadedCheckpoint:
        candidates: list[tuple[int, int, Path]] = []
        for head_name in ("head-a.json", "head-b.json"):
            try:
                head = json.loads((self.root / head_name).read_text(encoding="utf-8"))
                generation = int(head["generation"])
                relative = str(head["relative_path"])
                if relative != f"generation-{generation:08d}":
                    raise ValueError("checkpoint head path binding mismatch")
                candidates.append((generation, 1, self.root / relative))
            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        if self.persistent_root is not None:
            for path in self.persistent_root.glob("generation-[0-9]*"):
                if not path.is_dir() or path.is_symlink() or path.name.endswith(".partial"):
                    continue
                try:
                    generation = int(path.name.split("-")[1])
                except (IndexError, ValueError):
                    continue
                candidates.append((generation, 0, path))
        for generation, _priority, path in sorted(candidates, reverse=True):
            try:
                return self._load_verified(generation, path)
            except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
                continue
        raise RuntimeError("no verified checkpoint generation is recoverable")

    def repair(self) -> dict[str, Any]:
        """Quarantine interrupted writes and promote the newest verified headless generation."""
        quarantine = self.root / ".quarantine"
        quarantined: list[str] = []
        for partial in sorted(self.root.glob("generation-[0-9]*.partial")):
            quarantine.mkdir(mode=0o700, exist_ok=True)
            destination = quarantine / partial.name
            suffix = 0
            while destination.exists():
                suffix += 1
                destination = quarantine / f"{partial.name}.{suffix}"
            os.replace(partial, destination)
            quarantined.append(partial.name)
        if quarantined:
            _fsync_directory(quarantine)
            _fsync_directory(self.root)

        headed: set[int] = set()
        for head_name in ("head-a.json", "head-b.json"):
            try:
                headed.add(int(json.loads((self.root / head_name).read_text(encoding="utf-8"))["generation"]))
            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        verified: list[tuple[int, Path]] = []
        for path in self.root.glob("generation-[0-9]*"):
            if not path.is_dir() or path.name.endswith(".partial"):
                continue
            try:
                generation = int(path.name.split("-")[1])
                self._load_verified(generation, path)
            except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
                continue
            verified.append((generation, path))
        promoted: int | None = None
        if verified:
            generation, path = max(verified)
            if generation not in headed:
                head_slot = "head-a.json" if generation % 2 == 0 else "head-b.json"
                _atomic_json(
                    self.root / head_slot,
                    {"generation": generation, "relative_path": path.name},
                )
                promoted = generation
        return {
            "artifact_kind": "twc_checkpoint_repair_receipt_v1",
            "schema_version": 1,
            "quarantined_partials": quarantined,
            "promoted_generation": promoted,
        }

    def validate_reference(self, reference: CheckpointReference) -> LoadedCheckpoint:
        try:
            path = reference.path.resolve()
            root = self.root.resolve()
        except OSError as error:
            raise ValueError("checkpoint reference path cannot be resolved") from error
        if path.parent != root:
            raise ValueError("checkpoint reference is outside the managed checkpoint root")
        loaded = self._load_verified(reference.generation, path)
        if loaded.durability != reference.durability:
            raise ValueError("checkpoint reference durability is not verified")
        if (reference.manifest_sha256 is None) != (reference.state_sha256 is None):
            raise ValueError("checkpoint reference must bind both content hashes")
        if (
            reference.manifest_sha256 is not None
            and reference.manifest_sha256 != loaded.manifest_sha256
        ):
            raise ValueError("checkpoint manifest identity mismatch")
        if (
            reference.state_sha256 is not None
            and reference.state_sha256 != loaded.state_sha256
        ):
            raise ValueError("checkpoint state identity mismatch")
        return loaded

    def _load_verified(self, generation: int, path: Path) -> LoadedCheckpoint:
        manifest_path, ready_path, state_path = path / "manifest.json", path / "READY.json", path / "state.pt"
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        manifest_sha256 = _sha256(manifest_path)
        if int(ready["generation"]) != generation or ready["manifest_sha256"] != manifest_sha256:
            raise ValueError("checkpoint READY binding mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state_sha256 = _sha256(state_path)
        if int(manifest["generation"]) != generation or manifest["state_sha256"] != state_sha256:
            raise ValueError("checkpoint state binding mismatch")
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        return LoadedCheckpoint(
            generation=generation,
            durability="persistent_verified" if self._persistent_verified(path, generation) else "local_verified",
            run_identity=str(manifest["run_identity"]),
            attempt_hash=str(manifest["attempt_hash"]),
            step=int(manifest["step"]),
            consumed_loss_tokens=int(manifest["consumed_loss_tokens"]),
            next_global_ordinal=int(manifest["next_global_ordinal"]),
            state=dict(saved["state"]),
            cpu_rng_state=saved["cpu_rng_state"],
            cuda_rng_states=tuple(saved["cuda_rng_states"]),
            manifest_sha256=manifest_sha256,
            state_sha256=state_sha256,
        )

    def _next_generation(self) -> int:
        generations = []
        for path in self.root.glob("generation-[0-9]*"):
            if path.name.endswith(".partial"):
                continue
            try:
                generations.append(int(path.name.split("-")[1]))
            except (IndexError, ValueError):
                pass
        return max(generations, default=-1) + 1

    def _replicate_and_verify(self, source: Path, generation: int) -> None:
        if self.persistent_root is None:
            raise ValueError("persistent_verified requires a configured persistent_root")
        destination = self.persistent_root / source.name
        partial = self.persistent_root / f"{source.name}.partial"
        if destination.exists() or partial.exists():
            raise FileExistsError("persistent checkpoint generation already exists")
        shutil.copytree(source, partial)
        self._load_verified(generation, partial)
        os.replace(partial, destination)
        _fsync_directory(self.persistent_root)
        receipt = {
            "artifact_kind": "twc_persistent_checkpoint_receipt_v1",
            "generation": generation,
            "relative_path": destination.name,
            "manifest_sha256": _sha256(destination / "manifest.json"),
            "state_sha256": _sha256(destination / "state.pt"),
        }
        _atomic_json(source / "PERSISTENT.json", receipt)

    def _persistent_verified(self, local_path: Path, generation: int) -> bool:
        if self.topology == "direct_persistent":
            return local_path.parent.resolve() == self.root.resolve()
        if self.persistent_root is None:
            return False
        if local_path.parent.resolve() == self.persistent_root.resolve():
            return True
        try:
            receipt = json.loads((local_path / "PERSISTENT.json").read_text(encoding="utf-8"))
            destination = self.persistent_root / str(receipt["relative_path"])
            if int(receipt["generation"]) != generation:
                return False
            loaded = self._load_verified(generation, destination)
            return (
                loaded.generation == generation
                and receipt["manifest_sha256"] == _sha256(destination / "manifest.json")
                and receipt["state_sha256"] == _sha256(destination / "state.pt")
            )
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError):
            return False

    def _prune_unreferenced(self) -> None:
        generations = sorted(
            (path for path in self.root.glob("generation-[0-9]*") if not path.name.endswith(".partial")),
            key=lambda path: path.name,
            reverse=True,
        )
        for path in generations[2:]:
            shutil.rmtree(path)
