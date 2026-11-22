"""Tokenizer construction and exact clean-run tokenization contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twc.repro.hashing import stable_json_hash
from twc.repro.public_artifacts import validate_public_no_raw_payload

from .contracts import hash_without, require_int, require_sha256


@dataclass(frozen=True)
class CleanRunText:
    split: str
    clean_run_id_hash: str
    source_document_id_hash: str
    text: str


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int
    model_type: str
    special_tokens: tuple[str, ...]
    normalizer: str = "none"
    pre_tokenizer: str = "byte_level"
    byte_fallback: bool = True
    min_frequency: int = 2


@dataclass(frozen=True)
class TokenizerBuildOutput:
    artifact_bytes: bytes
    actual_vocab_size: int
    special_token_ids: Mapping[str, int]
    unknown_token_count: int
    roundtrip_failure_count: int
    deterministic_rebuild: bool
    chars_per_token: float
    vocab_inspection_passed: bool
    forbidden_whole_token_count: int


@dataclass(frozen=True)
class TokenizerBuildResult:
    manifest: dict[str, Any]
    public_summary: dict[str, Any]
    artifact_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class TokenizerPolicy:
    version: int
    config: TokenizerConfig
    train_on_accepted_text_only: bool
    validation_split_must_remain_untouched: bool
    deterministic_rebuild_required: bool
    diagnostics_required: tuple[str, ...]
    manifest_required: bool


TokenizerTextFactory = Callable[[], Iterator[str]]
TokenizerBuilder = Callable[[TokenizerTextFactory, TokenizerConfig], TokenizerBuildOutput]


def _atomic_write(path: Path, data: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(partial, 0o600)
    os.replace(partial, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: TokenizerConfig) -> None:
    if type(config.vocab_size) is not int or config.vocab_size < 8 or config.vocab_size > 65_536:
        raise ValueError("vocab_size must be an integer between 8 and 65536")
    if config.model_type not in {"bpe", "unigram"}:
        raise ValueError("unsupported tokenizer model type")
    if not config.special_tokens or len(config.special_tokens) != len(set(config.special_tokens)):
        raise ValueError("special_tokens must be unique and non-empty")
    if type(config.min_frequency) is not int or config.min_frequency < 1:
        raise ValueError("min_frequency must be positive")


def load_tokenizer_policy(path: Path) -> TokenizerPolicy:
    """Strictly consume every key in the checked tokenizer policy."""

    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tokenizer config must be a mapping")
    expected = {
        "version",
        "vocab_size",
        "model_type",
        "normalizer",
        "pre_tokenizer",
        "byte_fallback",
        "min_frequency",
        "special_tokens",
        "train_on_accepted_text_only",
        "validation_split_must_remain_untouched",
        "deterministic_rebuild_required",
        "diagnostics_required",
        "manifest_required",
    }
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        raise ValueError(f"tokenizer config keys mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")
    for field in (
        "train_on_accepted_text_only",
        "validation_split_must_remain_untouched",
        "deterministic_rebuild_required",
        "manifest_required",
    ):
        if payload[field] is not True:
            raise ValueError(f"tokenizer policy requires {field}=true")
    diagnostics = payload["diagnostics_required"]
    required_diagnostics = {
        "chars_per_token",
        "vocab_inspection",
        "forbidden_whole_token_report",
        "unknown_token_count",
        "roundtrip_failure_count",
    }
    if not isinstance(diagnostics, list) or set(diagnostics) != required_diagnostics:
        raise ValueError("tokenizer diagnostics_required is incomplete or unknown")
    config = TokenizerConfig(
        vocab_size=payload["vocab_size"],
        model_type=payload["model_type"],
        special_tokens=tuple(payload["special_tokens"]),
        normalizer=payload["normalizer"],
        pre_tokenizer=payload["pre_tokenizer"],
        byte_fallback=payload["byte_fallback"],
        min_frequency=payload["min_frequency"],
    )
    _validate_config(config)
    if payload["version"] != 2:
        raise ValueError("unsupported tokenizer policy version")
    return TokenizerPolicy(
        version=2,
        config=config,
        train_on_accepted_text_only=True,
        validation_split_must_remain_untouched=True,
        deterministic_rebuild_required=True,
        diagnostics_required=tuple(diagnostics),
        manifest_required=True,
    )


def build_tokenizer(
    *,
    records: Iterable[CleanRunText],
    config: TokenizerConfig,
    corpus_manifest_hash: str,
    output_root: Path,
    builder: TokenizerBuilder,
) -> TokenizerBuildResult:
    """Build a tokenizer over training clean runs only via an injected backend."""

    _validate_config(config)
    require_sha256(corpus_manifest_hash, "corpus_manifest_hash")
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(str(output_root))
    output_root.mkdir(parents=True, mode=0o700)
    os.chmod(output_root, 0o700)
    spool_path = output_root / ".tokenizer-training-input.jsonl"
    train_count = validation_count = 0
    training_order_digest = hashlib.sha256()
    seen: set[str] = set()
    with spool_path.open("wb") as spool:
        for record in records:
            if not isinstance(record, CleanRunText) or record.split not in {"train", "validation"}:
                raise ValueError("records must be split CleanRunText")
            require_sha256(record.clean_run_id_hash, "clean_run_id_hash")
            require_sha256(record.source_document_id_hash, "source_document_id_hash")
            if record.clean_run_id_hash in seen:
                raise ValueError("duplicate clean-run identity")
            seen.add(record.clean_run_id_hash)
            if not record.text:
                raise ValueError("clean-run text must be non-empty")
            if record.split == "train":
                spool.write((json.dumps(record.text, ensure_ascii=True) + "\n").encode())
                training_order_digest.update((record.clean_run_id_hash + "\n").encode())
                train_count += 1
            else:
                validation_count += 1
        spool.flush()
        os.fsync(spool.fileno())
    os.chmod(spool_path, 0o600)
    if not train_count:
        raise ValueError("tokenizer training requires training clean runs")
    if not validation_count:
        raise ValueError("tokenizer validation requires untouched validation clean runs")

    def text_factory() -> Iterator[str]:
        with spool_path.open(encoding="utf-8") as handle:
            for line in handle:
                yield str(json.loads(line))

    output = builder(text_factory, config)
    if not isinstance(output, TokenizerBuildOutput) or not output.artifact_bytes:
        raise ValueError("tokenizer builder returned no artifact")
    if output.actual_vocab_size != config.vocab_size:
        raise ValueError("tokenizer actual vocabulary size mismatch")
    expected_specials = set(config.special_tokens)
    if set(output.special_token_ids) != expected_specials:
        raise ValueError("tokenizer special-token mapping mismatch")
    ids = list(output.special_token_ids.values())
    if any(type(value) is not int or value < 0 or value >= config.vocab_size for value in ids):
        raise ValueError("invalid special-token id")
    if len(ids) != len(set(ids)):
        raise ValueError("special-token ids must be unique")
    ready = (
        output.unknown_token_count == 0
        and output.roundtrip_failure_count == 0
        and output.deterministic_rebuild is True
        and output.chars_per_token > 0
        and output.vocab_inspection_passed is True
        and output.forbidden_whole_token_count == 0
    )
    if not ready:
        raise ValueError("tokenizer diagnostics are not ready")

    spool_path.unlink()
    artifact_path = output_root / "tokenizer.bin"
    _atomic_write(artifact_path, output.artifact_bytes)
    artifact_hash = hashlib.sha256(output.artifact_bytes).hexdigest()
    config_payload = {
        "vocab_size": config.vocab_size,
        "model_type": config.model_type,
        "special_tokens": list(config.special_tokens),
        "normalizer": config.normalizer,
        "pre_tokenizer": config.pre_tokenizer,
        "byte_fallback": config.byte_fallback,
        "min_frequency": config.min_frequency,
    }
    training_order_root = training_order_digest.hexdigest()
    manifest: dict[str, Any] = {
        "artifact_kind": "twc_tokenizer_manifest_v1",
        "schema_version": 1,
        "corpus_manifest_hash": corpus_manifest_hash,
        "training_clean_run_count": train_count,
        "validation_clean_run_count": validation_count,
        "training_order_root": training_order_root,
        "config": config_payload,
        "config_hash": stable_json_hash(config_payload),
        "artifact_sha256": artifact_hash,
        "actual_vocab_size": output.actual_vocab_size,
        "special_token_ids": dict(sorted(output.special_token_ids.items())),
        "diagnostics": {
            "unknown_token_count": output.unknown_token_count,
            "roundtrip_failure_count": output.roundtrip_failure_count,
            "deterministic_rebuild": output.deterministic_rebuild,
            "chars_per_token": output.chars_per_token,
            "vocab_inspection_passed": output.vocab_inspection_passed,
            "forbidden_whole_token_count": output.forbidden_whole_token_count,
            "ready": ready,
        },
        "validation_used_for_training": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    manifest["manifest_hash"] = stable_json_hash(manifest)
    manifest_path = output_root / "tokenizer_manifest.json"
    _atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    summary = {
        "artifact_kind": "twc_tokenizer_public_summary_v1",
        "schema_version": 1,
        "manifest_hash": manifest["manifest_hash"],
        "artifact_sha256": artifact_hash,
        "training_clean_run_count": train_count,
        "validation_clean_run_count": validation_count,
        "diagnostics": manifest["diagnostics"],
        "validation_used_for_training": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    validate_public_no_raw_payload(summary)
    return TokenizerBuildResult(manifest, summary, artifact_path, manifest_path)


def validate_tokenizer_artifacts(
    manifest: Mapping[str, Any], tokenizer_root: Path
) -> None:
    """Validate the complete tokenizer manifest and its physical artifact."""

    expected = {
        "artifact_kind", "schema_version", "corpus_manifest_hash",
        "training_clean_run_count", "validation_clean_run_count",
        "training_order_root", "config", "config_hash", "artifact_sha256",
        "actual_vocab_size", "special_token_ids", "diagnostics",
        "validation_used_for_training", "raw_text_serialized",
        "raw_url_serialized", "token_serialized", "manifest_hash",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected:
        raise ValueError("tokenizer manifest fields are invalid")
    if (
        manifest.get("artifact_kind") != "twc_tokenizer_manifest_v1"
        or manifest.get("schema_version") != 1
        or manifest.get("manifest_hash") != hash_without(manifest, "manifest_hash")
    ):
        raise ValueError("tokenizer manifest identity is invalid")
    require_sha256(manifest.get("corpus_manifest_hash"), "corpus_manifest_hash")
    require_sha256(manifest.get("training_order_root"), "training_order_root")
    require_sha256(manifest.get("artifact_sha256"), "artifact_sha256")
    config = manifest.get("config")
    if (
        not isinstance(config, Mapping)
        or set(config) != {
            "vocab_size", "model_type", "special_tokens", "normalizer",
            "pre_tokenizer", "byte_fallback", "min_frequency",
        }
        or manifest.get("config_hash") != stable_json_hash(config)
    ):
        raise ValueError("tokenizer config binding is invalid")
    _validate_config(
        TokenizerConfig(
            vocab_size=config.get("vocab_size"),
            model_type=config.get("model_type"),
            special_tokens=tuple(config.get("special_tokens", ())),
            normalizer=config.get("normalizer"),
            pre_tokenizer=config.get("pre_tokenizer"),
            byte_fallback=config.get("byte_fallback"),
            min_frequency=config.get("min_frequency"),
        )
    )
    if manifest.get("actual_vocab_size") != config.get("vocab_size"):
        raise ValueError("tokenizer vocabulary size is invalid")
    require_int(manifest.get("training_clean_run_count"), "training_clean_run_count", minimum=1)
    require_int(
        manifest.get("validation_clean_run_count"),
        "validation_clean_run_count",
        minimum=1,
    )
    special_ids = manifest.get("special_token_ids")
    if (
        not isinstance(special_ids, Mapping)
        or set(special_ids) != set(config["special_tokens"])
        or len(set(special_ids.values())) != len(special_ids)
        or any(
            type(value) is not int or value < 0 or value >= config["vocab_size"]
            for value in special_ids.values()
        )
    ):
        raise ValueError("tokenizer special-token mapping is invalid")
    diagnostics = manifest.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {
        "unknown_token_count", "roundtrip_failure_count", "deterministic_rebuild",
        "chars_per_token", "vocab_inspection_passed",
        "forbidden_whole_token_count", "ready",
    } or diagnostics.get("ready") is not True:
        raise ValueError("tokenizer diagnostics are invalid")
    if (
        diagnostics["unknown_token_count"] != 0
        or diagnostics["roundtrip_failure_count"] != 0
        or diagnostics["deterministic_rebuild"] is not True
        or type(diagnostics["chars_per_token"]) not in {int, float}
        or not math.isfinite(diagnostics["chars_per_token"])
        or diagnostics["chars_per_token"] <= 0
        or diagnostics["vocab_inspection_passed"] is not True
        or diagnostics["forbidden_whole_token_count"] != 0
    ):
        raise ValueError("tokenizer diagnostics are not production-ready")
    if manifest.get("validation_used_for_training") is not False:
        raise ValueError("tokenizer validation split policy is invalid")
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if manifest.get(field) is not False:
            raise ValueError(f"tokenizer manifest {field} must be false")
    artifact = Path(tokenizer_root) / "tokenizer.bin"
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("tokenizer artifact must be a regular non-symlink file")
    if _sha256_file(artifact) != manifest.get("artifact_sha256"):
        raise ValueError("tokenizer artifact hash mismatch")

def huggingface_tokenizers_builder(
    text_factory: TokenizerTextFactory, config: TokenizerConfig
) -> TokenizerBuildOutput:
    """Production BPE backend; imported lazily from the optional ``data`` extra."""

    if config.model_type != "bpe":
        raise ValueError("the production backend currently supports bpe only")
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("install the project data extra to build the tokenizer") from exc

    def train_once() -> Any:
        tokenizer = Tokenizer(models.BPE(byte_fallback=config.byte_fallback))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=config.vocab_size,
            min_frequency=config.min_frequency,
            special_tokens=list(config.special_tokens),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(text_factory(), trainer=trainer)
        return tokenizer

    first = train_once()
    second = train_once()
    first_bytes = first.to_str().encode()
    second_bytes = second.to_str().encode()
    unknown_count = roundtrip_failures = total_chars = total_tokens = 0
    for text in text_factory():
        encoded = first.encode(text)
        total_chars += len(text)
        total_tokens += len(encoded.ids)
        if first.decode(encoded.ids) != text:
            roundtrip_failures += 1
    special_ids = {token: first.token_to_id(token) for token in config.special_tokens}
    if any(value is None for value in special_ids.values()):
        raise ValueError("production tokenizer omitted a special token")
    vocabulary = first.get_vocab()
    vocab_inspection_passed = (
        len(vocabulary) == first.get_vocab_size()
        and len(set(vocabulary.values())) == len(vocabulary)
        and all(0 <= value < first.get_vocab_size() for value in vocabulary.values())
    )
    forbidden_fragments = ("http://", "https://", "api_key=", "password=", "secret=")
    forbidden_count = sum(
        any(fragment in token.casefold() for fragment in forbidden_fragments)
        for token in vocabulary
    )
    return TokenizerBuildOutput(
        artifact_bytes=first_bytes,
        actual_vocab_size=first.get_vocab_size(),
        special_token_ids={key: int(value) for key, value in special_ids.items()},
        unknown_token_count=unknown_count,
        roundtrip_failure_count=roundtrip_failures,
        deterministic_rebuild=first_bytes == second_bytes,
        chars_per_token=total_chars / max(total_tokens, 1),
        vocab_inspection_passed=vocab_inspection_passed,
        forbidden_whole_token_count=forbidden_count,
    )
