"""No-raw adoption and continuity contracts for the C2/C3 composite pack.

This module is deliberately pure.  It validates already-measured identities and
emits content-bound receipts; it never opens the private proxy SQLite spool,
reads token shards, copies parent bytes, or materializes an extension.  The
runtime operator owns those mechanisms and must feed their measured digests
through these contracts before a C2/C3 study can bind the resulting views.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping

from twc.pretraining.data_pipeline.contracts import hash_without
from twc.repro.hashing import stable_json_hash
from twc.repro.public_artifacts import validate_public_no_raw_payload


C1_TRAIN_PREFIX_LOSS_TOKENS = 163_228_160
FROZEN_VALIDATION_LOSS_TOKENS = 8_388_608
C2_TRAIN_LOSS_TOKENS = 613_086_720
C3_TRAIN_LOSS_TOKENS = 982_275_840

_TOKENIZER_MANIFEST_FIELDS = {
    "artifact_kind",
    "schema_version",
    "corpus_manifest_hash",
    "training_clean_run_count",
    "validation_clean_run_count",
    "training_order_root",
    "config",
    "config_hash",
    "artifact_sha256",
    "actual_vocab_size",
    "special_token_ids",
    "diagnostics",
    "validation_used_for_training",
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "manifest_hash",
}
_TOKENIZER_CONFIG_FIELDS = {
    "vocab_size",
    "model_type",
    "special_tokens",
    "normalizer",
    "pre_tokenizer",
    "byte_fallback",
    "min_frequency",
}
_TOKENIZER_DIAGNOSTIC_FIELDS = {
    "unknown_token_count",
    "roundtrip_failure_count",
    "deterministic_rebuild",
    "chars_per_token",
    "vocab_inspection_passed",
    "forbidden_whole_token_count",
    "ready",
}
_SPOOL_RECEIPT_FIELDS = {
    "artifact_kind",
    "schema_version",
    "production_unlock_eligible",
    "accepted_manifest",
    "plan_hash",
    "source_rows_seen",
    "source_row_identity_root",
    "selected_document_identity_root",
    "train_document_count",
    "validation_document_count",
    "train_character_count",
    "validation_character_count",
    "target_train_characters",
    "target_validation_characters",
    "complete",
    "raw_text_serialized",
    "raw_url_serialized",
    "receipt_hash",
}
_CORPUS_MANIFEST_FIELDS = {
    "artifact_kind",
    "schema_version",
    "production_unlock_eligible",
    "accepted_manifest_unlock_eligible",
    "accepted_manifest",
    "plan_hash",
    "source_snapshot_hash",
    "source_rows_seen",
    "source_row_identity_root",
    "selected_document_identity_root",
    "train_document_count",
    "validation_document_count",
    "train_character_count",
    "validation_character_count",
    "traversal_policy",
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "manifest_hash",
}
_VIEW_FIELDS = {
    "artifact_kind",
    "schema_version",
    "pack_manifest_hash",
    "split",
    "loss_token_count",
    "full_segment_count",
    "final_segment_loss_tokens",
    "finite",
    "repeat_allowed",
    "order_policy",
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "view_hash",
}
_DURABLE_METADATA_FIELDS = {
    "artifact_kind",
    "schema_version",
    "production_unlock_eligible",
    "f1_unlock_eligible",
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
}
_HASH_CHARACTERS = frozenset("0123456789abcdef")


def _require_hash(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HASH_CHARACTERS for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{label} fields are invalid")


def _require_content_hash(
    payload: Mapping[str, Any], hash_field: str, label: str
) -> str:
    value = _require_hash(payload.get(hash_field), f"{label} {hash_field}")
    if value != hash_without(payload, hash_field):
        raise ValueError(f"{label} hash mismatch")
    return value


def _receipt_payload(kind: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_kind": kind,
        "schema_version": 1,
        "production_unlock_eligible": False,
        "f1_unlock_eligible": False,
        **values,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }


@dataclass(frozen=True, slots=True)
class C1TokenizerAdoptionReceiptV1:
    tokenizer_manifest_hash: str
    artifact_sha256: str
    config_hash: str
    corpus_manifest_hash: str
    training_order_root: str
    training_clean_run_count: int
    validation_clean_run_count: int
    actual_vocab_size: int
    validation_used_for_training: bool
    immutable_parent: bool

    def __post_init__(self) -> None:
        for field in (
            "tokenizer_manifest_hash",
            "artifact_sha256",
            "config_hash",
            "corpus_manifest_hash",
            "training_order_root",
        ):
            _require_hash(getattr(self, field), field)
        _require_int(self.training_clean_run_count, "training clean-run count", minimum=1)
        _require_int(self.validation_clean_run_count, "validation clean-run count", minimum=1)
        _require_int(self.actual_vocab_size, "actual vocab size", minimum=1)
        if self.actual_vocab_size != 32_768:
            raise ValueError("adopted C1 tokenizer vocabulary must be 32768")
        if self.validation_used_for_training is not False:
            raise ValueError("adopted C1 tokenizer validation must remain unused")
        if self.immutable_parent is not True:
            raise ValueError("adopted C1 tokenizer must remain immutable")

    @property
    def receipt_hash(self) -> str:
        return stable_json_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return _receipt_payload(
            "twc_c1_tokenizer_adoption_receipt_v1", asdict(self)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_hash"] = stable_json_hash(payload)
        return payload


def adopt_c1_tokenizer(
    *,
    tokenizer_manifest: Mapping[str, Any],
    tokenizer_artifact_bytes: bytes,
    expected_manifest_hash: str,
    expected_corpus_manifest_hash: str,
    expected_config_hash: str,
) -> C1TokenizerAdoptionReceiptV1:
    """Adopt the exact C1 tokenizer after validating bytes and full lineage."""

    if not isinstance(tokenizer_manifest, Mapping):
        raise ValueError("C1 tokenizer manifest must be a mapping")
    _require_exact_keys(tokenizer_manifest, _TOKENIZER_MANIFEST_FIELDS, "C1 tokenizer manifest")
    _require_content_hash(tokenizer_manifest, "manifest_hash", "C1 tokenizer manifest")
    if (
        tokenizer_manifest.get("artifact_kind") != "twc_tokenizer_manifest_v1"
        or type(tokenizer_manifest.get("schema_version")) is not int
        or tokenizer_manifest.get("schema_version") != 1
        or tokenizer_manifest.get("raw_text_serialized") is not False
        or tokenizer_manifest.get("raw_url_serialized") is not False
        or tokenizer_manifest.get("token_serialized") is not False
    ):
        raise ValueError("C1 tokenizer manifest boundary is invalid")
    if not isinstance(tokenizer_artifact_bytes, bytes) or not tokenizer_artifact_bytes:
        raise ValueError("C1 tokenizer artifact bytes must be nonempty bytes")
    artifact_sha256 = hashlib.sha256(tokenizer_artifact_bytes).hexdigest()
    if tokenizer_manifest.get("artifact_sha256") != artifact_sha256:
        raise ValueError("C1 tokenizer artifact bytes do not match the manifest")

    expected_corpus = _require_hash(
        expected_corpus_manifest_hash, "expected C1 corpus manifest hash"
    )
    corpus_hash = _require_hash(
        tokenizer_manifest.get("corpus_manifest_hash"), "C1 tokenizer corpus manifest hash"
    )
    if corpus_hash != expected_corpus:
        raise ValueError("C1 tokenizer corpus binding mismatch")

    config = tokenizer_manifest.get("config")
    if not isinstance(config, Mapping) or set(config) != _TOKENIZER_CONFIG_FIELDS:
        raise ValueError("C1 tokenizer config is invalid")
    config_hash = _require_hash(tokenizer_manifest.get("config_hash"), "C1 tokenizer config hash")
    if config_hash != stable_json_hash(config):
        raise ValueError("C1 tokenizer config hash mismatch")
    if config_hash != _require_hash(expected_config_hash, "expected C1 tokenizer config hash"):
        raise ValueError("C1 tokenizer config differs from the adopted C1 config")
    if (
        type(config.get("vocab_size")) is not int
        or config.get("vocab_size") != 32_768
        or config.get("model_type") != "bpe"
        or config.get("normalizer") != "none"
        or config.get("pre_tokenizer") != "byte_level"
        or config.get("byte_fallback") is not True
        or type(config.get("min_frequency")) is not int
        or config.get("min_frequency") < 1
        or not isinstance(config.get("special_tokens"), list)
        or not config["special_tokens"]
        or len(config["special_tokens"]) != len(set(config["special_tokens"]))
    ):
        raise ValueError("C1 tokenizer config semantics are invalid")

    actual_vocab_size = _require_int(
        tokenizer_manifest.get("actual_vocab_size"), "C1 tokenizer actual vocab size", minimum=1
    )
    if actual_vocab_size != config["vocab_size"]:
        raise ValueError("C1 tokenizer vocabulary differs from its config")
    special_ids = tokenizer_manifest.get("special_token_ids")
    if (
        not isinstance(special_ids, Mapping)
        or set(special_ids) != set(config["special_tokens"])
        or any(type(value) is not int or not 0 <= value < actual_vocab_size for value in special_ids.values())
        or len(set(special_ids.values())) != len(special_ids)
    ):
        raise ValueError("C1 tokenizer special-token mapping is invalid")

    diagnostics = tokenizer_manifest.get("diagnostics")
    chars_per_token = diagnostics.get("chars_per_token") if isinstance(diagnostics, Mapping) else None
    if (
        not isinstance(diagnostics, Mapping)
        or set(diagnostics) != _TOKENIZER_DIAGNOSTIC_FIELDS
        or diagnostics.get("unknown_token_count") != 0
        or type(diagnostics.get("unknown_token_count")) is not int
        or diagnostics.get("roundtrip_failure_count") != 0
        or type(diagnostics.get("roundtrip_failure_count")) is not int
        or diagnostics.get("deterministic_rebuild") is not True
        or isinstance(chars_per_token, bool)
        or not isinstance(chars_per_token, (int, float))
        or not math.isfinite(chars_per_token)
        or chars_per_token <= 0
        or diagnostics.get("vocab_inspection_passed") is not True
        or diagnostics.get("forbidden_whole_token_count") != 0
        or type(diagnostics.get("forbidden_whole_token_count")) is not int
        or diagnostics.get("ready") is not True
    ):
        raise ValueError("C1 tokenizer diagnostics are not ready")
    if tokenizer_manifest.get("validation_used_for_training") is not False:
        raise ValueError("C1 tokenizer validation boundary is invalid")

    manifest_hash = _require_hash(
        tokenizer_manifest.get("manifest_hash"), "C1 tokenizer manifest hash"
    )
    if manifest_hash != _require_hash(expected_manifest_hash, "expected C1 tokenizer manifest hash"):
        raise ValueError("C1 tokenizer manifest differs from the immutable adopted manifest")
    training_count = _require_int(
        tokenizer_manifest.get("training_clean_run_count"),
        "C1 tokenizer training clean-run count",
        minimum=1,
    )
    validation_count = _require_int(
        tokenizer_manifest.get("validation_clean_run_count"),
        "C1 tokenizer validation clean-run count",
        minimum=1,
    )
    training_order_root = _require_hash(
        tokenizer_manifest.get("training_order_root"), "C1 tokenizer training order root"
    )
    return C1TokenizerAdoptionReceiptV1(
        tokenizer_manifest_hash=manifest_hash,
        artifact_sha256=artifact_sha256,
        config_hash=config_hash,
        corpus_manifest_hash=corpus_hash,
        training_order_root=training_order_root,
        training_clean_run_count=training_count,
        validation_clean_run_count=validation_count,
        actual_vocab_size=actual_vocab_size,
        validation_used_for_training=False,
        immutable_parent=True,
    )


@dataclass(frozen=True, slots=True)
class ParentSpoolExtensionPlanV1:
    parent_spool_receipt_hash: str
    parent_corpus_manifest_hash: str
    parent_plan_hash: str
    source_snapshot_hash: str
    extension_start_source_row: int
    parent_source_row_identity_root: str
    parent_selected_document_identity_root: str
    parent_component_identity_index_hash: str
    parent_component_identity_count: int
    parent_train_document_count: int
    parent_validation_document_count: int
    parent_train_character_count: int
    parent_validation_character_count: int
    target_train_character_count: int
    minimum_composite_train_loss_tokens: int
    traversal_policy: str
    validation_policy: str
    candidate_exclusion_policy: str
    parent_root_mutation_allowed: bool
    extension_emits_validation_documents: bool

    def __post_init__(self) -> None:
        for field in (
            "parent_spool_receipt_hash",
            "parent_corpus_manifest_hash",
            "parent_plan_hash",
            "source_snapshot_hash",
            "parent_source_row_identity_root",
            "parent_selected_document_identity_root",
            "parent_component_identity_index_hash",
        ):
            _require_hash(getattr(self, field), field)
        for field in (
            "extension_start_source_row",
            "parent_component_identity_count",
            "parent_train_document_count",
            "parent_validation_document_count",
            "parent_train_character_count",
            "parent_validation_character_count",
            "target_train_character_count",
            "minimum_composite_train_loss_tokens",
        ):
            _require_int(getattr(self, field), field, minimum=1)
        if self.parent_component_identity_count != (
            self.parent_train_document_count + self.parent_validation_document_count
        ):
            raise ValueError("parent component identity count is incomplete")
        if self.target_train_character_count <= self.parent_train_character_count:
            raise ValueError("target train character count must increase the parent")
        if self.minimum_composite_train_loss_tokens < C3_TRAIN_LOSS_TOKENS:
            raise ValueError("composite train capacity must cover the exact C3 target")
        if self.traversal_policy != "pinned-parquet-row-group-round-robin-v1":
            raise ValueError("parent extension traversal policy is invalid")
        if self.validation_policy != "reuse-parent-validation-only-v1":
            raise ValueError("parent extension validation policy is invalid")
        if self.candidate_exclusion_policy != "exclude-parent-components-v1":
            raise ValueError("parent extension candidate exclusion policy is invalid")
        if self.parent_root_mutation_allowed is not False:
            raise ValueError("parent extension must not allow parent mutation")
        if self.extension_emits_validation_documents is not False:
            raise ValueError("parent extension must preserve parent-only validation")

    @property
    def receipt_hash(self) -> str:
        return stable_json_hash(self._payload())

    @property
    def identity(self) -> str:
        return self.receipt_hash

    def _payload(self) -> dict[str, Any]:
        return _receipt_payload("twc_parent_spool_extension_plan_v1", asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_hash"] = stable_json_hash(payload)
        return payload


def _validate_parent_spool_receipt(payload: Mapping[str, Any]) -> str:
    _require_exact_keys(payload, _SPOOL_RECEIPT_FIELDS, "parent spool receipt")
    receipt_hash = _require_content_hash(payload, "receipt_hash", "parent spool receipt")
    if (
        payload.get("artifact_kind") != "twc_configuration_proxy_spool_receipt_v1"
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
        or payload.get("accepted_manifest") is not False
        or payload.get("raw_text_serialized") is not False
        or payload.get("raw_url_serialized") is not False
    ):
        raise ValueError("parent spool receipt boundary is invalid")
    for field in (
        "source_rows_seen",
        "train_document_count",
        "validation_document_count",
        "train_character_count",
        "validation_character_count",
        "target_train_characters",
        "target_validation_characters",
    ):
        _require_int(payload.get(field), f"parent spool {field}", minimum=1)
    for field in (
        "plan_hash",
        "source_row_identity_root",
        "selected_document_identity_root",
    ):
        _require_hash(payload.get(field), f"parent spool {field}")
    if payload.get("complete") is not True:
        raise ValueError("parent spool must be complete before adoption")
    if (
        payload["source_rows_seen"]
        < payload["train_document_count"] + payload["validation_document_count"]
        or payload["train_character_count"] < payload["target_train_characters"]
        or payload["validation_character_count"] < payload["target_validation_characters"]
    ):
        raise ValueError("parent spool completed counts are inconsistent")
    return receipt_hash


def _validate_parent_corpus_manifest(payload: Mapping[str, Any]) -> str:
    _require_exact_keys(payload, _CORPUS_MANIFEST_FIELDS, "parent corpus manifest")
    manifest_hash = _require_content_hash(payload, "manifest_hash", "parent corpus manifest")
    if (
        payload.get("artifact_kind") != "twc_configuration_proxy_corpus_manifest_v1"
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
        or payload.get("accepted_manifest_unlock_eligible") is not False
        or payload.get("accepted_manifest") is not False
        or payload.get("raw_text_serialized") is not False
        or payload.get("raw_url_serialized") is not False
        or payload.get("token_serialized") is not False
    ):
        raise ValueError("parent corpus manifest boundary is invalid")
    for field in (
        "source_rows_seen",
        "train_document_count",
        "validation_document_count",
        "train_character_count",
        "validation_character_count",
    ):
        _require_int(payload.get(field), f"parent corpus {field}", minimum=1)
    for field in (
        "plan_hash",
        "source_snapshot_hash",
        "source_row_identity_root",
        "selected_document_identity_root",
    ):
        _require_hash(payload.get(field), f"parent corpus {field}")
    if payload.get("traversal_policy") != "pinned-parquet-row-group-round-robin-v1":
        raise ValueError("parent corpus traversal policy is invalid")
    return manifest_hash


def plan_parent_spool_extension(
    *,
    parent_spool_receipt: Mapping[str, Any],
    parent_corpus_manifest: Mapping[str, Any],
    expected_parent_spool_receipt_hash: str,
    expected_parent_corpus_manifest_hash: str,
    parent_component_identity_index_hash: str,
    parent_component_identity_count: int,
    target_train_character_count: int,
    minimum_composite_train_loss_tokens: int,
) -> ParentSpoolExtensionPlanV1:
    """Bind a new append-only extension without granting parent write authority."""

    if not isinstance(parent_spool_receipt, Mapping) or not isinstance(
        parent_corpus_manifest, Mapping
    ):
        raise ValueError("parent spool and corpus artifacts must be mappings")
    spool_hash = _validate_parent_spool_receipt(parent_spool_receipt)
    corpus_hash = _validate_parent_corpus_manifest(parent_corpus_manifest)
    for field in (
        "plan_hash",
        "source_rows_seen",
        "source_row_identity_root",
        "selected_document_identity_root",
        "train_document_count",
        "train_character_count",
    ):
        if parent_spool_receipt[field] != parent_corpus_manifest[field]:
            label = field.replace("_", "-")
            raise ValueError(f"parent spool/corpus {label} mismatch")
    for field in (
        "validation_document_count",
        "validation_character_count",
    ):
        if parent_spool_receipt[field] != parent_corpus_manifest[field]:
            raise ValueError("parent spool/corpus validation accounting mismatch")
    if spool_hash != _require_hash(
        expected_parent_spool_receipt_hash, "expected parent spool receipt hash"
    ):
        raise ValueError("parent spool receipt differs from the immutable C1 parent")
    if corpus_hash != _require_hash(
        expected_parent_corpus_manifest_hash, "expected parent corpus manifest hash"
    ):
        raise ValueError("parent corpus manifest differs from the immutable C1 parent")

    component_count = _require_int(
        parent_component_identity_count, "parent component identity count", minimum=1
    )
    expected_components = (
        parent_corpus_manifest["train_document_count"]
        + parent_corpus_manifest["validation_document_count"]
    )
    if component_count != expected_components:
        raise ValueError("parent component identity count is incomplete")
    identity_index_hash = _require_hash(
        parent_component_identity_index_hash, "parent component identity index hash"
    )
    target_characters = _require_int(
        target_train_character_count, "target train character count", minimum=1
    )
    if target_characters <= parent_corpus_manifest["train_character_count"]:
        raise ValueError("target train character count must increase the parent")
    minimum_tokens = _require_int(
        minimum_composite_train_loss_tokens,
        "minimum composite train loss tokens",
        minimum=1,
    )
    if minimum_tokens < C3_TRAIN_LOSS_TOKENS:
        raise ValueError("composite train capacity must cover the exact C3 target")

    return ParentSpoolExtensionPlanV1(
        parent_spool_receipt_hash=spool_hash,
        parent_corpus_manifest_hash=corpus_hash,
        parent_plan_hash=parent_corpus_manifest["plan_hash"],
        source_snapshot_hash=parent_corpus_manifest["source_snapshot_hash"],
        extension_start_source_row=parent_corpus_manifest["source_rows_seen"],
        parent_source_row_identity_root=parent_corpus_manifest["source_row_identity_root"],
        parent_selected_document_identity_root=parent_corpus_manifest[
            "selected_document_identity_root"
        ],
        parent_component_identity_index_hash=identity_index_hash,
        parent_component_identity_count=component_count,
        parent_train_document_count=parent_corpus_manifest["train_document_count"],
        parent_validation_document_count=parent_corpus_manifest[
            "validation_document_count"
        ],
        parent_train_character_count=parent_corpus_manifest["train_character_count"],
        parent_validation_character_count=parent_corpus_manifest[
            "validation_character_count"
        ],
        target_train_character_count=target_characters,
        minimum_composite_train_loss_tokens=minimum_tokens,
        traversal_policy=parent_corpus_manifest["traversal_policy"],
        validation_policy="reuse-parent-validation-only-v1",
        candidate_exclusion_policy="exclude-parent-components-v1",
        parent_root_mutation_allowed=False,
        extension_emits_validation_documents=False,
    )


@dataclass(frozen=True, slots=True)
class SequenceSliceDigestV1:
    """No-token digest summary measured by the private pack readback operator.

    ``component_order_root`` must chain stable clean-run and source-component
    identity hashes, not pack-local ordinals: appending train rows legitimately
    shifts the global ordinals of unchanged validation rows.  Likewise,
    ``index_offset_root`` is over view-local logical offsets plus those stable
    identities.  This distinction lets the receipt prove semantic continuity
    across a parent-bound repack without demanding identical shard layout.
    """

    pack_manifest_hash: str
    view_hash: str
    split: str
    loss_token_count: int
    token_stream_sha256: str
    component_order_root: str
    segment_boundary_root: str
    index_offset_root: str
    full_segment_count: int
    final_segment_loss_tokens: int
    pack_readback_receipt_hash: str
    reader_code_hash: str

    def __post_init__(self) -> None:
        for field in (
            "pack_manifest_hash",
            "view_hash",
            "token_stream_sha256",
            "component_order_root",
            "segment_boundary_root",
            "index_offset_root",
            "pack_readback_receipt_hash",
            "reader_code_hash",
        ):
            _require_hash(getattr(self, field), field)
        if self.split not in {"train", "validation"}:
            raise ValueError("sequence digest split must be train or validation")
        _require_int(self.loss_token_count, "sequence digest loss token count", minimum=1)
        _require_int(self.full_segment_count, "sequence digest full segment count")
        final = _require_int(
            self.final_segment_loss_tokens,
            "sequence digest final segment loss tokens",
            minimum=1,
        )
        if final > 4_096:
            raise ValueError("sequence digest final segment exceeds context length")

    @property
    def identity(self) -> str:
        return stable_json_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return _receipt_payload("twc_sequence_slice_digest_v1", asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["digest_hash"] = stable_json_hash(payload)
        return payload


def _validate_pack_manifest(
    payload: Mapping[str, Any], label: str, *, input_artifact_kind: str
) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    manifest_hash = _require_content_hash(payload, "manifest_hash", label)
    if (
        payload.get("artifact_kind") != "twc_isolated_sequence_pack_manifest_v1"
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("input_artifact_kind") != input_artifact_kind
        or type(payload.get("context_length")) is not int
        or payload.get("context_length") != 4_096
        or type(payload.get("vocab_size")) is not int
        or payload.get("vocab_size") != 32_768
        or payload.get("dtype") != "uint16-le"
        or payload.get("index_record") != "<QQIIBBH"
        or payload.get("component_identity_kind") != "source_document_id_hash"
        or payload.get("raw_text_serialized") is not False
        or payload.get("raw_url_serialized") is not False
        or payload.get("token_serialized") is not False
    ):
        raise ValueError(f"{label} boundary is invalid")
    for field in ("build_contract_hash", "tokenizer_manifest_hash", "input_manifest_hash"):
        _require_hash(payload.get(field), f"{label} {field}")
    build_contract = payload.get("build_contract")
    if (
        not isinstance(build_contract, Mapping)
        or stable_json_hash(build_contract) != payload["build_contract_hash"]
        or build_contract.get("artifact_kind")
        != "twc_isolated_sequence_pack_build_contract_v1"
    ):
        raise ValueError(f"{label} build contract is invalid")
    for field in (
        "tokenizer_manifest_hash",
        "input_artifact_kind",
        "input_manifest_hash",
        "context_length",
        "vocab_size",
        "dtype",
        "index_record",
        "component_identity_kind",
    ):
        if build_contract.get(field) != payload.get(field):
            raise ValueError(f"{label} build contract {field} mismatch")
    expected_order = (
        "parent_train_then_extension_train_then_frozen_parent_validation_v1"
        if input_artifact_kind
        == "twc_configuration_composite_corpus_manifest_v1"
        else "upstream_canonical_content_hash_order"
    )
    if build_contract.get("order_policy") != expected_order:
        raise ValueError(f"{label} build contract order policy is invalid")
    if build_contract.get("segmentation_policy") != "isolated_overlap_one_v1":
        raise ValueError(f"{label} build contract segmentation policy is invalid")
    _require_int(
        build_contract.get("target_shard_bytes"),
        f"{label} target shard bytes",
        minimum=2,
    )
    split_counts = payload.get("split_loss_token_counts")
    if not isinstance(split_counts, Mapping) or set(split_counts) != {"train", "validation"}:
        raise ValueError(f"{label} split loss-token counts are invalid")
    for split in ("train", "validation"):
        _require_int(split_counts[split], f"{label} {split} loss tokens", minimum=1)
    return manifest_hash


def _validate_view(
    payload: Mapping[str, Any],
    *,
    label: str,
    pack_manifest_hash: str,
    split: str,
    loss_token_count: int,
) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    _require_exact_keys(payload, _VIEW_FIELDS, label)
    view_hash = _require_content_hash(payload, "view_hash", label)
    if (
        payload.get("artifact_kind") != "twc_finite_sequence_view_v1"
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("pack_manifest_hash") != pack_manifest_hash
        or payload.get("split") != split
        or payload.get("finite") is not True
        or payload.get("repeat_allowed") is not False
        or payload.get("order_policy") != "pack_canonical_order"
        or payload.get("raw_text_serialized") is not False
        or payload.get("raw_url_serialized") is not False
        or payload.get("token_serialized") is not False
    ):
        if payload.get("pack_manifest_hash") != pack_manifest_hash:
            raise ValueError(f"{label} is not bound to the composite pack" if "C2" in label or "C3" in label else f"{label} pack binding mismatch")
        raise ValueError(f"{label} must be a canonical finite nonrepeating view")
    actual_loss = _require_int(payload.get("loss_token_count"), f"{label} loss tokens", minimum=1)
    if actual_loss != loss_token_count:
        raise ValueError(f"{label} must contain exactly {loss_token_count} loss tokens")
    _require_int(payload.get("full_segment_count"), f"{label} full segment count")
    final = _require_int(
        payload.get("final_segment_loss_tokens"),
        f"{label} final segment loss tokens",
        minimum=1,
    )
    if final > 4_096:
        raise ValueError(f"{label} final segment exceeds the pack context length")
    return view_hash


def _validate_digest(
    digest: SequenceSliceDigestV1,
    *,
    view: Mapping[str, Any],
    label: str,
) -> None:
    if not isinstance(digest, SequenceSliceDigestV1):
        raise ValueError(f"{label} must be a SequenceSliceDigestV1")
    if (
        digest.pack_manifest_hash != view["pack_manifest_hash"]
        or digest.view_hash != view["view_hash"]
        or digest.split != view["split"]
        or digest.loss_token_count != view["loss_token_count"]
    ):
        raise ValueError(f"{label} is not bound to its exact view")
    if digest.full_segment_count != view["full_segment_count"]:
        raise ValueError(f"{label} segment count differs from its view")
    if digest.final_segment_loss_tokens != view["final_segment_loss_tokens"]:
        raise ValueError(f"{label} final segment differs from its view")


@dataclass(frozen=True, slots=True)
class C1PackContinuityReceiptV1:
    tokenizer_adoption_receipt_hash: str
    tokenizer_manifest_hash: str
    parent_pack_manifest_hash: str
    composite_pack_manifest_hash: str
    parent_train_view_hash: str
    composite_train_prefix_view_hash: str
    parent_validation_view_hash: str
    composite_validation_view_hash: str
    parent_train_readback_receipt_hash: str
    composite_train_readback_receipt_hash: str
    parent_validation_readback_receipt_hash: str
    composite_validation_readback_receipt_hash: str
    reader_code_hash: str
    c1_train_loss_tokens: int
    validation_loss_tokens: int
    composite_train_capacity_loss_tokens: int
    validation_split_capacity_loss_tokens: int
    c1_train_full_segment_count: int
    c1_train_final_segment_loss_tokens: int
    train_token_stream_sha256: str
    train_component_order_root: str
    train_segment_boundary_root: str
    train_index_offset_root: str
    validation_token_stream_sha256: str
    validation_component_order_root: str
    validation_segment_boundary_root: str
    validation_index_offset_root: str
    exact_parent_train_prefix: bool
    validation_semantically_identical: bool

    def __post_init__(self) -> None:
        for field in (
            "tokenizer_adoption_receipt_hash",
            "tokenizer_manifest_hash",
            "parent_pack_manifest_hash",
            "composite_pack_manifest_hash",
            "parent_train_view_hash",
            "composite_train_prefix_view_hash",
            "parent_validation_view_hash",
            "composite_validation_view_hash",
            "parent_train_readback_receipt_hash",
            "composite_train_readback_receipt_hash",
            "parent_validation_readback_receipt_hash",
            "composite_validation_readback_receipt_hash",
            "reader_code_hash",
            "train_token_stream_sha256",
            "train_component_order_root",
            "train_segment_boundary_root",
            "train_index_offset_root",
            "validation_token_stream_sha256",
            "validation_component_order_root",
            "validation_segment_boundary_root",
            "validation_index_offset_root",
        ):
            _require_hash(getattr(self, field), field)
        _require_int(self.c1_train_loss_tokens, "C1 train loss tokens", minimum=1)
        _require_int(self.validation_loss_tokens, "validation loss tokens", minimum=1)
        _require_int(
            self.composite_train_capacity_loss_tokens,
            "composite train capacity loss tokens",
            minimum=1,
        )
        if self.c1_train_loss_tokens != C1_TRAIN_PREFIX_LOSS_TOKENS:
            raise ValueError("continuity receipt C1 train target is invalid")
        if self.validation_loss_tokens != FROZEN_VALIDATION_LOSS_TOKENS:
            raise ValueError("continuity receipt validation target is invalid")
        if self.composite_train_capacity_loss_tokens < C3_TRAIN_LOSS_TOKENS:
            raise ValueError("continuity receipt composite capacity is insufficient")
        _require_int(
            self.validation_split_capacity_loss_tokens,
            "validation split capacity",
            minimum=FROZEN_VALIDATION_LOSS_TOKENS,
        )
        _require_int(self.c1_train_full_segment_count, "C1 train full segment count")
        _require_int(
            self.c1_train_final_segment_loss_tokens,
            "C1 train final segment loss tokens",
            minimum=1,
        )
        if self.c1_train_final_segment_loss_tokens > 4_096:
            raise ValueError("C1 train final segment exceeds context length")
        if self.exact_parent_train_prefix is not True:
            raise ValueError("continuity receipt must prove the exact parent train prefix")
        if self.validation_semantically_identical is not True:
            raise ValueError("continuity receipt must prove identical validation semantics")

    @property
    def receipt_hash(self) -> str:
        return stable_json_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return _receipt_payload("twc_c1_pack_continuity_receipt_v1", asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_hash"] = stable_json_hash(payload)
        return payload


def _require_equal_digest_dimension(
    parent: SequenceSliceDigestV1,
    composite: SequenceSliceDigestV1,
    field: str,
    label: str,
) -> None:
    if getattr(parent, field) != getattr(composite, field):
        raise ValueError(f"C1 {label} mismatch between parent and composite")


def prove_c1_pack_continuity(
    *,
    tokenizer_adoption: C1TokenizerAdoptionReceiptV1,
    parent_pack_manifest: Mapping[str, Any],
    composite_pack_manifest: Mapping[str, Any],
    expected_parent_pack_manifest_hash: str,
    expected_parent_train_view_hash: str,
    expected_parent_validation_view_hash: str,
    parent_train_view: Mapping[str, Any],
    composite_train_prefix_view: Mapping[str, Any],
    parent_validation_view: Mapping[str, Any],
    composite_validation_view: Mapping[str, Any],
    parent_train_digest: SequenceSliceDigestV1,
    composite_train_digest: SequenceSliceDigestV1,
    parent_validation_digest: SequenceSliceDigestV1,
    composite_validation_digest: SequenceSliceDigestV1,
) -> C1PackContinuityReceiptV1:
    """Prove exact C1 train-prefix and frozen-validation continuity."""

    if not isinstance(tokenizer_adoption, C1TokenizerAdoptionReceiptV1):
        raise ValueError("C1 tokenizer adoption receipt is required")
    parent_pack_hash = _validate_pack_manifest(
        parent_pack_manifest,
        "parent pack manifest",
        input_artifact_kind="twc_configuration_proxy_corpus_manifest_v1",
    )
    composite_pack_hash = _validate_pack_manifest(
        composite_pack_manifest,
        "composite pack manifest",
        input_artifact_kind="twc_configuration_composite_corpus_manifest_v1",
    )
    if parent_pack_hash != _require_hash(
        expected_parent_pack_manifest_hash, "expected immutable C1 pack manifest hash"
    ):
        raise ValueError("parent pack differs from the immutable C1 pack")
    for label, pack in (
        ("parent", parent_pack_manifest),
        ("composite", composite_pack_manifest),
    ):
        if pack.get("tokenizer_manifest_hash") != tokenizer_adoption.tokenizer_manifest_hash:
            raise ValueError(f"{label} pack tokenizer binding differs from adopted C1 tokenizer")
    parent_counts = parent_pack_manifest["split_loss_token_counts"]
    composite_counts = composite_pack_manifest["split_loss_token_counts"]
    if parent_counts["train"] < C1_TRAIN_PREFIX_LOSS_TOKENS:
        raise ValueError("parent pack cannot cover the exact C1 train prefix")
    if composite_counts["train"] < C3_TRAIN_LOSS_TOKENS:
        raise ValueError("composite pack cannot cover the exact C3 target")
    if parent_counts["validation"] != composite_counts["validation"]:
        raise ValueError("parent/composite validation split capacity differs")
    if parent_counts["validation"] < FROZEN_VALIDATION_LOSS_TOKENS:
        raise ValueError("parent validation split cannot cover the frozen validation view")

    _validate_view(
        parent_train_view,
        label="parent C1 train view",
        pack_manifest_hash=parent_pack_hash,
        split="train",
        loss_token_count=C1_TRAIN_PREFIX_LOSS_TOKENS,
    )
    _validate_view(
        composite_train_prefix_view,
        label="composite C1 train-prefix view",
        pack_manifest_hash=composite_pack_hash,
        split="train",
        loss_token_count=C1_TRAIN_PREFIX_LOSS_TOKENS,
    )
    _validate_view(
        parent_validation_view,
        label="parent frozen validation view",
        pack_manifest_hash=parent_pack_hash,
        split="validation",
        loss_token_count=FROZEN_VALIDATION_LOSS_TOKENS,
    )
    _validate_view(
        composite_validation_view,
        label="composite frozen validation view",
        pack_manifest_hash=composite_pack_hash,
        split="validation",
        loss_token_count=FROZEN_VALIDATION_LOSS_TOKENS,
    )
    if parent_train_view["view_hash"] != _require_hash(
        expected_parent_train_view_hash, "expected immutable C1 train view hash"
    ):
        raise ValueError("parent train view differs from the immutable C1 train view")
    if parent_validation_view["view_hash"] != _require_hash(
        expected_parent_validation_view_hash,
        "expected immutable C1 validation view hash",
    ):
        raise ValueError("parent validation view differs from the immutable C1 validation view")
    digest_pairs = (
        (parent_train_digest, parent_train_view, "parent train digest"),
        (composite_train_digest, composite_train_prefix_view, "composite train digest"),
        (parent_validation_digest, parent_validation_view, "parent validation digest"),
        (
            composite_validation_digest,
            composite_validation_view,
            "composite validation digest",
        ),
    )
    for digest, view, label in digest_pairs:
        _validate_digest(digest, view=view, label=label)
    reader_code_hashes = {digest.reader_code_hash for digest, _, _ in digest_pairs}
    if len(reader_code_hashes) != 1:
        raise ValueError("continuity digests must use one reader code identity")

    for parent, composite in (
        (parent_train_digest, composite_train_digest),
        (parent_validation_digest, composite_validation_digest),
    ):
        for field, label in (
            ("token_stream_sha256", "token stream"),
            ("component_order_root", "component order"),
            ("segment_boundary_root", "segment boundary"),
            ("index_offset_root", "index offset"),
            ("full_segment_count", "segment count"),
            ("final_segment_loss_tokens", "final segment"),
        ):
            _require_equal_digest_dimension(parent, composite, field, label)

    receipt = C1PackContinuityReceiptV1(
        tokenizer_adoption_receipt_hash=tokenizer_adoption.receipt_hash,
        tokenizer_manifest_hash=tokenizer_adoption.tokenizer_manifest_hash,
        parent_pack_manifest_hash=parent_pack_hash,
        composite_pack_manifest_hash=composite_pack_hash,
        parent_train_view_hash=parent_train_view["view_hash"],
        composite_train_prefix_view_hash=composite_train_prefix_view["view_hash"],
        parent_validation_view_hash=parent_validation_view["view_hash"],
        composite_validation_view_hash=composite_validation_view["view_hash"],
        parent_train_readback_receipt_hash=parent_train_digest.pack_readback_receipt_hash,
        composite_train_readback_receipt_hash=composite_train_digest.pack_readback_receipt_hash,
        parent_validation_readback_receipt_hash=parent_validation_digest.pack_readback_receipt_hash,
        composite_validation_readback_receipt_hash=composite_validation_digest.pack_readback_receipt_hash,
        reader_code_hash=parent_train_digest.reader_code_hash,
        c1_train_loss_tokens=C1_TRAIN_PREFIX_LOSS_TOKENS,
        validation_loss_tokens=FROZEN_VALIDATION_LOSS_TOKENS,
        composite_train_capacity_loss_tokens=composite_counts["train"],
        validation_split_capacity_loss_tokens=composite_counts["validation"],
        c1_train_full_segment_count=parent_train_digest.full_segment_count,
        c1_train_final_segment_loss_tokens=parent_train_digest.final_segment_loss_tokens,
        train_token_stream_sha256=parent_train_digest.token_stream_sha256,
        train_component_order_root=parent_train_digest.component_order_root,
        train_segment_boundary_root=parent_train_digest.segment_boundary_root,
        train_index_offset_root=parent_train_digest.index_offset_root,
        validation_token_stream_sha256=parent_validation_digest.token_stream_sha256,
        validation_component_order_root=parent_validation_digest.component_order_root,
        validation_segment_boundary_root=parent_validation_digest.segment_boundary_root,
        validation_index_offset_root=parent_validation_digest.index_offset_root,
        exact_parent_train_prefix=True,
        validation_semantically_identical=True,
    )
    validate_public_no_raw_payload(receipt.to_dict())
    return receipt


@dataclass(frozen=True, slots=True)
class NestedCompositeViewsReceiptV1:
    continuity_receipt_hash: str
    composite_pack_manifest_hash: str
    train_view_hashes: tuple[str, str, str]
    train_loss_token_counts: tuple[int, int, int]
    validation_view_hash: str
    order_policy: str
    finite: bool
    repeat_allowed: bool
    c1_is_exact_composite_prefix: bool
    c2_is_exact_prefix_of_c3: bool

    def __post_init__(self) -> None:
        _require_hash(self.continuity_receipt_hash, "continuity_receipt_hash")
        _require_hash(self.composite_pack_manifest_hash, "composite_pack_manifest_hash")
        if len(self.train_view_hashes) != 3:
            raise ValueError("nested receipt requires exactly three train view hashes")
        for index, value in enumerate(self.train_view_hashes):
            _require_hash(value, f"train_view_hashes[{index}]")
        for index, value in enumerate(self.train_loss_token_counts):
            _require_int(value, f"train_loss_token_counts[{index}]", minimum=1)
        if self.train_loss_token_counts != (
            C1_TRAIN_PREFIX_LOSS_TOKENS,
            C2_TRAIN_LOSS_TOKENS,
            C3_TRAIN_LOSS_TOKENS,
        ):
            raise ValueError("nested receipt train loss-token targets are invalid")
        _require_hash(self.validation_view_hash, "validation_view_hash")
        if self.order_policy != "pack_canonical_order" or self.finite is not True:
            raise ValueError("nested receipt requires canonical finite views")
        if self.repeat_allowed is not False:
            raise ValueError("nested receipt requires nonrepeating views")
        if (
            self.c1_is_exact_composite_prefix is not True
            or self.c2_is_exact_prefix_of_c3 is not True
        ):
            raise ValueError("nested receipt must prove exact prefix nesting")

    @property
    def receipt_hash(self) -> str:
        return stable_json_hash(self._payload())

    def _payload(self) -> dict[str, Any]:
        return _receipt_payload("twc_nested_composite_views_receipt_v1", asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_hash"] = stable_json_hash(payload)
        return payload


def prove_nested_c2_c3_views(
    *,
    continuity: C1PackContinuityReceiptV1,
    c2_train_view: Mapping[str, Any],
    c3_train_view: Mapping[str, Any],
) -> NestedCompositeViewsReceiptV1:
    """Bind exact C1/C2/C3 canonical prefixes of one validated composite pack."""

    if not isinstance(continuity, C1PackContinuityReceiptV1):
        raise ValueError("C1 pack continuity receipt is required")
    _validate_view(
        c2_train_view,
        label="C2 train view",
        pack_manifest_hash=continuity.composite_pack_manifest_hash,
        split="train",
        loss_token_count=C2_TRAIN_LOSS_TOKENS,
    )
    _validate_view(
        c3_train_view,
        label="C3 train view",
        pack_manifest_hash=continuity.composite_pack_manifest_hash,
        split="train",
        loss_token_count=C3_TRAIN_LOSS_TOKENS,
    )
    boundaries = (
        (
            continuity.c1_train_full_segment_count,
            continuity.c1_train_final_segment_loss_tokens,
        ),
        (
            c2_train_view["full_segment_count"],
            c2_train_view["final_segment_loss_tokens"],
        ),
        (
            c3_train_view["full_segment_count"],
            c3_train_view["final_segment_loss_tokens"],
        ),
    )
    if not boundaries[0] < boundaries[1] < boundaries[2]:
        raise ValueError("C1/C2/C3 nested boundary order is invalid")
    receipt = NestedCompositeViewsReceiptV1(
        continuity_receipt_hash=continuity.receipt_hash,
        composite_pack_manifest_hash=continuity.composite_pack_manifest_hash,
        train_view_hashes=(
            continuity.composite_train_prefix_view_hash,
            c2_train_view["view_hash"],
            c3_train_view["view_hash"],
        ),
        train_loss_token_counts=(
            C1_TRAIN_PREFIX_LOSS_TOKENS,
            C2_TRAIN_LOSS_TOKENS,
            C3_TRAIN_LOSS_TOKENS,
        ),
        validation_view_hash=continuity.composite_validation_view_hash,
        order_policy="pack_canonical_order",
        finite=True,
        repeat_allowed=False,
        c1_is_exact_composite_prefix=True,
        c2_is_exact_prefix_of_c3=True,
    )
    validate_public_no_raw_payload(receipt.to_dict())
    return receipt


def _load_durable_values(
    payload: Mapping[str, Any],
    *,
    kind: str,
    value_fields: set[str],
    hash_field: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    expected = _DURABLE_METADATA_FIELDS | value_fields | {hash_field}
    _require_exact_keys(payload, expected, label)
    _require_content_hash(payload, hash_field, label)
    if (
        payload.get("artifact_kind") != kind
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
        or payload.get("f1_unlock_eligible") is not False
        or payload.get("raw_text_serialized") is not False
        or payload.get("raw_url_serialized") is not False
        or payload.get("token_serialized") is not False
    ):
        raise ValueError(f"{label} boundary is invalid")
    validate_public_no_raw_payload(payload)
    return {field: payload[field] for field in value_fields}


def load_c1_tokenizer_adoption_receipt(
    payload: Mapping[str, Any],
) -> C1TokenizerAdoptionReceiptV1:
    """Strictly load a durable C1 tokenizer adoption receipt."""

    fields = set(C1TokenizerAdoptionReceiptV1.__dataclass_fields__)
    values = _load_durable_values(
        payload,
        kind="twc_c1_tokenizer_adoption_receipt_v1",
        value_fields=fields,
        hash_field="receipt_hash",
        label="C1 tokenizer adoption receipt",
    )
    receipt = C1TokenizerAdoptionReceiptV1(**values)
    if receipt.receipt_hash != payload["receipt_hash"]:
        raise ValueError("C1 tokenizer adoption receipt hash mismatch")
    return receipt


def load_parent_spool_extension_plan(
    payload: Mapping[str, Any],
) -> ParentSpoolExtensionPlanV1:
    """Strictly load a durable parent-bound extension plan."""

    fields = set(ParentSpoolExtensionPlanV1.__dataclass_fields__)
    values = _load_durable_values(
        payload,
        kind="twc_parent_spool_extension_plan_v1",
        value_fields=fields,
        hash_field="receipt_hash",
        label="parent spool extension plan",
    )
    plan = ParentSpoolExtensionPlanV1(**values)
    if plan.receipt_hash != payload["receipt_hash"]:
        raise ValueError("parent spool extension plan hash mismatch")
    return plan


def load_sequence_slice_digest(payload: Mapping[str, Any]) -> SequenceSliceDigestV1:
    """Strictly load a no-token sequence-slice digest."""

    fields = set(SequenceSliceDigestV1.__dataclass_fields__)
    values = _load_durable_values(
        payload,
        kind="twc_sequence_slice_digest_v1",
        value_fields=fields,
        hash_field="digest_hash",
        label="sequence slice digest",
    )
    digest = SequenceSliceDigestV1(**values)
    if digest.identity != payload["digest_hash"]:
        raise ValueError("sequence slice digest hash mismatch")
    return digest


def load_c1_pack_continuity_receipt(
    payload: Mapping[str, Any],
) -> C1PackContinuityReceiptV1:
    """Strictly load a durable C1 pack-continuity receipt."""

    fields = set(C1PackContinuityReceiptV1.__dataclass_fields__)
    values = _load_durable_values(
        payload,
        kind="twc_c1_pack_continuity_receipt_v1",
        value_fields=fields,
        hash_field="receipt_hash",
        label="C1 pack continuity receipt",
    )
    receipt = C1PackContinuityReceiptV1(**values)
    if receipt.receipt_hash != payload["receipt_hash"]:
        raise ValueError("C1 pack continuity receipt hash mismatch")
    return receipt


def load_nested_composite_views_receipt(
    payload: Mapping[str, Any],
) -> NestedCompositeViewsReceiptV1:
    """Strictly load the exact nested C1/C2/C3 finite-view receipt."""

    fields = set(NestedCompositeViewsReceiptV1.__dataclass_fields__)
    values = _load_durable_values(
        payload,
        kind="twc_nested_composite_views_receipt_v1",
        value_fields=fields,
        hash_field="receipt_hash",
        label="nested composite views receipt",
    )
    hashes = values.get("train_view_hashes")
    counts = values.get("train_loss_token_counts")
    if not isinstance(hashes, (list, tuple)) or not isinstance(counts, (list, tuple)):
        raise ValueError("nested composite views receipt tuple fields are invalid")
    values["train_view_hashes"] = tuple(hashes)
    values["train_loss_token_counts"] = tuple(counts)
    receipt = NestedCompositeViewsReceiptV1(**values)
    if receipt.receipt_hash != payload["receipt_hash"]:
        raise ValueError("nested composite views receipt hash mismatch")
    return receipt
