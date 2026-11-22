"""Schemas and guardrails for local classifier artifacts."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping

from twc.data.decision_schema import stable_json_hash


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
RAW_TEXT_KEYS = {
    "raw_text",
    "text",
    "document_text",
    "normalized_text",
    "snippet",
    "raw_context",
    "raw_context_snippet",
    "matched_text",
    "rationale",
    "rationale_short",
    "risk_spans",
}
ALLOWED_SPLITS = {"train", "calibration", "eval", "audit"}


class ClassifierDecision(StrEnum):
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    PENDING_REQUIRED_GATES = "pending_required_gates"


@dataclass(frozen=True)
class ApprovalPolicy:
    classifiers_may_reject: bool
    classifiers_may_approve_alone: bool
    reject_on_missing_classifier: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ApprovalPolicy":
        required = {
            "classifiers_may_reject",
            "classifiers_may_approve_alone",
            "reject_on_missing_classifier",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"Missing approval policy keys: {missing}")
        policy = cls(
            classifiers_may_reject=bool(payload["classifiers_may_reject"]),
            classifiers_may_approve_alone=bool(payload["classifiers_may_approve_alone"]),
            reject_on_missing_classifier=bool(payload["reject_on_missing_classifier"]),
        )
        if not policy.classifiers_may_reject:
            raise ValueError("Classifier policy must allow rejection")
        if policy.classifiers_may_approve_alone:
            raise ValueError("Classifiers must not approve alone")
        if not policy.reject_on_missing_classifier:
            raise ValueError("Classifier policy must fail closed on missing classifiers")
        return policy


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    kind: str
    target: str
    threshold: float
    threshold_direction: str
    min_calibration_examples: int = 2
    min_positive_examples: int = 1
    min_negative_examples: int = 1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelSpec":
        for key in ("model_id", "kind", "target", "threshold", "threshold_direction"):
            if key not in payload:
                raise ValueError(f"Classifier model spec missing {key}")
        threshold = float(payload["threshold"])
        if not 0 <= threshold <= 1:
            raise ValueError(f"Classifier threshold must be in [0, 1]: {payload['model_id']}")
        direction = str(payload["threshold_direction"])
        if direction not in {"reject_at_or_above", "pass_at_or_above"}:
            raise ValueError(f"Unsupported threshold direction for {payload['model_id']}: {direction}")
        return cls(
            model_id=str(payload["model_id"]),
            kind=str(payload["kind"]),
            target=str(payload["target"]),
            threshold=threshold,
            threshold_direction=direction,
            min_calibration_examples=int(payload.get("min_calibration_examples", 2)),
            min_positive_examples=int(payload.get("min_positive_examples", 1)),
            min_negative_examples=int(payload.get("min_negative_examples", 1)),
        )


@dataclass(frozen=True)
class ClassifierPipelineConfig:
    version: int
    budget_cap_usd: float
    approval_policy: ApprovalPolicy
    model_specs: tuple[ModelSpec, ...]
    config_hash: str
    calibration: dict[str, Any] = field(default_factory=dict)
    artifact_policy: dict[str, Any] = field(default_factory=dict)
    training_methodology: dict[str, Any] = field(default_factory=dict)
    compute_policy: dict[str, Any] = field(default_factory=dict)

    @property
    def required_model_ids(self) -> tuple[str, ...]:
        return tuple(spec.model_id for spec in self.model_specs)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ClassifierPipelineConfig":
        if "budget_cap_usd" not in payload:
            raise ValueError("classifier.yaml must include budget_cap_usd")
        budget_cap_usd = float(payload["budget_cap_usd"])
        if budget_cap_usd <= 0:
            raise ValueError("budget_cap_usd must be positive")
        policy = ApprovalPolicy.from_mapping(payload.get("approval_policy", {}))
        components = payload.get("components")
        if components is None:
            components = _legacy_components(payload.get("models", []))
        specs = tuple(ModelSpec.from_mapping(component) for component in components)
        if not specs:
            raise ValueError("At least one classifier component is required")
        duplicate_ids = _duplicates(spec.model_id for spec in specs)
        if duplicate_ids:
            raise ValueError(f"Duplicate classifier model IDs: {duplicate_ids}")
        canonical_payload = {
            "version": int(payload.get("version", 1)),
            "budget_cap_usd": budget_cap_usd,
            "approval_policy": asdict(policy),
            "components": [asdict(spec) for spec in specs],
            "training_methodology": dict(payload.get("training_methodology", {})),
            "compute_policy": dict(payload.get("compute_policy", {})),
            "calibration": dict(payload.get("calibration", {})),
            "artifact_policy": dict(payload.get("artifact_policy", {})),
        }
        return cls(
            version=canonical_payload["version"],
            budget_cap_usd=budget_cap_usd,
            approval_policy=policy,
            model_specs=specs,
            training_methodology=canonical_payload["training_methodology"],
            compute_policy=canonical_payload["compute_policy"],
            calibration=canonical_payload["calibration"],
            artifact_policy=canonical_payload["artifact_policy"],
            config_hash=stable_json_hash(canonical_payload),
        )


@dataclass(frozen=True)
class TrainingExample:
    schema_version: int
    source_doc_id_hash: str
    text_sha256: str
    normalized_text_sha256: str
    label: str
    split: str
    teacher_model: str
    teacher_prompt_version: str
    confidence: float
    risk_spans_present: bool
    features: dict[str, float] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None
    metadata_hash: str = ""

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "TrainingExample":
        raw_keys = sorted(RAW_TEXT_KEYS & set(record))
        if raw_keys:
            raise ValueError(f"Training record contains raw-text-bearing keys: {raw_keys}")
        required = [
            "schema_version",
            "source_doc_id",
            "text_sha256",
            "normalized_text_sha256",
            "label",
            "split",
            "teacher_model",
            "teacher_prompt_version",
            "confidence",
        ]
        for key in required:
            if key not in record:
                raise ValueError(f"Training record missing {key}")
        text_sha256 = _require_sha256(record["text_sha256"], "text_sha256")
        normalized_text_sha256 = _require_sha256(
            record["normalized_text_sha256"],
            "normalized_text_sha256",
        )
        split = str(record["split"])
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported classifier split: {split}")
        confidence = float(record["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        source_doc_id_hash = stable_json_hash({"source_doc_id": str(record["source_doc_id"])})
        metadata_hash = stable_json_hash(record.get("metadata", {}))
        return cls(
            schema_version=int(record["schema_version"]),
            source_doc_id_hash=source_doc_id_hash,
            text_sha256=text_sha256,
            normalized_text_sha256=normalized_text_sha256,
            label=str(record["label"]),
            split=split,
            teacher_model=str(record["teacher_model"]),
            teacher_prompt_version=str(record["teacher_prompt_version"]),
            confidence=confidence,
            risk_spans_present=bool(record.get("risk_spans_present", False)),
            features=_parse_features(record),
            embedding=_parse_embedding(record),
            metadata_hash=metadata_hash,
        )

    def redacted_identity(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_doc_id_hash": self.source_doc_id_hash,
            "text_sha256": self.text_sha256,
            "normalized_text_sha256": self.normalized_text_sha256,
            "label": self.label,
            "split": self.split,
            "teacher_model": self.teacher_model,
            "teacher_prompt_version": self.teacher_prompt_version,
            "confidence": self.confidence,
            "risk_spans_present": self.risk_spans_present,
            "feature_hashes": sorted(self.features),
            "embedding_dim": len(self.embedding) if self.embedding is not None else 0,
            "metadata_hash": self.metadata_hash,
        }


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    kind: str
    target: str
    threshold: float
    threshold_direction: str
    config_hash: str
    label_set_hash: str
    training_data_manifest_hash: str
    feature_code_version: str
    metrics: dict[str, float | int | str]
    cost_ledger: dict[str, Any]
    model_payload_hash: str
    artifact_sha256: str

    def to_public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnsembleManifest:
    schema_version: int
    decision_capability: str
    approval_policy: ApprovalPolicy
    config_hash: str
    label_set_hash: str
    training_data_manifest_hash: str
    feature_code_version: str
    cost_ledger: dict[str, Any]
    model_artifacts: dict[str, ModelArtifact]
    ensemble_sha256: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_capability": self.decision_capability,
            "approval_policy": asdict(self.approval_policy),
            "config_hash": self.config_hash,
            "label_set_hash": self.label_set_hash,
            "training_data_manifest_hash": self.training_data_manifest_hash,
            "feature_code_version": self.feature_code_version,
            "cost_ledger": self.cost_ledger,
            "model_artifacts": {
                model_id: artifact.to_public_dict()
                for model_id, artifact in sorted(self.model_artifacts.items())
            },
            "ensemble_sha256": self.ensemble_sha256,
        }


@dataclass(frozen=True)
class ClassifierEnsembleDecision:
    status: ClassifierDecision
    scores: dict[str, float]
    rejection_reasons: tuple[str, ...] = ()
    quarantine_reasons: tuple[str, ...] = ()
    pending_reasons: tuple[str, ...] = ()
    all_required_gates_present: bool = False
    decision_capability: str = "reject_only_classifier_ensemble"

    def to_decision_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = str(self.status)
        payload["decision_status"] = str(self.status)
        payload["decision_row_hash"] = stable_json_hash(payload)
        return payload


def validate_training_examples(records: Iterable[Mapping[str, Any]] | Iterable[TrainingExample]) -> tuple[TrainingExample, ...]:
    examples: list[TrainingExample] = []
    for record in records:
        if isinstance(record, TrainingExample):
            examples.append(record)
        else:
            examples.append(TrainingExample.from_record(record))
    if not examples:
        raise ValueError("Classifier training requires at least one training example")
    splits = {example.split for example in examples}
    if "train" not in splits:
        raise ValueError("Classifier training examples must include a train split")
    return tuple(examples)


def hash_label_set(examples: Iterable[TrainingExample]) -> str:
    identities = sorted(
        (example.redacted_identity() for example in examples),
        key=lambda item: (
            item["text_sha256"],
            item["normalized_text_sha256"],
            item["label"],
            item["split"],
        ),
    )
    return stable_json_hash(identities)


def validate_budget_ledger(payload: Mapping[str, Any], *, budget_cap_usd: float) -> dict[str, Any]:
    required = {"budget_cap_usd", "estimated_usd", "actual_usd", "compute_backend"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Budget ledger missing keys: {missing}; budget_cap_usd is required")
    ledger = dict(payload)
    ledger["budget_cap_usd"] = float(ledger["budget_cap_usd"])
    ledger["estimated_usd"] = float(ledger["estimated_usd"])
    ledger["actual_usd"] = float(ledger["actual_usd"])
    if ledger["budget_cap_usd"] != float(budget_cap_usd):
        raise ValueError("Budget ledger cap must match classifier config budget_cap_usd")
    if ledger["estimated_usd"] > budget_cap_usd or ledger["actual_usd"] > budget_cap_usd:
        raise ValueError("Classifier compute estimate or actual spend exceeds budget cap")
    if ledger["estimated_usd"] < 0 or ledger["actual_usd"] < 0:
        raise ValueError("Classifier compute costs must be non-negative")
    ledger.setdefault("paid_api_used", False)
    ledger.setdefault("gpu_rental_used", False)
    ledger.setdefault("notes", "placeholder_or_local_cpu_accounting")
    return ledger


def _legacy_components(model_ids: Iterable[str]) -> list[dict[str, Any]]:
    targets = {
        "hashed_ngram_linear_risk": ("hashed_ngram_linear", "risk_any", "reject_at_or_above", 0.72),
        "modernbert_window_mil_risk": ("transformer_window_mil", "risk_any", "reject_at_or_above", 0.72),
        "lexical_ngram_risk": ("lexical_ngram", "risk_any", "reject_at_or_above", 0.72),
        "fasttext_or_linear_risk": ("hashed_linear", "risk_any", "reject_at_or_above", 0.72),
        "embedding_semantic_risk": ("embedding_centroid", "risk_any", "reject_at_or_above", 0.72),
        "safe_topic": ("hashed_linear", "safe_external", "pass_at_or_above", 0.72),
        "fiction_narrative_dialogue": (
            "hashed_linear",
            "risk_fiction_dialogue",
            "reject_at_or_above",
            0.72,
        ),
    }
    components = []
    for model_id in model_ids:
        if model_id not in targets:
            raise ValueError(f"Unknown classifier model in legacy config: {model_id}")
        kind, target, direction, threshold = targets[model_id]
        components.append(
            {
                "model_id": model_id,
                "kind": kind,
                "target": target,
                "threshold": threshold,
                "threshold_direction": direction,
            }
        )
    return components


def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _require_sha256(value: Any, field_name: str) -> str:
    candidate = str(value)
    if not HASH_RE.fullmatch(candidate):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return candidate


def _parse_features(record: Mapping[str, Any]) -> dict[str, float]:
    if "features" in record:
        raw = record["features"]
        if not isinstance(raw, Mapping):
            raise ValueError("features must be a mapping of hashed feature ID to numeric value")
        return {str(key): float(value) for key, value in raw.items()}
    if "feature_hashes" in record:
        raw_hashes = record["feature_hashes"]
        if not isinstance(raw_hashes, list | tuple):
            raise ValueError("feature_hashes must be a list")
        return {str(value): 1.0 for value in raw_hashes}
    return {}


def _parse_embedding(record: Mapping[str, Any]) -> tuple[float, ...] | None:
    if "embedding" not in record:
        return None
    raw = record["embedding"]
    if not isinstance(raw, list | tuple) or not raw:
        raise ValueError("embedding must be a non-empty list of numbers")
    return tuple(float(value) for value in raw)
