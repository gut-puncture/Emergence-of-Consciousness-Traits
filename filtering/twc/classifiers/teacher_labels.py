"""DeepSeek seed-label ingestion for provisional active-learning scoring."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from twc.classifiers.pipeline import load_classifier_config, train_classifier_ensemble
from twc.classifiers.schema import ClassifierPipelineConfig, EnsembleManifest, TrainingExample
from twc.data.decision_schema import stable_json_hash
from twc.data.teacher_labeling import (
    TEACHER_LABEL_SEED_HOLDOUT_TARGET,
    TEACHER_LABEL_SEED_TARGET,
    TEACHER_LABEL_SEED_TRAIN_TARGET,
)


FEATURE_CODE_VERSION = "teacher_seed_redacted_features_v1"
PROVISIONAL_SELECTOR_DECISION_CAPABILITY = "reject_only_provisional_active_learning_selector"
FINAL_TRAINING_REQUIRED_TOTAL_LABELS = 35_000
FINAL_TRAINING_REQUIRED_NON_BORDERLINE_LABELS = 31_000
FINAL_TRAINING_REQUIRED_BORDERLINE_LABELS = 4_000
PROVISIONAL_TRAIN_SPLIT = "provisional_train"
LABELED_HOLDOUT_SPLIT = "labeled_holdout"

FORBIDDEN_INPUT_KEYS = {
    "raw_text",
    "text",
    "document_text",
    "content",
    "raw_content",
    "snippet",
    "raw_context",
    "raw_context_snippet",
    "matched_text",
    "rationale",
    "rationale_short",
    "teacher_rationale",
    "model_raw_output",
    "raw_model_output",
    "response_body",
    "raw_response_body",
    "request_body",
}

FORBIDDEN_PUBLIC_KEYS = FORBIDDEN_INPUT_KEYS | {
    "source_doc_id",
    "document_id",
    "id",
    "risk_spans",
    "url",
    "raw_url",
    "source_url",
    "query",
    "query_string",
}

CLASSIFIER_SPLIT_BY_SEED_SPLIT = {
    PROVISIONAL_TRAIN_SPLIT: "train",
    LABELED_HOLDOUT_SPLIT: "calibration",
}


@dataclass(frozen=True)
class SeedLabelTrainingBundle:
    examples: tuple[TrainingExample, ...]
    manifest: dict[str, Any]
    training_data_manifest_hash: str
    split_counts: dict[str, int]
    teacher_label_counts: dict[str, int]


@dataclass(frozen=True)
class ProvisionalClassifierTrainingResult:
    ensemble: EnsembleManifest
    training_data_manifest_hash: str
    training_data_manifest_path: Path
    artifact_dir: Path
    split_counts: dict[str, int]
    teacher_label_counts: dict[str, int]


def load_seed_label_training_rows(
    path: str | Path,
    *,
    expected_total: int = TEACHER_LABEL_SEED_TARGET,
    expected_train: int = TEACHER_LABEL_SEED_TRAIN_TARGET,
    expected_holdout: int = TEACHER_LABEL_SEED_HOLDOUT_TARGET,
    allow_smoke: bool = False,
) -> SeedLabelTrainingBundle:
    source = Path(path)
    rows = _read_jsonl(source)
    bundle = build_training_examples_from_seed_label_rows(
        rows,
        expected_total=expected_total,
        expected_train=expected_train,
        expected_holdout=expected_holdout,
        allow_smoke=allow_smoke,
        source_label_file_sha256=_file_sha256(source),
    )
    return bundle


def build_training_examples_from_seed_label_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_total: int = TEACHER_LABEL_SEED_TARGET,
    expected_train: int = TEACHER_LABEL_SEED_TRAIN_TARGET,
    expected_holdout: int = TEACHER_LABEL_SEED_HOLDOUT_TARGET,
    allow_smoke: bool = False,
    source_label_file_sha256: str | None = None,
) -> SeedLabelTrainingBundle:
    ordered_rows = sorted((dict(row) for row in rows), key=_seed_selection_sort_key)
    _validate_seed_rows(
        ordered_rows,
        expected_total=expected_total,
        expected_train=expected_train,
        expected_holdout=expected_holdout,
        allow_smoke=allow_smoke,
    )
    examples = tuple(_row_to_training_example(row) for row in ordered_rows)
    split_counts = dict(Counter(example.split for example in examples))
    teacher_label_counts = dict(Counter(str(row["label"]) for row in ordered_rows))
    manifest = _training_data_manifest(
        ordered_rows,
        examples,
        expected_total=expected_total,
        expected_train=expected_train,
        expected_holdout=expected_holdout,
        allow_smoke=allow_smoke,
        source_label_file_sha256=source_label_file_sha256,
    )
    manifest_hash = stable_json_hash(manifest)
    manifest["training_data_manifest_hash"] = manifest_hash
    _validate_public_payload(manifest)
    return SeedLabelTrainingBundle(
        examples=examples,
        manifest=manifest,
        training_data_manifest_hash=manifest_hash,
        split_counts=split_counts,
        teacher_label_counts=teacher_label_counts,
    )


def train_provisional_classifier_from_seed_label_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: ClassifierPipelineConfig,
    artifact_dir: str | Path,
    expected_total: int = TEACHER_LABEL_SEED_TARGET,
    expected_train: int = TEACHER_LABEL_SEED_TRAIN_TARGET,
    expected_holdout: int = TEACHER_LABEL_SEED_HOLDOUT_TARGET,
    allow_smoke: bool = False,
    source_label_file_sha256: str | None = None,
) -> ProvisionalClassifierTrainingResult:
    artifact_path = Path(artifact_dir)
    bundle = build_training_examples_from_seed_label_rows(
        rows,
        expected_total=expected_total,
        expected_train=expected_train,
        expected_holdout=expected_holdout,
        allow_smoke=allow_smoke,
        source_label_file_sha256=source_label_file_sha256,
    )
    artifact_path.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_path / "training_data_manifest.json"
    _write_public_json(manifest_path, bundle.manifest)
    ensemble = train_classifier_ensemble(
        bundle.examples,
        config=config,
        artifact_dir=artifact_path,
        training_data_manifest_hash=bundle.training_data_manifest_hash,
        feature_code_version=FEATURE_CODE_VERSION,
        cost_ledger={
            "budget_cap_usd": config.budget_cap_usd,
            "estimated_usd": 0.0,
            "actual_usd": 0.0,
            "compute_backend": "local_cpu_provisional_classifier",
            "paid_api_used": False,
            "gpu_rental_used": False,
            "notes": "seed_label_ingestion_and_local_cpu_provisional_active_learning_only",
        },
        decision_capability=PROVISIONAL_SELECTOR_DECISION_CAPABILITY,
    )
    _validate_public_artifact_dir(artifact_path)
    return ProvisionalClassifierTrainingResult(
        ensemble=ensemble,
        training_data_manifest_hash=bundle.training_data_manifest_hash,
        training_data_manifest_path=manifest_path,
        artifact_dir=artifact_path,
        split_counts=bundle.split_counts,
        teacher_label_counts=bundle.teacher_label_counts,
    )


def train_provisional_classifier_from_seed_label_jsonl(
    label_jsonl: str | Path,
    *,
    config: ClassifierPipelineConfig,
    artifact_dir: str | Path,
    expected_total: int = TEACHER_LABEL_SEED_TARGET,
    expected_train: int = TEACHER_LABEL_SEED_TRAIN_TARGET,
    expected_holdout: int = TEACHER_LABEL_SEED_HOLDOUT_TARGET,
    allow_smoke: bool = False,
) -> ProvisionalClassifierTrainingResult:
    source = Path(label_jsonl)
    rows = _read_jsonl(source)
    return train_provisional_classifier_from_seed_label_rows(
        rows,
        config=config,
        artifact_dir=artifact_dir,
        expected_total=expected_total,
        expected_train=expected_train,
        expected_holdout=expected_holdout,
        allow_smoke=allow_smoke,
        source_label_file_sha256=_file_sha256(source),
    )


def _validate_seed_rows(
    rows: list[dict[str, Any]],
    *,
    expected_total: int,
    expected_train: int,
    expected_holdout: int,
    allow_smoke: bool,
) -> None:
    if not allow_smoke and len(rows) != TEACHER_LABEL_SEED_TARGET:
        raise ValueError(f"incomplete seed label set: expected exact 6000 rows, found {len(rows)}")
    if len(rows) != expected_total:
        raise ValueError(f"seed label row count mismatch: expected {expected_total}, found {len(rows)}")
    if expected_train + expected_holdout != expected_total:
        raise ValueError("expected train and holdout counts must sum to expected_total")

    seen_candidates: set[str] = set()
    split_indexes: dict[str, set[int]] = {PROVISIONAL_TRAIN_SPLIT: set(), LABELED_HOLDOUT_SPLIT: set()}
    split_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        _reject_forbidden_input(row)
        if row.get("parse_status") != "parsed":
            raise ValueError(f"seed label row {row_index} parse_status must be parsed")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError(f"seed label row {row_index} missing candidate_id")
        if candidate_id in seen_candidates:
            raise ValueError(f"duplicate candidate_id in seed labels: {candidate_id}")
        seen_candidates.add(candidate_id)
        split = str(row.get("seed_split") or "")
        if split not in CLASSIFIER_SPLIT_BY_SEED_SPLIT:
            raise ValueError(f"unsupported seed_split: {split}")
        split_counts[split] += 1
        split_index = _int_field(row, "seed_split_index")
        if split_index in split_indexes[split]:
            raise ValueError(f"duplicate seed_split_index for {split}: {split_index}")
        split_indexes[split].add(split_index)
        _map_teacher_label(row)

    if split_counts[PROVISIONAL_TRAIN_SPLIT] != expected_train:
        raise ValueError(f"expected {expected_train} provisional_train rows, found {split_counts[PROVISIONAL_TRAIN_SPLIT]}")
    if split_counts[LABELED_HOLDOUT_SPLIT] != expected_holdout:
        raise ValueError(f"expected {expected_holdout} labeled_holdout rows, found {split_counts[LABELED_HOLDOUT_SPLIT]}")
    if split_indexes[PROVISIONAL_TRAIN_SPLIT] != set(range(expected_train)):
        raise ValueError("provisional_train seed_split_index values must be contiguous from zero")
    if split_indexes[LABELED_HOLDOUT_SPLIT] != set(range(expected_holdout)):
        raise ValueError("labeled_holdout seed_split_index values must be contiguous from zero")


def _row_to_training_example(row: Mapping[str, Any]) -> TrainingExample:
    text_hash = _sha_field(row, "text_hash")
    source_doc_id = _required_str(row, "source_doc_id")
    normalized_text_hash = str(row.get("normalized_text_sha256") or text_hash)
    record = {
        "schema_version": 1,
        "source_doc_id": source_doc_id,
        "text_sha256": text_hash,
        "normalized_text_sha256": normalized_text_hash,
        "label": _map_teacher_label(row),
        "split": CLASSIFIER_SPLIT_BY_SEED_SPLIT[str(row["seed_split"])],
        "teacher_model": _required_str(row, "teacher_model"),
        "teacher_prompt_version": _required_str(row, "teacher_prompt_version"),
        "confidence": float(row["confidence"]),
        "risk_spans_present": bool(row.get("risk_spans")),
        "features": _hashed_feature_map(row),
        "metadata": {
            "candidate_id_hash": stable_json_hash({"candidate_id": row.get("candidate_id")}),
            "source_identity_hash": _sha_field(row, "source_identity_hash"),
            "seed_candidate_manifest_hash": _sha_field(row, "seed_candidate_manifest_hash"),
            "source_candidate_manifest_hash": _sha_field(row, "source_candidate_manifest_hash"),
            "label_row_hash": str(row.get("label_row_hash") or stable_json_hash(row)),
            "seed_selection_index": _int_field(row, "seed_selection_index"),
            "seed_split_index": _int_field(row, "seed_split_index"),
            "label_unit": _required_str(row, "label_unit"),
            "normalized_text_hash_source": (
                "provided_normalized_text_sha256"
                if row.get("normalized_text_sha256")
                else "same_as_text_hash_no_normalized_text_available"
            ),
        },
    }
    return TrainingExample.from_record(record)


def _map_teacher_label(row: Mapping[str, Any]) -> str:
    label = _required_str(row, "label")
    if row.get("teacher_prompt_version") == "twc_teacher_v3" or int(row.get("schema_version") or 0) >= 2:
        raise ValueError(
            "prompt-v3 teacher labels are not supported by the dormant seed classifier ingestion path"
        )
    risk_categories = _str_list(row, "risk_categories")
    safe_topic_categories = _str_list(row, "safe_topic_categories")
    wrapper_categories = _str_list(row, "wrapper_categories")
    reason_codes = _str_list(row, "reason_codes")
    risk_spans = row.get("risk_spans")
    if risk_spans is None:
        raise ValueError("teacher label row missing risk_spans")
    if not isinstance(risk_spans, list):
        raise ValueError("teacher label row risk_spans must be a list")

    if label == "safe_external_technical":
        if risk_categories:
            raise ValueError("safe teacher label cannot include risk_categories")
        if risk_spans:
            raise ValueError("safe teacher label cannot include risk_spans")
        if not safe_topic_categories:
            raise ValueError("safe teacher label must include safe_topic_categories")
        if bool(row.get("whole_document_rejection")):
            raise ValueError("safe teacher label cannot set whole_document_rejection")
        return "safe_external"

    if label in {"reject", "uncertain_reject"}:
        if row.get("whole_document_rejection") is not True:
            raise ValueError("reject teacher label must set whole_document_rejection")
        fiction_signals = set(risk_categories) | set(wrapper_categories) | set(reason_codes)
        if {"fiction_dialogue", "fiction_or_dialogue", "dialogue"} & fiction_signals:
            return "risk_fiction_dialogue"
        if label == "uncertain_reject":
            return "risk_uncertain_reject"
        return "risk_consciousness"

    raise ValueError(f"unsupported teacher label for classifier mapping: {label}")


def _hashed_feature_map(row: Mapping[str, Any]) -> dict[str, float]:
    features: dict[str, float] = {}
    for field in (
        "label_unit",
        "text_hash_status",
        "prompt_variant",
        "finish_reason",
        "seed_split",
    ):
        value = row.get(field)
        if value not in (None, ""):
            features[_feature_hash(field, str(value))] = 1.0
    for field in (
        "risk_categories",
        "wrapper_categories",
        "safe_topic_categories",
        "evidence_categories",
        "reason_codes",
        "uncertainty_notes",
    ):
        for value in _str_list(row, field):
            features[_feature_hash(field, value)] = 1.0
    text_hash = _sha_field(row, "text_hash")
    source_identity_hash = _sha_field(row, "source_identity_hash")
    for prefix_length in (2, 4):
        features[_feature_hash("text_hash_prefix", text_hash[:prefix_length])] = 1.0
        features[_feature_hash("source_identity_prefix", source_identity_hash[:prefix_length])] = 1.0
    return features


def _training_data_manifest(
    rows: list[dict[str, Any]],
    examples: tuple[TrainingExample, ...],
    *,
    expected_total: int,
    expected_train: int,
    expected_holdout: int,
    allow_smoke: bool,
    source_label_file_sha256: str | None,
) -> dict[str, Any]:
    split_counts = dict(Counter(example.split for example in examples))
    classifier_label_counts = dict(Counter(example.label for example in examples))
    teacher_label_counts = dict(Counter(str(row["label"]) for row in rows))
    manifest = {
        "schema_version": 1,
        "manifest_kind": "provisional_classifier_seed_training_data",
        "classifier_scope": "provisional_active_learning_selector",
        "intended_use": "mine_borderline_4000_candidates_only",
        "eligible_for_final_filtering": False,
        "final_classifier_training_allowed": False,
        "requires_full_35000_reviewed_labels_before_final_training": True,
        "final_training_required_total_labels": FINAL_TRAINING_REQUIRED_TOTAL_LABELS,
        "final_training_required_non_borderline_labels": FINAL_TRAINING_REQUIRED_NON_BORDERLINE_LABELS,
        "final_training_required_borderline_labels": FINAL_TRAINING_REQUIRED_BORDERLINE_LABELS,
        "source_label_set": "deepseek_v4_pro_seed_6000",
        "allow_smoke_fixture": allow_smoke,
        "expected_total": expected_total,
        "expected_provisional_train": expected_train,
        "expected_labeled_holdout": expected_holdout,
        "classifier_split_counts": split_counts,
        "classifier_label_counts": classifier_label_counts,
        "teacher_label_counts": teacher_label_counts,
        "seed_candidate_manifest_hashes": sorted({str(row.get("seed_candidate_manifest_hash")) for row in rows}),
        "source_candidate_manifest_hashes": sorted({str(row.get("source_candidate_manifest_hash")) for row in rows}),
        "source_label_file_sha256": source_label_file_sha256,
        "feature_code_version": FEATURE_CODE_VERSION,
        "fit_split": "train",
        "holdout_split": "calibration",
        "holdout_use": "calibration_and_eval_only",
        "public_artifact_redacted": True,
        "labels_are_corpus_approval": False,
        "accepted_manifests_created": False,
        "training_shards_created": False,
        "examples": [example.redacted_identity() for example in examples],
    }
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(payload)
    return rows


def _write_public_json(path: Path, payload: Mapping[str, Any]) -> None:
    _validate_public_payload(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_public_artifact_dir(path: Path) -> None:
    for artifact in path.rglob("*.json"):
        _validate_public_payload(json.loads(artifact.read_text(encoding="utf-8")))


def _validate_public_payload(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"public classifier artifact contains forbidden key: {path}.{key_text}")
            _validate_public_payload(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_public_payload(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if value.startswith(("http://", "https://")):
            raise ValueError(f"public classifier artifact contains a raw URL at {path}")


def _reject_forbidden_input(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_INPUT_KEYS:
                raise ValueError(f"seed label row contains forbidden field {key_text} at {path}.{key_text}")
            _reject_forbidden_input(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_input(child, path=f"{path}[{index}]")


def _seed_selection_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    value = row.get("seed_selection_index")
    if value is None:
        return (10**12, str(row.get("candidate_id") or ""))
    return (int(value), str(row.get("candidate_id") or ""))


def _feature_hash(namespace: str, value: str) -> str:
    return stable_json_hash({"feature": namespace, "value": value})


def _sha_field(row: Mapping[str, Any], field: str) -> str:
    value = _required_str(row, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase sha256 hex digest")
    return value


def _required_str(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None or str(value) == "":
        raise ValueError(f"teacher label row missing {field}")
    return str(value)


def _str_list(row: Mapping[str, Any], field: str) -> list[str]:
    value = row.get(field)
    if value is None:
        raise ValueError(f"teacher label row missing {field}")
    if not isinstance(value, list):
        raise ValueError(f"teacher label row {field} must be a list")
    return [str(item) for item in value]


def _int_field(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    if value is None:
        raise ValueError(f"teacher label row missing {field}")
    return int(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a provisional active-learning selector from DeepSeek seed labels.")
    parser.add_argument("--labels-jsonl", required=True)
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--artifact-dir", default="reports/classifiers/provisional_active_learning_seed_6000")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--expected-total", type=int, default=TEACHER_LABEL_SEED_TARGET)
    parser.add_argument("--expected-train", type=int, default=TEACHER_LABEL_SEED_TRAIN_TARGET)
    parser.add_argument("--expected-holdout", type=int, default=TEACHER_LABEL_SEED_HOLDOUT_TARGET)
    args = parser.parse_args(argv)

    config = load_classifier_config(args.config)
    try:
        result = train_provisional_classifier_from_seed_label_jsonl(
            args.labels_jsonl,
            config=config,
            artifact_dir=args.artifact_dir,
            expected_total=args.expected_total,
            expected_train=args.expected_train,
            expected_holdout=args.expected_holdout,
            allow_smoke=args.allow_smoke,
        )
    except ValueError as exc:
        print(json.dumps({"run_status": "failed_closed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(
        {
            "artifact_dir": str(result.artifact_dir),
            "classifier_scope": "provisional_active_learning_selector",
            "eligible_for_final_filtering": False,
            "training_data_manifest_hash": result.training_data_manifest_hash,
            "ensemble_sha256": result.ensemble.ensemble_sha256,
            "split_counts": result.split_counts,
            "teacher_label_counts": result.teacher_label_counts,
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
