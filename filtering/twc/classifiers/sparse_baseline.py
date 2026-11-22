"""Local CPU hashed n-gram sparse classifier baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twc.classifiers.feature_extraction import collect_final_text_feature_rows
from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.classifiers.metrics import binary_calibration_metrics
from twc.classifiers.pipeline import load_classifier_config
from twc.classifiers.private_text_cache import DEFAULT_CACHE_ROOT, PrivateTextCacheRowProvider
from twc.classifiers.schema import ClassifierPipelineConfig, validate_budget_ledger
from twc.data.decision_log import (
    DEFAULT_HF_ROW_READ_BLOCK_SIZE_BYTES,
    HFFsspecParquetRowProvider,
    LocalParquetRowProvider,
)
from twc.data.decision_schema import stable_json_hash


SPARSE_BASELINE_REPORT_KIND = "hashed_ngram_sparse_baseline_report"
SPARSE_BASELINE_CHECKPOINT_KIND = "hashed_ngram_sparse_baseline_checkpoint"
SPARSE_BASELINE_PROGRESS_KIND = "hashed_ngram_sparse_baseline_progress"
SPARSE_BASELINE_PLAN_KIND = "hashed_ngram_sparse_baseline_training_plan"
FEATURE_CODE_VERSION = "hashed_ngram_sparse_baseline_v1"
MODEL_ID = "hashed_ngram_linear_risk"


def train_hashed_ngram_sparse_baseline(
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    config: ClassifierPipelineConfig,
    artifact_dir: str | Path,
    batch_size: int = 512,
    n_features: int = 2**20,
    fetch_concurrency: int = 1,
    batch_order: str = "deterministic_stratified_shuffle",
    target_recall: float = 0.995,
    random_state: int = 20260608,
    max_train_batches: int | None = None,
) -> dict[str, Any]:
    """Train the local hashed n-gram challenger with durable checkpoints."""

    deps = _load_sparse_dependencies()
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    report_path = artifact_path / f"{MODEL_ID}_report.json"
    progress_path = artifact_path / f"{MODEL_ID}_progress.json"
    checkpoint_path = artifact_path / f"{MODEL_ID}_checkpoint.json"
    model_path = artifact_path / f"{MODEL_ID}_model.joblib"

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if fetch_concurrency < 1:
        raise ValueError("fetch_concurrency must be positive")
    if batch_order not in {"source_file", "manifest", "deterministic_stratified_shuffle"}:
        raise ValueError("batch_order must be source_file, manifest, or deterministic_stratified_shuffle")
    if fetch_concurrency > 1 and bool(getattr(row_provider, "manage_hf_cache_env", False)):
        raise ValueError("parallel sparse-baseline collection requires row_provider.manage_hf_cache_env=false")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")

    _require_component(config)
    rows = list(training_manifest.get("rows") or [])
    training_data_manifest_hash = _training_manifest_hash(training_manifest)
    fit_rows = [row for row in rows if row.get("fit_eligible") is not False and str(row.get("split")) != "audit"]
    audit_row_count = len(rows) - len(fit_rows)
    _validate_fit_rows(fit_rows)

    train_rows = [row for row in fit_rows if row["split"] == "train"]
    calibration_rows = [row for row in fit_rows if row["split"] == "calibration"]
    eval_rows = [row for row in fit_rows if row["split"] == "eval"]
    _validate_split_rows(train_rows, calibration_rows, eval_rows)
    train_rows = _order_rows_for_refetch(
        train_rows,
        candidates=candidates,
        schedule=schedule,
        batch_order=batch_order,
        random_state=random_state,
    )
    calibration_rows = _order_rows_for_refetch(
        calibration_rows,
        candidates=candidates,
        schedule=schedule,
        batch_order=batch_order,
        random_state=random_state,
    )
    eval_rows = _order_rows_for_refetch(
        eval_rows,
        candidates=candidates,
        schedule=schedule,
        batch_order=batch_order,
        random_state=random_state,
    )

    train_batches = _batches(train_rows, batch_size)
    feature_config = _feature_config(n_features=n_features)
    feature_config_hash = stable_json_hash(feature_config)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_sparse_baseline_report(report)
        _validate_existing_report_for_invocation(
            report,
            model_path=model_path,
            training_data_manifest_hash=training_data_manifest_hash,
            config_hash=config.config_hash,
            feature_config_hash=feature_config_hash,
            fit_eligible_row_count=len(fit_rows),
            audit_row_count=audit_row_count,
            split_counts=dict(sorted(Counter(str(row["split"]) for row in fit_rows).items())),
            train_label_counts=dict(sorted(Counter(str(row["classifier_label"]) for row in train_rows).items())),
            calibration_label_counts=dict(sorted(Counter(str(row["classifier_label"]) for row in calibration_rows).items())),
            eval_label_counts=dict(sorted(Counter(str(row["classifier_label"]) for row in eval_rows).items())),
            batch_size=batch_size,
            batch_order=batch_order,
            random_state=random_state,
            target_recall=target_recall,
            total_train_batches=len(train_batches),
        )
        return report
    cost_ledger = validate_budget_ledger(
        {
            "budget_cap_usd": config.budget_cap_usd,
            "estimated_usd": 0.0,
            "actual_usd": 0.0,
            "compute_backend": "local_cpu_sklearn_sgd",
            "paid_api_used": False,
            "gpu_rental_used": False,
            "notes": "local_cpu_sparse_baseline_no_paid_calls_no_gpu",
        },
        budget_cap_usd=config.budget_cap_usd,
    )

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        _validate_checkpoint(
            checkpoint,
            feature_config_hash=feature_config_hash,
            training_data_manifest_hash=training_data_manifest_hash,
            batch_size=batch_size,
            total_train_batches=len(train_batches),
            batch_order=batch_order,
            random_state=random_state,
            model_path=model_path,
        )
        completed_train_batches = int(checkpoint["completed_train_batches"])
        model = deps["joblib"].load(model_path)
    else:
        completed_train_batches = 0
        model = deps["SGDClassifier"](
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            random_state=int(random_state),
            learning_rate="optimal",
            tol=None,
        )

    class_weights = _balanced_sample_weights([_label_to_int(row["classifier_label"]) for row in train_rows])
    max_batches_allowed = len(train_batches)
    if max_train_batches is not None:
        max_batches_allowed = min(max_batches_allowed, int(max_train_batches))

    for batch_index in range(completed_train_batches, max_batches_allowed):
        batch = train_batches[batch_index]
        feature_rows = _collect_feature_rows(
            batch,
            training_manifest=training_manifest,
            candidates=candidates,
            schedule=schedule,
            row_provider=row_provider,
            n_features=n_features,
            fetch_concurrency=fetch_concurrency,
        )
        matrix = _feature_rows_to_csr(feature_rows, n_features=n_features, deps=deps)
        labels = [_label_to_int(row["classifier_label"]) for row in batch]
        sample_weights = [class_weights[label] for label in labels]
        model.partial_fit(
            matrix,
            deps["numpy"].array(labels, dtype=deps["numpy"].int64),
            classes=deps["numpy"].array([0, 1], dtype=deps["numpy"].int64),
            sample_weight=deps["numpy"].array(sample_weights, dtype=deps["numpy"].float64),
        )
        deps["joblib"].dump(model, model_path)
        completed_train_batches = batch_index + 1
        checkpoint = _checkpoint_payload(
            completed_train_batches=completed_train_batches,
            total_train_batches=len(train_batches),
            seen_train_rows=sum(len(batch) for batch in train_batches[:completed_train_batches]),
            training_data_manifest_hash=training_data_manifest_hash,
            feature_config_hash=feature_config_hash,
            batch_size=batch_size,
            batch_order=batch_order,
            random_state=random_state,
            model_file_sha256=_sha256_file(model_path),
        )
        _write_json_atomic(checkpoint_path, checkpoint)
        _write_json_atomic(
            progress_path,
            _progress_payload(
                status="training",
                completed_train_batches=completed_train_batches,
                total_train_batches=len(train_batches),
                fit_eligible_row_count=len(fit_rows),
                audit_row_count=audit_row_count,
            ),
        )

    if completed_train_batches < len(train_batches):
        progress = _progress_payload(
            status="incomplete",
            completed_train_batches=completed_train_batches,
            total_train_batches=len(train_batches),
            fit_eligible_row_count=len(fit_rows),
            audit_row_count=audit_row_count,
        )
        _write_json_atomic(progress_path, progress)
        return progress

    calibration_scores, calibration_labels = _score_rows(
        calibration_rows,
        model=model,
        training_manifest=training_manifest,
        candidates=candidates,
        schedule=schedule,
        row_provider=row_provider,
        n_features=n_features,
        fetch_concurrency=fetch_concurrency,
        deps=deps,
    )
    eval_scores, eval_labels = _score_rows(
        eval_rows,
        model=model,
        training_manifest=training_manifest,
        candidates=candidates,
        schedule=schedule,
        row_provider=row_provider,
        n_features=n_features,
        fetch_concurrency=fetch_concurrency,
        deps=deps,
    )
    threshold = _select_threshold_for_recall(
        y_true=calibration_labels,
        y_score=calibration_scores,
        target_recall=target_recall,
    )
    calibration_metrics = binary_calibration_metrics(
        y_true=calibration_labels,
        y_score=calibration_scores,
        threshold=threshold,
        bins=int(config.calibration.get("bins", 10)),
    )
    eval_metrics = binary_calibration_metrics(
        y_true=eval_labels,
        y_score=eval_scores,
        threshold=threshold,
        bins=int(config.calibration.get("bins", 10)),
    )

    deps["joblib"].dump(model, model_path)
    model_file_sha256 = _sha256_file(model_path)
    model_numeric_hash = _model_numeric_hash(model, deps=deps)
    provider_observations = (
        row_provider.observations()
        if hasattr(row_provider, "observations")
        else {"provider_class": type(row_provider).__name__}
    )
    report = {
        "artifact_kind": SPARSE_BASELINE_REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "model_id": MODEL_ID,
        "model_kind": "hashed_ngram_linear",
        "feature_code_version": FEATURE_CODE_VERSION,
        "training_data_manifest_hash": training_data_manifest_hash,
        "config_hash": config.config_hash,
        "feature_config": feature_config,
        "feature_config_hash": feature_config_hash,
        "fit_eligible_row_count": len(fit_rows),
        "audit_row_count": audit_row_count,
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in fit_rows).items())),
        "train_label_counts": dict(sorted(Counter(str(row["classifier_label"]) for row in train_rows).items())),
        "calibration_label_counts": dict(sorted(Counter(str(row["classifier_label"]) for row in calibration_rows).items())),
        "eval_label_counts": dict(sorted(Counter(str(row["classifier_label"]) for row in eval_rows).items())),
        "batch_size": int(batch_size),
        "fetch_concurrency": int(fetch_concurrency),
        "batch_order": batch_order,
        "random_state": int(random_state),
        "completed_train_batches": completed_train_batches,
        "total_train_batches": len(train_batches),
        "threshold": threshold,
        "threshold_selection": {
            "split": "calibration",
            "target_reject_recall": float(target_recall),
            "direction": "reject_at_or_above",
        },
        "calibration_metrics": calibration_metrics,
        "eval_metrics": eval_metrics,
        "cost_ledger": cost_ledger,
        "model_file_name": model_path.name,
        "model_file_sha256": model_file_sha256,
        "model_numeric_hash": model_numeric_hash,
        "provider_observations": provider_observations,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    report["model_payload_hash"] = stable_json_hash(
        {
            "model_id": report["model_id"],
            "model_kind": report["model_kind"],
            "model_numeric_hash": model_numeric_hash,
            "feature_config_hash": feature_config_hash,
            "training_data_manifest_hash": training_data_manifest_hash,
        }
    )
    report["report_hash"] = _report_hash(report)
    validate_sparse_baseline_report(report)
    _write_json_atomic(report_path, report)
    _write_json_atomic(
        progress_path,
        _progress_payload(
            status="complete",
            completed_train_batches=completed_train_batches,
            total_train_batches=len(train_batches),
            fit_eligible_row_count=len(fit_rows),
            audit_row_count=audit_row_count,
        ),
    )
    return report


def build_hashed_ngram_sparse_baseline_plan(
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    config: ClassifierPipelineConfig,
    artifact_dir: str | Path | None = None,
    batch_size: int = 512,
    n_features: int = 2**20,
    fetch_concurrency: int = 1,
    batch_order: str = "deterministic_stratified_shuffle",
    target_recall: float = 0.995,
    random_state: int = 20260608,
    expected_fit_rows: int | None = 34_992,
) -> dict[str, Any]:
    """Validate sparse-baseline readiness without fetching text or fitting."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if fetch_concurrency < 1:
        raise ValueError("fetch_concurrency must be positive")
    if batch_order not in {"source_file", "manifest", "deterministic_stratified_shuffle"}:
        raise ValueError("batch_order must be source_file, manifest, or deterministic_stratified_shuffle")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")

    _require_component(config)
    rows = list(training_manifest.get("rows") or [])
    training_data_manifest_hash = _training_manifest_hash(training_manifest)
    fit_rows = [row for row in rows if row.get("fit_eligible") is not False and str(row.get("split")) != "audit"]
    audit_row_count = len(rows) - len(fit_rows)
    if expected_fit_rows is not None and len(fit_rows) != int(expected_fit_rows):
        raise ValueError(f"expected {expected_fit_rows} fit-eligible rows, found {len(fit_rows)}")
    _validate_fit_rows(fit_rows)

    train_rows = [row for row in fit_rows if row["split"] == "train"]
    calibration_rows = [row for row in fit_rows if row["split"] == "calibration"]
    eval_rows = [row for row in fit_rows if row["split"] == "eval"]
    _validate_split_rows(train_rows, calibration_rows, eval_rows)
    ordered_train_rows = _order_rows_for_refetch(
        train_rows,
        candidates=candidates,
        schedule=schedule,
        batch_order=batch_order,
        random_state=random_state,
    )
    ordered_calibration_rows = _order_rows_for_refetch(
        calibration_rows,
        candidates=candidates,
        schedule=schedule,
        batch_order=batch_order,
        random_state=random_state,
    )
    ordered_eval_rows = _order_rows_for_refetch(
        eval_rows,
        candidates=candidates,
        schedule=schedule,
        batch_order=batch_order,
        random_state=random_state,
    )
    feature_config = _feature_config(n_features=n_features)
    feature_config_hash = stable_json_hash(feature_config)
    split_identity_hashes = {
        "train": _row_identity_hash(ordered_train_rows),
        "calibration": _row_identity_hash(ordered_calibration_rows),
        "eval": _row_identity_hash(ordered_eval_rows),
    }
    plan = {
        "artifact_kind": SPARSE_BASELINE_PLAN_KIND,
        "schema_version": 1,
        "status": "ready_for_sparse_training",
        "model_id": MODEL_ID,
        "model_kind": "hashed_ngram_linear",
        "feature_code_version": FEATURE_CODE_VERSION,
        "training_data_manifest_hash": training_data_manifest_hash,
        "config_hash": config.config_hash,
        "artifact_dir_configured": artifact_dir is not None,
        "feature_config": feature_config,
        "feature_config_hash": feature_config_hash,
        "fit_eligible_row_count": len(fit_rows),
        "expected_fit_row_count": int(expected_fit_rows) if expected_fit_rows is not None else None,
        "audit_row_count": audit_row_count,
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in fit_rows).items())),
        "train_label_counts": dict(sorted(Counter(str(row["classifier_label"]) for row in train_rows).items())),
        "calibration_label_counts": dict(sorted(Counter(str(row["classifier_label"]) for row in calibration_rows).items())),
        "eval_label_counts": dict(sorted(Counter(str(row["classifier_label"]) for row in eval_rows).items())),
        "split_identity_hashes": split_identity_hashes,
        "batch_size": int(batch_size),
        "fetch_concurrency": int(fetch_concurrency),
        "batch_order": batch_order,
        "random_state": int(random_state),
        "target_recall": float(target_recall),
        "threshold_selection": {
            "split": "calibration",
            "target_reject_recall": float(target_recall),
            "direction": "reject_at_or_above",
        },
        "eval_policy": {
            "split": "eval",
            "used_once_after_threshold_selection": True,
            "eval_split_used_for_threshold_selection": False,
        },
        "feature_input_contract": "private_raw_document_text_ngrams_only",
        "forbidden_metadata_features": [
            "candidate_id",
            "candidate_hash",
            "source_locator",
            "raw_url",
            "bucket_id",
            "provenance",
            "borderline_origin",
            "teacher_route",
            "split",
            "label_unit",
            "byte_length",
            "word_count",
        ],
        "row_provider_instantiated": False,
        "private_text_fetched": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "paid_api_used": False,
        "gpu_rental_used": False,
    }
    plan["plan_hash"] = _plan_hash(plan)
    validate_sparse_baseline_plan(plan)
    return plan


def validate_sparse_baseline_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("artifact_kind") != SPARSE_BASELINE_PLAN_KIND:
        raise ValueError("unsupported sparse baseline plan kind")
    validate_public_no_raw_payload(plan)
    if plan.get("status") != "ready_for_sparse_training":
        raise ValueError("sparse baseline plan must be ready_for_sparse_training")
    if plan.get("model_id") != MODEL_ID:
        raise ValueError("sparse baseline plan model id mismatch")
    if plan.get("classifier_training_performed") is not False:
        raise ValueError("sparse baseline plan must not perform classifier training")
    if plan.get("private_text_fetched") is not False:
        raise ValueError("sparse baseline plan must not fetch private text")
    if plan.get("row_provider_instantiated") is not False:
        raise ValueError("sparse baseline plan must not instantiate row provider")
    if plan.get("training_shards_created") is not False:
        raise ValueError("sparse baseline plan must not create training shards")
    if plan.get("plan_hash") != _plan_hash(plan):
        raise ValueError("sparse baseline plan hash mismatch")
    _require_sha256(str(plan.get("training_data_manifest_hash") or ""), "training_data_manifest_hash")
    _require_sha256(str(plan.get("feature_config_hash") or ""), "feature_config_hash")


def validate_sparse_baseline_report(report: Mapping[str, Any]) -> None:
    if report.get("artifact_kind") != SPARSE_BASELINE_REPORT_KIND:
        raise ValueError("unsupported sparse baseline report kind")
    validate_public_no_raw_payload(report)
    if report.get("status") != "complete":
        raise ValueError("sparse baseline report must be complete")
    if report.get("report_hash") != _report_hash(report):
        raise ValueError("sparse baseline report hash mismatch")
    _require_sha256(str(report.get("training_data_manifest_hash") or ""), "training_data_manifest_hash")
    _require_sha256(str(report.get("feature_config_hash") or ""), "feature_config_hash")
    _require_sha256(str(report.get("model_file_sha256") or ""), "model_file_sha256")


def _validate_existing_report_for_invocation(
    report: Mapping[str, Any],
    *,
    model_path: Path,
    training_data_manifest_hash: str,
    config_hash: str,
    feature_config_hash: str,
    fit_eligible_row_count: int,
    audit_row_count: int,
    split_counts: Mapping[str, int],
    train_label_counts: Mapping[str, int],
    calibration_label_counts: Mapping[str, int],
    eval_label_counts: Mapping[str, int],
    batch_size: int,
    batch_order: str,
    random_state: int,
    target_recall: float,
    total_train_batches: int,
) -> None:
    _expect_report_value(report, "training_data_manifest_hash", training_data_manifest_hash, "training manifest")
    _expect_report_value(report, "config_hash", config_hash, "config")
    _expect_report_value(report, "feature_config_hash", feature_config_hash, "feature config")
    _expect_report_value(report, "fit_eligible_row_count", int(fit_eligible_row_count), "fit row count")
    _expect_report_value(report, "audit_row_count", int(audit_row_count), "audit row count")
    _expect_report_value(report, "split_counts", dict(split_counts), "split counts")
    _expect_report_value(report, "train_label_counts", dict(train_label_counts), "train label counts")
    _expect_report_value(report, "calibration_label_counts", dict(calibration_label_counts), "calibration label counts")
    _expect_report_value(report, "eval_label_counts", dict(eval_label_counts), "eval label counts")
    _expect_report_value(report, "batch_size", int(batch_size), "batch size")
    _expect_report_value(report, "batch_order", batch_order, "batch order")
    _expect_report_value(report, "random_state", int(random_state), "random state")
    _expect_report_value(report, "total_train_batches", int(total_train_batches), "total train batches")
    threshold_selection = report.get("threshold_selection")
    if not isinstance(threshold_selection, Mapping):
        raise ValueError("sparse baseline report threshold selection missing")
    if float(threshold_selection.get("target_reject_recall", -1.0)) != float(target_recall):
        raise ValueError("sparse baseline report target recall drift")
    if not model_path.exists():
        raise ValueError("sparse baseline report model file missing")
    if str(report.get("model_file_name") or "") != model_path.name:
        raise ValueError("sparse baseline report model file name drift")
    if str(report.get("model_file_sha256") or "") != _sha256_file(model_path):
        raise ValueError("sparse baseline report model file hash drift")


def _expect_report_value(report: Mapping[str, Any], field: str, expected: Any, label: str) -> None:
    if report.get(field) != expected:
        raise ValueError(f"sparse baseline report {label} drift")


def _load_sparse_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import numpy
        from scipy.sparse import csr_matrix
        from sklearn.linear_model import SGDClassifier
    except ImportError as exc:  # pragma: no cover - exercised only in minimal envs.
        raise RuntimeError(
            "hashed n-gram sparse baseline requires optional filter dependencies: "
            "scikit-learn, scipy, numpy, and joblib"
        ) from exc
    return {
        "SGDClassifier": SGDClassifier,
        "csr_matrix": csr_matrix,
        "joblib": joblib,
        "numpy": numpy,
    }


def _require_component(config: ClassifierPipelineConfig) -> None:
    for spec in config.model_specs:
        if spec.model_id == MODEL_ID and spec.kind == "hashed_ngram_linear":
            return
    raise ValueError(f"classifier config must include {MODEL_ID} as hashed_ngram_linear")


def _feature_config(*, n_features: int) -> dict[str, Any]:
    return {
        "family": "hashed_word_char_ngrams",
        "n_features": int(n_features),
        "word_ngram_range": [1, 2],
        "char_ngram_range": [3, 5],
        "normalization": "sublinear_tf_l2",
    }


def _training_manifest_hash(training_manifest: Mapping[str, Any]) -> str:
    value = training_manifest.get("training_manifest_hash") or training_manifest.get("manifest_hash")
    if not value:
        value = stable_json_hash(training_manifest.get("rows") or [])
    return _require_sha256(str(value), "training_manifest_hash")


def _validate_fit_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    allowed_labels = {"safe", "reject"}
    bad_labels = sorted({str(row.get("classifier_label")) for row in rows} - allowed_labels)
    if bad_labels:
        raise ValueError(f"unsupported classifier labels: {bad_labels}")
    bad_splits = sorted({str(row.get("split")) for row in rows} - {"train", "calibration", "eval"})
    if bad_splits:
        raise ValueError(f"unsupported sparse-baseline fit splits: {bad_splits}")


def _validate_split_rows(
    train_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> None:
    if not train_rows:
        raise ValueError("sparse baseline requires train rows")
    if not calibration_rows:
        raise ValueError("sparse baseline requires calibration rows")
    if not eval_rows:
        raise ValueError("sparse baseline requires eval rows")
    train_labels = {str(row["classifier_label"]) for row in train_rows}
    if train_labels != {"safe", "reject"}:
        raise ValueError("sparse baseline requires both safe and reject train examples")
    for split_name, split_rows in (("calibration", calibration_rows), ("eval", eval_rows)):
        labels = {str(row["classifier_label"]) for row in split_rows}
        if labels != {"safe", "reject"}:
            raise ValueError(f"sparse baseline requires both safe and reject {split_name} examples")


def _batches(rows: Sequence[Mapping[str, Any]], batch_size: int) -> list[list[Mapping[str, Any]]]:
    return [list(rows[index : index + batch_size]) for index in range(0, len(rows), batch_size)]


def _order_rows_for_refetch(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    batch_order: str,
    random_state: int = 20260607,
) -> list[Mapping[str, Any]]:
    if batch_order == "manifest":
        return list(rows)
    if batch_order == "deterministic_stratified_shuffle":
        return _deterministic_stratified_shuffle(rows, random_state=random_state)
    candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    schedule_by_key = {
        (str(entry["source_file_id"]), int(entry["row_group_index"])): entry
        for entry in schedule.get("entries") or []
    }

    def sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, str]:
        candidate_id = str(row["candidate_id"])
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"candidate missing for training row: {candidate_id}")
        source_file_id = str(candidate["source_file_id"])
        row_group_index = int(candidate["row_group_index"])
        entry = schedule_by_key.get((source_file_id, row_group_index))
        if entry is None:
            raise ValueError("candidate physical row is not scheduled")
        source_key = str(entry.get("parquet_url_or_path") or source_file_id)
        return (
            source_key,
            source_file_id,
            row_group_index,
            int(candidate["row_group_local_row_index"]),
            candidate_id,
        )

    return sorted(rows, key=sort_key)


def _deterministic_stratified_shuffle(
    rows: Sequence[Mapping[str, Any]],
    *,
    random_state: int,
) -> list[Mapping[str, Any]]:
    by_label: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = str(row.get("classifier_label"))
        by_label.setdefault(label, []).append(row)
    for label, label_rows in by_label.items():
        by_label[label] = sorted(
            label_rows,
            key=lambda row: stable_json_hash(
                {
                    "batch_order": "deterministic_stratified_shuffle",
                    "random_state": int(random_state),
                    "split": str(row.get("split") or ""),
                    "classifier_label": label,
                    "candidate_id": str(row.get("candidate_id") or ""),
                }
            ),
        )
    labels = sorted(by_label)
    shuffled: list[Mapping[str, Any]] = []
    while any(by_label[label] for label in labels):
        for label in labels:
            if by_label[label]:
                shuffled.append(by_label[label].pop(0))
    return shuffled


def _row_identity_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return stable_json_hash(
        [
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "candidate_hash": str(row.get("candidate_hash") or ""),
                "split": str(row.get("split") or ""),
                "classifier_label": str(row.get("classifier_label") or ""),
                "fit_eligible": row.get("fit_eligible") is not False,
                "training_weight": float(row.get("training_weight", 1.0)),
            }
            for row in rows
        ]
    )


def _collect_feature_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    n_features: int,
    fetch_concurrency: int,
) -> list[Mapping[str, Any]]:
    subset_manifest = {
        "artifact_kind": training_manifest.get("artifact_kind", "final_35k_classifier_training_manifest"),
        "training_manifest_hash": _training_manifest_hash(training_manifest),
        "rows": list(rows),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    collected = collect_final_text_feature_rows(
        training_manifest=subset_manifest,
        candidates=candidates,
        schedule=schedule,
        row_provider=row_provider,
        start_row_index=0,
        max_rows=len(rows),
        include_audit=False,
        include_feature_counts=True,
        n_features=n_features,
        fetch_concurrency=fetch_concurrency,
    )
    return list(collected["rows"])


def _feature_rows_to_csr(rows: Sequence[Mapping[str, Any]], *, n_features: int, deps: Mapping[str, Any]) -> Any:
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in rows:
        counts = row.get("feature_hash_counts")
        if not isinstance(counts, Mapping) or not counts:
            raise ValueError("feature row is missing feature_hash_counts")
        row_values: list[tuple[int, float]] = []
        for feature_id, count in sorted(counts.items()):
            column = _feature_id_to_column(str(feature_id), n_features=n_features)
            value = 1.0 + math.log(float(count))
            row_values.append((column, value))
        norm = math.sqrt(sum(value * value for _column, value in row_values)) or 1.0
        for column, value in row_values:
            indices.append(column)
            data.append(value / norm)
        indptr.append(len(indices))
    return deps["csr_matrix"](
        (
            deps["numpy"].array(data, dtype=deps["numpy"].float64),
            deps["numpy"].array(indices, dtype=deps["numpy"].int32),
            deps["numpy"].array(indptr, dtype=deps["numpy"].int32),
        ),
        shape=(len(rows), int(n_features)),
    )


def _score_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    n_features: int,
    fetch_concurrency: int,
    deps: Mapping[str, Any],
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    for batch in _batches(rows, 512):
        feature_rows = _collect_feature_rows(
            batch,
            training_manifest=training_manifest,
            candidates=candidates,
            schedule=schedule,
            row_provider=row_provider,
            n_features=n_features,
            fetch_concurrency=fetch_concurrency,
        )
        matrix = _feature_rows_to_csr(feature_rows, n_features=n_features, deps=deps)
        probabilities = model.predict_proba(matrix)[:, 1]
        scores.extend(float(score) for score in probabilities)
        labels.extend(_label_to_int(row["classifier_label"]) for row in batch)
    return scores, labels


def _balanced_sample_weights(labels: Sequence[int]) -> dict[int, float]:
    counts = Counter(labels)
    total = len(labels)
    return {
        0: total / (2.0 * counts[0]),
        1: total / (2.0 * counts[1]),
    }


def _label_to_int(label: Any) -> int:
    value = str(label)
    if value == "reject":
        return 1
    if value == "safe":
        return 0
    raise ValueError(f"unsupported classifier label: {value}")


def _select_threshold_for_recall(*, y_true: Sequence[int], y_score: Sequence[float], target_recall: float) -> float:
    positives = sum(1 for label in y_true if int(label) == 1)
    if positives == 0:
        raise ValueError("calibration split must contain reject examples")
    for threshold in sorted({float(score) for score in y_score}, reverse=True):
        true_positive = sum(
            1
            for label, score in zip(y_true, y_score, strict=True)
            if int(label) == 1 and float(score) >= threshold
        )
        if true_positive / positives >= target_recall:
            return float(threshold)
    return 0.0


def _feature_id_to_column(feature_id: str, *, n_features: int) -> int:
    if not feature_id.startswith("h"):
        raise ValueError(f"unsupported feature id: {feature_id}")
    column = int(feature_id[1:], 16)
    if not 0 <= column < n_features:
        raise ValueError(f"feature id out of configured range: {feature_id}")
    return column


def _checkpoint_payload(
    *,
    completed_train_batches: int,
    total_train_batches: int,
    seen_train_rows: int,
    training_data_manifest_hash: str,
    feature_config_hash: str,
    batch_size: int,
    batch_order: str,
    random_state: int,
    model_file_sha256: str,
) -> dict[str, Any]:
    payload = {
        "artifact_kind": SPARSE_BASELINE_CHECKPOINT_KIND,
        "schema_version": 1,
        "completed_train_batches": int(completed_train_batches),
        "total_train_batches": int(total_train_batches),
        "seen_train_rows": int(seen_train_rows),
        "training_data_manifest_hash": training_data_manifest_hash,
        "feature_config_hash": feature_config_hash,
        "batch_size": int(batch_size),
        "batch_order": batch_order,
        "random_state": int(random_state),
        "model_file_name": f"{MODEL_ID}_model.joblib",
        "model_file_sha256": model_file_sha256,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    payload["checkpoint_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "checkpoint_hash"})
    validate_public_no_raw_payload(payload)
    return payload


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    feature_config_hash: str,
    training_data_manifest_hash: str,
    batch_size: int,
    total_train_batches: int,
    batch_order: str,
    random_state: int,
    model_path: Path,
) -> None:
    if checkpoint.get("artifact_kind") != SPARSE_BASELINE_CHECKPOINT_KIND:
        raise ValueError("unsupported sparse baseline checkpoint kind")
    validate_public_no_raw_payload(checkpoint)
    if checkpoint.get("feature_config_hash") != feature_config_hash:
        raise ValueError("sparse baseline checkpoint feature config drift")
    if checkpoint.get("training_data_manifest_hash") != training_data_manifest_hash:
        raise ValueError("sparse baseline checkpoint training manifest drift")
    if int(checkpoint.get("batch_size", -1)) != int(batch_size):
        raise ValueError("sparse baseline checkpoint batch size drift")
    if int(checkpoint.get("total_train_batches", -1)) != int(total_train_batches):
        raise ValueError("sparse baseline checkpoint total train batches drift")
    if checkpoint.get("batch_order") != batch_order:
        raise ValueError("sparse baseline checkpoint batch order drift")
    if int(checkpoint.get("random_state", 20260607)) != int(random_state):
        raise ValueError("sparse baseline checkpoint random_state drift")
    completed = int(checkpoint.get("completed_train_batches", -1))
    if completed < 0 or completed > int(total_train_batches):
        raise ValueError("sparse baseline checkpoint completed batch count invalid")
    if not model_path.exists():
        raise ValueError("sparse baseline checkpoint model file missing")
    if str(checkpoint.get("model_file_name") or "") != model_path.name:
        raise ValueError("sparse baseline checkpoint model file name drift")
    if str(checkpoint.get("model_file_sha256") or "") != _sha256_file(model_path):
        raise ValueError("sparse baseline checkpoint model file hash drift")
    expected_hash = stable_json_hash({key: value for key, value in checkpoint.items() if key != "checkpoint_hash"})
    if checkpoint.get("checkpoint_hash") != expected_hash:
        raise ValueError("sparse baseline checkpoint hash mismatch")


def _progress_payload(
    *,
    status: str,
    completed_train_batches: int,
    total_train_batches: int,
    fit_eligible_row_count: int,
    audit_row_count: int,
) -> dict[str, Any]:
    payload = {
        "artifact_kind": SPARSE_BASELINE_PROGRESS_KIND,
        "schema_version": 1,
        "status": status,
        "completed_train_batches": int(completed_train_batches),
        "total_train_batches": int(total_train_batches),
        "fit_eligible_row_count": int(fit_eligible_row_count),
        "audit_row_count": int(audit_row_count),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    payload["progress_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "progress_hash"})
    validate_public_no_raw_payload(payload)
    return payload


def _model_numeric_hash(model: Any, *, deps: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for attribute in ("coef_", "intercept_"):
        value = getattr(model, attribute, None)
        if value is None:
            continue
        array = deps["numpy"].ascontiguousarray(value)
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_hash(report: Mapping[str, Any]) -> str:
    return stable_json_hash({key: value for key, value in report.items() if key != "report_hash"})


def _plan_hash(plan: Mapping[str, Any]) -> str:
    return stable_json_hash({key: value for key, value in plan.items() if key != "plan_hash"})


def _require_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(payload)
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(serialized + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_candidates(path: str | Path) -> list[Mapping[str, Any]]:
    payload = _read_json(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate manifest must contain candidates list")
    return candidates


def _build_row_provider(args: argparse.Namespace) -> Any:
    if args.row_provider == "private-cache":
        return PrivateTextCacheRowProvider(cache_root=args.private_cache_root)
    if args.row_provider == "local-parquet":
        return LocalParquetRowProvider(columns=("text",))
    if args.row_provider == "hf-fsspec":
        if not args.hf_cache_dir:
            raise ValueError("--hf-cache-dir is required for --row-provider hf-fsspec")
        return HFFsspecParquetRowProvider(
            cache_dir=args.hf_cache_dir,
            cache_delta_cap_bytes=args.max_cache_delta_bytes,
            block_size_bytes=args.block_size_bytes,
            columns=("text",),
            max_cache_blocks=args.max_cache_blocks,
            manage_hf_cache_env=False,
            row_group_batch_size=args.row_group_batch_size,
        )
    raise ValueError(f"unsupported row provider: {args.row_provider}")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument(
        "--artifact-dir",
        default="reports/classifiers/final_35k_sparse_baseline_from_local_cache_2026_06_08",
    )
    parser.add_argument("--row-provider", choices=["private-cache", "local-parquet", "hf-fsspec"], default="private-cache")
    parser.add_argument("--private-cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-features", type=int, default=2**20)
    parser.add_argument("--fetch-concurrency", type=int, default=1)
    parser.add_argument(
        "--batch-order",
        choices=["source_file", "manifest", "deterministic_stratified_shuffle"],
        default="deterministic_stratified_shuffle",
    )
    parser.add_argument("--target-recall", type=float, default=0.995)
    parser.add_argument("--random-state", type=int, default=20260608)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--row-group-batch-size", type=int, default=64)
    parser.add_argument("--block-size-bytes", type=int, default=DEFAULT_HF_ROW_READ_BLOCK_SIZE_BYTES)
    parser.add_argument("--max-cache-blocks", type=int, default=1024)
    parser.add_argument("--max-cache-delta-bytes", type=int, default=30_000_000_000)
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--expected-fit-rows", type=int, default=34_992)
    args = parser.parse_args(argv)

    if args.dry_run_plan:
        plan = build_hashed_ngram_sparse_baseline_plan(
            training_manifest=_read_json(args.training_manifest),
            candidates=_load_candidates(args.candidates),
            schedule=_read_json(args.schedule),
            config=load_classifier_config(args.config),
            artifact_dir=args.artifact_dir,
            batch_size=args.batch_size,
            n_features=args.n_features,
            fetch_concurrency=args.fetch_concurrency,
            batch_order=args.batch_order,
            target_recall=args.target_recall,
            random_state=args.random_state,
            expected_fit_rows=args.expected_fit_rows,
        )
        print(json.dumps(plan, sort_keys=True))
        return 0

    report = train_hashed_ngram_sparse_baseline(
        training_manifest=_read_json(args.training_manifest),
        candidates=_load_candidates(args.candidates),
        schedule=_read_json(args.schedule),
        row_provider=_build_row_provider(args),
        config=load_classifier_config(args.config),
        artifact_dir=args.artifact_dir,
        batch_size=args.batch_size,
        n_features=args.n_features,
        fetch_concurrency=args.fetch_concurrency,
        batch_order=args.batch_order,
        target_recall=args.target_recall,
        random_state=args.random_state,
        max_train_batches=args.max_train_batches,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "model_id": report.get("model_id", MODEL_ID),
                "artifact_dir": str(args.artifact_dir),
                "completed_train_batches": report.get("completed_train_batches"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
