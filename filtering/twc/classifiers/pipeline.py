"""Train, evaluate, and freeze local classifier gate artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.classifiers.metrics import binary_calibration_metrics
from twc.classifiers.schema import (
    ClassifierPipelineConfig,
    EnsembleManifest,
    ModelArtifact,
    ModelSpec,
    TrainingExample,
    hash_label_set,
    validate_budget_ledger,
    validate_training_examples,
)
from twc.data.decision_schema import stable_json_hash


ENSEMBLE_DECISION_CAPABILITY = "reject_only_classifier_ensemble"
SPARSE_MODEL_ID = "hashed_ngram_linear_risk"
WINDOW_MIL_MODEL_ID = "modernbert_window_mil_risk"
SPARSE_USER_ACCEPTANCE_REPORT_KIND = "hashed_ngram_sparse_user_accepted_eval_report"
WINDOW_MIL_USER_ACCEPTANCE_REPORT_KIND = "modernbert_window_mil_user_accepted_rebind_report"
FINAL_FIT_ELIGIBLE_ROW_COUNT = 34_992
FINAL_AUDIT_ROW_COUNT = 8
FINAL_TARGET_REJECT_RECALL = 0.995


def load_classifier_config(source: str | Path | Mapping[str, Any]) -> ClassifierPipelineConfig:
    if isinstance(source, str | Path):
        with Path(source).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    else:
        payload = dict(source)
    if not isinstance(payload, Mapping):
        raise ValueError("Classifier config must parse to a mapping")
    return ClassifierPipelineConfig.from_mapping(payload)


def train_classifier_ensemble(
    records: Iterable[Mapping[str, Any]] | Iterable[TrainingExample],
    *,
    config: ClassifierPipelineConfig,
    artifact_dir: Path,
    training_data_manifest_hash: str,
    feature_code_version: str,
    cost_ledger: Mapping[str, Any],
    decision_capability: str = ENSEMBLE_DECISION_CAPABILITY,
) -> EnsembleManifest:
    """Train fixture-compatible classifier artifacts and freeze public metadata.

    The implementation is intentionally dependency-light: it builds deterministic
    score prototypes over redacted feature hashes and optional precomputed
    embeddings, then records the same artifact contracts future sklearn,
    fastText, or embedding-classifier trainers must satisfy.
    """

    examples = validate_training_examples(records)
    _require_sha256(training_data_manifest_hash, "training_data_manifest_hash")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ledger = validate_budget_ledger(cost_ledger, budget_cap_usd=config.budget_cap_usd)
    label_set_hash = hash_label_set(examples)
    model_artifacts: dict[str, ModelArtifact] = {}

    for spec in config.model_specs:
        artifact = _train_component(
            spec,
            examples=examples,
            config=config,
            label_set_hash=label_set_hash,
            training_data_manifest_hash=training_data_manifest_hash,
            feature_code_version=feature_code_version,
            cost_ledger=ledger,
            artifact_dir=artifact_dir,
        )
        model_artifacts[spec.model_id] = artifact

    ensemble_payload = {
        "schema_version": 1,
        "decision_capability": decision_capability,
        "approval_policy": {
            "classifiers_may_reject": config.approval_policy.classifiers_may_reject,
            "classifiers_may_approve_alone": config.approval_policy.classifiers_may_approve_alone,
            "reject_on_missing_classifier": config.approval_policy.reject_on_missing_classifier,
        },
        "config_hash": config.config_hash,
        "label_set_hash": label_set_hash,
        "training_data_manifest_hash": training_data_manifest_hash,
        "feature_code_version": feature_code_version,
        "cost_ledger": ledger,
        "model_artifacts": {
            model_id: artifact.to_public_dict()
            for model_id, artifact in sorted(model_artifacts.items())
        },
    }
    ensemble_sha256 = stable_json_hash(ensemble_payload)
    ensemble = EnsembleManifest(
        schema_version=1,
        decision_capability=decision_capability,
        approval_policy=config.approval_policy,
        config_hash=config.config_hash,
        label_set_hash=label_set_hash,
        training_data_manifest_hash=training_data_manifest_hash,
        feature_code_version=feature_code_version,
        cost_ledger=ledger,
        model_artifacts=model_artifacts,
        ensemble_sha256=ensemble_sha256,
    )
    _write_public_json(artifact_dir / "ensemble_manifest.json", ensemble.to_public_dict())
    return ensemble


def bind_trained_classifier_ensemble(
    reports: Iterable[Mapping[str, Any]],
    *,
    config: ClassifierPipelineConfig,
    artifact_dir: Path,
    training_data_manifest_hash: str,
    decision_capability: str = ENSEMBLE_DECISION_CAPABILITY,
    score_ledgers_by_model_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[EnsembleManifest, dict[str, Any]]:
    """Bind completed classifier reports into a reject-only freeze-review manifest.

    This does not train, approve corpus rows, create accepted manifests, or
    create training shards. It is the production binder for already completed
    sparse and window-MIL reports.
    """

    if decision_capability != ENSEMBLE_DECISION_CAPABILITY:
        raise ValueError(
            "production classifier binder requires "
            f"decision_capability={ENSEMBLE_DECISION_CAPABILITY!r}"
        )
    _require_sha256(training_data_manifest_hash, "training_data_manifest_hash")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    by_model_id = _validated_report_map(reports)
    required = set(config.required_model_ids)
    missing = sorted(required - set(by_model_id))
    extra = sorted(set(by_model_id) - required)
    if missing:
        raise ValueError(f"missing trained classifier reports: {missing}")
    if extra:
        raise ValueError(f"unexpected trained classifier reports: {extra}")

    model_artifacts: dict[str, ModelArtifact] = {}
    component_ledgers: dict[str, Any] = {}
    score_ledgers = dict(score_ledgers_by_model_id or {})
    total_estimated = 0.0
    total_actual = 0.0
    for spec in config.model_specs:
        report = by_model_id[spec.model_id]
        _validate_report_against_config(
            report,
            spec=spec,
            config=config,
            training_hash=training_data_manifest_hash,
            score_ledger=score_ledgers.get(spec.model_id),
        )
        cost = dict(report.get("cost_ledger") or {})
        component_ledgers[spec.model_id] = cost
        total_estimated += float(cost.get("estimated_usd", 0.0))
        total_actual += float(cost.get("actual_usd", 0.0))
        model_artifacts[spec.model_id] = _artifact_from_report(
            report,
            spec=spec,
            config=config,
            training_hash=training_data_manifest_hash,
        )

    ledger = validate_budget_ledger(
        {
            "budget_cap_usd": config.budget_cap_usd,
            "estimated_usd": total_estimated,
            "actual_usd": total_actual,
            "compute_backend": "bound_completed_classifier_reports",
            "component_cost_ledgers": component_ledgers,
            "paid_api_used": False,
            "gpu_rental_used": any(bool(ledger.get("gpu_rental_used")) for ledger in component_ledgers.values() if isinstance(ledger, Mapping)),
            "notes": "production_classifier_report_binder_no_corpus_approval",
        },
        budget_cap_usd=config.budget_cap_usd,
    )
    label_set_hash = training_data_manifest_hash
    feature_code_version = "classifier_report_binding_v1"
    ensemble_payload = {
        "schema_version": 1,
        "decision_capability": decision_capability,
        "approval_policy": {
            "classifiers_may_reject": config.approval_policy.classifiers_may_reject,
            "classifiers_may_approve_alone": config.approval_policy.classifiers_may_approve_alone,
            "reject_on_missing_classifier": config.approval_policy.reject_on_missing_classifier,
        },
        "config_hash": config.config_hash,
        "label_set_hash": label_set_hash,
        "training_data_manifest_hash": training_data_manifest_hash,
        "feature_code_version": feature_code_version,
        "cost_ledger": ledger,
        "model_artifacts": {
            model_id: artifact.to_public_dict()
            for model_id, artifact in sorted(model_artifacts.items())
        },
    }
    ensemble_sha256 = stable_json_hash(ensemble_payload)
    ensemble = EnsembleManifest(
        schema_version=1,
        decision_capability=decision_capability,
        approval_policy=config.approval_policy,
        config_hash=config.config_hash,
        label_set_hash=label_set_hash,
        training_data_manifest_hash=training_data_manifest_hash,
        feature_code_version=feature_code_version,
        cost_ledger=ledger,
        model_artifacts=model_artifacts,
        ensemble_sha256=ensemble_sha256,
    )
    freeze_manifest = _classifier_freeze_manifest(
        ensemble=ensemble,
        reports=by_model_id,
        config=config,
    )
    _write_public_json(artifact_dir / "ensemble_manifest.json", ensemble.to_public_dict())
    _write_public_json(artifact_dir / "classifier_freeze_manifest.json", freeze_manifest)
    return ensemble, freeze_manifest


def build_window_mil_user_acceptance_report(
    parent_report: Mapping[str, Any],
    *,
    repaired_training_data_manifest_hash: str,
    accepted_by: str,
    accepted_at_utc: str,
    repaired_fit_eligible_row_count: int = FINAL_FIT_ELIGIBLE_ROW_COUNT,
    repaired_audit_row_count: int = FINAL_AUDIT_ROW_COUNT,
) -> dict[str, Any]:
    """Rebind a stale Window-MIL report by explicit user acceptance.

    This is intentionally narrower than normal report validation. It preserves
    the original report hash/counts and only makes the old report usable as
    filter-freeze-review evidence for the repaired manifest.
    """

    from twc.classifiers.window_mil import validate_window_mil_report

    validate_window_mil_report(parent_report)
    validate_public_no_raw_payload(parent_report)
    _require_sha256(repaired_training_data_manifest_hash, "repaired_training_data_manifest_hash")
    parent_training_hash = str(parent_report.get("training_data_manifest_hash") or "")
    if parent_training_hash == repaired_training_data_manifest_hash:
        raise ValueError("Window-MIL accepted parent training manifest must differ from repaired manifest")
    if str(parent_report.get("model_id") or "") != WINDOW_MIL_MODEL_ID:
        raise ValueError("Window-MIL acceptance parent model id mismatch")
    if int(parent_report.get("fit_eligible_row_count", -1)) == int(repaired_fit_eligible_row_count):
        raise ValueError("Window-MIL acceptance must preserve stale parent fit row count")
    if int(parent_report.get("audit_row_count", -1)) == int(repaired_audit_row_count):
        raise ValueError("Window-MIL acceptance must preserve stale parent audit row count")
    accepted_by = str(accepted_by).strip()
    if not accepted_by:
        raise ValueError("Window-MIL acceptance accepted_by is required")
    accepted_at_utc = str(accepted_at_utc).strip()
    if not accepted_at_utc:
        raise ValueError("Window-MIL acceptance accepted_at_utc is required")
    eval_metrics = parent_report.get("eval_metrics")
    if not isinstance(eval_metrics, Mapping):
        raise ValueError("Window-MIL acceptance parent eval metrics missing")
    accepted_eval_recall = float(eval_metrics.get("recall", -1.0))
    payload = {
        "artifact_kind": WINDOW_MIL_USER_ACCEPTANCE_REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "model_id": WINDOW_MIL_MODEL_ID,
        "model_kind": str(parent_report.get("model_kind") or ""),
        "feature_code_version": str(parent_report.get("feature_code_version") or ""),
        "training_data_manifest_hash": repaired_training_data_manifest_hash,
        "config_hash": str(parent_report.get("config_hash") or ""),
        "fit_eligible_row_count": int(repaired_fit_eligible_row_count),
        "audit_row_count": int(repaired_audit_row_count),
        "threshold": float(parent_report.get("threshold")),
        "threshold_selection": dict(parent_report.get("threshold_selection") or {}),
        "calibration_metrics": dict(parent_report.get("calibration_metrics") or {}),
        "eval_metrics": dict(eval_metrics),
        "cost_ledger": dict(parent_report.get("cost_ledger") or {}),
        "model_file_name": str(parent_report.get("model_file_name") or ""),
        "model_file_sha256": str(parent_report.get("model_file_sha256") or ""),
        "model_numeric_hash": str(parent_report.get("model_numeric_hash") or ""),
        "model_payload_hash": str(parent_report.get("model_payload_hash") or ""),
        "acceptance": {
            "kind": "explicit_user_acceptance_of_stale_window_mil_report",
            "accepted_by": accepted_by,
            "accepted_at_utc": accepted_at_utc,
            "parent_report_hash": str(parent_report.get("report_hash") or ""),
            "parent_training_data_manifest_hash": parent_training_hash,
            "parent_config_hash": str(parent_report.get("config_hash") or ""),
            "parent_fit_eligible_row_count": int(parent_report.get("fit_eligible_row_count", -1)),
            "parent_audit_row_count": int(parent_report.get("audit_row_count", -1)),
            "parent_eval_recall": accepted_eval_recall,
            "accepted_eval_recall": accepted_eval_recall,
            "target_reject_recall": FINAL_TARGET_REJECT_RECALL,
            "eval_recall_below_target_accepted": accepted_eval_recall + 1e-12 < FINAL_TARGET_REJECT_RECALL,
            "stale_manifest_caveat_present": True,
            "rerun_required": False,
            "not_new_training_run": True,
            "accepted_for_filter_freeze_review_only": True,
        },
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    payload["report_hash"] = _binding_hash(payload)
    validate_window_mil_user_acceptance_report(payload)
    return payload


def build_sparse_user_acceptance_report(
    parent_report: Mapping[str, Any],
    *,
    accepted_by: str,
    accepted_at_utc: str,
) -> dict[str, Any]:
    """Accept a below-target sparse eval result without retuning on eval."""

    from twc.classifiers.sparse_baseline import validate_sparse_baseline_report

    validate_sparse_baseline_report(parent_report)
    validate_public_no_raw_payload(parent_report)
    if str(parent_report.get("model_id") or "") != SPARSE_MODEL_ID:
        raise ValueError("sparse acceptance parent model id mismatch")
    if int(parent_report.get("fit_eligible_row_count", -1)) != FINAL_FIT_ELIGIBLE_ROW_COUNT:
        raise ValueError("sparse acceptance parent fit row count drift")
    if int(parent_report.get("audit_row_count", -1)) != FINAL_AUDIT_ROW_COUNT:
        raise ValueError("sparse acceptance parent audit row count drift")
    accepted_by = str(accepted_by).strip()
    if not accepted_by:
        raise ValueError("sparse acceptance accepted_by is required")
    accepted_at_utc = str(accepted_at_utc).strip()
    if not accepted_at_utc:
        raise ValueError("sparse acceptance accepted_at_utc is required")
    eval_metrics = parent_report.get("eval_metrics")
    if not isinstance(eval_metrics, Mapping):
        raise ValueError("sparse acceptance parent eval metrics missing")
    accepted_eval_recall = float(eval_metrics.get("recall", -1.0))
    if accepted_eval_recall + 1e-12 >= FINAL_TARGET_REJECT_RECALL:
        raise ValueError("sparse acceptance parent eval recall is not below target")
    payload = {
        "artifact_kind": SPARSE_USER_ACCEPTANCE_REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "model_id": SPARSE_MODEL_ID,
        "model_kind": str(parent_report.get("model_kind") or ""),
        "feature_code_version": str(parent_report.get("feature_code_version") or ""),
        "training_data_manifest_hash": str(parent_report.get("training_data_manifest_hash") or ""),
        "config_hash": str(parent_report.get("config_hash") or ""),
        "feature_config_hash": str(parent_report.get("feature_config_hash") or ""),
        "fit_eligible_row_count": int(parent_report.get("fit_eligible_row_count", -1)),
        "audit_row_count": int(parent_report.get("audit_row_count", -1)),
        "split_counts": dict(parent_report.get("split_counts") or {}),
        "train_label_counts": dict(parent_report.get("train_label_counts") or {}),
        "calibration_label_counts": dict(parent_report.get("calibration_label_counts") or {}),
        "eval_label_counts": dict(parent_report.get("eval_label_counts") or {}),
        "batch_size": int(parent_report.get("batch_size") or 0),
        "fetch_concurrency": int(parent_report.get("fetch_concurrency") or 0),
        "batch_order": str(parent_report.get("batch_order") or ""),
        "random_state": int(parent_report.get("random_state") or 0),
        "completed_train_batches": int(parent_report.get("completed_train_batches") or 0),
        "total_train_batches": int(parent_report.get("total_train_batches") or 0),
        "threshold": float(parent_report.get("threshold")),
        "threshold_selection": dict(parent_report.get("threshold_selection") or {}),
        "calibration_metrics": dict(parent_report.get("calibration_metrics") or {}),
        "eval_metrics": dict(eval_metrics),
        "cost_ledger": dict(parent_report.get("cost_ledger") or {}),
        "model_file_name": str(parent_report.get("model_file_name") or ""),
        "model_file_sha256": str(parent_report.get("model_file_sha256") or ""),
        "model_numeric_hash": str(parent_report.get("model_numeric_hash") or ""),
        "model_payload_hash": str(parent_report.get("model_payload_hash") or ""),
        "provider_observations": dict(parent_report.get("provider_observations") or {}),
        "acceptance": {
            "kind": "explicit_user_acceptance_of_sparse_eval_recall_miss",
            "accepted_by": accepted_by,
            "accepted_at_utc": accepted_at_utc,
            "parent_report_hash": str(parent_report.get("report_hash") or ""),
            "accepted_eval_recall": accepted_eval_recall,
            "target_reject_recall": FINAL_TARGET_REJECT_RECALL,
            "eval_recall_below_target_accepted": True,
            "eval_used_for_threshold_selection": False,
            "rerun_required": False,
            "not_new_training_run": True,
            "accepted_for_filter_freeze_review_only": True,
        },
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    payload["report_hash"] = _binding_hash(payload)
    validate_sparse_user_acceptance_report(payload)
    return payload


def validate_sparse_user_acceptance_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != SPARSE_USER_ACCEPTANCE_REPORT_KIND:
        raise ValueError("unsupported sparse user acceptance report kind")
    if report.get("status") != "complete":
        raise ValueError("sparse user acceptance report must be complete")
    if report.get("model_id") != SPARSE_MODEL_ID:
        raise ValueError("sparse user acceptance model id mismatch")
    _require_sha256(str(report.get("training_data_manifest_hash") or ""), "training_data_manifest_hash")
    _require_sha256(str(report.get("config_hash") or ""), "config_hash")
    _require_sha256(str(report.get("feature_config_hash") or ""), "feature_config_hash")
    _require_sha256(str(report.get("model_file_sha256") or ""), "model_file_sha256")
    _require_sha256(str(report.get("model_payload_hash") or ""), "model_payload_hash")
    _require_sha256(str(report.get("model_numeric_hash") or ""), "model_numeric_hash")
    for field in (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "labels_are_corpus_approval",
        "training_shards_created",
    ):
        if report.get(field) is not False:
            raise ValueError(f"sparse user acceptance report must keep {field}=false")
    if int(report.get("fit_eligible_row_count", -1)) != FINAL_FIT_ELIGIBLE_ROW_COUNT:
        raise ValueError("sparse user acceptance fit row count drift")
    if int(report.get("audit_row_count", -1)) != FINAL_AUDIT_ROW_COUNT:
        raise ValueError("sparse user acceptance audit row count drift")
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("sparse user acceptance details missing")
    if acceptance.get("kind") != "explicit_user_acceptance_of_sparse_eval_recall_miss":
        raise ValueError("sparse user acceptance kind mismatch")
    if not str(acceptance.get("accepted_by") or "").strip():
        raise ValueError("sparse user acceptance accepted_by is required")
    if not str(acceptance.get("accepted_at_utc") or "").strip():
        raise ValueError("sparse user acceptance accepted_at_utc is required")
    _require_sha256(str(acceptance.get("parent_report_hash") or ""), "acceptance.parent_report_hash")
    eval_metrics = report.get("eval_metrics")
    if not isinstance(eval_metrics, Mapping):
        raise ValueError("sparse user acceptance eval metrics missing")
    eval_recall = float(eval_metrics.get("recall", -1.0))
    if eval_recall + 1e-12 >= FINAL_TARGET_REJECT_RECALL:
        raise ValueError("sparse user acceptance eval recall is not below target")
    accepted_eval_recall = float(acceptance.get("accepted_eval_recall", -1.0))
    if abs(eval_recall - accepted_eval_recall) > 1e-12:
        raise ValueError("sparse user acceptance eval recall mismatch")
    if acceptance.get("eval_recall_below_target_accepted") is not True:
        raise ValueError("sparse user acceptance must explicitly accept below-target eval recall")
    if acceptance.get("eval_used_for_threshold_selection") is not False:
        raise ValueError("sparse user acceptance must preserve calibration-only thresholding")
    if acceptance.get("rerun_required") is not False:
        raise ValueError("sparse user acceptance rerun_required must be false")
    if acceptance.get("not_new_training_run") is not True:
        raise ValueError("sparse user acceptance must state not_new_training_run")
    if acceptance.get("accepted_for_filter_freeze_review_only") is not True:
        raise ValueError("sparse user acceptance must be filter-freeze-review only")
    if report.get("report_hash") != _binding_hash(report):
        raise ValueError("sparse user acceptance report hash mismatch")


def validate_window_mil_user_acceptance_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != WINDOW_MIL_USER_ACCEPTANCE_REPORT_KIND:
        raise ValueError("unsupported Window-MIL user acceptance report kind")
    if report.get("status") != "complete":
        raise ValueError("Window-MIL user acceptance report must be complete")
    if report.get("model_id") != WINDOW_MIL_MODEL_ID:
        raise ValueError("Window-MIL user acceptance model id mismatch")
    _require_sha256(str(report.get("training_data_manifest_hash") or ""), "training_data_manifest_hash")
    _require_sha256(str(report.get("config_hash") or ""), "config_hash")
    _require_sha256(str(report.get("model_file_sha256") or ""), "model_file_sha256")
    _require_sha256(str(report.get("model_payload_hash") or ""), "model_payload_hash")
    _require_sha256(str(report.get("model_numeric_hash") or ""), "model_numeric_hash")
    for field in (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "labels_are_corpus_approval",
        "training_shards_created",
    ):
        if report.get(field) is not False:
            raise ValueError(f"Window-MIL user acceptance report must keep {field}=false")
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("Window-MIL user acceptance details missing")
    if not str(acceptance.get("accepted_by") or "").strip():
        raise ValueError("Window-MIL user acceptance accepted_by is required")
    if not str(acceptance.get("accepted_at_utc") or "").strip():
        raise ValueError("Window-MIL user acceptance accepted_at_utc is required")
    if acceptance.get("kind") != "explicit_user_acceptance_of_stale_window_mil_report":
        raise ValueError("Window-MIL user acceptance kind mismatch")
    for field in (
        "parent_report_hash",
        "parent_training_data_manifest_hash",
        "parent_config_hash",
    ):
        _require_sha256(str(acceptance.get(field) or ""), f"acceptance.{field}")
    if acceptance.get("parent_training_data_manifest_hash") == report.get("training_data_manifest_hash"):
        raise ValueError("Window-MIL accepted parent training manifest must differ from repaired manifest")
    if int(report.get("fit_eligible_row_count", -1)) != FINAL_FIT_ELIGIBLE_ROW_COUNT:
        raise ValueError("Window-MIL user acceptance repaired fit row count drift")
    if int(report.get("audit_row_count", -1)) != FINAL_AUDIT_ROW_COUNT:
        raise ValueError("Window-MIL user acceptance repaired audit row count drift")
    if int(acceptance.get("parent_fit_eligible_row_count", -1)) == FINAL_FIT_ELIGIBLE_ROW_COUNT:
        raise ValueError("Window-MIL user acceptance parent fit row count must remain stale")
    if int(acceptance.get("parent_audit_row_count", -1)) == FINAL_AUDIT_ROW_COUNT:
        raise ValueError("Window-MIL user acceptance parent audit row count must remain stale")
    if acceptance.get("stale_manifest_caveat_present") is not True:
        raise ValueError("Window-MIL user acceptance stale manifest caveat missing")
    if acceptance.get("rerun_required") is not False:
        raise ValueError("Window-MIL user acceptance rerun_required must be false")
    if acceptance.get("not_new_training_run") is not True:
        raise ValueError("Window-MIL user acceptance must state not_new_training_run")
    if acceptance.get("accepted_for_filter_freeze_review_only") is not True:
        raise ValueError("Window-MIL user acceptance must be filter-freeze-review only")
    eval_metrics = report.get("eval_metrics")
    if not isinstance(eval_metrics, Mapping):
        raise ValueError("Window-MIL user acceptance eval metrics missing")
    eval_recall = float(eval_metrics.get("recall", -1.0))
    accepted_eval_recall = float(acceptance.get("accepted_eval_recall", -1.0))
    if abs(eval_recall - accepted_eval_recall) > 1e-12:
        raise ValueError("Window-MIL user acceptance eval recall mismatch")
    if eval_recall + 1e-12 < FINAL_TARGET_REJECT_RECALL and acceptance.get("eval_recall_below_target_accepted") is not True:
        raise ValueError("Window-MIL user acceptance must explicitly accept below-target eval recall")
    if report.get("report_hash") != _binding_hash(report):
        raise ValueError("Window-MIL user acceptance report hash mismatch")


def _train_component(
    spec: ModelSpec,
    *,
    examples: tuple[TrainingExample, ...],
    config: ClassifierPipelineConfig,
    label_set_hash: str,
    training_data_manifest_hash: str,
    feature_code_version: str,
    cost_ledger: dict[str, Any],
    artifact_dir: Path,
) -> ModelArtifact:
    y_true = [_target_label(example, spec) for example in examples]
    train_indices = [
        index
        for index, example in enumerate(examples)
        if example.split == "train"
    ]
    if not train_indices:
        raise ValueError(f"{spec.model_id} requires train examples")
    fit_examples = tuple(examples[index] for index in train_indices)
    fit_y_true = [y_true[index] for index in train_indices]
    positive_count = sum(fit_y_true)
    negative_count = len(fit_y_true) - positive_count
    if positive_count < spec.min_positive_examples:
        raise ValueError(f"{spec.model_id} lacks positive examples for target {spec.target}")
    if negative_count < spec.min_negative_examples:
        raise ValueError(f"{spec.model_id} lacks negative examples for target {spec.target}")

    scores = [_score_example(example, spec, fit_examples, fit_y_true) for example in examples]
    eval_indices = [
        index
        for index, example in enumerate(examples)
        if example.split in {"calibration", "eval"}
    ]
    if not eval_indices:
        raise ValueError(f"{spec.model_id} requires disjoint calibration/eval examples")
    if len(eval_indices) < spec.min_calibration_examples:
        raise ValueError(f"{spec.model_id} lacks calibration/eval examples")
    metrics = binary_calibration_metrics(
        y_true=[y_true[index] for index in eval_indices],
        y_score=[scores[index] for index in eval_indices],
        threshold=spec.threshold,
        bins=int(config.calibration.get("bins", 10)),
    )
    metrics["target"] = spec.target
    metrics["score_source"] = spec.kind
    metrics["threshold_direction"] = spec.threshold_direction

    model_payload = {
        "model_id": spec.model_id,
        "kind": spec.kind,
        "target": spec.target,
        "threshold": spec.threshold,
        "threshold_direction": spec.threshold_direction,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "fit_split": "train",
        "training_split_counts": dict(Counter(example.split for example in examples)),
        "score_summary": {
            "min": min(scores),
            "max": max(scores),
            "mean": mean(scores),
        },
    }
    model_payload_hash = stable_json_hash(model_payload)
    artifact_payload = {
        "model_id": spec.model_id,
        "kind": spec.kind,
        "target": spec.target,
        "threshold": spec.threshold,
        "threshold_direction": spec.threshold_direction,
        "config_hash": config.config_hash,
        "label_set_hash": label_set_hash,
        "training_data_manifest_hash": training_data_manifest_hash,
        "feature_code_version": feature_code_version,
        "metrics": metrics,
        "cost_ledger": cost_ledger,
        "model_payload_hash": model_payload_hash,
    }
    artifact_sha256 = stable_json_hash(artifact_payload)
    artifact = ModelArtifact(
        model_id=spec.model_id,
        kind=spec.kind,
        target=spec.target,
        threshold=spec.threshold,
        threshold_direction=spec.threshold_direction,
        config_hash=config.config_hash,
        label_set_hash=label_set_hash,
        training_data_manifest_hash=training_data_manifest_hash,
        feature_code_version=feature_code_version,
        metrics=metrics,
        cost_ledger=cost_ledger,
        model_payload_hash=model_payload_hash,
        artifact_sha256=artifact_sha256,
    )
    _write_public_json(artifact_dir / f"{spec.model_id}.json", artifact.to_public_dict())
    return artifact


def _validated_report_map(reports: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    from twc.classifiers.sparse_baseline import validate_sparse_baseline_report
    from twc.classifiers.window_mil import validate_window_mil_report

    by_model_id: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        model_id = str(report.get("model_id") or "")
        if model_id == SPARSE_MODEL_ID:
            if report.get("artifact_kind") == SPARSE_USER_ACCEPTANCE_REPORT_KIND:
                validate_sparse_user_acceptance_report(report)
            else:
                validate_sparse_baseline_report(report)
        elif model_id == WINDOW_MIL_MODEL_ID:
            if report.get("artifact_kind") == WINDOW_MIL_USER_ACCEPTANCE_REPORT_KIND:
                validate_window_mil_user_acceptance_report(report)
            else:
                validate_window_mil_report(report)
        else:
            raise ValueError(f"unsupported trained classifier report model_id: {model_id}")
        if model_id in by_model_id:
            raise ValueError(f"duplicate trained classifier report: {model_id}")
        by_model_id[model_id] = report
    return by_model_id


def _validate_report_against_config(
    report: Mapping[str, Any],
    *,
    spec: ModelSpec,
    config: ClassifierPipelineConfig,
    training_hash: str,
    score_ledger: Mapping[str, Any] | None = None,
) -> None:
    if report.get("training_data_manifest_hash") != training_hash:
        raise ValueError(f"{spec.model_id} training manifest drift")
    if report.get("config_hash") != config.config_hash:
        raise ValueError(f"{spec.model_id} classifier config drift")
    if report.get("status") != "complete":
        raise ValueError(f"{spec.model_id} report must be complete")
    accepted_sparse = report.get("artifact_kind") == SPARSE_USER_ACCEPTANCE_REPORT_KIND
    accepted_window_mil = report.get("artifact_kind") == WINDOW_MIL_USER_ACCEPTANCE_REPORT_KIND
    if accepted_sparse:
        _validate_sparse_user_acceptance_against_config(
            report,
            spec=spec,
            config=config,
            training_hash=training_hash,
        )
    if accepted_window_mil:
        _validate_window_mil_user_acceptance_against_config(
            report,
            spec=spec,
            config=config,
            training_hash=training_hash,
        )
    if int(report.get("fit_eligible_row_count", -1)) != FINAL_FIT_ELIGIBLE_ROW_COUNT:
        raise ValueError(
            f"{spec.model_id} fit row count must be {FINAL_FIT_ELIGIBLE_ROW_COUNT}"
        )
    if int(report.get("audit_row_count", -1)) != FINAL_AUDIT_ROW_COUNT:
        raise ValueError(f"{spec.model_id} audit row count must be {FINAL_AUDIT_ROW_COUNT}")
    if report.get("raw_text_serialized") is not False or report.get("raw_url_serialized") is not False:
        raise ValueError(f"{spec.model_id} report must be no-raw")
    if report.get("token_serialized") is not False:
        raise ValueError(f"{spec.model_id} report must not serialize tokens")
    for field in ("accepted_manifest", "labels_are_corpus_approval", "training_shards_created"):
        if report.get(field) is not False:
            raise ValueError(f"{spec.model_id} report must keep {field}=false")
    direction = str((report.get("threshold_selection") or {}).get("direction") or "")
    if direction != spec.threshold_direction:
        raise ValueError(f"{spec.model_id} threshold direction drift")
    threshold_selection = report.get("threshold_selection")
    if not isinstance(threshold_selection, Mapping):
        raise ValueError(f"{spec.model_id} threshold selection missing")
    if threshold_selection.get("split") != "calibration":
        raise ValueError(f"{spec.model_id} threshold selection split must be calibration")
    target_recall = float(threshold_selection.get("target_reject_recall", -1.0))
    if target_recall + 1e-12 < FINAL_TARGET_REJECT_RECALL:
        raise ValueError(f"{spec.model_id} target recall must be at least {FINAL_TARGET_REJECT_RECALL}")
    if target_recall > FINAL_TARGET_REJECT_RECALL + 1e-12:
        _validate_calibration_only_rethreshold_provenance(
            report,
            threshold_selection=threshold_selection,
            model_id=spec.model_id,
            score_ledger=score_ledger,
            bins=int(config.calibration.get("bins", 10)),
        )
    for split_name in ("calibration", "eval"):
        metrics = report.get(f"{split_name}_metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{spec.model_id} {split_name} metrics missing")
        recall = metrics.get("recall")
        if not isinstance(recall, int | float):
            raise ValueError(f"{spec.model_id} {split_name} recall missing")
        below_target = float(recall) + 1e-12 < FINAL_TARGET_REJECT_RECALL
        accepted_sparse_below_target_eval = (
            accepted_sparse
            and spec.model_id == SPARSE_MODEL_ID
            and split_name == "eval"
            and (report.get("acceptance") or {}).get("eval_recall_below_target_accepted") is True
        )
        accepted_below_target_eval = (
            accepted_window_mil
            and spec.model_id == WINDOW_MIL_MODEL_ID
            and split_name == "eval"
            and (report.get("acceptance") or {}).get("eval_recall_below_target_accepted") is True
        )
        if below_target and not (accepted_sparse_below_target_eval or accepted_below_target_eval):
            raise ValueError(
                f"{spec.model_id} {split_name} recall below target {FINAL_TARGET_REJECT_RECALL}"
            )


def _validate_sparse_user_acceptance_against_config(
    report: Mapping[str, Any],
    *,
    spec: ModelSpec,
    config: ClassifierPipelineConfig,
    training_hash: str,
) -> None:
    if spec.model_id != SPARSE_MODEL_ID or spec.kind != "hashed_ngram_linear":
        raise ValueError("sparse user acceptance can only satisfy the sparse spec")
    validate_sparse_user_acceptance_report(report)
    if report.get("training_data_manifest_hash") != training_hash:
        raise ValueError("sparse user acceptance training manifest drift")
    if report.get("config_hash") != config.config_hash:
        raise ValueError("sparse user acceptance classifier config drift")


def _validate_window_mil_user_acceptance_against_config(
    report: Mapping[str, Any],
    *,
    spec: ModelSpec,
    config: ClassifierPipelineConfig,
    training_hash: str,
) -> None:
    if spec.model_id != WINDOW_MIL_MODEL_ID or spec.kind != "transformer_window_mil":
        raise ValueError("Window-MIL user acceptance can only satisfy the Window-MIL spec")
    validate_window_mil_user_acceptance_report(report)
    if report.get("training_data_manifest_hash") != training_hash:
        raise ValueError("Window-MIL user acceptance repaired training manifest drift")
    if report.get("config_hash") != config.config_hash:
        raise ValueError("Window-MIL user acceptance classifier config drift")
    acceptance = report.get("acceptance") or {}
    if acceptance.get("parent_config_hash") != config.config_hash:
        raise ValueError("Window-MIL user acceptance parent classifier config drift")


def _validate_calibration_only_rethreshold_provenance(
    report: Mapping[str, Any],
    *,
    threshold_selection: Mapping[str, Any],
    model_id: str,
    score_ledger: Mapping[str, Any] | None,
    bins: int,
) -> None:
    if score_ledger is None:
        raise ValueError(f"{model_id} stricter target requires score ledger")
    derivation = report.get("derivation")
    if not isinstance(derivation, Mapping) or derivation.get("kind") != "calibration_only_rethreshold":
        raise ValueError(f"{model_id} stricter target requires calibration-only rethreshold provenance")
    if derivation.get("eval_split_used_for_threshold_selection") is not False:
        raise ValueError(f"{model_id} calibration-only rethreshold provenance must not use eval for threshold")
    if threshold_selection.get("eval_split_used_for_threshold_selection") is not False:
        raise ValueError(f"{model_id} threshold selection must not use eval split")
    for field in (
        "parent_report_hash",
        "parent_model_file_sha256",
        "parent_model_payload_hash",
        "parent_model_numeric_hash",
        "score_ledger_hash",
    ):
        value = str(derivation.get(field) or "")
        _require_sha256(value, f"{model_id}.{field}")
        if threshold_selection.get(field) != value:
            raise ValueError(f"{model_id} calibration-only rethreshold provenance mismatch for {field}")
    if derivation.get("parent_model_file_sha256") != report.get("model_file_sha256"):
        raise ValueError(f"{model_id} calibration-only rethreshold must reuse parent model file")
    _validate_rethreshold_score_ledger_metrics(
        report,
        score_ledger=score_ledger,
        model_id=model_id,
        bins=bins,
    )


def _validate_rethreshold_score_ledger_metrics(
    report: Mapping[str, Any],
    *,
    score_ledger: Mapping[str, Any],
    model_id: str,
    bins: int,
) -> None:
    from twc.classifiers.sparse_baseline import _select_threshold_for_recall
    from twc.classifiers.window_mil import validate_window_mil_score_ledger

    derivation = report.get("derivation")
    threshold_selection = report.get("threshold_selection")
    if not isinstance(derivation, Mapping) or not isinstance(threshold_selection, Mapping):
        raise ValueError(f"{model_id} calibration-only rethreshold provenance missing")
    vectors = validate_window_mil_score_ledger(
        score_ledger,
        expected_training_data_manifest_hash=str(report["training_data_manifest_hash"]),
        expected_parent_report_hash=str(derivation["parent_report_hash"]),
        expected_parent_model_file_sha256=str(derivation["parent_model_file_sha256"]),
        expected_parent_model_payload_hash=str(derivation["parent_model_payload_hash"]),
        expected_parent_model_numeric_hash=str(derivation["parent_model_numeric_hash"]),
        expected_split_counts=report.get("split_counts"),
        expected_label_counts={
            "calibration": report.get("calibration_label_counts"),
            "eval": report.get("eval_label_counts"),
        },
    )
    if score_ledger.get("score_ledger_hash") != derivation.get("score_ledger_hash"):
        raise ValueError(f"{model_id} score ledger hash mismatch")
    target_recall = float(threshold_selection["target_reject_recall"])
    calibration_labels, calibration_scores = vectors["calibration"]
    eval_labels, eval_scores = vectors["eval"]
    threshold = _select_threshold_for_recall(
        y_true=calibration_labels,
        y_score=calibration_scores,
        target_recall=target_recall,
    )
    if abs(float(report["threshold"]) - float(threshold)) > 1e-12:
        raise ValueError(f"{model_id} rethreshold score ledger threshold mismatch")
    expected_calibration_metrics = binary_calibration_metrics(
        y_true=calibration_labels,
        y_score=calibration_scores,
        threshold=threshold,
        bins=int(bins),
    )
    expected_eval_metrics = binary_calibration_metrics(
        y_true=eval_labels,
        y_score=eval_scores,
        threshold=threshold,
        bins=int(bins),
    )
    _require_metric_subset(report.get("calibration_metrics"), expected_calibration_metrics, model_id, "calibration")
    _require_metric_subset(report.get("eval_metrics"), expected_eval_metrics, model_id, "eval")

def _require_metric_subset(
    actual: Any,
    expected: Mapping[str, Any],
    model_id: str,
    split: str,
) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError(f"{model_id} {split} metrics missing")
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            if abs(float(actual_value) - expected_value) > 1e-12:
                raise ValueError(f"{model_id} {split} metric {key} mismatch")
        elif actual_value != expected_value:
            raise ValueError(f"{model_id} {split} metric {key} mismatch")


def _artifact_from_report(
    report: Mapping[str, Any],
    *,
    spec: ModelSpec,
    config: ClassifierPipelineConfig,
    training_hash: str,
) -> ModelArtifact:
    metrics = _flatten_report_metrics(report)
    model_payload_hash = str(report.get("model_payload_hash") or report.get("model_numeric_hash") or "")
    _require_sha256(model_payload_hash, f"{spec.model_id}.model_payload_hash")
    artifact_payload = {
        "model_id": spec.model_id,
        "kind": spec.kind,
        "target": spec.target,
        "threshold": float(report["threshold"]),
        "threshold_direction": spec.threshold_direction,
        "config_hash": config.config_hash,
        "label_set_hash": training_hash,
        "training_data_manifest_hash": training_hash,
        "feature_code_version": str(report.get("feature_code_version") or ""),
        "metrics": metrics,
        "cost_ledger": dict(report.get("cost_ledger") or {}),
        "model_payload_hash": model_payload_hash,
        "artifact_sha256": str(report.get("model_file_sha256") or ""),
    }
    return ModelArtifact(
        model_id=spec.model_id,
        kind=spec.kind,
        target=spec.target,
        threshold=float(report["threshold"]),
        threshold_direction=spec.threshold_direction,
        config_hash=config.config_hash,
        label_set_hash=training_hash,
        training_data_manifest_hash=training_hash,
        feature_code_version=str(report.get("feature_code_version") or ""),
        metrics=metrics,
        cost_ledger=dict(report.get("cost_ledger") or {}),
        model_payload_hash=model_payload_hash,
        artifact_sha256=stable_json_hash(artifact_payload),
    )


def _flatten_report_metrics(report: Mapping[str, Any]) -> dict[str, float | int | str]:
    flattened: dict[str, float | int | str] = {}
    for prefix in ("calibration", "eval"):
        metrics = report.get(f"{prefix}_metrics") or {}
        if isinstance(metrics, Mapping):
            for key, value in sorted(metrics.items()):
                if isinstance(value, bool):
                    flattened[f"{prefix}_{key}"] = int(value)
                elif isinstance(value, int | float | str):
                    flattened[f"{prefix}_{key}"] = value
    flattened["fit_eligible_row_count"] = int(report.get("fit_eligible_row_count") or 0)
    flattened["audit_row_count"] = int(report.get("audit_row_count") or 0)
    return flattened


def _classifier_freeze_manifest(
    *,
    ensemble: EnsembleManifest,
    reports: Mapping[str, Mapping[str, Any]],
    config: ClassifierPipelineConfig,
) -> dict[str, Any]:
    payload = {
        "artifact_kind": "classifier_filter_freeze_readiness_manifest",
        "schema_version": 1,
        "status": "ready_for_filter_freeze_review",
        "classifier_training_complete": True,
        "filter_freeze_finalized": False,
        "requires_filter_freeze_review_rounds": True,
        "decision_capability": ensemble.decision_capability,
        "classifiers_may_reject": config.approval_policy.classifiers_may_reject,
        "classifiers_may_approve_alone": config.approval_policy.classifiers_may_approve_alone,
        "reject_on_missing_classifier": config.approval_policy.reject_on_missing_classifier,
        "config_hash": config.config_hash,
        "ensemble_sha256": ensemble.ensemble_sha256,
        "training_data_manifest_hash": ensemble.training_data_manifest_hash,
        "model_report_hashes": {
            model_id: str(report.get("report_hash") or "")
            for model_id, report in sorted(reports.items())
        },
        "model_file_hashes": {
            model_id: str(report.get("model_file_sha256") or "")
            for model_id, report in sorted(reports.items())
        },
        "accepted_model_evidence": {
            model_id: _accepted_model_evidence(report)
            for model_id, report in sorted(reports.items())
            if report.get("artifact_kind") in {SPARSE_USER_ACCEPTANCE_REPORT_KIND, WINDOW_MIL_USER_ACCEPTANCE_REPORT_KIND}
        },
        "cost_ledger": ensemble.cost_ledger,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "tokenizer_shards_created": False,
        "base_pretraining_started": False,
    }
    payload["freeze_readiness_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "freeze_readiness_hash"})
    return payload


def _accepted_model_evidence(report: Mapping[str, Any]) -> dict[str, Any]:
    acceptance = report.get("acceptance")
    if not isinstance(acceptance, Mapping):
        return {}
    return {
        "artifact_kind": str(report.get("artifact_kind") or ""),
        "report_hash": str(report.get("report_hash") or ""),
        "parent_report_hash": str(acceptance.get("parent_report_hash") or ""),
        "parent_training_data_manifest_hash": str(acceptance.get("parent_training_data_manifest_hash") or report.get("training_data_manifest_hash") or ""),
        "parent_fit_eligible_row_count": int(acceptance.get("parent_fit_eligible_row_count") or report.get("fit_eligible_row_count") or 0),
        "parent_audit_row_count": int(acceptance.get("parent_audit_row_count") or report.get("audit_row_count") or 0),
        "accepted_eval_recall": float(acceptance.get("accepted_eval_recall") or 0.0),
        "eval_recall_below_target_accepted": acceptance.get("eval_recall_below_target_accepted") is True,
        "stale_manifest_caveat_present": acceptance.get("stale_manifest_caveat_present") is True,
        "eval_used_for_threshold_selection": acceptance.get("eval_used_for_threshold_selection") is True,
        "not_new_training_run": acceptance.get("not_new_training_run") is True,
        "accepted_for_filter_freeze_review_only": acceptance.get("accepted_for_filter_freeze_review_only") is True,
    }


def _target_label(example: TrainingExample, spec: ModelSpec) -> int:
    if spec.target == "risk_any":
        return 0 if example.label in {"safe", "safe_external"} else 1
    if spec.target in {"safe", "safe_external"}:
        return 1 if example.label in {"safe", "safe_external"} else 0
    return 1 if example.label == spec.target else 0


def _score_example(
    example: TrainingExample,
    spec: ModelSpec,
    examples: tuple[TrainingExample, ...],
    y_true: list[int],
) -> float:
    if spec.kind == "embedding_centroid":
        return _embedding_centroid_score(example, examples, y_true)
    if spec.kind in {"lexical_ngram", "hashed_linear", "hashed_ngram_linear", "transformer_window_mil"}:
        return _redacted_feature_score(example, spec, y_true)
    raise ValueError(f"Unsupported classifier kind: {spec.kind}")


def _redacted_feature_score(example: TrainingExample, spec: ModelSpec, y_true: list[int]) -> float:
    if example.features:
        weighted_sum = sum(_stable_unit_interval(f"{spec.model_id}:{key}") * value for key, value in example.features.items())
        return _clip01(weighted_sum / max(1.0, sum(abs(value) for value in example.features.values())))
    return _clip01(sum(y_true) / len(y_true))


def _embedding_centroid_score(
    example: TrainingExample,
    examples: tuple[TrainingExample, ...],
    y_true: list[int],
) -> float:
    if example.embedding is None:
        return 0.5
    positive_embeddings = [
        candidate.embedding
        for candidate, label in zip(examples, y_true, strict=True)
        if label == 1 and candidate.embedding is not None
    ]
    negative_embeddings = [
        candidate.embedding
        for candidate, label in zip(examples, y_true, strict=True)
        if label == 0 and candidate.embedding is not None
    ]
    if not positive_embeddings or not negative_embeddings:
        return 0.5
    positive_centroid = _centroid(positive_embeddings)
    negative_centroid = _centroid(negative_embeddings)
    positive_distance = _squared_distance(example.embedding, positive_centroid)
    negative_distance = _squared_distance(example.embedding, negative_centroid)
    if positive_distance + negative_distance == 0:
        return 0.5
    return _clip01(negative_distance / (positive_distance + negative_distance))


def _centroid(vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
    dims = len(vectors[0])
    for vector in vectors:
        if len(vector) != dims:
            raise ValueError("Embedding dimensions must be consistent")
    return tuple(sum(vector[dim] for vector in vectors) / len(vectors) for dim in range(dims))


def _squared_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match")
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _stable_unit_interval(seed: str) -> float:
    digest = stable_json_hash({"seed": seed})
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")


def _write_public_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2)
    forbidden_key_fragments = [
        '"raw_text":',
        '"normalized_text":',
        '"source_doc_id":',
        '"rationale":',
        '"risk_spans":',
        '"matched_text":',
        '"raw_context_snippet":',
    ]
    for token in forbidden_key_fragments:
        if token in serialized:
            raise ValueError(f"Refusing to write raw-text-bearing classifier artifact token: {token}")
    path.write_text(serialized + "\n", encoding="utf-8")


def _binding_hash(payload: Mapping[str, Any]) -> str:
    return stable_json_hash({key: value for key, value in payload.items() if key != "report_hash"})
