"""Chunk-level classifier evidence cache, training runners, and freeze checks.

This module is the production-facing chunk classifier subsystem. Public JSON
reports are no-raw; private core+halo text lives only in the caller-supplied
SQLite evidence cache outside the public repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

from twc.classifiers.chunk_dataset import (
    ACTIVE_LEARNING_EVIDENCE_TIER,
    ACTIVE_LEARNING_PRIVATE_CACHE_ROUTE,
    CHUNK_CLASSIFIER_SCHEMA_VERSION_V2,
    FRESH_REPAIR_EVIDENCE_TIER,
    INHERITED_PARENT_SAFE_EVIDENCE_TIER,
    STRONG_EVIDENCE_TIER,
    validate_chunk_classifier_training_manifest,
)
from twc.classifiers.active_learning_private_maps import load_active_learning_private_index
from twc.classifiers.chunk_training_smoke import (
    DEFAULT_HALO_CHARS,
    _complete_stage_dirs_from_root,
    _fetch_core_halo_texts,
    _load_source_schedule_index,
    _load_stage_private_index,
)
from twc.classifiers.feature_extraction import hashed_ngram_feature_counts
from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.classifiers.metrics import binary_calibration_metrics
from twc.classifiers.private_text_cache import DEFAULT_CACHE_ROOT, PrivateTextCacheRowProvider
from twc.classifiers.sparse_baseline import (
    _batches,
    _label_to_int,
    _select_threshold_for_recall,
    _sha256_file,
)
from twc.classifiers.window_mil import (
    DEFAULT_MODEL_NAME,
    WINDOW_POOLING_MAX,
    WINDOW_POOLING_TOPK_MEAN,
    default_model_bundle_factory,
    _autocast_context,
    _device_observations,
    _load_torch,
    _make_grad_scaler,
    _model_numeric_hash as _torch_model_numeric_hash,
    _resolve_device,
    _score_document_batch,
    _score_split_halo_document_batch,
    _train_one_document_batch,
    _train_split_halo_document_batch,
)
from twc.data.decision_log import (
    DEFAULT_HF_ROW_READ_BLOCK_SIZE_BYTES,
    HFFsspecParquetRowProvider,
)
from twc.data.decision_schema import stable_json_hash
from twc.data.chunk_relabeling import build_inherited_safe_parent_chunk_artifacts


CHUNK_PRIVATE_EVIDENCE_CACHE_KIND = "twc_chunk_classifier_private_evidence_cache"
CHUNK_EVIDENCE_CACHE_REPORT_KIND = "twc_chunk_classifier_evidence_cache_report"
CHUNK_SPARSE_REPORT_KIND = "twc_chunk_hashed_ngram_sparse_report"
CHUNK_SPARSE_PROGRESS_KIND = "twc_chunk_hashed_ngram_sparse_progress"
CHUNK_SPARSE_CHECKPOINT_KIND = "twc_chunk_hashed_ngram_sparse_checkpoint"
CHUNK_WINDOW_MIL_REPORT_KIND = "twc_chunk_modernbert_window_mil_report"
CHUNK_WINDOW_MIL_PROGRESS_KIND = "twc_chunk_modernbert_window_mil_progress"
CHUNK_WINDOW_MIL_CHECKPOINT_KIND = "twc_chunk_modernbert_window_mil_checkpoint"
CHUNK_FREEZE_REPORT_KIND = "twc_chunk_classifier_freeze_report"
CHUNK_VARIANT_REGISTRY_KIND = "twc_chunk_variant_registry_v1"
CHUNK_TRAIN_SCORE_LEDGER_KIND = "twc_chunk_train_score_ledger_v1"
CHUNK_TRAIN_SCORE_LEDGER_SCHEMA_VERSION = 2
CHUNK_TRAIN_SCORE_RANK_POLICY_VERSION = "train_label_score_percentile_v1_no_calibration_thresholds"
CHUNK_HARD_CURRICULUM_KIND = "twc_chunk_hard_curriculum_v1"
CHUNK_HARD_CURRICULUM_SCHEMA_VERSION = 3
CHUNK_HARD_CURRICULUM_REASON_POLICY_VERSION = "chunk_hard_curriculum_reason_policy_v3_train_score_ranks"

CHUNK_SPARSE_MODEL_ID = "chunk_hashed_ngram_linear_risk"
CHUNK_WINDOW_MIL_MODEL_ID = "chunk_modernbert_window_mil_risk"
CHUNK_SPARSE_FEATURE_CODE_VERSION = "chunk_hashed_word_char_ngrams_v1"
CHUNK_WINDOW_FEATURE_CODE_VERSION = "chunk_modernbert_window_mil_v1"
CHUNK_WINDOW_FEATURE_CODE_VERSION_V2 = "chunk_modernbert_window_mil_v2"
CHUNK_WINDOW_ACTIVATION_CHECKPOINT_WINDOWS = True

WINDOW_INPUT_FULL_EVIDENCE = "full_evidence"
WINDOW_INPUT_CORE_ONLY = "core_only"
WINDOW_INPUT_SPLIT_HALO_CAPPED = "split_halo_capped"
SUPPORTED_WINDOW_INPUT_MODES = frozenset(
    {WINDOW_INPUT_FULL_EVIDENCE, WINDOW_INPUT_CORE_ONLY, WINDOW_INPUT_SPLIT_HALO_CAPPED}
)

DEFAULT_TARGET_RECALL = 0.9942
DEFAULT_SPARSE_N_FEATURES = 2**20
DEFAULT_BATCH_SIZE = 512
DEFAULT_RANDOM_STATE = 20260619
TRAIN_SCORE_HIGH_RISK_PERCENTILE = 0.90
TRAIN_SCORE_LOW_RISK_PERCENTILE = 0.10

CURRICULUM_REASON_WEIGHTS: dict[str, float] = {
    "normal": 1.0,
    "hard_or_borderline_source_bucket": 1.5,
    "safe_high_risk_score_one_model": 2.0,
    "safe_high_risk_score_two_models": 2.5,
    "active_policy_reject_strict_safe_overreject": 2.5,
    "active_sampling_one_model_reject_strict_safe": 2.0,
    "active_sampling_both_model_reject_strict_safe": 2.5,
    "reject_low_risk_score_one_model": 3.0,
    "reject_low_risk_score_two_models": 4.0,
    "active_policy_safe_reject_or_borderline_miss": 4.0,
    "active_sampling_one_model_safe_reject_or_borderline": 3.0,
    "active_sampling_both_model_safe_reject_or_borderline": 4.0,
    "active_teacher_split_vote": 3.0,
    "active_teacher_borderline": 3.0,
    "active_teacher_uncertain_reject_cap": 3.0,
}


def build_chunk_evidence_cache(
    *,
    classifier_manifest: Mapping[str, Any],
    stage_dirs: Iterable[str | Path],
    safe_parent_private_maps: Iterable[str | Path] = (),
    active_learning_private_maps: Iterable[str | Path] = (),
    source_schedule: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    row_provider: Any,
    cache_db: str | Path,
    output_json: str | Path | None = None,
    halo_chars: int = DEFAULT_HALO_CHARS,
    fetch_batch_size: int = 512,
    max_rows: int | None = None,
    progress_json: str | Path | None = None,
    require_private_cache_routes: Iterable[str] = (),
    require_active_learning_batches: Iterable[str] = (),
) -> dict[str, Any]:
    """Build or resume a private SQLite cache of core+halo evidence text."""

    validate_chunk_classifier_training_manifest(classifier_manifest)
    validate_public_no_raw_payload(classifier_manifest)
    if halo_chars < 0:
        raise ValueError("halo_chars must be nonnegative")
    if fetch_batch_size < 1:
        raise ValueError("fetch_batch_size must be positive")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when supplied")

    cache_path = Path(cache_db)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_private_sqlite_files(cache_path)
    manifest_hash = _required_sha256(str(classifier_manifest["training_manifest_hash"]), "classifier_manifest_hash")
    stage_index = _load_optional_stage_private_index(stage_dirs)
    safe_parent_index = _load_safe_parent_private_index(safe_parent_private_maps, classifier_manifest=classifier_manifest)
    active_learning_index = load_active_learning_private_index(
        active_learning_private_maps,
        classifier_manifest=classifier_manifest,
    )
    private_index = _merge_private_indexes(stage_index, safe_parent_index, active_learning_index)
    source_schedule_hash, schedule_by_entry, template_by_source_file = _load_combined_source_schedule_index(source_schedule)
    derived_stage_schedule_entry_count = _augment_schedule_index_from_stage_candidates(
        stage_index,
        schedule_by_entry=schedule_by_entry,
        template_by_source_file=template_by_source_file,
    )
    derived_safe_parent_schedule_entry_count = _augment_schedule_index_from_stage_candidates(
        safe_parent_index,
        schedule_by_entry=schedule_by_entry,
        template_by_source_file=template_by_source_file,
    )
    derived_active_learning_schedule_entry_count = _augment_schedule_index_from_stage_candidates(
        active_learning_index,
        schedule_by_entry=schedule_by_entry,
        template_by_source_file=template_by_source_file,
    )
    derived_schedule_entry_count = (
        derived_stage_schedule_entry_count
        + derived_safe_parent_schedule_entry_count
        + derived_active_learning_schedule_entry_count
    )
    rows = _cacheable_manifest_rows(
        classifier_manifest,
        stage_index=stage_index,
        safe_parent_index=safe_parent_index,
        active_learning_index=active_learning_index,
        schedule_by_entry=schedule_by_entry,
    )
    required_routes = {str(route) for route in require_private_cache_routes}
    required_active_batches = {str(batch) for batch in require_active_learning_batches}
    if max_rows is not None:
        if required_routes or required_active_batches:
            rows = _select_required_cache_canary_rows(
                rows,
                max_rows=int(max_rows),
                required_routes=required_routes,
                required_active_batches=required_active_batches,
            )
        else:
            rows = rows[: int(max_rows)]
    _validate_cache_canary_requirements(
        rows,
        required_routes=required_routes,
        required_active_batches=required_active_batches,
    )
    if not rows:
        raise ValueError("no cacheable chunk classifier rows")

    started = time.monotonic()
    with closing(sqlite3.connect(cache_path)) as con:
        _ensure_cache_schema(con)
        _validate_or_initialize_cache_metadata(
            con,
            classifier_manifest_hash=manifest_hash,
            source_schedule_hash=source_schedule_hash,
            halo_chars=halo_chars,
        )
        existing = _load_existing_cache_rows(con)
        for row in existing.values():
            _validate_private_cache_row(row)

        completed = 0
        total_rows = len(rows)
        completed_rows = 0
        for batch_index, batch in enumerate(_batches(rows, fetch_batch_size)):
            _write_cache_build_progress(
                progress_json,
                status="fetching",
                classifier_manifest_hash=manifest_hash,
                completed_rows=completed_rows,
                total_rows=total_rows,
                batch_index=batch_index,
                batch_row_count=len(batch),
                batch_source_group_count=_batch_source_group_count(batch, private_index=private_index),
            )
            missing_batch = [
                row
                for row in batch
                if str(row["candidate_id"]) not in existing
            ]
            if missing_batch:
                try:
                    extracted = _fetch_core_halo_texts(
                        missing_batch,
                        stage_index=private_index,
                        schedule_by_entry=schedule_by_entry,
                        row_provider=row_provider,
                        halo_chars=halo_chars,
                    )
                except Exception as exc:
                    _write_cache_build_progress(
                        progress_json,
                        status="failed",
                        classifier_manifest_hash=manifest_hash,
                        completed_rows=completed_rows,
                        total_rows=total_rows,
                        batch_index=batch_index,
                        batch_row_count=len(batch),
                        batch_source_group_count=_batch_source_group_count(batch, private_index=private_index),
                        error_type=type(exc).__name__,
                    )
                    raise
                by_candidate = {str(item["candidate_id"]): item for item in extracted}
                for manifest_row in missing_batch:
                    candidate_id = str(manifest_row["candidate_id"])
                    item = by_candidate.get(candidate_id)
                    if item is None:
                        raise ValueError("private evidence extraction missing candidate")
                    cache_row = _cache_row_from_extracted(manifest_row, item)
                    _insert_cache_row(con, cache_row)
                    existing[candidate_id] = cache_row
                con.commit()
            for manifest_row in batch:
                candidate_id = str(manifest_row["candidate_id"])
                cache_row = existing.get(candidate_id)
                if cache_row is None:
                    raise ValueError("private evidence cache missing completed candidate")
                _validate_cache_row_matches_manifest(cache_row, manifest_row)
                _validate_private_cache_row(cache_row)
                completed += 1
            completed_rows = completed
            _write_cache_build_progress(
                progress_json,
                status="building",
                classifier_manifest_hash=manifest_hash,
                completed_rows=completed_rows,
                total_rows=total_rows,
                batch_index=batch_index,
                batch_row_count=len(batch),
                batch_source_group_count=_batch_source_group_count(batch, private_index=private_index),
            )
        if completed != total_rows:
            raise ValueError("private evidence cache completion count mismatch")

        public_rows = sorted(
            (_public_cache_row(existing[str(row["candidate_id"])]) for row in rows),
            key=lambda row: str(row["candidate_id"]),
        )
        evidence_cache_hash = stable_json_hash(public_rows)
        _set_metadata(con, "evidence_cache_hash", evidence_cache_hash)
        _set_metadata(con, "row_count", str(len(rows)))
        _set_metadata(con, "status", "complete")
        con.commit()
    _chmod_private_sqlite_files(cache_path)

    label_counts = Counter(str(row["classifier_label"]) for row in public_rows)
    split_counts = Counter(str(row["split"]) for row in public_rows)
    evidence_tier_counts = Counter(str(row["evidence_tier"]) for row in public_rows)
    private_cache_route_counts = Counter(_private_cache_route_for_manifest_row(row) for row in rows)
    active_learning_batch_counts = Counter(
        str(row.get("active_learning_batch_id") or "")
        for row in rows
        if str(row.get("evidence_tier") or "") == ACTIVE_LEARNING_EVIDENCE_TIER
    )
    report = {
        "artifact_kind": CHUNK_EVIDENCE_CACHE_REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "classifier_manifest_hash": manifest_hash,
        "source_schedule_hash": source_schedule_hash,
        "evidence_cache_hash": evidence_cache_hash,
        "cache_db_path_hash": stable_json_hash({"cache_db_path": str(cache_path)}),
        "row_count": len(public_rows),
        "fit_eligible_row_count": sum(1 for row in public_rows if row["fit_eligible"]),
        "label_counts": _sorted_counter(label_counts),
        "split_counts": _sorted_counter(split_counts),
        "evidence_tier_counts": _sorted_counter(evidence_tier_counts),
        "private_cache_route_counts": _sorted_counter(private_cache_route_counts),
        "active_learning_batch_counts": _sorted_counter(active_learning_batch_counts),
        "required_private_cache_routes": sorted(required_routes),
        "required_active_learning_batches": sorted(required_active_batches),
        "halo_chars": int(halo_chars),
        "fetch_batch_size": int(fetch_batch_size),
        "max_rows_requested": None if max_rows is None else int(max_rows),
        "derived_schedule_entry_count": int(derived_schedule_entry_count),
        "derived_stage_schedule_entry_count": int(derived_stage_schedule_entry_count),
        "derived_safe_parent_schedule_entry_count": int(derived_safe_parent_schedule_entry_count),
        "derived_active_learning_schedule_entry_count": int(derived_active_learning_schedule_entry_count),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "provider_observations": (
            row_provider.observations()
            if hasattr(row_provider, "observations")
            else {"provider_class": type(row_provider).__name__, "raw_text_serialized": False, "raw_url_serialized": False, "token_serialized": False}
        ),
        "core_text_hashes_validated": True,
        "parent_text_hashes_validated": True,
        "core_halo_extraction_validated": True,
        "interrupted_resume_supported": True,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "paid_api_used": False,
        "gpu_rental_used": False,
    }
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_evidence_cache_report(report)
    if output_json is not None:
        _write_public_json(Path(output_json), report)
    _write_cache_build_progress(
        progress_json,
        status="complete",
        classifier_manifest_hash=manifest_hash,
        completed_rows=len(public_rows),
        total_rows=len(public_rows),
        batch_index=None,
        batch_row_count=0,
        batch_source_group_count=0,
        evidence_cache_hash=evidence_cache_hash,
    )
    return report


def validate_chunk_evidence_cache_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_EVIDENCE_CACHE_REPORT_KIND:
        raise ValueError("unsupported chunk evidence cache report kind")
    if report.get("status") != "complete":
        raise ValueError("chunk evidence cache report must be complete")
    for field in (
        "classifier_manifest_hash",
        "source_schedule_hash",
        "evidence_cache_hash",
        "cache_db_path_hash",
    ):
        _required_sha256(str(report.get(field) or ""), field)
    if int(report.get("row_count", -1)) < 1:
        raise ValueError("chunk evidence cache report requires rows")
    if report.get("core_text_hashes_validated") is not True:
        raise ValueError("chunk evidence cache report must validate core text hashes")
    if report.get("core_halo_extraction_validated") is not True:
        raise ValueError("chunk evidence cache report must validate core+halo extraction")
    route_counts = dict(report.get("private_cache_route_counts") or {})
    for route in report.get("required_private_cache_routes") or []:
        if int(route_counts.get(str(route), 0)) < 1:
            raise ValueError("chunk evidence cache report missing required private cache route")
    active_batch_counts = dict(report.get("active_learning_batch_counts") or {})
    for batch_id in report.get("required_active_learning_batches") or []:
        if int(active_batch_counts.get(str(batch_id), 0)) < 1:
            raise ValueError("chunk evidence cache report missing required active-learning batch")
    _require_false_report_flags(report)
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("chunk evidence cache report hash mismatch")


def train_chunk_sparse_classifier(
    *,
    classifier_manifest: Mapping[str, Any],
    cache_db: str | Path,
    artifact_dir: str | Path,
    evidence_cache_report: Mapping[str, Any] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_features: int = DEFAULT_SPARSE_N_FEATURES,
    target_recall: float = DEFAULT_TARGET_RECALL,
    random_state: int = DEFAULT_RANDOM_STATE,
    alpha: float = 1e-4,
    curriculum: Mapping[str, Any] | None = None,
    max_train_batches: int | None = None,
) -> dict[str, Any]:
    """Train the chunk sparse reject classifier over cached core+halo text."""

    deps = _load_sparse_dependencies()
    validate_chunk_classifier_training_manifest(classifier_manifest)
    manifest_hash = _required_sha256(str(classifier_manifest["training_manifest_hash"]), "classifier_manifest_hash")
    if evidence_cache_report is not None:
        validate_chunk_evidence_cache_report(evidence_cache_report)
        if evidence_cache_report["classifier_manifest_hash"] != manifest_hash:
            raise ValueError("sparse training evidence cache report manifest hash mismatch")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0:
        raise ValueError("alpha must be finite and positive")

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    model_path = artifact_path / f"{CHUNK_SPARSE_MODEL_ID}_model.joblib"
    checkpoint_path = artifact_path / f"{CHUNK_SPARSE_MODEL_ID}_checkpoint.json"
    progress_path = artifact_path / f"{CHUNK_SPARSE_MODEL_ID}_progress.json"
    report_path = artifact_path / f"{CHUNK_SPARSE_MODEL_ID}_report.json"
    view = load_cache_backed_training_view(
        cache_db,
        classifier_manifest=classifier_manifest,
        evidence_cache_report=evidence_cache_report,
        context="chunk sparse",
    )
    curriculum_view = _resolve_curriculum_view(curriculum, view=view, classifier_manifest_hash=manifest_hash)
    fit_rows = view["fit_rows"]
    train_rows, calibration_rows, eval_rows = view["train_rows"], view["calibration_rows"], view["eval_rows"]
    train_rows = _deterministic_stratified_shuffle(train_rows, random_state=random_state)
    calibration_rows = _deterministic_stratified_shuffle(calibration_rows, random_state=random_state)
    eval_rows = _deterministic_stratified_shuffle(eval_rows, random_state=random_state)
    train_batches = _batches(train_rows, batch_size)
    feature_config = _sparse_feature_config(n_features=n_features)
    feature_config_hash = stable_json_hash(feature_config)
    training_config = {
        "model_id": CHUNK_SPARSE_MODEL_ID,
        "variant_id": curriculum_view["variant_id"],
        "alpha": float(alpha),
        "curriculum_hash": curriculum_view["curriculum_hash"],
        "uses_curriculum_weights": curriculum_view["uses_curriculum_weights"],
    }
    training_config_hash = stable_json_hash(training_config)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_chunk_sparse_report(report)
        _validate_report_file_hash(report, model_path)
        _validate_existing_chunk_sparse_report_matches_invocation(
            report,
            classifier_manifest_hash=manifest_hash,
            evidence_cache_hash=view["evidence_cache_hash"],
            target_recall=target_recall,
            batch_size=batch_size,
            n_features=n_features,
            random_state=random_state,
            alpha=alpha,
            feature_config_hash=feature_config_hash,
            training_config_hash=training_config_hash,
            curriculum_hash=curriculum_view["curriculum_hash"],
        )
        return report

    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        _validate_chunk_sparse_checkpoint(
            checkpoint,
            model_path=model_path,
            classifier_manifest_hash=manifest_hash,
            evidence_cache_hash=view["evidence_cache_hash"],
            feature_config_hash=feature_config_hash,
            training_config_hash=training_config_hash,
            batch_size=batch_size,
            total_train_batches=len(train_batches),
            random_state=random_state,
        )
        completed_train_batches = int(checkpoint["completed_train_batches"])
        model = deps["joblib"].load(model_path)
    else:
        completed_train_batches = 0
        model = deps["SGDClassifier"](
            loss="log_loss",
            penalty="l2",
            alpha=float(alpha),
            random_state=int(random_state),
            learning_rate="optimal",
            tol=None,
        )

    class_weights = _balanced_sample_weights([_label_to_int(row["classifier_label"]) for row in train_rows])
    max_batches_allowed = len(train_batches)
    if max_train_batches is not None:
        max_batches_allowed = min(max_batches_allowed, int(max_train_batches))
    started = time.monotonic()
    for batch_index in range(completed_train_batches, max_batches_allowed):
        batch = train_batches[batch_index]
        matrix = _cache_rows_to_csr(batch, n_features=n_features, deps=deps)
        labels = [_label_to_int(row["classifier_label"]) for row in batch]
        weights = [
            class_weights[label] * _curriculum_weight_for_row(row, curriculum_view=curriculum_view)
            for row, label in zip(batch, labels, strict=True)
        ]
        model.partial_fit(
            matrix,
            deps["numpy"].array(labels, dtype=deps["numpy"].int64),
            classes=deps["numpy"].array([0, 1], dtype=deps["numpy"].int64),
            sample_weight=deps["numpy"].array(weights, dtype=deps["numpy"].float64),
        )
        deps["joblib"].dump(model, model_path)
        completed_train_batches = batch_index + 1
        checkpoint = _chunk_sparse_checkpoint(
            completed_train_batches=completed_train_batches,
            total_train_batches=len(train_batches),
            classifier_manifest_hash=manifest_hash,
            evidence_cache_hash=view["evidence_cache_hash"],
            feature_config_hash=feature_config_hash,
            training_config_hash=training_config_hash,
            batch_size=batch_size,
            random_state=random_state,
            model_file_sha256=_sha256_file(model_path),
        )
        _write_public_json(checkpoint_path, checkpoint)
        _write_public_json(
            progress_path,
            _chunk_progress_payload(
                artifact_kind=CHUNK_SPARSE_PROGRESS_KIND,
                status="training",
                completed_train_batches=completed_train_batches,
                total_train_batches=len(train_batches),
                fit_eligible_row_count=len(fit_rows),
            ),
        )

    if completed_train_batches < len(train_batches):
        progress = _chunk_progress_payload(
            artifact_kind=CHUNK_SPARSE_PROGRESS_KIND,
            status="incomplete",
            completed_train_batches=completed_train_batches,
            total_train_batches=len(train_batches),
            fit_eligible_row_count=len(fit_rows),
        )
        _write_public_json(progress_path, progress)
        return progress

    calibration_scores, calibration_labels = _score_sparse_rows(calibration_rows, model=model, n_features=n_features, deps=deps)
    eval_scores, eval_labels = _score_sparse_rows(eval_rows, model=model, n_features=n_features, deps=deps)
    _require_finite_scores(calibration_scores, "sparse calibration")
    _require_finite_scores(eval_scores, "sparse eval")
    threshold = _select_threshold_for_recall(
        y_true=calibration_labels,
        y_score=calibration_scores,
        target_recall=target_recall,
    )
    calibration_metrics = binary_calibration_metrics(y_true=calibration_labels, y_score=calibration_scores, threshold=threshold)
    eval_metrics = binary_calibration_metrics(y_true=eval_labels, y_score=eval_scores, threshold=threshold)
    deps["joblib"].dump(model, model_path)
    model_file_sha256 = _sha256_file(model_path)
    report = {
        "artifact_kind": CHUNK_SPARSE_REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "model_id": CHUNK_SPARSE_MODEL_ID,
        "model_kind": "hashed_ngram_linear",
        "feature_code_version": CHUNK_SPARSE_FEATURE_CODE_VERSION,
        "classifier_manifest_hash": manifest_hash,
        "evidence_cache_hash": view["evidence_cache_hash"],
        "dataset_view_hash": view["dataset_view_hash"],
        "feature_config": feature_config,
        "feature_config_hash": feature_config_hash,
        "training_config": training_config,
        "training_config_hash": training_config_hash,
        "curriculum_hash": curriculum_view["curriculum_hash"],
        "curriculum_summary": curriculum_view["summary"],
        "alpha": float(alpha),
        "fit_eligible_row_count": len(fit_rows),
        "split_counts": view["split_counts"],
        "evidence_tier_counts": view["evidence_tier_counts"],
        "train_label_counts": _label_counts(train_rows),
        "calibration_label_counts": _label_counts(calibration_rows),
        "eval_label_counts": _label_counts(eval_rows),
        "batch_size": int(batch_size),
        "random_state": int(random_state),
        "completed_train_batches": completed_train_batches,
        "total_train_batches": len(train_batches),
        "threshold": threshold,
        "threshold_selection": {
            "split": "calibration",
            "target_reject_recall": float(target_recall),
            "direction": "reject_at_or_above",
            "eval_split_used_for_threshold_selection": False,
        },
        "calibration_metrics": calibration_metrics,
        "eval_metrics": eval_metrics,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "compute_backend": "local_cpu_sklearn_sgd",
        "model_file_name": model_path.name,
        "model_file_sha256": model_file_sha256,
        "model_numeric_hash": _sparse_model_numeric_hash(model, deps=deps),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "paid_api_used": False,
        "gpu_rental_used": False,
    }
    report["model_payload_hash"] = stable_json_hash(
        {
            "model_id": report["model_id"],
            "model_kind": report["model_kind"],
            "model_numeric_hash": report["model_numeric_hash"],
            "feature_config_hash": feature_config_hash,
            "training_config_hash": training_config_hash,
            "classifier_manifest_hash": manifest_hash,
            "evidence_cache_hash": view["evidence_cache_hash"],
            "dataset_view_hash": view["dataset_view_hash"],
        }
    )
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_sparse_report(report)
    _write_public_json(report_path, report)
    _write_public_json(
        progress_path,
        _chunk_progress_payload(
            artifact_kind=CHUNK_SPARSE_PROGRESS_KIND,
            status="complete",
            completed_train_batches=completed_train_batches,
            total_train_batches=len(train_batches),
            fit_eligible_row_count=len(fit_rows),
        ),
    )
    return report


def validate_chunk_sparse_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_SPARSE_REPORT_KIND:
        raise ValueError("unsupported chunk sparse report kind")
    if report.get("status") != "complete":
        raise ValueError("chunk sparse report must be complete")
    for field in ("classifier_manifest_hash", "evidence_cache_hash", "feature_config_hash", "model_file_sha256"):
        _required_sha256(str(report.get(field) or ""), field)
    if "training_config_hash" in report:
        _required_sha256(str(report.get("training_config_hash") or ""), "training_config_hash")
    if "curriculum_hash" in report:
        _required_sha256(str(report.get("curriculum_hash") or ""), "curriculum_hash")
    if report.get("model_id") != CHUNK_SPARSE_MODEL_ID:
        raise ValueError("chunk sparse report model id mismatch")
    if report.get("threshold_selection", {}).get("split") != "calibration":
        raise ValueError("chunk sparse threshold must be selected on calibration")
    if report.get("threshold_selection", {}).get("eval_split_used_for_threshold_selection") is not False:
        raise ValueError("chunk sparse eval split must not select threshold")
    if "dataset_view_hash" in report:
        _required_sha256(str(report.get("dataset_view_hash") or ""), "dataset_view_hash")
    _require_metric_block(report, "calibration_metrics")
    _require_metric_block(report, "eval_metrics")
    _validate_metric_counts_match_splits(report)
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("chunk sparse report hash mismatch")


def train_chunk_window_mil_classifier(
    *,
    classifier_manifest: Mapping[str, Any],
    cache_db: str | Path,
    artifact_dir: str | Path,
    evidence_cache_report: Mapping[str, Any] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    model_bundle_factory: Any | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    stride: int = 128,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    target_recall: float = DEFAULT_TARGET_RECALL,
    curriculum: Mapping[str, Any] | None = None,
    input_mode: str = WINDOW_INPUT_FULL_EVIDENCE,
    pooling_mode: str = WINDOW_POOLING_MAX,
    pooling_top_k: int = 1,
    halo_weight: float = 0.25,
    halo_logit_clamp: float = 2.0,
    device: str = "auto",
    require_cuda: bool = False,
    amp: bool = True,
    max_windows_per_device_batch: int = 512,
    document_microbatch_size: int | None = None,
    checkpoint_every_batches: int = 1,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    """Train the chunk ModernBERT Window-MIL classifier over cached evidence."""

    torch = _load_torch()
    validate_chunk_classifier_training_manifest(classifier_manifest)
    if int(classifier_manifest.get("schema_version", 0)) != CHUNK_CLASSIFIER_SCHEMA_VERSION_V2:
        raise ValueError("chunk Window-MIL v2 training requires schema_version=2")
    manifest_hash = _required_sha256(str(classifier_manifest["training_manifest_hash"]), "classifier_manifest_hash")
    if evidence_cache_report is not None:
        validate_chunk_evidence_cache_report(evidence_cache_report)
        if evidence_cache_report["classifier_manifest_hash"] != manifest_hash:
            raise ValueError("window-MIL evidence cache report manifest hash mismatch")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    if not 0 <= stride < max_length:
        raise ValueError("stride must be in [0, max_length)")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if checkpoint_every_batches < 1:
        raise ValueError("checkpoint_every_batches must be positive")
    if max_windows_per_device_batch < 1:
        raise ValueError("max_windows_per_device_batch must be positive")
    effective_document_microbatch_size = _effective_document_microbatch_size(
        document_microbatch_size,
        batch_size=batch_size,
    )
    _validate_window_variant_config(
        input_mode=input_mode,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        halo_weight=halo_weight,
        halo_logit_clamp=halo_logit_clamp,
    )

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    model_path = artifact_path / f"{CHUNK_WINDOW_MIL_MODEL_ID}_model.pt"
    checkpoint_path = artifact_path / f"{CHUNK_WINDOW_MIL_MODEL_ID}_checkpoint.pt"
    checkpoint_meta_path = artifact_path / f"{CHUNK_WINDOW_MIL_MODEL_ID}_checkpoint.json"
    progress_path = artifact_path / f"{CHUNK_WINDOW_MIL_MODEL_ID}_progress.json"
    report_path = artifact_path / f"{CHUNK_WINDOW_MIL_MODEL_ID}_report.json"

    view = load_cache_backed_training_view(
        cache_db,
        classifier_manifest=classifier_manifest,
        evidence_cache_report=evidence_cache_report,
        context="chunk window-MIL",
    )
    curriculum_view = _resolve_curriculum_view(curriculum, view=view, classifier_manifest_hash=manifest_hash)
    window_variant_config = _window_variant_config(
        input_mode=input_mode,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        halo_weight=halo_weight,
        halo_logit_clamp=halo_logit_clamp,
    )
    window_variant_config_hash = stable_json_hash(window_variant_config)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_chunk_window_mil_report(report)
        _validate_report_file_hash(report, model_path)
        _validate_existing_chunk_window_report_matches_invocation(
            report,
            classifier_manifest_hash=manifest_hash,
            evidence_cache_hash=view["evidence_cache_hash"],
            target_recall=target_recall,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_windows_per_device_batch=max_windows_per_device_batch,
            document_microbatch_size=effective_document_microbatch_size,
            checkpoint_every_batches=checkpoint_every_batches,
            window_variant_config_hash=window_variant_config_hash,
            curriculum_hash=curriculum_view["curriculum_hash"],
        )
        return report
    fit_rows = view["fit_rows"]
    train_rows, calibration_rows, eval_rows = view["train_rows"], view["calibration_rows"], view["eval_rows"]
    train_rows = _deterministic_stratified_shuffle(train_rows, random_state=DEFAULT_RANDOM_STATE)
    calibration_rows = _deterministic_stratified_shuffle(calibration_rows, random_state=DEFAULT_RANDOM_STATE)
    eval_rows = _deterministic_stratified_shuffle(eval_rows, random_state=DEFAULT_RANDOM_STATE)
    train_batches = _batches(train_rows, batch_size)
    total_train_batches = len(train_batches) * int(epochs)

    resolved_device = _resolve_device(torch, device=device, require_cuda=require_cuda)
    factory = model_bundle_factory or default_model_bundle_factory
    tokenizer, model, resolved_model_name = factory(model_name=model_name, torch_module=torch)
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    scaler = _make_grad_scaler(torch, enabled=amp and resolved_device.type == "cuda")
    autocast = _autocast_context(torch, enabled=amp and resolved_device.type == "cuda")
    completed_train_batches = 0
    if checkpoint_path.exists() and checkpoint_meta_path.exists():
        _validate_chunk_window_checkpoint_meta(
            json.loads(checkpoint_meta_path.read_text(encoding="utf-8")),
            checkpoint_path=checkpoint_path,
            classifier_manifest_hash=manifest_hash,
            evidence_cache_hash=view["evidence_cache_hash"],
            model_name=resolved_model_name,
            max_length=max_length,
            stride=stride,
            batch_size=batch_size,
            epochs=epochs,
            total_train_batches=total_train_batches,
            document_microbatch_size=effective_document_microbatch_size,
            window_variant_config_hash=window_variant_config_hash,
            curriculum_hash=curriculum_view["curriculum_hash"],
        )
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and hasattr(scaler, "load_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        _restore_chunk_window_rng_state(torch=torch, checkpoint=checkpoint, resolved_device=resolved_device)
        completed_train_batches = int(checkpoint["completed_train_batches"])

    class_weights = _balanced_sample_weights([_label_to_int(row["classifier_label"]) for row in train_rows])
    max_batches_allowed = total_train_batches
    if max_train_batches is not None:
        max_batches_allowed = min(max_batches_allowed, int(max_train_batches))
    loss_sum = 0.0
    trained_rows = 0
    train_step_seconds = 0.0
    checkpoint_write_seconds = 0.0
    started = time.monotonic()
    for global_batch_index in range(completed_train_batches, max_batches_allowed):
        batch_index = global_batch_index % len(train_batches)
        epoch_index = global_batch_index // len(train_batches)
        batch = train_batches[batch_index]
        labels = [_label_to_int(row["classifier_label"]) for row in batch]
        weights = [
            class_weights[label] * _curriculum_weight_for_row(row, curriculum_view=curriculum_view)
            for row, label in zip(batch, labels, strict=True)
        ]
        step_started = time.monotonic()
        batch_loss = _train_window_batch_for_input_mode(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scaler=scaler,
            autocast=autocast,
            rows=batch,
            labels=labels,
            sample_weights=weights,
            device=resolved_device,
            max_length=max_length,
            stride=stride,
            max_windows_per_device_batch=max_windows_per_device_batch,
            document_microbatch_size=effective_document_microbatch_size,
            window_variant_config=window_variant_config,
        )
        train_step_seconds += time.monotonic() - step_started
        batch_loss = float(batch_loss)
        if not math.isfinite(batch_loss):
            raise ValueError("non-finite training loss")
        loss_sum += batch_loss * len(batch)
        trained_rows += len(batch)
        completed_train_batches = global_batch_index + 1
        if (
            completed_train_batches % checkpoint_every_batches == 0
            or completed_train_batches == max_batches_allowed
            or completed_train_batches == total_train_batches
        ):
            checkpoint_started = time.monotonic()
            _save_chunk_window_checkpoint(
                torch=torch,
                checkpoint_path=checkpoint_path,
                checkpoint_meta_path=checkpoint_meta_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                completed_train_batches=completed_train_batches,
                total_train_batches=total_train_batches,
                classifier_manifest_hash=manifest_hash,
                evidence_cache_hash=view["evidence_cache_hash"],
                model_name=resolved_model_name,
                max_length=max_length,
                stride=stride,
                batch_size=batch_size,
                epochs=epochs,
                epoch_index=epoch_index,
                batch_index=batch_index,
                document_microbatch_size=effective_document_microbatch_size,
                window_variant_config_hash=window_variant_config_hash,
                curriculum_hash=curriculum_view["curriculum_hash"],
            )
            checkpoint_write_seconds += time.monotonic() - checkpoint_started
        _write_public_json(
            progress_path,
            _chunk_progress_payload(
                artifact_kind=CHUNK_WINDOW_MIL_PROGRESS_KIND,
                status="training",
                completed_train_batches=completed_train_batches,
                total_train_batches=total_train_batches,
                fit_eligible_row_count=len(fit_rows),
                mean_train_loss=_mean_train_loss(loss_sum, trained_rows),
            ),
        )

    if completed_train_batches < total_train_batches:
        progress = _chunk_progress_payload(
            artifact_kind=CHUNK_WINDOW_MIL_PROGRESS_KIND,
            status="incomplete",
            completed_train_batches=completed_train_batches,
            total_train_batches=total_train_batches,
            fit_eligible_row_count=len(fit_rows),
            mean_train_loss=_mean_train_loss(loss_sum, trained_rows),
        )
        _write_public_json(progress_path, progress)
        return progress

    calibration_scores, calibration_labels = _score_window_rows(
        calibration_rows,
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        max_length=max_length,
        stride=stride,
        batch_size=batch_size,
        max_windows_per_device_batch=max_windows_per_device_batch,
        max_batches=max_eval_batches,
        window_variant_config=window_variant_config,
    )
    eval_scores, eval_labels = _score_window_rows(
        eval_rows,
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        max_length=max_length,
        stride=stride,
        batch_size=batch_size,
        max_windows_per_device_batch=max_windows_per_device_batch,
        max_batches=max_eval_batches,
        window_variant_config=window_variant_config,
    )
    _require_finite_scores(calibration_scores, "window-MIL calibration")
    _require_finite_scores(eval_scores, "window-MIL eval")
    threshold = _select_threshold_for_recall(
        y_true=calibration_labels,
        y_score=calibration_scores,
        target_recall=target_recall,
    )
    calibration_metrics = binary_calibration_metrics(y_true=calibration_labels, y_score=calibration_scores, threshold=threshold)
    eval_metrics = binary_calibration_metrics(y_true=eval_labels, y_score=eval_scores, threshold=threshold)
    torch.save({"model_state_dict": model.state_dict(), "model_name": resolved_model_name}, model_path)
    model_file_sha256 = _sha256_file(model_path)
    report = {
        "artifact_kind": CHUNK_WINDOW_MIL_REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "model_id": CHUNK_WINDOW_MIL_MODEL_ID,
        "model_kind": "transformer_window_mil",
        "feature_code_version": CHUNK_WINDOW_FEATURE_CODE_VERSION,
        "window_feature_code_version": CHUNK_WINDOW_FEATURE_CODE_VERSION_V2,
        "classifier_manifest_hash": manifest_hash,
        "evidence_cache_hash": view["evidence_cache_hash"],
        "dataset_view_hash": view["dataset_view_hash"],
        "base_model_name": resolved_model_name,
        "base_model_name_hash": stable_json_hash({"base_model_name": resolved_model_name}),
        "window_variant_config": window_variant_config,
        "window_variant_config_hash": window_variant_config_hash,
        "curriculum_hash": curriculum_view["curriculum_hash"],
        "curriculum_summary": curriculum_view["summary"],
        "fit_eligible_row_count": len(fit_rows),
        "split_counts": view["split_counts"],
        "evidence_tier_counts": view["evidence_tier_counts"],
        "train_label_counts": _label_counts(train_rows),
        "calibration_label_counts": _label_counts(calibration_rows),
        "eval_label_counts": _label_counts(eval_rows),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "stride": int(stride),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "completed_train_batches": completed_train_batches,
        "total_train_batches": total_train_batches,
        "mean_train_loss": _mean_train_loss(loss_sum, trained_rows),
        "threshold": threshold,
        "threshold_selection": {
            "split": "calibration",
            "target_reject_recall": float(target_recall),
            "direction": "reject_at_or_above",
            "eval_split_used_for_threshold_selection": False,
        },
        "calibration_metrics": calibration_metrics,
        "eval_metrics": eval_metrics,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "device": _device_observations(torch, resolved_device),
        "amp_enabled": bool(amp and resolved_device.type == "cuda"),
        "max_windows_per_device_batch": int(max_windows_per_device_batch),
        "document_microbatch_size": int(effective_document_microbatch_size),
        "activation_checkpoint_windows": CHUNK_WINDOW_ACTIVATION_CHECKPOINT_WINDOWS,
        "checkpoint_every_batches": int(checkpoint_every_batches),
        "runtime_profile": _window_runtime_profile(
            trained_rows=trained_rows,
            completed_train_batches=completed_train_batches,
            elapsed_seconds=time.monotonic() - started,
            train_step_seconds=train_step_seconds,
            checkpoint_write_seconds=checkpoint_write_seconds,
            rows=fit_rows,
            max_length=max_length,
            stride=stride,
            resolved_device_type=str(resolved_device.type),
        ),
        "compute_backend": "chunk_window_mil_gpu_or_local_same_entrypoint",
        "model_file_name": model_path.name,
        "model_file_sha256": model_file_sha256,
        "model_numeric_hash": _torch_model_numeric_hash(torch, model),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "paid_api_used": False,
        "gpu_rental_used": resolved_device.type == "cuda",
    }
    report["model_payload_hash"] = stable_json_hash(
        {
            "model_id": report["model_id"],
            "model_kind": report["model_kind"],
            "model_numeric_hash": report["model_numeric_hash"],
            "classifier_manifest_hash": manifest_hash,
            "evidence_cache_hash": view["evidence_cache_hash"],
            "dataset_view_hash": view["dataset_view_hash"],
            "base_model_name_hash": report["base_model_name_hash"],
            "window_variant_config_hash": window_variant_config_hash,
            "curriculum_hash": curriculum_view["curriculum_hash"],
        }
    )
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_window_mil_report(report)
    _write_public_json(report_path, report)
    _write_public_json(
        progress_path,
        _chunk_progress_payload(
            artifact_kind=CHUNK_WINDOW_MIL_PROGRESS_KIND,
            status="complete",
            completed_train_batches=completed_train_batches,
            total_train_batches=total_train_batches,
            fit_eligible_row_count=len(fit_rows),
            mean_train_loss=_mean_train_loss(loss_sum, trained_rows),
        ),
    )
    return report


def _effective_document_microbatch_size(document_microbatch_size: int | None, *, batch_size: int) -> int:
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if document_microbatch_size is None:
        return batch_size
    value = int(document_microbatch_size)
    if value < 1:
        raise ValueError("document_microbatch_size must be positive")
    if value > batch_size:
        raise ValueError("document_microbatch_size must not exceed batch_size")
    return value


def validate_chunk_window_mil_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_WINDOW_MIL_REPORT_KIND:
        raise ValueError("unsupported chunk window-MIL report kind")
    if report.get("status") != "complete":
        raise ValueError("chunk window-MIL report must be complete")
    for field in ("classifier_manifest_hash", "evidence_cache_hash", "model_file_sha256", "model_numeric_hash"):
        _required_sha256(str(report.get(field) or ""), field)
    if report.get("model_id") != CHUNK_WINDOW_MIL_MODEL_ID:
        raise ValueError("chunk window-MIL report model id mismatch")
    if report.get("threshold_selection", {}).get("split") != "calibration":
        raise ValueError("chunk window-MIL threshold must be selected on calibration")
    if report.get("threshold_selection", {}).get("eval_split_used_for_threshold_selection") is not False:
        raise ValueError("chunk window-MIL eval split must not select threshold")
    _require_metric_block(report, "calibration_metrics")
    _require_metric_block(report, "eval_metrics")
    _validate_metric_counts_match_splits(report)
    mean_loss = float(report.get("mean_train_loss", float("nan")))
    if not math.isfinite(mean_loss):
        raise ValueError("chunk window-MIL mean_train_loss must be finite")
    if "document_microbatch_size" in report:
        document_microbatch_size = int(report.get("document_microbatch_size", 0))
        batch_size = int(report.get("batch_size", 0))
        if document_microbatch_size < 1 or batch_size < 1 or document_microbatch_size > batch_size:
            raise ValueError("chunk window-MIL document_microbatch_size invalid")
    if report.get("activation_checkpoint_windows") is not True:
        raise ValueError("chunk window-MIL report must confirm activation checkpointing")
    if "dataset_view_hash" in report:
        _required_sha256(str(report.get("dataset_view_hash") or ""), "dataset_view_hash")
    if "window_variant_config_hash" in report:
        _required_sha256(str(report.get("window_variant_config_hash") or ""), "window_variant_config_hash")
        config = report.get("window_variant_config")
        if not isinstance(config, Mapping):
            raise ValueError("chunk window-MIL report missing window_variant_config")
        _validate_window_variant_config(
            input_mode=str(config.get("input_mode") or ""),
            pooling_mode=str(config.get("pooling_mode") or ""),
            pooling_top_k=int(config.get("pooling_top_k", 0)),
            halo_weight=float(config.get("halo_weight", float("nan"))),
            halo_logit_clamp=float(config.get("halo_logit_clamp", float("nan"))),
        )
        if report["window_variant_config_hash"] != stable_json_hash(config):
            raise ValueError("chunk window-MIL variant config hash mismatch")
    if "curriculum_hash" in report:
        _required_sha256(str(report.get("curriculum_hash") or ""), "curriculum_hash")
    if "runtime_profile" in report:
        _validate_runtime_profile(report["runtime_profile"])
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("chunk window-MIL report hash mismatch")


def build_chunk_classifier_freeze_report(
    *,
    classifier_manifest: Mapping[str, Any],
    evidence_cache_report: Mapping[str, Any],
    sparse_report: Mapping[str, Any],
    window_mil_report: Mapping[str, Any],
    output_json: str | Path | None = None,
    target_recall: float = DEFAULT_TARGET_RECALL,
) -> dict[str, Any]:
    """Bind completed chunk classifier reports into a freeze-readiness report."""

    validate_chunk_classifier_training_manifest(classifier_manifest)
    validate_chunk_evidence_cache_report(evidence_cache_report)
    validate_chunk_sparse_report(sparse_report)
    validate_chunk_window_mil_report(window_mil_report)
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    manifest_hash = str(classifier_manifest["training_manifest_hash"])
    evidence_cache_hash = str(evidence_cache_report["evidence_cache_hash"])
    for label, report in (("sparse", sparse_report), ("window-MIL", window_mil_report)):
        if report.get("classifier_manifest_hash") != manifest_hash:
            raise ValueError(f"{label} classifier manifest hash mismatch")
        if report.get("evidence_cache_hash") != evidence_cache_hash:
            raise ValueError(f"{label} evidence cache hash mismatch")
        if report.get("threshold_selection", {}).get("eval_split_used_for_threshold_selection") is not False:
            raise ValueError(f"{label} eval split was used for threshold selection")

    gates = {
        "sparse_eval_reject_recall": float(sparse_report["eval_metrics"]["recall"]) >= float(target_recall),
        "window_mil_eval_reject_recall": float(window_mil_report["eval_metrics"]["recall"]) >= float(target_recall),
        "sparse_scores_finite": True,
        "window_mil_scores_finite": True,
        "same_classifier_manifest": True,
        "same_evidence_cache": True,
        "thresholds_from_calibration_only": True,
        "no_raw_public_reports": True,
    }
    status = "ready" if all(gates.values()) else "blocked_metrics"
    report = {
        "artifact_kind": CHUNK_FREEZE_REPORT_KIND,
        "schema_version": 1,
        "status": status,
        "classifier_manifest_hash": manifest_hash,
        "evidence_cache_report_hash": str(evidence_cache_report["report_hash"]),
        "evidence_cache_hash": evidence_cache_hash,
        "sparse_report_hash": str(sparse_report["report_hash"]),
        "sparse_model_file_sha256": str(sparse_report["model_file_sha256"]),
        "sparse_model_payload_hash": str(sparse_report["model_payload_hash"]),
        "window_mil_report_hash": str(window_mil_report["report_hash"]),
        "window_mil_model_file_sha256": str(window_mil_report["model_file_sha256"]),
        "window_mil_model_payload_hash": str(window_mil_report["model_payload_hash"]),
        "target_reject_recall": float(target_recall),
        "metric_gates": gates,
        "decision_rule": {
            "reject": "reject_if_sparse_or_window_mil_score_at_or_above_frozen_threshold",
            "quarantine": "quarantine_on_uncertainty_disagreement_malformed_input_missing_evidence_or_ood",
            "allow": "allow_only_if_all_required_gates_pass_downstream",
            "classifier_may_approve_alone": False,
        },
        "thresholds": {
            "sparse": float(sparse_report["threshold"]),
            "window_mil": float(window_mil_report["threshold"]),
        },
        "eval_metrics": {
            "sparse": dict(sparse_report["eval_metrics"]),
            "window_mil": dict(window_mil_report["eval_metrics"]),
        },
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "paid_api_used": False,
        "gpu_rental_used": bool(window_mil_report.get("gpu_rental_used") is True),
    }
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_classifier_freeze_report(report)
    if output_json is not None:
        _write_public_json(Path(output_json), report)
    return report


def validate_chunk_classifier_freeze_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_FREEZE_REPORT_KIND:
        raise ValueError("unsupported chunk classifier freeze report kind")
    if report.get("status") not in {"ready", "blocked_metrics"}:
        raise ValueError("chunk classifier freeze status must be ready or blocked_metrics")
    for field in (
        "classifier_manifest_hash",
        "evidence_cache_report_hash",
        "evidence_cache_hash",
        "sparse_report_hash",
        "sparse_model_file_sha256",
        "sparse_model_payload_hash",
        "window_mil_report_hash",
        "window_mil_model_file_sha256",
        "window_mil_model_payload_hash",
    ):
        _required_sha256(str(report.get(field) or ""), field)
    _require_false_report_flags(report, allow_gpu_rental=True)
    gates = report.get("metric_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("chunk classifier freeze report requires metric gates")
    if report.get("status") == "ready" and not all(bool(value) for value in gates.values()):
        raise ValueError("chunk classifier freeze report ready status has failed gates")
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("chunk classifier freeze report hash mismatch")


def _cacheable_manifest_rows(
    classifier_manifest: Mapping[str, Any],
    *,
    stage_index: Mapping[str, Mapping[str, Any]],
    safe_parent_index: Mapping[str, Mapping[str, Any]],
    active_learning_index: Mapping[str, Mapping[str, Any]],
    schedule_by_entry: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows = []
    for row in classifier_manifest.get("rows") or []:
        if row.get("fit_eligible") is not True or str(row.get("split")) == "audit":
            continue
        tier = str(row.get("evidence_tier") or "")
        candidate_id = str(row.get("candidate_id") or "")
        route = _private_cache_route_for_manifest_row(row)
        if route == "rejected_chunk_stage_private_map":
            if tier not in {STRONG_EVIDENCE_TIER, FRESH_REPAIR_EVIDENCE_TIER}:
                raise ValueError("stage private cache route only supports labelled rejected-parent chunks")
            if candidate_id not in stage_index:
                raise ValueError("private evidence cache missing stage candidate")
            candidate = stage_index[candidate_id]["candidate"]
        elif route == "safe_parent_source_refetch":
            if tier != INHERITED_PARENT_SAFE_EVIDENCE_TIER:
                raise ValueError("safe-parent private cache route only supports inherited-safe rows")
            if candidate_id not in safe_parent_index:
                raise ValueError("private evidence cache missing inherited safe candidate")
            candidate = safe_parent_index[candidate_id]["candidate"]
        elif route == ACTIVE_LEARNING_PRIVATE_CACHE_ROUTE:
            if tier != ACTIVE_LEARNING_EVIDENCE_TIER:
                raise ValueError("active-learning private cache route only supports active-learning rows")
            if candidate_id not in active_learning_index:
                raise ValueError("private evidence cache missing active-learning private candidate")
            candidate = active_learning_index[candidate_id]["candidate"]
        else:
            raise ValueError("unsupported private evidence cache route")
        if str(candidate.get("schedule_entry_hash") or "") not in schedule_by_entry:
            raise ValueError("private evidence cache missing source schedule entry")
        rows.append(row)
    private_index = _merge_private_indexes(stage_index, safe_parent_index, active_learning_index)
    return sorted(rows, key=lambda row: _cache_fetch_sort_key(row, private_index=private_index))


def _select_required_cache_canary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_rows: int,
    required_routes: set[str],
    required_active_batches: set[str],
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    selected_ids: set[str] = set()

    def add_first(predicate: Any, missing_message: str) -> None:
        for row in rows:
            candidate_id = str(row["candidate_id"])
            if candidate_id in selected_ids:
                continue
            if predicate(row):
                selected.append(row)
                selected_ids.add(candidate_id)
                return
        raise ValueError(missing_message)

    for route in sorted(required_routes):
        if any(_private_cache_route_for_manifest_row(row) == route for row in selected):
            continue
        add_first(
            lambda row, route=route: _private_cache_route_for_manifest_row(row) == route,
            f"missing required private cache route for canary: {route}",
        )
    for batch_id in sorted(required_active_batches):
        if any(str(row.get("active_learning_batch_id") or "") == batch_id for row in selected):
            continue
        add_first(
            lambda row, batch_id=batch_id: str(row.get("active_learning_batch_id") or "") == batch_id,
            f"missing required active-learning batch for canary: {batch_id}",
        )
    if len(selected) > max_rows:
        raise ValueError("max_rows too small for required private cache canary coverage")
    for row in rows:
        if len(selected) >= max_rows:
            break
        candidate_id = str(row["candidate_id"])
        if candidate_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(candidate_id)
    return selected


def _validate_cache_canary_requirements(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_routes: set[str],
    required_active_batches: set[str],
) -> None:
    route_counts = Counter(_private_cache_route_for_manifest_row(row) for row in rows)
    for route in sorted(required_routes):
        if int(route_counts.get(route, 0)) < 1:
            raise ValueError(f"required private cache route not exercised: {route}")
    batch_counts = Counter(
        str(row.get("active_learning_batch_id") or "")
        for row in rows
        if str(row.get("evidence_tier") or "") == ACTIVE_LEARNING_EVIDENCE_TIER
    )
    for batch_id in sorted(required_active_batches):
        if int(batch_counts.get(batch_id, 0)) < 1:
            raise ValueError(f"required active-learning batch not exercised: {batch_id}")


def _cache_fetch_sort_key(row: Mapping[str, Any], *, private_index: Mapping[str, Mapping[str, Any]]) -> tuple[str, int, int, str]:
    candidate = private_index[str(row["candidate_id"])]["candidate"]
    return (
        str(candidate.get("source_file_id") or ""),
        int(candidate.get("row_group_index") or 0),
        int(candidate.get("row_group_local_row_index") or 0),
        str(row["candidate_id"]),
    )


def _batch_source_group_count(
    rows: Sequence[Mapping[str, Any]],
    *,
    private_index: Mapping[str, Mapping[str, Any]],
) -> int:
    groups = set()
    for row in rows:
        candidate = private_index[str(row["candidate_id"])]["candidate"]
        groups.add((str(candidate.get("source_file_id") or ""), int(candidate.get("row_group_index") or 0)))
    return len(groups)


def _load_optional_stage_private_index(stage_dirs: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    dirs = [Path(path) for path in stage_dirs]
    if not dirs:
        return {}
    return _load_stage_private_index(dirs)


def _load_safe_parent_private_index(
    safe_parent_private_maps: Iterable[str | Path],
    *,
    classifier_manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    paths = [Path(path) for path in safe_parent_private_maps]
    manifest_rows = {
        str(row["candidate_id"]): row
        for row in classifier_manifest.get("rows") or []
        if str(row.get("evidence_tier") or "") == INHERITED_PARENT_SAFE_EVIDENCE_TIER
    }
    if not paths:
        return {}
    index: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=lambda item: str(item)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_public_no_raw_payload(payload)
        if payload.get("artifact_kind") != "twc_inherited_safe_parent_private_map":
            raise ValueError("unsupported inherited safe private map kind")
        if payload.get("private_map_route") != "safe_parent_source_refetch":
            raise ValueError("inherited safe private map route mismatch")
        chunks = list(payload.get("chunks") or [])
        if int(payload.get("chunk_count", -1)) != len(chunks):
            raise ValueError("inherited safe private map chunk_count mismatch")
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise ValueError("inherited safe private map chunk must be an object")
            candidate_id = _required_sha256(str(chunk.get("candidate_id") or ""), "candidate_id")
            manifest_row = manifest_rows.get(candidate_id)
            if manifest_row is None:
                raise ValueError("inherited safe private map contains candidate outside manifest")
            if candidate_id in index:
                raise ValueError("duplicate inherited safe private candidate")
            private_chunk = {
                "candidate_id": candidate_id,
                "parent_candidate_id": _required_sha256(str(chunk.get("parent_candidate_id") or ""), "parent_candidate_id"),
                "chunk_text_sha256": _required_sha256(str(chunk.get("chunk_text_sha256") or ""), "chunk_text_sha256"),
                "parent_text_hash_sha256": _required_sha256(str(chunk.get("parent_text_hash_sha256") or ""), "parent_text_hash_sha256"),
                "parent_text_hash_algorithm": str(chunk.get("parent_text_hash_algorithm") or "utf8_sha256_v1"),
                "source_file_id": str(chunk.get("source_file_id") or ""),
                "source_schedule_hash": _required_sha256(str(chunk.get("source_schedule_hash") or ""), "source_schedule_hash"),
                "schedule_entry_hash": _required_sha256(str(chunk.get("schedule_entry_hash") or ""), "schedule_entry_hash"),
                "row_group_index": _nonnegative_int(chunk.get("row_group_index"), "row_group_index"),
                "row_group_local_row_index": _nonnegative_int(chunk.get("row_group_local_row_index"), "row_group_local_row_index"),
                "raw_char_start": _nonnegative_int(chunk.get("raw_char_start"), "raw_char_start"),
                "raw_char_end": _positive_int(chunk.get("raw_char_end"), "raw_char_end"),
                "block_start_index": _nonnegative_int(chunk.get("block_start_index", 0), "block_start_index"),
                "block_end_index_exclusive": _positive_int(chunk.get("block_end_index_exclusive", 1), "block_end_index_exclusive"),
            }
            if not private_chunk["source_file_id"]:
                raise ValueError("inherited safe private map missing source_file_id")
            if private_chunk["raw_char_end"] <= private_chunk["raw_char_start"]:
                raise ValueError("inherited safe private map span is empty")
            if private_chunk["parent_candidate_id"] != manifest_row["parent_candidate_id"]:
                raise ValueError("inherited safe private map parent_candidate_id drift")
            if private_chunk["chunk_text_sha256"] != manifest_row["text_hash_sha256"]:
                raise ValueError("inherited safe private map chunk hash drift")
            if private_chunk["parent_text_hash_sha256"] != manifest_row["parent_text_hash_sha256"]:
                raise ValueError("inherited safe private map parent hash drift")
            candidate_for_fetch = {
                **dict(manifest_row),
                "source_file_id": private_chunk["source_file_id"],
                "source_schedule_hash": private_chunk["source_schedule_hash"],
                "schedule_entry_hash": private_chunk["schedule_entry_hash"],
                "row_group_index": private_chunk["row_group_index"],
                "row_group_local_row_index": private_chunk["row_group_local_row_index"],
            }
            index[candidate_id] = {"candidate": candidate_for_fetch, "private_chunk": private_chunk, "private_map_hash": payload.get("private_map_hash")}
    return index


def _merge_private_indexes(
    *indexes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    merged: dict[str, Mapping[str, Any]] = {}
    for source in indexes:
        for candidate_id, bundle in source.items():
            if candidate_id in merged:
                raise ValueError("duplicate private cache candidate across routes")
            merged[candidate_id] = bundle
    return merged


def _private_cache_route_for_manifest_row(row: Mapping[str, Any]) -> str:
    route = str(row.get("private_cache_route") or "")
    if route:
        return route
    tier = str(row.get("evidence_tier") or "")
    if tier in {STRONG_EVIDENCE_TIER, FRESH_REPAIR_EVIDENCE_TIER}:
        return "rejected_chunk_stage_private_map"
    if tier == INHERITED_PARENT_SAFE_EVIDENCE_TIER:
        return "safe_parent_source_refetch"
    if tier == ACTIVE_LEARNING_EVIDENCE_TIER:
        return ACTIVE_LEARNING_PRIVATE_CACHE_ROUTE
    return ""


def _write_cache_build_progress(
    progress_json: str | Path | None,
    *,
    status: str,
    classifier_manifest_hash: str,
    completed_rows: int,
    total_rows: int,
    batch_index: int | None,
    batch_row_count: int,
    batch_source_group_count: int,
    private_cache_route_counts: Mapping[str, int] | None = None,
    error_type: str | None = None,
    evidence_cache_hash: str | None = None,
) -> None:
    if progress_json is None:
        return
    payload = {
        "artifact_kind": "twc_chunk_classifier_evidence_cache_progress",
        "schema_version": 1,
        "status": status,
        "classifier_manifest_hash": classifier_manifest_hash,
        "completed_rows": int(completed_rows),
        "total_rows": int(total_rows),
        "batch_index": None if batch_index is None else int(batch_index),
        "batch_row_count": int(batch_row_count),
        "batch_source_group_count": int(batch_source_group_count),
        "private_cache_route_counts": _sorted_counter(Counter(private_cache_route_counts or {})),
        "error_type": error_type,
        "evidence_cache_hash": evidence_cache_hash,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    payload["progress_hash"] = _hash_without(payload, "progress_hash")
    _write_public_json(Path(progress_json), payload)


def _load_combined_source_schedule_index(
    source_schedule: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    schedules = _normalize_source_schedules(source_schedule)
    schedule_hashes: list[str] = []
    by_entry: dict[str, Mapping[str, Any]] = {}
    template_by_source_file: dict[str, Mapping[str, Any]] = {}
    template_locator_hash_by_source_file: dict[str, str] = {}
    for schedule in schedules:
        schedule_hash, entries, _templates = _load_source_schedule_index(schedule)
        schedule_hashes.append(schedule_hash)
        for entry_hash, entry in entries.items():
            existing = by_entry.get(entry_hash)
            if existing is not None and stable_json_hash(existing) != stable_json_hash(entry):
                raise ValueError("source schedules contain conflicting entry_hash")
            by_entry[entry_hash] = entry
            source_file_id = str(entry.get("source_file_id") or "")
            if not source_file_id:
                continue
            locator_hash = stable_json_hash(
                {
                    "parquet_url_or_path": str(entry.get("parquet_url_or_path") or ""),
                    "dataset_repo_id": str(entry.get("dataset_repo_id") or ""),
                    "config": str(entry.get("config") or entry.get("dataset_config") or ""),
                    "split": str(entry.get("split") or entry.get("dataset_split") or ""),
                    "resolved_repo_sha": str(entry.get("resolved_repo_sha") or ""),
                }
            )
            previous_hash = template_locator_hash_by_source_file.get(source_file_id)
            if previous_hash is not None and previous_hash != locator_hash:
                raise ValueError("source schedules contain conflicting source_file_id locators")
            template_locator_hash_by_source_file[source_file_id] = locator_hash
            template_by_source_file[source_file_id] = entry
    if not by_entry:
        raise ValueError("no source schedule entries found")
    if len(schedule_hashes) == 1:
        schedule_set_hash = schedule_hashes[0]
    else:
        schedule_set_hash = stable_json_hash({"source_schedule_hashes": sorted(schedule_hashes)})
    return schedule_set_hash, by_entry, template_by_source_file


def _normalize_source_schedules(source_schedule: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(source_schedule, Mapping):
        return [source_schedule]
    schedules = list(source_schedule)
    if not schedules:
        raise ValueError("at least one source schedule is required")
    for schedule in schedules:
        if not isinstance(schedule, Mapping):
            raise ValueError("source schedule must be a mapping")
    return schedules


def _augment_schedule_index_from_stage_candidates(
    stage_index: Mapping[str, Mapping[str, Any]],
    *,
    schedule_by_entry: dict[str, Mapping[str, Any]],
    template_by_source_file: Mapping[str, Mapping[str, Any]],
) -> int:
    grouped: dict[str, dict[str, Any]] = {}
    for bundle in stage_index.values():
        candidate = bundle["candidate"]
        entry_hash = str(candidate.get("schedule_entry_hash") or "")
        if not entry_hash or entry_hash in schedule_by_entry:
            continue
        source_file_id = str(candidate.get("source_file_id") or "")
        template = template_by_source_file.get(source_file_id)
        if template is None:
            continue
        row_group_index = int(candidate.get("row_group_index", -1))
        if row_group_index < 0:
            raise ValueError("stage candidate row_group_index is invalid")
        offset = int(candidate.get("row_group_local_row_index", -1))
        if offset < 0:
            raise ValueError("stage candidate row_group_local_row_index is invalid")
        existing = grouped.setdefault(
            entry_hash,
            {
                "template": template,
                "source_file_id": source_file_id,
                "row_group_index": row_group_index,
                "offsets": set(),
            },
        )
        if existing["source_file_id"] != source_file_id or int(existing["row_group_index"]) != row_group_index:
            raise ValueError("stage candidates disagree on derived schedule entry identity")
        existing["offsets"].add(offset)
    for entry_hash, bundle in grouped.items():
        template = dict(bundle["template"])
        entry = dict(template)
        entry["entry_hash"] = entry_hash
        entry["source_file_id"] = bundle["source_file_id"]
        entry["row_group_index"] = int(bundle["row_group_index"])
        entry["row_group_local_offsets"] = sorted(int(offset) for offset in bundle["offsets"])
        entry.pop("row_group_num_rows", None)
        entry["derived_from_same_source_file_template"] = True
        schedule_by_entry[entry_hash] = entry
    return len(grouped)


def _ensure_cache_schema(con: sqlite3.Connection) -> None:
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS rows (
            candidate_id TEXT PRIMARY KEY,
            candidate_hash TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            classifier_label TEXT NOT NULL,
            split TEXT NOT NULL,
            evidence_tier TEXT NOT NULL,
            fit_eligible INTEGER NOT NULL,
            eligible_for_eval INTEGER NOT NULL,
            training_weight REAL NOT NULL,
            core_text TEXT NOT NULL,
            evidence_text TEXT NOT NULL,
            core_text_sha256 TEXT NOT NULL,
            evidence_text_sha256 TEXT NOT NULL,
            core_char_length INTEGER NOT NULL,
            evidence_char_length INTEGER NOT NULL,
            halo_left_chars INTEGER NOT NULL,
            halo_right_chars INTEGER NOT NULL
        )
        """
    )
    con.commit()


def _validate_or_initialize_cache_metadata(
    con: sqlite3.Connection,
    *,
    classifier_manifest_hash: str,
    source_schedule_hash: str,
    halo_chars: int,
) -> None:
    metadata = _metadata(con)
    expected = {
        "artifact_kind": CHUNK_PRIVATE_EVIDENCE_CACHE_KIND,
        "schema_version": "1",
        "classifier_manifest_hash": classifier_manifest_hash,
        "source_schedule_hash": source_schedule_hash,
        "halo_chars": str(int(halo_chars)),
    }
    if not metadata:
        for key, value in expected.items():
            _set_metadata(con, key, value)
        _set_metadata(con, "status", "building")
        con.commit()
        return
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"private evidence cache metadata drift: {key}")


def _metadata(con: sqlite3.Connection) -> dict[str, str]:
    return {str(key): str(value) for key, value in con.execute("SELECT key, value FROM metadata").fetchall()}


def _set_metadata(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, value))


def _load_existing_cache_rows(con: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for values in con.execute(
        """
        SELECT candidate_id, candidate_hash, row_hash, classifier_label, split,
               evidence_tier, fit_eligible, eligible_for_eval, training_weight,
               core_text, evidence_text, core_text_sha256, evidence_text_sha256,
               core_char_length, evidence_char_length, halo_left_chars, halo_right_chars
        FROM rows
        """
    ):
        row = _cache_row_from_sql(values)
        if row["candidate_id"] in rows:
            raise ValueError("private evidence cache duplicate candidate")
        rows[row["candidate_id"]] = row
    return rows


def _cache_row_from_sql(values: Sequence[Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(values[0]),
        "candidate_hash": str(values[1]),
        "row_hash": str(values[2]),
        "classifier_label": str(values[3]),
        "split": str(values[4]),
        "evidence_tier": str(values[5]),
        "fit_eligible": bool(int(values[6])),
        "eligible_for_eval": bool(int(values[7])),
        "training_weight": float(values[8]),
        "core_text": str(values[9]),
        "evidence_text": str(values[10]),
        "core_text_sha256": str(values[11]),
        "evidence_text_sha256": str(values[12]),
        "core_char_length": int(values[13]),
        "evidence_char_length": int(values[14]),
        "halo_left_chars": int(values[15]),
        "halo_right_chars": int(values[16]),
    }


def _cache_row_from_extracted(manifest_row: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    if str(manifest_row["candidate_id"]) != str(item["candidate_id"]):
        raise ValueError("private evidence candidate mismatch")
    row = {
        "candidate_id": str(manifest_row["candidate_id"]),
        "candidate_hash": str(manifest_row["candidate_hash"]),
        "classifier_label": str(manifest_row["classifier_label"]),
        "split": str(manifest_row["split"]),
        "evidence_tier": str(manifest_row["evidence_tier"]),
        "fit_eligible": bool(manifest_row.get("fit_eligible") is True),
        "eligible_for_eval": bool(manifest_row.get("eligible_for_eval") is True),
        "training_weight": float(manifest_row.get("training_weight", 1.0)),
        "core_text": str(item["core_text"]),
        "evidence_text": str(item["evidence_text"]),
        "core_text_sha256": str(item["core_text_sha256"]),
        "evidence_text_sha256": str(item["evidence_text_sha256"]),
        "core_char_length": int(item["core_char_length"]),
        "evidence_char_length": int(item["evidence_char_length"]),
        "halo_left_chars": int(item["halo_left_chars"]),
        "halo_right_chars": int(item["halo_right_chars"]),
    }
    row["row_hash"] = _private_cache_row_hash(row)
    _validate_private_cache_row(row)
    _validate_cache_row_matches_manifest(row, manifest_row)
    return row


def _insert_cache_row(con: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    con.execute(
        """
        INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["candidate_id"],
            row["candidate_hash"],
            row["row_hash"],
            row["classifier_label"],
            row["split"],
            row["evidence_tier"],
            1 if row["fit_eligible"] else 0,
            1 if row["eligible_for_eval"] else 0,
            float(row["training_weight"]),
            row["core_text"],
            row["evidence_text"],
            row["core_text_sha256"],
            row["evidence_text_sha256"],
            int(row["core_char_length"]),
            int(row["evidence_char_length"]),
            int(row["halo_left_chars"]),
            int(row["halo_right_chars"]),
        ),
    )


def _validate_private_cache_row(row: Mapping[str, Any]) -> None:
    for field in ("candidate_id", "candidate_hash", "row_hash", "core_text_sha256", "evidence_text_sha256"):
        _required_sha256(str(row.get(field) or ""), field)
    if row.get("classifier_label") not in {"safe", "reject"}:
        raise ValueError("private evidence cache row label must be safe or reject")
    if row.get("split") not in {"train", "calibration", "eval"}:
        raise ValueError("private evidence cache row split must be train/calibration/eval")
    if row.get("evidence_tier") not in {
        STRONG_EVIDENCE_TIER,
        FRESH_REPAIR_EVIDENCE_TIER,
        INHERITED_PARENT_SAFE_EVIDENCE_TIER,
        ACTIVE_LEARNING_EVIDENCE_TIER,
    }:
        raise ValueError("private evidence cache row evidence tier unsupported")
    core_text = row.get("core_text")
    evidence_text = row.get("evidence_text")
    if not isinstance(core_text, str) or not core_text:
        raise ValueError("private evidence cache row missing core text")
    if not isinstance(evidence_text, str) or not evidence_text:
        raise ValueError("private evidence cache row missing evidence text")
    if _sha256_text(core_text) != row["core_text_sha256"]:
        raise ValueError("private evidence cache row hash drift")
    if _sha256_text(evidence_text) != row["evidence_text_sha256"]:
        raise ValueError("private evidence cache row hash drift")
    if int(row["core_char_length"]) != len(core_text):
        raise ValueError("private evidence cache row core length drift")
    if int(row["evidence_char_length"]) != len(evidence_text):
        raise ValueError("private evidence cache row evidence length drift")
    if int(row["halo_left_chars"]) < 0 or int(row["halo_right_chars"]) < 0:
        raise ValueError("private evidence cache row halo lengths must be nonnegative")
    if row.get("row_hash") != _private_cache_row_hash(row):
        raise ValueError("private evidence cache row hash drift")


def _validate_cache_row_matches_manifest(cache_row: Mapping[str, Any], manifest_row: Mapping[str, Any]) -> None:
    for field in ("candidate_id", "candidate_hash", "classifier_label", "split", "evidence_tier"):
        if str(cache_row[field]) != str(manifest_row[field]):
            raise ValueError(f"private evidence cache row {field} drift")
    if bool(cache_row["fit_eligible"]) != bool(manifest_row.get("fit_eligible") is True):
        raise ValueError("private evidence cache row fit_eligible drift")
    if bool(cache_row["eligible_for_eval"]) != bool(manifest_row.get("eligible_for_eval") is True):
        raise ValueError("private evidence cache row eligible_for_eval drift")
    if abs(float(cache_row["training_weight"]) - float(manifest_row.get("training_weight", 1.0))) > 1e-12:
        raise ValueError("private evidence cache row training_weight drift")
    if str(cache_row["core_text_sha256"]) != str(manifest_row["text_hash_sha256"]):
        raise ValueError("private evidence cache row core text hash drift")
    if str(manifest_row.get("evidence_tier") or "") == ACTIVE_LEARNING_EVIDENCE_TIER:
        if str(cache_row["evidence_text_sha256"]) != str(manifest_row.get("teacher_input_text_hash_sha256") or ""):
            raise ValueError("private evidence cache row active-learning teacher-input hash drift")


def _public_cache_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "candidate_hash": str(row["candidate_hash"]),
        "row_hash": str(row["row_hash"]),
        "classifier_label": str(row["classifier_label"]),
        "split": str(row["split"]),
        "evidence_tier": str(row["evidence_tier"]),
        "fit_eligible": bool(row["fit_eligible"]),
        "eligible_for_eval": bool(row["eligible_for_eval"]),
        "training_weight": float(row["training_weight"]),
        "core_text_sha256": str(row["core_text_sha256"]),
        "evidence_text_sha256": str(row["evidence_text_sha256"]),
        "core_char_length": int(row["core_char_length"]),
        "evidence_char_length": int(row["evidence_char_length"]),
        "halo_left_chars": int(row["halo_left_chars"]),
        "halo_right_chars": int(row["halo_right_chars"]),
    }


def _private_cache_row_hash(row: Mapping[str, Any]) -> str:
    return stable_json_hash(
        {
            key: row[key]
            for key in (
                "candidate_id",
                "candidate_hash",
                "classifier_label",
                "split",
                "evidence_tier",
                "fit_eligible",
                "eligible_for_eval",
                "training_weight",
                "core_text_sha256",
                "evidence_text_sha256",
                "core_char_length",
                "evidence_char_length",
                "halo_left_chars",
                "halo_right_chars",
            )
        }
    )


def _load_cache_for_training(
    cache_db: str | Path,
    *,
    classifier_manifest: Mapping[str, Any],
    evidence_cache_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest_hash = _required_sha256(str(classifier_manifest["training_manifest_hash"]), "classifier_manifest_hash")
    cache_path = Path(cache_db)
    if not cache_path.exists():
        raise ValueError("private evidence cache does not exist")
    with closing(sqlite3.connect(cache_path)) as con:
        _ensure_cache_schema(con)
        metadata = _metadata(con)
        if metadata.get("artifact_kind") != CHUNK_PRIVATE_EVIDENCE_CACHE_KIND:
            raise ValueError("private evidence cache artifact kind mismatch")
        if metadata.get("classifier_manifest_hash") != manifest_hash:
            raise ValueError("private evidence cache classifier manifest hash mismatch")
        if metadata.get("status") != "complete":
            raise ValueError("private evidence cache is not complete")
        rows_by_candidate = _load_existing_cache_rows(con)
    rows = [_validate_and_return_row(row) for row in rows_by_candidate.values()]
    rows = sorted(rows, key=lambda row: row["candidate_id"])
    evidence_cache_hash = stable_json_hash([_public_cache_row(row) for row in rows])
    if metadata.get("evidence_cache_hash") != evidence_cache_hash:
        raise ValueError("private evidence cache hash mismatch")
    if int(metadata.get("row_count", -1)) != len(rows):
        raise ValueError("private evidence cache row_count mismatch")
    if evidence_cache_report is not None and evidence_cache_report["evidence_cache_hash"] != evidence_cache_hash:
        raise ValueError("private evidence cache report hash mismatch")

    cache_ids = {row["candidate_id"] for row in rows}
    expected_ids = {
        str(row["candidate_id"])
        for row in classifier_manifest.get("rows") or []
        if row.get("fit_eligible") is True and str(row.get("split")) != "audit"
    }
    if cache_ids != expected_ids:
        raise ValueError("private evidence cache candidate coverage mismatch")
    manifest_by_candidate = {str(row["candidate_id"]): row for row in classifier_manifest.get("rows") or []}
    for row in rows:
        _validate_cache_row_matches_manifest(row, manifest_by_candidate[row["candidate_id"]])
    return {"rows": rows, "evidence_cache_hash": evidence_cache_hash}


def load_cache_backed_training_view(
    cache_db: str | Path,
    *,
    classifier_manifest: Mapping[str, Any],
    evidence_cache_report: Mapping[str, Any] | None = None,
    context: str,
) -> dict[str, Any]:
    """Return one validated cache-backed split view for all chunk trainers.

    The returned rows contain private text and must not be serialized as a
    public artifact. Public report fields are emitted separately as counts and
    hashes.
    """

    validate_chunk_classifier_training_manifest(classifier_manifest)
    if evidence_cache_report is not None:
        validate_chunk_evidence_cache_report(evidence_cache_report)
    cache = _load_cache_for_training(
        cache_db,
        classifier_manifest=classifier_manifest,
        evidence_cache_report=evidence_cache_report,
    )
    rows = cache["rows"]
    fit_rows = [row for row in rows if row["fit_eligible"] and row["split"] != "audit"]
    train_rows, calibration_rows, eval_rows = _split_fit_rows(fit_rows, context=context)
    public_row_identities = [
        _training_row_identity(row)
        for row in sorted(fit_rows, key=lambda item: str(item["candidate_id"]))
    ]
    view = {
        "rows": rows,
        "fit_rows": fit_rows,
        "train_rows": train_rows,
        "calibration_rows": calibration_rows,
        "eval_rows": eval_rows,
        "evidence_cache_hash": cache["evidence_cache_hash"],
        "dataset_view_hash": stable_json_hash(public_row_identities),
        "row_identity_hashes": [row["row_identity_hash"] for row in public_row_identities],
        "label_counts": _label_counts(fit_rows),
        "split_counts": _split_counts(fit_rows),
        "evidence_tier_counts": _evidence_tier_counts(fit_rows),
    }
    return view


def build_chunk_variant_registry(
    *,
    classifier_manifest: Mapping[str, Any],
    evidence_cache_report: Mapping[str, Any],
    output_json: str | Path | None = None,
    artifact_root: str = "reports/classifiers/chunk_classifier_v2_multi_variant",
    target_recall: float = DEFAULT_TARGET_RECALL,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Pre-register the final sparse and Window-MIL variant set."""

    validate_chunk_classifier_training_manifest(classifier_manifest)
    validate_chunk_evidence_cache_report(evidence_cache_report)
    manifest_hash = str(classifier_manifest["training_manifest_hash"])
    if evidence_cache_report["classifier_manifest_hash"] != manifest_hash:
        raise ValueError("variant registry evidence cache manifest hash mismatch")
    if not 0 < float(target_recall) <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    artifact_root = _safe_relative_artifact_root(artifact_root)
    sparse_variants = [
        _sparse_variant("S0_sparse_baseline_repro", artifact_root=artifact_root, alpha=1e-4, uses_curriculum=False),
        _sparse_variant("S1_sparse_weighted_default", artifact_root=artifact_root, alpha=1e-4, uses_curriculum=True),
        _sparse_variant("S2_sparse_weighted_low_reg", artifact_root=artifact_root, alpha=3e-5, uses_curriculum=True),
        _sparse_variant("S3_sparse_weighted_high_reg", artifact_root=artifact_root, alpha=3e-4, uses_curriculum=True),
    ]
    window_variants = [
        _window_variant(
            "W0_current_max_baseline",
            artifact_root=artifact_root,
            input_mode=WINDOW_INPUT_FULL_EVIDENCE,
            pooling_mode=WINDOW_POOLING_MAX,
            pooling_top_k=1,
            uses_curriculum=False,
            adopt_existing_if_hashes_validate=True,
        ),
        _window_variant(
            "W1_full_evidence_topk3_weighted",
            artifact_root=artifact_root,
            input_mode=WINDOW_INPUT_FULL_EVIDENCE,
            pooling_mode=WINDOW_POOLING_TOPK_MEAN,
            pooling_top_k=3,
            uses_curriculum=True,
        ),
        _window_variant(
            "W2_core_only_topk3_weighted",
            artifact_root=artifact_root,
            input_mode=WINDOW_INPUT_CORE_ONLY,
            pooling_mode=WINDOW_POOLING_TOPK_MEAN,
            pooling_top_k=3,
            uses_curriculum=True,
        ),
        _window_variant(
            "W3_split_halo_capped_topk3_weighted",
            artifact_root=artifact_root,
            input_mode=WINDOW_INPUT_SPLIT_HALO_CAPPED,
            pooling_mode=WINDOW_POOLING_TOPK_MEAN,
            pooling_top_k=3,
            uses_curriculum=True,
        ),
    ]
    report = {
        "artifact_kind": CHUNK_VARIANT_REGISTRY_KIND,
        "schema_version": 1,
        "status": "complete",
        "classifier_manifest_hash": manifest_hash,
        "evidence_cache_hash": str(evidence_cache_report["evidence_cache_hash"]),
        "evidence_cache_report_hash": str(evidence_cache_report["report_hash"]),
        "target_reject_recall": float(target_recall),
        "random_state": int(random_state),
        "artifact_root": artifact_root,
        "selection_split": "calibration",
        "threshold_selection_objective": "maximize_accepted_core_token_lower_bound_after_reject_recall_and_slice_gates",
        "policy_selection_rule": "calibration_only_then_eval_once_for_reporting",
        "eval_split_used_for_variant_selection": False,
        "variant_set_locked_before_training": True,
        "sparse_variants": sparse_variants,
        "window_mil_variants": window_variants,
        "rejected_variant_families": [
            "noisy_or_pooling",
            "metadata_auxiliary_heads",
            "broad_encoder_family_sweep",
            "eval_mined_hard_examples",
            "split_changes",
            "production_yield_forced_training",
        ],
        **_training_public_false_flags(),
    }
    report["registry_hash"] = _hash_without(report, "registry_hash")
    validate_chunk_variant_registry(report)
    if output_json is not None:
        _write_public_json(Path(output_json), report)
    return report


def validate_chunk_variant_registry(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_VARIANT_REGISTRY_KIND:
        raise ValueError("unsupported chunk variant registry kind")
    if report.get("status") != "complete":
        raise ValueError("chunk variant registry must be complete")
    for field in ("classifier_manifest_hash", "evidence_cache_hash", "evidence_cache_report_hash"):
        _required_sha256(str(report.get(field) or ""), field)
    _safe_relative_artifact_root(str(report.get("artifact_root") or ""))
    if report.get("selection_split") != "calibration":
        raise ValueError("variant registry selection split must be calibration")
    if report.get("threshold_selection_objective") != "maximize_accepted_core_token_lower_bound_after_reject_recall_and_slice_gates":
        raise ValueError("variant registry threshold selection objective mismatch")
    if report.get("policy_selection_rule") != "calibration_only_then_eval_once_for_reporting":
        raise ValueError("variant registry policy selection rule mismatch")
    if report.get("eval_split_used_for_variant_selection") is not False:
        raise ValueError("eval must not select variants")
    variant_ids: list[str] = []
    for variant in list(report.get("sparse_variants") or []) + list(report.get("window_mil_variants") or []):
        if not isinstance(variant, Mapping):
            raise ValueError("variant registry variants must be objects")
        variant_id = str(variant.get("variant_id") or "")
        if not variant_id:
            raise ValueError("variant missing variant_id")
        variant_ids.append(variant_id)
        _required_sha256(str(variant.get("variant_hash") or ""), "variant_hash")
        _validate_variant_artifact_paths(variant)
        if variant.get("variant_hash") != _hash_without(variant, "variant_hash"):
            raise ValueError("variant hash mismatch")
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("variant registry duplicate variant id")
    _require_false_report_flags(report)
    if report.get("registry_hash") != _hash_without(report, "registry_hash"):
        raise ValueError("chunk variant registry hash mismatch")


def build_chunk_training_score_ledger(
    *,
    classifier_manifest: Mapping[str, Any],
    evidence_cache_report: Mapping[str, Any],
    sparse_report: Mapping[str, Any],
    window_mil_report: Mapping[str, Any],
    cache_db: str | Path,
    sparse_scores_by_candidate_id: Mapping[str, float] | None = None,
    window_mil_scores_by_candidate_id: Mapping[str, float] | None = None,
    sparse_model_path: str | Path | None = None,
    window_mil_model_path: str | Path | None = None,
    output_json: str | Path | None = None,
    device: str = "auto",
    require_cuda: bool = False,
    sparse_score_batch_size: int = 512,
    window_mil_score_batch_size: int = 32,
) -> dict[str, Any]:
    """Build a no-raw score ledger over train rows only for curriculum mining."""

    validate_chunk_evidence_cache_report(evidence_cache_report)
    validate_chunk_sparse_report(sparse_report)
    validate_chunk_window_mil_report(window_mil_report)
    manifest_hash = str(classifier_manifest["training_manifest_hash"])
    for label, report in (("sparse", sparse_report), ("window-MIL", window_mil_report)):
        if str(report.get("classifier_manifest_hash")) != manifest_hash:
            raise ValueError(f"{label} report manifest hash mismatch")
        if str(report.get("evidence_cache_hash")) != str(evidence_cache_report["evidence_cache_hash"]):
            raise ValueError(f"{label} report evidence cache hash mismatch")
    view = load_cache_backed_training_view(
        cache_db,
        classifier_manifest=classifier_manifest,
        evidence_cache_report=evidence_cache_report,
        context="chunk training score ledger",
    )
    manifest_by_candidate = {
        str(row["candidate_id"]): row
        for row in classifier_manifest.get("rows") or []
    }
    rows = sorted(view["train_rows"], key=lambda row: str(row["candidate_id"]))
    sparse_scores = _resolve_training_scores(
        rows,
        model_id=CHUNK_SPARSE_MODEL_ID,
        supplied_scores=sparse_scores_by_candidate_id,
        sparse_report=sparse_report,
        window_mil_report=window_mil_report,
        sparse_model_path=sparse_model_path,
        window_mil_model_path=window_mil_model_path,
        device=device,
        require_cuda=require_cuda,
        score_batch_size=sparse_score_batch_size,
    )
    window_scores = _resolve_training_scores(
        rows,
        model_id=CHUNK_WINDOW_MIL_MODEL_ID,
        supplied_scores=window_mil_scores_by_candidate_id,
        sparse_report=sparse_report,
        window_mil_report=window_mil_report,
        sparse_model_path=sparse_model_path,
        window_mil_model_path=window_mil_model_path,
        device=device,
        require_cuda=require_cuda,
        score_batch_size=window_mil_score_batch_size,
    )
    ledger_rows = [
        _training_score_ledger_row(
            row,
            manifest_row=manifest_by_candidate[str(row["candidate_id"])],
            sparse_score=sparse_scores[str(row["candidate_id"])],
            window_mil_score=window_scores[str(row["candidate_id"])],
            sparse_report=sparse_report,
            window_mil_report=window_mil_report,
        )
        for row in rows
    ]
    _attach_train_label_score_rank_fields(ledger_rows)
    report = {
        "artifact_kind": CHUNK_TRAIN_SCORE_LEDGER_KIND,
        "schema_version": CHUNK_TRAIN_SCORE_LEDGER_SCHEMA_VERSION,
        "status": "complete",
        "classifier_manifest_hash": manifest_hash,
        "evidence_cache_hash": str(evidence_cache_report["evidence_cache_hash"]),
        "evidence_cache_report_hash": str(evidence_cache_report["report_hash"]),
        "dataset_view_hash": str(view["dataset_view_hash"]),
        "sparse_report_hash": str(sparse_report["report_hash"]),
        "sparse_model_payload_hash": str(sparse_report["model_payload_hash"]),
        "window_mil_report_hash": str(window_mil_report["report_hash"]),
        "window_mil_model_payload_hash": str(window_mil_report["model_payload_hash"]),
        "split": "train",
        "row_count": len(ledger_rows),
        "score_rank_policy": CHUNK_TRAIN_SCORE_RANK_POLICY_VERSION,
        "score_rank_high_risk_percentile": TRAIN_SCORE_HIGH_RISK_PERCENTILE,
        "score_rank_low_risk_percentile": TRAIN_SCORE_LOW_RISK_PERCENTILE,
        "calibration_thresholds_used_for_curriculum": False,
        "label_counts": _label_counts(ledger_rows),
        "evidence_tier_counts": _evidence_tier_counts(ledger_rows),
        "source_bucket_hash_counts": _count_optional_field(ledger_rows, "source_bucket_hash"),
        "token_count_bucket_counts": _count_optional_field(ledger_rows, "token_count_bucket"),
        "row_identity_universe_hash": stable_json_hash([row["row_identity_hash"] for row in ledger_rows]),
        "score_ledger_hash": stable_json_hash(ledger_rows),
        "rows": ledger_rows,
        **_training_public_false_flags(),
    }
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_training_score_ledger(report)
    if output_json is not None:
        _write_public_json(Path(output_json), report)
    return report


def validate_chunk_training_score_ledger(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_TRAIN_SCORE_LEDGER_KIND:
        raise ValueError("unsupported chunk training score ledger kind")
    if report.get("status") != "complete":
        raise ValueError("chunk training score ledger must be complete")
    if int(report.get("schema_version", 1)) != CHUNK_TRAIN_SCORE_LEDGER_SCHEMA_VERSION:
        raise ValueError("unsupported chunk training score ledger schema_version")
    if report.get("score_rank_policy") != CHUNK_TRAIN_SCORE_RANK_POLICY_VERSION:
        raise ValueError("training score ledger rank policy mismatch")
    if report.get("calibration_thresholds_used_for_curriculum") is not False:
        raise ValueError("training score ledger must not use calibration thresholds for curriculum")
    for field, expected in (
        ("score_rank_high_risk_percentile", TRAIN_SCORE_HIGH_RISK_PERCENTILE),
        ("score_rank_low_risk_percentile", TRAIN_SCORE_LOW_RISK_PERCENTILE),
    ):
        actual = float(report.get(field, float("nan")))
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"training score ledger {field} mismatch")
    for field in (
        "classifier_manifest_hash",
        "evidence_cache_hash",
        "evidence_cache_report_hash",
        "dataset_view_hash",
        "sparse_report_hash",
        "sparse_model_payload_hash",
        "window_mil_report_hash",
        "window_mil_model_payload_hash",
        "row_identity_universe_hash",
        "score_ledger_hash",
    ):
        _required_sha256(str(report.get(field) or ""), field)
    if report.get("split") != "train":
        raise ValueError("training score ledger must contain train rows only")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("training score ledger requires rows")
    if int(report.get("row_count", -1)) != len(rows):
        raise ValueError("training score ledger row_count mismatch")
    seen: set[str] = set()
    for row in rows:
        _validate_training_score_ledger_row(row)
        if row["candidate_id"] in seen:
            raise ValueError("training score ledger duplicate candidate")
        seen.add(str(row["candidate_id"]))
    _validate_train_label_score_rank_fields(rows)
    if report["row_identity_universe_hash"] != stable_json_hash([row["row_identity_hash"] for row in rows]):
        raise ValueError("training score ledger row identity universe hash mismatch")
    if report["score_ledger_hash"] != stable_json_hash(rows):
        raise ValueError("training score ledger hash mismatch")
    _require_false_report_flags(report)
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("training score ledger report hash mismatch")


def build_chunk_hard_curriculum(
    *,
    classifier_manifest: Mapping[str, Any],
    evidence_cache_report: Mapping[str, Any],
    training_score_ledger: Mapping[str, Any],
    hard_source_bucket_hashes: Iterable[str] = (),
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Build capped train-only curriculum weights from no-raw train diagnostics."""

    validate_chunk_classifier_training_manifest(classifier_manifest)
    validate_chunk_evidence_cache_report(evidence_cache_report)
    validate_chunk_training_score_ledger(training_score_ledger)
    manifest_hash = str(classifier_manifest["training_manifest_hash"])
    if training_score_ledger["classifier_manifest_hash"] != manifest_hash:
        raise ValueError("curriculum training score ledger manifest hash mismatch")
    if training_score_ledger["evidence_cache_hash"] != evidence_cache_report["evidence_cache_hash"]:
        raise ValueError("curriculum training score ledger evidence cache hash mismatch")
    manifest_by_candidate = {
        str(row["candidate_id"]): row
        for row in classifier_manifest.get("rows") or []
    }
    hard_bucket_hashes = {_required_sha256(str(value), "hard_source_bucket_hash") for value in hard_source_bucket_hashes}
    rows = []
    group_weight_mass: dict[str, float] = {}
    group_base_mass: dict[str, float] = {}
    for score_row in training_score_ledger["rows"]:
        candidate_id = str(score_row["candidate_id"])
        manifest_row = manifest_by_candidate.get(candidate_id)
        if manifest_row is None:
            raise ValueError("curriculum score row missing manifest row")
        if str(score_row["split"]) != "train":
            raise ValueError("curriculum can only weight train rows")
        base_weight = 1.0
        multiplier, reasons = _curriculum_multiplier_and_reasons(
            score_row,
            manifest_row,
            hard_source_bucket_hashes=hard_bucket_hashes,
        )
        source_group = str(manifest_row.get("source_document_identity_hash") or manifest_row.get("leakage_group_hash") or candidate_id)
        group_base_mass[source_group] = group_base_mass.get(source_group, 0.0) + base_weight
        group_weight_mass[source_group] = group_weight_mass.get(source_group, 0.0) + multiplier
        row = {
            "candidate_id": candidate_id,
            "candidate_hash": str(score_row["candidate_hash"]),
            "row_hash": str(score_row["row_hash"]),
            "row_identity_hash": str(score_row["row_identity_hash"]),
            "classifier_label": str(score_row["classifier_label"]),
            "split": "train",
            "evidence_tier": str(score_row["evidence_tier"]),
            "source_document_identity_hash": source_group,
            "source_bucket_hash": score_row.get("source_bucket_hash"),
            "token_count_bucket": str(score_row.get("token_count_bucket") or "unknown"),
            "curriculum_multiplier": float(multiplier),
            "hard_reason_codes": reasons,
            "eligible_for_training": True,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        row.update(_active_learning_public_training_metadata(manifest_row))
        row["curriculum_row_hash"] = _hash_without(row, "curriculum_row_hash")
        rows.append(row)
    rows = sorted(rows, key=lambda row: str(row["candidate_id"]))
    _enforce_curriculum_group_caps(rows, group_base_mass=group_base_mass)
    summary = _curriculum_summary(rows)
    report = {
        "artifact_kind": CHUNK_HARD_CURRICULUM_KIND,
        "schema_version": CHUNK_HARD_CURRICULUM_SCHEMA_VERSION,
        "status": "complete",
        "classifier_manifest_hash": manifest_hash,
        "evidence_cache_hash": str(evidence_cache_report["evidence_cache_hash"]),
        "evidence_cache_report_hash": str(evidence_cache_report["report_hash"]),
        "training_score_ledger_hash": str(training_score_ledger["score_ledger_hash"]),
        "training_score_ledger_report_hash": str(training_score_ledger["report_hash"]),
        "split": "train",
        "row_count": len(rows),
        "weighted_row_count": sum(1 for row in rows if float(row["curriculum_multiplier"]) > 1.0),
        "max_row_multiplier": 4.0,
        "source_group_weight_cap_multiplier": 2.0,
        "reason_policy_version": CHUNK_HARD_CURRICULUM_REASON_POLICY_VERSION,
        "weight_policy": dict(CURRICULUM_REASON_WEIGHTS),
        "hard_source_bucket_hashes": sorted(hard_bucket_hashes),
        "summary": summary,
        "curriculum_hash": stable_json_hash(rows),
        "rows": rows,
        **_training_public_false_flags(),
    }
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_hard_curriculum(report)
    if output_json is not None:
        _write_public_json(Path(output_json), report)
    return report


def validate_chunk_hard_curriculum(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_HARD_CURRICULUM_KIND:
        raise ValueError("unsupported chunk hard curriculum kind")
    if report.get("status") != "complete":
        raise ValueError("chunk hard curriculum must be complete")
    schema_version = int(report.get("schema_version", 1))
    if schema_version not in {1, CHUNK_HARD_CURRICULUM_SCHEMA_VERSION}:
        raise ValueError("unsupported chunk hard curriculum schema_version")
    if schema_version >= CHUNK_HARD_CURRICULUM_SCHEMA_VERSION:
        if report.get("reason_policy_version") != CHUNK_HARD_CURRICULUM_REASON_POLICY_VERSION:
            raise ValueError("hard curriculum reason policy version mismatch")
        if dict(report.get("weight_policy") or {}) != CURRICULUM_REASON_WEIGHTS:
            raise ValueError("hard curriculum weight policy mismatch")
    for field in (
        "classifier_manifest_hash",
        "evidence_cache_hash",
        "evidence_cache_report_hash",
        "training_score_ledger_hash",
        "training_score_ledger_report_hash",
        "curriculum_hash",
    ):
        _required_sha256(str(report.get(field) or ""), field)
    if report.get("split") != "train":
        raise ValueError("hard curriculum must be train-only")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("hard curriculum requires rows")
    if int(report.get("row_count", -1)) != len(rows):
        raise ValueError("hard curriculum row_count mismatch")
    allowed_reason_codes = set(CURRICULUM_REASON_WEIGHTS) if schema_version >= CHUNK_HARD_CURRICULUM_SCHEMA_VERSION else None
    seen: set[str] = set()
    for row in rows:
        _validate_curriculum_row(row, allowed_reason_codes=allowed_reason_codes)
        if row["candidate_id"] in seen:
            raise ValueError("hard curriculum duplicate candidate")
        seen.add(str(row["candidate_id"]))
    if schema_version >= CHUNK_HARD_CURRICULUM_SCHEMA_VERSION:
        if int(report.get("weighted_row_count", -1)) != sum(
            1 for row in rows if float(row["curriculum_multiplier"]) > 1.0
        ):
            raise ValueError("hard curriculum weighted_row_count mismatch")
        if dict(report.get("summary") or {}) != _curriculum_summary(rows):
            raise ValueError("hard curriculum summary mismatch")
    if report["curriculum_hash"] != stable_json_hash(rows):
        raise ValueError("hard curriculum hash mismatch")
    _require_false_report_flags(report)
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("hard curriculum report hash mismatch")


def validate_chunk_hard_curriculum_binding(
    *,
    classifier_manifest: Mapping[str, Any],
    evidence_cache_report: Mapping[str, Any],
    training_score_ledger: Mapping[str, Any],
    curriculum: Mapping[str, Any],
) -> None:
    validate_chunk_hard_curriculum(curriculum)
    expected = build_chunk_hard_curriculum(
        classifier_manifest=classifier_manifest,
        evidence_cache_report=evidence_cache_report,
        training_score_ledger=training_score_ledger,
        hard_source_bucket_hashes=curriculum.get("hard_source_bucket_hashes") or (),
    )
    for field in (
        "training_score_ledger_hash",
        "training_score_ledger_report_hash",
        "curriculum_hash",
        "weighted_row_count",
        "summary",
        "rows",
        "report_hash",
    ):
        if curriculum.get(field) != expected.get(field):
            raise ValueError(f"hard curriculum binding {field} mismatch")


def _resolve_curriculum_view(
    curriculum: Mapping[str, Any] | None,
    *,
    view: Mapping[str, Any],
    classifier_manifest_hash: str,
) -> dict[str, Any]:
    if curriculum is None:
        no_curriculum_hash = stable_json_hash({"curriculum": "none", "classifier_manifest_hash": classifier_manifest_hash})
        return {
            "variant_id": "no_curriculum",
            "curriculum_hash": no_curriculum_hash,
            "uses_curriculum_weights": False,
            "weights_by_candidate": {},
            "summary": {
                "row_count": 0,
                "weighted_row_count": 0,
                "max_multiplier": 1.0,
                "reason_counts": {},
            },
        }
    validate_chunk_hard_curriculum(curriculum)
    if curriculum["classifier_manifest_hash"] != classifier_manifest_hash:
        raise ValueError("curriculum classifier manifest hash mismatch")
    view_train_ids = {str(row["candidate_id"]) for row in view["train_rows"]}
    rows = list(curriculum["rows"])
    row_ids = {str(row["candidate_id"]) for row in rows}
    if row_ids != view_train_ids:
        raise ValueError("curriculum train row coverage mismatch")
    return {
        "variant_id": "hard_curriculum_v1",
        "curriculum_hash": str(curriculum["curriculum_hash"]),
        "uses_curriculum_weights": True,
        "weights_by_candidate": {str(row["candidate_id"]): float(row["curriculum_multiplier"]) for row in rows},
        "summary": dict(curriculum["summary"]),
    }


def _curriculum_weight_for_row(row: Mapping[str, Any], *, curriculum_view: Mapping[str, Any]) -> float:
    if not curriculum_view.get("uses_curriculum_weights"):
        return float(row.get("training_weight", 1.0))
    weight = float(curriculum_view["weights_by_candidate"].get(str(row["candidate_id"]), float("nan")))
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("curriculum weight missing or invalid")
    return float(row.get("training_weight", 1.0)) * weight


def _sparse_variant(variant_id: str, *, artifact_root: str, alpha: float, uses_curriculum: bool) -> dict[str, Any]:
    artifact_paths = _variant_artifact_paths(
        artifact_root=artifact_root,
        variant_id=variant_id,
        model_id=CHUNK_SPARSE_MODEL_ID,
        model_suffix="joblib",
    )
    payload = {
        "variant_id": variant_id,
        "model_family": CHUNK_SPARSE_MODEL_ID,
        "alpha": float(alpha),
        "n_features": DEFAULT_SPARSE_N_FEATURES,
        "random_state": DEFAULT_RANDOM_STATE,
        "target_reject_recall": DEFAULT_TARGET_RECALL,
        "uses_curriculum": bool(uses_curriculum),
        "input_mode": "core_plus_halo_evidence_text",
        **artifact_paths,
    }
    payload["variant_hash"] = _hash_without(payload, "variant_hash")
    return payload


def _window_variant(
    variant_id: str,
    *,
    artifact_root: str,
    input_mode: str,
    pooling_mode: str,
    pooling_top_k: int,
    uses_curriculum: bool,
    adopt_existing_if_hashes_validate: bool = False,
) -> dict[str, Any]:
    artifact_paths = _variant_artifact_paths(
        artifact_root=artifact_root,
        variant_id=variant_id,
        model_id=CHUNK_WINDOW_MIL_MODEL_ID,
        model_suffix="pt",
    )
    config = _window_variant_config(
        input_mode=input_mode,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        halo_weight=0.25,
        halo_logit_clamp=2.0,
    )
    payload = {
        "variant_id": variant_id,
        "model_family": CHUNK_WINDOW_MIL_MODEL_ID,
        "base_model_name": DEFAULT_MODEL_NAME,
        "epochs": 2,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "max_length": 512,
        "stride": 128,
        "seed": DEFAULT_RANDOM_STATE,
        "target_reject_recall": DEFAULT_TARGET_RECALL,
        "uses_curriculum": bool(uses_curriculum),
        "adopt_existing_if_hashes_validate": bool(adopt_existing_if_hashes_validate),
        "window_variant_config": config,
        "window_variant_config_hash": stable_json_hash(config),
        **artifact_paths,
    }
    payload["variant_hash"] = _hash_without(payload, "variant_hash")
    return payload


def _variant_artifact_paths(*, artifact_root: str, variant_id: str, model_id: str, model_suffix: str) -> dict[str, str]:
    artifact_dir = f"{_safe_relative_artifact_root(artifact_root)}/{_safe_artifact_name(variant_id)}"
    return {
        "artifact_dir": artifact_dir,
        "report_path": f"{artifact_dir}/{model_id}_report.json",
        "model_path": f"{artifact_dir}/{model_id}_model.{model_suffix}",
        "progress_path": f"{artifact_dir}/{model_id}_progress.json",
        "checkpoint_path": f"{artifact_dir}/{model_id}_checkpoint.{model_suffix if model_suffix == 'pt' else 'json'}",
    }


def _validate_variant_artifact_paths(variant: Mapping[str, Any]) -> None:
    variant_id = _safe_artifact_name(str(variant.get("variant_id") or ""))
    artifact_dir = _safe_relative_artifact_root(str(variant.get("artifact_dir") or ""))
    if not artifact_dir.endswith(f"/{variant_id}"):
        raise ValueError("variant artifact_dir must end with variant_id")
    for field in ("report_path", "model_path", "progress_path", "checkpoint_path"):
        value = str(variant.get(field) or "")
        _safe_relative_artifact_root(value)
        if not value.startswith(f"{artifact_dir}/"):
            raise ValueError(f"variant {field} must stay under artifact_dir")


def _safe_relative_artifact_root(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text or text.startswith("/") or "://" in text:
        raise ValueError("artifact paths must be relative public repo paths")
    parts = [part for part in text.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError("artifact paths must not contain traversal")
    return "/".join(_safe_artifact_name(part) for part in parts)


def _safe_artifact_name(value: str) -> str:
    text = str(value).strip()
    if not text or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in text):
        raise ValueError("artifact names may only contain ASCII letters, numbers, dot, underscore, and hyphen")
    return text


def _resolve_training_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    supplied_scores: Mapping[str, float] | None,
    sparse_report: Mapping[str, Any],
    window_mil_report: Mapping[str, Any],
    sparse_model_path: str | Path | None,
    window_mil_model_path: str | Path | None,
    device: str,
    require_cuda: bool,
    score_batch_size: int,
) -> dict[str, float]:
    if supplied_scores is not None:
        scores = {}
        for row in rows:
            candidate_id = str(row["candidate_id"])
            if candidate_id not in supplied_scores:
                raise ValueError(f"{model_id} supplied train scores missing candidate")
            scores[candidate_id] = _score01(supplied_scores[candidate_id], f"{model_id} score")
        return scores
    if int(score_batch_size) < 1:
        raise ValueError(f"{model_id} score_batch_size must be positive")
    if model_id == CHUNK_SPARSE_MODEL_ID:
        if sparse_model_path is None:
            raise ValueError("sparse model path required when sparse train scores are not supplied")
        from twc.pretraining.chunk_filtering import ChunkSparseRiskScorer

        scorer = ChunkSparseRiskScorer.from_report(sparse_report, sparse_model_path)
    elif model_id == CHUNK_WINDOW_MIL_MODEL_ID:
        if window_mil_model_path is None:
            raise ValueError("Window-MIL model path required when Window-MIL train scores are not supplied")
        from twc.pretraining.chunk_filtering import ChunkWindowMilRiskScorer

        scorer = ChunkWindowMilRiskScorer.from_report(
            window_mil_report,
            window_mil_model_path,
            device=device,
            require_cuda=require_cuda,
        )
    else:
        raise ValueError(f"unsupported model id: {model_id}")
    return _score_training_rows_in_batches(
        rows,
        scorer=scorer,
        model_id=model_id,
        score_batch_size=int(score_batch_size),
    )


def _score_training_rows_in_batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    scorer: Any,
    model_id: str,
    score_batch_size: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    batches = _batches(rows, int(score_batch_size))
    total_rows = len(rows)
    completed_rows = 0
    started = time.monotonic()
    _emit_train_score_progress(
        model_id=model_id,
        completed_rows=0,
        total_rows=total_rows,
        completed_batches=0,
        total_batches=len(batches),
        started=started,
    )
    for batch_index, batch in enumerate(batches, start=1):
        texts = [str(row["evidence_text"]) for row in batch]
        metadata = [_training_runtime_metadata(row) for row in batch]
        batch_scores = scorer.score_many(texts, metadata)
        if len(batch_scores) != len(batch):
            raise ValueError(f"{model_id} score count mismatch")
        for row, score in zip(batch, batch_scores, strict=True):
            scores[str(row["candidate_id"])] = _score01(score, f"{model_id} score")
        completed_rows += len(batch)
        if batch_index == 1 or batch_index == len(batches) or batch_index % 10 == 0:
            _emit_train_score_progress(
                model_id=model_id,
                completed_rows=completed_rows,
                total_rows=total_rows,
                completed_batches=batch_index,
                total_batches=len(batches),
                started=started,
            )
    if len(scores) != len(rows):
        raise ValueError(f"{model_id} scored candidate count mismatch")
    return scores


def _emit_train_score_progress(
    *,
    model_id: str,
    completed_rows: int,
    total_rows: int,
    completed_batches: int,
    total_batches: int,
    started: float,
) -> None:
    elapsed = max(0.0, time.monotonic() - float(started))
    payload = {
        "event": "train_score_ledger_progress",
        "model_id": model_id,
        "completed_rows": int(completed_rows),
        "total_rows": int(total_rows),
        "completed_batches": int(completed_batches),
        "total_batches": int(total_batches),
        "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": 0.0 if elapsed <= 0 else round(float(completed_rows) / elapsed, 3),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    validate_public_no_raw_payload(payload)
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


def _training_score_ledger_row(
    row: Mapping[str, Any],
    *,
    manifest_row: Mapping[str, Any],
    sparse_score: float,
    window_mil_score: float,
    sparse_report: Mapping[str, Any],
    window_mil_report: Mapping[str, Any],
) -> dict[str, Any]:
    sparse = _score01(sparse_score, "sparse_score")
    window = _score01(window_mil_score, "window_mil_score")
    identity = _training_row_identity(row)
    source_document_identity_hash = str(
        manifest_row.get("source_document_identity_hash")
        or stable_json_hash({"parent_candidate_id": str(manifest_row.get("parent_candidate_id") or row["candidate_id"])})
    )
    payload = {
        **identity,
        "source_document_identity_hash": _required_sha256(source_document_identity_hash, "source_document_identity_hash"),
        "source_bucket_hash": _optional_sha256(manifest_row.get("source_bucket_hash"), "source_bucket_hash"),
        "token_count_bucket": str(manifest_row.get("token_count_bucket") or "unknown"),
        "sparse_report_hash": str(sparse_report["report_hash"]),
        "sparse_model_payload_hash": str(sparse_report["model_payload_hash"]),
        "sparse_score": sparse,
        "window_mil_report_hash": str(window_mil_report["report_hash"]),
        "window_mil_model_payload_hash": str(window_mil_report["model_payload_hash"]),
        "window_mil_score": window,
        "score_rank_policy": CHUNK_TRAIN_SCORE_RANK_POLICY_VERSION,
        "score_split_role": "train_curriculum_mining_only",
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload.update(_active_learning_public_training_metadata(manifest_row))
    payload["score_row_hash"] = _hash_without(payload, "score_row_hash")
    validate_public_no_raw_payload(payload)
    return payload


def _attach_train_label_score_rank_fields(rows: Sequence[dict[str, Any]]) -> None:
    for prefix, score_field in (
        ("sparse", "sparse_score"),
        ("window_mil", "window_mil_score"),
    ):
        for label in ("safe", "reject"):
            label_rows = [row for row in rows if str(row["classifier_label"]) == label]
            percentiles = _label_score_percentiles(label_rows, score_field=score_field)
            for row in label_rows:
                percentile = percentiles[str(row["candidate_id"])]
                row[f"{prefix}_train_label_score_percentile"] = percentile
                row[f"{prefix}_train_label_high_risk_score"] = (
                    label == "safe" and percentile >= TRAIN_SCORE_HIGH_RISK_PERCENTILE
                )
                row[f"{prefix}_train_label_low_risk_score"] = (
                    label == "reject" and percentile <= TRAIN_SCORE_LOW_RISK_PERCENTILE
                )
    for row in rows:
        row["score_row_hash"] = _hash_without(row, "score_row_hash")
        validate_public_no_raw_payload(row)


def _label_score_percentiles(rows: Sequence[Mapping[str, Any]], *, score_field: str) -> dict[str, float]:
    if not rows:
        return {}
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row[score_field]),
            str(row["candidate_id"]),
        ),
    )
    if len(ordered) == 1:
        return {str(ordered[0]["candidate_id"]): 0.5}
    result: dict[str, float] = {}
    index = 0
    denominator = float(len(ordered) - 1)
    while index < len(ordered):
        score = float(ordered[index][score_field])
        end = index + 1
        while end < len(ordered) and math.isclose(float(ordered[end][score_field]), score, rel_tol=0.0, abs_tol=0.0):
            end += 1
        average_rank = (index + end - 1) / 2.0
        percentile = float(average_rank / denominator)
        for row in ordered[index:end]:
            result[str(row["candidate_id"])] = percentile
        index = end
    return result


def _validate_train_label_score_rank_fields(rows: Sequence[Mapping[str, Any]]) -> None:
    for prefix, score_field in (
        ("sparse", "sparse_score"),
        ("window_mil", "window_mil_score"),
    ):
        for label in ("safe", "reject"):
            label_rows = [row for row in rows if str(row["classifier_label"]) == label]
            expected = _label_score_percentiles(label_rows, score_field=score_field)
            for row in label_rows:
                candidate_id = str(row["candidate_id"])
                actual = float(row[f"{prefix}_train_label_score_percentile"])
                if not math.isclose(actual, expected[candidate_id], rel_tol=0.0, abs_tol=1e-15):
                    raise ValueError("training score ledger rank percentile mismatch")
                high = bool(row[f"{prefix}_train_label_high_risk_score"])
                low = bool(row[f"{prefix}_train_label_low_risk_score"])
                if high != (label == "safe" and expected[candidate_id] >= TRAIN_SCORE_HIGH_RISK_PERCENTILE):
                    raise ValueError("training score ledger high-risk rank flag mismatch")
                if low != (label == "reject" and expected[candidate_id] <= TRAIN_SCORE_LOW_RISK_PERCENTILE):
                    raise ValueError("training score ledger low-risk rank flag mismatch")


def _validate_training_score_ledger_row(row: Mapping[str, Any]) -> None:
    for field in (
        "candidate_id",
        "candidate_hash",
        "row_hash",
        "row_identity_hash",
        "core_text_sha256",
        "evidence_text_sha256",
        "source_document_identity_hash",
        "sparse_report_hash",
        "sparse_model_payload_hash",
        "window_mil_report_hash",
        "window_mil_model_payload_hash",
    ):
        _required_sha256(str(row.get(field) or ""), field)
    if row.get("split") != "train":
        raise ValueError("training score ledger row must be train split")
    if row.get("classifier_label") not in {"safe", "reject"}:
        raise ValueError("training score ledger row label invalid")
    _score01(row.get("sparse_score"), "sparse_score")
    _score01(row.get("window_mil_score"), "window_mil_score")
    if row.get("score_rank_policy") != CHUNK_TRAIN_SCORE_RANK_POLICY_VERSION:
        raise ValueError("training score ledger row rank policy mismatch")
    for prefix in ("sparse", "window_mil"):
        _score01(row.get(f"{prefix}_train_label_score_percentile"), f"{prefix}_train_label_score_percentile")
        for suffix in ("high_risk_score", "low_risk_score"):
            field = f"{prefix}_train_label_{suffix}"
            if row.get(field) not in {True, False}:
                raise ValueError(f"training score ledger row {field} invalid")
    label = str(row["classifier_label"])
    for prefix in ("sparse", "window_mil"):
        percentile = float(row[f"{prefix}_train_label_score_percentile"])
        high = bool(row[f"{prefix}_train_label_high_risk_score"])
        low = bool(row[f"{prefix}_train_label_low_risk_score"])
        if high != (label == "safe" and percentile >= TRAIN_SCORE_HIGH_RISK_PERCENTILE):
            raise ValueError("training score ledger high-risk rank flag mismatch")
        if low != (label == "reject" and percentile <= TRAIN_SCORE_LOW_RISK_PERCENTILE):
            raise ValueError("training score ledger low-risk rank flag mismatch")
    if row.get("score_split_role") != "train_curriculum_mining_only":
        raise ValueError("training score ledger row role mismatch")
    _validate_emitted_active_learning_public_training_metadata(row)
    if row.get("score_row_hash") != _hash_without(row, "score_row_hash"):
        raise ValueError("training score ledger row hash mismatch")


def _curriculum_multiplier_and_reasons(
    score_row: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    *,
    hard_source_bucket_hashes: set[str],
) -> tuple[float, list[str]]:
    label = str(score_row["classifier_label"])
    reasons: list[str] = []
    multiplier = 1.0
    source_bucket_hash = manifest_row.get("source_bucket_hash")
    if source_bucket_hash in hard_source_bucket_hashes:
        multiplier = max(multiplier, 1.5)
        reasons.append("hard_or_borderline_source_bucket")
    if label == "safe":
        high_risk_count = int(bool(score_row["sparse_train_label_high_risk_score"])) + int(
            bool(score_row["window_mil_train_label_high_risk_score"])
        )
        if high_risk_count == 1:
            multiplier = max(multiplier, 2.0)
            reasons.append("safe_high_risk_score_one_model")
        elif high_risk_count == 2:
            multiplier = max(multiplier, 2.5)
            reasons.append("safe_high_risk_score_two_models")
    else:
        low_risk_count = int(bool(score_row["sparse_train_label_low_risk_score"])) + int(
            bool(score_row["window_mil_train_label_low_risk_score"])
        )
        if low_risk_count == 1:
            multiplier = max(multiplier, 3.0)
            reasons.append("reject_low_risk_score_one_model")
        elif low_risk_count == 2:
            multiplier = max(multiplier, 4.0)
            reasons.append("reject_low_risk_score_two_models")
    active_metadata = _active_learning_public_training_metadata(manifest_row)
    if active_metadata:
        active_sparse_rejects = bool(active_metadata["active_learning_sampling_sparse_reject"])
        active_window_rejects = bool(active_metadata["active_learning_sampling_window_mil_reject"])
        route = str(active_metadata["active_learning_route"])
        if label == "safe":
            active_false_positive_count = int(active_sparse_rejects) + int(active_window_rejects)
            if route in {"policy_reject", "quarantine", "policy_quarantine"}:
                multiplier = max(multiplier, 2.5)
                reasons.append("active_policy_reject_strict_safe_overreject")
            if active_false_positive_count == 1:
                multiplier = max(multiplier, 2.0)
                reasons.append("active_sampling_one_model_reject_strict_safe")
            elif active_false_positive_count == 2:
                multiplier = max(multiplier, 2.5)
                reasons.append("active_sampling_both_model_reject_strict_safe")
        else:
            active_false_negative_count = int(not active_sparse_rejects) + int(not active_window_rejects)
            if route == "policy_safe":
                multiplier = max(multiplier, 4.0)
                reasons.append("active_policy_safe_reject_or_borderline_miss")
            if active_false_negative_count == 1:
                multiplier = max(multiplier, 3.0)
                reasons.append("active_sampling_one_model_safe_reject_or_borderline")
            elif active_false_negative_count == 2:
                multiplier = max(multiplier, 4.0)
                reasons.append("active_sampling_both_model_safe_reject_or_borderline")
            if int(active_metadata["active_learning_teacher_safe_vote_count"]) > 0:
                multiplier = max(multiplier, 3.0)
                reasons.append("active_teacher_split_vote")
            if int(active_metadata["active_learning_borderline_vote_count"]) > 0:
                multiplier = max(multiplier, 3.0)
                reasons.append("active_teacher_borderline")
            if bool(active_metadata["active_learning_teacher_uncertain"]) and multiplier > 3.0:
                multiplier = 3.0
                reasons.append("active_teacher_uncertain_reject_cap")
    multiplier = min(float(multiplier), 4.0)
    if not reasons:
        reasons = ["normal"]
    return multiplier, reasons


def _active_learning_public_training_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    if row.get("evidence_tier") != ACTIVE_LEARNING_EVIDENCE_TIER:
        return {}
    safe_vote_count = _nonnegative_count(row.get("teacher_safe_vote_count"), "teacher_safe_vote_count")
    reject_vote_count = _nonnegative_count(row.get("teacher_reject_vote_count"), "teacher_reject_vote_count")
    borderline_vote_count = _nonnegative_count(row.get("borderline_vote_count"), "borderline_vote_count")
    payload = {
        "active_learning_final_label": str(row.get("active_learning_final_label") or ""),
        "active_learning_batch_id": str(row.get("active_learning_batch_id") or ""),
        "active_learning_route": str(row.get("route") or ""),
        "active_learning_primary_bucket": str(row.get("primary_bucket") or ""),
        "active_learning_teacher_safe_vote_count": safe_vote_count,
        "active_learning_teacher_reject_vote_count": reject_vote_count,
        "active_learning_borderline_vote_count": borderline_vote_count,
        "active_learning_teacher_uncertain": bool(safe_vote_count > 0 or borderline_vote_count > 0),
        "active_learning_sampling_sparse_reject": bool(row.get("sparse_decision_reject")),
        "active_learning_sampling_window_mil_reject": bool(row.get("window_mil_decision_reject")),
    }
    if payload["active_learning_final_label"] not in {"safe", "reject_or_borderline"}:
        raise ValueError("active-learning training metadata final label invalid")
    if not payload["active_learning_batch_id"]:
        raise ValueError("active-learning training metadata missing batch id")
    if safe_vote_count + reject_vote_count != 3:
        raise ValueError("active-learning training metadata vote count mismatch")
    validate_public_no_raw_payload(payload)
    return payload


def _nonnegative_count(value: Any, field_name: str) -> int:
    count = int(value)
    if count < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return count


def _validate_emitted_active_learning_public_training_metadata(row: Mapping[str, Any]) -> None:
    if row.get("evidence_tier") != ACTIVE_LEARNING_EVIDENCE_TIER:
        return
    for field in (
        "active_learning_final_label",
        "active_learning_batch_id",
        "active_learning_route",
        "active_learning_primary_bucket",
    ):
        if not isinstance(row.get(field), str) or not str(row.get(field)):
            raise ValueError(f"{field} missing from active-learning training metadata")
    if row.get("active_learning_final_label") not in {"safe", "reject_or_borderline"}:
        raise ValueError("active-learning training metadata final label invalid")
    safe_vote_count = _nonnegative_count(
        row.get("active_learning_teacher_safe_vote_count"),
        "active_learning_teacher_safe_vote_count",
    )
    reject_vote_count = _nonnegative_count(
        row.get("active_learning_teacher_reject_vote_count"),
        "active_learning_teacher_reject_vote_count",
    )
    borderline_vote_count = _nonnegative_count(
        row.get("active_learning_borderline_vote_count"),
        "active_learning_borderline_vote_count",
    )
    if safe_vote_count + reject_vote_count != 3:
        raise ValueError("active-learning training metadata vote count mismatch")
    expected_uncertain = bool(safe_vote_count > 0 or borderline_vote_count > 0)
    if row.get("active_learning_teacher_uncertain") is not expected_uncertain:
        raise ValueError("active-learning training metadata uncertainty drift")
    if not isinstance(row.get("active_learning_sampling_sparse_reject"), bool):
        raise ValueError("active-learning sparse sampling decision must be boolean")
    if not isinstance(row.get("active_learning_sampling_window_mil_reject"), bool):
        raise ValueError("active-learning window-MIL sampling decision must be boolean")


def _enforce_curriculum_group_caps(
    rows: list[dict[str, Any]],
    *,
    group_base_mass: Mapping[str, float],
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_document_identity_hash"]), []).append(row)
    for group, group_rows in grouped.items():
        cap = 2.0 * float(group_base_mass.get(group, len(group_rows)))
        mass = sum(float(row["curriculum_multiplier"]) for row in group_rows)
        if mass <= cap or mass <= 0:
            continue
        scale = cap / mass
        for row in group_rows:
            row["curriculum_multiplier"] = max(1.0, round(float(row["curriculum_multiplier"]) * scale, 12))
            row["group_cap_applied"] = True
            row["curriculum_row_hash"] = _hash_without(row, "curriculum_row_hash")


def _curriculum_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    for row in rows:
        for reason in row["hard_reason_codes"]:
            reason_counts[str(reason)] += 1
    multipliers = [float(row["curriculum_multiplier"]) for row in rows]
    return {
        "row_count": len(rows),
        "weighted_row_count": sum(1 for value in multipliers if value > 1.0),
        "max_multiplier": max(multipliers) if multipliers else 1.0,
        "reason_counts": _sorted_counter(reason_counts),
        "label_counts": _label_counts(rows),
        "evidence_tier_counts": _evidence_tier_counts(rows),
    }


def _validate_curriculum_row(row: Mapping[str, Any], *, allowed_reason_codes: set[str] | None = None) -> None:
    for field in (
        "candidate_id",
        "candidate_hash",
        "row_hash",
        "row_identity_hash",
        "source_document_identity_hash",
        "curriculum_row_hash",
    ):
        _required_sha256(str(row.get(field) or ""), field)
    if row.get("split") != "train":
        raise ValueError("hard curriculum row must be train split")
    if row.get("classifier_label") not in {"safe", "reject"}:
        raise ValueError("hard curriculum row label invalid")
    multiplier = float(row.get("curriculum_multiplier", float("nan")))
    if not math.isfinite(multiplier) or multiplier < 1.0 or multiplier > 4.0:
        raise ValueError("hard curriculum multiplier invalid")
    if row.get("eligible_for_training") is not True:
        raise ValueError("hard curriculum row must be training eligible")
    reasons = row.get("hard_reason_codes")
    if not isinstance(reasons, list) or not reasons or any(not isinstance(reason, str) for reason in reasons):
        raise ValueError("hard curriculum row requires reason codes")
    if allowed_reason_codes is not None:
        unknown_reasons = sorted(set(str(reason) for reason in reasons) - allowed_reason_codes)
        if unknown_reasons:
            raise ValueError(f"hard curriculum row unknown reason codes: {unknown_reasons}")
    _validate_emitted_active_learning_public_training_metadata(row)
    if row.get("curriculum_row_hash") != _hash_without(row, "curriculum_row_hash"):
        raise ValueError("hard curriculum row hash mismatch")


def _training_runtime_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "candidate_id_hash": str(row["candidate_id"]),
        "row_identity_hash": stable_json_hash({"candidate_id": str(row["candidate_id"]), "row_hash": str(row["row_hash"])}),
        "split": str(row["split"]),
        "evidence_tier": str(row["evidence_tier"]),
        "core_char_start": int(row["halo_left_chars"]),
        "core_char_end": int(row["halo_left_chars"]) + int(row["core_char_length"]),
        "core_text_sha256": str(row["core_text_sha256"]),
        "evidence_text_sha256": str(row["evidence_text_sha256"]),
    }
    validate_public_no_raw_payload(payload)
    return payload


def _score01(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite score in [0, 1]") from exc
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
        raise ValueError(f"{field_name} must be a finite score in [0, 1]")
    return parsed


def _optional_sha256(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_sha256(str(value), field_name)


def _count_optional_field(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return _sorted_counter(Counter(str(row.get(field) or "none") for row in rows))


def _training_public_false_flags() -> dict[str, bool]:
    return {
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "paid_api_used": False,
        "gpu_rental_used": False,
    }


def _training_row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "candidate_id": str(row["candidate_id"]),
        "candidate_hash": str(row["candidate_hash"]),
        "row_hash": str(row["row_hash"]),
        "classifier_label": str(row["classifier_label"]),
        "split": str(row["split"]),
        "evidence_tier": str(row["evidence_tier"]),
        "fit_eligible": bool(row["fit_eligible"]),
        "eligible_for_eval": bool(row["eligible_for_eval"]),
        "training_weight": float(row["training_weight"]),
        "core_text_sha256": str(row["core_text_sha256"]),
        "evidence_text_sha256": str(row["evidence_text_sha256"]),
    }
    payload["row_identity_hash"] = stable_json_hash(payload)
    return payload


def _validate_and_return_row(row: dict[str, Any]) -> dict[str, Any]:
    _validate_private_cache_row(row)
    return row


def _load_sparse_dependencies() -> dict[str, Any]:
    try:
        import joblib
        import numpy
        from scipy.sparse import csr_matrix
        from sklearn.linear_model import SGDClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency failure.
        raise RuntimeError("chunk sparse training requires scikit-learn, scipy, numpy, and joblib") from exc
    return {"joblib": joblib, "numpy": numpy, "csr_matrix": csr_matrix, "SGDClassifier": SGDClassifier}


def _cache_rows_to_csr(rows: Sequence[Mapping[str, Any]], *, n_features: int, deps: Mapping[str, Any]) -> Any:
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for row in rows:
        counts = hashed_ngram_feature_counts(str(row["evidence_text"]), n_features=n_features)
        if not counts:
            raise ValueError("chunk sparse feature row is empty")
        values: list[tuple[int, float]] = []
        for feature_id, count in sorted(counts.items()):
            column = _feature_id_to_column(feature_id, n_features=n_features)
            values.append((column, 1.0 + math.log(float(count))))
        norm = math.sqrt(sum(value * value for _column, value in values)) or 1.0
        for column, value in values:
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


def _score_sparse_rows(rows: Sequence[Mapping[str, Any]], *, model: Any, n_features: int, deps: Mapping[str, Any]) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    for batch in _batches(rows, 512):
        matrix = _cache_rows_to_csr(batch, n_features=n_features, deps=deps)
        probabilities = model.predict_proba(matrix)[:, 1]
        scores.extend(float(score) for score in probabilities)
        labels.extend(_label_to_int(row["classifier_label"]) for row in batch)
    return scores, labels


def _score_window_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    device: Any,
    max_length: int,
    stride: int,
    batch_size: int,
    max_windows_per_device_batch: int,
    max_batches: int | None,
    window_variant_config: Mapping[str, Any] | None = None,
) -> tuple[list[float], list[int]]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    batches = _batches(rows, batch_size)
    if max_batches is not None:
        batches = batches[: int(max_batches)]
    with torch.no_grad():
        for batch in batches:
            batch_scores = _score_window_batch_for_input_mode(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                rows=batch,
                device=device,
                max_length=max_length,
                stride=stride,
                max_windows_per_device_batch=max_windows_per_device_batch,
                window_variant_config=window_variant_config or _window_variant_config(
                    input_mode=WINDOW_INPUT_FULL_EVIDENCE,
                    pooling_mode=WINDOW_POOLING_MAX,
                    pooling_top_k=1,
                    halo_weight=0.25,
                    halo_logit_clamp=2.0,
                ),
            )
            scores.extend(batch_scores)
            labels.extend(_label_to_int(row["classifier_label"]) for row in batch)
    return scores, labels


def _train_window_batch_for_input_mode(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scaler: Any,
    autocast: Any,
    rows: Sequence[Mapping[str, Any]],
    labels: list[int],
    sample_weights: list[float],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    document_microbatch_size: int,
    window_variant_config: Mapping[str, Any],
) -> float:
    input_mode = str(window_variant_config["input_mode"])
    pooling_mode = str(window_variant_config["pooling_mode"])
    pooling_top_k = int(window_variant_config["pooling_top_k"])
    if input_mode == WINDOW_INPUT_SPLIT_HALO_CAPPED:
        core_texts, halo_texts = _core_and_halo_texts(rows)
        return _train_split_halo_document_batch(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scaler=scaler,
            autocast=autocast,
            core_texts=core_texts,
            halo_texts=halo_texts,
            labels=labels,
            sample_weights=sample_weights,
            device=device,
            max_length=max_length,
            stride=stride,
            max_windows_per_device_batch=max_windows_per_device_batch,
            pooling_mode=pooling_mode,
            pooling_top_k=pooling_top_k,
            halo_weight=float(window_variant_config["halo_weight"]),
            halo_logit_clamp=float(window_variant_config["halo_logit_clamp"]),
            document_microbatch_size=document_microbatch_size,
        )
    return _train_one_document_batch(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scaler=scaler,
        autocast=autocast,
        texts=_window_texts_for_input_mode(rows, input_mode=input_mode),
        labels=labels,
        sample_weights=sample_weights,
        device=device,
        max_length=max_length,
        stride=stride,
        max_windows_per_device_batch=max_windows_per_device_batch,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        document_microbatch_size=document_microbatch_size,
    )


def _score_window_batch_for_input_mode(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    window_variant_config: Mapping[str, Any],
) -> list[float]:
    input_mode = str(window_variant_config["input_mode"])
    pooling_mode = str(window_variant_config["pooling_mode"])
    pooling_top_k = int(window_variant_config["pooling_top_k"])
    if input_mode == WINDOW_INPUT_SPLIT_HALO_CAPPED:
        core_texts, halo_texts = _core_and_halo_texts(rows)
        return _score_split_halo_document_batch(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            core_texts=core_texts,
            halo_texts=halo_texts,
            device=device,
            max_length=max_length,
            stride=stride,
            max_windows_per_device_batch=max_windows_per_device_batch,
            pooling_mode=pooling_mode,
            pooling_top_k=pooling_top_k,
            halo_weight=float(window_variant_config["halo_weight"]),
            halo_logit_clamp=float(window_variant_config["halo_logit_clamp"]),
        )
    return _score_document_batch(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        texts=_window_texts_for_input_mode(rows, input_mode=input_mode),
        device=device,
        max_length=max_length,
        stride=stride,
        max_windows_per_device_batch=max_windows_per_device_batch,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
    )


def _window_texts_for_input_mode(rows: Sequence[Mapping[str, Any]], *, input_mode: str) -> list[str]:
    if input_mode == WINDOW_INPUT_FULL_EVIDENCE:
        return [str(row["evidence_text"]) for row in rows]
    if input_mode == WINDOW_INPUT_CORE_ONLY:
        return [str(row["core_text"]) for row in rows]
    if input_mode == WINDOW_INPUT_SPLIT_HALO_CAPPED:
        core_texts, _halo_texts = _core_and_halo_texts(rows)
        return core_texts
    raise ValueError(f"unsupported Window-MIL input mode: {input_mode}")


def _core_and_halo_texts(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    core_texts: list[str] = []
    halo_texts: list[str] = []
    for row in rows:
        evidence_text = str(row["evidence_text"])
        core_text = str(row["core_text"])
        left_len = int(row["halo_left_chars"])
        core_len = int(row["core_char_length"])
        right_start = left_len + core_len
        if evidence_text[left_len:right_start] != core_text:
            raise ValueError("private evidence cache core/evidence span drift")
        core_texts.append(core_text)
        halo_texts.append((evidence_text[:left_len] + "\n" + evidence_text[right_start:]).strip())
    return core_texts, halo_texts


def _window_variant_config(
    *,
    input_mode: str,
    pooling_mode: str,
    pooling_top_k: int,
    halo_weight: float,
    halo_logit_clamp: float,
) -> dict[str, Any]:
    _validate_window_variant_config(
        input_mode=input_mode,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        halo_weight=halo_weight,
        halo_logit_clamp=halo_logit_clamp,
    )
    return {
        "input_mode": str(input_mode),
        "pooling_mode": str(pooling_mode),
        "pooling_top_k": int(pooling_top_k),
        "halo_weight": float(halo_weight),
        "halo_logit_clamp": float(halo_logit_clamp),
        "halo_logits_terminal_veto": False,
    }


def _validate_window_variant_config(
    *,
    input_mode: str,
    pooling_mode: str,
    pooling_top_k: int,
    halo_weight: float,
    halo_logit_clamp: float,
) -> None:
    if input_mode not in SUPPORTED_WINDOW_INPUT_MODES:
        raise ValueError(f"unsupported Window-MIL input mode: {input_mode}")
    if pooling_mode not in {WINDOW_POOLING_MAX, WINDOW_POOLING_TOPK_MEAN}:
        raise ValueError(f"unsupported Window-MIL pooling mode: {pooling_mode}")
    if int(pooling_top_k) < 1:
        raise ValueError("pooling_top_k must be positive")
    if pooling_mode == WINDOW_POOLING_MAX and int(pooling_top_k) != 1:
        raise ValueError("max pooling requires pooling_top_k=1")
    if not math.isfinite(float(halo_weight)) or float(halo_weight) < 0:
        raise ValueError("halo_weight must be finite and nonnegative")
    if not math.isfinite(float(halo_logit_clamp)) or float(halo_logit_clamp) <= 0:
        raise ValueError("halo_logit_clamp must be finite and positive")
    if input_mode == WINDOW_INPUT_SPLIT_HALO_CAPPED and pooling_mode == WINDOW_POOLING_MAX:
        raise ValueError("split_halo_capped must not use max pooling")


def _split_fit_rows(rows: Sequence[Mapping[str, Any]], *, context: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    eval_rows = [row for row in rows if row["split"] == "eval"]
    if not train_rows:
        raise ValueError(f"{context} requires train rows")
    if not calibration_rows:
        raise ValueError(f"{context} requires calibration rows")
    if not eval_rows:
        raise ValueError(f"{context} requires eval rows")
    for split_name, split_rows in (("train", train_rows), ("calibration", calibration_rows), ("eval", eval_rows)):
        labels = {str(row["classifier_label"]) for row in split_rows}
        if labels != {"safe", "reject"}:
            raise ValueError(f"{context} requires both safe and reject {split_name} examples")
    return train_rows, calibration_rows, eval_rows


def _balanced_sample_weights(labels: Sequence[int]) -> dict[int, float]:
    counts = Counter(int(label) for label in labels)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError("balanced sample weights require both labels")
    total = len(labels)
    return {0: total / (2.0 * counts[0]), 1: total / (2.0 * counts[1])}


def _deterministic_stratified_shuffle(rows: Sequence[Mapping[str, Any]], *, random_state: int) -> list[Mapping[str, Any]]:
    by_label: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["classifier_label"]), []).append(row)
    for label, label_rows in by_label.items():
        by_label[label] = sorted(
            label_rows,
            key=lambda row: stable_json_hash(
                {
                    "random_state": int(random_state),
                    "split": str(row["split"]),
                    "classifier_label": label,
                    "candidate_id": str(row["candidate_id"]),
                }
            ),
        )
    labels = sorted(by_label)
    ordered: list[Mapping[str, Any]] = []
    while any(by_label[label] for label in labels):
        for label in labels:
            if by_label[label]:
                ordered.append(by_label[label].pop(0))
    return ordered


def _chunk_sparse_checkpoint(
    *,
    completed_train_batches: int,
    total_train_batches: int,
    classifier_manifest_hash: str,
    evidence_cache_hash: str,
    feature_config_hash: str,
    training_config_hash: str,
    batch_size: int,
    random_state: int,
    model_file_sha256: str,
) -> dict[str, Any]:
    payload = {
        "artifact_kind": CHUNK_SPARSE_CHECKPOINT_KIND,
        "schema_version": 1,
        "completed_train_batches": int(completed_train_batches),
        "total_train_batches": int(total_train_batches),
        "classifier_manifest_hash": classifier_manifest_hash,
        "evidence_cache_hash": evidence_cache_hash,
        "feature_config_hash": feature_config_hash,
        "training_config_hash": training_config_hash,
        "batch_size": int(batch_size),
        "random_state": int(random_state),
        "model_file_name": f"{CHUNK_SPARSE_MODEL_ID}_model.joblib",
        "model_file_sha256": model_file_sha256,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    payload["checkpoint_hash"] = _hash_without(payload, "checkpoint_hash")
    validate_public_no_raw_payload(payload)
    return payload


def _validate_chunk_sparse_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model_path: Path,
    classifier_manifest_hash: str,
    evidence_cache_hash: str,
    feature_config_hash: str,
    training_config_hash: str,
    batch_size: int,
    total_train_batches: int,
    random_state: int,
) -> None:
    validate_public_no_raw_payload(checkpoint)
    if checkpoint.get("artifact_kind") != CHUNK_SPARSE_CHECKPOINT_KIND:
        raise ValueError("unsupported chunk sparse checkpoint kind")
    expected = {
        "classifier_manifest_hash": classifier_manifest_hash,
        "evidence_cache_hash": evidence_cache_hash,
        "feature_config_hash": feature_config_hash,
        "training_config_hash": training_config_hash,
        "batch_size": int(batch_size),
        "total_train_batches": int(total_train_batches),
        "random_state": int(random_state),
        "model_file_name": model_path.name,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"chunk sparse checkpoint {key} drift")
    completed = int(checkpoint.get("completed_train_batches", -1))
    if completed < 0 or completed > total_train_batches:
        raise ValueError("chunk sparse checkpoint completed batch count invalid")
    if not model_path.exists():
        raise ValueError("chunk sparse checkpoint model file missing")
    if str(checkpoint.get("model_file_sha256") or "") != _sha256_file(model_path):
        raise ValueError("chunk sparse checkpoint model file hash drift")
    if checkpoint.get("checkpoint_hash") != _hash_without(checkpoint, "checkpoint_hash"):
        raise ValueError("chunk sparse checkpoint hash mismatch")


def _save_chunk_window_checkpoint(
    *,
    torch: Any,
    checkpoint_path: Path,
    checkpoint_meta_path: Path,
    model: Any,
    optimizer: Any,
    scaler: Any,
    completed_train_batches: int,
    total_train_batches: int,
    classifier_manifest_hash: str,
    evidence_cache_hash: str,
    model_name: str,
    max_length: int,
    stride: int,
    batch_size: int,
    document_microbatch_size: int,
    epochs: int,
    epoch_index: int,
    batch_index: int,
    window_variant_config_hash: str,
    curriculum_hash: str,
) -> None:
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if hasattr(scaler, "state_dict") else {},
        "torch_rng_state": torch.get_rng_state(),
        "completed_train_batches": int(completed_train_batches),
    }
    cuda_rng_state_recorded = bool(getattr(torch, "cuda", None) is not None and torch.cuda.is_available())
    if cuda_rng_state_recorded:
        checkpoint_payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint_payload, checkpoint_path)
    meta = {
        "artifact_kind": CHUNK_WINDOW_MIL_CHECKPOINT_KIND,
        "schema_version": 1,
        "completed_train_batches": int(completed_train_batches),
        "total_train_batches": int(total_train_batches),
        "classifier_manifest_hash": classifier_manifest_hash,
        "evidence_cache_hash": evidence_cache_hash,
        "base_model_name_hash": stable_json_hash({"base_model_name": model_name}),
        "window_variant_config_hash": window_variant_config_hash,
        "curriculum_hash": curriculum_hash,
        "max_length": int(max_length),
        "stride": int(stride),
        "batch_size": int(batch_size),
        "document_microbatch_size": int(document_microbatch_size),
        "activation_checkpoint_windows": CHUNK_WINDOW_ACTIVATION_CHECKPOINT_WINDOWS,
        "epochs": int(epochs),
        "epoch_index": int(epoch_index),
        "batch_index": int(batch_index),
        "scaler_state_recorded": hasattr(scaler, "state_dict"),
        "rng_state_recorded": True,
        "cuda_rng_state_recorded": cuda_rng_state_recorded,
        "checkpoint_file_sha256": _sha256_file(checkpoint_path),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    meta["checkpoint_hash"] = _hash_without(meta, "checkpoint_hash")
    _write_public_json(checkpoint_meta_path, meta)


def _validate_chunk_window_checkpoint_meta(
    meta: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    classifier_manifest_hash: str,
    evidence_cache_hash: str,
    model_name: str,
    max_length: int,
    stride: int,
    batch_size: int,
    document_microbatch_size: int,
    epochs: int,
    total_train_batches: int,
    window_variant_config_hash: str,
    curriculum_hash: str,
) -> None:
    validate_public_no_raw_payload(meta)
    if meta.get("artifact_kind") != CHUNK_WINDOW_MIL_CHECKPOINT_KIND:
        raise ValueError("unsupported chunk window-MIL checkpoint kind")
    expected = {
        "classifier_manifest_hash": classifier_manifest_hash,
        "evidence_cache_hash": evidence_cache_hash,
        "base_model_name_hash": stable_json_hash({"base_model_name": model_name}),
        "window_variant_config_hash": window_variant_config_hash,
        "curriculum_hash": curriculum_hash,
        "max_length": int(max_length),
        "stride": int(stride),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "total_train_batches": int(total_train_batches),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"chunk window-MIL checkpoint {key} drift")
    if meta.get("activation_checkpoint_windows") is not True:
        raise ValueError("chunk window-MIL checkpoint activation_checkpoint_windows drift")
    if "document_microbatch_size" in meta:
        if int(meta["document_microbatch_size"]) != int(document_microbatch_size):
            raise ValueError("chunk window-MIL checkpoint document_microbatch_size drift")
    elif int(document_microbatch_size) != int(batch_size):
        raise ValueError("chunk window-MIL checkpoint document_microbatch_size drift")
    completed = int(meta.get("completed_train_batches", -1))
    if completed < 0 or completed > total_train_batches:
        raise ValueError("chunk window-MIL checkpoint completed batch count invalid")
    if meta.get("checkpoint_file_sha256") != _sha256_file(checkpoint_path):
        raise ValueError("chunk window-MIL checkpoint file hash drift")
    if meta.get("checkpoint_hash") != _hash_without(meta, "checkpoint_hash"):
        raise ValueError("chunk window-MIL checkpoint hash mismatch")


def _restore_chunk_window_rng_state(*, torch: Any, checkpoint: Mapping[str, Any], resolved_device: Any) -> None:
    if "torch_rng_state" not in checkpoint:
        raise ValueError("chunk window-MIL checkpoint missing torch RNG state")
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if str(getattr(resolved_device, "type", "")) != "cuda":
        return
    if "cuda_rng_state" not in checkpoint:
        raise ValueError("chunk window-MIL checkpoint missing CUDA RNG state")
    cuda_rng_state = checkpoint["cuda_rng_state"]
    if hasattr(torch.cuda, "set_rng_state_all"):
        torch.cuda.set_rng_state_all(cuda_rng_state)
    else:  # pragma: no cover - old torch compatibility
        first_state = cuda_rng_state[0] if isinstance(cuda_rng_state, list) else cuda_rng_state
        torch.cuda.set_rng_state(first_state)


def _validate_existing_chunk_window_report_matches_invocation(
    report: Mapping[str, Any],
    *,
    classifier_manifest_hash: str,
    evidence_cache_hash: str,
    target_recall: float,
    batch_size: int,
    max_length: int,
    stride: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_windows_per_device_batch: int,
    document_microbatch_size: int,
    checkpoint_every_batches: int,
    window_variant_config_hash: str,
    curriculum_hash: str,
) -> None:
    exact_expected = {
        "classifier_manifest_hash": classifier_manifest_hash,
        "evidence_cache_hash": evidence_cache_hash,
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "stride": int(stride),
        "epochs": int(epochs),
        "max_windows_per_device_batch": int(max_windows_per_device_batch),
        "checkpoint_every_batches": int(checkpoint_every_batches),
        "activation_checkpoint_windows": CHUNK_WINDOW_ACTIVATION_CHECKPOINT_WINDOWS,
        "window_variant_config_hash": window_variant_config_hash,
        "curriculum_hash": curriculum_hash,
    }
    for key, expected in exact_expected.items():
        if report.get(key) != expected:
            raise ValueError(f"existing chunk window-MIL report {key.replace('_', ' ')} drift")
    if "document_microbatch_size" in report:
        if int(report["document_microbatch_size"]) != int(document_microbatch_size):
            raise ValueError("existing chunk window-MIL report document microbatch size drift")
    elif int(document_microbatch_size) != int(batch_size):
        raise ValueError("existing chunk window-MIL report document microbatch size drift")
    float_expected = {
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
    }
    for key, expected in float_expected.items():
        actual = float(report.get(key, float("nan")))
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"existing chunk window-MIL report {key.replace('_', ' ')} drift")
    threshold_target = float(report.get("threshold_selection", {}).get("target_reject_recall", float("nan")))
    if not math.isfinite(threshold_target) or not math.isclose(
        threshold_target,
        float(target_recall),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("existing chunk window-MIL report target reject recall drift")


def _validate_existing_chunk_sparse_report_matches_invocation(
    report: Mapping[str, Any],
    *,
    classifier_manifest_hash: str,
    evidence_cache_hash: str,
    target_recall: float,
    batch_size: int,
    n_features: int,
    random_state: int,
    alpha: float,
    feature_config_hash: str,
    training_config_hash: str,
    curriculum_hash: str,
) -> None:
    feature_config = report.get("feature_config")
    if not isinstance(feature_config, Mapping) or int(feature_config.get("n_features", -1)) != int(n_features):
        raise ValueError("existing chunk sparse report n features drift")
    alpha_actual = float(report.get("alpha", float("nan")))
    if not math.isfinite(alpha_actual) or not math.isclose(alpha_actual, float(alpha), rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("existing chunk sparse report alpha drift")
    exact_expected = {
        "classifier_manifest_hash": classifier_manifest_hash,
        "evidence_cache_hash": evidence_cache_hash,
        "feature_config_hash": feature_config_hash,
        "training_config_hash": training_config_hash,
        "curriculum_hash": curriculum_hash,
        "batch_size": int(batch_size),
        "random_state": int(random_state),
    }
    for key, expected in exact_expected.items():
        if report.get(key) != expected:
            raise ValueError(f"existing chunk sparse report {key.replace('_', ' ')} drift")
    threshold_target = float(report.get("threshold_selection", {}).get("target_reject_recall", float("nan")))
    if not math.isfinite(threshold_target) or not math.isclose(
        threshold_target,
        float(target_recall),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("existing chunk sparse report target reject recall drift")


def _chunk_progress_payload(
    *,
    artifact_kind: str,
    status: str,
    completed_train_batches: int,
    total_train_batches: int,
    fit_eligible_row_count: int,
    mean_train_loss: float | None = None,
) -> dict[str, Any]:
    if mean_train_loss is not None and not math.isfinite(float(mean_train_loss)):
        raise ValueError("chunk progress mean_train_loss must be finite")
    payload = {
        "artifact_kind": artifact_kind,
        "schema_version": 1,
        "status": status,
        "completed_train_batches": int(completed_train_batches),
        "total_train_batches": int(total_train_batches),
        "fit_eligible_row_count": int(fit_eligible_row_count),
        "mean_train_loss": None if mean_train_loss is None else float(mean_train_loss),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    payload["progress_hash"] = _hash_without(payload, "progress_hash")
    validate_public_no_raw_payload(payload)
    return payload


def _mean_train_loss(loss_sum: float, trained_rows: int) -> float:
    result = float(loss_sum) / max(1, int(trained_rows))
    if not math.isfinite(result):
        raise ValueError("non-finite training loss")
    return result


def _sparse_feature_config(*, n_features: int) -> dict[str, Any]:
    return {
        "family": "chunk_hashed_word_char_ngrams",
        "n_features": int(n_features),
        "word_ngram_range": [1, 2],
        "char_ngram_range": [3, 5],
        "normalization": "sublinear_tf_l2",
        "input": "core_plus_halo_evidence_text",
    }


def _feature_id_to_column(feature_id: str, *, n_features: int) -> int:
    if not feature_id.startswith("h"):
        raise ValueError(f"unsupported feature id: {feature_id}")
    column = int(feature_id[1:], 16)
    if not 0 <= column < n_features:
        raise ValueError(f"feature id out of configured range: {feature_id}")
    return column


def _sparse_model_numeric_hash(model: Any, *, deps: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for attribute in ("coef_", "intercept_"):
        value = getattr(model, attribute, None)
        if value is None:
            continue
        array = deps["numpy"].ascontiguousarray(value)
        digest.update(array.tobytes())
    return digest.hexdigest()


def _require_metric_block(report: Mapping[str, Any], field: str) -> None:
    metrics = report.get(field)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{field} missing")
    for key in ("count", "positive_count", "negative_count", "recall", "false_negative"):
        if key not in metrics:
            raise ValueError(f"{field} missing {key}")
    if int(metrics["positive_count"]) < 1 or int(metrics["negative_count"]) < 1:
        raise ValueError(f"{field} must contain both labels")
    _require_finite_scores([float(metrics["recall"])], field)


def _validate_metric_counts_match_splits(report: Mapping[str, Any]) -> None:
    split_counts = report.get("split_counts")
    if not isinstance(split_counts, Mapping):
        raise ValueError("report split_counts missing")
    expected = {
        "calibration": int(split_counts.get("calibration", -1)),
        "eval": int(split_counts.get("eval", -1)),
    }
    actual = {
        "calibration": int(report["calibration_metrics"]["count"]),
        "eval": int(report["eval_metrics"]["count"]),
    }
    for split_name in ("calibration", "eval"):
        if actual[split_name] != expected[split_name]:
            raise ValueError(f"{split_name} metric count mismatch")


def _require_finite_scores(scores: Sequence[float], label: str) -> None:
    if not scores:
        raise ValueError(f"{label} scores are empty")
    for score in scores:
        if not math.isfinite(float(score)):
            raise ValueError(f"{label} contains non-finite score")


def _require_false_report_flags(report: Mapping[str, Any], *, allow_gpu_rental: bool = False) -> None:
    fields = (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "labels_are_corpus_approval",
        "accepted_manifest_created",
        "tokenizer_shards_created",
        "training_shards_created",
        "classifier_training_performed",
        "base_pretraining_started",
        "pretraining_started",
        "paid_api_used",
        "gpu_rental_used",
    )
    for field in fields:
        if field not in report:
            continue
        if field == "gpu_rental_used" and allow_gpu_rental:
            continue
        if report.get(field) is not False:
            raise ValueError(f"report must keep {field}=false")


def _validate_report_file_hash(report: Mapping[str, Any], model_path: Path) -> None:
    if not model_path.exists():
        raise ValueError("model file missing for existing report")
    if report.get("model_file_name") != model_path.name:
        raise ValueError("model file name drift for existing report")
    if report.get("model_file_sha256") != _sha256_file(model_path):
        raise ValueError("model file hash drift for existing report")


def _label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return _sorted_counter(Counter(str(row["classifier_label"]) for row in rows))


def _split_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return _sorted_counter(Counter(str(row["split"]) for row in rows))


def _evidence_tier_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return _sorted_counter(Counter(str(row["evidence_tier"]) for row in rows))


def _window_runtime_profile(
    *,
    trained_rows: int,
    completed_train_batches: int,
    elapsed_seconds: float,
    train_step_seconds: float,
    checkpoint_write_seconds: float,
    rows: Sequence[Mapping[str, Any]],
    max_length: int,
    stride: int,
    resolved_device_type: str,
) -> dict[str, Any]:
    char_lengths = [int(row["evidence_char_length"]) for row in rows]
    estimated_windows = [_estimated_window_count(length, max_length=max_length, stride=stride) for length in char_lengths]
    payload = {
        "input_pipeline": "cache_text_bucket_ready_v1",
        "private_text_source": "sqlite_evidence_cache",
        "checkpoint_resume_support": True,
        "activation_checkpoint_windows": CHUNK_WINDOW_ACTIVATION_CHECKPOINT_WINDOWS,
        "scaler_state_checkpointed": True,
        "rng_state_checkpointed": True,
        "async_tokenization_workers_enabled": False,
        "pinned_memory_expected_on_cuda": resolved_device_type == "cuda",
        "completed_train_batches": int(completed_train_batches),
        "trained_rows": int(trained_rows),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "train_step_seconds": round(float(train_step_seconds), 6),
        "checkpoint_write_seconds": round(float(checkpoint_write_seconds), 6),
        "rows_per_second": 0.0 if elapsed_seconds <= 0 else float(trained_rows) / float(elapsed_seconds),
        "evidence_char_length": _distribution_summary(char_lengths),
        "estimated_window_count": _distribution_summary(estimated_windows),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    validate_public_no_raw_payload(payload)
    return payload


def _estimated_window_count(char_length: int, *, max_length: int, stride: int) -> int:
    # Public telemetry uses an intentionally rough character proxy, never token IDs.
    estimated_tokens = max(1, math.ceil(int(char_length) / 4))
    effective_step = max(1, int(max_length) - int(stride))
    if estimated_tokens <= max_length:
        return 1
    return 1 + math.ceil((estimated_tokens - max_length) / effective_step)


def _distribution_summary(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(int(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[int], q: float) -> int:
    if not ordered:
        return 0
    index = min(len(ordered) - 1, max(0, math.ceil(float(q) * len(ordered)) - 1))
    return int(ordered[index])


def _validate_runtime_profile(profile: Any) -> None:
    if not isinstance(profile, Mapping):
        raise ValueError("window-MIL runtime_profile must be an object")
    validate_public_no_raw_payload(profile)
    if profile.get("checkpoint_resume_support") is not True:
        raise ValueError("window-MIL runtime_profile must confirm checkpoint resume support")
    if profile.get("activation_checkpoint_windows") is not True:
        raise ValueError("window-MIL runtime_profile must confirm activation checkpointing")
    for field in ("elapsed_seconds", "train_step_seconds", "checkpoint_write_seconds", "rows_per_second"):
        value = float(profile.get(field, -1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"window-MIL runtime_profile {field} invalid")


def validate_window_mil_gpu_canary_gate(
    report: Mapping[str, Any],
    *,
    min_rows_per_second: float,
    max_checkpoint_fraction: float,
    min_trained_rows: int = 1,
    require_cuda: bool = True,
    require_amp: bool = True,
    require_full_eval: bool = True,
) -> dict[str, Any]:
    validate_chunk_window_mil_report(report)
    if min_rows_per_second < 0:
        raise ValueError("min_rows_per_second must be nonnegative")
    if not 0 <= max_checkpoint_fraction <= 1:
        raise ValueError("max_checkpoint_fraction must be in [0, 1]")
    runtime = report.get("runtime_profile")
    if not isinstance(runtime, Mapping):
        raise ValueError("window-MIL GPU canary requires runtime_profile")
    device = report.get("device")
    if not isinstance(device, Mapping):
        raise ValueError("window-MIL GPU canary requires device observations")
    if require_cuda and device.get("device_type") != "cuda":
        raise ValueError("window-MIL GPU canary did not run on CUDA")
    if require_cuda and report.get("gpu_rental_used") is not True:
        raise ValueError("window-MIL GPU canary report does not mark gpu_rental_used=true")
    if require_amp and report.get("amp_enabled") is not True:
        raise ValueError("window-MIL GPU canary did not use AMP")
    trained_rows = int(runtime.get("trained_rows", 0))
    if trained_rows < int(min_trained_rows):
        raise ValueError("window-MIL GPU canary trained too few rows")
    completed = int(report.get("completed_train_batches", -1))
    total = int(report.get("total_train_batches", -1))
    if completed < 1 or completed != total:
        raise ValueError("window-MIL GPU canary did not complete all requested batches")
    rows_per_second = float(runtime.get("rows_per_second", -1.0))
    if not math.isfinite(rows_per_second) or rows_per_second < float(min_rows_per_second):
        raise ValueError("window-MIL GPU canary rows_per_second below threshold")
    elapsed = float(runtime.get("elapsed_seconds", 0.0))
    checkpoint_write = float(runtime.get("checkpoint_write_seconds", 0.0))
    checkpoint_fraction = 0.0 if elapsed <= 0 else checkpoint_write / elapsed
    if checkpoint_fraction > float(max_checkpoint_fraction):
        raise ValueError("window-MIL GPU canary checkpoint overhead above threshold")
    if require_full_eval:
        split_counts = report.get("split_counts")
        if not isinstance(split_counts, Mapping):
            raise ValueError("window-MIL GPU canary missing split counts")
        for split_name, metric_name in (("calibration", "calibration_metrics"), ("eval", "eval_metrics")):
            metrics = report.get(metric_name)
            if not isinstance(metrics, Mapping) or int(metrics.get("count", -1)) != int(split_counts.get(split_name, -2)):
                raise ValueError(f"window-MIL GPU canary {split_name} metrics are truncated")
    payload = {
        "artifact_kind": "twc_chunk_window_mil_gpu_canary_gate_v1",
        "status": "passed",
        "window_mil_report_hash": str(report["report_hash"]),
        "model_payload_hash": str(report["model_payload_hash"]),
        "device_type": str(device.get("device_type")),
        "trained_rows": trained_rows,
        "rows_per_second": rows_per_second,
        "checkpoint_fraction": checkpoint_fraction,
        "min_rows_per_second": float(min_rows_per_second),
        "max_checkpoint_fraction": float(max_checkpoint_fraction),
        "require_cuda": bool(require_cuda),
        "require_amp": bool(require_amp),
        "require_full_eval": bool(require_full_eval),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["gate_hash"] = _hash_without(payload, "gate_hash")
    validate_public_no_raw_payload(payload)
    return payload


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _required_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return result


def _positive_int(value: Any, field_name: str) -> int:
    result = _nonnegative_int(value, field_name)
    if result < 1:
        raise ValueError(f"{field_name} must be positive")
    return result


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _hash_without(payload: Mapping[str, Any], field: str) -> str:
    return stable_json_hash({key: value for key, value in payload.items() if key != field})


def _write_public_json(path: Path, payload: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _chmod_private_sqlite_files(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch(mode=0o600)
    for sqlite_path in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if sqlite_path.exists():
            sqlite_path.chmod(0o600)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_and_validate_cli_curriculum(
    args: argparse.Namespace,
    *,
    classifier_manifest: Mapping[str, Any],
    evidence_cache_report: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    curriculum_path = getattr(args, "curriculum", None)
    if not curriculum_path:
        return None
    if evidence_cache_report is None:
        raise ValueError("--evidence-cache-report is required when --curriculum is set")
    score_ledger_path = getattr(args, "curriculum_score_ledger", None)
    if not score_ledger_path:
        raise ValueError("--curriculum-score-ledger is required when --curriculum is set")
    curriculum = _read_json(curriculum_path)
    validate_chunk_hard_curriculum_binding(
        classifier_manifest=classifier_manifest,
        evidence_cache_report=evidence_cache_report,
        training_score_ledger=_read_json(score_ledger_path),
        curriculum=curriculum,
    )
    return curriculum


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(payload)
    return rows


class _FallbackRowProvider:
    """Try a fast private row provider before falling back to remote refetch."""

    manage_hf_cache_env = False

    def __init__(self, *, primary: Any, fallback: Any) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_hit_group_count = 0
        self.fallback_group_count = 0
        self.fetch_call_count = 0
        self.last_primary_error_type: str | None = None

    def fetch_rows(self, entry: dict[str, Any], offsets: list[int]) -> dict[int, Any]:
        key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
        return self.fetch_row_groups([(entry, offsets)])[key]

    def fetch_row_groups(self, entry_offsets: list[tuple[dict[str, Any], list[int]]]) -> dict[tuple[str, int], dict[int, Any]]:
        self.fetch_call_count += 1
        resolved: dict[tuple[str, int], dict[int, Any]] = {}
        fallback_inputs: list[tuple[dict[str, Any], list[int]]] = []
        for entry, offsets in entry_offsets:
            key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
            try:
                primary_result = self.primary.fetch_row_groups([(entry, offsets)])
            except Exception as exc:
                self.last_primary_error_type = type(exc).__name__
                fallback_inputs.append((entry, offsets))
                continue
            if key not in primary_result:
                fallback_inputs.append((entry, offsets))
                continue
            resolved[key] = primary_result[key]
            self.primary_hit_group_count += 1
        if fallback_inputs:
            fallback_result = self.fallback.fetch_row_groups(fallback_inputs)
            for key, rows in fallback_result.items():
                if key in resolved:
                    raise ValueError("fallback row provider returned duplicate row group")
                resolved[key] = rows
                self.fallback_group_count += 1
        return resolved

    def observations(self) -> dict[str, Any]:
        payload = {
            "provider_class": type(self).__name__,
            "primary_provider_class": type(self.primary).__name__,
            "fallback_provider_class": type(self.fallback).__name__,
            "fetch_call_count": int(self.fetch_call_count),
            "primary_hit_group_count": int(self.primary_hit_group_count),
            "fallback_group_count": int(self.fallback_group_count),
            "last_primary_error_type": self.last_primary_error_type,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
        validate_public_no_raw_payload(payload)
        return payload


def _build_row_provider(args: argparse.Namespace) -> Any:
    row_provider = getattr(args, "row_provider", "hf-fsspec")
    if row_provider == "hf-fsspec":
        return _build_hf_provider(args)
    if row_provider == "private-cache":
        return PrivateTextCacheRowProvider(cache_root=args.private_cache_root)
    if row_provider == "private-cache-with-hf-fallback":
        return _FallbackRowProvider(
            primary=PrivateTextCacheRowProvider(cache_root=args.private_cache_root),
            fallback=_build_hf_provider(args),
        )
    raise ValueError(f"unsupported row provider: {row_provider}")


def _build_hf_provider(args: argparse.Namespace) -> HFFsspecParquetRowProvider:
    if not args.hf_cache_dir:
        raise ValueError("--hf-cache-dir is required")
    return HFFsspecParquetRowProvider(
        cache_dir=args.hf_cache_dir,
        cache_delta_cap_bytes=args.max_cache_delta_bytes,
        block_size_bytes=args.block_size_bytes,
        columns=("text",),
        max_cache_blocks=args.max_cache_blocks,
        manage_hf_cache_env=False,
        row_group_batch_size=args.row_group_batch_size,
        source_concurrency=args.source_concurrency,
        source_retry_count=args.source_retry_count,
        source_retry_backoff_seconds=args.source_retry_backoff_seconds,
        http_total_timeout_seconds=args.http_total_timeout_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and train TWC chunk classifier artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_inherited_safe = subparsers.add_parser("build-inherited-safe")
    build_inherited_safe.add_argument("--source-manifest", required=True)
    build_inherited_safe.add_argument("--labels-jsonl", required=True)
    build_inherited_safe.add_argument("--source-schedule", required=True)
    build_inherited_safe.add_argument("--output-candidates-jsonl", required=True)
    build_inherited_safe.add_argument("--output-private-map-json", required=True)
    build_inherited_safe.add_argument("--output-report-json")
    build_inherited_safe.add_argument("--target-proxy-tokens", type=int, default=1024)
    build_inherited_safe.add_argument("--max-parent-docs", type=int)
    build_inherited_safe.add_argument("--skip-parent-docs", type=int, default=0)
    build_inherited_safe.add_argument(
        "--row-provider",
        choices=("hf-fsspec", "private-cache", "private-cache-with-hf-fallback"),
        default="hf-fsspec",
    )
    build_inherited_safe.add_argument("--private-cache-root", default=str(DEFAULT_CACHE_ROOT))
    build_inherited_safe.add_argument("--hf-cache-dir")
    build_inherited_safe.add_argument("--max-cache-delta-bytes", type=int, default=8_000_000_000)
    build_inherited_safe.add_argument("--block-size-bytes", type=int, default=DEFAULT_HF_ROW_READ_BLOCK_SIZE_BYTES)
    build_inherited_safe.add_argument("--max-cache-blocks", type=int, default=64)
    build_inherited_safe.add_argument("--row-group-batch-size", type=int, default=8)
    build_inherited_safe.add_argument("--source-concurrency", type=int, default=4)
    build_inherited_safe.add_argument("--source-retry-count", type=int, default=4)
    build_inherited_safe.add_argument("--source-retry-backoff-seconds", type=float, default=1.0)
    build_inherited_safe.add_argument("--http-total-timeout-seconds", type=float, default=240.0)

    build_cache = subparsers.add_parser("build-cache")
    build_cache.add_argument("--classifier-manifest", required=True)
    build_cache.add_argument("--source-schedule", action="append", required=True)
    build_cache.add_argument("--stage-dir", action="append", default=[])
    build_cache.add_argument("--stage-root")
    build_cache.add_argument("--stage-glob", default="rejected_35k_chunk_relabel_stage*")
    build_cache.add_argument("--safe-parent-private-map", action="append", default=[])
    build_cache.add_argument("--active-learning-private-map", action="append", default=[])
    build_cache.add_argument("--cache-db", required=True)
    build_cache.add_argument("--output", required=True)
    build_cache.add_argument("--progress-output")
    build_cache.add_argument(
        "--row-provider",
        choices=("hf-fsspec", "private-cache", "private-cache-with-hf-fallback"),
        default="hf-fsspec",
    )
    build_cache.add_argument("--private-cache-root", default=str(DEFAULT_CACHE_ROOT))
    build_cache.add_argument("--halo-chars", type=int, default=DEFAULT_HALO_CHARS)
    build_cache.add_argument("--fetch-batch-size", type=int, default=512)
    build_cache.add_argument("--max-rows", type=int)
    build_cache.add_argument("--require-private-cache-route", action="append", default=[])
    build_cache.add_argument("--require-active-learning-batch", action="append", default=[])
    build_cache.add_argument("--hf-cache-dir")
    build_cache.add_argument("--max-cache-delta-bytes", type=int, default=8_000_000_000)
    build_cache.add_argument("--block-size-bytes", type=int, default=DEFAULT_HF_ROW_READ_BLOCK_SIZE_BYTES)
    build_cache.add_argument("--max-cache-blocks", type=int, default=64)
    build_cache.add_argument("--row-group-batch-size", type=int, default=8)
    build_cache.add_argument("--source-concurrency", type=int, default=4)
    build_cache.add_argument("--source-retry-count", type=int, default=4)
    build_cache.add_argument("--source-retry-backoff-seconds", type=float, default=1.0)
    build_cache.add_argument("--http-total-timeout-seconds", type=float, default=240.0)

    train_sparse = subparsers.add_parser("train-sparse")
    train_sparse.add_argument("--classifier-manifest", required=True)
    train_sparse.add_argument("--evidence-cache-report")
    train_sparse.add_argument("--cache-db", required=True)
    train_sparse.add_argument("--artifact-dir", required=True)
    train_sparse.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    train_sparse.add_argument("--n-features", type=int, default=DEFAULT_SPARSE_N_FEATURES)
    train_sparse.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    train_sparse.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    train_sparse.add_argument("--alpha", type=float, default=1e-4)
    train_sparse.add_argument("--curriculum")
    train_sparse.add_argument("--curriculum-score-ledger")
    train_sparse.add_argument("--max-train-batches", type=int)

    train_window = subparsers.add_parser("train-window")
    train_window.add_argument("--classifier-manifest", required=True)
    train_window.add_argument("--evidence-cache-report")
    train_window.add_argument("--cache-db", required=True)
    train_window.add_argument("--artifact-dir", required=True)
    train_window.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    train_window.add_argument("--batch-size", type=int, default=8)
    train_window.add_argument("--max-length", type=int, default=512)
    train_window.add_argument("--stride", type=int, default=128)
    train_window.add_argument("--epochs", type=int, default=1)
    train_window.add_argument("--learning-rate", type=float, default=2e-5)
    train_window.add_argument("--weight-decay", type=float, default=0.01)
    train_window.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    train_window.add_argument("--curriculum")
    train_window.add_argument("--curriculum-score-ledger")
    train_window.add_argument(
        "--input-mode",
        choices=sorted(SUPPORTED_WINDOW_INPUT_MODES),
        default=WINDOW_INPUT_FULL_EVIDENCE,
    )
    train_window.add_argument(
        "--pooling-mode",
        choices=sorted({WINDOW_POOLING_MAX, WINDOW_POOLING_TOPK_MEAN}),
        default=WINDOW_POOLING_MAX,
    )
    train_window.add_argument("--pooling-top-k", type=int, default=1)
    train_window.add_argument("--halo-weight", type=float, default=0.25)
    train_window.add_argument("--halo-logit-clamp", type=float, default=2.0)
    train_window.add_argument("--device", default="auto")
    train_window.add_argument("--require-cuda", action="store_true")
    train_window.add_argument("--disable-amp", action="store_true")
    train_window.add_argument("--max-windows-per-device-batch", type=int, default=512)
    train_window.add_argument("--document-microbatch-size", type=int)
    train_window.add_argument("--checkpoint-every-batches", type=int, default=1)
    train_window.add_argument("--max-train-batches", type=int)
    train_window.add_argument("--max-eval-batches", type=int)

    validate_gpu_canary = subparsers.add_parser("validate-window-gpu-canary")
    validate_gpu_canary.add_argument("--window-mil-report", required=True)
    validate_gpu_canary.add_argument("--output")
    validate_gpu_canary.add_argument("--min-rows-per-second", type=float, required=True)
    validate_gpu_canary.add_argument("--max-checkpoint-fraction", type=float, default=0.25)
    validate_gpu_canary.add_argument("--min-trained-rows", type=int, default=1)
    validate_gpu_canary.add_argument("--allow-cpu", action="store_true")
    validate_gpu_canary.add_argument("--allow-amp-disabled", action="store_true")
    validate_gpu_canary.add_argument("--allow-eval-truncation", action="store_true")

    variant_registry = subparsers.add_parser("build-variant-registry")
    variant_registry.add_argument("--classifier-manifest", required=True)
    variant_registry.add_argument("--evidence-cache-report", required=True)
    variant_registry.add_argument("--output", required=True)
    variant_registry.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    variant_registry.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    variant_registry.add_argument("--artifact-root", default="reports/classifiers/chunk_classifier_v2_multi_variant")

    train_score_ledger = subparsers.add_parser("build-train-score-ledger")
    train_score_ledger.add_argument("--classifier-manifest", required=True)
    train_score_ledger.add_argument("--evidence-cache-report", required=True)
    train_score_ledger.add_argument("--sparse-report", required=True)
    train_score_ledger.add_argument("--window-mil-report", required=True)
    train_score_ledger.add_argument("--cache-db", required=True)
    train_score_ledger.add_argument("--sparse-model-file", required=True)
    train_score_ledger.add_argument("--window-mil-model-file", required=True)
    train_score_ledger.add_argument("--output", required=True)
    train_score_ledger.add_argument("--device", default="auto")
    train_score_ledger.add_argument("--require-cuda", action="store_true")
    train_score_ledger.add_argument("--sparse-score-batch-size", type=int, default=512)
    train_score_ledger.add_argument("--window-mil-score-batch-size", type=int, default=32)

    hard_curriculum = subparsers.add_parser("build-hard-curriculum")
    hard_curriculum.add_argument("--classifier-manifest", required=True)
    hard_curriculum.add_argument("--evidence-cache-report", required=True)
    hard_curriculum.add_argument("--train-score-ledger", required=True)
    hard_curriculum.add_argument("--hard-source-bucket-hash", action="append", default=[])
    hard_curriculum.add_argument("--output", required=True)

    validate_hard_curriculum_binding = subparsers.add_parser("validate-hard-curriculum-binding")
    validate_hard_curriculum_binding.add_argument("--classifier-manifest", required=True)
    validate_hard_curriculum_binding.add_argument("--evidence-cache-report", required=True)
    validate_hard_curriculum_binding.add_argument("--train-score-ledger", required=True)
    validate_hard_curriculum_binding.add_argument("--curriculum", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--classifier-manifest", required=True)
    freeze.add_argument("--evidence-cache-report", required=True)
    freeze.add_argument("--sparse-report", required=True)
    freeze.add_argument("--window-mil-report", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)

    score_ledger = subparsers.add_parser("build-score-ledger")
    score_ledger.add_argument("--classifier-manifest", required=True)
    score_ledger.add_argument("--evidence-cache-report", required=True)
    score_ledger.add_argument("--sparse-report", required=True)
    score_ledger.add_argument("--window-mil-report", required=True)
    score_ledger.add_argument("--cache-db", required=True)
    score_ledger.add_argument("--sparse-model-file", required=True)
    score_ledger.add_argument("--window-mil-model-file", required=True)
    score_ledger.add_argument("--output", required=True)
    score_ledger.add_argument("--device", default="auto")
    score_ledger.add_argument("--require-cuda", action="store_true")
    score_ledger.add_argument(
        "--window-mil-precision",
        choices=("fp32", "tf32", "amp_bf16"),
        default="fp32",
    )

    deterministic_ledger = subparsers.add_parser("build-deterministic-ledger")
    deterministic_ledger.add_argument("--classifier-manifest", required=True)
    deterministic_ledger.add_argument("--evidence-cache-report", required=True)
    deterministic_ledger.add_argument("--cache-db", required=True)
    deterministic_ledger.add_argument("--deterministic-ontology", required=True)
    deterministic_ledger.add_argument("--deterministic-policy-payload")
    deterministic_ledger.add_argument("--output", required=True)

    sweep_policy = subparsers.add_parser("sweep-policy")
    sweep_policy.add_argument("--score-ledger", required=True)
    sweep_policy.add_argument("--deterministic-ledger", required=True)
    sweep_policy.add_argument("--output", required=True)
    sweep_policy.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    sweep_policy.add_argument("--max-reported-operating-points", type=int, default=128)
    sweep_policy.add_argument("--fixed-sparse-threshold", type=float)
    sweep_policy.add_argument("--fixed-window-mil-threshold", type=float)

    simulate_clean_runs = subparsers.add_parser("simulate-clean-runs")
    simulate_clean_runs.add_argument("--score-ledger", required=True)
    simulate_clean_runs.add_argument("--deterministic-ledger", required=True)
    simulate_clean_runs.add_argument("--policy-sweep", required=True)
    simulate_clean_runs.add_argument("--split", choices=("calibration", "eval"), required=True)
    simulate_clean_runs.add_argument("--output", required=True)

    freeze_eval = subparsers.add_parser("freeze-eval")
    freeze_eval.add_argument("--score-ledger", required=True)
    freeze_eval.add_argument("--deterministic-ledger", required=True)
    freeze_eval.add_argument("--policy-sweep", required=True)
    freeze_eval.add_argument("--output", required=True)
    freeze_eval.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)

    validate = subparsers.add_parser("validate-report")
    validate.add_argument(
        "--kind",
        choices=(
            "cache",
            "sparse",
            "window",
            "freeze",
            "variant-registry",
            "train-score-ledger",
            "hard-curriculum",
            "score-ledger",
            "deterministic-ledger",
            "policy-sweep",
            "clean-run-simulation",
            "policy-freeze",
        ),
        required=True,
    )
    validate.add_argument("--input", required=True)

    args = parser.parse_args(argv)
    if args.command == "build-inherited-safe":
        provider = _build_row_provider(args)
        artifacts = build_inherited_safe_parent_chunk_artifacts(
            source_manifest=_read_json(args.source_manifest),
            final_labels=_read_jsonl(args.labels_jsonl),
            source_schedule=_read_json(args.source_schedule),
            row_provider=provider,
            output_candidates_jsonl=args.output_candidates_jsonl,
            output_private_map_json=args.output_private_map_json,
            output_report_json=args.output_report_json,
            target_proxy_tokens=args.target_proxy_tokens,
            max_parent_docs=args.max_parent_docs,
            skip_parent_docs=args.skip_parent_docs,
        )
        report = artifacts["report"]
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "parent_document_count": report["parent_document_count"],
                    "candidate_count": report["candidate_count"],
                    "private_map_hash": report["private_map_hash"],
                    "candidate_rows_hash": report["candidate_rows_hash"],
                    "paid_api_calls_performed": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "build-cache":
        stage_dirs = [Path(path) for path in args.stage_dir]
        if args.stage_root:
            stage_dirs.extend(_complete_stage_dirs_from_root(Path(args.stage_root), args.stage_glob))
        provider = _build_row_provider(args)
        report = build_chunk_evidence_cache(
            classifier_manifest=_read_json(args.classifier_manifest),
            stage_dirs=stage_dirs,
            safe_parent_private_maps=args.safe_parent_private_map,
            active_learning_private_maps=args.active_learning_private_map,
            source_schedule=[_read_json(path) for path in args.source_schedule],
            row_provider=provider,
            cache_db=args.cache_db,
            output_json=args.output,
            halo_chars=args.halo_chars,
            fetch_batch_size=args.fetch_batch_size,
            max_rows=args.max_rows,
            progress_json=args.progress_output,
            require_private_cache_routes=args.require_private_cache_route,
            require_active_learning_batches=args.require_active_learning_batch,
        )
        print(json.dumps({"status": report["status"], "row_count": report["row_count"], "evidence_cache_hash": report["evidence_cache_hash"]}, sort_keys=True))
        return 0
    if args.command == "train-sparse":
        classifier_manifest = _read_json(args.classifier_manifest)
        evidence_cache_report = _read_json(args.evidence_cache_report) if args.evidence_cache_report else None
        report = train_chunk_sparse_classifier(
            classifier_manifest=classifier_manifest,
            cache_db=args.cache_db,
            artifact_dir=args.artifact_dir,
            evidence_cache_report=evidence_cache_report,
            batch_size=args.batch_size,
            n_features=args.n_features,
            target_recall=args.target_recall,
            random_state=args.random_state,
            alpha=args.alpha,
            curriculum=_read_and_validate_cli_curriculum(
                args,
                classifier_manifest=classifier_manifest,
                evidence_cache_report=evidence_cache_report,
            ),
            max_train_batches=args.max_train_batches,
        )
        print(json.dumps({"status": report["status"], "model_id": report.get("model_id"), "report_hash": report.get("report_hash")}, sort_keys=True))
        return 0
    if args.command == "train-window":
        classifier_manifest = _read_json(args.classifier_manifest)
        evidence_cache_report = _read_json(args.evidence_cache_report) if args.evidence_cache_report else None
        report = train_chunk_window_mil_classifier(
            classifier_manifest=classifier_manifest,
            cache_db=args.cache_db,
            artifact_dir=args.artifact_dir,
            evidence_cache_report=evidence_cache_report,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            stride=args.stride,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            target_recall=args.target_recall,
            curriculum=_read_and_validate_cli_curriculum(
                args,
                classifier_manifest=classifier_manifest,
                evidence_cache_report=evidence_cache_report,
            ),
            input_mode=args.input_mode,
            pooling_mode=args.pooling_mode,
            pooling_top_k=args.pooling_top_k,
            halo_weight=args.halo_weight,
            halo_logit_clamp=args.halo_logit_clamp,
            device=args.device,
            require_cuda=args.require_cuda,
            amp=not args.disable_amp,
            max_windows_per_device_batch=args.max_windows_per_device_batch,
            document_microbatch_size=args.document_microbatch_size,
            checkpoint_every_batches=args.checkpoint_every_batches,
            max_train_batches=args.max_train_batches,
            max_eval_batches=args.max_eval_batches,
        )
        print(json.dumps({"status": report["status"], "model_id": report.get("model_id"), "report_hash": report.get("report_hash")}, sort_keys=True))
        return 0
    if args.command == "validate-window-gpu-canary":
        gate = validate_window_mil_gpu_canary_gate(
            _read_json(args.window_mil_report),
            min_rows_per_second=args.min_rows_per_second,
            max_checkpoint_fraction=args.max_checkpoint_fraction,
            min_trained_rows=args.min_trained_rows,
            require_cuda=not args.allow_cpu,
            require_amp=not args.allow_amp_disabled,
            require_full_eval=not args.allow_eval_truncation,
        )
        if args.output:
            _write_public_json(Path(args.output), gate)
        print(json.dumps({"status": gate["status"], "gate_hash": gate["gate_hash"]}, sort_keys=True))
        return 0
    if args.command == "build-variant-registry":
        report = build_chunk_variant_registry(
            classifier_manifest=_read_json(args.classifier_manifest),
            evidence_cache_report=_read_json(args.evidence_cache_report),
            output_json=args.output,
            artifact_root=args.artifact_root,
            target_recall=args.target_recall,
            random_state=args.random_state,
        )
        print(json.dumps({"status": report["status"], "registry_hash": report["registry_hash"]}, sort_keys=True))
        return 0
    if args.command == "build-train-score-ledger":
        report = build_chunk_training_score_ledger(
            classifier_manifest=_read_json(args.classifier_manifest),
            evidence_cache_report=_read_json(args.evidence_cache_report),
            sparse_report=_read_json(args.sparse_report),
            window_mil_report=_read_json(args.window_mil_report),
            cache_db=args.cache_db,
            sparse_model_path=args.sparse_model_file,
            window_mil_model_path=args.window_mil_model_file,
            output_json=args.output,
            device=args.device,
            require_cuda=args.require_cuda,
            sparse_score_batch_size=args.sparse_score_batch_size,
            window_mil_score_batch_size=args.window_mil_score_batch_size,
        )
        print(json.dumps({"status": report["status"], "row_count": report["row_count"], "score_ledger_hash": report["score_ledger_hash"]}, sort_keys=True))
        return 0
    if args.command == "build-hard-curriculum":
        report = build_chunk_hard_curriculum(
            classifier_manifest=_read_json(args.classifier_manifest),
            evidence_cache_report=_read_json(args.evidence_cache_report),
            training_score_ledger=_read_json(args.train_score_ledger),
            hard_source_bucket_hashes=args.hard_source_bucket_hash,
            output_json=args.output,
        )
        print(json.dumps({"status": report["status"], "row_count": report["row_count"], "curriculum_hash": report["curriculum_hash"]}, sort_keys=True))
        return 0
    if args.command == "validate-hard-curriculum-binding":
        curriculum = _read_json(args.curriculum)
        validate_chunk_hard_curriculum_binding(
            classifier_manifest=_read_json(args.classifier_manifest),
            evidence_cache_report=_read_json(args.evidence_cache_report),
            training_score_ledger=_read_json(args.train_score_ledger),
            curriculum=curriculum,
        )
        print(json.dumps({"status": "valid", "curriculum_hash": curriculum["curriculum_hash"]}, sort_keys=True))
        return 0
    if args.command == "freeze":
        report = build_chunk_classifier_freeze_report(
            classifier_manifest=_read_json(args.classifier_manifest),
            evidence_cache_report=_read_json(args.evidence_cache_report),
            sparse_report=_read_json(args.sparse_report),
            window_mil_report=_read_json(args.window_mil_report),
            output_json=args.output,
            target_recall=args.target_recall,
        )
        print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}, sort_keys=True))
        return 0
    if args.command == "build-score-ledger":
        from twc.classifiers.chunk_policy_freeze import build_chunk_policy_score_ledger

        report = build_chunk_policy_score_ledger(
            classifier_manifest=_read_json(args.classifier_manifest),
            evidence_cache_report=_read_json(args.evidence_cache_report),
            sparse_report=_read_json(args.sparse_report),
            window_mil_report=_read_json(args.window_mil_report),
            cache_db=args.cache_db,
            sparse_model_path=args.sparse_model_file,
            window_mil_model_path=args.window_mil_model_file,
            output_json=args.output,
            device=args.device,
            require_cuda=args.require_cuda,
            window_mil_precision=args.window_mil_precision,
        )
        print(json.dumps({"status": report["status"], "row_count": report["row_count"], "score_ledger_hash": report["score_ledger_hash"]}, sort_keys=True))
        return 0
    if args.command == "build-deterministic-ledger":
        from twc.classifiers.chunk_policy_freeze import build_chunk_deterministic_ledger
        from twc.data.risk_ontology import load_ontology
        from twc.pretraining.deterministic_filter import build_deterministic_filter_policy_payload

        ontology = load_ontology(args.deterministic_ontology)
        policy_payload = (
            _read_json(args.deterministic_policy_payload)
            if args.deterministic_policy_payload
            else build_deterministic_filter_policy_payload(ontology)
        )
        report = build_chunk_deterministic_ledger(
            classifier_manifest=_read_json(args.classifier_manifest),
            evidence_cache_report=_read_json(args.evidence_cache_report),
            cache_db=args.cache_db,
            deterministic_policy_payload=policy_payload,
            ontology=ontology,
            output_json=args.output,
        )
        print(json.dumps({"status": report["status"], "row_count": report["row_count"], "deterministic_ledger_hash": report["deterministic_ledger_hash"]}, sort_keys=True))
        return 0
    if args.command == "sweep-policy":
        from twc.classifiers.chunk_policy_freeze import build_chunk_policy_sweep_report

        report = build_chunk_policy_sweep_report(
            score_ledger=_read_json(args.score_ledger),
            deterministic_ledger=_read_json(args.deterministic_ledger),
            output_json=args.output,
            target_recall=args.target_recall,
            max_reported_operating_points=args.max_reported_operating_points,
            fixed_sparse_threshold=args.fixed_sparse_threshold,
            fixed_window_mil_threshold=args.fixed_window_mil_threshold,
        )
        print(json.dumps({"status": report["status"], "selected_policy_hash": report.get("selected_policy_hash"), "valid_policy_count": report["valid_policy_count"]}, sort_keys=True))
        return 0
    if args.command == "simulate-clean-runs":
        from twc.classifiers.chunk_policy_freeze import build_chunk_clean_run_simulation_report

        sweep_report = _read_json(args.policy_sweep)
        report = build_chunk_clean_run_simulation_report(
            score_ledger=_read_json(args.score_ledger),
            deterministic_ledger=_read_json(args.deterministic_ledger),
            selected_policy=sweep_report["selected_policy"],
            split=args.split,
            output_json=args.output,
        )
        print(json.dumps({"status": report["status"], "split": report["split"], "clean_run_count": report["clean_run_count"], "gap_event_count": report["gap_event_count"]}, sort_keys=True))
        return 0
    if args.command == "freeze-eval":
        from twc.classifiers.chunk_policy_freeze import build_chunk_policy_eval_freeze_report

        report = build_chunk_policy_eval_freeze_report(
            score_ledger=_read_json(args.score_ledger),
            deterministic_ledger=_read_json(args.deterministic_ledger),
            policy_sweep_report=_read_json(args.policy_sweep),
            output_json=args.output,
            target_recall=args.target_recall,
        )
        print(json.dumps({"status": report["status"], "report_hash": report["report_hash"], "selected_policy_hash": report["selected_policy_hash"]}, sort_keys=True))
        return 0
    if args.command == "validate-report":
        payload = _read_json(args.input)
        if args.kind == "cache":
            validate_chunk_evidence_cache_report(payload)
        elif args.kind == "sparse":
            validate_chunk_sparse_report(payload)
        elif args.kind == "window":
            validate_chunk_window_mil_report(payload)
        elif args.kind == "freeze":
            validate_chunk_classifier_freeze_report(payload)
        elif args.kind == "variant-registry":
            validate_chunk_variant_registry(payload)
        elif args.kind == "train-score-ledger":
            validate_chunk_training_score_ledger(payload)
        elif args.kind == "hard-curriculum":
            validate_chunk_hard_curriculum(payload)
        else:
            from twc.classifiers.chunk_policy_freeze import (
                validate_chunk_clean_run_simulation_report,
                validate_chunk_deterministic_ledger,
                validate_chunk_policy_eval_freeze_report,
                validate_chunk_policy_score_ledger,
                validate_chunk_policy_sweep_report,
            )

            if args.kind == "score-ledger":
                validate_chunk_policy_score_ledger(payload)
            elif args.kind == "deterministic-ledger":
                validate_chunk_deterministic_ledger(payload)
            elif args.kind == "policy-sweep":
                validate_chunk_policy_sweep_report(payload)
            elif args.kind == "clean-run-simulation":
                validate_chunk_clean_run_simulation_report(payload)
            elif args.kind == "policy-freeze":
                validate_chunk_policy_eval_freeze_report(payload)
            else:
                raise AssertionError(args.kind)
        print(json.dumps({"status": "valid"}, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
