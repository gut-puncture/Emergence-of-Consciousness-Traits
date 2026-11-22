"""Combined chunk-filter policy search and eval-only freeze reports.

This module owns the row-level, no-raw artifact spine that binds the v2 chunk
classifier scores, deterministic filter outputs, calibration-only policy
selection, clean-run simulation, and eval-only freeze reporting.
"""

from __future__ import annotations

import json
import math
import heapq
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twc.classifiers.chunk_training import (
    CHUNK_SPARSE_MODEL_ID,
    CHUNK_WINDOW_MIL_MODEL_ID,
    DEFAULT_TARGET_RECALL,
    load_cache_backed_training_view,
    validate_chunk_evidence_cache_report,
    validate_chunk_sparse_report,
    validate_chunk_window_mil_report,
)
from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.classifiers.precision_policy import (
    W1_PRECISION_FP32,
    build_w1_precision_policy,
    normalize_w1_precision_mode,
    validate_w1_precision_policy,
)
from twc.data.decision_schema import stable_json_hash
from twc.data.risk_ontology import RiskOntology
from twc.pretraining.deterministic_filter import (
    DETERMINISTIC_FILTER_POLICY_VERSION,
    evaluate_chunk_candidate_deterministic_filter,
    validate_deterministic_filter_policy_payload,
)


CHUNK_SCORE_LEDGER_KIND = "twc_chunk_score_ledger_v1"
CHUNK_DETERMINISTIC_LEDGER_KIND = "twc_chunk_deterministic_ledger_v1"
CHUNK_POLICY_SWEEP_KIND = "twc_chunk_policy_sweep_v1"
CHUNK_CLEAN_RUN_SIMULATION_KIND = "twc_chunk_clean_run_simulation_v1"
CHUNK_POLICY_EVAL_FREEZE_KIND = "twc_chunk_classifier_freeze_v2_eval_report"
COMBINED_POLICY_FAMILY = "deterministic_terminal_or_sparse_or_window_mil_veto"
DEFAULT_TOKEN_BYTES_PER_TOKEN_LOWER_BOUND = 8
DEFAULT_MAX_REPORTED_OPERATING_POINTS = 128


def build_chunk_policy_score_ledger(
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
    window_mil_precision: str = W1_PRECISION_FP32,
) -> dict[str, Any]:
    """Build a no-raw row-level score ledger for calibration/eval rows."""

    validate_chunk_evidence_cache_report(evidence_cache_report)
    validate_chunk_sparse_report(sparse_report)
    validate_chunk_window_mil_report(window_mil_report)
    _validate_report_bindings(
        classifier_manifest=classifier_manifest,
        evidence_cache_report=evidence_cache_report,
        sparse_report=sparse_report,
        window_mil_report=window_mil_report,
    )
    view = load_cache_backed_training_view(
        cache_db,
        classifier_manifest=classifier_manifest,
        evidence_cache_report=evidence_cache_report,
        context="chunk policy score ledger",
    )
    manifest_by_candidate = _manifest_rows_by_candidate(classifier_manifest)
    rows = [
        row
        for row in sorted(view["fit_rows"], key=lambda item: str(item["candidate_id"]))
        if str(row["split"]) in {"calibration", "eval"}
    ]
    if not rows:
        raise ValueError("score ledger requires calibration/eval rows")
    precision_policy = build_w1_precision_policy(window_mil_precision)
    sparse_scores = _resolve_model_scores(
        rows,
        model_id=CHUNK_SPARSE_MODEL_ID,
        supplied_scores=sparse_scores_by_candidate_id,
        sparse_report=sparse_report,
        window_mil_report=window_mil_report,
        sparse_model_path=sparse_model_path,
        window_mil_model_path=window_mil_model_path,
        device=device,
        require_cuda=require_cuda,
        window_mil_precision=W1_PRECISION_FP32,
    )
    window_scores = _resolve_model_scores(
        rows,
        model_id=CHUNK_WINDOW_MIL_MODEL_ID,
        supplied_scores=window_mil_scores_by_candidate_id,
        sparse_report=sparse_report,
        window_mil_report=window_mil_report,
        sparse_model_path=sparse_model_path,
        window_mil_model_path=window_mil_model_path,
        device=device,
        require_cuda=require_cuda,
        window_mil_precision=str(precision_policy["precision_mode"]),
    )
    ledger_rows: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        manifest_row = manifest_by_candidate[candidate_id]
        ledger_row = _score_ledger_row(
            row,
            manifest_row=manifest_row,
            sparse_score=sparse_scores[candidate_id],
            window_mil_score=window_scores[candidate_id],
            sparse_report=sparse_report,
            window_mil_report=window_mil_report,
        )
        ledger_rows.append(ledger_row)
    ledger_rows = sorted(ledger_rows, key=lambda item: str(item["candidate_id"]))
    report = {
        "artifact_kind": CHUNK_SCORE_LEDGER_KIND,
        "schema_version": 1,
        "status": "complete",
        "classifier_manifest_hash": str(classifier_manifest["training_manifest_hash"]),
        "evidence_cache_hash": str(evidence_cache_report["evidence_cache_hash"]),
        "evidence_cache_report_hash": str(evidence_cache_report["report_hash"]),
        "dataset_view_hash": str(sparse_report["dataset_view_hash"]),
        "sparse_report_hash": str(sparse_report["report_hash"]),
        "sparse_model_payload_hash": str(sparse_report["model_payload_hash"]),
        "window_mil_report_hash": str(window_mil_report["report_hash"]),
        "window_mil_model_payload_hash": str(window_mil_report["model_payload_hash"]),
        "row_count": len(ledger_rows),
        "split_counts": _count_field(ledger_rows, "split"),
        "label_counts": _count_field(ledger_rows, "classifier_label"),
        "evidence_tier_counts": _count_field(ledger_rows, "evidence_tier"),
        "finite_score_counts": {
            CHUNK_SPARSE_MODEL_ID: sum(1 for row in ledger_rows if row["sparse_score_finite"]),
            CHUNK_WINDOW_MIL_MODEL_ID: sum(1 for row in ledger_rows if row["window_mil_score_finite"]),
        },
        "window_mil_precision_policy": precision_policy,
        "window_mil_precision_policy_hash": precision_policy["precision_policy_hash"],
        "row_identity_universe_hash": stable_json_hash(
            [row["row_identity_hash"] for row in ledger_rows]
        ),
        "score_ledger_hash": stable_json_hash(ledger_rows),
        "rows": ledger_rows,
        **_public_false_flags(),
    }
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_policy_score_ledger(report)
    if output_json is not None:
        _write_public_json(Path(output_json), report)
    return report


def build_chunk_deterministic_ledger(
    *,
    classifier_manifest: Mapping[str, Any],
    evidence_cache_report: Mapping[str, Any],
    cache_db: str | Path,
    deterministic_policy_payload: Mapping[str, Any],
    ontology: RiskOntology,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Build a no-raw deterministic-filter ledger over calibration/eval rows."""

    validate_chunk_evidence_cache_report(evidence_cache_report)
    validate_deterministic_filter_policy_payload(deterministic_policy_payload)
    view = load_cache_backed_training_view(
        cache_db,
        classifier_manifest=classifier_manifest,
        evidence_cache_report=evidence_cache_report,
        context="chunk deterministic ledger",
    )
    manifest_by_candidate = _manifest_rows_by_candidate(classifier_manifest)
    rows = [
        row
        for row in sorted(view["fit_rows"], key=lambda item: str(item["candidate_id"]))
        if str(row["split"]) in {"calibration", "eval"}
    ]
    ledger_rows: list[dict[str, Any]] = []
    for row in rows:
        manifest_row = manifest_by_candidate[str(row["candidate_id"])]
        ledger_rows.append(
            _deterministic_ledger_row(
                row,
                manifest_row=manifest_row,
                deterministic_policy_payload=deterministic_policy_payload,
                ontology=ontology,
            )
        )
    ledger_rows = sorted(ledger_rows, key=lambda item: str(item["candidate_id"]))
    terminal_count = sum(1 for row in ledger_rows if int(row["terminal_hit_count"]) > 0)
    report = {
        "artifact_kind": CHUNK_DETERMINISTIC_LEDGER_KIND,
        "schema_version": 1,
        "status": "complete",
        "policy_version": DETERMINISTIC_FILTER_POLICY_VERSION,
        "deterministic_filter_policy_hash": str(deterministic_policy_payload["deterministic_filter_policy_hash"]),
        "classifier_manifest_hash": str(classifier_manifest["training_manifest_hash"]),
        "evidence_cache_hash": str(evidence_cache_report["evidence_cache_hash"]),
        "evidence_cache_report_hash": str(evidence_cache_report["report_hash"]),
        "ontology_semantic_hash": str(deterministic_policy_payload["ontology_semantic_hash"]),
        "ontology_source_tree_hash": str(deterministic_policy_payload["ontology_source_tree_hash"]),
        "row_count": len(ledger_rows),
        "split_counts": _count_field(ledger_rows, "split"),
        "label_counts": _count_field(ledger_rows, "classifier_label"),
        "deterministic_terminal_reject_count": terminal_count,
        "deterministic_audit_only_candidate_count": sum(1 for row in ledger_rows if int(row["audit_only_hit_count"]) > 0),
        "deterministic_halo_evidence_candidate_count": sum(1 for row in ledger_rows if int(row["halo_evidence_hit_count"]) > 0),
        "deterministic_error_count": 0,
        "row_identity_universe_hash": stable_json_hash(
            [row["row_identity_hash"] for row in ledger_rows]
        ),
        "deterministic_ledger_hash": stable_json_hash(ledger_rows),
        "rows": ledger_rows,
        **_public_false_flags(),
    }
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_deterministic_ledger(report)
    if output_json is not None:
        _write_public_json(Path(output_json), report)
    return report


def build_chunk_policy_sweep_report(
    *,
    score_ledger: Mapping[str, Any],
    deterministic_ledger: Mapping[str, Any],
    output_json: str | Path | None = None,
    target_recall: float = DEFAULT_TARGET_RECALL,
    max_reported_operating_points: int = DEFAULT_MAX_REPORTED_OPERATING_POINTS,
    fixed_sparse_threshold: float | None = None,
    fixed_window_mil_threshold: float | None = None,
) -> dict[str, Any]:
    """Search combined-policy thresholds on calibration only."""

    validate_chunk_policy_score_ledger(score_ledger)
    validate_chunk_deterministic_ledger(deterministic_ledger)
    pairs = _aligned_ledger_pairs(score_ledger, deterministic_ledger)
    calibration_pairs = [pair for pair in pairs if pair[0]["split"] == "calibration"]
    if not calibration_pairs:
        raise ValueError("policy sweep requires calibration rows")
    if not 0 < float(target_recall) <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    precision_binding = _score_ledger_precision_binding(score_ledger)

    sweep = _search_threshold_grid(
        calibration_pairs,
        target_recall=float(target_recall),
        max_reported_operating_points=max_reported_operating_points,
        fixed_sparse_threshold=fixed_sparse_threshold,
        fixed_window_mil_threshold=fixed_window_mil_threshold,
    )
    selected_point = sweep["selected_point"]
    selected_policy: dict[str, Any] | None = None
    selected_policy_hash: str | None = None
    status = "blocked_no_valid_policy"
    if selected_point is not None:
        selected_policy = {
            "policy_family": COMBINED_POLICY_FAMILY,
            "selection_split": "calibration",
            "target_reject_recall": float(target_recall),
            "sparse_threshold": float(selected_point["sparse_threshold"]),
            "window_mil_threshold": float(selected_point["window_mil_threshold"]),
            "threshold_direction": "reject_at_or_above",
            "deterministic_terminal_core_reject_fixed": True,
            "halo_hits_terminal": False,
            "audit_only_hits_terminal": False,
            "missing_or_non_finite_score_action": "block_freeze",
            "classifier_can_approve_corpus": False,
        }
        selected_policy_hash = stable_json_hash(selected_policy)
        status = "selected"

    report = {
        "artifact_kind": CHUNK_POLICY_SWEEP_KIND,
        "schema_version": 1,
        "status": status,
        "classifier_manifest_hash": str(score_ledger["classifier_manifest_hash"]),
        "evidence_cache_hash": str(score_ledger["evidence_cache_hash"]),
        "dataset_view_hash": str(score_ledger["dataset_view_hash"]),
        "score_ledger_hash": str(score_ledger["score_ledger_hash"]),
        "score_ledger_report_hash": str(score_ledger.get("report_hash") or ""),
        **precision_binding,
        "deterministic_ledger_hash": str(deterministic_ledger["deterministic_ledger_hash"]),
        "deterministic_filter_policy_hash": str(deterministic_ledger["deterministic_filter_policy_hash"]),
        "row_identity_universe_hash": str(score_ledger["row_identity_universe_hash"]),
        "selection_split": "calibration",
        "eval_split_used_for_policy_selection": False,
        "target_reject_recall": float(target_recall),
        "policy_family": COMBINED_POLICY_FAMILY,
        "threshold_grid": {
            "sparse_threshold_count": int(sweep["sparse_threshold_count"]),
            "window_mil_threshold_count": int(sweep["window_mil_threshold_count"]),
            "fixed_sparse_threshold": None if fixed_sparse_threshold is None else float(fixed_sparse_threshold),
            "fixed_window_mil_threshold": (
                None if fixed_window_mil_threshold is None else float(fixed_window_mil_threshold)
            ),
            "exact_unique_score_grid": fixed_sparse_threshold is None and fixed_window_mil_threshold is None,
        },
        "searched_policy_count": int(sweep["searched_policy_count"]),
        "valid_policy_count": int(sweep["valid_policy_count"]),
        "rejected_policy_count": int(sweep["searched_policy_count"]) - int(sweep["valid_policy_count"]),
        "selected_policy": selected_policy,
        "selected_policy_hash": selected_policy_hash,
        "selected_calibration_metrics": None if selected_point is None else selected_point["metrics"],
        "calibration_operating_table": _format_reported_operating_points(sweep["reported_points"]),
        **_public_false_flags(),
    }
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_policy_sweep_report(report)
    if output_json is not None:
        _write_public_json(Path(output_json), report)
    return report


def build_chunk_clean_run_simulation_report(
    *,
    score_ledger: Mapping[str, Any],
    deterministic_ledger: Mapping[str, Any],
    selected_policy: Mapping[str, Any],
    split: str,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Simulate gap-aware clean-run survival from no-raw candidate ledgers."""

    validate_chunk_policy_score_ledger(score_ledger)
    validate_chunk_deterministic_ledger(deterministic_ledger)
    policy = _validate_selected_policy(selected_policy)
    if split not in {"calibration", "eval"}:
        raise ValueError("clean-run simulation split must be calibration or eval")
    precision_binding = _score_ledger_precision_binding(score_ledger)
    pairs = [pair for pair in _aligned_ledger_pairs(score_ledger, deterministic_ledger) if pair[0]["split"] == split]
    decisions = [_combined_policy_decision(score_row, det_row, policy) for score_row, det_row in pairs]
    simulation = _clean_run_simulation_from_decisions(pairs, decisions)
    report = {
        "artifact_kind": CHUNK_CLEAN_RUN_SIMULATION_KIND,
        "schema_version": 1,
        "status": "complete",
        "split": split,
        "policy_family": COMBINED_POLICY_FAMILY,
        "selected_policy_hash": stable_json_hash(policy),
        "score_ledger_hash": str(score_ledger["score_ledger_hash"]),
        "score_ledger_report_hash": str(score_ledger.get("report_hash") or ""),
        **precision_binding,
        "deterministic_ledger_hash": str(deterministic_ledger["deterministic_ledger_hash"]),
        "row_identity_universe_hash": str(score_ledger["row_identity_universe_hash"]),
        "halos_are_classifier_evidence_only": True,
        **simulation,
        **_public_false_flags(),
    }
    report["simulation_hash"] = _hash_without_many(report, {"simulation_hash", "report_hash"})
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_clean_run_simulation_report(report)
    if output_json is not None:
        _write_public_json(Path(output_json), report)
    return report


def build_chunk_policy_eval_freeze_report(
    *,
    score_ledger: Mapping[str, Any],
    deterministic_ledger: Mapping[str, Any],
    policy_sweep_report: Mapping[str, Any],
    output_json: str | Path | None = None,
    target_recall: float = DEFAULT_TARGET_RECALL,
) -> dict[str, Any]:
    """Apply the selected calibration policy to eval once and freeze evidence."""

    validate_chunk_policy_score_ledger(score_ledger)
    validate_chunk_deterministic_ledger(deterministic_ledger)
    validate_chunk_policy_sweep_report(policy_sweep_report)
    if policy_sweep_report.get("status") != "selected":
        raise ValueError("eval freeze requires a selected calibration policy")
    if policy_sweep_report.get("selection_split") != "calibration":
        raise ValueError("eval freeze policy must be selected on calibration")
    if policy_sweep_report.get("eval_split_used_for_policy_selection") is not False:
        raise ValueError("eval split must not select policy")
    _validate_sweep_bindings(score_ledger, deterministic_ledger, policy_sweep_report)
    precision_binding = _score_ledger_precision_binding(score_ledger)
    policy = _validate_selected_policy(policy_sweep_report["selected_policy"])
    pairs = _aligned_ledger_pairs(score_ledger, deterministic_ledger)
    eval_pairs = [pair for pair in pairs if pair[0]["split"] == "eval"]
    if not eval_pairs:
        raise ValueError("eval freeze requires eval rows")
    eval_decisions = [_combined_policy_decision(score_row, det_row, policy) for score_row, det_row in eval_pairs]
    combined_metrics = _decision_metrics(eval_pairs, eval_decisions)
    simulation = build_chunk_clean_run_simulation_report(
        score_ledger=score_ledger,
        deterministic_ledger=deterministic_ledger,
        selected_policy=policy,
        split="eval",
    )
    gates = {
        "eval_reject_recall": float(combined_metrics["recall"]) >= float(target_recall),
        "eval_false_negative_ceiling": int(combined_metrics["false_negative"]) <= _allowed_false_negatives(
            int(combined_metrics["positive_count"]),
            float(target_recall),
        ),
        "policy_selected_on_calibration": True,
        "eval_not_used_for_policy_selection": True,
        "ledger_identity_alignment": True,
        "deterministic_error_count_zero": int(deterministic_ledger["deterministic_error_count"]) == 0,
        "missing_or_non_finite_scores_zero": _score_ledger_missing_or_nonfinite_count(score_ledger) == 0,
        "no_raw_public_reports": True,
        "classifier_cannot_approve_corpus": True,
    }
    status = "ready" if all(gates.values()) else "blocked_metrics"
    report = {
        "artifact_kind": CHUNK_POLICY_EVAL_FREEZE_KIND,
        "schema_version": 1,
        "status": status,
        "classifier_manifest_hash": str(score_ledger["classifier_manifest_hash"]),
        "evidence_cache_hash": str(score_ledger["evidence_cache_hash"]),
        "dataset_view_hash": str(score_ledger["dataset_view_hash"]),
        "score_ledger_hash": str(score_ledger["score_ledger_hash"]),
        "score_ledger_report_hash": str(score_ledger.get("report_hash") or ""),
        **precision_binding,
        "deterministic_ledger_hash": str(deterministic_ledger["deterministic_ledger_hash"]),
        "deterministic_filter_policy_hash": str(deterministic_ledger["deterministic_filter_policy_hash"]),
        "policy_sweep_report_hash": str(policy_sweep_report["report_hash"]),
        "selected_policy_hash": str(policy_sweep_report["selected_policy_hash"]),
        "selected_policy": policy,
        "policy_selected_on_split": "calibration",
        "eval_policy_selection_count": 0,
        "target_reject_recall": float(target_recall),
        "metric_gates": gates,
        "eval_metrics": {
            "combined_policy": combined_metrics,
            "disagreement_matrix": _disagreement_matrix(eval_pairs, eval_decisions),
        },
        "clean_run_simulation": {
            key: simulation[key]
            for key in (
                "clean_run_count",
                "gap_event_count",
                "accepted_core_token_lower_bound",
                "clean_run_token_lower_bound_distribution",
                "document_count",
                "documents_with_any_clean_run_count",
                "halos_are_classifier_evidence_only",
            )
        },
        "classifier_can_approve_corpus": False,
        "full_filtering_still_requires_1000_row_canary": True,
        **_public_false_flags(),
    }
    report["report_hash"] = _hash_without(report, "report_hash")
    validate_chunk_policy_eval_freeze_report(report)
    if output_json is not None:
        _write_public_json(Path(output_json), report)
    return report


def validate_chunk_policy_score_ledger(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_SCORE_LEDGER_KIND:
        raise ValueError("unsupported chunk score ledger kind")
    if report.get("status") != "complete":
        raise ValueError("chunk score ledger must be complete")
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
        _require_sha256(report.get(field), field)
    rows = _require_rows(report)
    if int(report.get("row_count", -1)) != len(rows):
        raise ValueError("score ledger row_count mismatch")
    _require_sorted_unique_candidates(rows, "score ledger")
    for row in rows:
        _validate_score_ledger_row(row)
    if report["score_ledger_hash"] != stable_json_hash(rows):
        raise ValueError("score ledger hash mismatch")
    if report["row_identity_universe_hash"] != stable_json_hash([row["row_identity_hash"] for row in rows]):
        raise ValueError("score ledger identity universe hash mismatch")
    _score_ledger_precision_binding(report)
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("score ledger report hash mismatch")


def _score_ledger_precision_binding(score_ledger: Mapping[str, Any]) -> dict[str, Any]:
    policy = score_ledger.get("window_mil_precision_policy")
    if policy is None:
        return {
            "window_mil_precision_policy": None,
            "window_mil_precision_policy_hash": None,
            "window_mil_precision_legacy_default": True,
        }
    if not isinstance(policy, Mapping):
        raise ValueError("score ledger W1 precision policy must be an object")
    validate_w1_precision_policy(policy)
    policy_hash = str(policy["precision_policy_hash"])
    if score_ledger.get("window_mil_precision_policy_hash") != policy_hash:
        raise ValueError("score ledger W1 precision policy hash mismatch")
    return {
        "window_mil_precision_policy": dict(policy),
        "window_mil_precision_policy_hash": policy_hash,
        "window_mil_precision_legacy_default": False,
    }


def _validate_embedded_precision_binding(report: Mapping[str, Any]) -> None:
    policy = report.get("window_mil_precision_policy")
    if policy is None:
        if report.get("window_mil_precision_policy_hash") is not None:
            raise ValueError("embedded W1 precision policy hash without policy")
        if report.get("window_mil_precision_legacy_default") not in {None, True}:
            raise ValueError("embedded W1 precision legacy marker mismatch")
        return
    if not isinstance(policy, Mapping):
        raise ValueError("embedded W1 precision policy must be an object")
    validate_w1_precision_policy(policy)
    if report.get("window_mil_precision_policy_hash") != policy["precision_policy_hash"]:
        raise ValueError("embedded W1 precision policy hash mismatch")


def validate_chunk_deterministic_ledger(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_DETERMINISTIC_LEDGER_KIND:
        raise ValueError("unsupported chunk deterministic ledger kind")
    if report.get("status") != "complete":
        raise ValueError("chunk deterministic ledger must be complete")
    for field in (
        "deterministic_filter_policy_hash",
        "classifier_manifest_hash",
        "evidence_cache_hash",
        "evidence_cache_report_hash",
        "ontology_semantic_hash",
        "ontology_source_tree_hash",
        "row_identity_universe_hash",
        "deterministic_ledger_hash",
    ):
        _require_sha256(report.get(field), field)
    rows = _require_rows(report)
    if int(report.get("row_count", -1)) != len(rows):
        raise ValueError("deterministic ledger row_count mismatch")
    _require_sorted_unique_candidates(rows, "deterministic ledger")
    policy_hash = str(report["deterministic_filter_policy_hash"])
    for row in rows:
        _validate_deterministic_ledger_row(row, deterministic_policy_hash=policy_hash)
    if report["deterministic_ledger_hash"] != stable_json_hash(rows):
        raise ValueError("deterministic ledger hash mismatch")
    if report["row_identity_universe_hash"] != stable_json_hash([row["row_identity_hash"] for row in rows]):
        raise ValueError("deterministic ledger identity universe hash mismatch")
    if int(report.get("deterministic_error_count", -1)) != 0:
        raise ValueError("deterministic ledger error count must be zero for freeze")
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("deterministic ledger report hash mismatch")


def validate_chunk_policy_sweep_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_POLICY_SWEEP_KIND:
        raise ValueError("unsupported chunk policy sweep kind")
    if report.get("status") not in {"selected", "blocked_no_valid_policy"}:
        raise ValueError("policy sweep status invalid")
    for field in (
        "classifier_manifest_hash",
        "evidence_cache_hash",
        "dataset_view_hash",
        "score_ledger_hash",
        "deterministic_ledger_hash",
        "deterministic_filter_policy_hash",
        "row_identity_universe_hash",
    ):
        _require_sha256(report.get(field), field)
    if report.get("selection_split") != "calibration":
        raise ValueError("policy sweep must select on calibration")
    if report.get("eval_split_used_for_policy_selection") is not False:
        raise ValueError("policy sweep must not use eval split for calibration-only selection")
    if report.get("policy_family") != COMBINED_POLICY_FAMILY:
        raise ValueError("policy sweep family mismatch")
    _validate_embedded_precision_binding(report)
    if report.get("status") == "selected":
        policy = _validate_selected_policy(report.get("selected_policy"))
        if report.get("selected_policy_hash") != stable_json_hash(policy):
            raise ValueError("selected policy hash mismatch")
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("policy sweep report hash mismatch")


def validate_chunk_clean_run_simulation_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_CLEAN_RUN_SIMULATION_KIND:
        raise ValueError("unsupported clean-run simulation kind")
    if report.get("status") != "complete":
        raise ValueError("clean-run simulation must be complete")
    for field in ("selected_policy_hash", "score_ledger_hash", "deterministic_ledger_hash", "row_identity_universe_hash"):
        _require_sha256(report.get(field), field)
    if report.get("split") not in {"calibration", "eval"}:
        raise ValueError("clean-run simulation split invalid")
    if report.get("halos_are_classifier_evidence_only") is not True:
        raise ValueError("clean-run simulation must keep halos evidence-only")
    _validate_embedded_precision_binding(report)
    for field in ("safe_chunk_count", "gap_event_count", "clean_run_count", "accepted_core_token_lower_bound"):
        if int(report.get(field, -1)) < 0:
            raise ValueError(f"clean-run simulation {field} invalid")
    if report.get("simulation_hash") != _hash_without_many(report, {"simulation_hash", "report_hash"}):
        raise ValueError("clean-run simulation hash mismatch")
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("clean-run simulation report hash mismatch")


def validate_chunk_policy_eval_freeze_report(report: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_POLICY_EVAL_FREEZE_KIND:
        raise ValueError("unsupported chunk policy eval freeze kind")
    if report.get("status") not in {"ready", "blocked_metrics"}:
        raise ValueError("eval freeze status invalid")
    for field in (
        "classifier_manifest_hash",
        "evidence_cache_hash",
        "dataset_view_hash",
        "score_ledger_hash",
        "deterministic_ledger_hash",
        "deterministic_filter_policy_hash",
        "policy_sweep_report_hash",
        "selected_policy_hash",
    ):
        _require_sha256(report.get(field), field)
    if report.get("policy_selected_on_split") != "calibration":
        raise ValueError("eval freeze policy must be selected on calibration")
    if int(report.get("eval_policy_selection_count", -1)) != 0:
        raise ValueError("eval freeze must not select policy on eval")
    if report.get("classifier_can_approve_corpus") is not False:
        raise ValueError("classifier must not approve corpus")
    _validate_embedded_precision_binding(report)
    _validate_selected_policy(report.get("selected_policy"))
    metrics = report.get("eval_metrics", {}).get("combined_policy")
    if not isinstance(metrics, Mapping):
        raise ValueError("eval freeze missing combined policy metrics")
    _validate_decision_metrics(metrics)
    gates = report.get("metric_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("eval freeze requires metric gates")
    if report.get("status") == "ready" and not all(bool(value) for value in gates.values()):
        raise ValueError("eval freeze ready status has failed gates")
    expected_status = "ready" if all(bool(value) for value in gates.values()) else "blocked_metrics"
    if report.get("status") != expected_status:
        raise ValueError("eval freeze status does not match metric gates")
    if report.get("report_hash") != _hash_without(report, "report_hash"):
        raise ValueError("eval freeze report hash mismatch")


def _validate_report_bindings(
    *,
    classifier_manifest: Mapping[str, Any],
    evidence_cache_report: Mapping[str, Any],
    sparse_report: Mapping[str, Any],
    window_mil_report: Mapping[str, Any],
) -> None:
    manifest_hash = str(classifier_manifest["training_manifest_hash"])
    cache_hash = str(evidence_cache_report["evidence_cache_hash"])
    for label, report in (("sparse", sparse_report), ("window-MIL", window_mil_report)):
        if str(report.get("classifier_manifest_hash")) != manifest_hash:
            raise ValueError(f"{label} classifier manifest hash mismatch")
        if str(report.get("evidence_cache_hash")) != cache_hash:
            raise ValueError(f"{label} evidence cache hash mismatch")
        if report.get("threshold_selection", {}).get("eval_split_used_for_threshold_selection") is not False:
            raise ValueError(f"{label} eval split was used for threshold selection")
    if str(sparse_report.get("dataset_view_hash")) != str(window_mil_report.get("dataset_view_hash")):
        raise ValueError("sparse/window dataset view hash mismatch")


def _resolve_model_scores(
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
    window_mil_precision: str,
) -> dict[str, float]:
    if supplied_scores is not None:
        scores: dict[str, float] = {}
        for row in rows:
            candidate_id = str(row["candidate_id"])
            if candidate_id not in supplied_scores:
                raise ValueError(f"{model_id} supplied scores missing candidate")
            scores[candidate_id] = _score01(supplied_scores[candidate_id], f"{model_id} score")
        return scores
    texts = [str(row["evidence_text"]) for row in rows]
    metadata = [_public_runtime_metadata(row) for row in rows]
    if model_id == CHUNK_SPARSE_MODEL_ID:
        if sparse_model_path is None:
            raise ValueError("sparse model path required when sparse scores are not supplied")
        from twc.pretraining.chunk_filtering import ChunkSparseRiskScorer

        scorer = ChunkSparseRiskScorer.from_report(sparse_report, sparse_model_path)
    elif model_id == CHUNK_WINDOW_MIL_MODEL_ID:
        if window_mil_model_path is None:
            raise ValueError("Window-MIL model path required when Window-MIL scores are not supplied")
        from twc.pretraining.chunk_filtering import ChunkWindowMilRiskScorer

        scorer = ChunkWindowMilRiskScorer.from_report(
            window_mil_report,
            window_mil_model_path,
            device=device,
            require_cuda=require_cuda,
            window_mil_precision=normalize_w1_precision_mode(window_mil_precision),
        )
    else:
        raise ValueError(f"unsupported model id: {model_id}")
    raw_scores = scorer.score_many(texts, metadata)
    if len(raw_scores) != len(rows):
        raise ValueError(f"{model_id} score count mismatch")
    return {
        str(row["candidate_id"]): _score01(score, f"{model_id} score")
        for row, score in zip(rows, raw_scores, strict=True)
    }


def _score_ledger_row(
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
    payload = {
        **_shared_public_row_fields(row, manifest_row),
        "sparse_model_id": CHUNK_SPARSE_MODEL_ID,
        "sparse_report_hash": str(sparse_report["report_hash"]),
        "sparse_model_payload_hash": str(sparse_report["model_payload_hash"]),
        "sparse_score": sparse,
        "sparse_score_finite": True,
        "window_mil_model_id": CHUNK_WINDOW_MIL_MODEL_ID,
        "window_mil_report_hash": str(window_mil_report["report_hash"]),
        "window_mil_model_payload_hash": str(window_mil_report["model_payload_hash"]),
        "window_mil_score": window,
        "window_mil_score_finite": True,
    }
    payload["score_row_hash"] = _hash_without(payload, "score_row_hash")
    validate_public_no_raw_payload(payload)
    return payload


def _deterministic_ledger_row(
    row: Mapping[str, Any],
    *,
    manifest_row: Mapping[str, Any],
    deterministic_policy_payload: Mapping[str, Any],
    ontology: RiskOntology,
) -> dict[str, Any]:
    halo_left = int(row["halo_left_chars"])
    core_length = int(row["core_char_length"])
    evidence_text = str(row["evidence_text"])
    core_text = str(row["core_text"])
    if evidence_text[halo_left : halo_left + core_length] != core_text:
        raise ValueError("private evidence cache core/evidence span drift")
    candidate = {
        "candidate_id_hash": str(row["candidate_id"]),
        "core_char_start": halo_left,
        "core_char_end": halo_left + core_length,
        "evidence_char_start": 0,
        "evidence_char_end": int(row["evidence_char_length"]),
    }
    result = evaluate_chunk_candidate_deterministic_filter(
        candidate,
        raw_text=evidence_text,
        ontology=ontology,
        policy_payload=deterministic_policy_payload,
    )
    payload = {
        **_shared_public_row_fields(row, manifest_row),
        "policy_version": DETERMINISTIC_FILTER_POLICY_VERSION,
        "deterministic_filter_policy_hash": str(result["deterministic_filter_policy_hash"]),
        "deterministic_filter_result_hash": str(result["deterministic_filter_result_hash"]),
        "terminal_decision": result.get("terminal_decision"),
        "terminal_hit_count": int(result["terminal_hit_count"]),
        "audit_only_hit_count": int(result["audit_only_hit_count"]),
        "halo_evidence_hit_count": int(result["halo_evidence_hit_count"]),
        "terminal_rule_ids": list(result["terminal_rule_ids"]),
        "audit_only_rule_ids": list(result["audit_only_rule_ids"]),
        "halo_evidence_rule_ids": list(result["halo_evidence_rule_ids"]),
        "deterministic_result_status": "ok",
    }
    payload["deterministic_row_hash"] = _hash_without(payload, "deterministic_row_hash")
    validate_public_no_raw_payload(payload)
    return payload


def _shared_public_row_fields(row: Mapping[str, Any], manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "candidate_id": str(row["candidate_id"]),
        "candidate_hash": str(row["candidate_hash"]),
        "row_hash": str(row["row_hash"]),
        "classifier_label": str(row["classifier_label"]),
        "split": str(row["split"]),
        "evidence_tier": str(row["evidence_tier"]),
        "core_text_sha256": str(row["core_text_sha256"]),
        "evidence_text_sha256": str(row["evidence_text_sha256"]),
    }
    identity["row_identity_hash"] = stable_json_hash(identity)
    source_document_identity_hash = str(
        manifest_row.get("source_document_identity_hash")
        or manifest_row.get("source_doc_identity_hash")
        or stable_json_hash({"parent_candidate_id": str(manifest_row.get("parent_candidate_id") or row["candidate_id"])})
    )
    return {
        **identity,
        "source_document_identity_hash": _require_sha256(source_document_identity_hash, "source_document_identity_hash"),
        "source_chunk_manifest_hash": _optional_sha256(
            manifest_row.get("source_chunk_manifest_hash"),
            "source_chunk_manifest_hash",
        ),
        "source_bucket_hash": _optional_sha256(manifest_row.get("source_bucket_hash"), "source_bucket_hash"),
        "token_count_bucket": str(manifest_row.get("token_count_bucket") or "unknown"),
        "global_chunk_index": _nonnegative_int(manifest_row.get("global_chunk_index", 0), "global_chunk_index"),
        "parent_chunk_index": _nonnegative_int(manifest_row.get("parent_chunk_index", 0), "parent_chunk_index"),
        "parent_chunk_count": _positive_int(manifest_row.get("parent_chunk_count", 1), "parent_chunk_count"),
        "core_char_length": _nonnegative_int(row["core_char_length"], "core_char_length"),
        "evidence_char_length": _nonnegative_int(row["evidence_char_length"], "evidence_char_length"),
        "halo_left_chars": _nonnegative_int(row["halo_left_chars"], "halo_left_chars"),
        "halo_right_chars": _nonnegative_int(row["halo_right_chars"], "halo_right_chars"),
        "core_token_lower_bound": _core_token_lower_bound(int(row["core_char_length"])),
    }


def _manifest_rows_by_candidate(classifier_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {}
    for row in classifier_manifest.get("rows") or []:
        candidate_id = str(row.get("candidate_id") or "")
        _require_sha256(candidate_id, "candidate_id")
        if candidate_id in rows:
            raise ValueError("classifier manifest duplicate candidate")
        rows[candidate_id] = row
    if not rows:
        raise ValueError("classifier manifest contains no rows")
    return rows


def _validate_score_ledger_row(row: Mapping[str, Any]) -> None:
    _validate_shared_ledger_row(row)
    for field in ("sparse_report_hash", "sparse_model_payload_hash", "window_mil_report_hash", "window_mil_model_payload_hash"):
        _require_sha256(row.get(field), field)
    if row.get("sparse_model_id") != CHUNK_SPARSE_MODEL_ID:
        raise ValueError("score ledger sparse model id mismatch")
    if row.get("window_mil_model_id") != CHUNK_WINDOW_MIL_MODEL_ID:
        raise ValueError("score ledger Window-MIL model id mismatch")
    _score01(row.get("sparse_score"), "sparse_score")
    _score01(row.get("window_mil_score"), "window_mil_score")
    if row.get("sparse_score_finite") is not True or row.get("window_mil_score_finite") is not True:
        raise ValueError("score ledger finite flags must be true")
    if row.get("score_row_hash") != _hash_without(row, "score_row_hash"):
        raise ValueError("score ledger row hash mismatch")


def _validate_deterministic_ledger_row(row: Mapping[str, Any], *, deterministic_policy_hash: str) -> None:
    _validate_shared_ledger_row(row)
    if row.get("policy_version") != DETERMINISTIC_FILTER_POLICY_VERSION:
        raise ValueError("deterministic ledger policy version mismatch")
    if row.get("deterministic_filter_policy_hash") != deterministic_policy_hash:
        raise ValueError("deterministic policy hash mismatch")
    _require_sha256(row.get("deterministic_filter_result_hash"), "deterministic_filter_result_hash")
    if row.get("terminal_decision") not in {None, "reject"}:
        raise ValueError("deterministic ledger terminal decision invalid")
    for field in ("terminal_hit_count", "audit_only_hit_count", "halo_evidence_hit_count"):
        _nonnegative_int(row.get(field), field)
    for field in ("terminal_rule_ids", "audit_only_rule_ids", "halo_evidence_rule_ids"):
        if not isinstance(row.get(field), list) or any(not isinstance(item, str) for item in row[field]):
            raise ValueError(f"deterministic ledger {field} must be a string list")
    if row.get("deterministic_result_status") != "ok":
        raise ValueError("deterministic ledger result status must be ok")
    if row.get("deterministic_row_hash") != _hash_without(row, "deterministic_row_hash"):
        raise ValueError("deterministic ledger row hash mismatch")


def _validate_shared_ledger_row(row: Mapping[str, Any]) -> None:
    for field in ("candidate_id", "candidate_hash", "row_hash", "core_text_sha256", "evidence_text_sha256", "row_identity_hash"):
        _require_sha256(row.get(field), field)
    if row.get("classifier_label") not in {"safe", "reject"}:
        raise ValueError("ledger row classifier_label invalid")
    if row.get("split") not in {"calibration", "eval"}:
        raise ValueError("ledger row split must be calibration/eval")
    if stable_json_hash(
        {
            "candidate_id": row["candidate_id"],
            "candidate_hash": row["candidate_hash"],
            "row_hash": row["row_hash"],
            "classifier_label": row["classifier_label"],
            "split": row["split"],
            "evidence_tier": row["evidence_tier"],
            "core_text_sha256": row["core_text_sha256"],
            "evidence_text_sha256": row["evidence_text_sha256"],
        }
    ) != row["row_identity_hash"]:
        raise ValueError("ledger row identity hash mismatch")
    for field in ("source_document_identity_hash",):
        _require_sha256(row.get(field), field)
    for field in ("source_chunk_manifest_hash", "source_bucket_hash"):
        if row.get(field) is not None:
            _require_sha256(row.get(field), field)
    for field in ("global_chunk_index", "parent_chunk_index", "parent_chunk_count", "core_char_length", "evidence_char_length", "halo_left_chars", "halo_right_chars", "core_token_lower_bound"):
        _nonnegative_int(row.get(field), field)


def _search_threshold_grid(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    target_recall: float,
    max_reported_operating_points: int,
    fixed_sparse_threshold: float | None = None,
    fixed_window_mil_threshold: float | None = None,
) -> dict[str, Any]:
    if max_reported_operating_points < 1:
        raise ValueError("max_reported_operating_points must be positive")
    sparse_thresholds = _fixed_or_unique_thresholds(
        [pair[0]["sparse_score"] for pair in pairs],
        fixed_threshold=fixed_sparse_threshold,
        field="fixed_sparse_threshold",
    )
    window_thresholds = _fixed_or_unique_thresholds(
        [pair[0]["window_mil_score"] for pair in pairs],
        fixed_threshold=fixed_window_mil_threshold,
        field="fixed_window_mil_threshold",
    )
    reject_label_mask = 0
    safe_label_mask = 0
    terminal_mask = 0
    all_mask = 0
    token_weights: list[int] = []
    for index, (score_row, det_row) in enumerate(pairs):
        bit = 1 << index
        all_mask |= bit
        token_weights.append(int(score_row["core_token_lower_bound"]))
        if score_row["classifier_label"] == "reject":
            reject_label_mask |= bit
        else:
            safe_label_mask |= bit
        if int(det_row["terminal_hit_count"]) > 0:
            terminal_mask |= bit
    reject_total = reject_label_mask.bit_count()
    safe_total = safe_label_mask.bit_count()
    if reject_total < 1 or safe_total < 1:
        raise ValueError("policy sweep requires both labels on calibration")
    sparse_masks = [
        (threshold, _score_threshold_mask(pairs, field="sparse_score", threshold=threshold))
        for threshold in sparse_thresholds
    ]
    window_masks = [
        (threshold, _score_threshold_mask(pairs, field="window_mil_score", threshold=threshold))
        for threshold in window_thresholds
    ]
    searched_policy_count = 0
    valid_policy_count = 0
    selected_point: dict[str, Any] | None = None
    reported_heap: list[tuple[tuple[bool, int, int, float], int, dict[str, Any]]] = []
    report_sequence = 0
    for sparse_threshold, sparse_mask in sparse_masks:
        for window_threshold, window_mask in window_masks:
            searched_policy_count += 1
            reject_mask = terminal_mask | sparse_mask | window_mask
            tp = (reject_mask & reject_label_mask).bit_count()
            fn = reject_total - tp
            recall = _safe_div(tp, reject_total)
            if recall < float(target_recall):
                continue
            valid_policy_count += 1
            fp = (reject_mask & safe_label_mask).bit_count()
            tn = safe_total - fp
            accepted_token_lower_bound = _accepted_token_lower_bound_from_weights(
                reject_mask=reject_mask,
                all_mask=all_mask,
                token_weights=token_weights,
            )
            metrics = {
                "count": len(pairs),
                "positive_count": reject_total,
                "negative_count": safe_total,
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": _safe_div(tp, tp + fp),
                "recall": recall,
                "specificity": _safe_div(tn, tn + fp),
                "safe_clearance_count": tn,
                "safe_clearance_rate": _safe_div(tn, safe_total),
                "safe_decision_count": len(pairs) - reject_mask.bit_count(),
                "reject_decision_count": reject_mask.bit_count(),
                "accepted_core_token_lower_bound": accepted_token_lower_bound,
            }
            point = {
                "sparse_threshold": float(sparse_threshold),
                "window_mil_threshold": float(window_threshold),
                "meets_target_recall": True,
                "metrics": metrics,
            }
            if selected_point is None or _selected_operating_point_key(point) > _selected_operating_point_key(selected_point):
                selected_point = point
            report_key = _reported_operating_point_key(point)
            heap_item = (report_key, report_sequence, point)
            report_sequence += 1
            if len(reported_heap) < max_reported_operating_points:
                heapq.heappush(reported_heap, heap_item)
            elif report_key > reported_heap[0][0]:
                heapq.heapreplace(reported_heap, heap_item)
    return {
        "searched_policy_count": searched_policy_count,
        "valid_policy_count": valid_policy_count,
        "selected_point": selected_point,
        "sparse_threshold_count": len(sparse_thresholds),
        "window_mil_threshold_count": len(window_thresholds),
        "reported_points": [
            point
            for _key, _sequence, point in sorted(
                reported_heap,
                key=lambda item: (item[0], item[1]),
                reverse=True,
            )
        ],
    }


def _fixed_or_unique_thresholds(
    scores: Sequence[float],
    *,
    fixed_threshold: float | None,
    field: str,
) -> list[float]:
    if fixed_threshold is None:
        return _unique_thresholds(scores)
    return [_score01(fixed_threshold, field)]


def _selected_operating_point_key(point: Mapping[str, Any]) -> tuple[int, int, int, float, float]:
    return (
        int(point["metrics"]["accepted_core_token_lower_bound"]),
        int(point["metrics"]["safe_decision_count"]),
        int(point["metrics"]["safe_clearance_count"]),
        float(point["metrics"]["recall"]),
        -(float(point["sparse_threshold"]) + float(point["window_mil_threshold"])),
    )


def _reported_operating_point_key(point: Mapping[str, Any]) -> tuple[bool, int, int, float]:
    return (
        bool(point["meets_target_recall"]),
        int(point["metrics"]["accepted_core_token_lower_bound"]),
        int(point["metrics"]["safe_decision_count"]),
        float(point["metrics"]["recall"]),
    )


def _select_operating_point(points: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not points:
        return None
    return dict(max(points, key=_selected_operating_point_key))


def _format_reported_operating_points(
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "sparse_threshold": float(point["sparse_threshold"]),
            "window_mil_threshold": float(point["window_mil_threshold"]),
            "meets_target_recall": bool(point["meets_target_recall"]),
            "recall": float(point["metrics"]["recall"]),
            "false_negative": int(point["metrics"]["false_negative"]),
            "safe_decision_count": int(point["metrics"]["safe_decision_count"]),
            "accepted_core_token_lower_bound": int(point["metrics"]["accepted_core_token_lower_bound"]),
        }
        for point in points
    ]


def _reported_operating_points(
    points: Sequence[Mapping[str, Any]],
    *,
    max_reported_operating_points: int,
) -> list[dict[str, Any]]:
    if max_reported_operating_points < 1:
        raise ValueError("max_reported_operating_points must be positive")
    ordered = sorted(points, key=_reported_operating_point_key, reverse=True)
    return _format_reported_operating_points(ordered[: int(max_reported_operating_points)])


def _aligned_ledger_pairs(
    score_ledger: Mapping[str, Any],
    deterministic_ledger: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    validate_chunk_policy_score_ledger(score_ledger)
    validate_chunk_deterministic_ledger(deterministic_ledger)
    if score_ledger["classifier_manifest_hash"] != deterministic_ledger["classifier_manifest_hash"]:
        raise ValueError("score/deterministic classifier manifest hash mismatch")
    if score_ledger["evidence_cache_hash"] != deterministic_ledger["evidence_cache_hash"]:
        raise ValueError("score/deterministic evidence cache hash mismatch")
    if score_ledger["row_identity_universe_hash"] != deterministic_ledger["row_identity_universe_hash"]:
        raise ValueError("score/deterministic row identity universe hash mismatch")
    score_rows = list(score_ledger["rows"])
    det_rows = list(deterministic_ledger["rows"])
    if len(score_rows) != len(det_rows):
        raise ValueError("score/deterministic row count mismatch")
    pairs = []
    identity_fields = (
        "candidate_id",
        "candidate_hash",
        "row_hash",
        "row_identity_hash",
        "classifier_label",
        "split",
        "evidence_tier",
        "core_text_sha256",
        "evidence_text_sha256",
    )
    for score_row, det_row in zip(score_rows, det_rows, strict=True):
        for field in identity_fields:
            if score_row.get(field) != det_row.get(field):
                raise ValueError("score/deterministic candidate identity mismatch")
        pairs.append((score_row, det_row))
    return pairs


def _combined_policy_decision(
    score_row: Mapping[str, Any],
    det_row: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    sparse_score = _score01(score_row["sparse_score"], "sparse_score")
    window_score = _score01(score_row["window_mil_score"], "window_mil_score")
    sparse_reject = sparse_score >= float(policy["sparse_threshold"])
    window_reject = window_score >= float(policy["window_mil_threshold"])
    deterministic_reject = int(det_row["terminal_hit_count"]) > 0
    if deterministic_reject:
        status = "reject"
        route = "deterministic_terminal"
    elif sparse_reject or window_reject:
        status = "reject"
        if sparse_reject and window_reject:
            route = "sparse_and_window_mil"
        elif sparse_reject:
            route = "sparse_only"
        else:
            route = "window_mil_only"
    else:
        status = "safe"
        route = "all_vetoes_clear"
    return {
        "candidate_id": score_row["candidate_id"],
        "classifier_decision": status,
        "decision_route": route,
        "sparse_reject": sparse_reject,
        "window_mil_reject": window_reject,
        "deterministic_terminal_reject": deterministic_reject,
    }


def _decision_metrics(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    accepted_tokens = 0
    safe_label_accepted_tokens = 0
    for (score_row, _det_row), decision in zip(pairs, decisions, strict=True):
        label_reject = score_row["classifier_label"] == "reject"
        decision_reject = decision["classifier_decision"] != "safe"
        if label_reject and decision_reject:
            tp += 1
        elif label_reject and not decision_reject:
            fn += 1
        elif not label_reject and decision_reject:
            fp += 1
        else:
            tn += 1
        if not decision_reject:
            accepted_tokens += int(score_row["core_token_lower_bound"])
            if not label_reject:
                safe_label_accepted_tokens += int(score_row["core_token_lower_bound"])
    positive_count = tp + fn
    negative_count = tn + fp
    metrics = {
        "count": len(pairs),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, positive_count),
        "specificity": _safe_div(tn, negative_count),
        "safe_clearance_count": tn,
        "safe_clearance_rate": _safe_div(tn, negative_count),
        "safe_decision_count": sum(1 for decision in decisions if decision["classifier_decision"] == "safe"),
        "reject_decision_count": sum(1 for decision in decisions if decision["classifier_decision"] != "safe"),
        "accepted_core_token_lower_bound": accepted_tokens,
        "safe_label_accepted_core_token_lower_bound": safe_label_accepted_tokens,
    }
    _validate_decision_metrics(metrics)
    return metrics


def _validate_decision_metrics(metrics: Mapping[str, Any]) -> None:
    count = int(metrics.get("count", -1))
    positive = int(metrics.get("positive_count", -1))
    negative = int(metrics.get("negative_count", -1))
    tp = int(metrics.get("true_positive", -1))
    fp = int(metrics.get("false_positive", -1))
    tn = int(metrics.get("true_negative", -1))
    fn = int(metrics.get("false_negative", -1))
    if min(count, positive, negative, tp, fp, tn, fn) < 0:
        raise ValueError("decision metrics contain negative count")
    if positive + negative != count:
        raise ValueError("decision metrics label count mismatch")
    if tp + fn != positive:
        raise ValueError("decision metrics false-negative count mismatch")
    if tn + fp != negative:
        raise ValueError("decision metrics false-positive count mismatch")
    for field in ("precision", "recall", "specificity", "safe_clearance_rate"):
        value = float(metrics.get(field, float("nan")))
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"decision metrics {field} invalid")


def _disagreement_matrix(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    del pairs
    counts = Counter(str(decision["decision_route"]) for decision in decisions)
    return {key: int(counts[key]) for key in sorted(counts)}


def _clean_run_simulation_from_decisions(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_doc: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for (score_row, _det_row), decision in zip(pairs, decisions, strict=True):
        by_doc[str(score_row["source_document_identity_hash"])].append((score_row, decision))
    clean_run_lengths: list[int] = []
    safe_chunk_count = 0
    gap_count = 0
    docs_with_clean_run = 0
    for doc_rows in by_doc.values():
        current_tokens = 0
        doc_had_clean_run = False
        for score_row, decision in sorted(doc_rows, key=lambda item: (int(item[0]["global_chunk_index"]), str(item[0]["candidate_id"]))):
            if decision["classifier_decision"] == "safe":
                safe_chunk_count += 1
                current_tokens += int(score_row["core_token_lower_bound"])
                continue
            gap_count += 1
            if current_tokens > 0:
                clean_run_lengths.append(current_tokens)
                doc_had_clean_run = True
                current_tokens = 0
        if current_tokens > 0:
            clean_run_lengths.append(current_tokens)
            doc_had_clean_run = True
        if doc_had_clean_run:
            docs_with_clean_run += 1
    return {
        "document_count": len(by_doc),
        "documents_with_any_clean_run_count": docs_with_clean_run,
        "safe_chunk_count": safe_chunk_count,
        "gap_event_count": gap_count,
        "clean_run_count": len(clean_run_lengths),
        "accepted_core_token_lower_bound": sum(clean_run_lengths),
        "clean_run_token_lower_bound_distribution": _distribution_summary(clean_run_lengths),
    }


def _validate_selected_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, Mapping):
        raise ValueError("selected policy must be an object")
    if policy.get("policy_family") != COMBINED_POLICY_FAMILY:
        raise ValueError("selected policy family mismatch")
    if policy.get("selection_split") != "calibration":
        raise ValueError("selected policy must be from calibration")
    if policy.get("threshold_direction") != "reject_at_or_above":
        raise ValueError("selected policy threshold direction mismatch")
    if policy.get("deterministic_terminal_core_reject_fixed") is not True:
        raise ValueError("selected policy must fix deterministic terminal rejects")
    if policy.get("halo_hits_terminal") is not False:
        raise ValueError("selected policy must keep halo hits nonterminal")
    if policy.get("audit_only_hits_terminal") is not False:
        raise ValueError("selected policy must keep audit hits nonterminal")
    if policy.get("classifier_can_approve_corpus") is not False:
        raise ValueError("selected policy must not approve corpus")
    _score01(policy.get("sparse_threshold"), "selected sparse_threshold")
    _score01(policy.get("window_mil_threshold"), "selected window_mil_threshold")
    return dict(policy)


def _validate_sweep_bindings(
    score_ledger: Mapping[str, Any],
    deterministic_ledger: Mapping[str, Any],
    sweep: Mapping[str, Any],
) -> None:
    expected = {
        "classifier_manifest_hash": score_ledger["classifier_manifest_hash"],
        "evidence_cache_hash": score_ledger["evidence_cache_hash"],
        "dataset_view_hash": score_ledger["dataset_view_hash"],
        "score_ledger_hash": score_ledger["score_ledger_hash"],
        "deterministic_ledger_hash": deterministic_ledger["deterministic_ledger_hash"],
        "deterministic_filter_policy_hash": deterministic_ledger["deterministic_filter_policy_hash"],
        "row_identity_universe_hash": score_ledger["row_identity_universe_hash"],
    }
    for key, value in expected.items():
        if sweep.get(key) != value:
            raise ValueError(f"policy sweep {key} drift")
    if score_ledger.get("report_hash") is not None and sweep.get("score_ledger_report_hash") not in {None, "", score_ledger["report_hash"]}:
        raise ValueError("policy sweep score_ledger_report_hash drift")
    expected_precision = _score_ledger_precision_binding(score_ledger)
    for key, value in expected_precision.items():
        observed = sweep.get(key)
        if key == "window_mil_precision_legacy_default" and value is True and observed in {None, True}:
            continue
        if observed != value:
            raise ValueError(f"policy sweep {key} drift")


def _score_threshold_mask(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    field: str,
    threshold: float,
) -> int:
    mask = 0
    for index, (score_row, _det_row) in enumerate(pairs):
        if float(score_row[field]) >= float(threshold):
            mask |= 1 << index
    return mask


def _accepted_token_lower_bound(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    reject_mask: int,
    all_mask: int,
) -> int:
    accepted_mask = all_mask & ~reject_mask
    total = 0
    for index, (score_row, _det_row) in enumerate(pairs):
        if accepted_mask & (1 << index):
            total += int(score_row["core_token_lower_bound"])
    return total


def _accepted_token_lower_bound_from_weights(
    *,
    reject_mask: int,
    all_mask: int,
    token_weights: Sequence[int],
) -> int:
    accepted_mask = all_mask & ~reject_mask
    total = 0
    while accepted_mask:
        bit = accepted_mask & -accepted_mask
        total += int(token_weights[bit.bit_length() - 1])
        accepted_mask ^= bit
    return total


def _unique_thresholds(scores: Sequence[float]) -> list[float]:
    values = {_score01(score, "threshold score") for score in scores}
    values.add(0.0)
    values.add(1.0)
    return sorted(values)


def _score_ledger_missing_or_nonfinite_count(score_ledger: Mapping[str, Any]) -> int:
    total = 0
    for row in score_ledger.get("rows") or []:
        if row.get("sparse_score_finite") is not True or row.get("window_mil_score_finite") is not True:
            total += 1
    return total


def _allowed_false_negatives(positive_count: int, target_recall: float) -> int:
    if positive_count < 1:
        return 0
    return max(0, math.floor((1.0 - float(target_recall)) * int(positive_count)))


def _public_runtime_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "candidate_id_hash": str(row["candidate_id"]),
        "row_identity_hash": stable_json_hash(
            {
                "candidate_id": str(row["candidate_id"]),
                "row_hash": str(row["row_hash"]),
            }
        ),
        "split": str(row["split"]),
        "evidence_tier": str(row["evidence_tier"]),
        "core_char_start": int(row["halo_left_chars"]),
        "core_char_end": int(row["halo_left_chars"]) + int(row["core_char_length"]),
        "core_text_sha256": str(row["core_text_sha256"]),
        "evidence_text_sha256": str(row["evidence_text_sha256"]),
    }
    validate_public_no_raw_payload(payload)
    return payload


def _require_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("report requires non-empty rows")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("report rows must be objects")
    return rows


def _require_sorted_unique_candidates(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    candidate_ids = [str(row.get("candidate_id") or "") for row in rows]
    if candidate_ids != sorted(candidate_ids):
        raise ValueError(f"{label} candidate identity mismatch: rows must be sorted by candidate_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"{label} candidate identity mismatch: duplicate candidate")


def _count_field(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(str(row[field]) for row in rows)
    return {key: int(counter[key]) for key in sorted(counter)}


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
    index = min(len(ordered) - 1, max(0, math.ceil(float(q) * len(ordered)) - 1))
    return int(ordered[index])


def _core_token_lower_bound(core_char_length: int) -> int:
    return max(1, int(core_char_length) // DEFAULT_TOKEN_BYTES_PER_TOKEN_LOWER_BOUND)


def _score01(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite score in [0, 1]") from exc
    if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
        raise ValueError(f"{field_name} must be a finite score in [0, 1]")
    return parsed


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _nonnegative_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return parsed


def _positive_int(value: Any, field_name: str) -> int:
    parsed = _nonnegative_int(value, field_name)
    if parsed < 1:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return value


def _optional_sha256(value: Any, field_name: str) -> str | None:
    if value is None or value == "":
        return None
    return _require_sha256(value, field_name)


def _public_false_flags() -> dict[str, bool]:
    return {
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "accepted_manifest_created": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "paid_api_used": False,
        "gpu_rental_used": False,
    }


def _hash_without(payload: Mapping[str, Any], field: str) -> str:
    return stable_json_hash({key: value for key, value in payload.items() if key != field})


def _hash_without_many(payload: Mapping[str, Any], fields: set[str]) -> str:
    return stable_json_hash({key: value for key, value in payload.items() if key not in fields})


def _write_public_json(path: Path, payload: Mapping[str, Any]) -> None:
    validate_public_no_raw_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
