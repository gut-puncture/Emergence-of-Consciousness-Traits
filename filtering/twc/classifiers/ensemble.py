"""Reject-only ensemble semantics for local classifier gates."""

from __future__ import annotations

from collections.abc import Mapping

from twc.classifiers.schema import (
    ClassifierDecision,
    ClassifierEnsembleDecision,
    EnsembleManifest,
)


def evaluate_classifier_ensemble(
    ensemble: EnsembleManifest,
    scores: Mapping[str, float],
) -> ClassifierEnsembleDecision:
    """Evaluate classifier scores as a required reject/quarantine gate.

    This function intentionally has no acceptance branch. Even when every score
    looks safe, the decision remains pending because final approval requires the
    full strict filter, audits, freeze lineage, and post-tokenization checks.
    """

    normalized_scores = {str(model_id): _score01(score) for model_id, score in scores.items()}
    required_model_ids = set(ensemble.model_artifacts)
    missing = sorted(required_model_ids - set(normalized_scores))
    unknown = sorted(set(normalized_scores) - required_model_ids)
    quarantine_reasons: list[str] = []
    rejection_reasons: list[str] = []
    pending_reasons: list[str] = []

    if missing and ensemble.approval_policy.reject_on_missing_classifier:
        quarantine_reasons.extend(f"missing_classifier:{model_id}" for model_id in missing)
    if unknown:
        quarantine_reasons.extend(f"unknown_classifier_score:{model_id}" for model_id in unknown)

    if quarantine_reasons:
        return ClassifierEnsembleDecision(
            status=ClassifierDecision.QUARANTINED,
            scores=normalized_scores,
            quarantine_reasons=tuple(quarantine_reasons),
            decision_capability=ensemble.decision_capability,
        )

    for model_id, artifact in sorted(ensemble.model_artifacts.items()):
        score = normalized_scores[model_id]
        if artifact.threshold_direction == "reject_at_or_above":
            if score >= artifact.threshold:
                rejection_reasons.append(model_id)
        elif artifact.threshold_direction == "pass_at_or_above":
            if score < artifact.threshold:
                quarantine_reasons.append(f"{model_id}_below_threshold")
        else:
            quarantine_reasons.append(f"unsupported_threshold_direction:{model_id}")

    if rejection_reasons:
        return ClassifierEnsembleDecision(
            status=ClassifierDecision.REJECTED,
            scores=normalized_scores,
            rejection_reasons=tuple(rejection_reasons),
            decision_capability=ensemble.decision_capability,
        )

    if quarantine_reasons:
        return ClassifierEnsembleDecision(
            status=ClassifierDecision.QUARANTINED,
            scores=normalized_scores,
            quarantine_reasons=tuple(quarantine_reasons),
            decision_capability=ensemble.decision_capability,
        )

    pending_reasons.append("classifiers_cannot_approve_alone")
    return ClassifierEnsembleDecision(
        status=ClassifierDecision.PENDING_REQUIRED_GATES,
        scores=normalized_scores,
        pending_reasons=tuple(pending_reasons),
        all_required_gates_present=False,
        decision_capability=ensemble.decision_capability,
    )


def _score01(value: float) -> float:
    score = float(value)
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score
