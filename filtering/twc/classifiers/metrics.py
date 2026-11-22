"""Calibration and threshold metrics for local classifier gates."""

from __future__ import annotations

from collections.abc import Sequence


def binary_calibration_metrics(
    *,
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
    bins: int = 10,
) -> dict[str, float | int]:
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")
    if not y_true:
        raise ValueError("At least one scored example is required")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if bins <= 0:
        raise ValueError("bins must be positive")

    labels = [1 if int(label) else 0 for label in y_true]
    scores = [_clip01(float(score)) for score in y_score]
    count = len(labels)

    true_positive = sum(1 for label, score in zip(labels, scores, strict=True) if label == 1 and score >= threshold)
    false_positive = sum(1 for label, score in zip(labels, scores, strict=True) if label == 0 and score >= threshold)
    true_negative = sum(1 for label, score in zip(labels, scores, strict=True) if label == 0 and score < threshold)
    false_negative = sum(1 for label, score in zip(labels, scores, strict=True) if label == 1 and score < threshold)

    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, true_positive + false_negative)
    specificity = _safe_div(true_negative, true_negative + false_positive)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    brier = sum((score - label) ** 2 for label, score in zip(labels, scores, strict=True)) / count
    ece = _expected_calibration_error(labels, scores, bins)

    return {
        "count": count,
        "positive_count": sum(labels),
        "negative_count": count - sum(labels),
        "threshold": threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "brier_score": brier,
        "expected_calibration_error": ece,
    }


def _expected_calibration_error(labels: list[int], scores: list[float], bins: int) -> float:
    total = len(labels)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if bin_index == bins - 1:
            selected = [(label, score) for label, score in zip(labels, scores, strict=True) if lower <= score <= upper]
        else:
            selected = [(label, score) for label, score in zip(labels, scores, strict=True) if lower <= score < upper]
        if not selected:
            continue
        bin_labels = [label for label, _ in selected]
        bin_scores = [score for _, score in selected]
        accuracy = sum(bin_labels) / len(bin_labels)
        confidence = sum(bin_scores) / len(bin_scores)
        ece += (len(selected) / total) * abs(accuracy - confidence)
    return ece


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _clip01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value
