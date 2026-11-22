"""Local classifier gates for the TWC strict filter."""

from twc.classifiers.ensemble import evaluate_classifier_ensemble
from twc.classifiers.pipeline import load_classifier_config, train_classifier_ensemble

__all__ = [
    "evaluate_classifier_ensemble",
    "load_classifier_config",
    "train_classifier_ensemble",
]
