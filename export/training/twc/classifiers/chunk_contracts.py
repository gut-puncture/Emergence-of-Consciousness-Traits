"""Dependency-light contracts for current chunk classifiers and reports."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from twc.repro.public_artifacts import validate_public_no_raw_payload
from twc.repro.hashing import stable_json_hash


CHUNK_SPARSE_REPORT_KIND = "twc_chunk_hashed_ngram_sparse_report"
CHUNK_WINDOW_MIL_REPORT_KIND = "twc_chunk_modernbert_window_mil_report"
CHUNK_SPARSE_MODEL_ID = "chunk_hashed_ngram_linear_risk"
CHUNK_WINDOW_MIL_MODEL_ID = "chunk_modernbert_window_mil_risk"

WINDOW_INPUT_FULL_EVIDENCE = "full_evidence"
WINDOW_INPUT_CORE_ONLY = "core_only"
WINDOW_INPUT_SPLIT_HALO_CAPPED = "split_halo_capped"
SUPPORTED_WINDOW_INPUT_MODES = frozenset(
    {WINDOW_INPUT_FULL_EVIDENCE, WINDOW_INPUT_CORE_ONLY, WINDOW_INPUT_SPLIT_HALO_CAPPED}
)

DEFAULT_HALO_CHARS = 512
DEFAULT_TARGET_RECALL = 0.9942


class ChunkFilteringScorer(Protocol):
    model_id: str

    def score_many(self, texts: list[str], metadata: list[dict[str, Any]]) -> list[float]:
        """Return reject-risk scores in the same order as ``texts``."""


def build_chunk_runtime_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
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


def coerce_risk_score(value: Any, field_name: str | None = None) -> float:
    """Return a finite probability while preserving existing error contracts."""

    if field_name is not None:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a finite score in [0, 1]") from exc
        if not math.isfinite(parsed) or parsed < 0 or parsed > 1:
            raise ValueError(f"{field_name} must be a finite score in [0, 1]")
        return parsed

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("classifier score must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError("classifier score must be finite")
    if parsed < 0 or parsed > 1:
        raise ValueError("classifier score must be in [0, 1]")
    return parsed


def coerce_optional_risk_score(value: Any) -> float | None:
    try:
        return coerce_risk_score(value)
    except ValueError:
        return None


def validate_chunk_sparse_inference_report(report: Mapping[str, Any]) -> None:
    """Validate the bounded report contract needed to load a sparse artifact."""

    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_SPARSE_REPORT_KIND:
        raise ValueError("unsupported chunk sparse report kind")
    if report.get("status") != "complete":
        raise ValueError("chunk sparse report must be complete")
    if report.get("model_id") != CHUNK_SPARSE_MODEL_ID:
        raise ValueError("chunk sparse report model id mismatch (chunk sparse model_id mismatch)")
    _validate_inference_threshold(report.get("threshold"), "chunk sparse threshold")
    for field in (
        "classifier_manifest_hash",
        "evidence_cache_hash",
        "feature_config_hash",
        "model_file_sha256",
        "model_payload_hash",
    ):
        require_sha256(str(report.get(field) or ""), field)
    feature_config = report.get("feature_config")
    if not isinstance(feature_config, Mapping) or int(feature_config.get("n_features", 0)) < 2:
        raise ValueError("chunk sparse report missing valid feature_config")
    _validate_optional_report_hash(report, "chunk sparse report")


def validate_chunk_window_mil_inference_report(report: Mapping[str, Any]) -> None:
    """Validate the bounded report contract needed to load a Window-MIL artifact."""

    validate_public_no_raw_payload(report)
    if report.get("artifact_kind") != CHUNK_WINDOW_MIL_REPORT_KIND:
        raise ValueError("unsupported chunk window-MIL report kind")
    if report.get("status") != "complete":
        raise ValueError("chunk window-MIL report must be complete")
    if report.get("model_id") != CHUNK_WINDOW_MIL_MODEL_ID:
        raise ValueError("chunk window-MIL report model id mismatch (chunk Window-MIL model_id mismatch)")
    _validate_inference_threshold(report.get("threshold"), "chunk Window-MIL threshold")
    for field in (
        "classifier_manifest_hash",
        "evidence_cache_hash",
        "model_file_sha256",
        "model_payload_hash",
    ):
        require_sha256(str(report.get(field) or ""), field)
    if not str(report.get("base_model_name") or ""):
        raise ValueError("chunk Window-MIL report missing base_model_name")
    max_length = _positive_int(report.get("max_length"), "max_length")
    if max_length < 8:
        raise ValueError("chunk Window-MIL max_length must be at least 8")
    stride = _nonnegative_int(report.get("stride"), "stride")
    if stride >= max_length:
        raise ValueError("chunk Window-MIL stride must be less than max_length")
    _positive_int(report.get("max_windows_per_device_batch"), "max_windows_per_device_batch")
    _validate_optional_report_hash(report, "chunk Window-MIL report")


def validate_chunk_sparse_report(report: Mapping[str, Any]) -> None:
    validate_chunk_sparse_inference_report(report)
    if "training_config_hash" in report:
        require_sha256(str(report.get("training_config_hash") or ""), "training_config_hash")
    if "curriculum_hash" in report:
        require_sha256(str(report.get("curriculum_hash") or ""), "curriculum_hash")
    if report.get("threshold_selection", {}).get("split") != "calibration":
        raise ValueError("chunk sparse threshold must be selected on calibration")
    if report.get("threshold_selection", {}).get("eval_split_used_for_threshold_selection") is not False:
        raise ValueError("chunk sparse eval split must not select threshold")
    if "dataset_view_hash" in report:
        require_sha256(str(report.get("dataset_view_hash") or ""), "dataset_view_hash")
    _require_metric_block(report, "calibration_metrics")
    _require_metric_block(report, "eval_metrics")
    _validate_metric_counts_match_splits(report)
    if report.get("report_hash") != hash_without(report, "report_hash"):
        raise ValueError("chunk sparse report hash mismatch")


def validate_chunk_window_mil_report(report: Mapping[str, Any]) -> None:
    validate_chunk_window_mil_inference_report(report)
    require_sha256(str(report.get("model_numeric_hash") or ""), "model_numeric_hash")
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
        require_sha256(str(report.get("dataset_view_hash") or ""), "dataset_view_hash")
    if "window_variant_config_hash" in report:
        require_sha256(
            str(report.get("window_variant_config_hash") or ""),
            "window_variant_config_hash",
        )
        config = report.get("window_variant_config")
        if not isinstance(config, Mapping):
            raise ValueError("chunk window-MIL report missing window_variant_config")
        validate_window_variant_config(
            input_mode=str(config.get("input_mode") or ""),
            pooling_mode=str(config.get("pooling_mode") or ""),
            pooling_top_k=int(config.get("pooling_top_k", 0)),
            halo_weight=float(config.get("halo_weight", float("nan"))),
            halo_logit_clamp=float(config.get("halo_logit_clamp", float("nan"))),
        )
        if report["window_variant_config_hash"] != stable_json_hash(config):
            raise ValueError("chunk window-MIL variant config hash mismatch")
    if "curriculum_hash" in report:
        require_sha256(str(report.get("curriculum_hash") or ""), "curriculum_hash")
    if "runtime_profile" in report:
        _validate_runtime_profile(report["runtime_profile"])
    if report.get("report_hash") != hash_without(report, "report_hash"):
        raise ValueError("chunk window-MIL report hash mismatch")


def validate_window_variant_config(
    *,
    input_mode: str,
    pooling_mode: str,
    pooling_top_k: int,
    halo_weight: float,
    halo_logit_clamp: float,
) -> None:
    if input_mode not in SUPPORTED_WINDOW_INPUT_MODES:
        raise ValueError(f"unsupported Window-MIL input mode: {input_mode}")
    if pooling_mode not in {"max", "topk_mean_logits"}:
        raise ValueError(f"unsupported Window-MIL pooling mode: {pooling_mode}")
    if int(pooling_top_k) < 1:
        raise ValueError("pooling_top_k must be positive")
    if pooling_mode == "max" and int(pooling_top_k) != 1:
        raise ValueError("max pooling requires pooling_top_k=1")
    if not math.isfinite(float(halo_weight)) or float(halo_weight) < 0:
        raise ValueError("halo_weight must be finite and nonnegative")
    if not math.isfinite(float(halo_logit_clamp)) or float(halo_logit_clamp) <= 0:
        raise ValueError("halo_logit_clamp must be finite and positive")
    if input_mode == WINDOW_INPUT_SPLIT_HALO_CAPPED and pooling_mode == "max":
        raise ValueError("split_halo_capped must not use max pooling")


def require_finite_scores(scores: Sequence[float], label: str) -> None:
    if not scores:
        raise ValueError(f"{label} scores are empty")
    for score in scores:
        if not math.isfinite(float(score)):
            raise ValueError(f"{label} contains non-finite score")


def require_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return value


def hash_without(payload: Mapping[str, Any], field: str) -> str:
    return stable_json_hash({key: value for key, value in payload.items() if key != field})


def _validate_inference_threshold(value: Any, label: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric and finite") from exc
    if not math.isfinite(score):
        raise ValueError(f"{label} must be numeric and finite")
    if not 0 <= score <= 1:
        raise ValueError(f"{label} must be in [0, 1]")
    return score


def _positive_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _validate_optional_report_hash(report: Mapping[str, Any], label: str) -> None:
    if report.get("report_hash") is None:
        return
    if report.get("report_hash") != hash_without(report, "report_hash"):
        raise ValueError(f"{label} report_hash content mismatch")


def _require_metric_block(report: Mapping[str, Any], field: str) -> None:
    metrics = report.get(field)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{field} missing")
    for key in ("count", "positive_count", "negative_count", "recall", "false_negative"):
        if key not in metrics:
            raise ValueError(f"{field} missing {key}")
    if int(metrics["positive_count"]) < 1 or int(metrics["negative_count"]) < 1:
        raise ValueError(f"{field} must contain both labels")
    require_finite_scores([float(metrics["recall"])], field)


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


def _validate_runtime_profile(profile: Any) -> None:
    if not isinstance(profile, Mapping):
        raise ValueError("window-MIL runtime_profile must be an object")
    validate_public_no_raw_payload(profile)
    if profile.get("checkpoint_resume_support") is not True:
        raise ValueError("window-MIL runtime_profile must confirm checkpoint resume support")
    if profile.get("activation_checkpoint_windows") is not True:
        raise ValueError("window-MIL runtime_profile must confirm activation checkpointing")
    for field in (
        "elapsed_seconds",
        "train_step_seconds",
        "checkpoint_write_seconds",
        "rows_per_second",
    ):
        value = float(profile.get(field, -1.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"window-MIL runtime_profile {field} invalid")
