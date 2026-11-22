"""Window-level transformer MIL classifier training for final 35k labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from twc.classifiers.final_labels import validate_public_no_raw_payload
from twc.classifiers.metrics import binary_calibration_metrics
from twc.classifiers.pipeline import load_classifier_config
from twc.classifiers.schema import ClassifierPipelineConfig, validate_budget_ledger
from twc.classifiers.sparse_baseline import (
    _batches,
    _label_to_int,
    _order_rows_for_refetch,
    _select_threshold_for_recall,
    _sha256_file,
    _training_manifest_hash,
    _validate_fit_rows,
    _validate_split_rows,
    _write_json_atomic,
)
from twc.data.decision_schema import stable_json_hash


WINDOW_MIL_REPORT_KIND = "modernbert_window_mil_report"
WINDOW_MIL_SCORE_LEDGER_KIND = "modernbert_window_mil_score_ledger"
WINDOW_MIL_CHECKPOINT_KIND = "modernbert_window_mil_checkpoint"
WINDOW_MIL_PROGRESS_KIND = "modernbert_window_mil_progress"
WINDOW_MIL_LOCAL_MICROBENCH_KIND = "twc_window_mil_local_microbench_v1"
FEATURE_CODE_VERSION = "modernbert_window_mil_v1"
MODEL_ID = "modernbert_window_mil_risk"
DEFAULT_MODEL_NAME = "answerdotai/ModernBERT-base"
W1_ATTN_IMPLEMENTATION_SDPA = "sdpa"
W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2 = "flash_attention_2"
SUPPORTED_W1_ATTN_IMPLEMENTATIONS = frozenset(
    {W1_ATTN_IMPLEMENTATION_SDPA, W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2}
)
ACTIVATION_CHECKPOINT_WINDOWS = True
WINDOW_POOLING_MAX = "max"
WINDOW_POOLING_TOPK_MEAN = "topk_mean_logits"
SUPPORTED_WINDOW_POOLING_MODES = frozenset({WINDOW_POOLING_MAX, WINDOW_POOLING_TOPK_MEAN})
TEXT_HASH_ALGORITHM_UTF8_SHA256_V1 = "utf8_sha256_v1"
LEGACY_PRIVATE_MINING_TEXT_HASH_ALGORITHM = "legacy_private_mining_stable_json_raw_text_v1"
TRANSIENT_FETCH_ERROR_NAMES = frozenset(
    {
        "ClientConnectionError",
        "ClientPayloadError",
        "ConnectionError",
        "ContentLengthError",
        "IncompleteReadError",
        "OSError",
        "TimeoutError",
    }
)
FETCH_RETRY_ATTEMPTS = 4
FETCH_RETRY_BASE_SLEEP_SECONDS = 1.0


def normalize_w1_attn_implementation(value: str | None) -> str:
    normalized = str(value or W1_ATTN_IMPLEMENTATION_SDPA).strip().lower()
    if normalized not in SUPPORTED_W1_ATTN_IMPLEMENTATIONS:
        allowed = ", ".join(sorted(SUPPORTED_W1_ATTN_IMPLEMENTATIONS))
        raise ValueError(f"w1_attn_implementation must be one of: {allowed}")
    return normalized


def _observed_attn_implementation(encoder: Any, *, requested: str) -> str:
    config = getattr(encoder, "config", None)
    for attr in ("_attn_implementation", "attn_implementation"):
        observed = getattr(config, attr, None) if config is not None else None
        if observed:
            return str(observed).strip().lower()
    return requested


@dataclass(frozen=True)
class PreparedWindowMilBatch:
    """CPU-tokenized window batch for exact inference-only W1 scoring."""

    encoded: Mapping[str, Any]
    mapping: Any
    document_count: int
    window_count: int
    max_windows_per_document: int
    padded_token_count: int
    pooling_mode: str
    pooling_top_k: int
    pin_memory_requested: bool = False
    pin_memory_effective: bool = False
    telemetry: Mapping[str, Any] | None = None
    metadata: tuple[Mapping[str, Any], ...] = ()
    candidate_ids: tuple[str, ...] = ()

    def slice(self, start: int, end: int) -> "PreparedWindowMilBatch":
        if start < 0 or end < start or end > int(self.document_count):
            raise ValueError("prepared W1 batch slice bounds are invalid")
        if start == 0 and end == int(self.document_count):
            return self
        local_indexes = [
            index
            for index, value in enumerate(self.mapping.detach().cpu().tolist())
            if int(start) <= int(value) < int(end)
        ]
        if not local_indexes:
            raise ValueError("prepared W1 batch slice selected no windows")
        index_tensor = self.mapping.new_tensor(local_indexes, dtype=self.mapping.dtype)
        sliced_mapping = self.mapping.index_select(0, index_tensor) - int(start)
        sliced_encoded = {
            key: (
                value.index_select(0, index_tensor)
                if hasattr(value, "index_select") and int(getattr(value, "shape", (0,))[0]) == int(self.window_count)
                else value
            )
            for key, value in self.encoded.items()
        }
        telemetry = dict(self.telemetry or {})
        window_count = int(sliced_mapping.numel())
        first_tensor = next((value for value in sliced_encoded.values() if hasattr(value, "numel")), None)
        padded_token_count = int(first_tensor.numel()) if first_tensor is not None else window_count
        return replace(
            self,
            encoded=sliced_encoded,
            mapping=sliced_mapping,
            document_count=int(end) - int(start),
            window_count=window_count,
            padded_token_count=padded_token_count,
            telemetry=telemetry,
            metadata=tuple(self.metadata[int(start):int(end)]),
            candidate_ids=tuple(self.candidate_ids[int(start):int(end)]),
        )


def train_window_mil_transformer(
    *,
    training_manifest: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    config: ClassifierPipelineConfig,
    artifact_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
    model_bundle_factory: Any | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    stride: int = 128,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    fetch_concurrency: int = 1,
    batch_order: str = "source_file",
    target_recall: float = 0.995,
    device: str = "auto",
    require_cuda: bool = False,
    amp: bool = True,
    max_windows_per_device_batch: int = 512,
    checkpoint_every_batches: int = 1,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    budget_cap_usd: float | None = None,
    estimated_usd: float = 0.0,
    actual_usd: float = 0.0,
    compute_backend: str = "prime_gpu_transformer_window_mil",
) -> dict[str, Any]:
    """Train the primary transformer MIL reject-risk classifier.

    Raw text is refetched only in memory from private row providers. All
    persisted JSON artifacts are validated no-raw public metadata.
    """

    torch = _load_torch()
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    report_path = artifact_path / f"{MODEL_ID}_report.json"
    progress_path = artifact_path / f"{MODEL_ID}_progress.json"
    checkpoint_path = artifact_path / f"{MODEL_ID}_checkpoint.pt"
    checkpoint_meta_path = artifact_path / f"{MODEL_ID}_checkpoint.json"
    model_path = artifact_path / f"{MODEL_ID}_model.pt"

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validate_window_mil_report(report)
        return report

    _require_component(config)
    _validate_schedule_no_raw(schedule)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    if not 0 <= stride < max_length:
        raise ValueError("stride must be in [0, max_length)")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if fetch_concurrency < 1:
        raise ValueError("fetch_concurrency must be positive")
    if fetch_concurrency > 1 and bool(getattr(row_provider, "manage_hf_cache_env", False)):
        raise ValueError("parallel window-MIL collection requires row_provider.manage_hf_cache_env=false")
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    if checkpoint_every_batches < 1:
        raise ValueError("checkpoint_every_batches must be positive")

    rows = list(training_manifest.get("rows") or [])
    training_data_manifest_hash = _training_manifest_hash(training_manifest)
    fit_rows = [row for row in rows if row.get("fit_eligible") is not False and str(row.get("split")) != "audit"]
    audit_row_count = len(rows) - len(fit_rows)
    _validate_fit_rows(fit_rows)
    train_rows = [row for row in fit_rows if row["split"] == "train"]
    calibration_rows = [row for row in fit_rows if row["split"] == "calibration"]
    eval_rows = [row for row in fit_rows if row["split"] == "eval"]
    _validate_split_rows(train_rows, calibration_rows, eval_rows)

    train_rows = _order_rows_for_refetch(train_rows, candidates=candidates, schedule=schedule, batch_order=batch_order)
    calibration_rows = _order_rows_for_refetch(
        calibration_rows,
        candidates=candidates,
        schedule=schedule,
        batch_order=batch_order,
    )
    eval_rows = _order_rows_for_refetch(eval_rows, candidates=candidates, schedule=schedule, batch_order=batch_order)

    resolved_device = _resolve_device(torch, device=device, require_cuda=require_cuda)
    tokenizer, model, resolved_model_name = _load_model_bundle(
        model_name=model_name,
        model_bundle_factory=model_bundle_factory,
        torch=torch,
    )
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    scaler = _make_grad_scaler(torch, enabled=amp and resolved_device.type == "cuda")
    autocast = _autocast_context(torch, enabled=amp and resolved_device.type == "cuda")

    train_batches = _batches(train_rows, batch_size)
    total_train_batches = len(train_batches) * int(epochs)
    completed_train_batches = 0
    if checkpoint_path.exists() and checkpoint_meta_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed_train_batches = int(checkpoint["completed_train_batches"])
        _validate_checkpoint_meta(
            json.loads(checkpoint_meta_path.read_text(encoding="utf-8")),
            training_data_manifest_hash=training_data_manifest_hash,
            model_name=resolved_model_name,
            max_length=max_length,
            stride=stride,
            batch_order=batch_order,
        )

    class_weights = _balanced_sample_weights([_label_to_int(row["classifier_label"]) for row in train_rows])
    max_batches_allowed = total_train_batches
    if max_train_batches is not None:
        max_batches_allowed = min(max_batches_allowed, int(max_train_batches))
    loss_sum = 0.0
    trained_rows = 0

    for global_batch_index in range(completed_train_batches, max_batches_allowed):
        epoch_index = global_batch_index // len(train_batches)
        batch_index = global_batch_index % len(train_batches)
        batch = train_batches[batch_index]
        texts = _fetch_private_texts_for_rows_with_retries(
            batch,
            candidates=candidates,
            schedule=schedule,
            row_provider=row_provider,
            fetch_concurrency=fetch_concurrency,
        )
        labels = [_label_to_int(row["classifier_label"]) for row in batch]
        weights = [class_weights[label] for label in labels]
        batch_loss = _train_one_document_batch(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scaler=scaler,
            autocast=autocast,
            texts=texts,
            labels=labels,
            sample_weights=weights,
            device=resolved_device,
            max_length=max_length,
            stride=stride,
            max_windows_per_device_batch=max_windows_per_device_batch,
        )
        loss_sum += batch_loss * len(batch)
        trained_rows += len(batch)
        completed_train_batches = global_batch_index + 1
        should_checkpoint = (
            completed_train_batches % int(checkpoint_every_batches) == 0
            or completed_train_batches == max_batches_allowed
            or completed_train_batches == total_train_batches
        )
        if should_checkpoint:
            _save_checkpoint(
                torch=torch,
                checkpoint_path=checkpoint_path,
                checkpoint_meta_path=checkpoint_meta_path,
                model=model,
                optimizer=optimizer,
                completed_train_batches=completed_train_batches,
                total_train_batches=total_train_batches,
                training_data_manifest_hash=training_data_manifest_hash,
                model_name=resolved_model_name,
                max_length=max_length,
                stride=stride,
                batch_order=batch_order,
                epoch_index=epoch_index,
                batch_index=batch_index,
                checkpoint_every_batches=checkpoint_every_batches,
            )
        _write_json_atomic(
            progress_path,
            _progress_payload(
                status="training",
                completed_train_batches=completed_train_batches,
                total_train_batches=total_train_batches,
                fit_eligible_row_count=len(fit_rows),
                audit_row_count=audit_row_count,
                mean_train_loss=loss_sum / max(1, trained_rows),
            ),
        )

    if completed_train_batches < total_train_batches:
        progress = _progress_payload(
            status="incomplete",
            completed_train_batches=completed_train_batches,
            total_train_batches=total_train_batches,
            fit_eligible_row_count=len(fit_rows),
            audit_row_count=audit_row_count,
            mean_train_loss=loss_sum / max(1, trained_rows),
        )
        _write_json_atomic(progress_path, progress)
        return progress

    calibration_scores, calibration_labels = _score_rows(
        calibration_rows,
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        candidates=candidates,
        schedule=schedule,
        row_provider=row_provider,
        device=resolved_device,
        max_length=max_length,
        stride=stride,
        batch_size=batch_size,
        fetch_concurrency=fetch_concurrency,
        max_windows_per_device_batch=max_windows_per_device_batch,
        max_batches=max_eval_batches,
    )
    eval_scores, eval_labels = _score_rows(
        eval_rows,
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        candidates=candidates,
        schedule=schedule,
        row_provider=row_provider,
        device=resolved_device,
        max_length=max_length,
        stride=stride,
        batch_size=batch_size,
        fetch_concurrency=fetch_concurrency,
        max_windows_per_device_batch=max_windows_per_device_batch,
        max_batches=max_eval_batches,
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

    torch.save({"model_state_dict": model.state_dict(), "model_name": resolved_model_name}, model_path)
    model_file_sha256 = _sha256_file(model_path)
    budget = validate_budget_ledger(
        {
            "budget_cap_usd": float(config.budget_cap_usd if budget_cap_usd is None else budget_cap_usd),
            "estimated_usd": float(estimated_usd),
            "actual_usd": float(actual_usd),
            "compute_backend": compute_backend,
            "paid_api_used": False,
            "gpu_rental_used": resolved_device.type == "cuda",
            "notes": "window_mil_transformer_no_paid_api_no_accepted_manifest",
        },
        budget_cap_usd=config.budget_cap_usd,
    )
    provider_observations = (
        row_provider.observations()
        if hasattr(row_provider, "observations")
        else {"provider_class": type(row_provider).__name__}
    )
    report = {
        "artifact_kind": WINDOW_MIL_REPORT_KIND,
        "schema_version": 1,
        "status": "complete",
        "model_id": MODEL_ID,
        "model_kind": "transformer_window_mil",
        "feature_code_version": FEATURE_CODE_VERSION,
        "training_data_manifest_hash": training_data_manifest_hash,
        "config_hash": config.config_hash,
        "base_model_name": resolved_model_name,
        "base_model_name_hash": stable_json_hash({"base_model_name": resolved_model_name}),
        "fit_eligible_row_count": len(fit_rows),
        "audit_row_count": audit_row_count,
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in fit_rows).items())),
        "train_label_counts": dict(sorted(Counter(str(row["classifier_label"]) for row in train_rows).items())),
        "calibration_label_counts": dict(sorted(Counter(str(row["classifier_label"]) for row in calibration_rows).items())),
        "eval_label_counts": dict(sorted(Counter(str(row["classifier_label"]) for row in eval_rows).items())),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "stride": int(stride),
        "fetch_concurrency": int(fetch_concurrency),
        "batch_order": batch_order,
        "completed_train_batches": completed_train_batches,
        "total_train_batches": total_train_batches,
        "mean_train_loss": loss_sum / max(1, trained_rows),
        "threshold": threshold,
        "threshold_selection": {
            "split": "calibration",
            "target_reject_recall": float(target_recall),
            "direction": "reject_at_or_above",
        },
        "calibration_metrics": calibration_metrics,
        "eval_metrics": eval_metrics,
        "cost_ledger": budget,
        "device": _device_observations(torch, resolved_device),
        "amp_enabled": bool(amp and resolved_device.type == "cuda"),
        "max_windows_per_device_batch": int(max_windows_per_device_batch),
        "activation_checkpoint_windows": ACTIVATION_CHECKPOINT_WINDOWS,
        "checkpoint_every_batches": int(checkpoint_every_batches),
        "model_file_name": model_path.name,
        "model_file_sha256": model_file_sha256,
        "model_numeric_hash": _model_numeric_hash(torch, model),
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
            "model_numeric_hash": report["model_numeric_hash"],
            "training_data_manifest_hash": training_data_manifest_hash,
            "base_model_name_hash": report["base_model_name_hash"],
        }
    )
    report["report_hash"] = _report_hash(report)
    validate_window_mil_report(report)
    _write_json_atomic(report_path, report)
    _write_json_atomic(
        progress_path,
        _progress_payload(
            status="complete",
            completed_train_batches=completed_train_batches,
            total_train_batches=total_train_batches,
            fit_eligible_row_count=len(fit_rows),
            audit_row_count=audit_row_count,
            mean_train_loss=loss_sum / max(1, trained_rows),
        ),
    )
    return report


def validate_window_mil_report(report: Mapping[str, Any]) -> None:
    if report.get("artifact_kind") != WINDOW_MIL_REPORT_KIND:
        raise ValueError("unsupported window-MIL report kind")
    validate_public_no_raw_payload(report)
    if report.get("status") != "complete":
        raise ValueError("window-MIL report must be complete")
    if report.get("report_hash") != _report_hash(report):
        raise ValueError("window-MIL report hash mismatch")
    _require_sha256(str(report.get("training_data_manifest_hash") or ""), "training_data_manifest_hash")
    _require_sha256(str(report.get("model_file_sha256") or ""), "model_file_sha256")
    if report.get("activation_checkpoint_windows") is not True:
        raise ValueError("window-MIL report must confirm activation checkpointing")


def derive_window_mil_calibration_rethreshold_report(
    parent_report: Mapping[str, Any],
    *,
    score_ledger: Mapping[str, Any],
    target_recall: float,
    bins: int = 10,
) -> dict[str, Any]:
    """Derive a stricter threshold report from scores without retraining.

    The threshold is selected from calibration scores only. Eval scores are used
    only after threshold selection to validate the derived artifact.
    """

    validate_window_mil_report(parent_report)
    if not 0 < float(target_recall) <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    vectors = validate_window_mil_score_ledger(
        score_ledger,
        expected_training_data_manifest_hash=str(parent_report["training_data_manifest_hash"]),
        expected_parent_report_hash=str(parent_report["report_hash"]),
        expected_parent_model_file_sha256=str(parent_report["model_file_sha256"]),
        expected_parent_model_payload_hash=str(parent_report.get("model_payload_hash") or ""),
        expected_parent_model_numeric_hash=str(parent_report.get("model_numeric_hash") or ""),
        expected_split_counts=parent_report.get("split_counts"),
        expected_label_counts={
            "calibration": parent_report.get("calibration_label_counts"),
            "eval": parent_report.get("eval_label_counts"),
        },
    )
    calibration_y, calibration_s = vectors["calibration"]
    eval_y, eval_s = vectors["eval"]
    threshold = _select_threshold_for_recall(
        y_true=calibration_y,
        y_score=calibration_s,
        target_recall=float(target_recall),
    )
    calibration_metrics = binary_calibration_metrics(
        y_true=calibration_y,
        y_score=calibration_s,
        threshold=threshold,
        bins=bins,
    )
    eval_metrics = binary_calibration_metrics(
        y_true=eval_y,
        y_score=eval_s,
        threshold=threshold,
        bins=bins,
    )
    score_ledger_hash = str(score_ledger["score_ledger_hash"])
    derived = dict(parent_report)
    derived.pop("report_hash", None)
    parent_report_hash = str(parent_report["report_hash"])
    parent_model_file_sha256 = str(parent_report["model_file_sha256"])
    derived["threshold"] = threshold
    derived["threshold_selection"] = {
        "split": "calibration",
        "target_reject_recall": float(target_recall),
        "direction": "reject_at_or_above",
        "eval_split_used_for_threshold_selection": False,
        "parent_report_hash": parent_report_hash,
        "parent_model_file_sha256": parent_model_file_sha256,
        "parent_model_payload_hash": str(parent_report.get("model_payload_hash") or ""),
        "parent_model_numeric_hash": str(parent_report.get("model_numeric_hash") or ""),
        "score_ledger_hash": score_ledger_hash,
    }
    derived["calibration_metrics"] = calibration_metrics
    derived["eval_metrics"] = eval_metrics
    derived["derivation"] = {
        "kind": "calibration_only_rethreshold",
        "parent_report_hash": parent_report_hash,
        "parent_model_file_sha256": parent_model_file_sha256,
        "parent_model_payload_hash": str(parent_report.get("model_payload_hash") or ""),
        "parent_model_numeric_hash": str(parent_report.get("model_numeric_hash") or ""),
        "score_ledger_hash": score_ledger_hash,
        "eval_split_used_for_threshold_selection": False,
        "score_ledger_split_counts": {
            "calibration": len(calibration_y),
            "eval": len(eval_y),
        },
        "score_ledger_label_counts": {
            "calibration": _score_label_counts(calibration_y),
            "eval": _score_label_counts(eval_y),
        },
    }
    derived["report_hash"] = _report_hash(derived)
    validate_window_mil_report(derived)
    return derived


def validate_window_mil_score_ledger(
    score_ledger: Mapping[str, Any],
    *,
    expected_training_data_manifest_hash: str,
    expected_parent_report_hash: str,
    expected_parent_model_file_sha256: str,
    expected_parent_model_payload_hash: str,
    expected_parent_model_numeric_hash: str,
    expected_split_counts: Any,
    expected_label_counts: Mapping[str, Any],
) -> dict[str, tuple[list[int], list[float]]]:
    if not isinstance(score_ledger, Mapping):
        raise ValueError("window-MIL score ledger must be a mapping")
    validate_public_no_raw_payload(score_ledger)
    if score_ledger.get("artifact_kind") != WINDOW_MIL_SCORE_LEDGER_KIND:
        raise ValueError("unsupported window-MIL score ledger kind")
    if score_ledger.get("model_id") != MODEL_ID:
        raise ValueError("window-MIL score ledger model_id mismatch")
    if score_ledger.get("training_data_manifest_hash") != expected_training_data_manifest_hash:
        raise ValueError("window-MIL score ledger training manifest drift")
    if score_ledger.get("parent_report_hash") != expected_parent_report_hash:
        raise ValueError("window-MIL score ledger parent report drift")
    if score_ledger.get("parent_model_file_sha256") != expected_parent_model_file_sha256:
        raise ValueError("window-MIL score ledger parent model file drift")
    if score_ledger.get("parent_model_payload_hash") != expected_parent_model_payload_hash:
        raise ValueError("window-MIL score ledger parent model payload drift")
    if score_ledger.get("parent_model_numeric_hash") != expected_parent_model_numeric_hash:
        raise ValueError("window-MIL score ledger parent model numeric drift")
    for field in (
        "raw_text_serialized",
        "raw_url_serialized",
        "token_serialized",
        "accepted_manifest",
        "labels_are_corpus_approval",
        "training_shards_created",
    ):
        if score_ledger.get(field) is not False:
            raise ValueError(f"window-MIL score ledger must keep {field}=false")
    expected_hash = stable_json_hash({key: value for key, value in score_ledger.items() if key != "score_ledger_hash"})
    if score_ledger.get("score_ledger_hash") != expected_hash:
        raise ValueError("window-MIL score ledger hash mismatch")
    split_counts = expected_split_counts if isinstance(expected_split_counts, Mapping) else {}
    splits = score_ledger.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("window-MIL score ledger splits missing")
    outputs: dict[str, tuple[list[int], list[float]]] = {}
    seen_identities: set[str] = set()
    for split in ("calibration", "eval"):
        entries = splits.get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{split} score ledger must be a non-empty list")
        expected_count = int(split_counts.get(split, -1))
        if len(entries) != expected_count:
            raise ValueError(f"{split} score ledger count mismatch")
        labels: list[int] = []
        scores: list[float] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"{split} score ledger entry must be a mapping")
            extra = sorted(set(entry) - {"row_identity_hash", "label", "score"})
            if extra:
                raise ValueError(f"{split} score ledger entry contains unsupported fields: {extra}")
            row_identity_hash = _require_sha256(str(entry.get("row_identity_hash") or ""), "row_identity_hash")
            if row_identity_hash in seen_identities:
                raise ValueError("window-MIL score ledger duplicate row identity")
            seen_identities.add(row_identity_hash)
            labels.append(int(entry.get("label")))
            scores.append(float(entry.get("score")))
        normalized_labels, normalized_scores = _validated_score_vectors(labels, scores, label=split)
        expected_counts = expected_label_counts.get(split)
        if not isinstance(expected_counts, Mapping):
            raise ValueError(f"{split} score ledger expected label counts missing")
        if _score_label_counts(normalized_labels) != {
            "reject": int(expected_counts.get("reject", 0)),
            "safe": int(expected_counts.get("safe", 0)),
        }:
            raise ValueError(f"{split} score ledger label counts mismatch")
        outputs[split] = (normalized_labels, normalized_scores)
    return outputs


class _TransformerWindowMilModel:
    """Light wrapper around an HF encoder with a binary risk head."""

    def __new__(cls, torch: Any, encoder: Any) -> Any:
        class TorchModule(torch.nn.Module):
            def __init__(self, wrapped_encoder: Any) -> None:
                super().__init__()
                self.encoder = wrapped_encoder
                hidden_size = int(getattr(wrapped_encoder.config, "hidden_size", 768))
                self.classifier = torch.nn.Linear(hidden_size, 1)

            def forward(self, **encoded: Any) -> Any:
                try:
                    outputs = self.encoder(**encoded)
                except TypeError:
                    encoded = {key: value for key, value in encoded.items() if key != "token_type_ids"}
                    outputs = self.encoder(**encoded)
                hidden = outputs.last_hidden_state
                attention_mask = encoded.get("attention_mask")
                if attention_mask is None:
                    pooled = hidden[:, 0, :]
                else:
                    weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
                return self.classifier(pooled).squeeze(-1)

        return TorchModule(encoder)


def default_model_bundle_factory(
    *,
    model_name: str,
    torch_module: Any,
    attn_implementation: str = W1_ATTN_IMPLEMENTATION_SDPA,
) -> tuple[Any, Any, str]:
    requested_attn_implementation = normalize_w1_attn_implementation(attn_implementation)
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional train dependency
        raise RuntimeError("transformer window-MIL training requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    try:
        encoder = AutoModel.from_pretrained(model_name, attn_implementation=requested_attn_implementation)
    except Exception as exc:
        if requested_attn_implementation == W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2:
            raise RuntimeError(
                "W1 flash_attention_2 attention backend requested but unavailable; "
                "install compatible flash-attn/transformers dependencies or use sdpa"
            ) from exc
        raise
    observed_attn_implementation = _observed_attn_implementation(
        encoder,
        requested=requested_attn_implementation,
    )
    if (
        requested_attn_implementation == W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2
        and observed_attn_implementation != W1_ATTN_IMPLEMENTATION_FLASH_ATTENTION_2
    ):
        raise RuntimeError(
            "W1 flash_attention_2 attention backend requested but unavailable; "
            f"observed attention backend {observed_attn_implementation!r}"
        )
    config = getattr(encoder, "config", None)
    if config is not None and hasattr(config, "reference_compile"):
        setattr(config, "reference_compile", False)
    model = _TransformerWindowMilModel(torch_module, encoder)
    setattr(model, "w1_attn_implementation_requested", requested_attn_implementation)
    setattr(model, "w1_attn_implementation_observed", observed_attn_implementation)
    return tokenizer, model, model_name


def _load_model_bundle(
    *,
    model_name: str,
    model_bundle_factory: Any | None,
    torch: Any,
    attn_implementation: str = W1_ATTN_IMPLEMENTATION_SDPA,
) -> tuple[Any, Any, str]:
    factory = model_bundle_factory or default_model_bundle_factory
    return factory(
        model_name=model_name,
        torch_module=torch,
        attn_implementation=normalize_w1_attn_implementation(attn_implementation),
    )


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional train dependency
        raise RuntimeError("window-MIL training requires torch") from exc
    return torch


def _resolve_device(torch: Any, *, device: str, require_cuda: bool) -> Any:
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = device
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this window-MIL run but is not available")
    if require_cuda and not resolved.startswith("cuda"):
        raise RuntimeError("CUDA is required for this window-MIL run but device resolved to CPU")
    return torch.device(resolved)


def _make_grad_scaler(torch: Any, *, enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(torch: Any, *, enabled: bool) -> Any:
    def factory() -> Any:
        if not enabled:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=True)
        return torch.cuda.amp.autocast(enabled=True)

    return factory


def _resolve_document_microbatch_size(value: int | None, total_documents: int) -> int:
    if int(total_documents) < 1:
        raise ValueError("training batch must contain at least one document")
    if value is None:
        return int(total_documents)
    size = int(value)
    if size < 1:
        raise ValueError("document_microbatch_size must be positive")
    return min(size, int(total_documents))


def _validate_train_batch_lengths(texts: Sequence[str], labels: Sequence[int], sample_weights: Sequence[float]) -> None:
    if len(texts) != len(labels) or len(texts) != len(sample_weights):
        raise ValueError("training inputs must have matching lengths")
    if not texts:
        raise ValueError("training batch must contain at least one document")


def _train_one_document_batch(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scaler: Any,
    autocast: Any,
    texts: list[str],
    labels: list[int],
    sample_weights: list[float],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str = WINDOW_POOLING_MAX,
    pooling_top_k: int = 1,
    document_microbatch_size: int | None = None,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    _validate_train_batch_lengths(texts, labels, sample_weights)
    total_documents = len(texts)
    microbatch_size = _resolve_document_microbatch_size(document_microbatch_size, total_documents)
    loss_total = 0.0
    for start in range(0, total_documents, microbatch_size):
        end = min(total_documents, start + microbatch_size)
        with autocast():
            doc_logits = _document_logits_for_texts(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                texts=texts[start:end],
                device=device,
                max_length=max_length,
                stride=stride,
                max_windows_per_device_batch=max_windows_per_device_batch,
                pooling_mode=pooling_mode,
                pooling_top_k=pooling_top_k,
                checkpoint_window_chunks=True,
            )
            targets = torch.tensor(labels[start:end], dtype=torch.float32, device=device)
            weights = torch.tensor(sample_weights[start:end], dtype=torch.float32, device=device)
            loss_values = torch.nn.functional.binary_cross_entropy_with_logits(doc_logits, targets, reduction="none")
            loss = (loss_values * weights).sum() / float(total_documents)
        loss_total += float(loss.detach().cpu().item())
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return float(loss_total)


def _train_split_halo_document_batch(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scaler: Any,
    autocast: Any,
    core_texts: list[str],
    halo_texts: list[str],
    labels: list[int],
    sample_weights: list[float],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str = WINDOW_POOLING_TOPK_MEAN,
    pooling_top_k: int = 3,
    halo_weight: float = 0.25,
    halo_logit_clamp: float = 2.0,
    document_microbatch_size: int | None = None,
) -> float:
    if len(core_texts) != len(halo_texts) or len(core_texts) != len(labels) or len(core_texts) != len(sample_weights):
        raise ValueError("split-halo training inputs must have matching lengths")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_documents = len(core_texts)
    microbatch_size = _resolve_document_microbatch_size(document_microbatch_size, total_documents)
    loss_total = 0.0
    for start in range(0, total_documents, microbatch_size):
        end = min(total_documents, start + microbatch_size)
        with autocast():
            doc_logits = _split_halo_document_logits(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                core_texts=core_texts[start:end],
                halo_texts=halo_texts[start:end],
                device=device,
                max_length=max_length,
                stride=stride,
                max_windows_per_device_batch=max_windows_per_device_batch,
                pooling_mode=pooling_mode,
                pooling_top_k=pooling_top_k,
                halo_weight=halo_weight,
                halo_logit_clamp=halo_logit_clamp,
                checkpoint_window_chunks=True,
            )
            targets = torch.tensor(labels[start:end], dtype=torch.float32, device=device)
            weights = torch.tensor(sample_weights[start:end], dtype=torch.float32, device=device)
            loss_values = torch.nn.functional.binary_cross_entropy_with_logits(doc_logits, targets, reduction="none")
            loss = (loss_values * weights).sum() / float(total_documents)
        loss_total += float(loss.detach().cpu().item())
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return float(loss_total)


def _score_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    device: Any,
    max_length: int,
    stride: int,
    batch_size: int,
    fetch_concurrency: int,
    max_windows_per_device_batch: int,
    max_batches: int | None,
) -> tuple[list[float], list[int]]:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    batches = _batches(rows, batch_size)
    if max_batches is not None:
        batches = batches[: int(max_batches)]
    with torch.no_grad():
        for batch in batches:
            texts = _fetch_private_texts_for_rows_with_retries(
                batch,
                candidates=candidates,
                schedule=schedule,
                row_provider=row_provider,
                fetch_concurrency=fetch_concurrency,
            )
            batch_scores = _score_document_batch(
                torch=torch,
                model=model,
                tokenizer=tokenizer,
                texts=texts,
                device=device,
                max_length=max_length,
                stride=stride,
                max_windows_per_device_batch=max_windows_per_device_batch,
            )
            scores.extend(batch_scores)
            labels.extend(_label_to_int(row["classifier_label"]) for row in batch)
    return scores, labels


def _score_document_batch(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str = WINDOW_POOLING_MAX,
    pooling_top_k: int = 1,
    telemetry: MutableMapping[str, Any] | None = None,
) -> list[float]:
    doc_logits = _document_logits_for_texts(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        max_length=max_length,
        stride=stride,
        max_windows_per_device_batch=max_windows_per_device_batch,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        telemetry=telemetry,
    )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    postprocess_at = time.monotonic()
    scores = [float(value) for value in torch.sigmoid(doc_logits).detach().cpu().tolist()]
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "postprocess_seconds", time.monotonic() - postprocess_at)
    return scores


def _score_split_halo_document_batch(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    core_texts: list[str],
    halo_texts: list[str],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str = WINDOW_POOLING_TOPK_MEAN,
    pooling_top_k: int = 3,
    halo_weight: float = 0.25,
    halo_logit_clamp: float = 2.0,
    telemetry: MutableMapping[str, Any] | None = None,
) -> list[float]:
    doc_logits = _split_halo_document_logits(
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
        halo_weight=halo_weight,
        halo_logit_clamp=halo_logit_clamp,
        telemetry=telemetry,
    )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    postprocess_at = time.monotonic()
    scores = [float(value) for value in torch.sigmoid(doc_logits).detach().cpu().tolist()]
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "postprocess_seconds", time.monotonic() - postprocess_at)
    return scores


def _telemetry_add(telemetry: MutableMapping[str, Any] | None, key: str, value: float | int) -> None:
    if telemetry is None:
        return
    telemetry[key] = telemetry.get(key, 0) + value


def _telemetry_max(telemetry: MutableMapping[str, Any] | None, key: str, value: int) -> None:
    if telemetry is None:
        return
    telemetry[key] = max(int(telemetry.get(key, 0)), int(value))


def _telemetry_merge_prepared_payload(
    telemetry: MutableMapping[str, Any] | None,
    payload: Mapping[str, Any] | None,
) -> None:
    if telemetry is None or not isinstance(payload, Mapping):
        return
    for key, value in payload.items():
        if isinstance(value, bool):
            telemetry[key] = bool(telemetry.get(key, False)) or value
        elif isinstance(value, int) and key.startswith("max_"):
            _telemetry_max(telemetry, key, value)
        elif isinstance(value, (int, float)):
            _telemetry_add(telemetry, key, value)
        elif value is not None:
            telemetry[key] = value


def _device_type_name(device: Any) -> str:
    device_type = getattr(device, "type", None)
    if device_type is not None:
        return str(device_type)
    return str(device).split(":", 1)[0]


def _synchronize_cuda_for_telemetry(*, torch: Any, device: Any, telemetry: MutableMapping[str, Any] | None) -> None:
    if telemetry is None or not bool(telemetry.get("synchronize_cuda_timing")):
        return
    if _device_type_name(device) != "cuda":
        return
    cuda_module = getattr(torch, "cuda", None)
    synchronize = getattr(cuda_module, "synchronize", None)
    if not callable(synchronize):
        return
    try:
        synchronize(device)
    except TypeError:
        synchronize()
    telemetry["cuda_synchronized_timing"] = True
    _telemetry_add(telemetry, "cuda_synchronization_count", 1)


def _validated_cpu_document_mapping_counts(
    torch: Any,
    mapping: Any,
    *,
    document_count: int,
) -> tuple[Any, Any, int]:
    if int(document_count) < 1:
        raise ValueError("document_count must be positive")
    if not torch.is_tensor(mapping):
        mapping = torch.tensor(mapping, dtype=torch.long)
    mapping = mapping.to(dtype=torch.long)
    if _device_type_name(getattr(mapping, "device", "cpu")) != "cpu":
        raise ValueError("document mapping validation requires CPU tokenizer output")
    if int(mapping.numel()) < int(document_count):
        raise ValueError("tokenizer returned fewer windows than documents")
    valid_mask = (mapping >= 0) & (mapping < int(document_count))
    if not bool(valid_mask.all()):
        raise ValueError("tokenizer returned window mapping outside document range")
    counts = torch.bincount(mapping, minlength=int(document_count))
    if bool((counts == 0).any()):
        raise ValueError("document has no tokenized windows")
    return mapping, counts, int(counts.max().item())


def _record_tokenized_window_shape_telemetry(
    torch: Any,
    *,
    encoded: Mapping[str, Any],
    mapping: Any,
    document_count: int,
    telemetry: MutableMapping[str, Any] | None,
) -> int:
    _mapping, _counts, max_windows_per_document = _validated_cpu_document_mapping_counts(
        torch,
        mapping,
        document_count=document_count,
    )
    if telemetry is None:
        return max_windows_per_document

    tensor_values = [value for value in encoded.values() if torch.is_tensor(value)]
    first_tensor = tensor_values[0] if tensor_values else None
    input_cell_count = int(first_tensor.numel()) if first_tensor is not None else 0
    max_sequence_length = 0
    attention_token_count = 0
    padding_cell_count = 0
    attention_mask = encoded.get("attention_mask")
    if torch.is_tensor(attention_mask):
        attention = attention_mask.to(dtype=torch.long)
        input_cell_count = int(attention.numel())
        attention_token_count = int(attention.sum().item())
        padding_cell_count = max(0, input_cell_count - attention_token_count)
        if int(attention.numel()) > 0:
            if getattr(attention, "ndim", 1) >= 2:
                max_sequence_length = int(attention.sum(dim=1).max().item())
            else:
                max_sequence_length = int(attention.sum().item())
    elif first_tensor is not None:
        attention_token_count = input_cell_count
        if getattr(first_tensor, "ndim", 1) >= 2:
            max_sequence_length = int(first_tensor.shape[-1])
        else:
            max_sequence_length = input_cell_count

    _telemetry_add(telemetry, "tokenized_input_cell_count", input_cell_count)
    _telemetry_add(telemetry, "tokenized_attention_token_count", attention_token_count)
    _telemetry_add(telemetry, "tokenized_padding_cell_count", padding_cell_count)
    _telemetry_max(telemetry, "max_tokenized_sequence_length", max_sequence_length)
    _telemetry_max(telemetry, "max_windows_per_document", max_windows_per_document)
    return max_windows_per_document


def _pin_tokenized_windows_for_cuda(
    torch: Any,
    *,
    encoded: Mapping[str, Any],
    mapping: Any,
    pin_memory: bool,
) -> tuple[dict[str, Any], Any, bool]:
    if not bool(pin_memory):
        return dict(encoded), mapping, False
    try:
        pinned_mapping = mapping.pin_memory() if hasattr(mapping, "pin_memory") else mapping
        pinned_encoded = {
            key: value.pin_memory() if torch.is_tensor(value) and hasattr(value, "pin_memory") else value
            for key, value in encoded.items()
            if torch.is_tensor(value)
        }
    except Exception:
        return dict(encoded), mapping, False
    if not pinned_encoded:
        return dict(encoded), mapping, False
    return pinned_encoded, pinned_mapping, True


def _prepare_document_logits_batch(
    *,
    torch: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    stride: int,
    pooling_mode: str,
    pooling_top_k: int,
    pin_memory: bool = False,
    metadata: Sequence[Mapping[str, Any]] | None = None,
    telemetry: MutableMapping[str, Any] | None = None,
) -> PreparedWindowMilBatch:
    _validate_pooling_config(pooling_mode=pooling_mode, pooling_top_k=pooling_top_k)
    if not texts:
        raise ValueError("prepared W1 batch requires at least one document")
    prepare_telemetry: dict[str, Any] = {}
    tokenize_at = time.monotonic()
    encoded, mapping = _tokenize_windows_cpu(
        torch=torch,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        stride=stride,
    )
    _telemetry_add(prepare_telemetry, "tokenizer_call_count", 1)
    _telemetry_add(prepare_telemetry, "tokenization_seconds", time.monotonic() - tokenize_at)
    window_count = int(mapping.numel())
    _telemetry_add(prepare_telemetry, "tokenized_window_count", window_count)
    _telemetry_max(prepare_telemetry, "max_tokenized_windows_per_document_batch", window_count)
    max_windows_per_document = _record_tokenized_window_shape_telemetry(
        torch,
        encoded=encoded,
        mapping=mapping,
        document_count=len(texts),
        telemetry=prepare_telemetry,
    )
    pin_at = time.monotonic()
    encoded, mapping, pin_memory_effective = _pin_tokenized_windows_for_cuda(
        torch,
        encoded=encoded,
        mapping=mapping,
        pin_memory=pin_memory,
    )
    if pin_memory:
        _telemetry_add(prepare_telemetry, "pin_memory_seconds", time.monotonic() - pin_at)
    prepare_telemetry["pin_memory_requested"] = bool(pin_memory)
    prepare_telemetry["pin_memory_effective"] = bool(pin_memory_effective)
    first_tensor = next((value for value in encoded.values() if torch.is_tensor(value)), None)
    padded_token_count = int(first_tensor.numel()) if first_tensor is not None else window_count
    prepared = PreparedWindowMilBatch(
        encoded=encoded,
        mapping=mapping,
        document_count=len(texts),
        window_count=window_count,
        max_windows_per_document=max_windows_per_document,
        padded_token_count=padded_token_count,
        pooling_mode=pooling_mode,
        pooling_top_k=int(pooling_top_k),
        pin_memory_requested=bool(pin_memory),
        pin_memory_effective=bool(pin_memory_effective),
        telemetry=dict(prepare_telemetry),
        metadata=tuple(dict(item) for item in (metadata or ())),
        candidate_ids=tuple(str(item.get("candidate_id_hash")) for item in (metadata or ()) if item.get("candidate_id_hash") is not None),
    )
    _telemetry_merge_prepared_payload(telemetry, prepared.telemetry)
    return prepared


def _document_logits_for_texts(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str,
    pooling_top_k: int,
    checkpoint_window_chunks: bool = False,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    _validate_pooling_config(pooling_mode=pooling_mode, pooling_top_k=pooling_top_k)
    if int(max_windows_per_device_batch) < 1:
        raise ValueError("max_windows_per_device_batch must be positive")
    prepared = _prepare_document_logits_batch(
        torch=torch,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        stride=stride,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        pin_memory=False,
        telemetry=None,
    )
    return _document_logits_from_prepared_batch(
        torch=torch,
        model=model,
        prepared=prepared,
        device=device,
        max_windows_per_device_batch=max_windows_per_device_batch,
        checkpoint_window_chunks=checkpoint_window_chunks,
        telemetry=telemetry,
    )


def _document_logits_from_prepared_batch(
    *,
    torch: Any,
    model: Any,
    prepared: PreparedWindowMilBatch,
    device: Any,
    max_windows_per_device_batch: int,
    checkpoint_window_chunks: bool = False,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    if int(max_windows_per_device_batch) < 1:
        raise ValueError("max_windows_per_device_batch must be positive")
    _validate_pooling_config(pooling_mode=prepared.pooling_mode, pooling_top_k=prepared.pooling_top_k)
    _telemetry_merge_prepared_payload(telemetry, prepared.telemetry)
    encoded = prepared.encoded
    mapping = prepared.mapping
    window_count = int(prepared.window_count)
    if window_count != int(mapping.numel()):
        raise ValueError("prepared W1 batch window count mismatch")
    if window_count > int(max_windows_per_device_batch):
        return _document_logits_from_encoded_window_chunks(
            torch=torch,
            model=model,
            encoded=encoded,
            mapping=mapping,
            document_count=int(prepared.document_count),
            device=device,
            max_windows_per_device_batch=max_windows_per_device_batch,
            pooling_mode=prepared.pooling_mode,
            pooling_top_k=int(prepared.pooling_top_k),
            checkpoint_window_chunks=checkpoint_window_chunks,
            max_windows_per_document=int(prepared.max_windows_per_document),
            telemetry=telemetry,
            non_blocking=bool(prepared.pin_memory_effective),
        )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    transfer_at = time.monotonic()
    encoded, mapping = _move_tokenized_windows_to_device(
        torch=torch,
        encoded=encoded,
        mapping=mapping,
        device=device,
        non_blocking=bool(prepared.pin_memory_effective),
    )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "device_transfer_seconds", time.monotonic() - transfer_at)
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    forward_at = time.monotonic()
    window_logits = _forward_window_logits(model, encoded)
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "window_forward_call_count", 1)
    _telemetry_add(telemetry, "window_forward_window_count", window_count)
    _telemetry_add(telemetry, "window_forward_seconds", time.monotonic() - forward_at)
    _telemetry_max(telemetry, "max_windows_per_forward_call", window_count)
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    pooling_at = time.monotonic()
    doc_logits = _pool_window_logits(
        torch,
        window_logits=window_logits,
        mapping=mapping,
        document_count=int(prepared.document_count),
        pooling_mode=prepared.pooling_mode,
        pooling_top_k=int(prepared.pooling_top_k),
        max_windows_per_document=int(prepared.max_windows_per_document),
        mapping_validated=True,
        use_ragged_topk_pooling=False,
        telemetry=telemetry,
    )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "pooling_seconds", time.monotonic() - pooling_at)
    return doc_logits


def _document_logits_from_encoded_window_chunks(
    *,
    torch: Any,
    model: Any,
    encoded: Mapping[str, Any],
    mapping: Any,
    document_count: int,
    device: Any,
    max_windows_per_device_batch: int,
    pooling_mode: str,
    pooling_top_k: int,
    checkpoint_window_chunks: bool = False,
    max_windows_per_document: int | None = None,
    telemetry: MutableMapping[str, Any] | None = None,
    non_blocking: bool = False,
) -> Any:
    window_count = int(mapping.numel())
    if window_count < 1:
        raise ValueError("tokenizer returned no windows")
    chunk_logits = []
    chunk_mappings = []
    for start in range(0, window_count, int(max_windows_per_device_batch)):
        end = min(window_count, start + int(max_windows_per_device_batch))
        encoded_chunk = {
            key: value[start:end] if torch.is_tensor(value) and int(value.shape[0]) == window_count else value
            for key, value in encoded.items()
            if torch.is_tensor(value)
        }
        _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
        transfer_at = time.monotonic()
        moved_chunk, moved_mapping = _move_tokenized_windows_to_device(
            torch=torch,
            encoded=encoded_chunk,
            mapping=mapping[start:end],
            device=device,
            non_blocking=bool(non_blocking),
        )
        _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
        _telemetry_add(telemetry, "device_transfer_seconds", time.monotonic() - transfer_at)
        _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
        forward_at = time.monotonic()
        logits = _forward_window_logits_for_training_chunk(
            torch=torch,
            model=model,
            encoded=moved_chunk,
            checkpoint_window_chunks=checkpoint_window_chunks,
        )
        _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
        chunk_logits.append(logits)
        _telemetry_add(telemetry, "window_forward_call_count", 1)
        _telemetry_add(telemetry, "window_forward_window_count", end - start)
        _telemetry_add(telemetry, "window_forward_seconds", time.monotonic() - forward_at)
        _telemetry_max(telemetry, "max_windows_per_forward_call", end - start)
        chunk_mappings.append(moved_mapping)
    if not chunk_logits:
        raise ValueError("tokenizer returned no windows")
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    pooling_at = time.monotonic()
    doc_logits = _pool_window_logits(
        torch,
        window_logits=torch.cat(chunk_logits, dim=0),
        mapping=torch.cat(chunk_mappings, dim=0),
        document_count=document_count,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        max_windows_per_document=max_windows_per_document,
        mapping_validated=True,
        use_ragged_topk_pooling=False,
        telemetry=telemetry,
    )
    _synchronize_cuda_for_telemetry(torch=torch, device=device, telemetry=telemetry)
    _telemetry_add(telemetry, "pooling_seconds", time.monotonic() - pooling_at)
    return doc_logits


def _split_halo_document_logits(
    *,
    torch: Any,
    model: Any,
    tokenizer: Any,
    core_texts: list[str],
    halo_texts: list[str],
    device: Any,
    max_length: int,
    stride: int,
    max_windows_per_device_batch: int,
    pooling_mode: str,
    pooling_top_k: int,
    halo_weight: float,
    halo_logit_clamp: float,
    checkpoint_window_chunks: bool = False,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    if len(core_texts) != len(halo_texts):
        raise ValueError("split-halo scoring inputs must have matching lengths")
    if not math.isfinite(float(halo_weight)) or float(halo_weight) < 0:
        raise ValueError("halo_weight must be a finite nonnegative value")
    if not math.isfinite(float(halo_logit_clamp)) or float(halo_logit_clamp) <= 0:
        raise ValueError("halo_logit_clamp must be a finite positive value")
    core_logits = _document_logits_for_texts(
        torch=torch,
        model=model,
        tokenizer=tokenizer,
        texts=core_texts,
        device=device,
        max_length=max_length,
        stride=stride,
        max_windows_per_device_batch=max_windows_per_device_batch,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
        checkpoint_window_chunks=checkpoint_window_chunks,
        telemetry=telemetry,
    )
    halo_logits = torch.zeros(len(core_texts), dtype=core_logits.dtype, device=device)
    halo_inputs = [(index, text) for index, text in enumerate(halo_texts) if str(text).strip()]
    if halo_inputs:
        indexes = [index for index, _text in halo_inputs]
        scored_halos = _document_logits_for_texts(
            torch=torch,
            model=model,
            tokenizer=tokenizer,
            texts=[text for _index, text in halo_inputs],
            device=device,
            max_length=max_length,
            stride=stride,
            max_windows_per_device_batch=max_windows_per_device_batch,
            pooling_mode=pooling_mode,
            pooling_top_k=pooling_top_k,
            checkpoint_window_chunks=checkpoint_window_chunks,
            telemetry=telemetry,
        )
        for position, index in enumerate(indexes):
            halo_logits[index] = scored_halos[position]
    return core_logits + float(halo_weight) * torch.clamp(
        halo_logits,
        min=-float(halo_logit_clamp),
        max=float(halo_logit_clamp),
    )


def _single_document_logit_from_window_chunks(
    *,
    torch: Any,
    model: Any,
    encoded: Mapping[str, Any],
    device: Any,
    max_windows_per_device_batch: int,
    pooling_mode: str,
    pooling_top_k: int,
    checkpoint_window_chunks: bool = False,
) -> Any:
    if max_windows_per_device_batch < 1:
        raise ValueError("max_windows_per_device_batch must be positive")
    selected_logits = []
    for chunk in _encoded_window_chunks(encoded, max_windows_per_device_batch):
        moved_chunk = {
            key: value.to(device)
            for key, value in chunk.items()
            if torch.is_tensor(value)
        }
        chunk_logits = _forward_window_logits_for_training_chunk(
            torch=torch,
            model=model,
            encoded=moved_chunk,
            checkpoint_window_chunks=checkpoint_window_chunks,
        )
        selected_logits.append(
            _select_candidate_logits_for_pooling(
                torch,
                chunk_logits,
                pooling_mode=pooling_mode,
                pooling_top_k=pooling_top_k,
            )
        )
    if not selected_logits:
        raise ValueError("single document produced no tokenized windows")
    return _pool_logits_for_one_document(
        torch,
        torch.cat(selected_logits, dim=0),
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
    )


def _forward_window_logits_for_training_chunk(
    *,
    torch: Any,
    model: Any,
    encoded: Mapping[str, Any],
    checkpoint_window_chunks: bool,
) -> Any:
    if not checkpoint_window_chunks or not torch.is_grad_enabled():
        return _forward_window_logits(model, encoded)
    try:
        from torch.utils.checkpoint import checkpoint as torch_checkpoint
    except ImportError as exc:  # pragma: no cover - torch builds normally include checkpointing.
        raise RuntimeError("torch.utils.checkpoint is required for checkpointed window-MIL training") from exc
    keys = tuple(sorted(encoded))
    tensors = tuple(encoded[key] for key in keys)
    if not tensors:
        raise ValueError("tokenizer returned no tensor inputs")

    def checkpointed_forward(*values: Any) -> Any:
        return _forward_window_logits(model, dict(zip(keys, values, strict=True)))

    return torch_checkpoint(
        checkpointed_forward,
        *tensors,
        use_reentrant=False,
        preserve_rng_state=True,
    )


def _encoded_window_chunks(encoded: Mapping[str, Any], max_windows_per_device_batch: int) -> list[dict[str, Any]]:
    tensor_values = [value for value in encoded.values() if hasattr(value, "shape")]
    if not tensor_values:
        raise ValueError("tokenizer returned no tensor inputs")
    window_count = int(tensor_values[0].shape[0])
    chunks = []
    for start in range(0, window_count, int(max_windows_per_device_batch)):
        end = min(window_count, start + int(max_windows_per_device_batch))
        chunks.append(
            {
                key: value[start:end] if hasattr(value, "shape") and int(value.shape[0]) == window_count else value
                for key, value in encoded.items()
            }
        )
    return chunks


def _tokenize_windows_cpu(
    *,
    torch: Any,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    stride: int,
) -> tuple[dict[str, Any], Any]:
    encoded = tokenizer(
        texts,
        max_length=int(max_length),
        stride=int(stride),
        truncation=True,
        padding=True,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    if "overflow_to_sample_mapping" not in encoded:
        raise ValueError("tokenizer did not return overflow_to_sample_mapping")
    mapping = encoded.pop("overflow_to_sample_mapping")
    if not torch.is_tensor(mapping):
        mapping = torch.tensor(mapping, dtype=torch.long)
    encoded_tensors = {
        key: value
        for key, value in encoded.items()
        if torch.is_tensor(value)
    }
    if not encoded_tensors:
        raise ValueError("tokenizer returned no tensor inputs")
    if int(mapping.numel()) < len(texts):
        raise ValueError("tokenizer returned fewer windows than documents")
    return encoded_tensors, mapping


def _move_tokenized_windows_to_device(
    *,
    torch: Any,
    encoded: Mapping[str, Any],
    mapping: Any,
    device: Any,
    non_blocking: bool = False,
) -> tuple[dict[str, Any], Any]:
    if not torch.is_tensor(mapping):
        mapping = torch.tensor(mapping, dtype=torch.long)
    moved_mapping = mapping.to(device, non_blocking=bool(non_blocking))
    moved_encoded = {
        key: value.to(device, non_blocking=bool(non_blocking))
        for key, value in encoded.items()
        if torch.is_tensor(value)
    }
    if not moved_encoded:
        raise ValueError("tokenizer returned no tensor inputs")
    return moved_encoded, moved_mapping


def _forward_window_logits(model: Any, encoded: Mapping[str, Any]) -> Any:
    outputs = model(**encoded)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if getattr(logits, "ndim", 1) == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if getattr(logits, "ndim", 1) != 1:
        raise ValueError("window-MIL model must return one logit per window")
    return logits


def _pool_window_logits(
    torch: Any,
    *,
    window_logits: Any,
    mapping: Any,
    document_count: int,
    pooling_mode: str = WINDOW_POOLING_MAX,
    pooling_top_k: int = 1,
    max_windows_per_document: int | None = None,
    mapping_validated: bool = False,
    use_ragged_topk_pooling: bool = False,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    _validate_pooling_config(pooling_mode=pooling_mode, pooling_top_k=pooling_top_k)
    if int(document_count) < 1:
        raise ValueError("document_count must be positive")
    if not torch.is_tensor(mapping):
        mapping = torch.tensor(mapping, dtype=torch.long, device=window_logits.device)
    else:
        if getattr(mapping, "device", None) != window_logits.device or getattr(mapping, "dtype", None) != torch.long:
            mapping = mapping.to(device=window_logits.device, dtype=torch.long)
    if int(window_logits.numel()) != int(mapping.numel()):
        raise ValueError("window logits and mapping must have matching lengths")
    if int(mapping.numel()) < int(document_count):
        raise ValueError("tokenizer returned fewer windows than documents")
    if not bool(mapping_validated):
        valid_mask = (mapping >= 0) & (mapping < int(document_count))
        if not bool(valid_mask.all()):
            mapping = mapping[valid_mask]
            window_logits = window_logits[valid_mask]
    counts = torch.bincount(mapping, minlength=int(document_count))
    if not bool(mapping_validated):
        if bool((counts == 0).any()):
            raise ValueError("document has no tokenized windows")
    if pooling_mode == WINDOW_POOLING_MAX and hasattr(torch.Tensor, "scatter_reduce_"):
        doc_logits = torch.full(
            (int(document_count),),
            -float("inf"),
            dtype=window_logits.dtype,
            device=window_logits.device,
        )
        return doc_logits.scatter_reduce_(0, mapping, window_logits, reduce="amax", include_self=True)
    if pooling_mode == WINDOW_POOLING_TOPK_MEAN:
        if bool(use_ragged_topk_pooling) and hasattr(torch.Tensor, "scatter_reduce_"):
            return _pool_window_logits_topk_mean_ragged(
                torch,
                window_logits=window_logits,
                mapping=mapping,
                counts=counts,
                document_count=int(document_count),
                pooling_top_k=int(pooling_top_k),
                max_windows_per_document=max_windows_per_document,
                telemetry=telemetry,
            )
        return _pool_window_logits_topk_mean_vectorized(
            torch,
            window_logits=window_logits,
            mapping=mapping,
            counts=counts,
            document_count=int(document_count),
            pooling_top_k=int(pooling_top_k),
            max_windows_per_document=max_windows_per_document,
            telemetry=telemetry,
        )
    return _pool_window_logits_reference(
        torch,
        window_logits=window_logits,
        mapping=mapping,
        document_count=document_count,
        pooling_mode=pooling_mode,
        pooling_top_k=pooling_top_k,
    )


def _pool_window_logits_reference(
    torch: Any,
    *,
    window_logits: Any,
    mapping: Any,
    document_count: int,
    pooling_mode: str = WINDOW_POOLING_MAX,
    pooling_top_k: int = 1,
) -> Any:
    _validate_pooling_config(pooling_mode=pooling_mode, pooling_top_k=pooling_top_k)
    doc_logits = []
    for document_index in range(int(document_count)):
        mask = mapping == document_index
        if not bool(mask.any()):
            raise ValueError("document has no tokenized windows")
        doc_logits.append(
            _pool_logits_for_one_document(
                torch,
                window_logits[mask],
                pooling_mode=pooling_mode,
                pooling_top_k=pooling_top_k,
            )
        )
    return torch.stack(doc_logits)


def _pool_window_logits_topk_mean_ragged(
    torch: Any,
    *,
    window_logits: Any,
    mapping: Any,
    counts: Any,
    document_count: int,
    pooling_top_k: int,
    max_windows_per_document: int | None = None,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    max_windows = (
        int(counts.max().detach().cpu().item())
        if max_windows_per_document is None
        else int(max_windows_per_document)
    )
    if max_windows < 1:
        raise ValueError("document has no tokenized windows")
    k = min(int(pooling_top_k), max_windows)
    window_count = int(window_logits.numel())
    _telemetry_add(telemetry, "pooling_ragged_topk_call_count", 1)
    _telemetry_add(telemetry, "pooling_ragged_topk_iteration_count", k)
    _telemetry_add(telemetry, "pooling_ragged_topk_input_window_count", window_count)
    _telemetry_add(telemetry, "pooling_ragged_topk_document_count", int(document_count))
    _telemetry_add(telemetry, "pooling_dense_padded_cell_count_avoided", int(document_count) * max_windows)
    if telemetry is not None:
        telemetry["pooling_backend"] = "ragged_topk"

    active = torch.ones((window_count,), dtype=torch.bool, device=window_logits.device)
    selected_sum = torch.zeros((int(document_count),), dtype=window_logits.dtype, device=window_logits.device)
    positions = torch.arange(window_count, dtype=torch.long, device=window_logits.device)
    sentinel_position = int(window_count)
    is_nan = torch.isnan(window_logits)
    zeros = torch.zeros_like(window_logits)

    for _iteration in range(k):
        active_nan = active & is_nan
        doc_has_nan = torch.zeros((int(document_count),), dtype=torch.long, device=window_logits.device)
        doc_has_nan.scatter_reduce_(
            0,
            mapping,
            active_nan.to(dtype=torch.long),
            reduce="amax",
            include_self=True,
        )
        use_nan_for_doc = doc_has_nan > 0
        finite_candidates = torch.where(
            active & ~is_nan,
            window_logits,
            torch.full_like(window_logits, -float("inf")),
        )
        max_per_doc = torch.full(
            (int(document_count),),
            -float("inf"),
            dtype=window_logits.dtype,
            device=window_logits.device,
        )
        max_per_doc.scatter_reduce_(0, mapping, finite_candidates, reduce="amax", include_self=True)
        winner_mask = (active_nan & use_nan_for_doc[mapping]) | (
            active
            & ~is_nan
            & ~use_nan_for_doc[mapping]
            & (window_logits == max_per_doc[mapping])
        )
        candidate_positions = torch.where(
            winner_mask,
            positions,
            torch.full_like(positions, sentinel_position),
        )
        winner_positions = torch.full(
            (int(document_count),),
            sentinel_position,
            dtype=torch.long,
            device=window_logits.device,
        )
        winner_positions.scatter_reduce_(
            0,
            mapping,
            candidate_positions,
            reduce="amin",
            include_self=True,
        )
        selected = winner_mask & (positions == winner_positions[mapping])
        selected_sum.scatter_add_(0, mapping, torch.where(selected, window_logits, zeros))
        active = active & ~selected

    selected_counts = torch.clamp(counts, max=k).to(dtype=window_logits.dtype)
    return selected_sum / selected_counts


def _pool_window_logits_topk_mean_vectorized(
    torch: Any,
    *,
    window_logits: Any,
    mapping: Any,
    counts: Any,
    document_count: int,
    pooling_top_k: int,
    max_windows_per_document: int | None = None,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    max_windows = (
        int(counts.max().detach().cpu().item())
        if max_windows_per_document is None
        else int(max_windows_per_document)
    )
    if max_windows < 1:
        raise ValueError("document has no tokenized windows")
    padded_cell_count = int(document_count) * max_windows
    if telemetry is not None:
        telemetry["pooling_backend"] = "vectorized_topk_mean"
    _telemetry_add(telemetry, "pooling_vectorized_topk_call_count", 1)
    _telemetry_add(telemetry, "pooling_vectorized_topk_input_window_count", int(window_logits.numel()))
    _telemetry_add(telemetry, "pooling_vectorized_topk_document_count", int(document_count))
    _telemetry_add(telemetry, "pooling_vectorized_padded_cell_count", padded_cell_count)
    if padded_cell_count > 1_000_000:
        return _pool_window_logits_topk_mean_padded_chunks(
            torch,
            window_logits=window_logits,
            mapping=mapping,
            counts=counts,
            document_count=document_count,
            pooling_top_k=pooling_top_k,
            max_windows_per_document=max_windows,
            telemetry=telemetry,
        )
    order = torch.argsort(mapping, stable=True)
    sorted_mapping = mapping[order]
    sorted_logits = window_logits[order]
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(int(sorted_logits.numel()), device=window_logits.device) - starts[sorted_mapping]
    padded = torch.full(
        (int(document_count), max_windows),
        -float("inf"),
        dtype=window_logits.dtype,
        device=window_logits.device,
    )
    padded[sorted_mapping, positions] = sorted_logits
    k = min(int(pooling_top_k), max_windows)
    top_values = torch.topk(padded, k=k, dim=1).values
    selected_counts = torch.clamp(counts, max=k).to(dtype=window_logits.dtype)
    selected_mask = torch.arange(k, device=window_logits.device).unsqueeze(0) < selected_counts.unsqueeze(1)
    selected = torch.where(selected_mask, top_values, torch.zeros_like(top_values))
    return selected.sum(dim=1) / selected_counts


def _pool_window_logits_topk_mean_padded_chunks(
    torch: Any,
    *,
    window_logits: Any,
    mapping: Any,
    counts: Any,
    document_count: int,
    pooling_top_k: int,
    max_windows_per_document: int,
    max_padded_cell_count: int = 1_000_000,
    telemetry: MutableMapping[str, Any] | None = None,
) -> Any:
    max_windows = int(max_windows_per_document)
    if max_windows < 1:
        raise ValueError("document has no tokenized windows")
    order = torch.argsort(mapping, stable=True)
    sorted_mapping = mapping[order]
    sorted_logits = window_logits[order]
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(int(sorted_logits.numel()), device=window_logits.device) - starts[sorted_mapping]
    k = min(int(pooling_top_k), max_windows)
    docs_per_chunk = max(1, int(max_padded_cell_count) // max_windows)
    if telemetry is not None:
        telemetry["pooling_backend"] = "vectorized_topk_mean_padded_chunks"
    _telemetry_add(telemetry, "pooling_vectorized_padded_chunk_call_count", 1)
    output = torch.empty((int(document_count),), dtype=window_logits.dtype, device=window_logits.device)
    for doc_start in range(0, int(document_count), docs_per_chunk):
        doc_end = min(int(document_count), doc_start + docs_per_chunk)
        row_mask = (sorted_mapping >= doc_start) & (sorted_mapping < doc_end)
        local_mapping = sorted_mapping[row_mask] - doc_start
        local_positions = positions[row_mask]
        local_logits = sorted_logits[row_mask]
        padded = torch.full(
            (doc_end - doc_start, max_windows),
            -float("inf"),
            dtype=window_logits.dtype,
            device=window_logits.device,
        )
        padded[local_mapping, local_positions] = local_logits
        top_values = torch.topk(padded, k=k, dim=1).values
        selected_counts = torch.clamp(counts[doc_start:doc_end], max=k).to(dtype=window_logits.dtype)
        selected_mask = torch.arange(k, device=window_logits.device).unsqueeze(0) < selected_counts.unsqueeze(1)
        selected = torch.where(selected_mask, top_values, torch.zeros_like(top_values))
        output[doc_start:doc_end] = selected.sum(dim=1) / selected_counts
    return output


def _pool_logits_for_one_document(torch: Any, logits: Any, *, pooling_mode: str, pooling_top_k: int) -> Any:
    if int(logits.numel()) < 1:
        raise ValueError("document has no tokenized windows")
    if pooling_mode == WINDOW_POOLING_MAX:
        return logits.max()
    if pooling_mode == WINDOW_POOLING_TOPK_MEAN:
        k = min(int(pooling_top_k), int(logits.numel()))
        return torch.topk(logits, k=k).values.mean()
    raise ValueError(f"unsupported pooling mode: {pooling_mode}")


def _select_candidate_logits_for_pooling(torch: Any, logits: Any, *, pooling_mode: str, pooling_top_k: int) -> Any:
    if int(logits.numel()) < 1:
        raise ValueError("document has no tokenized windows")
    if pooling_mode == WINDOW_POOLING_MAX:
        return logits.max().reshape(1)
    if pooling_mode == WINDOW_POOLING_TOPK_MEAN:
        k = min(int(pooling_top_k), int(logits.numel()))
        return torch.topk(logits, k=k).values
    raise ValueError(f"unsupported pooling mode: {pooling_mode}")


def _validate_pooling_config(*, pooling_mode: str, pooling_top_k: int) -> None:
    if pooling_mode not in SUPPORTED_WINDOW_POOLING_MODES:
        raise ValueError(f"unsupported pooling mode: {pooling_mode}")
    if int(pooling_top_k) < 1:
        raise ValueError("pooling_top_k must be positive")
    if pooling_mode == WINDOW_POOLING_MAX and int(pooling_top_k) != 1:
        raise ValueError("max pooling requires pooling_top_k=1")


def _summary_for_ints(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0, "mean": 0.0}
    ordered = sorted(int(value) for value in values)

    def percentile(fraction: float) -> int:
        index = min(len(ordered) - 1, max(0, math.ceil((len(ordered) * fraction) - 1)))
        return int(ordered[index])

    return {
        "count": len(ordered),
        "min": int(ordered[0]),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": int(ordered[-1]),
        "mean": round(float(sum(ordered)) / float(len(ordered)), 6),
    }


def _synthetic_microbench_text_lengths(*, document_count: int, max_length: int) -> list[int]:
    if int(document_count) < 1:
        raise ValueError("document_count must be positive")
    width = max(2, int(max_length) - 2)
    lengths: list[int] = []
    for index in range(int(document_count)):
        lengths.append(1 + ((index * 17) % (width * 3)))
    return lengths


def _synthetic_microbench_tokenize(
    torch: Any,
    *,
    text_lengths: Sequence[int],
    max_length: int,
    stride: int,
) -> tuple[dict[str, Any], Any, list[int]]:
    if int(max_length) < 2:
        raise ValueError("max_length must be at least 2")
    if not 0 <= int(stride) < int(max_length):
        raise ValueError("stride must be in [0, max_length)")
    step = max(1, int(max_length) - int(stride))
    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    mapping: list[int] = []
    window_lengths: list[int] = []
    for sample_index, raw_length in enumerate(text_lengths):
        token_count = max(1, int(raw_length))
        synthetic_tokens = [((sample_index + offset) % 127) + 1 for offset in range(token_count)]
        for start in range(0, token_count, step):
            window = synthetic_tokens[start : start + int(max_length)]
            if not window:
                continue
            window_length = len(window)
            input_ids.append(window + [0] * (int(max_length) - window_length))
            attention.append([1] * window_length + [0] * (int(max_length) - window_length))
            mapping.append(sample_index)
            window_lengths.append(window_length)
            if start + int(max_length) >= token_count:
                break
    encoded = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }
    return encoded, torch.tensor(mapping, dtype=torch.long), window_lengths


def _fake_forward_window_logits_for_microbench(torch: Any, encoded: Mapping[str, Any]) -> Any:
    input_ids = encoded["input_ids"].to(dtype=torch.float32)
    attention_mask = encoded["attention_mask"].to(dtype=torch.float32)
    weighted = input_ids * attention_mask
    return weighted.sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1.0)


def _microbench_rate(numerator: int | float, denominator: int | float) -> float:
    denominator_float = float(denominator)
    if denominator_float <= 0.0:
        return 0.0
    return round(float(numerator) / denominator_float, 6)


def build_window_mil_local_microbench_report(
    *,
    torch: Any | None = None,
    document_count: int = 64,
    max_length: int = 512,
    stride: int = 128,
    max_windows_per_device_batch: int = 512,
    pooling_top_k: int = 3,
    synchronize_cuda_timing: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Build a no-raw synthetic timing report for the W1 scorer hot path."""

    torch_module = torch if torch is not None else _load_torch()
    if int(max_windows_per_device_batch) < 1:
        raise ValueError("max_windows_per_device_batch must be positive")
    text_lengths = _synthetic_microbench_text_lengths(document_count=int(document_count), max_length=int(max_length))
    started_at = time.monotonic()
    tokenize_at = time.monotonic()
    encoded, mapping, window_lengths = _synthetic_microbench_tokenize(
        torch_module,
        text_lengths=text_lengths,
        max_length=int(max_length),
        stride=int(stride),
    )
    _mapping, _counts, max_windows_per_document = _validated_cpu_document_mapping_counts(
        torch_module,
        mapping,
        document_count=int(document_count),
    )
    tokenization_seconds = time.monotonic() - tokenize_at
    attention_mask = encoded.get("attention_mask")
    tokenized_input_cell_count = int(attention_mask.numel()) if attention_mask is not None else 0
    tokenized_attention_token_count = int(attention_mask.sum().item()) if attention_mask is not None else 0
    tokenized_padding_cell_count = max(0, tokenized_input_cell_count - tokenized_attention_token_count)

    resolved_device = torch_module.device(device)
    cuda_synchronization_count = 0
    cuda_synchronized_timing_observed = False
    if bool(synchronize_cuda_timing) and _device_type_name(resolved_device) == "cuda":
        cuda_module = getattr(torch_module, "cuda", None)
        synchronize = getattr(cuda_module, "synchronize", None)
        if callable(synchronize):
            try:
                synchronize(resolved_device)
            except TypeError:
                synchronize()
            cuda_synchronization_count += 1
            cuda_synchronized_timing_observed = True

    forward_at = time.monotonic()
    logits_by_chunk = []
    mapping_by_chunk = []
    window_count = int(mapping.numel())
    for start in range(0, window_count, int(max_windows_per_device_batch)):
        end = min(window_count, start + int(max_windows_per_device_batch))
        encoded_chunk = {key: value[start:end].to(resolved_device) for key, value in encoded.items()}
        logits_by_chunk.append(_fake_forward_window_logits_for_microbench(torch_module, encoded_chunk))
        mapping_by_chunk.append(mapping[start:end].to(resolved_device))
    window_logits = torch_module.cat(logits_by_chunk, dim=0)
    moved_mapping = torch_module.cat(mapping_by_chunk, dim=0)
    if cuda_synchronized_timing_observed:
        try:
            torch_module.cuda.synchronize(resolved_device)
        except TypeError:
            torch_module.cuda.synchronize()
        cuda_synchronization_count += 1
    fake_forward_seconds = time.monotonic() - forward_at

    pooling_telemetry: dict[str, Any] = {}
    pooling_at = time.monotonic()
    doc_logits = _pool_window_logits(
        torch_module,
        window_logits=window_logits,
        mapping=moved_mapping,
        document_count=int(document_count),
        pooling_mode=WINDOW_POOLING_TOPK_MEAN,
        pooling_top_k=int(pooling_top_k),
        max_windows_per_document=max_windows_per_document,
        mapping_validated=True,
        use_ragged_topk_pooling=False,
        telemetry=pooling_telemetry,
    )
    if cuda_synchronized_timing_observed:
        try:
            torch_module.cuda.synchronize(resolved_device)
        except TypeError:
            torch_module.cuda.synchronize()
        cuda_synchronization_count += 1
    pooling_seconds = time.monotonic() - pooling_at

    postprocess_at = time.monotonic()
    scores = torch_module.sigmoid(doc_logits).detach().cpu().tolist()
    postprocess_seconds = time.monotonic() - postprocess_at
    total_seconds = time.monotonic() - started_at

    phase_seconds = {
        "tokenization": round(tokenization_seconds, 6),
        "fake_forward": round(fake_forward_seconds, 6),
        "pooling": round(pooling_seconds, 6),
        "postprocess": round(postprocess_seconds, 6),
        "total": round(total_seconds, 6),
    }
    phase_candidates = {key: value for key, value in phase_seconds.items() if key != "total"}
    bottleneck_phase = max(phase_candidates, key=lambda key: float(phase_candidates[key])) if phase_candidates else None
    payload = {
        "artifact_kind": WINDOW_MIL_LOCAL_MICROBENCH_KIND,
        "schema_version": 1,
        "status": "complete",
        "claim_scope": "synthetic_local_microbench_only",
        "document_count": int(document_count),
        "max_length": int(max_length),
        "stride": int(stride),
        "max_windows_per_device_batch": int(max_windows_per_device_batch),
        "pooling_mode": WINDOW_POOLING_TOPK_MEAN,
        "pooling_top_k": int(pooling_top_k),
        "tokenized_window_count": window_count,
        "max_windows_per_document": int(max_windows_per_document),
        "tokenized_input_cell_count": tokenized_input_cell_count,
        "tokenized_attention_token_count": tokenized_attention_token_count,
        "tokenized_padding_cell_count": tokenized_padding_cell_count,
        "tokenized_padding_fraction": _microbench_rate(
            tokenized_padding_cell_count,
            tokenized_input_cell_count,
        ),
        "window_length_summary": _summary_for_ints(window_lengths),
        "synthetic_profile_hash": stable_json_hash(
            {
                "lengths": list(text_lengths),
                "max_length": int(max_length),
                "stride": int(stride),
            }
        ),
        "window_forward_call_count": len(logits_by_chunk),
        "max_windows_per_forward_call": max((int(logit.numel()) for logit in logits_by_chunk), default=0),
        "score_count": len(scores),
        "score_min": round(min(float(score) for score in scores), 12),
        "score_max": round(max(float(score) for score in scores), 12),
        "pooling_backend": str(pooling_telemetry.get("pooling_backend") or "unknown"),
        "pooling_ragged_topk_call_count": int(pooling_telemetry.get("pooling_ragged_topk_call_count", 0) or 0),
        "pooling_ragged_topk_iteration_count": int(pooling_telemetry.get("pooling_ragged_topk_iteration_count", 0) or 0),
        "pooling_ragged_topk_input_window_count": int(
            pooling_telemetry.get("pooling_ragged_topk_input_window_count", 0) or 0
        ),
        "pooling_ragged_topk_document_count": int(pooling_telemetry.get("pooling_ragged_topk_document_count", 0) or 0),
        "pooling_dense_padded_cell_count_avoided": int(
            pooling_telemetry.get("pooling_dense_padded_cell_count_avoided", 0) or 0
        ),
        "pooling_vectorized_topk_call_count": int(pooling_telemetry.get("pooling_vectorized_topk_call_count", 0) or 0),
        "pooling_vectorized_topk_input_window_count": int(
            pooling_telemetry.get("pooling_vectorized_topk_input_window_count", 0) or 0
        ),
        "pooling_vectorized_topk_document_count": int(
            pooling_telemetry.get("pooling_vectorized_topk_document_count", 0) or 0
        ),
        "pooling_vectorized_padded_cell_count": int(
            pooling_telemetry.get("pooling_vectorized_padded_cell_count", 0) or 0
        ),
        "pooling_vectorized_padded_chunk_call_count": int(
            pooling_telemetry.get("pooling_vectorized_padded_chunk_call_count", 0) or 0
        ),
        "phase_seconds": phase_seconds,
        "phase_rates": {
            "documents_per_total_second": _microbench_rate(int(document_count), total_seconds),
            "windows_per_total_second": _microbench_rate(window_count, total_seconds),
            "windows_per_fake_forward_second": _microbench_rate(window_count, fake_forward_seconds),
            "windows_per_pooling_second": _microbench_rate(window_count, pooling_seconds),
            "live_tokens_per_total_second": _microbench_rate(tokenized_attention_token_count, total_seconds),
            "padded_tokens_per_total_second": _microbench_rate(tokenized_input_cell_count, total_seconds),
        },
        "bottleneck_phase": bottleneck_phase,
        "synchronize_cuda_timing_requested": bool(synchronize_cuda_timing),
        "cuda_synchronized_timing_observed": bool(cuda_synchronized_timing_observed),
        "cuda_synchronization_count": int(cuda_synchronization_count),
        "device_type": _device_type_name(resolved_device),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["microbench_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "microbench_hash"})
    validate_public_no_raw_payload(payload)
    return payload


def _fetch_private_texts_for_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    fetch_concurrency: int,
) -> list[str]:
    candidate_by_id = _index_candidates(candidates)
    schedule_by_key = _schedule_entries_by_key(schedule)
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    for manifest_row in rows:
        candidate_id = _required_str(manifest_row, "candidate_id")
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            raise ValueError(f"candidate missing for training row: {candidate_id}")
        entry = _schedule_entry_for_candidate(candidate, schedule_by_key)
        selected.append((manifest_row, candidate, entry))

    grouped_offsets: dict[tuple[str, int], set[int]] = defaultdict(set)
    entry_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for _manifest_row, candidate, entry in selected:
        key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
        grouped_offsets[key].add(int(candidate["row_group_local_row_index"]))
        entry_by_key[key] = entry

    fetched: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    fetch_inputs = [
        (key, entry_by_key[key], sorted(grouped_offsets[key]))
        for key in sorted(grouped_offsets)
    ]
    fetch_outputs = _fetch_groups(row_provider, fetch_inputs, fetch_concurrency=fetch_concurrency)
    for key, offsets, rows_by_offset in fetch_outputs:
        for offset in offsets:
            row = rows_by_offset.get(offset)
            if isinstance(row, BaseException):
                raise ValueError(f"private row refetch failed: {type(row).__name__}")
            if not isinstance(row, Mapping):
                raise ValueError("private row provider returned a non-object row")
            fetched[(key[0], key[1], offset)] = row

    texts: list[str] = []
    for _manifest_row, candidate, entry in selected:
        key = (str(entry["source_file_id"]), int(entry["row_group_index"]), int(candidate["row_group_local_row_index"]))
        private_row = fetched[key]
        private_text = private_row.get("text")
        if not isinstance(private_text, str) or not private_text:
            raise ValueError("private row provider returned missing or empty text")
        _validate_text_hash(candidate, private_text)
        texts.append(private_text)
    return texts


def _fetch_private_texts_for_rows_with_retries(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidates: list[Mapping[str, Any]],
    schedule: Mapping[str, Any],
    row_provider: Any,
    fetch_concurrency: int,
    attempts: int = FETCH_RETRY_ATTEMPTS,
    base_sleep_seconds: float = FETCH_RETRY_BASE_SLEEP_SECONDS,
) -> list[str]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    last_error: Exception | None = None
    for attempt_index in range(int(attempts)):
        try:
            return _fetch_private_texts_for_rows(
                rows,
                candidates=candidates,
                schedule=schedule,
                row_provider=row_provider,
                fetch_concurrency=fetch_concurrency,
            )
        except Exception as exc:
            last_error = exc
            is_final_attempt = attempt_index + 1 >= int(attempts)
            if is_final_attempt or not _is_transient_fetch_error(exc):
                raise
            sleep_seconds = float(base_sleep_seconds) * (2**attempt_index)
            time.sleep(min(30.0, sleep_seconds))
    if last_error is not None:
        raise last_error
    raise RuntimeError("fetch retry loop exited unexpectedly")


def _is_transient_fetch_error(exc: Exception) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if type(current).__name__ in TRANSIENT_FETCH_ERROR_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


def _fetch_groups(
    row_provider: Any,
    fetch_inputs: list[tuple[tuple[str, int], Mapping[str, Any], list[int]]],
    *,
    fetch_concurrency: int,
) -> list[tuple[tuple[str, int], list[int], Mapping[int, Any]]]:
    if hasattr(row_provider, "fetch_row_groups"):
        outputs = []
        source_groups: dict[str, list[tuple[tuple[str, int], Mapping[str, Any], list[int]]]] = defaultdict(list)
        for key, entry, offsets in fetch_inputs:
            source_groups[_entry_source_group_key(entry)].append((key, entry, offsets))
        for source_key in sorted(source_groups):
            batch_result = row_provider.fetch_row_groups(
                [(dict(entry), offsets) for _key, entry, offsets in source_groups[source_key]]
            )
            for key, entry, offsets in source_groups[source_key]:
                result_key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
                rows_by_offset = batch_result.get(result_key)
                if not isinstance(rows_by_offset, Mapping):
                    raise ValueError("batch row provider did not return requested row group")
                outputs.append((key, offsets, rows_by_offset))
        return outputs
    if fetch_concurrency != 1:
        raise ValueError("fetch_concurrency > 1 requires a batch row provider")
    return [
        (key, offsets, row_provider.fetch_rows(dict(entry), offsets))
        for key, entry, offsets in fetch_inputs
    ]


def _validate_text_hash(candidate: Mapping[str, Any], private_text: str) -> None:
    expected = candidate.get("text_hash_sha256") or candidate.get("text_hash")
    if not expected:
        return
    algorithm = _required_str(candidate, "text_hash_algorithm")
    if algorithm == TEXT_HASH_ALGORITHM_UTF8_SHA256_V1:
        computed = hashlib.sha256(private_text.encode("utf-8", errors="surrogatepass")).hexdigest()
    elif algorithm == LEGACY_PRIVATE_MINING_TEXT_HASH_ALGORITHM:
        computed = stable_json_hash(["raw_text", private_text])
    else:
        raise ValueError(f"unsupported candidate text_hash_algorithm: {algorithm}")
    if str(expected) != computed:
        raise ValueError("private refetch text hash drift")


def _require_component(config: ClassifierPipelineConfig) -> None:
    for spec in config.model_specs:
        if spec.model_id == MODEL_ID and spec.kind == "transformer_window_mil":
            return
    raise ValueError(f"classifier config must include {MODEL_ID} as transformer_window_mil")


def _balanced_sample_weights(labels: Sequence[int]) -> dict[int, float]:
    counts = Counter(labels)
    total = len(labels)
    return {
        0: total / (2.0 * counts[0]),
        1: total / (2.0 * counts[1]),
    }


def _validate_schedule_no_raw(schedule: Mapping[str, Any]) -> None:
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if schedule.get(field) is not False:
            raise ValueError(f"schedule must set {field}=false")
    for entry in schedule.get("entries") or []:
        if entry.get("raw_text_serialized") is True or entry.get("raw_url_serialized") is True or entry.get("token_serialized") is True:
            raise ValueError("schedule entry must not serialize raw text, raw urls, or tokens")


def _index_candidates(candidates: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        candidate_id = _required_str(candidate, "candidate_id")
        if candidate_id in indexed:
            raise ValueError(f"duplicate candidate_id in candidates: {candidate_id}")
        indexed[candidate_id] = candidate
    return indexed


def _schedule_entries_by_key(schedule: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in schedule.get("entries") or []:
        source_file_id = _required_str(entry, "source_file_id")
        row_group_index = int(entry["row_group_index"])
        key = (source_file_id, row_group_index)
        if key in indexed:
            raise ValueError("schedule contains duplicate source_file_id/row_group_index entries")
        indexed[key] = entry
    return indexed


def _schedule_entry_for_candidate(
    candidate: Mapping[str, Any],
    schedule_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
) -> Mapping[str, Any]:
    source_file_id = _required_str(candidate, "source_file_id")
    row_group_index = int(candidate["row_group_index"])
    offset = int(candidate["row_group_local_row_index"])
    entry = schedule_by_key.get((source_file_id, row_group_index))
    if entry is None:
        raise ValueError("candidate physical row is not scheduled")
    offsets = {int(value) for value in entry.get("row_group_local_offsets") or []}
    if offset not in offsets:
        raise ValueError("candidate offset is not scheduled")
    if offset < 0 or offset >= int(entry["row_group_num_rows"]):
        raise ValueError("candidate offset is out of row-group bounds")
    return entry


def _entry_source_group_key(entry: Mapping[str, Any]) -> str:
    source = entry.get("parquet_url_or_path") or entry.get("source_file_id")
    if not isinstance(source, str) or not source:
        raise ValueError("schedule entry is missing source identity")
    return source


def _save_checkpoint(
    *,
    torch: Any,
    checkpoint_path: Path,
    checkpoint_meta_path: Path,
    model: Any,
    optimizer: Any,
    completed_train_batches: int,
    total_train_batches: int,
    training_data_manifest_hash: str,
    model_name: str,
    max_length: int,
    stride: int,
    batch_order: str,
    epoch_index: int,
    batch_index: int,
    checkpoint_every_batches: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_train_batches": int(completed_train_batches),
            "total_train_batches": int(total_train_batches),
        },
        checkpoint_path,
    )
    payload = {
        "artifact_kind": WINDOW_MIL_CHECKPOINT_KIND,
        "schema_version": 1,
        "completed_train_batches": int(completed_train_batches),
        "total_train_batches": int(total_train_batches),
        "training_data_manifest_hash": training_data_manifest_hash,
        "model_name_hash": stable_json_hash({"model_name": model_name}),
        "max_length": int(max_length),
        "stride": int(stride),
        "batch_order": batch_order,
        "activation_checkpoint_windows": ACTIVATION_CHECKPOINT_WINDOWS,
        "checkpoint_every_batches": int(checkpoint_every_batches),
        "epoch_index": int(epoch_index),
        "batch_index": int(batch_index),
        "checkpoint_file_name": checkpoint_path.name,
        "checkpoint_file_sha256": _sha256_file(checkpoint_path),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
    }
    payload["checkpoint_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "checkpoint_hash"})
    _write_json_atomic(checkpoint_meta_path, payload)


def _validate_checkpoint_meta(
    payload: Mapping[str, Any],
    *,
    training_data_manifest_hash: str,
    model_name: str,
    max_length: int,
    stride: int,
    batch_order: str,
) -> None:
    if payload.get("artifact_kind") != WINDOW_MIL_CHECKPOINT_KIND:
        raise ValueError("unsupported window-MIL checkpoint kind")
    validate_public_no_raw_payload(payload)
    if payload.get("training_data_manifest_hash") != training_data_manifest_hash:
        raise ValueError("window-MIL checkpoint training manifest drift")
    if payload.get("model_name_hash") != stable_json_hash({"model_name": model_name}):
        raise ValueError("window-MIL checkpoint model name drift")
    if int(payload.get("max_length") or 0) != int(max_length):
        raise ValueError("window-MIL checkpoint max_length drift")
    if int(payload.get("stride") or -1) != int(stride):
        raise ValueError("window-MIL checkpoint stride drift")
    if payload.get("batch_order") != batch_order:
        raise ValueError("window-MIL checkpoint batch order drift")
    if payload.get("activation_checkpoint_windows") is not True:
        raise ValueError("window-MIL checkpoint activation_checkpoint_windows drift")
    expected = stable_json_hash({key: value for key, value in payload.items() if key != "checkpoint_hash"})
    if payload.get("checkpoint_hash") != expected:
        raise ValueError("window-MIL checkpoint hash mismatch")


def _progress_payload(
    *,
    status: str,
    completed_train_batches: int,
    total_train_batches: int,
    fit_eligible_row_count: int,
    audit_row_count: int,
    mean_train_loss: float,
) -> dict[str, Any]:
    payload = {
        "artifact_kind": WINDOW_MIL_PROGRESS_KIND,
        "schema_version": 1,
        "status": status,
        "completed_train_batches": int(completed_train_batches),
        "total_train_batches": int(total_train_batches),
        "fit_eligible_row_count": int(fit_eligible_row_count),
        "audit_row_count": int(audit_row_count),
        "mean_train_loss": float(mean_train_loss),
        "updated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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


def _device_observations(torch: Any, device: Any) -> dict[str, Any]:
    payload = {
        "device_type": str(device.type),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        payload.update(
            {
                "cuda_device_index": int(index),
                "cuda_device_name": str(torch.cuda.get_device_name(index)),
                "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
        )
    return payload


def _model_numeric_hash(torch: Any, model: Any) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _report_hash(report: Mapping[str, Any]) -> str:
    return stable_json_hash({key: value for key, value in report.items() if key != "report_hash"})


def _validated_score_vectors(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    label: str,
) -> tuple[list[int], list[float]]:
    if len(labels) != len(scores):
        raise ValueError(f"{label} labels and scores must have the same length")
    if not labels:
        raise ValueError(f"{label} scores must be non-empty")
    normalized_labels: list[int] = []
    normalized_scores: list[float] = []
    for index, (raw_label, raw_score) in enumerate(zip(labels, scores, strict=True)):
        int_label = int(raw_label)
        if int_label not in {0, 1}:
            raise ValueError(f"{label} label at index {index} must be 0 or 1")
        score = float(raw_score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{label} score at index {index} must be finite and in [0, 1]")
        normalized_labels.append(int_label)
        normalized_scores.append(score)
    if sum(normalized_labels) == 0:
        raise ValueError(f"{label} scores must include reject examples")
    return normalized_labels, normalized_scores


def _score_ledger_rows(labels: Sequence[int], scores: Sequence[float]) -> list[dict[str, float | int]]:
    return [
        {"index": index, "label": int(label), "score": float(score)}
        for index, (label, score) in enumerate(zip(labels, scores, strict=True))
    ]


def _score_label_counts(labels: Sequence[int]) -> dict[str, int]:
    reject = sum(1 for label in labels if int(label) == 1)
    return {"reject": reject, "safe": len(labels) - reject}


def _require_sha256(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return value


def _required_str(row: Mapping[str, Any], key: str) -> str:
    if key not in row or row[key] in (None, ""):
        raise ValueError(f"missing required field {key}")
    return str(row[key])


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_row_provider(args: argparse.Namespace) -> Any:
    columns = tuple(part.strip() for part in str(args.columns).split(",") if part.strip())
    if args.row_provider == "local-parquet":
        from twc.data.decision_log import LocalParquetRowProvider

        return LocalParquetRowProvider(columns=columns or None)
    from twc.data.decision_log import HFFsspecParquetRowProvider

    return HFFsspecParquetRowProvider(
        cache_dir=args.hf_cache_dir,
        cache_delta_cap_bytes=args.max_cache_delta_bytes,
        block_size_bytes=args.block_size_bytes,
        columns=columns or ("text",),
        max_cache_blocks=args.max_cache_blocks,
        manage_hf_cache_env=False,
        row_group_batch_size=args.row_group_batch_size,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--row-provider", choices=["local-parquet", "hf-fsspec"], required=True)
    parser.add_argument("--hf-cache-dir", default="reports/classifiers/hf_window_mil_row_cache")
    parser.add_argument("--columns", default="text")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--fetch-concurrency", type=int, default=4)
    parser.add_argument("--row-group-batch-size", type=int, default=64)
    parser.add_argument("--block-size-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-cache-blocks", type=int, default=1024)
    parser.add_argument("--max-cache-delta-bytes", type=int, default=30_000_000_000)
    parser.add_argument("--max-windows-per-device-batch", type=int, default=128)
    parser.add_argument("--checkpoint-every-batches", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--target-recall", type=float, default=0.995)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--estimated-usd", type=float, default=0.0)
    parser.add_argument("--actual-usd", type=float, default=0.0)
    parser.add_argument("--compute-backend", default="prime_gpu_transformer_window_mil")
    args = parser.parse_args(argv)

    candidate_manifest = _read_json(args.candidates)
    report = train_window_mil_transformer(
        training_manifest=_read_json(args.training_manifest),
        candidates=list(candidate_manifest.get("candidates") or []),
        schedule=_read_json(args.schedule),
        row_provider=_build_row_provider(args),
        config=load_classifier_config(args.config),
        artifact_dir=args.artifact_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fetch_concurrency=args.fetch_concurrency,
        target_recall=args.target_recall,
        device=args.device,
        require_cuda=args.require_cuda,
        amp=not args.no_amp,
        max_windows_per_device_batch=args.max_windows_per_device_batch,
        checkpoint_every_batches=args.checkpoint_every_batches,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        estimated_usd=args.estimated_usd,
        actual_usd=args.actual_usd,
        compute_backend=args.compute_backend,
    )
    print(json.dumps({"status": report.get("status"), "artifact_dir": args.artifact_dir, "report_hash": report.get("report_hash")}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
