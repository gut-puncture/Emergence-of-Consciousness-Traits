"""CPU-prefilter execution and secure survivor-queue publication."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import math
import os
from pathlib import Path
import time
from typing import Any
import subprocess
import yaml

from twc.classifiers import chunk_contracts
from twc.data.decision_log import LocalParquetRowProvider
from twc.repro.hashing import stable_json_hash
from twc.repro.layout import execution_environment, find_repo_root
from twc.data.risk_ontology import RiskOntology
from twc.pretraining.filtering import artifacts as filtering_artifacts
from twc.pretraining.filtering import policy as filtering_policy
from twc.pretraining.filtering import source_prep
from twc.pretraining.filtering import stage_state_v1
from twc.pretraining.chunked_clean_run import SegmentationResult, SourceDocumentRecord, _segment_source_document_prefilter_fast, segment_source_document, validate_segmentation_result
from twc.pretraining.deterministic_filter import build_deterministic_filter_runtime, evaluate_chunk_candidate_deterministic_filter_with_runtime, validate_deterministic_filter_policy_payload
from twc.pretraining.filtering.prefilter_metrics import (
    PREFILTER_ROW_GROUP_METRICS_FILENAME,
    PREFILTER_TIMING_COUNT_FIELDS,
    PREFILTER_TIMING_SECOND_FIELDS,
    _add_prefilter_row_group_elapsed,
    _attach_prefilter_runtime_receipt_fields,
    _increment_prefilter_count,
    _load_prefilter_row_group_metrics,
    _new_prefilter_row_group_metric_state,
    _new_prefilter_runtime_metrics,
    _prefilter_row_group_metric_key,
    _prefilter_row_group_metric_payload,
    _prefilter_runtime_receipt_fields,
    _record_prefilter_elapsed,
    _write_prefilter_row_group_metrics,
)
from twc.pretraining.filtering.prefilter_records import (
    _SurvivorQueueWriteProof,
    _acquire_survivor_queue_writer_lock,
    _build_prefilter_receipt,
    _directory_size,
    _insert_prefilter_candidate,
    _is_w1_compact_prefilter_receipt,
    _iter_prefilter_survivor_queue_rows,
    _legacy_core_estimated_survivor_queue_row,
    _metadata_token_count,
    _physical_row_key,
    _prefilter_candidate_insert_record,
    _prefilter_receipt_has_survivor_queue_id_index_proof,
    _prefilter_receipt_has_survivor_queue_id_proof,
    _prefilter_segmented_result,
    _prefilter_source_row_payload,
    _prefilter_survivor_queue_rows,
    _prefilter_terminal_offsets_for_entry,
    _source_document_id_for_schedule_row,
    _sparse_scorer_metadata,
    _sparse_scorer_requires_metadata,
    _survivor_queue_id_index_base_path,
    _survivor_queue_id_index_part_path,
    _survivor_queue_id_index_ready_path,
    _survivor_queue_rows_match_live_db,
    _validate_prefilter_ready_receipt,
    _validate_prefilter_ready_receipt_payload,
    _validate_survivor_queue_id_index_ready_file,
    _validate_survivor_queue_ready_file,
    _validate_survivor_queue_row,
    _validate_survivor_queue_rows,
    _validate_w1_compact_ready_db,
    _validate_w1_compact_receipt_proof_payload,
    _w1_compact_forbidden_fresh_files,
    _write_survivor_queue_partial_streaming,
    _write_survivor_queue_ready_streaming as _write_survivor_queue_ready_streaming_impl,
)
from twc.pretraining.filtering.prefilter_supervisor import (
    DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
    PREFILTER_SUPERVISOR_MODES,
    PREFILTER_SUPERVISOR_SCHEMA_VERSION,
    PREFILTER_SUPERVISOR_THREAD_ENV,
    SEGMENTATION_VALIDATION_MODES,
    _allocate_prefilter_supervisor_diagnostic_row_caps,
    _append_prefilter_supervisor_event,
    _clear_prefilter_supervisor_owned_shard_stop,
    _expected_prefilter_supervisor_shards,
    _local_parquet_prefilter_plan_receipt_fields,
    _prefilter_supervisor_manifest_hash_payload,
    _prefilter_supervisor_run_dir_sha256,
    _prefilter_supervisor_shard_command_hash,
    _prefilter_supervisor_shard_dir,
    _prefilter_supervisor_shard_terminal_status,
    _prefilter_supervisor_summary,
    _request_prefilter_supervisor_shard_stop,
    _segmentation_sample_modulus,
    _validate_prefilter_supervisor_child_binding,
    _validate_prefilter_supervisor_ready_shard,
    _validate_segmentation_validation_mode,
    _w1_start_prefilter_expected_status,
    build_prefilter_local_parquet_plan_supervisor_manifest,
    build_prefilter_local_parquet_plan_supervisor_shard_commands,
    validate_prefilter_local_parquet_plan_supervisor_manifest,
)
from twc.pretraining.filtering.prefilter_merge import (
    PREFILTER_MERGE_SPEC_FILENAME,
    _prefilter_merged_plan_receipt_fields,
    _prefilter_supervisor_merge_source_snapshot,
    _prefilter_supervisor_merge_spec,
    _prepare_prefilter_merge_incoming,
    _promote_prefilter_merge_directory,
    _validate_existing_prefilter_supervisor_merge,
    _validate_prefilter_merge_directory,
    _validate_prefilter_merge_spec_file,
)
from twc.pretraining.filtering.prefilter_handoff import (
    _elapsed_summary,
    _validate_sharded_w1_handoff_payload,
)

def _segment_for_prefilter(
    source: SourceDocumentRecord,
    *,
    validation_mode: str,
    sample_modulus: int,
    metrics: dict[str, Any] | None = None,
) -> SegmentationResult:
    mode = _validate_segmentation_validation_mode(validation_mode)
    modulus = _segmentation_sample_modulus(sample_modulus)
    if mode == "full":
        return segment_source_document(source)
    result = _segment_source_document_prefilter_fast(source)
    sampled_count = 0 if metrics is None else int(metrics["counts"].get("segmentation_sampled_validations", 0))
    if sampled_count == 0 or int(source.source_document_id_hash[:16], 16) % modulus == 0:
        validate_segmentation_result(result)
        if metrics is not None:
            _increment_prefilter_count(metrics, "segmentation_sampled_validations")
    return result


def _write_survivor_queue_ready_streaming(
    base_path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    stage_state: stage_state_v1.FilteringStageStateV1,
    config_payload: Mapping[str, Any],
) -> _SurvivorQueueWriteProof:
    """Preserve the historical prefilter monkeypatch seam for live-row checks."""

    return _write_survivor_queue_ready_streaming_impl(
        base_path,
        rows,
        stage_state=stage_state,
        config_payload=config_payload,
        _iter_live_rows=_iter_prefilter_survivor_queue_rows,
    )

def run_prefilter_cpu_stage_for_local_parquet_plan(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    row_provider: Any,
    sparse_scorer: chunk_contracts.ChunkFilteringScorer,
    run_dir: str | Path,
    deterministic_policy_payload: Mapping[str, Any],
    deterministic_ontology: RiskOntology,
    max_rows: int | None = None,
    cache_dir: str | Path | None = None,
    min_free_bytes: int = 1,
    row_group_shard_count: int = 1,
    row_group_shard_index: int = 0,
    row_group_sharding_policy: str = source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL,
    supervisor_manifest_hash: str | None = None,
    supervisor_command_hash: str | None = None,
    supervisor_run_dir_sha256: str | None = None,
    supervisor_manifest_path: str | Path | None = None,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> dict[str, Any]:
    filtering_policy.validate_chunk_filtering_config_payload(config_payload)
    if sparse_scorer.model_id != chunk_contracts.CHUNK_SPARSE_MODEL_ID:
        raise ValueError("prefilter-cpu local all-row mode requires the S1 sparse scorer only")
    if not isinstance(row_provider, LocalParquetRowProvider):
        raise ValueError("local Parquet prefilter requires LocalParquetRowProvider")
    if config_payload.get("deterministic_filter_policy_hash") != deterministic_policy_payload.get(
        "deterministic_filter_policy_hash"
    ):
        raise ValueError("prefilter-cpu deterministic filter policy hash mismatch")
    validate_deterministic_filter_policy_payload(deterministic_policy_payload)
    deterministic_runtime = build_deterministic_filter_runtime(deterministic_ontology, deterministic_policy_payload)
    sparse_scorer_metadata_required = _sparse_scorer_requires_metadata(sparse_scorer)
    segment_mode = _validate_segmentation_validation_mode(segmentation_validation_mode)
    segment_sample_modulus = _segmentation_sample_modulus(segmentation_validation_sample_modulus)
    plan = dict(prefilter_plan)
    source_prep._validate_local_parquet_prefilter_plan(plan)
    if plan["source_metadata_hash"] != config_payload["source_metadata_hash"]:
        raise ValueError("prefilter-cpu local Parquet plan source metadata hash mismatch")
    if plan["artifact_kind"] == filtering_artifacts.LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND and max_rows is not None:
        raise ValueError("token tranche prefilter does not support max_rows; build a smaller tranche plan")
    row_cap = None if max_rows is None else filtering_artifacts._positive_int(max_rows, "max_rows")
    row_group_fetch_batch_size = max(1, int(getattr(row_provider, "row_group_batch_size", 1)))
    shard_policy = source_prep._validate_row_group_sharding_policy(row_group_sharding_policy)
    supervisor_binding_values = (
        supervisor_manifest_hash,
        supervisor_command_hash,
        supervisor_run_dir_sha256,
        supervisor_manifest_path,
    )
    supervisor_receipt_fields: dict[str, Any] = {}
    if any(value is not None for value in supervisor_binding_values):
        if not all(value is not None for value in supervisor_binding_values):
            raise ValueError("prefilter supervisor receipt binding requires manifest path, manifest, command, and run-dir hashes")
        supervisor_receipt_fields = _validate_prefilter_supervisor_child_binding(
            prefilter_plan=plan,
            config_payload=config_payload,
            supervisor_manifest_path=supervisor_manifest_path,
            supervisor_manifest_hash=supervisor_manifest_hash,
            supervisor_command_hash=supervisor_command_hash,
            supervisor_run_dir_sha256=supervisor_run_dir_sha256,
            row_group_shard_count=row_group_shard_count,
            row_group_shard_index=row_group_shard_index,
            row_group_batch_size=row_group_fetch_batch_size,
            max_rows=row_cap,
            min_free_bytes=min_free_bytes,
            row_group_sharding_policy=shard_policy,
            segmentation_validation_mode=segment_mode,
            segmentation_validation_sample_modulus=segment_sample_modulus,
        )
    plan = source_prep._shard_local_parquet_prefilter_plan(
        plan,
        shard_count=row_group_shard_count,
        shard_index=row_group_shard_index,
        row_group_sharding_policy=shard_policy,
    )
    expected_rows = int(plan["expected_physical_rows"] if row_cap is None else min(row_cap, int(plan["expected_physical_rows"])))
    diagnostic_receipt_fields = (
        {
            "row_cap_applied": True,
            "max_rows": int(row_cap),
            "claims_scope": "throughput_diagnostic_only",
            "merge_allowed": False,
        }
        if row_cap is not None
        else {}
    )
    segmentation_receipt_fields = {
        "segmentation_validation_mode": segment_mode,
        "segmentation_validation_sample_modulus": segment_sample_modulus,
    }
    run_path = Path(run_dir)
    filtering_artifacts._ensure_private_dir(run_path)
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        filtering_artifacts._ensure_private_cache_dir(cache_path)
    filtering_artifacts._check_disk_watermark([run_path, *([cache_path] if cache_path is not None else [])], min_free_bytes=min_free_bytes)
    conn = stage_state_v1.create_or_open_prefilter_state(run_path, config_payload)
    conn.bind_prefilter_schedule(
        str(plan["schedule_hash"]),
        expected_rows,
        None,
    )
    start = time.monotonic()
    runtime_metrics = _new_prefilter_runtime_metrics()
    row_group_metrics_by_id = {
        metric_id: row
        for metric_id, row in _load_prefilter_row_group_metrics(run_path).items()
        if row.get("input_schedule_hash", row.get("source_schedule_hash")) == plan["schedule_hash"]
    }
    stop_requested = False
    rows_accounted = int(conn.prefilter_summary_facts()["terminal_rows"])
    try:
        entries = list(plan["entries"])
        entry_index = 0
        while entry_index < len(entries):
            fetch_batch: list[tuple[dict[str, Any], list[int]]] = []
            pending_batch_rows = 0
            while entry_index < len(entries) and len(fetch_batch) < row_group_fetch_batch_size:
                iteration_start = time.monotonic()
                if row_cap is not None and rows_accounted + pending_batch_rows >= row_cap:
                    _record_prefilter_elapsed(runtime_metrics, "row_group_iteration", iteration_start)
                    entry_index = len(entries)
                    break
                _increment_prefilter_count(runtime_metrics, "row_groups_considered")
                entry = dict(entries[entry_index])
                entry["source_schedule_hash"] = str(plan["schedule_hash"])
                entry["row_group_metric_source_schedule_hash"] = str(
                    plan.get("local_parquet_prefilter_plan_hash") or plan["schedule_hash"]
                )
                num_rows = int(entry["row_group_num_rows"])
                candidate_offsets: list[int] = []
                for offset in range(num_rows):
                    if row_cap is not None and rows_accounted + pending_batch_rows + len(candidate_offsets) >= row_cap:
                        break
                    candidate_offsets.append(offset)
                terminal_offsets = _prefilter_terminal_offsets_for_entry(conn, entry, candidate_offsets)
                rows_accounted += len(terminal_offsets)
                pending_offsets = [offset for offset in candidate_offsets if offset not in terminal_offsets]
                entry_index += 1
                if not pending_offsets:
                    _record_prefilter_elapsed(runtime_metrics, "row_group_iteration", iteration_start)
                    continue
                if (run_path / "STOP_REQUESTED").exists():
                    stop_requested = True
                    _record_prefilter_elapsed(runtime_metrics, "row_group_iteration", iteration_start)
                    break
                fetch_batch.append((entry, pending_offsets))
                pending_batch_rows += len(pending_offsets)
                _record_prefilter_elapsed(runtime_metrics, "row_group_iteration", iteration_start)
            if stop_requested:
                break
            if not fetch_batch:
                continue
            batch_group_metrics: dict[tuple[str, int], dict[str, Any]] = {}
            raw_group_to_metric_key: dict[tuple[str, int], tuple[str, int]] = {}
            for entry, _pending_offsets in fetch_batch:
                raw_group = (str(entry["source_file_id"]), int(entry["row_group_index"]))
                metric_key = _prefilter_row_group_metric_key(entry)
                raw_group_to_metric_key[raw_group] = metric_key
                batch_group_metrics[metric_key] = _new_prefilter_row_group_metric_state(entry)
            filtering_artifacts._check_disk_watermark([run_path, *([cache_path] if cache_path is not None else [])], min_free_bytes=min_free_bytes)
            fetch_start = time.monotonic()
            fetch_error_row_payloads: list[dict[str, Any]] = []
            fetch_error_group_keys: set[tuple[str, int]] = set()
            try:
                if hasattr(row_provider, "fetch_row_groups"):
                    fetched_by_group = row_provider.fetch_row_groups(
                        [(dict(entry), list(pending_offsets)) for entry, pending_offsets in fetch_batch]
                    )
                else:
                    fetched_by_group = {}
                    for entry, pending_offsets in fetch_batch:
                        fetched_by_group[(str(entry["source_file_id"]), int(entry["row_group_index"]))] = row_provider.fetch_rows(
                            dict(entry),
                            pending_offsets,
                        )
            except Exception as exc:
                fetched_by_group = {}
                for entry, pending_offsets in fetch_batch:
                    try:
                        fetched_by_group[(str(entry["source_file_id"]), int(entry["row_group_index"]))] = row_provider.fetch_rows(
                            dict(entry),
                            list(pending_offsets),
                        )
                    except Exception as row_group_exc:
                        failed_key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
                        fetch_error_group_keys.add(failed_key)
                        fetch_error_row_payloads.extend(
                            [
                                _prefilter_source_row_payload(
                                    entry=entry,
                                    offset=offset,
                                    row_status="fetch_error",
                                    error_class=type(row_group_exc).__name__,
                                    segmentation_receipt_hash=None,
                                    candidate_count=0,
                                    survivor_count=0,
                                )
                                for offset in pending_offsets
                            ]
                        )
                if not fetched_by_group and not fetch_error_row_payloads:
                    fetch_error_row_payloads = [
                        _prefilter_source_row_payload(
                            entry=entry,
                            offset=offset,
                            row_status="fetch_error",
                            error_class=type(exc).__name__,
                            segmentation_receipt_hash=None,
                            candidate_count=0,
                            survivor_count=0,
                        )
                        for entry, pending_offsets in fetch_batch
                        for offset in pending_offsets
                    ]
            fetch_elapsed = _record_prefilter_elapsed(runtime_metrics, "parquet_fetch", fetch_start)
            for state in batch_group_metrics.values():
                _add_prefilter_row_group_elapsed(
                    state,
                    "parquet_fetch",
                    fetch_elapsed / max(1, len(batch_group_metrics)),
                )
            for raw_group, rows in fetched_by_group.items():
                metric_key = raw_group_to_metric_key.get(raw_group)
                if metric_key in batch_group_metrics:
                    batch_group_metrics[metric_key]["rows_fetched"] = len(rows)
            _increment_prefilter_count(runtime_metrics, "row_groups_fetched", len(fetched_by_group))
            _increment_prefilter_count(runtime_metrics, "rows_fetched", sum(len(rows) for rows in fetched_by_group.values()))
            row_payloads: list[dict[str, Any]] = list(fetch_error_row_payloads)
            candidate_records: list[tuple[tuple[Any, ...], tuple[Any, ...] | None]] = []
            scorer_items: list[tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any], str]] = []
            scorer_texts: list[str] = []
            scorer_metadata: list[dict[str, Any]] = []
            row_infos: list[dict[str, Any]] = []
            for entry, pending_offsets in fetch_batch:
                group_key = (str(entry["source_file_id"]), int(entry["row_group_index"]))
                row_group_metric_key = raw_group_to_metric_key.get(group_key)
                row_group_metric = (
                    batch_group_metrics.get(row_group_metric_key)
                    if row_group_metric_key is not None
                    else None
                )
                if group_key not in fetched_by_group:
                    if group_key not in fetch_error_group_keys:
                        row_payloads.extend(
                            _prefilter_source_row_payload(
                                entry=entry,
                                offset=offset,
                                row_status="fetch_error",
                                error_class="missing_row_group_fetch_result",
                                segmentation_receipt_hash=None,
                                candidate_count=0,
                                survivor_count=0,
                            )
                            for offset in pending_offsets
                        )
                    continue
                fetched = fetched_by_group[group_key]
                for offset in pending_offsets:
                    row = fetched.get(offset)
                    if isinstance(row, BaseException):
                        row_payloads.append(
                            _prefilter_source_row_payload(
                                entry=entry,
                                offset=offset,
                                row_status="fetch_error",
                                error_class=type(row).__name__,
                                segmentation_receipt_hash=None,
                                candidate_count=0,
                                survivor_count=0,
                            )
                        )
                        continue
                    if not isinstance(row, Mapping):
                        row_payloads.append(
                            _prefilter_source_row_payload(
                                entry=entry,
                                offset=offset,
                                row_status="non_mapping_row",
                                error_class="non_mapping_row",
                                segmentation_receipt_hash=None,
                                candidate_count=0,
                                survivor_count=0,
                            )
                        )
                        continue
                    text = row.get("text")
                    if not isinstance(text, str):
                        segmentation_start = time.monotonic()
                        result = _segment_for_prefilter(
                            SourceDocumentRecord(
                                source_document_id_hash=_source_document_id_for_schedule_row(entry, offset),
                                split="train",
                                raw_text=text,
                                source_lineage_hash=_physical_row_key(entry, offset),
                                source_token_count_hint=_metadata_token_count(row),
                            ),
                            validation_mode=segment_mode,
                            sample_modulus=segment_sample_modulus,
                            metrics=runtime_metrics,
                        )
                        segment_elapsed = _record_prefilter_elapsed(runtime_metrics, "segmentation", segmentation_start)
                        if row_group_metric is not None:
                            _add_prefilter_row_group_elapsed(row_group_metric, "segmentation", segment_elapsed)
                        row_payloads.append(
                            _prefilter_source_row_payload(
                                entry=entry,
                                offset=offset,
                                row_status="missing_text",
                                error_class=str(result.receipt.document_status),
                                segmentation_receipt_hash=result.receipt.receipt_hash,
                                candidate_count=0,
                                survivor_count=0,
                            )
                        )
                        continue
                    segmentation_start = time.monotonic()
                    result = _segment_for_prefilter(
                        SourceDocumentRecord(
                            source_document_id_hash=_source_document_id_for_schedule_row(entry, offset),
                            split="train",
                            raw_text=text,
                            source_lineage_hash=_physical_row_key(entry, offset),
                            source_token_count_hint=_metadata_token_count(row),
                        ),
                        validation_mode=segment_mode,
                        sample_modulus=segment_sample_modulus,
                        metrics=runtime_metrics,
                    )
                    segment_elapsed = _record_prefilter_elapsed(runtime_metrics, "segmentation", segmentation_start)
                    if row_group_metric is not None:
                        _add_prefilter_row_group_elapsed(row_group_metric, "segmentation", segment_elapsed)
                    if result.receipt.document_status != "segmented":
                        row_payloads.append(
                            _prefilter_source_row_payload(
                                entry=entry,
                                offset=offset,
                                row_status="segmentation_error",
                                error_class=str(result.receipt.document_status),
                                segmentation_receipt_hash=result.receipt.receipt_hash,
                                candidate_count=0,
                                survivor_count=0,
                            )
                        )
                        continue
                    candidate_start = time.monotonic()
                    candidates = filtering_policy.build_filtering_candidates(
                        result=result,
                        raw_text=text,
                        config_hash=str(config_payload["config_hash"]),
                        target_proxy_tokens=int(config_payload["target_proxy_tokens"]),
                        halo_chars=int(config_payload["halo_chars"]),
                        deterministic_policy_hash=str(deterministic_policy_payload["deterministic_filter_policy_hash"]),
                    )
                    candidate_elapsed = _record_prefilter_elapsed(runtime_metrics, "candidate_construction", candidate_start)
                    if row_group_metric is not None:
                        _add_prefilter_row_group_elapsed(
                            row_group_metric,
                            "candidate_construction",
                            candidate_elapsed,
                        )
                    _increment_prefilter_count(runtime_metrics, "candidates_built", len(candidates))
                    row_info = {
                        "entry": entry,
                        "offset": offset,
                        "row_group_metric_key": row_group_metric_key,
                        "segmentation_receipt_hash": result.receipt.receipt_hash,
                        "candidate_count": len(candidates),
                        "survivor_count": 0,
                    }
                    row_infos.append(row_info)
                    deterministic_normalization_cache: dict[tuple[Any, ...], Any] = {}
                    for candidate in candidates:
                        deterministic_start = time.monotonic()
                        deterministic_result = evaluate_chunk_candidate_deterministic_filter_with_runtime(
                            candidate,
                            raw_text=text,
                            runtime=deterministic_runtime,
                            normalization_cache=deterministic_normalization_cache,
                        )
                        deterministic_elapsed = _record_prefilter_elapsed(
                            runtime_metrics,
                            "deterministic_filter",
                            deterministic_start,
                        )
                        if row_group_metric is not None:
                            _add_prefilter_row_group_elapsed(
                                row_group_metric,
                                "deterministic_filter",
                                deterministic_elapsed,
                            )
                        _increment_prefilter_count(runtime_metrics, "deterministic_candidates")
                        deterministic_decision = filtering_policy._deterministic_candidate_decision(candidate)
                        if deterministic_result.get("terminal_decision") == "reject":
                            if row_group_metric is not None:
                                row_group_metric["route_counts"]["deterministic_reject"] += 1
                            candidate_records.append(
                                _prefilter_candidate_insert_record(
                                    candidate,
                                    prefilter_route="deterministic_reject",
                                    deterministic_result=deterministic_result,
                                    s1_score=None,
                                    private_evidence_text=None,
                                )
                            )
                            continue
                        if deterministic_decision == "quarantine":
                            if row_group_metric is not None:
                                row_group_metric["route_counts"]["prefilter_quarantine"] += 1
                            candidate_records.append(
                                _prefilter_candidate_insert_record(
                                    candidate,
                                    prefilter_route="prefilter_quarantine",
                                    deterministic_result=deterministic_result,
                                    s1_score=None,
                                    private_evidence_text=None,
                                )
                            )
                            continue
                        evidence_text = text[int(candidate["evidence_char_start"]) : int(candidate["evidence_char_end"])]
                        scorer_items.append((row_info, candidate, deterministic_result, evidence_text))
                        scorer_texts.append(evidence_text)
                        if sparse_scorer_metadata_required:
                            scorer_metadata.append(_sparse_scorer_metadata(sparse_scorer, candidate))
            if scorer_items:
                s1_start = time.monotonic()
                scores = list(
                    sparse_scorer.score_many(
                        scorer_texts,
                        scorer_metadata if sparse_scorer_metadata_required else [],
                    )
                )
                s1_elapsed = _record_prefilter_elapsed(runtime_metrics, "s1_feature_score", s1_start)
                _increment_prefilter_count(runtime_metrics, "s1_scored_candidates", len(scorer_items))
                s1_group_counts = Counter(
                    row_info.get("row_group_metric_key")
                    for row_info, _candidate, _deterministic_result, _evidence_text in scorer_items
                    if row_info.get("row_group_metric_key") is not None
                )
                for metric_key, group_count in s1_group_counts.items():
                    state = batch_group_metrics.get(metric_key)
                    if state is not None:
                        _add_prefilter_row_group_elapsed(
                            state,
                            "s1_feature_score",
                            s1_elapsed * (int(group_count) / max(1, len(scorer_items))),
                        )
                if len(scores) != len(scorer_items):
                    scores = [math.nan] * len(scorer_items)
                for (row_info, candidate, deterministic_result, evidence_text), score in zip(
                    scorer_items, scores, strict=True
                ):
                    metric_key = row_info.get("row_group_metric_key")
                    row_group_metric = batch_group_metrics.get(metric_key) if metric_key is not None else None
                    parsed = chunk_contracts.coerce_optional_risk_score(score)
                    if parsed is None:
                        route = "prefilter_error"
                    elif parsed >= float(config_payload["sparse_threshold"]):
                        route = "s1_reject"
                    else:
                        route = "prefilter_survivor"
                        row_info["survivor_count"] = int(row_info["survivor_count"]) + 1
                    if row_group_metric is not None:
                        row_group_metric["route_counts"][route] += 1
                    candidate_records.append(
                        _prefilter_candidate_insert_record(
                            candidate,
                            prefilter_route=route,
                            deterministic_result=deterministic_result,
                            s1_score=parsed,
                            private_evidence_text=evidence_text if route == "prefilter_survivor" else None,
                        )
                    )
            row_payloads.extend(
                _prefilter_source_row_payload(
                    entry=row_info["entry"],
                    offset=int(row_info["offset"]),
                    row_status="prefiltered",
                    error_class=None,
                    segmentation_receipt_hash=str(row_info["segmentation_receipt_hash"]),
                    candidate_count=int(row_info["candidate_count"]),
                    survivor_count=int(row_info["survivor_count"]),
                    )
                for row_info in row_infos
            )
            for row_payload in row_payloads:
                metric_key = (str(row_payload["source_file_id_hash"]), int(row_payload["row_group_index"]))
                state = batch_group_metrics.get(metric_key)
                if state is None:
                    continue
                state["terminal_rows"] = int(state.get("terminal_rows", 0)) + 1
                state["candidate_count"] = int(state.get("candidate_count", 0)) + int(row_payload["candidate_count"])
                state["survivor_count"] = int(state.get("survivor_count", 0)) + int(row_payload["survivor_count"])
            sqlite_start = time.monotonic()
            conn.commit_prefilter_batch(row_payloads, candidate_records)
            sqlite_elapsed = _record_prefilter_elapsed(runtime_metrics, "sqlite_write", sqlite_start)
            terminal_rows_in_batch = sum(int(state.get("terminal_rows", 0)) for state in batch_group_metrics.values())
            for state in batch_group_metrics.values():
                share = int(state.get("terminal_rows", 0)) / max(1, terminal_rows_in_batch)
                _add_prefilter_row_group_elapsed(state, "sqlite_write", sqlite_elapsed * share)
                metric_payload = _prefilter_row_group_metric_payload(state)
                row_group_metrics_by_id[str(metric_payload["row_group_metric_id"])] = metric_payload
            rows_accounted += pending_batch_rows
            filtering_artifacts._check_disk_watermark([run_path, *([cache_path] if cache_path is not None else [])], min_free_bytes=min_free_bytes)
            if (run_path / "STOP_REQUESTED").exists():
                stop_requested = True
                break
        queue_path = run_path / "w1_survivor_queue.jsonl"
        if stop_requested:
            queue_start = time.monotonic()
            _write_survivor_queue_partial_streaming(
                queue_path,
                _iter_prefilter_survivor_queue_rows(conn, config_payload),
            )
            _record_prefilter_elapsed(runtime_metrics, "survivor_queue_export", queue_start)
            row_group_metric_fields = _write_prefilter_row_group_metrics(run_path, row_group_metrics_by_id)
            receipt_start = time.monotonic()
            report = _build_prefilter_receipt(
                conn,
                config_payload,
                run_path=run_path,
                status="stopped",
                stop_reason="stop_requested",
                elapsed_seconds=time.monotonic() - start,
                cache_dir=cache_path,
                min_free_bytes=min_free_bytes,
                extra_fields=_local_parquet_prefilter_plan_receipt_fields(plan)
                | supervisor_receipt_fields
                | diagnostic_receipt_fields
                | segmentation_receipt_fields
                | row_group_metric_fields,
            )
            _record_prefilter_elapsed(runtime_metrics, "receipt_build", receipt_start)
            report = _attach_prefilter_runtime_receipt_fields(report, runtime_metrics)
        else:
            final_row_counts = conn.prefilter_summary_facts()
            blocked_by_errors = bool(final_row_counts["row_error_counts"])
            queue_proof = None
            if blocked_by_errors or row_cap is not None:
                queue_hash = None
                queue_start = time.monotonic()
                _write_survivor_queue_partial_streaming(
                    queue_path,
                    _iter_prefilter_survivor_queue_rows(conn, config_payload),
                )
                _record_prefilter_elapsed(runtime_metrics, "survivor_queue_export", queue_start)
            else:
                queue_start = time.monotonic()
                queue_proof = _write_survivor_queue_ready_streaming(
                    queue_path,
                    _iter_prefilter_survivor_queue_rows(conn, config_payload),
                    stage_state=conn,
                    config_payload=config_payload,
                )
                queue_hash = queue_proof.survivor_queue_hash
                _record_prefilter_elapsed(runtime_metrics, "survivor_queue_export", queue_start)
            if row_cap is not None:
                ready_status = "prefilter_diagnostic_complete"
            else:
                ready_status = (
                    "prefilter_shard_ready_for_merge"
                    if int(plan.get("row_group_shard_count", 1)) > 1
                    else "prefilter_ready_for_w1"
                )
            row_group_metric_fields = _write_prefilter_row_group_metrics(run_path, row_group_metrics_by_id)
            receipt_start = time.monotonic()
            report = _build_prefilter_receipt(
                conn,
                config_payload,
                run_path=run_path,
                status="prefilter_blocked_errors" if blocked_by_errors else ready_status,
                stop_reason=(
                    "prefilter_row_errors"
                    if blocked_by_errors
                    else "diagnostic_row_cap_reached"
                    if row_cap is not None
                    else "prefilter_complete"
                ),
                elapsed_seconds=time.monotonic() - start,
                cache_dir=cache_path,
                min_free_bytes=min_free_bytes,
                survivor_queue_ready=not blocked_by_errors and row_cap is None,
                survivor_queue_part=blocked_by_errors or row_cap is not None,
                survivor_queue_hash=queue_hash,
                survivor_queue_row_count=None if queue_proof is None else queue_proof.survivor_queue_row_count,
                survivor_queue_id_index_hash=None if queue_proof is None else queue_proof.survivor_queue_id_index_hash,
                survivor_queue_id_index_row_count=None
                if queue_proof is None
                else queue_proof.survivor_queue_id_index_row_count,
                survivor_candidate_id_set_hash=None if queue_proof is None else queue_proof.survivor_candidate_id_set_hash,
                survivor_private_evidence_id_set_hash=None if queue_proof is None else queue_proof.survivor_private_evidence_id_set_hash,
                extra_fields=_local_parquet_prefilter_plan_receipt_fields(plan)
                | supervisor_receipt_fields
                | diagnostic_receipt_fields
                | segmentation_receipt_fields
                | row_group_metric_fields,
            )
            _record_prefilter_elapsed(runtime_metrics, "receipt_build", receipt_start)
            report = _attach_prefilter_runtime_receipt_fields(report, runtime_metrics)
        filtering_artifacts._write_json_atomic(run_path / "prefilter_cpu_receipt.json", report)
        return report
    finally:
        conn.close()

def run_prefilter_local_parquet_plan_supervisor(
    *,
    config_path: str | Path,
    selected_policy_report_path: str | Path,
    plan_path: str | Path,
    supervisor_run_dir: str | Path,
    cache_dir: str | Path,
    sparse_report_path: str | Path,
    sparse_model_path: str | Path,
    deterministic_ontology_path: str | Path,
    shard_count: int,
    worker_count: int,
    row_group_batch_size: int,
    manifest_mode: str,
    max_rows: int | None = None,
    min_free_bytes: int = 1,
    row_group_sharding_policy: str = source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL,
    merge_run_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
    poll_interval_seconds: float = 1.0,
    segmentation_validation_mode: str = "full",
    segmentation_validation_sample_modulus: int = DEFAULT_SEGMENTATION_VALIDATION_SAMPLE_MODULUS,
) -> dict[str, Any]:
    config_payload = filtering_policy.build_chunk_filtering_config_payload(_load_yaml(config_path))
    filtering_policy.validate_locked_chunk_filtering_config_payload(config_payload)
    _load_and_validate_selected_policy_report(selected_policy_report_path, config_payload, required=True)
    plan = filtering_artifacts._load_json(plan_path)
    supervisor_path = Path(supervisor_run_dir)
    filtering_artifacts._ensure_private_dir(supervisor_path)
    filtering_artifacts.validate_private_source_cache_dir(cache_dir)
    filtering_artifacts._check_disk_watermark([supervisor_path, Path(cache_dir)], min_free_bytes=min_free_bytes)
    manifest = build_prefilter_local_parquet_plan_supervisor_manifest(
        prefilter_plan=plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_path,
        shard_count=shard_count,
        worker_count=worker_count,
        row_group_batch_size=row_group_batch_size,
        manifest_mode=manifest_mode,
        max_rows=max_rows,
        min_free_bytes=min_free_bytes,
        row_group_sharding_policy=row_group_sharding_policy,
        segmentation_validation_mode=segmentation_validation_mode,
        segmentation_validation_sample_modulus=segmentation_validation_sample_modulus,
    )
    manifest_path = supervisor_path / "prefilter_supervisor_manifest.json"
    if manifest_path.exists():
        existing = filtering_artifacts._load_json(manifest_path)
        if existing != manifest:
            raise ValueError("prefilter supervisor manifest drift; resume requires the exact original manifest")
        manifest = existing
    else:
        filtering_artifacts._write_json_atomic(manifest_path, manifest)
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_path,
    )
    if merge_run_dir is not None:
        if not bool(manifest["merge_allowed"]):
            raise ValueError("prefilter supervisor manifest is not mergeable")
        merge_path = Path(merge_run_dir)
        if not merge_path.exists() and not merge_path.is_symlink():
            filtering_artifacts._check_disk_watermark([merge_path], min_free_bytes=min_free_bytes)
    commands = build_prefilter_local_parquet_plan_supervisor_shard_commands(
        manifest=manifest,
        supervisor_run_dir=supervisor_path,
        config_path=config_path,
        selected_policy_report_path=selected_policy_report_path,
        plan_path=plan_path,
        cache_dir=cache_dir,
        sparse_report_path=sparse_report_path,
        sparse_model_path=sparse_model_path,
        deterministic_ontology_path=deterministic_ontology_path,
        python_executable=python_executable,
    )
    events_path = supervisor_path / "prefilter_supervisor_events.jsonl"
    summary_path = supervisor_path / "prefilter_supervisor_summary.json"
    stop_path = supervisor_path / "STOP_REQUESTED"
    if not stop_path.exists():
        for spec in commands:
            _clear_prefilter_supervisor_owned_shard_stop(spec["shard_run_dir"])
    start = time.monotonic()
    launched = 0
    skipped = 0
    shard_statuses: dict[int, str] = {}
    pending: list[dict[str, Any]] = []
    shards_by_index = {int(shard["shard_index"]): shard for shard in manifest["shards"]}
    command_shard_indexes = {int(spec["shard_index"]) for spec in commands}
    for shard in shards_by_index.values():
        shard_index = int(shard["shard_index"])
        if shard_index not in command_shard_indexes:
            shard_statuses[shard_index] = "not_launched_zero_budget"
            _append_prefilter_supervisor_event(
                events_path,
                {
                    "event_type": "shard_skipped_zero_budget",
                    "manifest_hash": manifest["manifest_hash"],
                    "shard_index": shard_index,
                    "command_hash": str(shard["command_hash"]),
                },
            )
    for spec in commands:
        status = _prefilter_supervisor_shard_terminal_status(spec["shard_run_dir"])
        if status == "ready":
            shard = shards_by_index[int(spec["shard_index"])]
            _validate_prefilter_supervisor_ready_shard(
                shard_run_dir=spec["shard_run_dir"],
                shard=shard,
                manifest=manifest,
                prefilter_plan=plan,
                config_payload=config_payload,
            )
            skipped += 1
            shard_statuses[int(spec["shard_index"])] = "ready"
            _append_prefilter_supervisor_event(
                events_path,
                {
                    "event_type": "shard_skipped_ready",
                    "manifest_hash": manifest["manifest_hash"],
                    "shard_index": int(spec["shard_index"]),
                    "command_hash": spec["command_hash"],
                },
            )
        else:
            pending.append(spec)
    running: dict[subprocess.Popen[Any], tuple[dict[str, Any], Any]] = {}
    stop_requested = False
    failure_seen = False
    code_root = find_repo_root(__file__)
    try:
        while pending or running:
            if stop_path.exists():
                stop_requested = True
                for spec, _handle in running.values():
                    _request_prefilter_supervisor_shard_stop(spec["shard_run_dir"])
            while (
                not stop_requested
                and not failure_seen
                and pending
                and len(running) < int(manifest["worker_count"])
            ):
                spec = pending.pop(0)
                shard_dir = Path(spec["shard_run_dir"])
                filtering_artifacts._ensure_private_dir(shard_dir)
                log_path = shard_dir / "prefilter_shard_stdout_stderr.log"
                log_handle = log_path.open("a", encoding="utf-8")
                filtering_artifacts._chmod_private_file(log_path)
                env = os.environ.copy()
                env.update({str(key): str(value) for key, value in spec["env_overrides"].items()})
                env = execution_environment(
                    code_root,
                    code_identity_policy_version=3,
                    base=env,
                )
                try:
                    process = subprocess.Popen(
                        spec["command"],
                        cwd=code_root,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        text=True,
                    )
                except BaseException:
                    log_handle.close()
                    shard_statuses[int(spec["shard_index"])] = "launch_failed"
                    raise
                running[process] = (spec, log_handle)
                launched += 1
                _append_prefilter_supervisor_event(
                    events_path,
                    {
                        "event_type": "shard_launched",
                        "manifest_hash": manifest["manifest_hash"],
                        "shard_index": int(spec["shard_index"]),
                        "command_hash": spec["command_hash"],
                    },
                )
            completed: list[subprocess.Popen[Any]] = []
            for process, (spec, log_handle) in list(running.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                completed.append(process)
                log_handle.close()
                status = _prefilter_supervisor_shard_terminal_status(spec["shard_run_dir"])
                if status == "ready":
                    try:
                        shard = shards_by_index[int(spec["shard_index"])]
                        _validate_prefilter_supervisor_ready_shard(
                            shard_run_dir=spec["shard_run_dir"],
                            shard=shard,
                            manifest=manifest,
                            prefilter_plan=plan,
                            config_payload=config_payload,
                        )
                    except ValueError:
                        status = "failed"
                if int(return_code) != 0 and status != "ready":
                    status = "failed"
                if status == "stopped":
                    stop_requested = True
                elif status != "ready":
                    failure_seen = True
                shard_statuses[int(spec["shard_index"])] = status
                _append_prefilter_supervisor_event(
                    events_path,
                    {
                        "event_type": "shard_finished",
                        "manifest_hash": manifest["manifest_hash"],
                        "shard_index": int(spec["shard_index"]),
                        "command_hash": spec["command_hash"],
                        "return_code": int(return_code),
                        "shard_status": status,
                    },
                )
            for process in completed:
                running.pop(process, None)
            if failure_seen:
                for spec, _handle in running.values():
                    _request_prefilter_supervisor_shard_stop(spec["shard_run_dir"])
            if running and not completed:
                time.sleep(max(0.05, float(poll_interval_seconds)))
            if stop_requested and not running:
                break
        if stop_requested:
            for spec in pending:
                shard_statuses[int(spec["shard_index"])] = "not_launched_stop_requested"
            status = "supervisor_stopped"
            merge_receipt_hash = None
        elif failure_seen:
            for spec in pending:
                shard_statuses[int(spec["shard_index"])] = "not_launched_after_failure"
            status = "supervisor_failed"
            merge_receipt_hash = None
        else:
            for spec in commands:
                shard_index = int(spec["shard_index"])
                if shard_index not in shard_statuses:
                    status = _prefilter_supervisor_shard_terminal_status(spec["shard_run_dir"])
                    if status == "ready":
                        shard = shards_by_index[shard_index]
                        _validate_prefilter_supervisor_ready_shard(
                            shard_run_dir=spec["shard_run_dir"],
                            shard=shard,
                            manifest=manifest,
                            prefilter_plan=plan,
                            config_payload=config_payload,
                        )
                    shard_statuses[shard_index] = status
            successful_statuses = {"ready", "not_launched_zero_budget"}
            if all(value in successful_statuses for value in shard_statuses.values()):
                if manifest["manifest_mode"] == "diagnostic_sweep":
                    status = "supervisor_diagnostic_complete"
                    merge_receipt_hash = None
                elif merge_run_dir is not None:
                    merged = merge_prefilter_local_parquet_plan_supervisor_manifest(
                        prefilter_plan=plan,
                        config_payload=config_payload,
                        manifest=manifest,
                        supervisor_run_dir=supervisor_path,
                        run_dir=merge_run_dir,
                        min_free_bytes=min_free_bytes,
                    )
                    status = "supervisor_merged"
                    merge_receipt_hash = str(merged["stage_receipt_hash"])
                else:
                    status = "supervisor_ready_for_merge"
                    merge_receipt_hash = None
            else:
                status = "supervisor_failed"
                merge_receipt_hash = None
        counts = Counter(shard_statuses.values())
        summary = _prefilter_supervisor_summary(
            manifest=manifest,
            status=status,
            elapsed_seconds=time.monotonic() - start,
            shard_status_counts=counts,
            launched_shards=launched,
            skipped_ready_shards=skipped,
            merge_receipt_hash=merge_receipt_hash,
        )
        filtering_artifacts._write_json_atomic(summary_path, summary)
        return summary
    except BaseException:
        for spec in pending:
            shard_statuses[int(spec["shard_index"])] = "not_launched_after_exception"
        for process, (spec, log_handle) in list(running.items()):
            _request_prefilter_supervisor_shard_stop(spec["shard_run_dir"])
            shard_statuses[int(spec["shard_index"])] = "stop_requested_after_exception"
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass
            try:
                log_handle.close()
            except Exception:
                pass
        counts = Counter(shard_statuses.values())
        summary = _prefilter_supervisor_summary(
            manifest=manifest,
            status="supervisor_failed_exception",
            elapsed_seconds=time.monotonic() - start,
            shard_status_counts=counts,
            launched_shards=launched,
            skipped_ready_shards=skipped,
            merge_receipt_hash=None,
        )
        try:
            filtering_artifacts._write_json_atomic(summary_path, summary)
            _append_prefilter_supervisor_event(
                events_path,
                {
                    "event_type": "supervisor_exception",
                    "manifest_hash": manifest["manifest_hash"],
                    "summary_hash": summary["summary_hash"],
                },
            )
        except Exception:
            pass
        raise
    finally:
        for _process, (_spec, log_handle) in list(running.items()):
            try:
                log_handle.close()
            except Exception:
                pass

def merge_prefilter_local_parquet_plan_supervisor_manifest(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    run_dir: str | Path,
    min_free_bytes: int = 1,
) -> dict[str, Any]:
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=prefilter_plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
    )
    if not bool(manifest.get("merge_allowed")):
        raise ValueError("prefilter supervisor manifest is not mergeable")
    shard_dirs = []
    shard_snapshots: list[dict[str, Any]] = []
    for shard in manifest["shards"]:
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
        receipt = _validate_prefilter_supervisor_ready_shard(
            shard_run_dir=shard_dir,
            shard=shard,
            manifest=manifest,
            prefilter_plan=prefilter_plan,
            config_payload=config_payload,
        )
        shard_dirs.append(shard_dir)
        shard_snapshots.append(
            _prefilter_supervisor_merge_source_snapshot(
                shard_dir=shard_dir,
                shard_index=int(shard["shard_index"]),
                receipt=receipt,
            )
        )
    merge_spec = _prefilter_supervisor_merge_spec(
        prefilter_plan=prefilter_plan,
        config_payload=config_payload,
        manifest=manifest,
        shard_snapshots=shard_snapshots,
    )
    final_path = Path(run_dir)
    filtering_artifacts.validate_private_filtering_run_dir(final_path)
    incoming_path = Path(str(final_path) + ".incoming")
    lock_path = Path(str(final_path) + ".merge.lock")
    lock_fd = _acquire_survivor_queue_writer_lock(lock_path)
    try:
        if final_path.exists() or final_path.is_symlink():
            if final_path.is_symlink() or not final_path.is_dir():
                raise ValueError("prefilter merge final path is not a trusted directory")
            if incoming_path.exists() or incoming_path.is_symlink():
                raise ValueError("prefilter merge has both final and incoming directories")
            _validate_prefilter_merge_directory(final_path)
            _validate_prefilter_merge_spec_file(final_path, merge_spec)
            return _validate_existing_prefilter_supervisor_merge(
                prefilter_plan=prefilter_plan,
                config_payload=config_payload,
                manifest=manifest,
                supervisor_run_dir=supervisor_run_dir,
                run_dir=final_path,
            )

        _prepare_prefilter_merge_incoming(incoming_path, merge_spec)
        report = merge_prefilter_local_parquet_plan_shards(
            prefilter_plan=prefilter_plan,
            config_payload=config_payload,
            shard_run_dirs=shard_dirs,
            run_dir=incoming_path,
            min_free_bytes=min_free_bytes,
            row_group_sharding_policy=source_prep._validate_row_group_sharding_policy(
                manifest.get(
                    "row_group_sharding_policy",
                    source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL,
                )
            ),
            _merge_spec=merge_spec,
        )
        validated = _validate_existing_prefilter_supervisor_merge(
            prefilter_plan=prefilter_plan,
            config_payload=config_payload,
            manifest=manifest,
            supervisor_run_dir=supervisor_run_dir,
            run_dir=incoming_path,
        )
        if validated != report:
            raise ValueError("prefilter merge receipt changed during validation")
        current_snapshots = [
            _prefilter_supervisor_merge_source_snapshot(
                shard_dir=shard_dir,
                shard_index=int(shard["shard_index"]),
                receipt=_validate_prefilter_supervisor_ready_shard(
                    shard_run_dir=shard_dir,
                    shard=shard,
                    manifest=manifest,
                    prefilter_plan=prefilter_plan,
                    config_payload=config_payload,
                ),
            )
            for shard, shard_dir in zip(manifest["shards"], shard_dirs, strict=True)
        ]
        if current_snapshots != shard_snapshots:
            raise ValueError("prefilter merge source shard changed during merge")
        _validate_prefilter_merge_spec_file(incoming_path, merge_spec)
        _promote_prefilter_merge_directory(incoming_path, final_path)
        return report
    finally:
        os.close(lock_fd)


def merge_prefilter_local_parquet_all_row_shards(
    *,
    local_schedule: Mapping[str, Any],
    source_metadata_manifest: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    shard_run_dirs: Sequence[str | Path],
    run_dir: str | Path,
    min_free_bytes: int = 1,
) -> dict[str, Any]:
    plan = source_prep.build_local_parquet_all_rows_prefilter_plan(
        local_schedule=local_schedule,
        source_metadata_manifest=source_metadata_manifest,
    )
    return merge_prefilter_local_parquet_plan_shards(
        prefilter_plan=plan,
        config_payload=config_payload,
        shard_run_dirs=shard_run_dirs,
        run_dir=run_dir,
        min_free_bytes=min_free_bytes,
    )

def merge_prefilter_local_parquet_plan_shards(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    shard_run_dirs: Sequence[str | Path],
    run_dir: str | Path,
    min_free_bytes: int = 1,
    row_group_sharding_policy: str = source_prep.ROW_GROUP_SHARDING_POLICY_ORDINAL,
    _merge_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    filtering_policy.validate_chunk_filtering_config_payload(config_payload)
    plan = dict(prefilter_plan)
    source_prep._validate_local_parquet_prefilter_plan(plan)
    shard_policy = source_prep._validate_row_group_sharding_policy(row_group_sharding_policy)
    if plan["source_metadata_hash"] != config_payload["source_metadata_hash"]:
        raise ValueError("prefilter shard merge source metadata hash mismatch")
    shard_paths = [Path(path) for path in shard_run_dirs]
    if not shard_paths:
        raise ValueError("complete shard set is required for prefilter merge")
    run_path = Path(run_dir)
    existing_names = set() if not run_path.exists() else {path.name for path in run_path.iterdir()}
    allowed_names = {PREFILTER_MERGE_SPEC_FILENAME} if _merge_spec is not None else set()
    if existing_names != allowed_names:
        raise ValueError("merged prefilter run dir must be fresh and empty")
    if _merge_spec is not None:
        _validate_prefilter_merge_spec_file(run_path, _merge_spec)
    if any(path.resolve(strict=False) == run_path.resolve(strict=False) for path in shard_paths):
        raise ValueError("merged prefilter run dir must be separate from shard run dirs")
    filtering_artifacts._ensure_private_dir(run_path)
    filtering_artifacts._check_disk_watermark([run_path], min_free_bytes=min_free_bytes)
    start = time.monotonic()
    shard_count = len(shard_paths)
    expected_indexes = set(range(shard_count))
    receipts_by_index: dict[int, dict[str, Any]] = {}
    shard_conns: list[stage_state_v1.FilteringStageStateV1] = []
    try:
        for shard_path in shard_paths:
            filtering_artifacts.validate_private_filtering_run_dir(shard_path)
            queue_row_count = _validate_survivor_queue_ready_file(
                shard_path / "w1_survivor_queue.jsonl",
                config_payload,
                allow_empty=True,
            )
            conn = stage_state_v1.open_stage_state_readonly(shard_path, config_payload)
            shard_conns.append(conn)
            _validate_prefilter_ready_receipt(
                conn,
                shard_path,
                config_payload,
                queue_row_count=queue_row_count,
                expected_status="prefilter_shard_ready_for_merge",
            )
            receipt = filtering_artifacts._load_json(shard_path / "prefilter_cpu_receipt.json")
            if not isinstance(receipt, Mapping):
                raise ValueError("prefilter shard receipt must be an object")
            filtering_artifacts.validate_public_no_raw_payload(receipt)
            declared_count = int(receipt.get("row_group_shard_count", 1))
            if declared_count != shard_count:
                raise ValueError("complete shard set is required for prefilter merge")
            shard_index = int(receipt.get("row_group_shard_index", -1))
            if shard_index not in expected_indexes:
                raise ValueError("prefilter shard index is outside the complete shard set")
            if shard_index in receipts_by_index:
                raise ValueError("prefilter shard merge received duplicate shard index")
            expected_shard_plan = source_prep._shard_local_parquet_prefilter_plan(
                plan,
                shard_count=shard_count,
                shard_index=shard_index,
                row_group_sharding_policy=shard_policy,
            )
            expected_fields = _local_parquet_prefilter_plan_receipt_fields(expected_shard_plan)
            for field, expected_value in expected_fields.items():
                if receipt.get(field) != expected_value:
                    raise ValueError(f"prefilter shard receipt {field} mismatch")
            if receipt.get("input_schedule_hash") != expected_shard_plan["schedule_hash"]:
                raise ValueError("prefilter shard input schedule hash mismatch")
            if int(receipt.get("terminal_rows", -1)) != int(expected_shard_plan["expected_physical_rows"]):
                raise ValueError("prefilter shard terminal row count mismatch")
            receipts_by_index[shard_index] = dict(receipt)
        if set(receipts_by_index) != expected_indexes:
            raise ValueError("complete shard set is required for prefilter merge")
        if sum(int(item["expected_physical_rows"]) for item in receipts_by_index.values()) != int(
            plan["expected_physical_rows"]
        ):
            raise ValueError("prefilter shard expected rows do not cover the all-row plan")
        if sum(int(item["row_group_shard_row_group_count"]) for item in receipts_by_index.values()) != int(
            plan["row_group_count"]
        ):
            raise ValueError("prefilter shard row groups do not cover the all-row plan")

        merged = stage_state_v1.create_or_open_prefilter_state(run_path, config_payload)
        try:
            merged.bind_prefilter_schedule(
                str(plan["schedule_hash"]),
                int(plan["expected_physical_rows"]),
                None,
            )
            merged.merge_prefilter_shards(shard_conns)
            queue_proof = _write_survivor_queue_ready_streaming(
                run_path / "w1_survivor_queue.jsonl",
                _iter_prefilter_survivor_queue_rows(merged, config_payload),
                stage_state=merged,
                config_payload=config_payload,
            )
            queue_hash = queue_proof.survivor_queue_hash
            queue_count = queue_proof.survivor_queue_row_count
            if queue_count != sum(int(item.get("survivor_count", 0)) for item in receipts_by_index.values()):
                raise ValueError("merged prefilter survivor queue row count mismatch")
            extra_fields = _prefilter_merged_plan_receipt_fields(plan, receipts_by_index)
            report = _build_prefilter_receipt(
                merged,
                config_payload,
                run_path=run_path,
                status="prefilter_ready_for_w1",
                stop_reason="prefilter_shards_merged",
                elapsed_seconds=time.monotonic() - start,
                cache_dir=None,
                min_free_bytes=min_free_bytes,
                survivor_queue_ready=True,
                survivor_queue_part=False,
                survivor_queue_hash=queue_hash,
                survivor_queue_row_count=queue_proof.survivor_queue_row_count,
                survivor_queue_id_index_hash=queue_proof.survivor_queue_id_index_hash,
                survivor_queue_id_index_row_count=queue_proof.survivor_queue_id_index_row_count,
                survivor_candidate_id_set_hash=queue_proof.survivor_candidate_id_set_hash,
                survivor_private_evidence_id_set_hash=queue_proof.survivor_private_evidence_id_set_hash,
                extra_fields=extra_fields,
            )
            filtering_artifacts._write_json_atomic(run_path / "prefilter_cpu_receipt.json", report)
            return report
        finally:
            merged.close()
    finally:
        for conn in shard_conns:
            conn.close()

def build_sharded_w1_handoff(
    *,
    prefilter_plan: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
    supervisor_run_dir: str | Path,
    handoff_path: str | Path | None = None,
    full_audit: bool = False,
) -> dict[str, Any]:
    filtering_policy.validate_chunk_filtering_config_payload(config_payload)
    plan = dict(prefilter_plan)
    validate_prefilter_local_parquet_plan_supervisor_manifest(
        manifest,
        prefilter_plan=plan,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
        expect_merge_allowed=True,
    )
    seen_candidates: set[str] | None = set()
    seen_evidence: set[str] | None = set()
    route_counts: Counter[str] = Counter()
    timing_sums: Counter[str] = Counter()
    shard_payloads: list[dict[str, Any]] = []
    elapsed_values: list[float] = []
    candidate_count = 0
    survivor_count = 0
    terminal_rows = 0
    prefilter_producer_config_hashes: set[str] = set()
    for shard in sorted(manifest["shards"], key=lambda item: int(item["shard_index"])):
        shard_dir = _prefilter_supervisor_shard_dir(supervisor_run_dir, shard)
        receipt = _validate_prefilter_supervisor_ready_shard(
            shard_run_dir=shard_dir,
            shard=shard,
            manifest=manifest,
            prefilter_plan=plan,
            config_payload=config_payload,
            seen_survivor_candidate_ids=seen_candidates,
            seen_private_evidence_ids=seen_evidence,
            audit_sqlite_state=full_audit,
        )
        prefilter_producer_config_hashes.add(
            filtering_artifacts._require_sha256(receipt.get("config_hash"), "sharded W1 handoff shard producer config_hash")
        )
        if full_audit:
            shard_prefilter_config_payload = filtering_policy._prefilter_input_config_payload_for_hash(
                config_payload,
                receipt.get("config_hash"),
                label="sharded W1 handoff shard prefilter receipt",
            )
            queue_rows = filtering_artifacts._load_ready_jsonl(shard_dir / "w1_survivor_queue.jsonl", "prefilter survivor queue")
            _validate_survivor_queue_rows(queue_rows, shard_prefilter_config_payload, allow_empty=True)
            conn = stage_state_v1.open_stage_state_readonly(
                shard_dir,
                shard_prefilter_config_payload,
                legacy_retained_config_payload=config_payload,
            )
            try:
                receipt = filtering_artifacts._load_json(shard_dir / "prefilter_cpu_receipt.json")
                if not isinstance(receipt, Mapping):
                    raise ValueError("sharded W1 handoff shard prefilter receipt must be an object")
                _validate_prefilter_ready_receipt(
                    conn,
                    shard_dir,
                    shard_prefilter_config_payload,
                    queue_rows=queue_rows,
                    expected_status=_w1_start_prefilter_expected_status(receipt),
                )
                if not _survivor_queue_rows_match_live_db(
                    queue_rows,
                    _prefilter_survivor_queue_rows(conn, shard_prefilter_config_payload),
                ):
                    raise ValueError("sharded W1 handoff survivor queue export does not match SQLite state")
            finally:
                conn.close()
        shard_route_counts = {str(key): int(value) for key, value in dict(receipt.get("route_counts") or {}).items()}
        route_counts.update(shard_route_counts)
        for field in PREFILTER_TIMING_SECOND_FIELDS:
            timing_sums[field] += float(receipt.get(f"prefilter_{field}_seconds", 0.0) or 0.0)
        candidate_count += int(receipt.get("candidate_count", 0))
        survivor_count += int(receipt.get("survivor_count", 0))
        terminal_rows += int(receipt.get("terminal_rows", 0))
        elapsed_values.append(float(receipt.get("elapsed_seconds", 0.0) or 0.0))
        has_queue_proof = _prefilter_receipt_has_survivor_queue_id_proof(receipt)
        shard_payloads.append(
            {
                "shard_index": int(shard["shard_index"]),
                "shard_run_dir_name": str(shard["shard_run_dir_name"]),
                "expected_physical_rows": int(shard["expected_physical_rows"]),
                "terminal_rows": int(receipt.get("terminal_rows", 0)),
                "candidate_count": int(receipt.get("candidate_count", 0)),
                "survivor_count": int(receipt.get("survivor_count", 0)),
                "route_counts": shard_route_counts,
                "elapsed_seconds": round(float(receipt.get("elapsed_seconds", 0.0) or 0.0), 6),
                "prefilter_stage_receipt_hash": receipt["stage_receipt_hash"],
                "survivor_queue_hash": receipt["survivor_queue_hash"],
                "survivor_queue_row_count": int(receipt["survivor_queue_row_count"]) if has_queue_proof else None,
                "survivor_queue_id_index_hash": receipt.get("survivor_queue_id_index_hash") if has_queue_proof else None,
                "survivor_queue_id_index_row_count": (
                    int(receipt["survivor_queue_id_index_row_count"])
                    if has_queue_proof and receipt.get("survivor_queue_id_index_row_count") is not None
                    else None
                ),
                "survivor_candidate_id_set_hash": receipt.get("survivor_candidate_id_set_hash") if has_queue_proof else None,
                "survivor_private_evidence_id_set_hash": receipt.get("survivor_private_evidence_id_set_hash") if has_queue_proof else None,
            }
        )
    if len(prefilter_producer_config_hashes) != 1:
        raise ValueError("sharded W1 handoff shards have inconsistent producer config hashes")
    prefilter_config_payload = filtering_policy._prefilter_input_config_payload_for_hash(
        config_payload,
        next(iter(prefilter_producer_config_hashes)),
        label="sharded W1 handoff shard producer",
    )
    payload = {
        "artifact_kind": filtering_artifacts.SHARDED_W1_HANDOFF_KIND,
        "schema_version": 1,
        "status": "sharded_prefilter_ready_for_w1",
        "next_required_phase": "w1-score-sharded",
        "prefilter_plan_hash": plan["schedule_hash"],
        "prefilter_plan_kind": plan["artifact_kind"],
        "supervisor_manifest_hash": manifest["manifest_hash"],
        "supervisor_run_dir_sha256": manifest["supervisor_run_dir_sha256"],
        "config_hash": prefilter_config_payload["config_hash"],
        "selected_policy_hash": prefilter_config_payload["selected_policy_hash"],
        "selected_policy_report_hash": prefilter_config_payload["selected_policy_report_hash"],
        "source_metadata_hash": prefilter_config_payload["source_metadata_hash"],
        "deterministic_filter_policy_hash": prefilter_config_payload["deterministic_filter_policy_hash"],
        "sparse_report_hash": prefilter_config_payload["sparse_report_hash"],
        "sparse_model_file_sha256": prefilter_config_payload["sparse_model_file_sha256"],
        "w1_report_hash": prefilter_config_payload["window_mil_report_hash"],
        "w1_model_file_sha256": prefilter_config_payload["window_mil_model_file_sha256"],
        "shard_count": int(manifest["shard_count"]),
        "expected_physical_rows": int(manifest["expected_physical_rows"]),
        "terminal_rows": int(terminal_rows),
        "candidate_count": int(candidate_count),
        "survivor_count": int(survivor_count),
        "route_counts": filtering_artifacts._sorted_counter(route_counts),
        "prefilter_timing_seconds": {field: round(float(timing_sums[field]), 6) for field in PREFILTER_TIMING_SECOND_FIELDS},
        "shard_elapsed_seconds_summary": _elapsed_summary(elapsed_values),
        "shards": shard_payloads,
        "accepted_manifest_created": False,
        "accepted_manifest": False,
        "tokenizer_shards_created": False,
        "packed_shards_created": False,
        "training_shards_created": False,
        "base_pretraining_started": False,
        "pretraining_started": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["handoff_hash"] = stable_json_hash({key: value for key, value in payload.items() if key != "handoff_hash"})
    _validate_sharded_w1_handoff_payload(
        payload,
        config_payload=config_payload,
        supervisor_run_dir=supervisor_run_dir,
        manifest=manifest,
        require_compact_w1_scores_empty=True,
    )
    if handoff_path is not None:
        filtering_artifacts._write_json_atomic(handoff_path, payload)
    return payload

def _load_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def _load_and_validate_selected_policy_report(
    path: str | Path | None,
    config_payload: Mapping[str, Any],
    *,
    required: bool,
) -> dict[str, Any] | None:
    if path is None:
        if required:
            raise ValueError("selected policy freeze report is required for production filtering stages")
        return None
    report = filtering_artifacts._load_json(path)
    if not isinstance(report, Mapping):
        raise ValueError("selected policy freeze report must be a JSON object")
    filtering_policy._validate_selected_policy_freeze_report(report, config_payload)
    return dict(report)
