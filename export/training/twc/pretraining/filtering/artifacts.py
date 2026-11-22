"""Filtering artifact contracts and safe publication primitives.

This module owns artifact identities, no-raw validation, private-path rules,
durable JSON/JSONL publication, and read-only W1 artifact validation. It has no
dependency on scoring engines, offline workflows, controllers, CLIs, or raw
corpus data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from twc.classifiers import chunk_contracts
from twc.repro.public_artifacts import (
    validate_public_no_raw_payload as _validate_classifier_public_no_raw_payload,
)
from twc.repro.hashing import (
    sha256_file as _sha256_file,
    stable_json_hash,
    self_hash_mismatches,
)
from twc.repro.layout import is_public_repo_root
from twc.repro.values import coerce_lowercase_sha256


CHUNK_FILTERING_CONFIG_KIND = "twc_chunk_filtering_config"
CHUNK_FILTER_DOCUMENT_REPORT_KIND = "twc_chunk_filter_document_report"
SELECTED_POLICY_FREEZE_REPORT_KIND = "twc_chunk_classifier_freeze_v2_eval_report"
LOCAL_PARQUET_EXECUTION_SCHEDULE_KIND = "twc_private_local_parquet_execution_schedule_v1"
LOCAL_PARQUET_MIRROR_REPORT_KIND = "twc_private_local_parquet_mirror_report_v1"
LOCAL_PARQUET_ALL_ROWS_PREFILTER_PLAN_KIND = "twc_local_parquet_all_rows_prefilter_plan_v1"
LOCAL_PARQUET_TOKEN_TARGET_TRANCHE_PLAN_KIND = "twc_local_parquet_token_target_tranche_plan_v1"
PREFILTER_SUPERVISOR_MANIFEST_KIND = "twc_local_parquet_prefilter_supervisor_manifest_v1"
PREFILTER_SUPERVISOR_SUMMARY_KIND = "twc_local_parquet_prefilter_supervisor_summary_v1"
PREFILTER_CPU_STAGE_KIND = "twc_chunk_filter_prefilter_cpu_stage_receipt_v1"
W1_SCORE_STAGE_KIND = "twc_chunk_filter_w1_score_stage_receipt_v1"
W1_RUNTIME_EXECUTION_ENVELOPE_KIND = "twc_chunk_filter_w1_runtime_execution_envelope_v1"
W1_ACCEPTED_TOKEN_LEDGER_KIND = "twc_w1_accepted_token_ledger_v1"
W1_COMPACT_START_STATE_KIND = "twc_w1_compact_start_state_v1"
ASSEMBLED_STAGE_REPORT_KIND = "twc_chunk_filter_assembled_report_v1"
SHARDED_W1_HANDOFF_KIND = "twc_local_parquet_sharded_prefilter_w1_handoff_v1"
SHARDED_W1_SCORE_SUMMARY_KIND = "twc_chunk_filter_sharded_w1_score_summary_v1"
W1_PRIVATE_SCORE_STATE_ARCHIVE_KIND = "twc_w1_private_score_state_archive_v1"
CLOSEOUT_SOURCE_MANIFEST_KIND = "twc_filtering_closeout_source_manifest_v1"
CLOSEOUT_REPORT_KIND = "twc_w1_preserved_selection_closeout_report_v1"
CLOSEOUT_UNIT_RECEIPT_KIND = "twc_filtering_closeout_unit_receipt_v1"
CLOSEOUT_PUBLICATION_SET_KIND = "twc_filtering_closeout_publication_set_v1"

PUBLIC_FALSE_FLAGS = (
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "accepted_manifest_created",
    "accepted_manifest",
    "tokenizer_shards_created",
    "packed_shards_created",
    "training_shards_created",
    "base_pretraining_started",
    "pretraining_started",
)

_PRIVATE_FILE_MODE = 0o600


def validate_public_no_raw_payload(payload: Any, *, path: str = "$") -> None:
    """Apply the canonical recursive no-raw contract to a filtering artifact."""

    _validate_classifier_public_no_raw_payload(payload, path=path)


def validate_closeout_source_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the compact, no-row descriptor consumed by R4 closeout."""

    validate_public_no_raw_payload(payload)
    if payload.get("artifact_kind") != CLOSEOUT_SOURCE_MANIFEST_KIND:
        raise ValueError("closeout source manifest kind mismatch")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("closeout source manifest schema mismatch")
    if payload.get("source_format") != "legacy_w1_private_score_state_archive_v1":
        raise ValueError("closeout source format is unsupported")
    for field in (
        "source_content_root",
        "archive_manifest_hash",
        "archive_sha256",
        "legacy_plan_hash",
        "ledger_content_root",
        "handoff_hash",
        "config_hash",
        "prefilter_producer_config_hash",
        "selected_policy_hash",
        "selected_policy_report_hash",
        "source_metadata_hash",
        "rank_assignment_hash",
        "shard_survivor_counts_hash",
        "original_logical_root_hash",
        "physical_source_root_hash",
        "semantic_identity_hash",
        "legacy_v1_runtime_envelope_hash",
    ):
        _require_sha256(payload.get(field), f"closeout source {field}")
    export_hash = payload.get("export_manifest_hash")
    if export_hash is not None:
        _require_sha256(export_hash, "closeout source export_manifest_hash")
    attempts = payload.get("execution_attempt_provenance_roots")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("closeout source execution attempt roots must be nonempty")
    for value in attempts:
        _require_sha256(value, "closeout source execution attempt root")
    if len(set(attempts)) != len(attempts):
        raise ValueError("closeout source execution attempt roots must be unique")
    if payload.get("identity_split_status") != "legacy_fused_validated_not_migrated":
        raise ValueError("closeout source identity split status mismatch")
    if payload.get("semantic_identity_hash") != closeout_semantic_identity_hash(payload):
        raise ValueError("closeout semantic identity projection mismatch")
    if attempts != [closeout_legacy_attempt_provenance_root(payload)]:
        raise ValueError("closeout legacy execution attempt projection mismatch")
    _positive_int(payload.get("configured_target_proxy_tokens"), "configured target proxy tokens")
    _positive_int(payload.get("source_survivor_count"), "closeout source survivor count")
    export_event = payload.get("export_event_proxy_tokens")
    if export_event is not None:
        _nonnegative_int(export_event, "export event proxy tokens")
    expectations = payload.get("expectations")
    expected_fields = {
        "persisted_accepted_proxy_tokens",
        "next_commit_ordinal",
        "export_accepted_row_count",
        "export_part_count",
        "export_last_global_ordinal",
        "export_accepted_proxy_tokens",
    }
    if not isinstance(expectations, Mapping) or set(expectations) != expected_fields:
        raise ValueError("closeout source expectations shape mismatch")
    for field in expected_fields:
        _nonnegative_int(expectations[field], f"closeout source expectation {field}")
    capabilities = payload.get("capabilities")
    expected_capabilities = {
        "compact_survivor_candidates": True,
        "private_evidence": True,
        "prefix_ledger": True,
        "natural_block_content_boundaries": False,
        "non_survivor_candidates": False,
        "canonical_clean_run_reconstruction": False,
    }
    if capabilities != expected_capabilities:
        raise ValueError("closeout source capabilities mismatch")
    for field in PUBLIC_FALSE_FLAGS:
        if payload.get(field) is not False:
            raise ValueError(f"closeout source manifest must set {field}=false")
    forbidden = {"rows", "candidates", "accepted_rows", "source_paths"}
    if forbidden.intersection(payload):
        raise ValueError("closeout source manifest must remain compact")
    if payload.get("source_content_root") != closeout_source_content_root(payload):
        raise ValueError("closeout source content root mismatch")
    return dict(payload)


def closeout_source_content_root(payload: Mapping[str, Any]) -> str:
    """Root immutable scientific source identity independently of physical paths."""

    return stable_json_hash(
        {
            "source_format": payload.get("source_format"),
            "archive_manifest_hash": payload.get("archive_manifest_hash"),
            "archive_sha256": payload.get("archive_sha256"),
            "legacy_plan_hash": payload.get("legacy_plan_hash"),
            "ledger_content_root": payload.get("ledger_content_root"),
            "handoff_hash": payload.get("handoff_hash"),
            "config_hash": payload.get("config_hash"),
            "prefilter_producer_config_hash": payload.get("prefilter_producer_config_hash"),
            "selected_policy_hash": payload.get("selected_policy_hash"),
            "selected_policy_report_hash": payload.get("selected_policy_report_hash"),
            "source_metadata_hash": payload.get("source_metadata_hash"),
            "rank_assignment_hash": payload.get("rank_assignment_hash"),
            "shard_survivor_counts_hash": payload.get("shard_survivor_counts_hash"),
            "source_survivor_count": payload.get("source_survivor_count"),
            "export_manifest_hash": payload.get("export_manifest_hash"),
            "semantic_identity_hash": payload.get("semantic_identity_hash"),
            "execution_attempt_provenance_roots": payload.get(
                "execution_attempt_provenance_roots"
            ),
            "legacy_v1_runtime_envelope_hash": payload.get(
                "legacy_v1_runtime_envelope_hash"
            ),
            "expectations": payload.get("expectations"),
        }
    )


def closeout_semantic_identity_hash(payload: Mapping[str, Any]) -> str:
    return stable_json_hash(
        {
            "config_hash": payload.get("config_hash"),
            "prefilter_producer_config_hash": payload.get(
                "prefilter_producer_config_hash"
            ),
            "selected_policy_hash": payload.get("selected_policy_hash"),
            "selected_policy_report_hash": payload.get("selected_policy_report_hash"),
            "source_metadata_hash": payload.get("source_metadata_hash"),
            "handoff_hash": payload.get("handoff_hash"),
            "shard_survivor_counts_hash": payload.get("shard_survivor_counts_hash"),
            "source_survivor_count": payload.get("source_survivor_count"),
        }
    )


def closeout_legacy_attempt_provenance_root(payload: Mapping[str, Any]) -> str:
    return stable_json_hash(
        {
            "legacy_v1_runtime_envelope_hash": payload.get(
                "legacy_v1_runtime_envelope_hash"
            ),
            "rank_assignment_hash": payload.get("rank_assignment_hash"),
        }
    )


def validate_private_clean_run_output_path(path: str | Path) -> None:
    resolved = Path(path).expanduser().resolve(strict=False)
    for parent in (resolved, *resolved.parents):
        if is_public_repo_root(parent):
            raise ValueError("private clean-run output must be outside the public repo")


def validate_private_filtering_run_dir(path: str | Path) -> None:
    resolved = Path(path).expanduser().resolve(strict=False)
    for parent in (resolved, *resolved.parents):
        if is_public_repo_root(parent):
            raise ValueError("private filtering run dir must be outside the public repo")
    if Path(path).expanduser().is_symlink():
        raise ValueError("private filtering run dir must not be a symlink")


def validate_private_source_cache_dir(path: str | Path) -> None:
    resolved = Path(path).expanduser().resolve(strict=False)
    for parent in (resolved, *resolved.parents):
        if is_public_repo_root(parent):
            raise ValueError("private source cache dir must be outside the public repo")
    if Path(path).expanduser().is_symlink():
        raise ValueError("private source cache dir must not be a symlink")


def _ensure_private_dir(path: Path) -> None:
    validate_private_filtering_run_dir(path)
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(path, 0o700)


def _ensure_private_cache_dir(path: Path) -> None:
    validate_private_source_cache_dir(path)
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(path, 0o700)


def _chmod_private_file(path: str | Path) -> None:
    if os.name == "posix":
        try:
            os.chmod(path, 0o600)
        except FileNotFoundError:
            return


def _chmod_private_sqlite_files(path: str | Path) -> None:
    base = Path(path)
    _chmod_private_file(base)
    _chmod_private_file(Path(str(base) + "-wal"))
    _chmod_private_file(Path(str(base) + "-shm"))


def _validate_private_partial_metadata(metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("private partial must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("private partial must be a regular file")
    if int(metadata.st_nlink) != 1:
        raise ValueError("private partial must have a single link")
    if os.name == "posix":
        if int(metadata.st_uid) != int(os.geteuid()):
            raise ValueError("private partial must belong to the current user")
        if stat.S_IMODE(metadata.st_mode) != _PRIVATE_FILE_MODE:
            raise ValueError("private partial must use mode 0600")


def _validate_owned_private_partial(path: str | Path) -> os.stat_result:
    """Validate a stale or publishable partial without following links."""

    partial = Path(path)
    try:
        metadata = partial.lstat()
    except FileNotFoundError:
        raise ValueError("private partial is missing") from None
    _validate_private_partial_metadata(metadata)
    return metadata


def _create_private_partial(path: str | Path) -> int:
    """Create a fresh owner-only partial, restarting only a trusted stale file."""

    partial = Path(path)
    partial.parent.mkdir(parents=True, exist_ok=True)
    try:
        _validate_owned_private_partial(partial)
    except ValueError:
        if partial.is_symlink() or partial.exists():
            raise
    else:
        partial.unlink()

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(partial, flags, _PRIVATE_FILE_MODE)
    try:
        if os.name == "posix":
            os.fchmod(fd, _PRIVATE_FILE_MODE)
    except BaseException:
        os.close(fd)
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
        raise
    return fd


def _flush_and_fsync(handle: Any) -> None:
    """Flush a writable file object through the operating-system boundary."""

    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(path: str | Path) -> None:
    """Persist directory-entry changes on POSIX filesystems."""

    if os.name != "posix":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(Path(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _promote_private_partial(partial_path: str | Path, destination_path: str | Path) -> None:
    """Durably replace a destination with a validated same-directory partial."""

    partial = Path(partial_path)
    destination = Path(destination_path)
    if partial.parent.resolve(strict=False) != destination.parent.resolve(strict=False):
        raise ValueError("private partial promotion requires one directory")
    expected = _validate_owned_private_partial(partial)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(partial, flags)
    promoted = False
    try:
        opened = os.fstat(fd)
        _validate_private_partial_metadata(opened)
        if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("private partial changed while opening")
        os.fsync(fd)
        current = _validate_owned_private_partial(partial)
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("private partial changed before promotion")
        if os.name == "posix":
            os.replace(partial, destination)
            promoted = True
    finally:
        os.close(fd)
    if not promoted:
        os.replace(partial, destination)
    _chmod_private_file(destination)
    _fsync_directory(destination.parent)


def _path_free_bytes(path: str | Path) -> int:
    probe = Path(path)
    existing = probe if probe.exists() else probe.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    return int(os.statvfs(existing).f_bavail * os.statvfs(existing).f_frsize)


def _check_disk_watermark(paths: Sequence[str | Path], *, min_free_bytes: int) -> None:
    minimum = _nonnegative_int(min_free_bytes, "min_free_bytes")
    for path in paths:
        if _path_free_bytes(path) < minimum:
            raise RuntimeError("disk hard stop: free bytes below configured minimum")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    return coerce_lowercase_sha256(value, label)


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


def _nonnegative_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be nonnegative and finite")
    return result


def _require_false_flags(payload: Mapping[str, Any], fields: Sequence[str], context: str) -> None:
    for field in fields:
        if field in payload and payload[field] is not False:
            raise ValueError(f"{context} must keep {field}=false")


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _validate_report_self_hash_required(report: Mapping[str, Any], label: str) -> None:
    if report.get("report_hash") is None:
        raise ValueError(f"{label} missing report_hash")
    if self_hash_mismatches(report, "report_hash"):
        raise ValueError(f"{label} report_hash content mismatch")


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        if os.name == "posix":
            os.fchmod(fd, _PRIVATE_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            _flush_and_fsync(handle)
        _promote_private_partial(tmp_name, output)
    except Exception:
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass
        raise


def _write_jsonl_path(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")
    _chmod_private_file(output)


def _write_jsonl_path_streaming(
    output: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[str, int]:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    _chmod_private_file(path)
    return digest.hexdigest(), count


def _write_jsonl_ready(base_path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base = Path(base_path)
    part = base.with_suffix(base.suffix + ".part")
    ready = base.with_suffix(base.suffix + ".ready")
    _write_jsonl_path(part, rows)
    part.replace(ready)
    _chmod_private_file(ready)


def _write_jsonl_ready_streaming(
    base_path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[str, int]:
    base = Path(base_path)
    part = base.with_suffix(base.suffix + ".part")
    ready = base.with_suffix(base.suffix + ".ready")
    digest, count = _write_jsonl_path_streaming(part, rows)
    part.replace(ready)
    _chmod_private_file(ready)
    return digest, count


def _iter_ready_jsonl(base_path: str | Path, label: str) -> Iterator[dict[str, Any]]:
    base = Path(base_path)
    part = base.with_suffix(base.suffix + ".part")
    ready = base.with_suffix(base.suffix + ".ready")
    if part.exists():
        raise ValueError(f"{label} has untrusted .part export")
    if not ready.exists():
        raise ValueError(f"{label} is missing ready export")
    with ready.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            yield payload


def _load_ready_jsonl(base_path: str | Path, label: str) -> list[dict[str, Any]]:
    return list(_iter_ready_jsonl(base_path, label))


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _w1_runtime_execution_envelope_hash(envelope: Mapping[str, Any]) -> str:
    envelope_hash = stable_json_hash(dict(envelope))
    return _require_sha256(envelope_hash, "w1_runtime_envelope_hash")


def _validated_w1_runtime_envelope_from_receipt(
    receipt: Mapping[str, Any],
    *,
    requested_w1_runtime_envelope_hash: str | None = None,
    label: str = "W1 score receipt",
) -> tuple[dict[str, Any], str]:
    w1_runtime_envelope = receipt.get("w1_runtime_envelope")
    if not isinstance(w1_runtime_envelope, Mapping):
        raise ValueError(f"{label} missing runtime envelope")
    w1_runtime_envelope_hash = _require_sha256(
        receipt.get("w1_runtime_envelope_hash"),
        "w1_runtime_envelope_hash",
    )
    if _w1_runtime_execution_envelope_hash(w1_runtime_envelope) != w1_runtime_envelope_hash:
        raise ValueError(f"{label} runtime envelope hash mismatch")
    if (
        requested_w1_runtime_envelope_hash is not None
        and w1_runtime_envelope_hash != requested_w1_runtime_envelope_hash
    ):
        raise ValueError(f"{label} runtime envelope mismatch")
    return dict(w1_runtime_envelope), w1_runtime_envelope_hash


def _w1_score_row_runtime_hashes(score_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for row in score_rows:
        hashes.add(_require_sha256(row.get("w1_runtime_envelope_hash"), "w1_runtime_envelope_hash"))
    return hashes


def _validate_w1_score_rows(
    rows: Sequence[Mapping[str, Any]],
    config_payload: Mapping[str, Any],
    *,
    survivor_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    allow_empty: bool = False,
) -> None:
    if not rows and not allow_empty:
        raise ValueError("W1 score export is empty")
    seen: set[str] = set()
    for row in rows:
        validate_public_no_raw_payload(row)
        candidate_id = _require_sha256(row.get("candidate_id_hash"), "candidate_id_hash")
        if candidate_id in seen:
            raise ValueError("W1 score export contains duplicate candidate ids")
        seen.add(candidate_id)
        if row.get("config_hash") != config_payload["config_hash"]:
            raise ValueError("W1 score config hash mismatch")
        if row.get("selected_policy_hash") != config_payload["selected_policy_hash"]:
            raise ValueError("W1 score selected policy mismatch")
        if row.get("selected_policy_report_hash") != config_payload["selected_policy_report_hash"]:
            raise ValueError("W1 score selected policy report mismatch")
        if row.get("source_metadata_hash") != config_payload["source_metadata_hash"]:
            raise ValueError("W1 score source metadata hash mismatch")
        if row.get("w1_report_hash") != config_payload["window_mil_report_hash"]:
            raise ValueError("W1 score report hash mismatch")
        if row.get("w1_model_file_sha256") != config_payload["window_mil_model_file_sha256"]:
            raise ValueError("W1 score model hash mismatch")
        _require_sha256(row.get("w1_runtime_envelope_hash"), "w1_runtime_envelope_hash")
        if float(row.get("w1_threshold")) != float(config_payload["window_mil_threshold"]):
            raise ValueError("W1 score threshold mismatch")
        if row.get("w1_route") not in {"w1_pass", "w1_reject", "w1_error"}:
            raise ValueError("W1 score route mismatch")
        score = row.get("w1_score")
        if row.get("w1_route") == "w1_error":
            if score is not None or not row.get("error_class"):
                raise ValueError("W1 error score row must carry only an error_class")
        else:
            parsed = chunk_contracts.coerce_optional_risk_score(score)
            if parsed is None:
                raise ValueError("W1 non-error score row must carry a finite score")
            if row.get("w1_route") == "w1_pass" and parsed >= float(config_payload["window_mil_threshold"]):
                raise ValueError("W1 score route contradicts threshold")
            if row.get("w1_route") == "w1_reject" and parsed < float(config_payload["window_mil_threshold"]):
                raise ValueError("W1 score route contradicts threshold")
        if survivor_by_id is not None:
            survivor = survivor_by_id.get(candidate_id)
            if survivor is None:
                raise ValueError("W1 score row missing matching survivor queue row")
            if row.get("survivor_stage_receipt_hash") != survivor.get("stage_receipt_hash"):
                raise ValueError("W1 score survivor receipt hash mismatch")
        _require_sha256(row.get("stage_receipt_hash"), "stage_receipt_hash")
        if self_hash_mismatches(row, "stage_receipt_hash"):
            raise ValueError("W1 score row receipt hash mismatch")


def _load_existing_complete_w1_receipt(
    run_path: Path,
    config_payload: Mapping[str, Any],
    *,
    queue_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    w1_runtime_envelope_hash: str,
) -> dict[str, Any] | None:
    receipt_path = run_path / "w1_score_receipt.json"
    if not receipt_path.exists():
        return None
    receipt = _load_json(receipt_path)
    if not isinstance(receipt, Mapping):
        raise ValueError("existing W1 score receipt must be an object")
    if receipt.get("status") != "w1_scored":
        return None
    if self_hash_mismatches(receipt, "stage_receipt_hash"):
        raise ValueError("existing W1 score receipt hash mismatch")
    if receipt.get("artifact_kind") != W1_SCORE_STAGE_KIND:
        raise ValueError("existing W1 score receipt kind mismatch")
    if receipt.get("phase") != "w1-score":
        raise ValueError("existing W1 score receipt phase mismatch")
    if receipt.get("config_hash") != config_payload["config_hash"]:
        raise ValueError("existing W1 score receipt config hash mismatch")
    if receipt.get("selected_policy_hash") != config_payload["selected_policy_hash"]:
        raise ValueError("existing W1 score receipt selected policy mismatch")
    if receipt.get("selected_policy_report_hash") != config_payload["selected_policy_report_hash"]:
        raise ValueError("existing W1 score receipt selected policy report mismatch")
    if receipt.get("source_metadata_hash") != config_payload["source_metadata_hash"]:
        raise ValueError("existing W1 score receipt source metadata mismatch")
    _validated_w1_runtime_envelope_from_receipt(
        receipt,
        requested_w1_runtime_envelope_hash=w1_runtime_envelope_hash,
        label="existing W1 score receipt",
    )
    if int(receipt.get("survivor_count", -1)) != len(queue_rows):
        raise ValueError("existing W1 score receipt survivor count mismatch")
    if int(receipt.get("scored_count", -1)) != len(score_rows) or len(score_rows) != len(queue_rows):
        raise ValueError("existing W1 score receipt scored count mismatch")
    survivor_by_id = {str(row["candidate_id_hash"]): row for row in queue_rows}
    _validate_w1_score_rows(score_rows, config_payload, survivor_by_id=survivor_by_id, allow_empty=True)
    if score_rows and _w1_score_row_runtime_hashes(score_rows) != {w1_runtime_envelope_hash}:
        raise ValueError("existing W1 score receipt score row runtime envelope mismatch")
    score_ids = {str(row["candidate_id_hash"]) for row in score_rows}
    if score_ids != set(survivor_by_id):
        raise ValueError("existing W1 score receipt scored set mismatch")
    expected_route_counts = _sorted_counter(Counter(str(row["w1_route"]) for row in score_rows))
    if receipt.get("route_counts") != expected_route_counts:
        raise ValueError("existing W1 score receipt route counts mismatch")
    ready_path = run_path / "w1_scores.jsonl.ready"
    if not ready_path.exists():
        raise ValueError("existing complete W1 score receipt requires ready score export")
    if receipt.get("w1_scores_hash") != _sha256_file(ready_path):
        raise ValueError("existing W1 score receipt ready export hash mismatch")
    ready_queue_path = run_path / "w1_survivor_queue.jsonl.ready"
    current_survivor_queue_hash = _sha256_file(ready_queue_path)
    if (
        receipt.get("survivor_queue_hash") is not None
        and receipt.get("survivor_queue_hash") != current_survivor_queue_hash
    ):
        raise ValueError("existing W1 score receipt survivor queue hash mismatch")
    part_path = run_path / "w1_scores.jsonl.part"
    if receipt.get("w1_scores_part_hash") is not None:
        if not part_path.exists() or receipt.get("w1_scores_part_hash") != _sha256_file(part_path):
            raise ValueError("existing W1 score receipt part export hash mismatch")
    _require_false_flags(receipt, PUBLIC_FALSE_FLAGS, "existing W1 score receipt")
    validate_public_no_raw_payload(receipt)
    return dict(receipt)
