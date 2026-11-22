"""Final 35k label freeze and no-raw classifier-training manifest helpers."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from twc.data.decision_schema import stable_json_hash


FINAL_LABEL_MANIFEST_KIND = "final_35k_classifier_training_manifest"
ALLOWED_FINAL_LABELS = {"safe", "reject"}
ALLOWED_FINAL_SECTIONS = {"borderline", "non_borderline"}
ALLOWED_LABEL_UNITS = {"document", "window_bundle"}
SERIALIZATION_FALSE_FLAGS = {
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "accepted_manifest",
    "labels_are_corpus_approval",
    "training_shards_created",
    "classifier_training_performed",
}
FORBIDDEN_PUBLIC_KEYS = {
    "document_text",
    "matched_text",
    "messages",
    "model_output",
    "normalized_text",
    "parquet_url_or_path",
    "prompt",
    "prompt_text",
    "raw_context",
    "raw_context_snippet",
    "raw_response",
    "local_source_path",
    "baseline_run_dir",
    "comparison_run_dir",
    "run_dir",
    "raw_text",
    "raw_url",
    "request_body",
    "response_body",
    "rationale",
    "rationale_short",
    "risk_spans",
    "source_doc_id",
    "source_domain",
    "source_locator",
    "private_locator",
    "private_source_locator",
    "raw_source_locator",
    "source_file_path",
    "source_path",
    "source_url",
    "snippet",
    "text",
}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
URL_RE = re.compile(r"https?://", re.IGNORECASE)


def build_final_training_manifest_from_rows(
    label_rows: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
    *,
    expected_total: int = 35_000,
    expected_borderline: int = 4_000,
    expected_non_borderline: int = 31_000,
    source_candidate_manifest_hash: str | None = None,
    split_seed: str = "twc-final-35k-v1",
) -> dict[str, Any]:
    """Build a redacted manifest that freezes final labels for classifier work.

    The manifest is deliberately no-raw and does not contain model responses,
    source document IDs, source domains, parquet URLs, raw text, tokens, or
    private locators. Rows with fail-closed parse errors are preserved for audit
    lineage but excluded from semantic fitting by assigning them to `audit` with
    zero training weight.
    """

    labels = [dict(row) for row in label_rows]
    candidates = [dict(row) for row in candidate_rows]
    _require_count("final label rows", labels, expected_total)
    _require_count("final candidate rows", candidates, expected_total)

    label_by_id = _index_unique(labels, key="candidate_id", label="final label")
    candidate_by_id = _index_unique(candidates, key="candidate_id", label="final candidate")
    label_ids = set(label_by_id)
    candidate_ids = set(candidate_by_id)
    if label_ids != candidate_ids:
        missing_labels = sorted(candidate_ids - label_ids)[:5]
        missing_candidates = sorted(label_ids - candidate_ids)[:5]
        raise ValueError(
            "final label/candidate identity mismatch; "
            f"missing_labels={missing_labels} missing_candidates={missing_candidates}"
        )

    for index, label in enumerate(labels):
        _validate_final_label_row(label, index, source_candidate_manifest_hash=source_candidate_manifest_hash)
    for index, candidate in enumerate(candidates):
        _validate_final_candidate_row(candidate, index)

    duplicate_counts = {
        "candidate_id": _duplicate_count(candidates, "candidate_id"),
        "candidate_hash": _duplicate_count(candidates, "candidate_hash"),
        "normalized_text_hash_sha256_present": _duplicate_count(candidates, "normalized_text_hash_sha256", present_only=True),
        "physical_source_row_key_hash": _duplicate_count(candidates, "physical_source_row_key_hash"),
        "text_hash_sha256_present": _duplicate_count(candidates, "text_hash_sha256", present_only=True),
    }
    duplicate_fields = [field for field, count in duplicate_counts.items() if count]
    if duplicate_fields:
        raise ValueError(f"final candidates contain duplicate identities: {duplicate_fields}")

    source_section_counts = Counter(str(candidate["final_manifest_source_section"]) for candidate in candidates)
    if source_section_counts != Counter({"borderline": expected_borderline, "non_borderline": expected_non_borderline}):
        raise ValueError(
            "final candidate section counts mismatch: "
            f"expected borderline={expected_borderline} non_borderline={expected_non_borderline}; "
            f"found={dict(sorted(source_section_counts.items()))}"
        )

    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_ids):
        candidate = candidate_by_id[candidate_id]
        label = label_by_id[candidate_id]
        if str(candidate["candidate_hash"]) != str(label["candidate_hash"]):
            raise ValueError(f"candidate_hash mismatch for candidate_id {candidate_id}")
        row = _redacted_manifest_row(candidate, label)
        rows.append(row)

    assignments = plan_dedupe_aware_splits(rows, split_seed=split_seed)
    for row in rows:
        row["split"] = assignments[row["candidate_id"]]
        row["row_hash"] = stable_json_hash(row)

    split_counts = Counter(row["split"] for row in rows)
    label_counts = Counter(row["classifier_label"] for row in rows)
    fit_rows = [row for row in rows if row["fit_eligible"]]
    fit_label_counts = Counter(row["classifier_label"] for row in fit_rows)
    audit_only_count = sum(1 for row in rows if not row["fit_eligible"])
    manifest_payload = {
        "manifest_kind": FINAL_LABEL_MANIFEST_KIND,
        "schema_version": 1,
        "candidate_count": len(rows),
        "fit_eligible_count": len(fit_rows),
        "audit_only_count": audit_only_count,
        "label_counts": _sorted_counter(label_counts),
        "fit_label_counts": _sorted_counter(fit_label_counts),
        "source_section_counts": _sorted_counter(source_section_counts),
        "bucket_counts": _sorted_counter(Counter(row["bucket_id"] for row in rows)),
        "label_unit_counts": _sorted_counter(Counter(row["label_unit"] for row in rows)),
        "split_counts": _sorted_counter(split_counts),
        "duplicate_counts": duplicate_counts,
        "private_text_refetch_required_count": sum(1 for row in rows if row["private_text_refetch_required"]),
        "parse_error_audit_only_count": sum(
            1
            for row in rows
            if row["aggregation_route"] == "parse_error_fail_closed_reject"
            or row["parse_error_vote_count"] > 0
        ),
        "source_candidate_manifest_hash": source_candidate_manifest_hash,
        "split_seed_hash": stable_json_hash({"split_seed": split_seed}),
        "split_policy": {
            "dedupe_group": "normalized_text_hash_then_text_hash_then_physical_row_key",
            "default_train_fraction": 0.8,
            "default_calibration_fraction": 0.1,
            "default_eval_fraction": 0.1,
            "audit_only_rows": "parse_error_fail_closed_or_fit_ineligible",
        },
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
        "rows": rows,
    }
    manifest_payload["training_manifest_hash"] = stable_json_hash(manifest_payload)
    validate_public_no_raw_payload(manifest_payload)
    return manifest_payload


def write_final_training_manifest_from_files(
    *,
    labels_jsonl: str | Path,
    candidates_json: str | Path,
    output_json: str | Path,
    split_seed: str = "twc-final-35k-v1",
) -> dict[str, Any]:
    """Write a final no-raw training manifest from local final-label artifacts."""

    labels_path = Path(labels_jsonl)
    candidates_path = Path(candidates_json)
    output_path = Path(output_json)
    labels = [
        json.loads(line)
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidate_manifest = json.loads(candidates_path.read_text(encoding="utf-8"))
    manifest = build_final_training_manifest_from_rows(
        labels,
        candidate_manifest["candidates"],
        expected_total=int(candidate_manifest["candidate_count"]),
        expected_borderline=int(candidate_manifest["quota_report"]["borderline_count"]),
        expected_non_borderline=int(candidate_manifest["quota_report"]["non_borderline_count"]),
        source_candidate_manifest_hash=str(candidate_manifest["manifest_hash"]),
        split_seed=split_seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True)
    validate_public_no_raw_payload(manifest)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    return manifest


def repair_training_manifest_for_text_hash_collisions(
    training_manifest: Mapping[str, Any],
    duplicate_audit_report: Mapping[str, Any],
    *,
    repair_policy: str = "prefer_holdout_over_train_duplicate",
) -> dict[str, Any]:
    """Exclude train-side private-text duplicates discovered after refetch.

    This is a no-raw repair for the final classifier-training manifest. It does
    not create accepted manifests or replacement corpus rows.
    """

    if repair_policy != "prefer_holdout_over_train_duplicate":
        raise ValueError("unsupported private text duplicate repair policy")
    validate_public_no_raw_payload(training_manifest)
    validate_public_no_raw_payload(duplicate_audit_report)
    if training_manifest.get("manifest_kind") != FINAL_LABEL_MANIFEST_KIND:
        raise ValueError("unsupported classifier training manifest kind")
    if duplicate_audit_report.get("artifact_kind") != "final_35k_private_text_cache_duplicate_audit":
        raise ValueError("unsupported duplicate audit report kind")
    rows = [dict(row) for row in training_manifest.get("rows") or []]
    row_by_hash = {
        stable_json_hash({"candidate_id": _required_hash(row.get("candidate_id"), "candidate_id")}): row
        for row in rows
    }
    excluded_by_hash: dict[str, str] = {}
    collisions = list(duplicate_audit_report.get("collisions") or [])
    for collision in collisions:
        if not isinstance(collision, Mapping):
            raise ValueError("duplicate collision entry must be a mapping")
        if collision.get("same_classifier_label") is not True:
            raise ValueError("cannot repair label-disagreeing private text duplicate collision")
        collision_hash = _required_hash(collision.get("collision_hash"), "collision_hash")
        members = collision.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError("duplicate collision must contain at least two members")
        resolved_members: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
        for member in members:
            if not isinstance(member, Mapping):
                raise ValueError("duplicate collision member must be a mapping")
            candidate_id_hash = _required_hash(member.get("candidate_id_hash"), "candidate_id_hash")
            row = row_by_hash.get(candidate_id_hash)
            if row is None:
                raise ValueError("duplicate collision member missing from training manifest")
            if member.get("split") != row.get("split") or member.get("classifier_label") != row.get("classifier_label"):
                raise ValueError("duplicate collision member drift from training manifest")
            resolved_members.append((member, row))
        holdout = [
            (member, row)
            for member, row in resolved_members
            if row.get("fit_eligible") is not False and row.get("split") in {"calibration", "eval"}
        ]
        train = [
            (member, row)
            for member, row in resolved_members
            if row.get("fit_eligible") is not False and row.get("split") == "train"
        ]
        if holdout and train:
            for member, _row in train:
                excluded_by_hash[_required_hash(member.get("candidate_id_hash"), "candidate_id_hash")] = collision_hash
            continue
        if len(train) > 1 and not holdout:
            ordered_train = sorted(train, key=lambda item: str(item[0]["candidate_id_hash"]))
            for member, _row in ordered_train[1:]:
                excluded_by_hash[_required_hash(member.get("candidate_id_hash"), "candidate_id_hash")] = collision_hash
            continue
        raise ValueError("private text duplicate collision requires coordinator review")

    for row in rows:
        candidate_hash = stable_json_hash({"candidate_id": row["candidate_id"]})
        collision_hash = excluded_by_hash.get(candidate_hash)
        if collision_hash is None:
            row["row_hash"] = stable_json_hash({key: value for key, value in row.items() if key != "row_hash"})
            continue
        row["fit_eligible"] = False
        row["training_weight"] = 0.0
        row["split"] = "audit"
        row["training_role"] = "audit_only_duplicate_text_hash_repair"
        row["duplicate_text_hash_excluded"] = True
        row["duplicate_text_hash_collision_hash"] = collision_hash
        row["duplicate_text_hash_repair_policy"] = repair_policy
        row["row_hash"] = stable_json_hash({key: value for key, value in row.items() if key != "row_hash"})

    repaired = dict(training_manifest)
    repaired.pop("training_manifest_hash", None)
    repaired["rows"] = rows
    repaired["fit_eligible_count"] = sum(1 for row in rows if row.get("fit_eligible") is not False and row.get("split") != "audit")
    repaired["audit_only_count"] = sum(1 for row in rows if row.get("fit_eligible") is False or row.get("split") == "audit")
    repaired["label_counts"] = _sorted_counter(Counter(str(row["classifier_label"]) for row in rows))
    fit_rows = [row for row in rows if row.get("fit_eligible") is not False and row.get("split") != "audit"]
    repaired["fit_label_counts"] = _sorted_counter(Counter(str(row["classifier_label"]) for row in fit_rows))
    repaired["bucket_counts"] = _sorted_counter(Counter(str(row["bucket_id"]) for row in rows))
    repaired["label_unit_counts"] = _sorted_counter(Counter(str(row["label_unit"]) for row in rows))
    repaired["split_counts"] = _sorted_counter(Counter(str(row["split"]) for row in rows))
    repaired["private_text_refetch_required_count"] = sum(1 for row in rows if row["private_text_refetch_required"])
    repaired["parse_error_audit_only_count"] = sum(
        1
        for row in rows
        if row["aggregation_route"] == "parse_error_fail_closed_reject"
        or row["parse_error_vote_count"] > 0
    )
    repaired["private_text_duplicate_repair"] = {
        "policy": repair_policy,
        "duplicate_audit_report_hash": _required_hash(duplicate_audit_report.get("report_hash"), "duplicate_audit_report_hash"),
        "collision_count": len(collisions),
        "excluded_fit_row_count": len(excluded_by_hash),
        "excluded_candidate_id_hashes": sorted(excluded_by_hash),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    repaired["training_manifest_hash"] = stable_json_hash(repaired)
    validate_public_no_raw_payload(repaired)
    return repaired


def plan_dedupe_aware_splits(
    rows: Iterable[Mapping[str, Any]],
    *,
    split_seed: str = "twc-final-35k-v1",
    train_fraction: float = 0.8,
    calibration_fraction: float = 0.1,
    eval_fraction: float = 0.1,
) -> dict[str, str]:
    """Assign deterministic splits while keeping dedupe groups together."""

    if abs((train_fraction + calibration_fraction + eval_fraction) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    by_stratum: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    assignments: dict[str, str] = {}
    for row in rows:
        candidate_id = _required_str(row, "candidate_id")
        if row.get("fit_eligible") is False:
            assignments[candidate_id] = "audit"
            continue
        group = _required_str(row, "dedupe_group_hash")
        stratum = _required_str(row, "stratum_key")
        by_stratum[stratum][group].append(candidate_id)

    for stratum, groups in sorted(by_stratum.items()):
        ordered_groups = sorted(
            groups,
            key=lambda group: stable_json_hash({"split_seed": split_seed, "stratum": stratum, "group": group}),
        )
        split_for_group = _split_sequence_for_group_count(
            len(ordered_groups),
            train_fraction=train_fraction,
            calibration_fraction=calibration_fraction,
            eval_fraction=eval_fraction,
        )
        for group, split in zip(ordered_groups, split_for_group, strict=True):
            for candidate_id in groups[group]:
                assignments[candidate_id] = split
    return assignments


def validate_public_no_raw_payload(payload: Any, *, path: str = "$") -> None:
    """Fail if a public classifier payload contains raw/private fields."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            key_lower = key_text.casefold()
            child_path = f"{path}.{key_text}"
            if key_lower in SERIALIZATION_FALSE_FLAGS:
                if value is not False:
                    raise ValueError(f"{child_path} must be false")
                continue
            if key_lower in FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"public payload contains forbidden key {key_text} at {child_path}")
            if "url" in key_lower:
                raise ValueError(f"public payload contains url-like key {key_text} at {child_path}")
            validate_public_no_raw_payload(value, path=child_path)
    elif isinstance(payload, list | tuple):
        for index, item in enumerate(payload):
            validate_public_no_raw_payload(item, path=f"{path}[{index}]")
    elif isinstance(payload, str) and URL_RE.search(payload):
        raise ValueError(f"public payload contains raw url-like string at {path}")


def _redacted_manifest_row(candidate: Mapping[str, Any], label: Mapping[str, Any]) -> dict[str, Any]:
    classifier_label = _required_str(label, "label")
    aggregation_route = _required_str(label, "aggregation_route")
    parse_error_vote_count = _required_int(label, "parse_error_vote_count")
    if (
        aggregation_route == "parse_error_fail_closed_reject"
        or parse_error_vote_count > 0
    ):
        if classifier_label != "reject":
            raise ValueError("parse-error fail-closed rows must be reject labels")
        training_role = "audit_only_fail_closed_reject"
        fit_eligible = False
        training_weight = 0.0
    elif classifier_label == "reject":
        training_role = "semantic_positive_reject"
        fit_eligible = True
        training_weight = 1.0
    else:
        training_role = "semantic_negative_safe"
        fit_eligible = True
        training_weight = 1.0

    source_section = _required_str(candidate, "final_manifest_source_section")
    bucket_id = _required_str(candidate, "bucket_id")
    label_unit = _required_str(candidate, "label_unit")
    text_hash = _optional_hash(candidate.get("text_hash_sha256"), "text_hash_sha256")
    normalized_text_hash = _optional_hash(
        candidate.get("normalized_text_hash_sha256"),
        "normalized_text_hash_sha256",
    )
    physical_key = _required_hash(candidate.get("physical_source_row_key_hash"), "physical_source_row_key_hash")
    dedupe_key = normalized_text_hash or text_hash or physical_key
    private_text_refetch_required = not bool(text_hash and normalized_text_hash)
    row = {
        "row_schema_version": 1,
        "candidate_id": _required_hash(candidate.get("candidate_id"), "candidate_id"),
        "candidate_hash": _required_hash(candidate.get("candidate_hash"), "candidate_hash"),
        "source_section": source_section,
        "bucket_id": bucket_id,
        "bucket_id_hash": stable_json_hash({"bucket_id": bucket_id}),
        "subbucket_id_hash": stable_json_hash({"subbucket_id": str(candidate.get("subbucket_id") or "")}),
        "label_unit": label_unit,
        "classifier_label": classifier_label,
        "aggregation_route": aggregation_route,
        "training_role": training_role,
        "fit_eligible": fit_eligible,
        "training_weight": training_weight,
        "parse_error_vote_count": parse_error_vote_count,
        "parsed_vote_count": _required_int(label, "parsed_vote_count"),
        "required_vote_count": _required_int(label, "required_vote_count"),
        "transport_failure_count": _required_int(label, "transport_failure_count"),
        "physical_source_row_key_hash": physical_key,
        "text_hash_sha256": text_hash,
        "normalized_text_hash_sha256": normalized_text_hash,
        "private_text_refetch_required": private_text_refetch_required,
        "row_group_index": _required_int(candidate, "row_group_index"),
        "row_group_local_row_index": _required_int(candidate, "row_group_local_row_index"),
        "schedule_entry_hash": _required_hash(candidate.get("schedule_entry_hash"), "schedule_entry_hash"),
        "source_schedule_hash": _required_hash(candidate.get("source_schedule_hash"), "source_schedule_hash"),
        "source_file_id_hash": stable_json_hash({"source_file_id": str(candidate.get("source_file_id") or "")}),
        "dedupe_group_hash": stable_json_hash({"dedupe_key": dedupe_key}),
        "stratum_key": ":".join([classifier_label, source_section, bucket_id, label_unit, training_role]),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
        "accepted_manifest": False,
        "labels_are_corpus_approval": False,
        "training_shards_created": False,
        "classifier_training_performed": False,
    }
    return row


def _validate_final_label_row(
    row: Mapping[str, Any],
    index: int,
    *,
    source_candidate_manifest_hash: str | None,
) -> None:
    validate_public_no_raw_payload(row, path=f"label_rows[{index}]")
    for field in (
        "candidate_id",
        "candidate_hash",
        "label",
        "final_label",
        "aggregation_route",
        "parse_status",
        "parse_error_vote_count",
        "parsed_vote_count",
        "required_vote_count",
        "transport_failure_count",
        "label_unit",
        "schedule_entry_hash",
    ):
        if field not in row:
            raise ValueError(f"final label row {index} missing {field}")
    _require_hash(row["candidate_id"], "candidate_id")
    _require_hash(row["candidate_hash"], "candidate_hash")
    label = _required_str(row, "label")
    final_label = _required_str(row, "final_label")
    if label not in ALLOWED_FINAL_LABELS:
        raise ValueError(f"unsupported final classifier label: {label}")
    if final_label != label:
        raise ValueError("final_label must match label")
    if _required_str(row, "parse_status") != "final_label":
        raise ValueError("final label row parse_status must be final_label")
    if _required_str(row, "label_unit") not in ALLOWED_LABEL_UNITS:
        raise ValueError("final label row label_unit must be document or window_bundle")
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if field in row:
            _require_false(row, field)
    for field in ("accepted_manifest", "labels_are_corpus_approval", "training_shards_created", "classifier_training_performed"):
        if field in row:
            _require_false(row, field)
    if source_candidate_manifest_hash is not None:
        _require_hash(source_candidate_manifest_hash, "source_candidate_manifest_hash")
        if row.get("source_candidate_manifest_hash") != source_candidate_manifest_hash:
            raise ValueError("source_candidate_manifest_hash mismatch in final labels")


def _validate_final_candidate_row(row: Mapping[str, Any], index: int) -> None:
    for forbidden in ("raw_text", "text", "document_text", "normalized_text", "matched_text", "snippet"):
        if forbidden in row:
            raise ValueError(f"final candidate row {index} contains forbidden raw field {forbidden}")
    for field in (
        "candidate_id",
        "candidate_hash",
        "final_manifest_source_section",
        "bucket_id",
        "label_unit",
        "physical_source_row_key_hash",
        "row_group_index",
        "row_group_local_row_index",
        "schedule_entry_hash",
        "source_schedule_hash",
    ):
        if field not in row:
            raise ValueError(f"final candidate row {index} missing {field}")
    _require_hash(row["candidate_id"], "candidate_id")
    _require_hash(row["candidate_hash"], "candidate_hash")
    _require_hash(row["physical_source_row_key_hash"], "physical_source_row_key_hash")
    _require_hash(row["schedule_entry_hash"], "schedule_entry_hash")
    _require_hash(row["source_schedule_hash"], "source_schedule_hash")
    if _required_str(row, "final_manifest_source_section") not in ALLOWED_FINAL_SECTIONS:
        raise ValueError("final candidate section must be borderline or non_borderline")
    if _required_str(row, "label_unit") not in ALLOWED_LABEL_UNITS:
        raise ValueError("final candidate label_unit must be document or window_bundle")
    for optional in ("text_hash_sha256", "normalized_text_hash_sha256"):
        _optional_hash(row.get(optional), optional)
    for field in ("raw_text_serialized", "raw_url_serialized", "token_serialized"):
        if field in row:
            _require_false(row, field)
    for field in ("accepted_manifest", "labels_are_corpus_approval", "training_shards_created", "classifier_training_performed"):
        if field in row:
            _require_false(row, field)


def _split_sequence_for_group_count(
    count: int,
    *,
    train_fraction: float,
    calibration_fraction: float,
    eval_fraction: float,
) -> list[str]:
    if count < 1:
        return []
    if count == 1:
        return ["train"]
    if count == 2:
        return ["train", "eval"]
    calibration_count = max(1, round(count * calibration_fraction))
    eval_count = max(1, round(count * eval_fraction))
    if calibration_count + eval_count >= count:
        calibration_count = 1
        eval_count = 1
    train_count = count - calibration_count - eval_count
    if train_count < 1:
        train_count = 1
        if calibration_count > eval_count:
            calibration_count -= 1
        else:
            eval_count -= 1
    return ["train"] * train_count + ["calibration"] * calibration_count + ["eval"] * eval_count


def _index_unique(rows: Iterable[Mapping[str, Any]], *, key: str, label: str) -> dict[str, Mapping[str, Any]]:
    by_key: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        value = _required_str(row, key)
        if value in by_key:
            raise ValueError(f"duplicate {key} in {label} rows: {value}")
        by_key[value] = row
    return by_key


def _duplicate_count(rows: Iterable[Mapping[str, Any]], key: str, *, present_only: bool = False) -> int:
    values: list[str] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            if present_only:
                continue
            values.append("")
        else:
            values.append(str(value))
    return len(values) - len(set(values))


def _require_count(label: str, rows: list[Mapping[str, Any]], expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"{label} count mismatch: expected {expected}, found {len(rows)}")


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _required_str(row: Mapping[str, Any], key: str) -> str:
    if key not in row or row[key] in (None, ""):
        raise ValueError(f"missing required field {key}")
    return str(row[key])


def _required_int(row: Mapping[str, Any], key: str) -> int:
    if key not in row:
        raise ValueError(f"missing required field {key}")
    return int(row[key])


def _require_hash(value: Any, field_name: str) -> str:
    candidate = str(value)
    if not HASH_RE.fullmatch(candidate):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return candidate


def _required_hash(value: Any, field_name: str) -> str:
    return _require_hash(value, field_name)


def _optional_hash(value: Any, field_name: str) -> str | None:
    if value in (None, ""):
        return None
    return _require_hash(value, field_name)


def _require_false(row: Mapping[str, Any], field_name: str) -> None:
    if row.get(field_name) is not False:
        raise ValueError(f"{field_name} must be false")
