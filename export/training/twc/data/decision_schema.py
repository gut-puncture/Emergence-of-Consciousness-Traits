"""Decision-row schema and consumer guardrails for TWC filtering stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from twc.repro.hashing import stable_json_hash as _stable_json_hash


PHASE2A_ID = "phase2a_reject_only_foundation"
PHASE2A_DECISION_CAPABILITY = "reject_only_foundation"
PHASE2A_STATUSES = {
    "rejected",
    "quarantined",
    "error",
    "pending_required_gates",
}
SPAN_PROJECTION_VERSION = "span_projection_v1"

REQUIRED_FINAL_ACCEPT_FIELDS = {
    "decision_status": "accepted",
    "all_required_gates_present": True,
}


@dataclass(frozen=True)
class FilterHit:
    category: str
    rule_id: str
    severity: str
    reason: str
    source_field: str
    match_view: str
    matched_text: str
    span_normalized: tuple[int, int] | None
    raw_span: tuple[int, int] | None
    raw_context_snippet: str
    span_projection_version: str = SPAN_PROJECTION_VERSION
    pattern_type: str = "literal"
    hit_weight: float = 1.0
    scan_scope: str = "whole_document"
    window_index: int | None = None
    window_start_normalized: int | None = None
    window_end_normalized: int | None = None
    window_word_start: int | None = None
    window_word_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FoundationDecision:
    decision_status: str
    raw_text_sha256: str
    normalized_text_sha256: str
    view_sha256s: dict[str, str]
    ontology_semantic_hash: str
    ontology_source_tree_hash: str
    config_hash: str
    hits: tuple[FilterHit, ...] = field(default_factory=tuple)
    phase_id: str = PHASE2A_ID
    decision_capability: str = PHASE2A_DECISION_CAPABILITY
    all_required_gates_present: bool = False
    executed_gates: tuple[str, ...] = (
        "source_channel_gate",
        "deterministic_regex_gate",
    )
    skipped_required_gates: tuple[str, ...] = (
        "density_gate",
        "safe_topic_gate",
        "local_classifier_ensemble",
        "teacher_audit_failure_gate",
        "post_tokenization_rescan",
    )
    pending_gate_reasons: tuple[str, ...] = (
        "safe_topic_gate_not_implemented",
        "local_classifier_ensemble_not_implemented",
        "teacher_audit_failure_gate_not_implemented",
        "post_tokenization_rescan_not_implemented",
    )
    normalizer_version: str = ""
    normalization_steps_applied: tuple[str, ...] = field(default_factory=tuple)
    regex_engine: str = ""
    regex_engine_version: str = ""
    yaml_parser_version: str = ""
    python_unicode_version: str = ""
    match_timeout_ms: int = 0
    source_metadata_presence: dict[str, bool] = field(default_factory=dict)
    quarantine_reason_code: str | None = None
    error_class: str | None = None
    error_stage: str | None = None
    error_message_hash: str | None = None
    run_id: str | None = None
    parent_run_id: str | None = None
    artifact_set_id: str | None = None
    freeze_id: str | None = None

    def __post_init__(self) -> None:
        if self.decision_status not in PHASE2A_STATUSES:
            raise ValueError(f"Phase 2A cannot emit status: {self.decision_status}")
        if self.all_required_gates_present:
            raise ValueError("Phase 2A must not claim all required gates are present")
        if self.decision_status == "pending_required_gates" and not self.skipped_required_gates:
            raise ValueError("pending_required_gates requires skipped gate metadata")

    def to_decision_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["hits"] = [hit.to_dict() for hit in self.hits]
        row["decision_row_schema_version"] = 1
        row["decision_row_hash"] = _stable_json_hash(
            {key: value for key, value in row.items() if key != "decision_row_hash"}
        )
        return row


def assert_training_ready(row: dict[str, Any]) -> None:
    """Hard guard for future tokenizer, packer, corpus writer, and trainer inputs."""

    for key, expected in REQUIRED_FINAL_ACCEPT_FIELDS.items():
        if row.get(key) != expected:
            raise ValueError(
                "Decision row is not training-ready: "
                f"{key}={row.get(key)!r}, expected {expected!r}"
            )
    if not row.get("freeze_id"):
        raise ValueError("Decision row is not training-ready: missing freeze_id")
    if row.get("decision_capability") == PHASE2A_DECISION_CAPABILITY:
        raise ValueError("Phase 2A reject-only rows cannot feed training")
