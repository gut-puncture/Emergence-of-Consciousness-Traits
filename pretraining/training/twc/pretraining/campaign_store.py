"""Crash-safe local persistence for provider-neutral campaign contracts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from decimal import Decimal
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

from twc.pretraining.campaign import (
    AttemptPlan,
    AttemptReceipt,
    AttemptStart,
    CampaignEvent,
    CampaignLedger,
    CampaignPlan,
    CheckpointBinding,
    ConfigurationPromotion,
    ResumeBinding,
    WaveClosed,
    WaveOpened,
    WavePlan,
)
from twc.repro.hashing import stable_json_hash


def _money(value: Decimal) -> str:
    return format(value, "f")


def _decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a decimal string")
    try:
        result = Decimal(value)
    except Exception as error:
        raise ValueError(f"{label} is not a decimal string") from error
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def campaign_plan_artifact(plan: CampaignPlan) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_kind": "twc_gpu_campaign_plan_v1",
        "schema_version": 1,
        "campaign_id": plan.campaign_id,
        "budget_cap_usd": _money(plan.budget_cap_usd),
        "reserve_usd": _money(plan.reserve_usd),
        "code_hash": plan.code_hash,
        "evaluation_policy_hash": plan.evaluation_policy_hash,
        "waves": [
            {
                "wave_id": wave.wave_id,
                "purpose": wave.purpose,
                "cap_usd": _money(wave.cap_usd),
                "prerequisite_wave_ids": list(wave.prerequisite_wave_ids),
                "attempts": [
                    {
                        "attempt_id": attempt.attempt_id,
                        "model_config_hash": attempt.model_config_hash,
                        "data_order_hash": attempt.data_order_hash,
                        "execution_path_hash": attempt.execution_path_hash,
                        "max_cost_usd": _money(attempt.max_cost_usd),
                        "recovery_mode": attempt.recovery_mode,
                        "predecessor_attempt_id": attempt.predecessor_attempt_id,
                    }
                    for attempt in wave.attempts
                ],
            }
            for wave in plan.waves
        ],
        "campaign_hash": plan.identity,
        "production_unlock": False,
        "f1_unlock": False,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload


def load_campaign_plan(path: Path) -> CampaignPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "artifact_kind", "schema_version", "campaign_id", "budget_cap_usd", "reserve_usd",
        "code_hash", "evaluation_policy_hash", "waves", "campaign_hash", "production_unlock",
        "f1_unlock", "receipt_hash",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("campaign plan artifact fields are invalid")
    if payload["artifact_kind"] != "twc_gpu_campaign_plan_v1" or payload["schema_version"] != 1:
        raise ValueError("campaign plan artifact version is invalid")
    if payload["production_unlock"] is not False or payload["f1_unlock"] is not False:
        raise ValueError("campaign plan cannot unlock production or F1")
    if payload["receipt_hash"] != stable_json_hash({key: value for key, value in payload.items() if key != "receipt_hash"}):
        raise ValueError("campaign plan receipt hash mismatch")
    wave_rows = payload["waves"]
    if not isinstance(wave_rows, list):
        raise ValueError("campaign plan waves must be a list")
    waves: list[WavePlan] = []
    wave_fields = {"wave_id", "purpose", "cap_usd", "prerequisite_wave_ids", "attempts"}
    legacy_attempt_fields = {"attempt_id", "model_config_hash", "data_order_hash", "execution_path_hash", "max_cost_usd"}
    attempt_fields = legacy_attempt_fields | {"recovery_mode", "predecessor_attempt_id"}
    for wave_row in wave_rows:
        if not isinstance(wave_row, dict) or set(wave_row) != wave_fields or not isinstance(wave_row["attempts"], list):
            raise ValueError("campaign plan wave fields are invalid")
        attempts: list[AttemptPlan] = []
        for attempt_row in wave_row["attempts"]:
            if not isinstance(attempt_row, dict) or frozenset(attempt_row) not in {
                frozenset(legacy_attempt_fields), frozenset(attempt_fields)
            }:
                raise ValueError("campaign plan attempt fields are invalid")
            attempts.append(
                AttemptPlan(
                    attempt_id=attempt_row["attempt_id"],
                    model_config_hash=attempt_row["model_config_hash"],
                    data_order_hash=attempt_row["data_order_hash"],
                    execution_path_hash=attempt_row["execution_path_hash"],
                    max_cost_usd=_decimal(attempt_row["max_cost_usd"], "max_cost_usd"),
                    recovery_mode=attempt_row.get("recovery_mode", "none"),
                    predecessor_attempt_id=attempt_row.get("predecessor_attempt_id"),
                )
            )
        prerequisites = wave_row["prerequisite_wave_ids"]
        if not isinstance(prerequisites, list) or any(not isinstance(value, str) for value in prerequisites):
            raise ValueError("campaign plan prerequisites are invalid")
        waves.append(
            WavePlan(
                wave_id=wave_row["wave_id"],
                purpose=wave_row["purpose"],
                cap_usd=_decimal(wave_row["cap_usd"], "wave cap"),
                attempts=tuple(attempts),
                prerequisite_wave_ids=tuple(prerequisites),
            )
        )
    plan = CampaignPlan(
        campaign_id=payload["campaign_id"],
        budget_cap_usd=_decimal(payload["budget_cap_usd"], "budget_cap_usd"),
        reserve_usd=_decimal(payload["reserve_usd"], "reserve_usd"),
        code_hash=payload["code_hash"],
        evaluation_policy_hash=payload["evaluation_policy_hash"],
        waves=tuple(waves),
    )
    if payload["campaign_hash"] != plan.identity:
        raise ValueError("campaign plan semantic identity mismatch")
    return plan


def _checkpoint(value: object) -> CheckpointBinding | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "checkpoint_manifest_hash", "generation", "step", "consumed_loss_tokens", "next_global_ordinal"
    }:
        raise ValueError("campaign checkpoint binding is invalid")
    return CheckpointBinding(**value)


def _event_payload(event: CampaignEvent) -> tuple[str, dict[str, Any]]:
    if isinstance(event, WaveOpened):
        return "wave_opened", asdict(event)
    if isinstance(event, WaveClosed):
        return "wave_closed", asdict(event)
    if isinstance(event, AttemptStart):
        resume = None
        if event.resume is not None:
            resume = {
                "prior_attempt_id": event.resume.prior_attempt_id,
                "prior_stop_receipt_hash": event.resume.prior_stop_receipt_hash,
                "checkpoint": asdict(event.resume.checkpoint),
            }
        return "attempt_started", {
            "request_id": event.request_id,
            "wave_id": event.wave_id,
            "attempt_id": event.attempt_id,
            "quoted_hourly_usd": _money(event.quoted_hourly_usd),
            "started_at_epoch": event.started_at_epoch,
            "resume": resume,
        }
    if isinstance(event, AttemptReceipt):
        return "attempt_stopped", {
            "request_id": event.request_id,
            "wave_id": event.wave_id,
            "attempt_id": event.attempt_id,
            "quoted_hourly_usd": _money(event.quoted_hourly_usd),
            "effective_hourly_usd": _money(event.effective_hourly_usd),
            "started_at_epoch": event.started_at_epoch,
            "stopped_at_epoch": event.stopped_at_epoch,
            "billed_seconds": event.billed_seconds,
            "billed_cost_usd": _money(event.billed_cost_usd),
            "estimated_cost_usd": _money(event.estimated_cost_usd),
            "cost_source": event.cost_source,
            "stop_reason": event.stop_reason,
            "checkpoint": asdict(event.checkpoint) if event.checkpoint else None,
            "copyback_manifest_hash": event.copyback_manifest_hash,
            "artifact_hashes": list(event.artifact_hashes),
            "receipt_hash": event.receipt_hash,
        }
    if isinstance(event, ConfigurationPromotion):
        payload = asdict(event)
        payload["decision_evidence_hashes"] = list(event.decision_evidence_hashes)
        return "configuration_promoted", payload
    raise TypeError("unsupported campaign event")


def _load_event(kind: object, row: object) -> CampaignEvent:
    if not isinstance(kind, str) or not isinstance(row, dict):
        raise ValueError("campaign event payload is invalid")
    if kind == "wave_opened":
        return WaveOpened(**row)
    if kind == "wave_closed":
        return WaveClosed(**row)
    if kind == "attempt_started":
        resume_row = row.get("resume")
        resume = None
        if resume_row is not None:
            if not isinstance(resume_row, dict) or set(resume_row) != {"prior_attempt_id", "prior_stop_receipt_hash", "checkpoint"}:
                raise ValueError("campaign resume binding fields are invalid")
            checkpoint = _checkpoint(resume_row["checkpoint"])
            if checkpoint is None:
                raise ValueError("campaign resume checkpoint is missing")
            resume = ResumeBinding(resume_row["prior_attempt_id"], resume_row["prior_stop_receipt_hash"], checkpoint)
        expected = {"request_id", "wave_id", "attempt_id", "quoted_hourly_usd", "started_at_epoch", "resume"}
        if set(row) != expected:
            raise ValueError("campaign attempt-start fields are invalid")
        return AttemptStart(
            row["request_id"], row["wave_id"], row["attempt_id"],
            _decimal(row["quoted_hourly_usd"], "quoted_hourly_usd"), row["started_at_epoch"], resume,
        )
    if kind == "attempt_stopped":
        expected = {
            "request_id", "wave_id", "attempt_id", "quoted_hourly_usd", "effective_hourly_usd",
            "started_at_epoch", "stopped_at_epoch", "billed_seconds", "billed_cost_usd",
            "estimated_cost_usd", "cost_source", "stop_reason", "checkpoint",
            "copyback_manifest_hash", "artifact_hashes", "receipt_hash",
        }
        if set(row) != expected or not isinstance(row["artifact_hashes"], list):
            raise ValueError("campaign attempt-stop fields are invalid")
        return AttemptReceipt(
            request_id=row["request_id"], wave_id=row["wave_id"], attempt_id=row["attempt_id"],
            quoted_hourly_usd=_decimal(row["quoted_hourly_usd"], "quoted_hourly_usd"),
            effective_hourly_usd=_decimal(row["effective_hourly_usd"], "effective_hourly_usd"),
            started_at_epoch=row["started_at_epoch"], stopped_at_epoch=row["stopped_at_epoch"],
            billed_seconds=row["billed_seconds"], billed_cost_usd=_decimal(row["billed_cost_usd"], "billed_cost_usd"),
            estimated_cost_usd=_decimal(row["estimated_cost_usd"], "estimated_cost_usd"),
            cost_source=row["cost_source"], stop_reason=row["stop_reason"], checkpoint=_checkpoint(row["checkpoint"]),
            copyback_manifest_hash=row["copyback_manifest_hash"], artifact_hashes=tuple(row["artifact_hashes"]),
            receipt_hash=row["receipt_hash"],
        )
    if kind == "configuration_promoted":
        values = dict(row)
        values["decision_evidence_hashes"] = tuple(values["decision_evidence_hashes"])
        return ConfigurationPromotion(**values)
    raise ValueError("unknown campaign event type")


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    if partial.exists():
        raise ValueError(f"unreconciled partial artifact exists: {partial.name}")
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


class CampaignStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.plan_path = self.root / "campaign-plan.json"
        self.events_root = self.root / "events"
        self._lock_handle: Any | None = None

    @classmethod
    def initialize(cls, root: Path, plan: CampaignPlan) -> "CampaignStore":
        target = root.expanduser().resolve()
        target.mkdir(mode=0o700, parents=True, exist_ok=False)
        os.chmod(target, 0o700)
        (target / "events").mkdir(mode=0o700)
        _write_atomic(target / "campaign-plan.json", campaign_plan_artifact(plan))
        return cls(target)

    @contextmanager
    def writer(self) -> Iterator["CampaignStore"]:
        if self._lock_handle is not None:
            raise RuntimeError("campaign writer is already held")
        handle = (self.root / ".writer.lock").open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise RuntimeError("campaign writer is held by another process") from error
        self._lock_handle = handle
        try:
            yield self
        finally:
            self._lock_handle = None
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def load(self) -> CampaignLedger:
        plan = load_campaign_plan(self.plan_path)
        if any(path.name.endswith(".partial") for path in self.events_root.iterdir()):
            raise ValueError("campaign event publication has an unreconciled partial")
        events: list[CampaignEvent] = []
        previous: str | None = None
        files = sorted(self.events_root.glob("*.json"))
        expected_outer = {
            "artifact_kind", "schema_version", "sequence", "previous_event_hash", "campaign_hash",
            "event_type", "event", "event_hash",
        }
        for sequence, path in enumerate(files, start=1):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != expected_outer:
                raise ValueError("campaign event artifact fields are invalid")
            expected_hash = stable_json_hash({key: value for key, value in payload.items() if key != "event_hash"})
            if (
                payload["artifact_kind"] != "twc_gpu_campaign_event_v1"
                or payload["schema_version"] != 1
                or payload["sequence"] != sequence
                or payload["previous_event_hash"] != previous
                or payload["campaign_hash"] != plan.identity
                or payload["event_hash"] != expected_hash
                or path.name != f"{sequence:06d}-{expected_hash}.json"
            ):
                raise ValueError("campaign event hash, order, or identity is invalid")
            events.append(_load_event(payload["event_type"], payload["event"]))
            previous = expected_hash
        return CampaignLedger(plan, tuple(events))

    def commit(self, ledger: CampaignLedger) -> CampaignLedger:
        if self._lock_handle is None:
            raise RuntimeError("campaign commit requires the writer lock")
        current = self.load()
        if ledger.plan.identity != current.plan.identity:
            raise ValueError("campaign commit plan identity mismatch")
        if ledger.events == current.events:
            return current
        if len(ledger.events) != len(current.events) + 1 or ledger.events[: len(current.events)] != current.events:
            raise ValueError("campaign commit must append exactly one event to current state")
        event = ledger.events[-1]
        kind, event_payload = _event_payload(event)
        previous_files = sorted(self.events_root.glob("*.json"))
        previous_hash = None
        if previous_files:
            previous_hash = json.loads(previous_files[-1].read_text(encoding="utf-8"))["event_hash"]
        sequence = len(current.events) + 1
        payload: dict[str, Any] = {
            "artifact_kind": "twc_gpu_campaign_event_v1",
            "schema_version": 1,
            "sequence": sequence,
            "previous_event_hash": previous_hash,
            "campaign_hash": current.plan.identity,
            "event_type": kind,
            "event": event_payload,
        }
        payload["event_hash"] = stable_json_hash(payload)
        _write_atomic(self.events_root / f"{sequence:06d}-{payload['event_hash']}.json", payload)
        return self.load()
