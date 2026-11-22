"""Provider-neutral, immutable orchestration contracts for GPU experiment campaigns.

This module owns scientific/execution identities, exact budget policy, append-only
lifecycle transitions, interruption resume bindings, and evidence freshness.  It
does not provision hardware, execute training, persist secrets, or unlock the
production pretraining path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from typing import Any, Literal

from twc.repro.hashing import stable_json_hash


Money = Decimal
WavePurpose = Literal[
    "diagnostic_canary",
    "corrective_recanary",
    "configuration_shape_probe",
    "configuration",
]
StopReason = Literal["completed", "controlled_stop", "spot_interruption", "failed"]

_MONEY_PLACES = Decimal("0.000001")
_PROOF_RANK = {"local_only": 0, "targeted_proof": 1, "full_canary": 2}
_EVIDENCE_STALENESS = {
    "docs": "local_only",
    "reporting": "local_only",
    "tests": "local_only",
    "data_reader": "targeted_proof",
    "checkpoint": "targeted_proof",
    "telemetry": "targeted_proof",
    "evaluation": "targeted_proof",
    "model_math": "full_canary",
    "backend": "full_canary",
    "environment": "full_canary",
}


def _validate_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, whitespace-trimmed string")


def _validate_int(value: int, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _validate_money(value: Money, name: str, *, positive: bool = False) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    if value.as_tuple().exponent < -6:
        raise ValueError(f"{name} supports at most six decimal places")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")


def _money(value: Money) -> str:
    return format(value, "f")


def _hash_payload(payload: dict[str, Any]) -> str:
    return stable_json_hash(payload)


@dataclass(frozen=True, slots=True)
class EvidenceDependency:
    """A named evidence input whose content hash controls proof freshness."""

    component: str
    evidence_hash: str

    def __post_init__(self) -> None:
        if self.component not in _EVIDENCE_STALENESS:
            raise ValueError(f"unknown evidence dependency: {self.component}")
        _validate_hash(self.evidence_hash, "evidence_hash")


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    dependencies: tuple[EvidenceDependency, ...]
    snapshot_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(item.component for item in self.dependencies)
        if not names or len(names) != len(set(names)):
            raise ValueError("evidence snapshot dependencies must be non-empty and unique")
        payload = [{"component": item.component, "evidence_hash": item.evidence_hash} for item in self.dependencies]
        object.__setattr__(self, "snapshot_hash", _hash_payload({"dependencies": payload}))


@dataclass(frozen=True, slots=True)
class AttemptPlan:
    attempt_id: str
    model_config_hash: str
    data_order_hash: str
    execution_path_hash: str
    max_cost_usd: Money
    recovery_mode: str = "none"
    predecessor_attempt_id: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.attempt_id, "attempt_id")
        for name in ("model_config_hash", "data_order_hash", "execution_path_hash"):
            _validate_hash(getattr(self, name), name)
        _validate_money(self.max_cost_usd, "max_cost_usd", positive=True)
        if self.recovery_mode not in {"none", "fresh_restart"}:
            raise ValueError("unsupported attempt recovery mode")
        if self.recovery_mode == "fresh_restart":
            if self.predecessor_attempt_id is None:
                raise ValueError("fresh restart requires a predecessor attempt")
            _validate_identifier(self.predecessor_attempt_id, "predecessor_attempt_id")
        elif self.predecessor_attempt_id is not None:
            raise ValueError("ordinary attempt cannot declare a predecessor")

    @property
    def identity(self) -> str:
        payload = {
                "artifact_kind": "twc_experiment_attempt_plan_v1",
                "attempt_id": self.attempt_id,
                "model_config_hash": self.model_config_hash,
                "data_order_hash": self.data_order_hash,
                "execution_path_hash": self.execution_path_hash,
                "max_cost_usd": _money(self.max_cost_usd),
            }
        if self.recovery_mode != "none":
            payload.update(
                recovery_mode=self.recovery_mode,
                predecessor_attempt_id=self.predecessor_attempt_id,
            )
        return _hash_payload(payload)


@dataclass(frozen=True, slots=True)
class WavePlan:
    wave_id: str
    purpose: WavePurpose
    cap_usd: Money
    attempts: tuple[AttemptPlan, ...]
    prerequisite_wave_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.wave_id, "wave_id")
        if self.purpose not in {
            "diagnostic_canary",
            "corrective_recanary",
            "configuration_shape_probe",
            "configuration",
        }:
            raise ValueError("unsupported campaign wave purpose")
        _validate_money(self.cap_usd, "cap_usd", positive=True)
        if not self.attempts:
            raise ValueError("campaign wave must contain predeclared attempts")
        attempt_ids = tuple(attempt.attempt_id for attempt in self.attempts)
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("campaign wave contains a duplicate attempt id")
        if sum((attempt.max_cost_usd for attempt in self.attempts), Decimal("0")) > self.cap_usd:
            raise ValueError("predeclared attempt caps exceed wave cap")
        if len(self.prerequisite_wave_ids) != len(set(self.prerequisite_wave_ids)) or self.wave_id in self.prerequisite_wave_ids:
            raise ValueError("wave prerequisites must be unique and cannot include the wave itself")

    def attempt(self, attempt_id: str) -> AttemptPlan:
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        raise ValueError(f"attempt is not predeclared in wave {self.wave_id}: {attempt_id}")

    @property
    def identity(self) -> str:
        return _hash_payload(
            {
                "artifact_kind": "twc_experiment_wave_plan_v1",
                "wave_id": self.wave_id,
                "purpose": self.purpose,
                "cap_usd": _money(self.cap_usd),
                "attempt_hashes": [attempt.identity for attempt in self.attempts],
                "prerequisite_wave_ids": list(self.prerequisite_wave_ids),
            }
        )


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    campaign_id: str
    budget_cap_usd: Money
    reserve_usd: Money
    code_hash: str
    evaluation_policy_hash: str
    waves: tuple[WavePlan, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.campaign_id, "campaign_id")
        _validate_money(self.budget_cap_usd, "budget_cap_usd", positive=True)
        _validate_money(self.reserve_usd, "reserve_usd")
        if self.reserve_usd >= self.budget_cap_usd:
            raise ValueError("campaign reserve must be smaller than the budget cap")
        _validate_hash(self.code_hash, "code_hash")
        _validate_hash(self.evaluation_policy_hash, "evaluation_policy_hash")
        wave_ids = tuple(wave.wave_id for wave in self.waves)
        if not wave_ids or len(wave_ids) != len(set(wave_ids)):
            raise ValueError("campaign waves must be non-empty and uniquely identified")
        known: set[str] = set()
        attempts_by_id: dict[str, AttemptPlan] = {}
        successors_by_predecessor: dict[str, str] = {}
        for wave in self.waves:
            if any(prerequisite not in known for prerequisite in wave.prerequisite_wave_ids):
                raise ValueError("wave prerequisites must refer to earlier predeclared waves")
            known.add(wave.wave_id)
            for attempt in wave.attempts:
                if attempt.attempt_id in attempts_by_id:
                    raise ValueError("attempt ids must be globally unique within a campaign")
                if attempt.recovery_mode == "fresh_restart":
                    predecessor = attempts_by_id.get(str(attempt.predecessor_attempt_id))
                    if predecessor is None:
                        raise ValueError("fresh-restart predecessor must be an earlier campaign attempt")
                    if str(attempt.predecessor_attempt_id) in successors_by_predecessor:
                        raise ValueError("campaign permits only one fresh-restart successor per predecessor")
                    if any(
                        getattr(attempt, field) != getattr(predecessor, field)
                        for field in (
                            "model_config_hash", "data_order_hash", "execution_path_hash"
                        )
                    ):
                        raise ValueError("fresh-restart scientific identities must match the predecessor")
                    successors_by_predecessor[str(attempt.predecessor_attempt_id)] = attempt.attempt_id
                attempts_by_id[attempt.attempt_id] = attempt

    def wave(self, wave_id: str) -> WavePlan:
        for wave in self.waves:
            if wave.wave_id == wave_id:
                return wave
        raise ValueError(f"wave is not predeclared: {wave_id}")

    @property
    def identity(self) -> str:
        return _hash_payload(
            {
                "artifact_kind": "twc_experiment_campaign_plan_v1",
                "campaign_id": self.campaign_id,
                "budget_cap_usd": _money(self.budget_cap_usd),
                "reserve_usd": _money(self.reserve_usd),
                "code_hash": self.code_hash,
                "evaluation_policy_hash": self.evaluation_policy_hash,
                "wave_hashes": [wave.identity for wave in self.waves],
            }
        )


@dataclass(frozen=True, slots=True)
class CheckpointBinding:
    checkpoint_manifest_hash: str
    generation: int
    step: int
    consumed_loss_tokens: int
    next_global_ordinal: int

    def __post_init__(self) -> None:
        _validate_hash(self.checkpoint_manifest_hash, "checkpoint_manifest_hash")
        for name in ("generation", "step", "consumed_loss_tokens", "next_global_ordinal"):
            _validate_int(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ResumeBinding:
    prior_attempt_id: str
    prior_stop_receipt_hash: str
    checkpoint: CheckpointBinding

    def __post_init__(self) -> None:
        _validate_identifier(self.prior_attempt_id, "prior_attempt_id")
        _validate_hash(self.prior_stop_receipt_hash, "prior_stop_receipt_hash")


@dataclass(frozen=True, slots=True)
class WaveOpened:
    request_id: str
    wave_id: str

    def __post_init__(self) -> None:
        _validate_identifier(self.request_id, "request_id")
        _validate_identifier(self.wave_id, "wave_id")


@dataclass(frozen=True, slots=True)
class WaveClosed:
    request_id: str
    wave_id: str
    reason: str

    def __post_init__(self) -> None:
        _validate_identifier(self.request_id, "request_id")
        _validate_identifier(self.wave_id, "wave_id")
        _validate_identifier(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class AttemptStart:
    request_id: str
    wave_id: str
    attempt_id: str
    quoted_hourly_usd: Money
    started_at_epoch: int
    resume: ResumeBinding | None

    def __post_init__(self) -> None:
        for name in ("request_id", "wave_id", "attempt_id"):
            _validate_identifier(getattr(self, name), name)
        _validate_money(self.quoted_hourly_usd, "quoted_hourly_usd", positive=True)
        _validate_int(self.started_at_epoch, "started_at_epoch", minimum=1)


@dataclass(frozen=True, slots=True)
class AttemptStop:
    request_id: str
    wave_id: str
    attempt_id: str
    effective_hourly_usd: Money
    stopped_at_epoch: int
    billed_seconds: int
    billed_cost_usd: Money
    estimated_cost_usd: Money
    cost_source: Literal["provider_billed", "rate_estimate"]
    stop_reason: StopReason
    checkpoint: CheckpointBinding | None
    copyback_manifest_hash: str
    artifact_hashes: tuple[str, ...]
    request_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("request_id", "wave_id", "attempt_id"):
            _validate_identifier(getattr(self, name), name)
        _validate_money(self.effective_hourly_usd, "effective_hourly_usd", positive=True)
        _validate_int(self.stopped_at_epoch, "stopped_at_epoch", minimum=1)
        _validate_int(self.billed_seconds, "billed_seconds", minimum=1)
        _validate_money(self.billed_cost_usd, "billed_cost_usd", positive=True)
        _validate_money(self.estimated_cost_usd, "estimated_cost_usd", positive=True)
        if self.cost_source not in {"provider_billed", "rate_estimate"}:
            raise ValueError("unsupported cost source")
        if self.stop_reason not in {"completed", "controlled_stop", "spot_interruption", "failed"}:
            raise ValueError("unsupported stop reason")
        expected = self.effective_hourly_usd * Decimal(self.billed_seconds) / Decimal(3600)
        if abs(self.estimated_cost_usd - expected) > _MONEY_PLACES:
            raise ValueError("estimated cost arithmetic does not match effective price and billed time")
        if abs(self.billed_cost_usd - expected) > _MONEY_PLACES:
            source = "rate-estimate" if self.cost_source == "rate_estimate" else "provider billed"
            raise ValueError(
                f"{source} billed cost arithmetic does not match effective price and billed time"
            )
        _validate_hash(self.copyback_manifest_hash, "copyback manifest hash")
        if not self.artifact_hashes or len(self.artifact_hashes) != len(set(self.artifact_hashes)):
            raise ValueError("copyback artifact hashes must be non-empty and unique")
        for value in self.artifact_hashes:
            _validate_hash(value, "copyback artifact hash")
        object.__setattr__(self, "request_hash", _hash_payload(self._payload()))

    @classmethod
    def create(cls, **values: Any) -> "AttemptStop":
        return cls(**values)

    def _payload(self) -> dict[str, Any]:
        checkpoint = asdict(self.checkpoint) if self.checkpoint is not None else None
        return {
            "artifact_kind": "twc_experiment_attempt_stop_request_v1",
            "request_id": self.request_id,
            "wave_id": self.wave_id,
            "attempt_id": self.attempt_id,
            "effective_hourly_usd": _money(self.effective_hourly_usd),
            "stopped_at_epoch": self.stopped_at_epoch,
            "billed_seconds": self.billed_seconds,
            "billed_cost_usd": _money(self.billed_cost_usd),
            "estimated_cost_usd": _money(self.estimated_cost_usd),
            "cost_source": self.cost_source,
            "stop_reason": self.stop_reason,
            "checkpoint": checkpoint,
            "copyback_manifest_hash": self.copyback_manifest_hash,
            "artifact_hashes": list(self.artifact_hashes),
        }


@dataclass(frozen=True, slots=True)
class AttemptReceipt:
    request_id: str
    wave_id: str
    attempt_id: str
    quoted_hourly_usd: Money
    effective_hourly_usd: Money
    started_at_epoch: int
    stopped_at_epoch: int
    billed_seconds: int
    billed_cost_usd: Money
    estimated_cost_usd: Money
    cost_source: str
    stop_reason: str
    checkpoint: CheckpointBinding | None
    copyback_manifest_hash: str
    artifact_hashes: tuple[str, ...]
    receipt_hash: str

    def __post_init__(self) -> None:
        stop = AttemptStop.create(
            request_id=self.request_id,
            wave_id=self.wave_id,
            attempt_id=self.attempt_id,
            effective_hourly_usd=self.effective_hourly_usd,
            stopped_at_epoch=self.stopped_at_epoch,
            billed_seconds=self.billed_seconds,
            billed_cost_usd=self.billed_cost_usd,
            estimated_cost_usd=self.estimated_cost_usd,
            cost_source=self.cost_source,
            stop_reason=self.stop_reason,
            checkpoint=self.checkpoint,
            copyback_manifest_hash=self.copyback_manifest_hash,
            artifact_hashes=self.artifact_hashes,
        )
        _validate_money(self.quoted_hourly_usd, "quoted_hourly_usd", positive=True)
        _validate_int(self.started_at_epoch, "started_at_epoch", minimum=1)
        payload = {
            **stop._payload(),
            "artifact_kind": "twc_experiment_attempt_receipt_v1",
            "quoted_hourly_usd": _money(self.quoted_hourly_usd),
            "started_at_epoch": self.started_at_epoch,
        }
        _validate_hash(self.receipt_hash, "attempt receipt hash")
        if self.receipt_hash != _hash_payload(payload):
            raise ValueError("attempt receipt hash mismatch")


@dataclass(frozen=True, slots=True)
class ConfigurationPromotion:
    request_id: str
    wave_id: str
    decision_evidence_hashes: tuple[str, ...]
    configuration_promoted: bool
    production_unlock: bool
    f1_unlock: bool
    receipt_hash: str

    def __post_init__(self) -> None:
        _validate_identifier(self.request_id, "request_id")
        _validate_identifier(self.wave_id, "wave_id")
        if len(self.decision_evidence_hashes) < 2 or len(self.decision_evidence_hashes) != len(set(self.decision_evidence_hashes)):
            raise ValueError("promotion requires at least two unique decision evidence hashes")
        for value in self.decision_evidence_hashes:
            _validate_hash(value, "decision evidence hash")
        if self.configuration_promoted is not True or self.production_unlock is not False or self.f1_unlock is not False:
            raise ValueError("experiment configuration promotion cannot unlock production or F1")
        payload = {
            "artifact_kind": "twc_experiment_configuration_promotion_v1",
            "request_id": self.request_id,
            "wave_id": self.wave_id,
            "decision_evidence_hashes": list(self.decision_evidence_hashes),
            "configuration_promoted": True,
            "production_unlock": False,
            "f1_unlock": False,
        }
        _validate_hash(self.receipt_hash, "configuration promotion receipt hash")
        if self.receipt_hash != _hash_payload(payload):
            raise ValueError("configuration promotion receipt hash mismatch")


CampaignEvent = WaveOpened | WaveClosed | AttemptStart | AttemptReceipt | ConfigurationPromotion


@dataclass(frozen=True, slots=True)
class CampaignLedger:
    """Immutable append-only campaign lifecycle with idempotent request handling."""

    plan: CampaignPlan
    events: tuple[CampaignEvent, ...] = ()

    def __post_init__(self) -> None:
        self._validate_event_chain()

    def _request_event(self, request_id: str) -> CampaignEvent | None:
        for event in self.events:
            if event.request_id == request_id:
                return event
        return None

    def _validate_event_chain(self) -> None:
        """Replay persisted events so direct construction cannot bypass policy."""

        request_ids: set[str] = set()
        opened: set[str] = set()
        closed: set[str] = set()
        starts: dict[str, AttemptStart] = {}
        stops: dict[str, AttemptReceipt] = {}
        cumulative_cost = Decimal("0")
        wave_costs: dict[str, Money] = {}
        for event in self.events:
            if event.request_id in request_ids:
                raise ValueError(f"duplicate request_id in campaign event chain: {event.request_id}")
            request_ids.add(event.request_id)
            wave = self.plan.wave(event.wave_id)
            if isinstance(event, WaveOpened):
                if event.wave_id in opened or event.wave_id in closed:
                    raise ValueError("campaign event chain reopens an existing wave")
                if any(prerequisite not in closed for prerequisite in wave.prerequisite_wave_ids):
                    raise ValueError("campaign event chain opens a wave before its prerequisites")
                active_commitment = sum(
                    (
                        max(
                            Decimal("0"),
                            self.plan.wave(active).cap_usd - wave_costs.get(active, Decimal("0")),
                        )
                        for active in opened - closed
                    ),
                    Decimal("0"),
                )
                if cumulative_cost + active_commitment + wave.cap_usd + self.plan.reserve_usd > self.plan.budget_cap_usd:
                    raise ValueError("campaign event chain exceeds cap plus reserve")
                opened.add(event.wave_id)
            elif isinstance(event, AttemptStart):
                if event.wave_id not in opened - closed:
                    raise ValueError("campaign event chain starts an attempt outside an open wave")
                attempt_plan = wave.attempt(event.attempt_id)
                if event.attempt_id in starts:
                    raise ValueError("campaign event chain starts an attempt more than once")
                if event.resume is not None:
                    prior = stops.get(event.resume.prior_attempt_id)
                    if (
                        prior is None
                        or prior.wave_id != event.wave_id
                        or prior.stop_reason != "spot_interruption"
                        or prior.receipt_hash != event.resume.prior_stop_receipt_hash
                        or prior.checkpoint != event.resume.checkpoint
                    ):
                        raise ValueError("campaign event chain contains an invalid resume binding")
                if attempt_plan.recovery_mode == "fresh_restart":
                    predecessor = stops.get(str(attempt_plan.predecessor_attempt_id))
                    if (
                        event.resume is not None
                        or predecessor is None
                        or predecessor.stop_reason not in {"spot_interruption", "failed"}
                    ):
                        raise ValueError("campaign event chain contains an invalid fresh restart")
                if wave_costs.get(event.wave_id, Decimal("0")) + wave.attempt(event.attempt_id).max_cost_usd > wave.cap_usd:
                    raise ValueError("campaign event chain starts an attempt without remaining wave budget")
                starts[event.attempt_id] = event
            elif isinstance(event, AttemptReceipt):
                start = starts.get(event.attempt_id)
                if start is None or start.wave_id != event.wave_id or event.attempt_id in stops:
                    raise ValueError("campaign event chain contains an unbound or duplicate attempt receipt")
                if event.stopped_at_epoch < start.started_at_epoch:
                    raise ValueError("campaign event chain stops an attempt before it starts")
                attempt = wave.attempt(event.attempt_id)
                if event.billed_cost_usd > attempt.max_cost_usd:
                    raise ValueError("campaign event chain exceeds attempt cost cap")
                stops[event.attempt_id] = event
                cumulative_cost += event.billed_cost_usd
                wave_costs[event.wave_id] = wave_costs.get(event.wave_id, Decimal("0")) + event.billed_cost_usd
                if wave_costs[event.wave_id] > wave.cap_usd:
                    raise ValueError("campaign event chain exceeds wave cost cap")
                if cumulative_cost + self.plan.reserve_usd > self.plan.budget_cap_usd:
                    raise ValueError("campaign event chain exceeds budget cap plus reserve")
            elif isinstance(event, WaveClosed):
                if event.wave_id not in opened - closed:
                    raise ValueError("campaign event chain closes a wave that is not open")
                if any(
                    attempt.attempt_id in starts and attempt.attempt_id not in stops
                    for attempt in wave.attempts
                ):
                    raise ValueError("campaign event chain closes a wave with an active attempt")
                closed.add(event.wave_id)
            elif isinstance(event, ConfigurationPromotion):
                if wave.purpose != "configuration" or event.wave_id not in opened - closed:
                    raise ValueError("campaign event chain contains an invalid configuration promotion")
                if any(
                    not self._configuration_attempt_satisfied(attempt, stops)
                    for attempt in wave.attempts
                ):
                    raise ValueError("campaign event chain promotes before every attempt stops")
                closed.add(event.wave_id)
            else:  # pragma: no cover - closed union; protects untyped persisted callers.
                raise TypeError("unsupported campaign event type")

    def _append(self, event: CampaignEvent) -> "CampaignLedger":
        prior = self._request_event(event.request_id)
        if prior is not None:
            if prior == event:
                return self
            raise ValueError(f"request_id conflict: {event.request_id}")
        return replace(self, events=(*self.events, event))

    @property
    def open_wave_ids(self) -> tuple[str, ...]:
        opened = [event.wave_id for event in self.events if isinstance(event, WaveOpened)]
        closed = {event.wave_id for event in self.events if isinstance(event, (WaveClosed, ConfigurationPromotion))}
        return tuple(wave_id for wave_id in opened if wave_id not in closed)

    @property
    def cumulative_cost_usd(self) -> Money:
        return sum((event.billed_cost_usd for event in self.events if isinstance(event, AttemptReceipt)), Decimal("0"))

    @property
    def remaining_unreserved_usd(self) -> Money:
        remaining_active_commitment = sum(
            (
                max(
                    Decimal("0"),
                    self.plan.wave(wave_id).cap_usd - self._wave_cost(wave_id),
                )
                for wave_id in self.open_wave_ids
            ),
            Decimal("0"),
        )
        return (
            self.plan.budget_cap_usd
            - self.plan.reserve_usd
            - self.cumulative_cost_usd
            - remaining_active_commitment
        )

    def open_wave(self, wave_id: str, *, request_id: str) -> "CampaignLedger":
        event = WaveOpened(request_id=request_id, wave_id=wave_id)
        prior = self._request_event(request_id)
        if prior is not None:
            return self._append(event)
        wave = self.plan.wave(wave_id)
        if wave_id in self.open_wave_ids or self._wave_closed(wave_id):
            raise ValueError("campaign wave has already been opened")
        if any(not self._wave_closed(prerequisite) for prerequisite in wave.prerequisite_wave_ids):
            raise ValueError("campaign wave prerequisites are incomplete")
        if wave.cap_usd > self.remaining_unreserved_usd:
            raise ValueError("wave cap plus reserve is unavailable under the campaign budget")
        return self._append(event)

    def close_wave(self, wave_id: str, *, request_id: str, reason: str) -> "CampaignLedger":
        event = WaveClosed(request_id=request_id, wave_id=wave_id, reason=reason)
        prior = self._request_event(request_id)
        if prior is not None:
            return self._append(event)
        if wave_id not in self.open_wave_ids:
            raise ValueError("campaign wave is not open")
        if any(self._active_start(attempt.attempt_id) is not None for attempt in self.plan.wave(wave_id).attempts):
            raise ValueError("campaign wave cannot close with an active attempt")
        return self._append(event)

    def start_attempt(self, start: AttemptStart) -> "CampaignLedger":
        prior = self._request_event(start.request_id)
        if prior is not None:
            return self._append(start)
        if start.wave_id not in self.open_wave_ids:
            raise ValueError("attempt wave is not open")
        wave = self.plan.wave(start.wave_id)
        attempt = wave.attempt(start.attempt_id)
        if self.attempt_start(start.attempt_id, required=False) is not None:
            raise ValueError("attempt has already started")
        if self._wave_cost(start.wave_id) + attempt.max_cost_usd > wave.cap_usd:
            raise ValueError("attempt cap is unavailable within the remaining wave budget")
        if start.resume is not None:
            self._validate_resume(start)
        if attempt.recovery_mode == "fresh_restart":
            predecessor = self.attempt_stop(str(attempt.predecessor_attempt_id), required=False)
            if (
                start.resume is not None
                or predecessor is None
                or predecessor.stop_reason not in {"spot_interruption", "failed"}
            ):
                raise ValueError("fresh restart requires the exact failed or interrupted predecessor")
        return self._append(start)

    def stop_attempt(self, stop: AttemptStop) -> "CampaignLedger":
        prior = self._request_event(stop.request_id)
        if prior is not None:
            if isinstance(prior, AttemptReceipt) and prior.receipt_hash == self._receipt(stop, self.attempt_start(stop.attempt_id)).receipt_hash:
                return self
            raise ValueError(f"request_id conflict: {stop.request_id}")
        start = self.attempt_start(stop.attempt_id)
        if start.wave_id != stop.wave_id:
            raise ValueError("attempt stop wave binding mismatch")
        if self.attempt_stop(stop.attempt_id, required=False) is not None:
            raise ValueError("attempt has already stopped")
        if stop.stopped_at_epoch < start.started_at_epoch:
            raise ValueError("attempt stop precedes its start")
        wave = self.plan.wave(stop.wave_id)
        attempt = wave.attempt(stop.attempt_id)
        if stop.billed_cost_usd > attempt.max_cost_usd:
            raise ValueError("attempt billed cost exceeds its predeclared cap")
        if self._wave_cost(stop.wave_id) + stop.billed_cost_usd > wave.cap_usd:
            raise ValueError("attempt billed cost exceeds the wave cap")
        if (
            self.cumulative_cost_usd
            + stop.billed_cost_usd
            + self.plan.reserve_usd
            > self.plan.budget_cap_usd
        ):
            raise ValueError("attempt billed cost exceeds campaign cap plus reserve")
        return self._append(self._receipt(stop, start))

    def promote_wave(
        self,
        wave_id: str,
        *,
        request_id: str,
        decision_evidence_hashes: tuple[str, ...],
    ) -> "CampaignLedger":
        for value in decision_evidence_hashes:
            _validate_hash(value, "decision evidence hash")
        if len(decision_evidence_hashes) < 2 or len(decision_evidence_hashes) != len(set(decision_evidence_hashes)):
            raise ValueError("promotion requires at least two unique decision evidence hashes")
        payload = {
            "artifact_kind": "twc_experiment_configuration_promotion_v1",
            "request_id": request_id,
            "wave_id": wave_id,
            "decision_evidence_hashes": list(decision_evidence_hashes),
            "configuration_promoted": True,
            "production_unlock": False,
            "f1_unlock": False,
        }
        event = ConfigurationPromotion(
            request_id=request_id,
            wave_id=wave_id,
            decision_evidence_hashes=decision_evidence_hashes,
            configuration_promoted=True,
            production_unlock=False,
            f1_unlock=False,
            receipt_hash=_hash_payload(payload),
        )
        prior = self._request_event(request_id)
        if prior is not None:
            return self._append(event)
        wave = self.plan.wave(wave_id)
        if wave.purpose != "configuration":
            raise ValueError("only a configuration wave can produce a configuration promotion")
        if wave_id not in self.open_wave_ids:
            raise ValueError("configuration wave is not open")
        stops = {
            attempt.attempt_id: stop
            for planned_wave in self.plan.waves
            for attempt in planned_wave.attempts
            if (stop := self.attempt_stop(attempt.attempt_id, required=False)) is not None
        }
        if any(
            not self._configuration_attempt_satisfied(attempt, stops)
            for attempt in wave.attempts
        ):
            raise ValueError("all required configuration attempts must complete before promotion")
        return self._append(event)

    @staticmethod
    def _configuration_attempt_satisfied(
        attempt: AttemptPlan, stops: dict[str, AttemptReceipt]
    ) -> bool:
        stop = stops.get(attempt.attempt_id)
        if attempt.recovery_mode == "none":
            return stop is not None and stop.stop_reason == "completed"
        predecessor = stops.get(str(attempt.predecessor_attempt_id))
        if predecessor is None:
            return False
        if predecessor.stop_reason == "completed":
            return stop is None
        return stop is not None and stop.stop_reason == "completed"

    def attempt_start(self, attempt_id: str, *, required: bool = True) -> AttemptStart | None:
        for event in self.events:
            if isinstance(event, AttemptStart) and event.attempt_id == attempt_id:
                return event
        if required:
            raise ValueError(f"attempt has not started: {attempt_id}")
        return None

    def attempt_stop(self, attempt_id: str, *, required: bool = True) -> AttemptReceipt | None:
        for event in self.events:
            if isinstance(event, AttemptReceipt) and event.attempt_id == attempt_id:
                return event
        if required:
            raise ValueError(f"attempt has not stopped: {attempt_id}")
        return None

    def promotion(self, wave_id: str) -> ConfigurationPromotion:
        for event in self.events:
            if isinstance(event, ConfigurationPromotion) and event.wave_id == wave_id:
                return event
        raise ValueError(f"wave has no configuration promotion: {wave_id}")

    def _wave_closed(self, wave_id: str) -> bool:
        return any(
            isinstance(event, (WaveClosed, ConfigurationPromotion)) and event.wave_id == wave_id
            for event in self.events
        )

    def _wave_cost(self, wave_id: str) -> Money:
        return sum(
            (
                event.billed_cost_usd
                for event in self.events
                if isinstance(event, AttemptReceipt) and event.wave_id == wave_id
            ),
            Decimal("0"),
        )

    def _active_start(self, attempt_id: str) -> AttemptStart | None:
        start = self.attempt_start(attempt_id, required=False)
        if start is None or self.attempt_stop(attempt_id, required=False) is not None:
            return None
        return start

    def _validate_resume(self, start: AttemptStart) -> None:
        assert start.resume is not None
        prior = self.attempt_stop(start.resume.prior_attempt_id, required=False)
        if (
            prior is None
            or prior.wave_id != start.wave_id
            or prior.stop_reason != "spot_interruption"
            or prior.receipt_hash != start.resume.prior_stop_receipt_hash
            or prior.checkpoint != start.resume.checkpoint
        ):
            raise ValueError("resume binding does not match an exact interrupted attempt checkpoint")

    @staticmethod
    def _receipt(stop: AttemptStop, start: AttemptStart) -> AttemptReceipt:
        payload = {
            **stop._payload(),
            "artifact_kind": "twc_experiment_attempt_receipt_v1",
            "quoted_hourly_usd": _money(start.quoted_hourly_usd),
            "started_at_epoch": start.started_at_epoch,
        }
        return AttemptReceipt(
            request_id=stop.request_id,
            wave_id=stop.wave_id,
            attempt_id=stop.attempt_id,
            quoted_hourly_usd=start.quoted_hourly_usd,
            effective_hourly_usd=stop.effective_hourly_usd,
            started_at_epoch=start.started_at_epoch,
            stopped_at_epoch=stop.stopped_at_epoch,
            billed_seconds=stop.billed_seconds,
            billed_cost_usd=stop.billed_cost_usd,
            estimated_cost_usd=stop.estimated_cost_usd,
            cost_source=stop.cost_source,
            stop_reason=stop.stop_reason,
            checkpoint=stop.checkpoint,
            copyback_manifest_hash=stop.copyback_manifest_hash,
            artifact_hashes=stop.artifact_hashes,
            receipt_hash=_hash_payload(payload),
        )


def classify_evidence_staleness(changed_components: tuple[str, ...]) -> str:
    """Return the strongest proof refresh invalidated by the named changes."""

    if not changed_components:
        raise ValueError("at least one evidence dependency change is required")
    requirements: list[str] = []
    for component in changed_components:
        try:
            requirements.append(_EVIDENCE_STALENESS[component])
        except KeyError as error:
            raise ValueError(f"unknown evidence dependency: {component}") from error
    return max(requirements, key=_PROOF_RANK.__getitem__)


def authorize_canary_pass(pass_number: int, *, justification: str | None) -> str:
    """Bound the diagnostic/fix/retest loop without hiding justified exceptions."""

    _validate_int(pass_number, "pass_number", minimum=1)
    if pass_number == 1:
        return "diagnostic"
    if pass_number == 2:
        return "corrective_recanary"
    if justification is None or not justification.strip():
        raise ValueError("a third canary pass requires a concrete new-evidence and cost justification")
    return "exception_recanary"
