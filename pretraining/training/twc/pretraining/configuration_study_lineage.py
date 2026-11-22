"""Strict, no-raw lineage receipts for adopting C1 into C2/C3.

The module deliberately reconstructs C1 from its preserved campaign evidence
root.  It does not accept caller-authored result hashes, screen decisions, or
unlock claims as adoption inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from twc.pretraining.campaign import AttemptReceipt, ConfigurationPromotion
from twc.pretraining.campaign_store import CampaignStore
from twc.pretraining.data_pipeline.contracts import hash_without
from twc.pretraining.gpu_provider_evidence import (
    finalize_configuration_provider_billing,
)
from twc.pretraining.training.experiments import (
    ConfigurationScreenDecisionReceipt,
    PairedComparisonReceipt,
    load_evaluation_manifest,
    load_experiment_result_receipt,
    load_experiment_training_contract,
)
from twc.repro.hashing import stable_json_hash
from twc.repro.public_artifacts import validate_public_no_raw_payload


_HASH_CHARACTERS = frozenset("0123456789abcdef")
_C1_PROFILES = ("s_wide", "s_deep")
_C1_LEARNING_RATES = (("400", 4e-4), ("600", 6e-4), ("900", 9e-4))
_C1_CAMPAIGN_ID = "twc-c1-configuration-screen-v1"
_C1_CLAIM_STATUS = "practical_scaling_observation_not_causal"

_CLOSEOUT_FIELDS = {
    "artifact_kind",
    "schema_version",
    "campaign_hash",
    "promotion_receipt_hash",
    "attempt_receipt_hashes",
    "copyback_manifest_hashes",
    "result_hashes",
    "shared_copyback_manifest_hash",
    "provider",
    "pod_identity_digest",
    "provider_offer_hash",
    "pre_teardown_provider_observation_hash",
    "pre_teardown_observed_at_epoch",
    "pre_teardown_provider_total_usd",
    "attempt_estimated_total_usd",
    "provider_authorized_total_cap_usd",
    "billing_finalized",
    "pod_teardown_authorized",
    "persistent_disk_teardown_authorized",
    "destructive_action_performed",
    "production_unlock",
    "f1_unlock",
    "raw_text_serialized",
    "secrets_serialized",
    "receipt_hash",
}
_BILLING_FIELDS = {
    "artifact_kind",
    "schema_version",
    "campaign_hash",
    "closeout_receipt_hash",
    "final_provider_observation_hash",
    "provider_final_total_usd",
    "provider_authorized_total_cap_usd",
    "attempt_estimated_total_usd",
    "provider_minus_attempt_estimate_usd",
    "billing_finalized",
    "production_unlock",
    "f1_unlock",
    "raw_text_serialized",
    "secrets_serialized",
    "receipt_hash",
}
_FINITE_VIEW_FIELDS = {
    "artifact_kind",
    "schema_version",
    "pack_manifest_hash",
    "split",
    "loss_token_count",
    "full_segment_count",
    "final_segment_loss_tokens",
    "finite",
    "repeat_allowed",
    "order_policy",
    "raw_text_serialized",
    "raw_url_serialized",
    "token_serialized",
    "view_hash",
}


def _require_hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HASH_CHARACTERS for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} must contain valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_hash_bound(
    payload: Mapping[str, Any], *, fields: set[str], hash_field: str, label: str
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError(f"{label} fields are invalid")
    expected = stable_json_hash(
        {key: value for key, value in payload.items() if key != hash_field}
    )
    if payload.get(hash_field) != expected:
        raise ValueError(f"{label} hash mismatch")
    return dict(payload)


def load_finite_sequence_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly load the hash-bound finite-view summary used by study lineage."""

    if not isinstance(payload, Mapping) or set(payload) != _FINITE_VIEW_FIELDS:
        raise ValueError("finite sequence view fields are invalid")
    view = dict(payload)
    if (
        view.get("artifact_kind") != "twc_finite_sequence_view_v1"
        or view.get("schema_version") != 1
        or view.get("view_hash") != hash_without(view, "view_hash")
        or view.get("split") not in {"train", "validation"}
        or type(view.get("loss_token_count")) is not int
        or view["loss_token_count"] <= 0
        or type(view.get("full_segment_count")) is not int
        or view["full_segment_count"] < 0
        or type(view.get("final_segment_loss_tokens")) is not int
        or view["final_segment_loss_tokens"] <= 0
        or view.get("finite") is not True
        or view.get("repeat_allowed") is not False
        or view.get("order_policy") != "pack_canonical_order"
        or view.get("raw_text_serialized") is not False
        or view.get("raw_url_serialized") is not False
        or view.get("token_serialized") is not False
    ):
        raise ValueError("finite sequence view boundary is invalid")
    _require_hash(view["pack_manifest_hash"], "finite view pack_manifest_hash")
    validate_public_no_raw_payload(view)
    return view


@dataclass(frozen=True, slots=True)
class C1ScreenAdoptionReceiptV1:
    """The exact closed C1 screen evidence adopted for C2/C3 planning."""

    campaign_hash: str
    screen_decision_hash: str
    campaign_closeout_receipt_hash: str
    billing_finalization_receipt_hash: str
    evaluator_manifest_hash: str
    result_hashes: tuple[str, str, str, str, str, str]
    comparison_hashes: tuple[str, str, str]
    selected_learning_rate: float
    selected_profile_ids: tuple[str, str]
    m_transfer_authorized: bool
    claim_status: str
    production_run_authorized: bool
    f1_authorized: bool

    def __post_init__(self) -> None:
        for field in (
            "campaign_hash",
            "screen_decision_hash",
            "campaign_closeout_receipt_hash",
            "billing_finalization_receipt_hash",
            "evaluator_manifest_hash",
        ):
            _require_hash(getattr(self, field), field)
        if len(self.result_hashes) != 6 or len(set(self.result_hashes)) != 6:
            raise ValueError("C1 adoption requires six unique result hashes")
        if len(self.comparison_hashes) != 3 or len(set(self.comparison_hashes)) != 3:
            raise ValueError("C1 adoption requires three unique comparison hashes")
        for index, value in enumerate(self.result_hashes):
            _require_hash(value, f"result_hashes[{index}]")
        for index, value in enumerate(self.comparison_hashes):
            _require_hash(value, f"comparison_hashes[{index}]")
        if self.selected_learning_rate != 9e-4:
            raise ValueError("C1 adoption requires the selected 9e-4 learning rate")
        if self.selected_profile_ids != _C1_PROFILES:
            raise ValueError("C1 adoption selected profile pair is invalid")
        if self.m_transfer_authorized is not True:
            raise ValueError("C1 adoption requires explicit M-transfer authorization")
        if self.claim_status != _C1_CLAIM_STATUS:
            raise ValueError("C1 adoption claim status is invalid")
        if self.production_run_authorized is not False or self.f1_authorized is not False:
            raise ValueError("C1 adoption cannot unlock production or F1")

    def _payload(self) -> dict[str, Any]:
        return {
            "artifact_kind": "twc_c1_screen_adoption_receipt_v1",
            "schema_version": 1,
            "production_unlock_eligible": False,
            "f1_unlock_eligible": False,
            **asdict(self),
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "secrets_serialized": False,
        }

    @property
    def receipt_hash(self) -> str:
        return stable_json_hash(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_hash"] = stable_json_hash(payload)
        return payload


def load_c1_screen_adoption_receipt(
    payload: Mapping[str, Any],
) -> C1ScreenAdoptionReceiptV1:
    value_fields = set(C1ScreenAdoptionReceiptV1.__dataclass_fields__)
    metadata = {
        "artifact_kind",
        "schema_version",
        "production_unlock_eligible",
        "f1_unlock_eligible",
        "raw_text_serialized",
        "raw_url_serialized",
        "secrets_serialized",
        "receipt_hash",
    }
    if not isinstance(payload, Mapping) or set(payload) != value_fields | metadata:
        raise ValueError("C1 screen adoption receipt fields are invalid")
    if (
        payload.get("artifact_kind") != "twc_c1_screen_adoption_receipt_v1"
        or payload.get("schema_version") != 1
        or payload.get("production_unlock_eligible") is not False
        or payload.get("f1_unlock_eligible") is not False
        or payload.get("raw_text_serialized") is not False
        or payload.get("raw_url_serialized") is not False
        or payload.get("secrets_serialized") is not False
    ):
        raise ValueError("C1 screen adoption receipt boundary is invalid")
    expected = stable_json_hash(
        {key: value for key, value in payload.items() if key != "receipt_hash"}
    )
    if payload.get("receipt_hash") != expected:
        raise ValueError("C1 screen adoption receipt hash mismatch")
    values = {field: payload[field] for field in value_fields}
    values["result_hashes"] = tuple(values["result_hashes"])
    values["comparison_hashes"] = tuple(values["comparison_hashes"])
    values["selected_profile_ids"] = tuple(values["selected_profile_ids"])
    receipt = C1ScreenAdoptionReceiptV1(**values)
    if receipt.receipt_hash != payload["receipt_hash"]:
        raise ValueError("C1 screen adoption receipt semantic hash mismatch")
    validate_public_no_raw_payload(payload)
    return receipt


def _validate_closeout(payload: Mapping[str, Any]) -> dict[str, Any]:
    closeout = _require_hash_bound(
        payload,
        fields=_CLOSEOUT_FIELDS,
        hash_field="receipt_hash",
        label="C1 campaign closeout",
    )
    if (
        closeout.get("artifact_kind")
        != "twc_configuration_campaign_pod_teardown_authorization_v2"
        or closeout.get("schema_version") != 2
        or closeout.get("billing_finalized") is not False
        or closeout.get("pod_teardown_authorized") is not True
        or closeout.get("persistent_disk_teardown_authorized") is not False
        or closeout.get("destructive_action_performed") is not False
        or closeout.get("production_unlock") is not False
        or closeout.get("f1_unlock") is not False
        or closeout.get("raw_text_serialized") is not False
        or closeout.get("secrets_serialized") is not False
    ):
        raise ValueError("C1 campaign closeout boundary is invalid")
    for field in (
        "campaign_hash",
        "promotion_receipt_hash",
        "shared_copyback_manifest_hash",
        "provider_offer_hash",
        "pre_teardown_provider_observation_hash",
    ):
        _require_hash(closeout[field], f"C1 closeout {field}")
    for field in ("attempt_receipt_hashes", "copyback_manifest_hashes", "result_hashes"):
        values = closeout.get(field)
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"C1 closeout {field} must contain unique hashes")
        for index, value in enumerate(values):
            _require_hash(value, f"C1 closeout {field}[{index}]")
    validate_public_no_raw_payload(closeout)
    return closeout


def _validate_billing(payload: Mapping[str, Any]) -> dict[str, Any]:
    billing = _require_hash_bound(
        payload,
        fields=_BILLING_FIELDS,
        hash_field="receipt_hash",
        label="C1 billing finalization",
    )
    if (
        billing.get("artifact_kind")
        != "twc_configuration_provider_billing_finalization_v1"
        or billing.get("schema_version") != 1
        or billing.get("billing_finalized") is not True
        or billing.get("production_unlock") is not False
        or billing.get("f1_unlock") is not False
        or billing.get("raw_text_serialized") is not False
        or billing.get("secrets_serialized") is not False
    ):
        raise ValueError("C1 billing finalization boundary is invalid")
    for field in (
        "campaign_hash",
        "closeout_receipt_hash",
        "final_provider_observation_hash",
    ):
        _require_hash(billing[field], f"C1 billing {field}")
    validate_public_no_raw_payload(billing)
    return billing


def build_c1_screen_adoption_receipt(
    evidence_root: Path,
) -> C1ScreenAdoptionReceiptV1:
    """Reconstruct and adopt the exact closed C1 campaign evidence root."""

    source_root = Path(evidence_root).expanduser()
    if source_root.is_symlink():
        raise ValueError("C1 evidence root must not be a symlink")
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("C1 evidence root must be a real directory")
    campaign_root = root / "campaign"
    if (
        campaign_root.is_symlink()
        or not campaign_root.is_dir()
        or (campaign_root / "campaign-plan.json").is_symlink()
        or not (campaign_root / "campaign-plan.json").is_file()
    ):
        raise ValueError("C1 evidence root is missing its regular campaign ledger")
    ledger = CampaignStore(campaign_root).load()
    if (
        ledger.plan.campaign_id != _C1_CAMPAIGN_ID
        or not ledger.plan.waves
        or any(wave.purpose != "configuration" for wave in ledger.plan.waves)
        or ledger.open_wave_ids
    ):
        raise ValueError("C1 campaign ledger is not the terminal screen campaign")
    promotions = [event for event in ledger.events if isinstance(event, ConfigurationPromotion)]
    stops = [event for event in ledger.events if isinstance(event, AttemptReceipt)]
    if len(promotions) != 1 or len(stops) != 6:
        raise ValueError("C1 campaign requires one promotion and six terminal attempts")

    slots: dict[tuple[str, float], tuple[Any, Any, Any]] = {}
    for profile_id in _C1_PROFILES:
        for suffix, learning_rate in _C1_LEARNING_RATES:
            attempt_id = f"c1-{profile_id.replace('_', '-')}-lr{suffix}-seed17"
            bundle_root = root / "bundles" / attempt_id
            training = load_experiment_training_contract(
                _read_object(bundle_root / "run-contract.json", f"{attempt_id} run contract")
            )
            result = load_experiment_result_receipt(
                _read_object(root / "results" / f"{attempt_id}.json", f"{attempt_id} result")
            )
            evaluator = load_evaluation_manifest(
                _read_object(
                    bundle_root / "evaluation-manifest.json",
                    f"{attempt_id} evaluation manifest",
                )
            )
            contract = training.experiment
            if (
                contract.profile_id != profile_id
                or contract.learning_rate != learning_rate
                or contract.campaign_hash != ledger.plan.identity
                or result.contract_hash != contract.identity
                or result.evaluator_manifest_hash != evaluator.identity
            ):
                raise ValueError("C1 result slot is cross-run or scientifically mismatched")
            slots[(profile_id, learning_rate)] = (result, contract, evaluator)

    comparisons: list[PairedComparisonReceipt] = []
    for _, learning_rate in _C1_LEARNING_RATES:
        candidate_result, candidate_contract, candidate_evaluator = slots[
            ("s_wide", learning_rate)
        ]
        control_result, control_contract, control_evaluator = slots[
            ("s_deep", learning_rate)
        ]
        if candidate_evaluator != control_evaluator:
            raise ValueError("C1 paired evaluators differ")
        comparisons.append(
            PairedComparisonReceipt(
                candidate_result=candidate_result,
                control_result=control_result,
                candidate_contract=candidate_contract,
                control_contract=control_contract,
                evaluation_manifest=candidate_evaluator,
                ranking_task_ids=("validation_nll",),
            )
        )
    decision = ConfigurationScreenDecisionReceipt(
        screen_id="c1-screen",
        campaign_hash=ledger.plan.identity,
        comparisons=tuple(comparisons),
    )
    screen = _read_object(root / "results" / "c1-screen.json", "C1 screen decision")
    if set(screen) != set(decision.to_dict()) or stable_json_hash(screen) != decision.identity:
        raise ValueError("C1 screen cannot be reconstructed from the six exact results")
    screen_hash = decision.identity

    result_hashes = tuple(
        result.identity
        for comparison in comparisons
        for result in (comparison.candidate_result, comparison.control_result)
    )
    comparison_hashes = tuple(comparison.identity for comparison in comparisons)
    evaluator_hashes = {
        comparison.evaluation_manifest.identity for comparison in comparisons
    }
    if len(evaluator_hashes) != 1:
        raise ValueError("C1 screen does not use one exact evaluator manifest")

    promotion = promotions[0]
    if promotion.decision_evidence_hashes != (screen_hash, *comparison_hashes):
        raise ValueError("C1 campaign promotion is not bound to the exact screen")
    closeout = _validate_closeout(
        _read_object(root / "configuration-closeout.json", "C1 campaign closeout")
    )
    if (
        closeout["campaign_hash"] != ledger.plan.identity
        or closeout["promotion_receipt_hash"] != promotion.receipt_hash
        or set(closeout["attempt_receipt_hashes"])
        != {stop.receipt_hash for stop in stops}
        or set(closeout["result_hashes"]) != set(result_hashes)
    ):
        raise ValueError("C1 closeout lineage differs from the reconstructed campaign")
    billing = _validate_billing(
        _read_object(
            root / "configuration-billing-final.json", "C1 billing finalization"
        )
    )
    rebuilt_billing = finalize_configuration_provider_billing(
        closeout=closeout,
        final_observation=_read_object(
            root / "shared" / "provider-billing-final.json",
            "C1 final provider observation",
        ),
        campaign_budget_cap_usd=format(ledger.plan.budget_cap_usd, "f"),
        campaign_reserve_usd=format(ledger.plan.reserve_usd, "f"),
        expected_campaign_hash=ledger.plan.identity,
    )
    if (
        rebuilt_billing != billing
        or billing["campaign_hash"] != ledger.plan.identity
        or billing["closeout_receipt_hash"] != closeout["receipt_hash"]
    ):
        raise ValueError("C1 billing finalization differs from the closed campaign")

    receipt = C1ScreenAdoptionReceiptV1(
        campaign_hash=ledger.plan.identity,
        screen_decision_hash=screen_hash,
        campaign_closeout_receipt_hash=closeout["receipt_hash"],
        billing_finalization_receipt_hash=billing["receipt_hash"],
        evaluator_manifest_hash=next(iter(evaluator_hashes)),
        result_hashes=result_hashes,
        comparison_hashes=comparison_hashes,
        selected_learning_rate=decision.selected_learning_rate,
        selected_profile_ids=decision.selected_profile_ids,
        m_transfer_authorized=decision.m_transfer_authorized,
        claim_status=decision.claim_status,
        production_run_authorized=False,
        f1_authorized=False,
    )
    validate_public_no_raw_payload(receipt.to_dict())
    return receipt
