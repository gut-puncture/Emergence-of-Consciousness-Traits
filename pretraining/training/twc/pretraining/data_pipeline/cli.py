"""Provider-neutral ``twc-data`` CPU data operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from twc.repro.hashing import stable_json_hash
from twc.repro.public_artifacts import validate_public_no_raw_payload

from .accepted_data import validate_accepted_clean_run_manifest
from twc.pretraining.materialization_handoff import (
    MaterializationInput,
    SelectedMaterializationFact,
    validate_selected_materialization_handoff,
)
from .accepted_data import SourceDocument, build_accepted_clean_runs
from .accepted_data import SourceDocumentUnit
from .sequence_pack import (
    build_finite_sequence_view,
    build_fused_isolated_sequence_pack,
    build_global_step_plan,
    iter_view_segments,
    relocate_and_verify_pack,
    validate_global_step_plan,
    validate_isolated_sequence_pack,
)
from .tokenizer import (
    CleanRunText,
    build_tokenizer,
    huggingface_tokenizers_builder,
    load_tokenizer_policy,
)


def _json(payload: dict[str, Any]) -> None:
    validate_public_no_raw_payload(payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact must be a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class _AcceptedRecordSchedule:
    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.manifest = manifest
        self.identity_hash = manifest["manifest_hash"]

    def __iter__(self) -> Iterator[CleanRunText]:
        return self.iter_from(0)

    def validate_committed_prefix(self, cursor: int) -> None:
        covered = 0
        for receipt in self.manifest["part_receipts"]:
            if covered >= cursor:
                break
            path = self.root / receipt["part_name"]
            metadata = path.stat()
            if receipt.get("file_identity") != {
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "byte_count": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
            }:
                raise ValueError("committed accepted input file identity changed")
            covered += int(receipt["record_count"])

    def iter_from(self, cursor: int) -> Iterator[CleanRunText]:
        if cursor < 0 or cursor > self.manifest["accepted_clean_run_count"]:
            raise ValueError("accepted-record cursor is out of range")
        skipped = 0
        for receipt in self.manifest["part_receipts"]:
            part_count = int(receipt["record_count"])
            if skipped + part_count <= cursor:
                skipped += part_count
                continue
            path = self.root / receipt["part_name"]
            with path.open(encoding="utf-8") as handle:
                for local_index, line in enumerate(handle):
                    if skipped + local_index < cursor:
                        continue
                    row = json.loads(line)
                    yield CleanRunText(
                        split=row["split"],
                        clean_run_id_hash=row["clean_run_id_hash"],
                        source_document_id_hash=row["source_document_id_hash"],
                        text=row["text"],
                    )
            skipped += part_count


def _plan(args: argparse.Namespace) -> None:
    policy = load_tokenizer_policy(args.tokenizer_config)
    payload = {
        "artifact_kind": "twc_data_plan_v1",
        "schema_version": 1,
        "status": "local_plan_only",
        "phases": [
            "accepted_clean_runs",
            "tokenizer",
            "fused_tokenize_and_isolated_pack",
            "finite_sequence_views",
            "semantic_global_step_plan",
        ],
        "tokenizer_config_hash": stable_json_hash(
            {
                "version": policy.version,
                "vocab_size": policy.config.vocab_size,
                "model_type": policy.config.model_type,
                "special_tokens": list(policy.config.special_tokens),
                "diagnostics_required": list(policy.diagnostics_required),
            }
        ),
        "provider_calls_authorized": False,
        "private_read_authorized": False,
        "isolated_sequence_pack_implemented": True,
        "tokenized_json_intermediate_required": False,
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "token_serialized": False,
    }
    payload["plan_hash"] = stable_json_hash(payload)
    _json(payload)


def _status(args: argparse.Namespace) -> None:
    root = args.root
    kinds = {
        "accepted": root / "accepted" / "accepted_clean_run_manifest.json",
        "tokenizer": root / "tokenizer" / "tokenizer_manifest.json",
        "isolated-pack": root / "isolated-pack" / "isolated_sequence_pack_manifest.json",
        "view": root / "views" / "train.view.json",
        "plan": root / "plans" / "train.global-plan.json",
    }
    present = [name for name, path in kinds.items() if path.is_file()]
    prerequisites = {
        "accepted": [],
        "tokenizer": ["accepted"],
        "isolated-pack": ["accepted", "tokenizer"],
        "view": ["isolated-pack"],
        "plan": ["view"],
    }
    blockers = {
        phase: [required for required in required_phases if required not in present]
        for phase, required_phases in prerequisites.items()
        if phase not in present
    }
    _json(
        {
            "artifact_kind": "twc_data_status_v1",
            "schema_version": 1,
            "status": (
                "all_artifacts_present_unvalidated"
                if len(present) == len(kinds)
                else ("partially_built" if present else "not_started")
            ),
            "present_artifact_kinds": present,
            "blocking_prerequisites": blockers,
            "provider_calls_performed": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
    )


def _validate(args: argparse.Namespace) -> None:
    phase_root = args.root / args.kind
    if args.kind == "accepted":
        manifest = _load(phase_root / "accepted_clean_run_manifest.json")
        validate_accepted_clean_run_manifest(manifest, root=phase_root)
    elif args.kind == "isolated-pack":
        manifest = _load(phase_root / "isolated_sequence_pack_manifest.json")
        validate_isolated_sequence_pack(manifest, phase_root)
    elif args.kind == "view":
        pack_root = args.root / "isolated-pack"
        pack = _load(pack_root / "isolated_sequence_pack_manifest.json")
        view = _load(args.root / "views" / f"{args.split}.view.json")
        observed = sum(
            segment.loss_token_count
            for segment in iter_view_segments(pack, pack_root, view=view)
        )
        if observed != view.get("loss_token_count"):
            raise ValueError("finite sequence view token coverage mismatch")
    elif args.kind == "plan":
        pack_root = args.root / "isolated-pack"
        pack = _load(pack_root / "isolated_sequence_pack_manifest.json")
        view = _load(args.root / "views" / f"{args.split}.view.json")
        plan = _load(args.root / "plans" / f"{args.split}.global-plan.json")
        validate_global_step_plan(plan, view=view, pack_manifest=pack, pack_root=pack_root)
    else:  # tokenizer
        manifest = _load(phase_root / "tokenizer_manifest.json")
        expected = stable_json_hash({k: v for k, v in manifest.items() if k != "manifest_hash"})
        if manifest.get("manifest_hash") != expected:
            raise ValueError("tokenizer manifest hash mismatch")
        artifact = phase_root / "tokenizer.bin"
        if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest.get(
            "artifact_sha256"
        ):
            raise ValueError("tokenizer artifact hash mismatch")
    _json(
        {
            "artifact_kind": "twc_data_validation_v1",
            "schema_version": 1,
            "validated_kind": args.kind,
            "valid": True,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
    )


class _JsonlSourceSchedule:
    def __init__(self, path: Path, receipt_path: Path) -> None:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("source JSONL must be a single-link regular file")
        receipt = _load(receipt_path)
        if receipt.get("receipt_hash") != stable_json_hash(
            {key: value for key, value in receipt.items() if key != "receipt_hash"}
        ):
            raise ValueError("source JSONL receipt hash mismatch")
        if receipt.get("byte_count") != metadata.st_size:
            raise ValueError("source JSONL receipt byte count mismatch")
        expected_identity = receipt.get("file_identity")
        observed_identity = {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "byte_count": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
        }
        if expected_identity != observed_identity:
            raise ValueError("source JSONL immutable file identity mismatch")
        self.path = path
        self.receipt = receipt
        self.identity_hash = receipt["receipt_hash"]

    def verify_content(self) -> None:
        if _sha256_file(self.path) != self.receipt.get("sha256"):
            raise ValueError("source JSONL content hash mismatch")

    def iter_from(self, cursor: int) -> Iterator[SourceDocumentUnit]:
        with self.path.open("rb") as handle:
            handle.seek(cursor)
            while line := handle.readline():
                next_cursor = handle.tell()
                row = json.loads(line)
                yield SourceDocumentUnit(
                    next_cursor,
                    SourceDocument(
                        row["source_document_id_hash"],
                        row["split"],
                        row["raw_text"],
                        row.get("source_lineage_hash"),
                        row.get("source_token_count_hint"),
                    ),
                )


def _materialization_input(path: Path, handoff: dict[str, Any]) -> MaterializationInput:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("selected index must be a single-link regular file")
    if metadata.st_mode & 0o077:
        raise ValueError("selected index permissions must be owner-only")
    uri = f"file:{path.resolve(strict=True)}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    if connection.execute("pragma integrity_check").fetchone() != ("ok",):
        connection.close()
        raise ValueError("selected index failed SQLite integrity check")
    identity_row = connection.execute(
        "select identity_json,identity_hash from selected_lookup_identity where id=1"
    ).fetchone()
    if identity_row is None or connection.execute(
        "select count(*) from selected_lookup_identity"
    ).fetchone() != (1,):
        raise ValueError("selected index identity is missing")
    identity = json.loads(identity_row[0])
    if stable_json_hash(identity) != identity_row[1]:
        raise ValueError("selected index identity hash mismatch")
    if identity.get("source_content_root") != handoff["source_content_root"]:
        raise ValueError("selected index source-content binding mismatch")
    shards = []
    for shard_number, receipt_json, receipt_hash in connection.execute(
        "select shard_number,receipt_json,receipt_hash from selected_lookup_shard_commits "
        "order by shard_number"
    ):
        receipt = json.loads(receipt_json)
        if (
            receipt.get("receipt_hash") != receipt_hash
            or stable_json_hash(
                {key: value for key, value in receipt.items() if key != "receipt_hash"}
            )
            != receipt_hash
            or receipt.get("shard_number") != int(shard_number)
            or receipt.get("source_content_root") != handoff["source_content_root"]
            or receipt.get("case_id") != identity.get("case_id")
        ):
            raise ValueError("selected index shard receipt binding mismatch")
        shards.append(
            {
                "shard_number": int(shard_number),
                "selected_count": int(receipt["selected_count"]),
                "selected_fact_root": receipt["selected_fact_root"],
                "receipt_hash": receipt_hash,
            }
        )
    selected_count = int(connection.execute("select count(*) from selected_lookup").fetchone()[0])
    descriptor = {
        "artifact_kind": "twc_filtering_closeout_selected_fact_index_v1",
        "identity_hash": identity_row[1],
        "selected_count": selected_count,
        "shards": shards,
    }

    class FactSchedule:
        def __iter__(self) -> Iterator[SelectedMaterializationFact]:
            return self.iter_from_shard(0)

        def iter_from_shard(self, start_shard: int) -> Iterator[SelectedMaterializationFact]:
            query = (
                "select shard_number,candidate_id_hash,global_ordinal,evidence_text_sha256,"
                "core_text_sha256,core_char_count,core_byte_count,core_proxy_token_count,"
                "first_block_index,last_block_index,core_char_start,core_char_end,"
                "source_document_id_hash,segmentation_receipt_hash from selected_lookup "
                "where shard_number>=? order by shard_number,candidate_id_hash"
            )
            for row in connection.execute(query, (start_shard,)):
                yield SelectedMaterializationFact(*row)

    return MaterializationInput(
        descriptor,
        handoff["topology_fact_root"],
        handoff["export_fact_root"],
        handoff["source_content_root"],
        FactSchedule(),
    )


def _build(args: argparse.Namespace) -> None:
    if not args.private_read_approval_id:
        raise ValueError("data build requires an explicit private-read approval id")
    run_root = args.root
    if args.kind == "accepted":
        legacy = (args.handoff, args.selected_index_sqlite, args.source_jsonl, args.source_jsonl_receipt)
        if args.input_mode == "verified-selected-core":
            if args.preserved_core_export_manifest is None or any(value is not None for value in legacy):
                raise ValueError("verified-selected-core mode requires only --preserved-core-export-manifest")
            export = _load(args.preserved_core_export_manifest)
            result = build_accepted_clean_runs(
                core_export_manifest=export, core_export_root=args.preserved_core_export_manifest.parent,
                output_root=run_root / "accepted", validation_basis_points=args.validation_basis_points,
            )
        else:
            if any(value is None for value in legacy) or args.preserved_core_export_manifest is not None:
                raise ValueError("source-rehydration mode requires handoff, selected index, source JSONL, and receipt")
            handoff = _load(args.handoff)
            validate_selected_materialization_handoff(handoff)
            result = build_accepted_clean_runs(
                handoff=handoff, materialization_input=_materialization_input(args.selected_index_sqlite, handoff),
                source_documents=_JsonlSourceSchedule(args.source_jsonl, args.source_jsonl_receipt),
                output_root=run_root / "accepted", validation_basis_points=args.validation_basis_points,
            )
    elif args.kind == "tokenizer":
        if args.tokenizer_config is None:
            raise ValueError("tokenizer build requires --tokenizer-config")
        accepted_root = run_root / "accepted"
        accepted = _load(accepted_root / "accepted_clean_run_manifest.json")
        validate_accepted_clean_run_manifest(accepted, root=accepted_root)
        policy = load_tokenizer_policy(args.tokenizer_config)
        result = build_tokenizer(
            records=_AcceptedRecordSchedule(accepted_root, accepted),
            config=policy.config,
            corpus_manifest_hash=accepted["manifest_hash"],
            output_root=run_root / "tokenizer",
            builder=huggingface_tokenizers_builder,
        )
    elif args.kind == "isolated-pack":
        accepted_root, tokenizer_root = run_root / "accepted", run_root / "tokenizer"
        accepted = _load(accepted_root / "accepted_clean_run_manifest.json")
        tokenizer_manifest = _load(tokenizer_root / "tokenizer_manifest.json")
        result = build_fused_isolated_sequence_pack(
            output_root=run_root / "isolated-pack",
            accepted_root=accepted_root,
            accepted_manifest=accepted,
            tokenizer_root=tokenizer_root,
            tokenizer_manifest=tokenizer_manifest,
            context_length=args.context_length,
            target_shard_bytes=args.target_shard_bytes,
        )
    elif args.kind == "view":
        if args.requested_loss_tokens is None:
            raise ValueError("view build requires --requested-loss-tokens")
        pack_root = run_root / "isolated-pack"
        pack = _load(pack_root / "isolated_sequence_pack_manifest.json")
        view = build_finite_sequence_view(
            pack_manifest=pack,
            pack_root=pack_root,
            split=args.split,
            requested_loss_tokens=args.requested_loss_tokens,
            output_path=run_root / "views" / f"{args.split}.view.json",
        )
        result = type("ViewResult", (), {"manifest": view})()
    else:  # plan
        pack_root = run_root / "isolated-pack"
        pack = _load(pack_root / "isolated_sequence_pack_manifest.json")
        view = _load(run_root / "views" / f"{args.split}.view.json")
        plan = build_global_step_plan(
            view=view,
            pack_manifest=pack,
            pack_root=pack_root,
            output_path=run_root / "plans" / f"{args.split}.global-plan.json",
            max_loss_tokens=args.max_loss_tokens,
            max_attention_cells=args.max_attention_cells,
            max_sequences=args.max_sequences,
        )
        result = type("PlanResult", (), {"manifest": plan})()
    _json(
        {
            "artifact_kind": "twc_data_build_receipt_v1",
            "schema_version": 1,
            "built_kind": args.kind,
            "manifest_hash": result.manifest.get(
                "manifest_hash",
                result.manifest.get("view_hash", result.manifest.get("plan_hash")),
            ),
            "approval_id_hash": stable_json_hash(args.private_read_approval_id),
            "provider_calls_performed": False,
            "raw_text_serialized": False,
            "raw_url_serialized": False,
            "token_serialized": False,
        }
    )


def _place(args: argparse.Namespace) -> None:
    if not args.approval_id:
        raise ValueError("data placement requires an explicit approval id")
    source = args.source.resolve(strict=True)
    manifest = _load(source / "isolated_sequence_pack_manifest.json")
    receipt = relocate_and_verify_pack(
        pack_manifest=manifest,
        source_root=source,
        destination_root=args.destination,
    )
    _json(receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="twc-data")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--tokenizer-config", type=Path, required=True)
    plan.set_defaults(func=_plan)
    status = sub.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    status.set_defaults(func=_status)
    place = sub.add_parser("place")
    place.add_argument("--source", type=Path, required=True)
    place.add_argument("--destination", type=Path, required=True)
    place.add_argument("--approval-id")
    place.set_defaults(func=_place)
    validate = sub.add_parser("validate")
    validate.add_argument(
        "kind",
        choices=("accepted", "tokenizer", "isolated-pack", "view", "plan"),
    )
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--split", choices=("train", "validation"), default="train")
    validate.set_defaults(func=_validate)
    build = sub.add_parser("build")
    build.add_argument(
        "kind", choices=("accepted", "tokenizer", "isolated-pack", "view", "plan")
    )
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--tokenizer-config", type=Path)
    build.add_argument("--handoff", type=Path)
    build.add_argument("--selected-index-sqlite", type=Path)
    build.add_argument("--source-jsonl", type=Path)
    build.add_argument("--source-jsonl-receipt", type=Path)
    build.add_argument("--input-mode", choices=("source-rehydration", "verified-selected-core"), default="source-rehydration")
    build.add_argument("--preserved-core-export-manifest", type=Path)
    build.add_argument("--validation-basis-points", type=int, default=500)
    build.add_argument("--context-length", type=int, default=4096)
    build.add_argument("--target-shard-bytes", type=int, default=256 * 1024 * 1024)
    build.add_argument("--split", choices=("train", "validation"), default="train")
    build.add_argument("--requested-loss-tokens", type=int)
    build.add_argument("--max-loss-tokens", type=int, default=262_144)
    build.add_argument("--max-attention-cells", type=int, default=1_073_741_824)
    build.add_argument("--max-sequences", type=int, default=4_096)
    build.add_argument("--private-read-approval-id")
    build.set_defaults(func=_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
