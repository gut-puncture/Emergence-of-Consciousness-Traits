"""Scientific source identity for the C2/C3 configuration study.

The physical release hash owns packaging and operator provenance.  This module
owns the narrower identity that must remain equal across C2 and C3: model and
training mathematics, optimizer/schedule/global-step behavior, finite-data
interpretation, and neutral evaluation semantics.  Mixed runtime modules are
projected to an explicit, fail-closed set of Python AST units so checkpoint,
prefetch, device-slicing, retry, and provider fixes do not rewrite science.
"""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping

from twc.repro.hashing import stable_json_hash


_WHOLE_FILE = "<file>"

# Exact scientific owners.  A full-file unit is used only when the module is a
# pure scientific contract or numerical implementation.  Mixed modules name
# the smallest coherent top-level symbols/methods.  Dependency validation below
# rejects a newly referenced local helper or constant until it is deliberately
# added here.
SEMANTIC_CODE_UNITS_V2: tuple[tuple[str, str], ...] = (
    ("configs/eval.yaml", _WHOLE_FILE),
    ("training/twc/pretraining/configuration_assessment.py", _WHOLE_FILE),
    ("training/twc/pretraining/configuration_composite_data.py", _WHOLE_FILE),
    ("training/twc/pretraining/configuration_study.py", _WHOLE_FILE),
    (
        "training/twc/pretraining/configuration_study_compiler.py",
        "ConfigurationCompilerInputReceiptV1",
    ),
    ("training/twc/pretraining/configuration_study_compiler.py", "VALIDATION_LOSS_TOKENS"),
    (
        "training/twc/pretraining/configuration_study_compiler.py",
        "CONFIGURATION_GLOBAL_STEP_CAPS_V2",
    ),
    ("training/twc/pretraining/configuration_study_compiler.py", "DATA_ORDER_POLICY_V2"),
    ("training/twc/pretraining/configuration_study_compiler.py", "SCHEDULE_POLICY_V2"),
    (
        "training/twc/pretraining/configuration_study_compiler.py",
        "VALIDATION_NORMALIZATION_V1",
    ),
    ("training/twc/pretraining/configuration_study_compiler.py", "_require_hash"),
    ("training/twc/pretraining/configuration_study_compiler.py", "_positive_int"),
    (
        "training/twc/pretraining/configuration_study_compiler.py",
        "bind_configuration_validation_nll",
    ),
    (
        "training/twc/pretraining/configuration_study_compiler.py",
        "_compile_configuration_profile",
    ),
    (
        "training/twc/pretraining/configuration_study_compiler.py",
        "compile_configuration_model_artifact",
    ),
    (
        "training/twc/pretraining/configuration_study_compiler.py",
        "compile_configuration_endpoint_model_artifact",
    ),
    (
        "training/twc/pretraining/configuration_study_compiler.py",
        "_validate_study_inputs",
    ),
    ("training/twc/pretraining/configuration_study_compiler.py", "_data_order_hash"),
    ("training/twc/pretraining/configuration_study_compiler.py", "_schedule"),
    ("training/twc/pretraining/configuration_study_compiler.py", "_bound_evaluation"),
    ("training/twc/pretraining/configuration_result_builder.py", "_VALIDATION_LOSS_TOKENS"),
    ("training/twc/pretraining/configuration_result_builder.py", "_HASH_CHARACTERS"),
    ("training/twc/pretraining/configuration_result_builder.py", "_hash_without"),
    ("training/twc/pretraining/configuration_result_builder.py", "_is_hash"),
    (
        "training/twc/pretraining/configuration_result_builder.py",
        "_validate_bound_global_step_plan",
    ),
    (
        "training/twc/pretraining/configuration_result_builder.py",
        "_expected_evaluation_positions",
    ),
    ("training/twc/pretraining/data_pipeline/sequence_pack.py", "INDEX_RECORD"),
    ("training/twc/pretraining/data_pipeline/sequence_pack.py", "_SPLIT_CODE"),
    ("training/twc/pretraining/data_pipeline/sequence_pack.py", "_CODE_SPLIT"),
    ("training/twc/pretraining/data_pipeline/sequence_pack.py", "IsolatedSequenceYield"),
    ("training/twc/pretraining/data_pipeline/sequence_pack.py", "PackedSegment"),
    ("training/twc/pretraining/data_pipeline/sequence_pack.py", "isolated_sequence_yield"),
    ("training/twc/pretraining/data_pipeline/sequence_pack.py", "_sha256_file"),
    ("training/twc/pretraining/data_pipeline/sequence_pack.py", "_relative_file"),
    (
        "training/twc/pretraining/data_pipeline/sequence_pack.py",
        "validate_isolated_sequence_pack",
    ),
    ("training/twc/pretraining/data_pipeline/sequence_pack.py", "iter_view_segments"),
    (
        "training/twc/pretraining/data_pipeline/sequence_pack.py",
        "validate_global_step_plan",
    ),
    ("training/twc/pretraining/training/data.py", "_SEQUENCE_INDEX"),
    ("training/twc/pretraining/training/data.py", "SequenceRecord"),
    ("training/twc/pretraining/training/data.py", "BatchSource._read"),
    ("training/twc/pretraining/training/data.py", "BatchSource.read_batch"),
    ("training/twc/pretraining/training/data.py", "SequenceTrainingArtifacts@schema"),
    ("training/twc/pretraining/training/data.py", "SequenceEvaluationArtifacts@schema"),
    ("training/twc/pretraining/training/data.py", "_hash_without"),
    ("training/twc/pretraining/training/data.py", "_json_object"),
    ("training/twc/pretraining/training/data.py", "_load_finite_view_records"),
    ("training/twc/pretraining/training/data.py", "load_sequence_evaluation_artifacts"),
    ("training/twc/pretraining/training/data.py", "load_sequence_training_artifacts"),
    ("training/twc/pretraining/training/engine.py", "EngineConfig@schema"),
    ("training/twc/pretraining/training/engine.py", "EngineConfig.from_contract"),
    ("training/twc/pretraining/training/engine.py", "StepResult@schema"),
    ("training/twc/pretraining/training/engine.py", "EvaluationResult@schema"),
    ("training/twc/pretraining/training/engine.py", "Microbatch"),
    ("training/twc/pretraining/training/engine.py", "wsd_token_multiplier"),
    ("training/twc/pretraining/training/engine.py", "TrainingEngine.__init__"),
    ("training/twc/pretraining/training/engine.py", "TrainingEngine.train_step"),
    ("training/twc/pretraining/training/engine.py", "TrainingEngine.evaluate"),
    (
        "training/twc/pretraining/training/engine.py",
        "TrainingEngine._schedule_multiplier",
    ),
    ("training/twc/pretraining/training/engine.py", "TrainingEngine._forward"),
    ("training/twc/pretraining/training/engine.py", "TrainingEngine._forward_context"),
    ("training/twc/pretraining/training/engine.py", "_microbatch_loss_tokens"),
    ("training/twc/pretraining/training/engine.py", "_optimizer_parameters"),
    (
        "training/twc/pretraining/training/worker.py",
        "_validate_plan_contract",
    ),
    (
        "training/twc/pretraining/training/worker.py",
        "_evaluation_groups",
    ),
    (
        "training/twc/pretraining/training/worker.py",
        "_evaluation_component_group_hashes",
    ),
    (
        "training/twc/pretraining/training/worker.py",
        "_validate_v2_attempt_cadence",
    ),
    (
        "training/twc/pretraining/training/worker.py",
        "_initialize_training_engine",
    ),
    ("training/twc/pretraining/training/worker.py", "_plan_groups"),
    (
        "training/twc/pretraining/training/worker.py",
        "_planned_microbatch_ordinals",
    ),
    ("training/twc/pretraining/training/experiments.py", _WHOLE_FILE),
    ("training/twc/pretraining/training/model.py", _WHOLE_FILE),
    ("training/twc/pretraining/training/planning.py", "GLOBAL_STEP_PLAN_KIND"),
    ("training/twc/pretraining/training/planning.py", "PlannedWork"),
    ("training/twc/pretraining/training/planning.py", "GlobalStepPlan"),
    ("training/twc/pretraining/training/planning.py", "build_global_step_plan"),
    (
        "training/twc/pretraining/training/planning.py",
        "global_plan_for_sequence_artifacts",
    ),
    ("training/twc/pretraining/training/planning.py", "_validate_lengths"),
    ("training/twc/pretraining/training/planning.py", "_validate_caps"),
    ("training/twc/pretraining/training/planning.py", "_next_fit"),
    ("training/twc/pretraining/training/planning.py", "_work_dict"),
    ("training/twc/pretraining/training/planning.py", "_validate_global_plan"),
)

SEMANTIC_CODE_PATHS_V2 = tuple(
    dict.fromkeys(path for path, _selector in SEMANTIC_CODE_UNITS_V2)
)


def _require_hash(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _top_level_owner(node: ast.stmt) -> tuple[str, ...]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return (node.name,)
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        return tuple(target.id for target in targets if isinstance(target, ast.Name))
    return ()


def _module_owners(tree: ast.Module) -> dict[str, ast.stmt]:
    owners: dict[str, ast.stmt] = {}
    for node in tree.body:
        for name in _top_level_owner(node):
            if name in owners:
                raise ValueError(f"semantic Python module defines {name} more than once")
            owners[name] = node
    return owners


def _schema_node(node: ast.ClassDef) -> ast.ClassDef:
    body = [
        copy.deepcopy(item)
        for item in node.body
        if isinstance(item, (ast.Assign, ast.AnnAssign))
    ]
    if not body:
        body = [ast.Pass()]
    return ast.ClassDef(
        name=node.name,
        bases=copy.deepcopy(node.bases),
        keywords=copy.deepcopy(node.keywords),
        body=body,
        decorator_list=copy.deepcopy(node.decorator_list),
        type_params=copy.deepcopy(getattr(node, "type_params", [])),
    )


def _selected_node(
    *, owners: Mapping[str, ast.stmt], selector: str, path: str
) -> ast.AST:
    if selector.endswith("@schema"):
        class_name = selector.removesuffix("@schema")
        owner = owners.get(class_name)
        if not isinstance(owner, ast.ClassDef):
            raise ValueError(f"semantic selector is not a class schema: {path}:{selector}")
        return _schema_node(owner)
    if "." in selector:
        class_name, member_name = selector.split(".", 1)
        owner = owners.get(class_name)
        if not isinstance(owner, ast.ClassDef):
            raise ValueError(f"semantic selector class is missing: {path}:{selector}")
        members = [
            item
            for item in owner.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == member_name
        ]
        if len(members) != 1:
            raise ValueError(f"semantic selector member is missing: {path}:{selector}")
        return members[0]
    owner = owners.get(selector)
    if owner is None:
        raise ValueError(f"semantic selector is missing: {path}:{selector}")
    return owner


def _canonical_ast_bytes(node: ast.AST) -> bytes:
    return (ast.dump(node, annotate_fields=True, include_attributes=False) + "\n").encode()


def _covered_owner_names(selectors: tuple[str, ...]) -> set[str]:
    covered: set[str] = set()
    for selector in selectors:
        if selector == _WHOLE_FILE or "." in selector:
            continue
        covered.add(selector.removesuffix("@schema"))
    return covered


def _python_units(path: Path, relative: str, selectors: tuple[str, ...]) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, SyntaxError, UnicodeError) as error:
        raise ValueError(f"semantic Python source is unreadable: {relative}") from error
    owners = _module_owners(tree)
    covered = _covered_owner_names(selectors)
    selected = {
        selector: _selected_node(owners=owners, selector=selector, path=relative)
        for selector in selectors
    }
    for selector, node in selected.items():
        dependencies = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        omitted = sorted(name for name in dependencies if name in owners and name not in covered)
        if omitted:
            raise ValueError(
                f"semantic selector dependency omitted for {relative}:{selector}: "
                + ", ".join(omitted)
            )
    rows: list[dict[str, Any]] = []
    for selector in selectors:
        payload = _canonical_ast_bytes(selected[selector])
        rows.append(
            {
                "path": relative,
                "selector": selector,
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    return rows


def _build_units(root: Path) -> list[dict[str, Any]]:
    by_path: dict[str, list[str]] = {}
    for relative, selector in SEMANTIC_CODE_UNITS_V2:
        by_path.setdefault(relative, []).append(selector)
    rows: list[dict[str, Any]] = []
    for relative in SEMANTIC_CODE_PATHS_V2:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing semantic source: {relative}")
        selectors = tuple(by_path[relative])
        if selectors == (_WHOLE_FILE,):
            payload = path.read_bytes()
            rows.append(
                {
                    "path": relative,
                    "selector": _WHOLE_FILE,
                    "bytes": len(payload),
                    "sha256": _sha256(payload),
                }
            )
        elif _WHOLE_FILE in selectors:
            raise ValueError(f"whole-file semantic selector cannot be combined: {relative}")
        else:
            rows.extend(_python_units(path, relative, selectors))
    return rows


def semantic_code_hash_from_inventory(units: object) -> str:
    """Validate and reduce the exact canonical semantic unit inventory."""

    expected_units = SEMANTIC_CODE_UNITS_V2
    if not isinstance(units, (list, tuple)) or tuple(
        (row.get("path"), row.get("selector"))
        if isinstance(row, Mapping)
        else (None, None)
        for row in units
    ) != expected_units:
        raise ValueError("configuration receipt requires the exact semantic unit set")
    canonical: list[dict[str, Any]] = []
    for row in units:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "selector", "bytes", "sha256"}
            or type(row.get("bytes")) is not int
            or int(row["bytes"]) < 0
        ):
            raise ValueError("configuration semantic code unit row is invalid")
        canonical.append(
            {
                "path": str(row["path"]),
                "selector": str(row["selector"]),
                "bytes": int(row["bytes"]),
                "sha256": _require_hash(row.get("sha256"), "semantic unit sha256"),
            }
        )
    if canonical != list(units):
        raise ValueError("configuration semantic code unit inventory is noncanonical")
    return stable_json_hash(
        {
            "artifact_kind": "twc_configuration_semantic_code_identity_v2",
            "schema_version": 2,
            "units": canonical,
        }
    )


def validate_configuration_semantic_code_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the projected source inventory that owns C2/C3 science."""

    expected = {
        "artifact_kind",
        "schema_version",
        "units",
        "semantic_code_hash",
        "raw_text_serialized",
        "raw_url_serialized",
        "secrets_serialized",
        "receipt_hash",
    }
    if (
        set(receipt) != expected
        or receipt.get("artifact_kind")
        != "twc_configuration_semantic_code_inventory_v2"
        or receipt.get("schema_version") != 2
        or receipt.get("raw_text_serialized") is not False
        or receipt.get("raw_url_serialized") is not False
        or receipt.get("secrets_serialized") is not False
    ):
        raise ValueError("configuration semantic code receipt boundary is invalid")
    if receipt.get("semantic_code_hash") != semantic_code_hash_from_inventory(
        receipt.get("units")
    ):
        raise ValueError("configuration semantic code hash mismatch")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if receipt.get("receipt_hash") != stable_json_hash(unsigned):
        raise ValueError("configuration semantic code receipt hash mismatch")
    return dict(receipt)


def build_configuration_semantic_code_receipt(code_root: Path) -> dict[str, Any]:
    """Project and hash only predeclared C2/C3 scientific source units."""

    root = Path(code_root).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("configuration code root must be a real directory")
    units = _build_units(root)
    receipt: dict[str, Any] = {
        "artifact_kind": "twc_configuration_semantic_code_inventory_v2",
        "schema_version": 2,
        "units": units,
        "semantic_code_hash": semantic_code_hash_from_inventory(units),
        "raw_text_serialized": False,
        "raw_url_serialized": False,
        "secrets_serialized": False,
    }
    receipt["receipt_hash"] = stable_json_hash(receipt)
    return validate_configuration_semantic_code_receipt(receipt)
