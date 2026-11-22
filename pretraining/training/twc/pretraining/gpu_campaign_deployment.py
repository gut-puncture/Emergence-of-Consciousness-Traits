"""Provider-neutral deployment contracts for the bounded GPU campaign.

This module deliberately performs no rental, network, or GPU operation.  It
builds the immutable inputs and evidence that an operator must bind before it
is allowed to spend GPU time.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from twc.pretraining.configuration_semantic_code import SEMANTIC_CODE_PATHS_V2
from twc.repro.hashing import stable_json_hash
from twc.runtime.host import host_identity


REQUIRED_HEAD_DIMENSIONS = (32, 40, 48, 56, 64, 80)

_FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".secrets",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "artifacts",
        "cache",
        "caches",
        "docs",
        "private",
        "reports",
        "results",
    }
)
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_SAFE_PACKAGE_PIN = re.compile(r"^[A-Za-z0-9._-]+==[A-Za-z0-9._-]+$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+()/-]{0,127}$")
_SENSITIVE_WORDS = ("credential", "password", "secret", "token")
_ALLOWED_COMPILE_MODES = frozenset({"disabled", "default", "reduce-overhead", "max-autotune"})
_CAMPAIGN_RUNTIME_MODULES = frozenset(
    {
        "campaign.py",
        "campaign_store.py",
        "configuration_assessment.py",
        "configuration_composite_data.py",
        "configuration_composite_runtime.py",
        "configuration_experiment_cli.py",
        "configuration_experiments.py",
        "configuration_performance.py",
        "configuration_promotion.py",
        "configuration_result_builder.py",
        "configuration_semantic_code.py",
        "configuration_shape_campaign.py",
        "configuration_shape_copyback.py",
        "configuration_shape_probe.py",
        "configuration_shape_runtime.py",
        "configuration_study.py",
        "configuration_study_compiler.py",
        "configuration_study_execution.py",
        "configuration_study_lineage.py",
        "configuration_stage_authorization.py",
        "configuration_stage_control.py",
        "gpu_bootstrap_entry.py",
        "gpu_campaign_deployment.py",
        "gpu_campaign_evidence.py",
        "gpu_provider_evidence.py",
        "gpu_engineering_canary.py",
        "gpu_engineering_verdict.py",
    }
)
_PASSIVE_HISTORICAL_RUNTIME_PATHS = frozenset(
    {
        "training/twc/pretraining/gpu_campaign_cli.py",
        "training/twc/pretraining/gpu_canary_fixture.py",
    }
)
C23_PAID_ENTRYPOINT_MODULES_V1 = (
    "twc.pretraining.gpu_bootstrap_entry",
    "twc.pretraining.training.cli",
    "twc.pretraining.training.watchdog",
    "twc.pretraining.training.worker",
)


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _validate_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX_256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest, not a raw identity")
    return value


def _validate_label(value: object, field: str) -> str:
    if not isinstance(value, str) or _SAFE_LABEL.fullmatch(value) is None or "://" in value:
        raise ValueError(f"{field} must be a sanitized non-URL label")
    return value


def _validate_requested_path(raw: object) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise ValueError("runtime path must be a non-empty relative POSIX path")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or raw != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(
            "runtime path must be a normalized relative path without escape components"
        )
    if _contains_forbidden_part(path):
        raise ValueError(
            "runtime path contains a forbidden private, cache, report, or VCS component"
        )
    if not _is_allowlisted_runtime_path(path):
        raise ValueError(f"runtime path is outside the thin release allowlist: {raw}")
    return path


def _validate_receipt_path(raw: object) -> PurePosixPath:
    if isinstance(raw, str) and raw in _PASSIVE_HISTORICAL_RUNTIME_PATHS:
        return PurePosixPath(raw)
    return _validate_requested_path(raw)


def _contains_forbidden_part(path: PurePosixPath) -> bool:
    return any(part.casefold() in _FORBIDDEN_PARTS for part in path.parts)


def _is_allowlisted_runtime_path(path: PurePosixPath) -> bool:
    text = path.as_posix()
    if text in {"pyproject.toml", "uv.lock"}:
        return True
    if text in {"training/twc/__init__.py", "training/twc/pretraining/__init__.py"}:
        return True
    if text.startswith("configs/"):
        return path.suffix in {".yaml", ".yml", ".json"}
    if text == "training/twc/pretraining/training" or text.startswith(
        "training/twc/pretraining/training/"
    ):
        return path.suffix in {"", ".py", ".json", ".yaml", ".yml"}
    if text == "training/twc/pretraining/data_pipeline" or text.startswith(
        "training/twc/pretraining/data_pipeline/"
    ):
        return path.suffix in {"", ".py", ".json", ".yaml", ".yml"}
    if text == "training/twc/repro" or text.startswith("training/twc/repro/"):
        return path.suffix in {"", ".py", ".json", ".yaml", ".yml"}
    if text == "training/twc/runtime" or text.startswith("training/twc/runtime/"):
        return path.suffix in {"", ".py", ".json", ".yaml", ".yml"}
    if path.parent.as_posix() == "training/twc/pretraining":
        return path.name in _CAMPAIGN_RUNTIME_MODULES
    return False


def _module_name_for_runtime_path(relative: str) -> tuple[str, str]:
    path = PurePosixPath(relative)
    prefix = ("training", "twc")
    if path.suffix != ".py" or path.parts[:2] != prefix:
        raise ValueError(f"local import source is not a twc Python module: {relative}")
    parts = list(path.parts[1:])
    if parts[-1] == "__init__.py":
        parts.pop()
        kind = "package"
    else:
        parts[-1] = path.stem
        kind = "module"
    return ".".join(parts), kind


def _runtime_path_for_module(
    code_root: Path, module_name: str, *, required: bool
) -> str | None:
    if module_name != "twc" and not module_name.startswith("twc."):
        return None
    parts = module_name.split(".")
    module_path = code_root / "training" / Path(*parts)
    file_candidate = module_path.with_suffix(".py")
    package_candidate = module_path / "__init__.py"
    candidates = [
        candidate
        for candidate in (file_candidate, package_candidate)
        if candidate.is_file() and not candidate.is_symlink()
    ]
    if len(candidates) > 1:
        raise ValueError(f"local module has ambiguous source paths: {module_name}")
    if candidates:
        return candidates[0].relative_to(code_root).as_posix()
    if module_path.is_dir() and not module_path.is_symlink():
        return None  # PEP 420 namespace package.
    if required:
        raise ValueError(f"local import has no source module: {module_name}")
    return None


def _package_initializers(code_root: Path, module_name: str) -> set[str]:
    rows: set[str] = set()
    parts = module_name.split(".")
    for stop in range(1, len(parts)):
        candidate = code_root / "training" / Path(*parts[:stop]) / "__init__.py"
        if candidate.is_symlink():
            raise ValueError(f"runtime release rejects symlink: {candidate}")
        if candidate.is_file():
            rows.add(candidate.relative_to(code_root).as_posix())
    return rows


def _resolved_import_base(
    *, node: ast.ImportFrom, module_name: str, module_kind: str
) -> str | None:
    if node.level == 0:
        return node.module
    package = module_name if module_kind == "package" else module_name.rpartition(".")[0]
    if not package:
        raise ValueError(f"relative import has no package: {module_name}")
    try:
        return importlib.util.resolve_name(
            "." * node.level + (node.module or ""), package
        )
    except (ImportError, ValueError) as error:
        raise ValueError(f"relative local import cannot be resolved: {module_name}") from error


def _local_import_dependencies(code_root: Path, relative: str) -> set[str]:
    path = code_root / relative
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, SyntaxError, UnicodeError) as error:
        raise ValueError(f"runtime Python source is unreadable: {relative}") from error
    module_name, module_kind = _module_name_for_runtime_path(relative)
    dependencies: set[str] = set()

    def add_module(name: str, *, required: bool) -> None:
        if name != "twc" and not name.startswith("twc."):
            return
        source = _runtime_path_for_module(code_root, name, required=required)
        if source is not None:
            dependencies.add(source)
        dependencies.update(_package_initializers(code_root, name))

    def module_scope_nodes(statements: Sequence[ast.stmt]) -> Iterable[ast.AST]:
        for statement in statements:
            yield statement
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for field in ("body", "orelse", "finalbody"):
                nested = getattr(statement, field, None)
                if isinstance(nested, list):
                    yield from module_scope_nodes(nested)
            handlers = getattr(statement, "handlers", None)
            if isinstance(handlers, list):
                for handler in handlers:
                    yield from module_scope_nodes(handler.body)

    for node in module_scope_nodes(tree.body):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_module(alias.name, required=True)
        elif isinstance(node, ast.ImportFrom):
            base = _resolved_import_base(
                node=node, module_name=module_name, module_kind=module_kind
            )
            if base is None or (base != "twc" and not base.startswith("twc.")):
                continue
            add_module(base, required=True)
            for alias in node.names:
                if alias.name != "*":
                    add_module(f"{base}.{alias.name}", required=False)
        elif isinstance(node, ast.Call) and node.args:
            target = node.func
            is_dynamic_import = (
                isinstance(target, ast.Name) and target.id == "__import__"
            ) or (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "importlib"
                and target.attr == "import_module"
            )
            if (
                is_dynamic_import
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                add_module(node.args[0].value, required=True)
    dependencies.discard(relative)
    return dependencies


def _expand_local_import_closure(
    code_root: Path, initial_paths: Iterable[str]
) -> tuple[str, ...]:
    selected = set(initial_paths)
    pending = list(selected)
    while pending:
        relative = pending.pop()
        if not relative.endswith(".py"):
            continue
        for dependency in _local_import_dependencies(code_root, relative):
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    return tuple(sorted(selected))


def _validate_selected_local_import_closure(
    code_root: Path, selected_paths: set[str]
) -> None:
    for owner in sorted(path for path in selected_paths if path.endswith(".py")):
        missing = _local_import_dependencies(code_root, owner) - selected_paths
        if missing:
            raise ValueError(
                f"runtime dependency missing for {owner}: {', '.join(sorted(missing))}"
            )


def c23_runtime_release_paths(code_root: Path) -> tuple[str, ...]:
    """Derive the one minimal C2/C3 paid-runtime manifest from real imports."""

    root = Path(code_root).expanduser().resolve(strict=True)
    entry_paths = {
        _runtime_path_for_module(root, module, required=True)
        for module in C23_PAID_ENTRYPOINT_MODULES_V1
    }
    initial = {
        "pyproject.toml",
        "uv.lock",
        *SEMANTIC_CODE_PATHS_V2,
        *(path for path in entry_paths if path is not None),
    }
    paths = _expand_local_import_closure(root, initial)
    for relative in paths:
        _validate_requested_path(relative)
    return paths


def _expand_runtime_paths(code_root: Path, requested: Sequence[PurePosixPath]) -> list[Path]:
    selected: dict[str, Path] = {}
    for relative in requested:
        candidate = code_root / relative.as_posix()
        if candidate.is_symlink():
            raise ValueError(f"runtime release rejects symlink: {relative.as_posix()}")
        if not candidate.exists():
            raise ValueError(f"runtime path does not exist: {relative.as_posix()}")
        if candidate.is_file():
            selected[relative.as_posix()] = candidate
            continue
        if not candidate.is_dir():
            raise ValueError(f"runtime path is nonregular: {relative.as_posix()}")
        for child in sorted(candidate.rglob("*")):
            child_relative = PurePosixPath(child.relative_to(code_root).as_posix())
            if child.is_symlink():
                raise ValueError(f"runtime release rejects symlink: {child_relative.as_posix()}")
            if child.is_dir():
                continue
            if not child.is_file():
                raise ValueError(f"runtime path is nonregular: {child_relative.as_posix()}")
            if _contains_forbidden_part(child_relative) or not _is_allowlisted_runtime_path(
                child_relative
            ):
                continue
            selected[child_relative.as_posix()] = child
    return [selected[key] for key in sorted(selected)]


def _inventory_files(root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError("runtime inventory accepts regular non-symlink files only")
        size, digest = _sha256_file(path)
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": digest,
            }
        )
    return sorted(rows, key=lambda row: str(row["path"]))


def _full_release_inventory(root: Path) -> list[dict[str, Any]]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("runtime release content contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("runtime release content contains a nonregular path")
        _validate_requested_path(path.relative_to(root).as_posix())
        files.append(path)
    return _inventory_files(root, files)


def build_thin_runtime_release(
    *,
    code_root: Path,
    release_root: Path,
    relative_paths: Sequence[str],
) -> dict[str, Any]:
    """Copy only explicit runtime inputs into an immutable content-addressed release."""

    if code_root.is_symlink():
        raise ValueError("code root must not be a symlink")
    source = code_root.expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError("code root must be a real directory")
    target_root = release_root.expanduser().resolve()
    if target_root == source or source in target_root.parents or target_root in source.parents:
        raise ValueError("release root and code root must be non-nested")

    requested = sorted({_validate_requested_path(value) for value in relative_paths}, key=str)
    if not requested:
        raise ValueError("thin runtime release requires at least one requested path")
    if PurePosixPath("uv.lock") not in requested:
        raise ValueError("thin runtime release must explicitly include pinned uv.lock")

    files = _expand_runtime_paths(source, requested)
    inventory = _inventory_files(source, files)
    selected_paths = {str(row["path"]) for row in inventory}
    _validate_selected_local_import_closure(source, selected_paths)
    lock_rows = [row for row in inventory if row["path"] == "uv.lock"]
    if len(lock_rows) != 1:
        raise ValueError("thin runtime release must contain exactly one uv.lock")
    release_hash = stable_json_hash(inventory)
    target_root.mkdir(parents=True, exist_ok=True)
    final = target_root / f"gpu-runtime-{release_hash}"
    partial = target_root / f"gpu-runtime-{release_hash}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    if not final.exists():
        partial.mkdir()
        for source_file, row in zip(files, inventory, strict=True):
            destination = partial / str(row["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, destination)
        if _full_release_inventory(partial) != inventory:
            shutil.rmtree(partial)
            raise ValueError("runtime release copy readback mismatch")
        os.replace(partial, final)
    elif _full_release_inventory(final) != inventory:
        raise ValueError("existing runtime release content does not match its identity")

    payload: dict[str, Any] = {
        "artifact_kind": "twc_gpu_runtime_release_v1",
        "schema_version": 1,
        "release_hash": release_hash,
        "dependency_lock_sha256": lock_rows[0]["sha256"],
        "path": str(final),
        "requested_paths": [path.as_posix() for path in requested],
        "file_count": len(inventory),
        "bytes": sum(int(row["bytes"]) for row in inventory),
        "inventory": inventory,
        "raw_text_serialized": False,
        "secrets_serialized": False,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload


def validate_thin_runtime_release_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate release receipt semantics without reading the original filesystem."""

    expected_fields = {
        "artifact_kind", "schema_version", "release_hash",
        "dependency_lock_sha256", "path", "requested_paths", "file_count",
        "bytes", "inventory", "raw_text_serialized", "secrets_serialized",
        "receipt_hash",
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("artifact_kind") != "twc_gpu_runtime_release_v1"
        or receipt.get("schema_version") != 1
    ):
        raise ValueError("runtime release receipt semantic schema is invalid")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if receipt.get("receipt_hash") != stable_json_hash(unsigned):
        raise ValueError("runtime release receipt hash mismatch")

    inventory = receipt.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("runtime release inventory is missing")
    canonical_inventory: list[dict[str, Any]] = []
    for row in inventory:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "bytes", "sha256"}
            or type(row.get("bytes")) is not int
            or int(row["bytes"]) < 0
        ):
            raise ValueError("runtime release inventory row schema is invalid")
        path = _validate_receipt_path(row.get("path")).as_posix()
        canonical_inventory.append(
            {
                "path": path,
                "bytes": int(row["bytes"]),
                "sha256": _validate_digest(row.get("sha256"), "runtime file digest"),
            }
        )
    inventory_paths = [row["path"] for row in canonical_inventory]
    if canonical_inventory != inventory or inventory_paths != sorted(set(inventory_paths)):
        raise ValueError("runtime release inventory is not canonical")
    if stable_json_hash(canonical_inventory) != receipt.get("release_hash"):
        raise ValueError("runtime release inventory does not match release identity")

    requested_raw = receipt.get("requested_paths")
    if not isinstance(requested_raw, list) or not requested_raw:
        raise ValueError("runtime release requested paths are missing")
    requested = [_validate_receipt_path(value).as_posix() for value in requested_raw]
    if requested != sorted(set(requested)) or "uv.lock" not in requested:
        raise ValueError("runtime release requested paths are not canonical")
    for path in requested:
        prefix = f"{path}/"
        if not any(value == path or value.startswith(prefix) for value in inventory_paths):
            raise ValueError("runtime release requested path has no inventory content")
    lock_rows = [row for row in canonical_inventory if row["path"] == "uv.lock"]
    if (
        len(lock_rows) != 1
        or lock_rows[0]["sha256"] != receipt.get("dependency_lock_sha256")
    ):
        raise ValueError("runtime release dependency lock semantics are invalid")
    release_path = receipt.get("path")
    if (
        not isinstance(release_path, str)
        or not Path(release_path).is_absolute()
        or Path(release_path).name != f"gpu-runtime-{receipt.get('release_hash')}"
    ):
        raise ValueError("runtime release canonical path identity is invalid")
    if (
        receipt.get("file_count") != len(canonical_inventory)
        or receipt.get("bytes") != sum(row["bytes"] for row in canonical_inventory)
        or receipt.get("raw_text_serialized") is not False
        or receipt.get("secrets_serialized") is not False
    ):
        raise ValueError("runtime release semantic counts or privacy flags are invalid")
    return dict(receipt)


def verify_thin_runtime_release(receipt: Mapping[str, Any]) -> dict[str, Any]:
    validate_thin_runtime_release_receipt(receipt)
    raw_root = Path(str(receipt.get("path")))
    if raw_root.is_symlink():
        raise ValueError("runtime release receipt path must not be a symlink")
    root = raw_root.resolve(strict=True)
    inventory = _full_release_inventory(root)
    if inventory != receipt.get("inventory") or stable_json_hash(inventory) != receipt.get(
        "release_hash"
    ):
        raise ValueError("runtime release inventory/content mismatch")
    if root.name != f"gpu-runtime-{receipt.get('release_hash')}":
        raise ValueError("runtime release canonical path identity mismatch")
    _validate_selected_local_import_closure(
        root, {str(row["path"]) for row in inventory}
    )
    return {"path": str(root), "release_hash": receipt["release_hash"], "verified": True}


def build_c23_runtime_release(*, code_root: Path, release_root: Path) -> dict[str, Any]:
    """Build the canonical import-complete C2/C3 paid-device release."""

    paths = c23_runtime_release_paths(code_root)
    receipt = build_thin_runtime_release(
        code_root=code_root,
        release_root=release_root,
        relative_paths=paths,
    )
    verify_c23_runtime_release(receipt)
    return receipt


def verify_c23_runtime_release(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Verify physical bytes and the canonical C2/C3 manifest/import closure."""

    verified = verify_thin_runtime_release(receipt)
    root = Path(str(verified["path"])).resolve(strict=True)
    expected = c23_runtime_release_paths(root)
    requested = receipt.get("requested_paths")
    inventory = receipt.get("inventory")
    if (
        not isinstance(requested, list)
        or tuple(requested) != expected
        or not isinstance(inventory, list)
        or tuple(str(row.get("path")) for row in inventory) != expected
    ):
        raise ValueError("runtime release is not the canonical C2/C3 paid manifest")
    return verified


def relocate_thin_runtime_release(receipt: Mapping[str, Any], new_path: Path) -> dict[str, Any]:
    """Rebind an intact copied release to its canonical path on a new host."""

    if receipt.get("receipt_hash") != stable_json_hash(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    ):
        raise ValueError("source runtime release receipt hash mismatch")
    relocated = dict(receipt)
    relocated["path"] = str(new_path.expanduser().resolve(strict=True))
    relocated["receipt_hash"] = stable_json_hash(
        {key: value for key, value in relocated.items() if key != "receipt_hash"}
    )
    verify_thin_runtime_release(relocated)
    return relocated


def _validate_command(command: Sequence[str]) -> list[str]:
    if not command:
        raise ValueError("bootstrap command must not be empty")
    result: list[str] = []
    for token in command:
        if (
            not isinstance(token, str)
            or not token
            or any(character in token for character in "\r\n\0")
        ):
            raise ValueError("bootstrap command contains an invalid token")
        lowered = token.casefold()
        if (
            "://" in token
            or ("=" in token and _SAFE_PACKAGE_PIN.fullmatch(token) is None)
            or any(word in lowered for word in _SENSITIVE_WORDS)
        ):
            raise ValueError("bootstrap command contains a URL, assignment, or sensitive value")
        result.append(token)
    return result


def validate_bootstrap_plan(
    bootstrap_plan: Mapping[str, Any], *, release_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a copied bounded bootstrap plan without executing it."""

    validate_thin_runtime_release_receipt(release_receipt)
    expected_fields = {
        "artifact_kind", "schema_version", "release_hash",
        "dependency_lock_sha256", "commands", "environment_variable_names",
        "max_minutes", "max_cost_usd", "raw_text_serialized",
        "secrets_serialized", "bootstrap_plan_hash",
    }
    if (
        set(bootstrap_plan) != expected_fields
        or bootstrap_plan.get("artifact_kind") != "twc_gpu_bootstrap_plan_v1"
        or bootstrap_plan.get("schema_version") != 1
    ):
        raise ValueError("bootstrap plan semantic schema is invalid")
    unsigned = {
        key: value for key, value in bootstrap_plan.items() if key != "bootstrap_plan_hash"
    }
    if bootstrap_plan.get("bootstrap_plan_hash") != stable_json_hash(unsigned):
        raise ValueError("bootstrap plan hash mismatch")
    if (
        bootstrap_plan.get("release_hash") != release_receipt.get("release_hash")
        or bootstrap_plan.get("dependency_lock_sha256")
        != release_receipt.get("dependency_lock_sha256")
    ):
        raise ValueError("bootstrap plan release or dependency lock mismatch")
    commands = bootstrap_plan.get("commands")
    if not isinstance(commands, list) or not commands or any(
        not isinstance(command, list) or _validate_command(command) != command
        for command in commands
    ):
        raise ValueError("bootstrap plan commands are invalid")
    names = bootstrap_plan.get("environment_variable_names")
    if (
        not isinstance(names, list)
        or names != sorted(set(names))
        or any(
            not isinstance(name, str)
            or _ENVIRONMENT_NAME.fullmatch(name) is None
            or any(word in name.casefold() for word in _SENSITIVE_WORDS)
            for name in names
        )
    ):
        raise ValueError("bootstrap environment variable names are invalid")
    minutes = bootstrap_plan.get("max_minutes")
    cost = bootstrap_plan.get("max_cost_usd")
    if type(minutes) is not int or not 0 < minutes <= 10:
        raise ValueError("bootstrap plan must have a positive cap of at most 10 minutes")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not 0 < float(cost) <= 0.50
    ):
        raise ValueError("bootstrap plan cost cap must be at most USD 0.50")
    if (
        bootstrap_plan.get("raw_text_serialized") is not False
        or bootstrap_plan.get("secrets_serialized") is not False
    ):
        raise ValueError("bootstrap plan privacy flags are invalid")
    return dict(bootstrap_plan)


def build_bootstrap_plan(
    *,
    release_receipt: Mapping[str, Any],
    commands: Sequence[Sequence[str]],
    environment_variable_names: Sequence[str],
    max_minutes: int,
    max_cost_usd: float,
) -> dict[str, Any]:
    """Create a no-secret plan for a bounded, lock-pinned remote bootstrap."""

    verify_thin_runtime_release(release_receipt)
    if (
        isinstance(max_minutes, bool)
        or not isinstance(max_minutes, int)
        or not 0 < max_minutes <= 10
    ):
        raise ValueError("bootstrap must have a positive cap of at most 10 minutes")
    if (
        isinstance(max_cost_usd, bool)
        or not isinstance(max_cost_usd, (int, float))
        or not 0 < float(max_cost_usd) <= 0.50
    ):
        raise ValueError("bootstrap must have a positive cost cap of at most USD 0.50")
    safe_commands = [_validate_command(command) for command in commands]
    if not safe_commands:
        raise ValueError("bootstrap plan requires at least one command")
    safe_names: list[str] = []
    for name in environment_variable_names:
        if not isinstance(name, str) or _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise ValueError("bootstrap environment variable name is invalid")
        if any(word in name.casefold() for word in _SENSITIVE_WORDS):
            raise ValueError("bootstrap environment variable names must not name secrets")
        safe_names.append(name)

    payload: dict[str, Any] = {
        "artifact_kind": "twc_gpu_bootstrap_plan_v1",
        "schema_version": 1,
        "release_hash": release_receipt["release_hash"],
        "dependency_lock_sha256": release_receipt["dependency_lock_sha256"],
        "commands": safe_commands,
        "environment_variable_names": sorted(set(safe_names)),
        "max_minutes": max_minutes,
        "max_cost_usd": float(max_cost_usd),
        "raw_text_serialized": False,
        "secrets_serialized": False,
    }
    payload["bootstrap_plan_hash"] = stable_json_hash(payload)
    validate_bootstrap_plan(payload, release_receipt=release_receipt)
    return payload


def build_bootstrap_receipt(
    *,
    bootstrap_plan: Mapping[str, Any],
    release_receipt: Mapping[str, Any],
    elapsed_seconds: float,
    observed_cost_usd: float,
    command_exit_codes: Sequence[int],
) -> dict[str, Any]:
    """Bind successful bootstrap evidence and fail closed on every paid cap."""

    verify_thin_runtime_release(release_receipt)
    validate_bootstrap_plan(bootstrap_plan, release_receipt=release_receipt)
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or elapsed_seconds < 0
    ):
        raise ValueError("bootstrap elapsed time must be non-negative")
    if elapsed_seconds > float(bootstrap_plan["max_minutes"]) * 60:
        raise ValueError("bootstrap exceeded time cap")
    if (
        isinstance(observed_cost_usd, bool)
        or not isinstance(observed_cost_usd, (int, float))
        or observed_cost_usd < 0
    ):
        raise ValueError("bootstrap observed cost must be non-negative")
    if observed_cost_usd > float(bootstrap_plan["max_cost_usd"]):
        raise ValueError("bootstrap exceeded cost cap")
    expected_commands = bootstrap_plan.get("commands")
    if not isinstance(expected_commands, list) or len(command_exit_codes) != len(expected_commands):
        raise ValueError("bootstrap command exit-code count mismatch")
    if any(
        isinstance(code, bool) or not isinstance(code, int) or code != 0
        for code in command_exit_codes
    ):
        raise ValueError("bootstrap command failed")

    payload: dict[str, Any] = {
        "artifact_kind": "twc_gpu_bootstrap_receipt_v1",
        "schema_version": 1,
        "status": "ready",
        "bootstrap_plan_hash": bootstrap_plan["bootstrap_plan_hash"],
        "release_hash": release_receipt["release_hash"],
        "dependency_lock_sha256": release_receipt["dependency_lock_sha256"],
        "elapsed_seconds": float(elapsed_seconds),
        "observed_cost_usd": float(observed_cost_usd),
        "command_exit_codes": list(command_exit_codes),
        "raw_text_serialized": False,
        "secrets_serialized": False,
    }
    payload["receipt_hash"] = stable_json_hash(payload)
    return payload


def execute_bootstrap_plan(
    *,
    bootstrap_plan: Mapping[str, Any],
    release_receipt: Mapping[str, Any],
    gpu_hourly_price_usd: float,
    run: Callable[..., Any] = subprocess.run,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Execute the exact argv plan without a shell and enforce one wall-time cap."""

    verify_thin_runtime_release(release_receipt)
    validate_bootstrap_plan(bootstrap_plan, release_receipt=release_receipt)
    if not isinstance(gpu_hourly_price_usd, (int, float)) or isinstance(gpu_hourly_price_usd, bool) or gpu_hourly_price_usd <= 0:
        raise ValueError("bootstrap GPU hourly price must be positive")
    commands = bootstrap_plan.get("commands")
    assert isinstance(commands, list) and commands
    started = clock()
    deadline = started + float(bootstrap_plan["max_minutes"]) * 60
    exit_codes: list[int] = []
    for command in commands:
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("bootstrap exhausted its aggregate time cap")
        completed = run(
            list(command),
            cwd=str(release_receipt["path"]),
            shell=False,
            check=False,
            timeout=remaining,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        exit_codes.append(int(completed.returncode))
        if completed.returncode != 0:
            raise RuntimeError(f"bootstrap command failed with exit code {completed.returncode}")
    elapsed = clock() - started
    verify_thin_runtime_release(release_receipt)
    observed_cost = elapsed * float(gpu_hourly_price_usd) / 3600
    return build_bootstrap_receipt(
        bootstrap_plan=bootstrap_plan,
        release_receipt=release_receipt,
        elapsed_seconds=elapsed,
        observed_cost_usd=observed_cost,
        command_exit_codes=tuple(exit_codes),
    )


def _normalized_head_probes(probes: Mapping[int, bool]) -> dict[str, bool]:
    normalized: dict[int, bool] = {}
    for dimension, result in probes.items():
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise ValueError("head-dimension probe keys must be integers")
        if dimension not in REQUIRED_HEAD_DIMENSIONS:
            raise ValueError(f"unexpected head-dimension probe: {dimension}")
        if not isinstance(result, bool):
            raise ValueError("head-dimension probe results must be booleans")
        normalized[dimension] = result
    return {
        str(dimension): normalized.get(dimension, False)
        for dimension in REQUIRED_HEAD_DIMENSIONS
    }


def build_gpu_capability_receipt(
    *,
    provider_label: str,
    gpu_model: str,
    gpu_uuid_digest: str,
    driver_version: str,
    cuda_runtime_version: str,
    torch_version: str,
    sm: tuple[int, int],
    vram_bytes: int,
    supports_bf16: bool,
    supports_tf32: bool,
    native_varlen_available: bool,
    fused_optimizer_available: bool,
    compile_mode: str,
    dependency_lock_sha256: str,
    release_hash: str,
    host_identity_digest: str,
    storage_identity_digest: str,
    head_dimension_probes: Mapping[int, bool],
) -> dict[str, Any]:
    """Build sanitized GPU evidence and explicitly record readiness blockers."""

    _validate_label(provider_label, "provider_label")
    _validate_label(gpu_model, "gpu_model")
    _validate_label(driver_version, "driver_version")
    _validate_label(cuda_runtime_version, "cuda_runtime_version")
    _validate_label(torch_version, "torch_version")
    _validate_digest(gpu_uuid_digest, "gpu_uuid_digest")
    _validate_digest(dependency_lock_sha256, "dependency_lock_sha256")
    _validate_digest(release_hash, "release_hash")
    _validate_digest(host_identity_digest, "host_identity_digest")
    _validate_digest(storage_identity_digest, "storage_identity_digest")
    if (
        not isinstance(sm, tuple)
        or len(sm) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in sm)
    ):
        raise ValueError("sm must be a non-negative (major, minor) integer tuple")
    if isinstance(vram_bytes, bool) or not isinstance(vram_bytes, int) or vram_bytes <= 0:
        raise ValueError("vram_bytes must be a positive integer")
    capabilities = {
        "supports_bf16": supports_bf16,
        "supports_tf32": supports_tf32,
        "native_varlen_available": native_varlen_available,
        "fused_optimizer_available": fused_optimizer_available,
    }
    if any(not isinstance(value, bool) for value in capabilities.values()):
        raise ValueError("GPU capability flags must be booleans")
    if compile_mode not in _ALLOWED_COMPILE_MODES:
        raise ValueError("compile_mode is unsupported")
    normalized_probes = _normalized_head_probes(head_dimension_probes)

    blockers: list[str] = []
    if not supports_bf16:
        blockers.append("bf16")
    if not native_varlen_available:
        blockers.append("native_varlen")
    if not fused_optimizer_available:
        blockers.append("fused_optimizer")
    blockers.extend(
        f"head_dimension:{dimension}"
        for dimension in REQUIRED_HEAD_DIMENSIONS
        if not normalized_probes[str(dimension)]
    )
    payload: dict[str, Any] = {
        "artifact_kind": "twc_gpu_capability_receipt_v1",
        "schema_version": 1,
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "provider_label": provider_label,
        "gpu_model": gpu_model,
        "gpu_uuid_digest": gpu_uuid_digest,
        "driver_version": driver_version,
        "cuda_runtime_version": cuda_runtime_version,
        "torch_version": torch_version,
        "sm": list(sm),
        "vram_bytes": vram_bytes,
        **capabilities,
        "compile_mode": compile_mode,
        "dependency_lock_sha256": dependency_lock_sha256,
        "release_hash": release_hash,
        "host_identity_digest": host_identity_digest,
        "storage_identity_digest": storage_identity_digest,
        "required_head_dimensions": list(REQUIRED_HEAD_DIMENSIONS),
        "head_dimension_probes": normalized_probes,
        "raw_identifiers_serialized": False,
        "raw_text_serialized": False,
        "secrets_serialized": False,
    }
    payload["capability_hash"] = stable_json_hash(payload)
    return payload


def validate_gpu_capability_receipt(
    receipt: Mapping[str, Any], *, require_ready: bool
) -> dict[str, Any]:
    unsigned = {key: value for key, value in receipt.items() if key != "capability_hash"}
    if receipt.get("artifact_kind") != "twc_gpu_capability_receipt_v1":
        raise ValueError("GPU capability receipt schema mismatch")
    if receipt.get("capability_hash") != stable_json_hash(unsigned):
        raise ValueError("GPU capability receipt hash mismatch")
    if receipt.get("required_head_dimensions") != list(REQUIRED_HEAD_DIMENSIONS):
        raise ValueError("GPU capability required head dimensions mismatch")
    probes = receipt.get("head_dimension_probes")
    if not isinstance(probes, dict) or set(probes) != {
        str(value) for value in REQUIRED_HEAD_DIMENSIONS
    }:
        raise ValueError("GPU capability head-dimension evidence mismatch")
    try:
        sm_values = receipt["sm"]
        expected = build_gpu_capability_receipt(
            provider_label=receipt["provider_label"],
            gpu_model=receipt["gpu_model"],
            gpu_uuid_digest=receipt["gpu_uuid_digest"],
            driver_version=receipt["driver_version"],
            cuda_runtime_version=receipt["cuda_runtime_version"],
            torch_version=receipt["torch_version"],
            sm=(sm_values[0], sm_values[1]),
            vram_bytes=receipt["vram_bytes"],
            supports_bf16=receipt["supports_bf16"],
            supports_tf32=receipt["supports_tf32"],
            native_varlen_available=receipt["native_varlen_available"],
            fused_optimizer_available=receipt["fused_optimizer_available"],
            compile_mode=receipt["compile_mode"],
            dependency_lock_sha256=receipt["dependency_lock_sha256"],
            release_hash=receipt["release_hash"],
            host_identity_digest=receipt["host_identity_digest"],
            storage_identity_digest=receipt["storage_identity_digest"],
            head_dimension_probes={int(key): value for key, value in probes.items()},
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("GPU capability receipt semantic fields are invalid") from error
    if dict(receipt) != expected:
        raise ValueError("GPU capability receipt semantic blockers or canonical fields mismatch")
    if require_ready and (receipt.get("status") != "ready" or receipt.get("blockers") != []):
        raise ValueError("GPU capability receipt is not ready")
    return {"capability_hash": receipt["capability_hash"], "verified": True}


def _live_gpu_facts(index: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is unavailable")
    properties = torch.cuda.get_device_properties(index)
    identity = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits",
            "-i", str(index),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip().splitlines()
    if len(identity) != 1 or "," not in identity[0]:
        raise RuntimeError("nvidia-smi returned an unexpected identity shape")
    raw_uuid, driver = (part.strip() for part in identity[0].split(",", 1))
    if not raw_uuid or not driver:
        raise RuntimeError("nvidia-smi GPU identity is incomplete")
    try:
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401
        native_available = True
    except (ImportError, AttributeError):
        native_available = False
    try:
        parameter = torch.nn.Parameter(torch.ones(1, device=f"cuda:{index}"))
        torch.optim.AdamW((parameter,), fused=True)
        fused_available = True
    except (RuntimeError, TypeError, ValueError):
        fused_available = False
    head_probes: dict[int, bool] = {}
    if native_available:
        from twc.pretraining.training.model import DecoderConfig, PackedCausalBatch, initialize_decoder

        for head_dim in REQUIRED_HEAD_DIMENSIONS:
            try:
                config = DecoderConfig(128, 16, 8 * head_dim, 1, 8, 2, 24 * head_dim)
                model = initialize_decoder(config, seed=head_dim).to(f"cuda:{index}")
                model.configure_attention("native_varlen", "auto")
                batch = PackedCausalBatch.create(
                    torch.tensor([1, 2, 3], dtype=torch.int64),
                    torch.tensor([2, 3, 4], dtype=torch.int64),
                    (0, 3),
                )
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch.to(f"cuda:{index}"))
                head_probes[head_dim] = bool(torch.isfinite(output.loss_sum))
            except (RuntimeError, TypeError, ValueError):
                head_probes[head_dim] = False
            finally:
                torch.cuda.empty_cache()
    return {
        "gpu_model": str(properties.name),
        "gpu_uuid_digest": hashlib.sha256(raw_uuid.encode("utf-8")).hexdigest(),
        "driver_version": driver,
        "cuda_runtime_version": str(torch.version.cuda or "unavailable"),
        "torch_version": str(torch.__version__),
        "sm": (int(properties.major), int(properties.minor)),
        "vram_bytes": int(properties.total_memory),
        "supports_bf16": bool(torch.cuda.is_bf16_supported()),
        "supports_tf32": bool(properties.major >= 8),
        "native_varlen_available": native_available,
        "fused_optimizer_available": fused_available,
        "head_dimension_probes": head_probes,
    }


def capture_live_gpu_capability_receipt(
    *,
    provider_label: str,
    release_receipt: Mapping[str, Any],
    storage_root: Path,
    compile_mode: str,
    device_index: int = 0,
) -> dict[str, Any]:
    """Capture sanitized live GPU/environment evidence without serializing raw identities."""

    verified_release = verify_thin_runtime_release(release_receipt)
    root = storage_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("GPU campaign storage root must be a directory")
    facts = _live_gpu_facts(device_index)
    host_digest = host_identity()
    stat = root.stat()
    storage_digest = stable_json_hash(
        {"device_id": stat.st_dev, "inode": stat.st_ino, "mode": stat.st_mode & 0o777}
    )
    return build_gpu_capability_receipt(
        provider_label=provider_label,
        dependency_lock_sha256=release_receipt["dependency_lock_sha256"],
        release_hash=verified_release["release_hash"],
        host_identity_digest=host_digest,
        storage_identity_digest=storage_digest,
        compile_mode=compile_mode,
        **facts,
    )
