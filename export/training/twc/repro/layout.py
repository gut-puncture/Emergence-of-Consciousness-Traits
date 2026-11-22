"""Canonical repository and Python-root layout for development and staged releases."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


LOGICAL_PACKAGE_ROOT = "twc"
PHYSICAL_PACKAGE_ROOT = Path("training/twc")
EXECUTION_IMPORT_ROOT = Path("training")
LAYOUT_VERSION = 2


def is_public_repo_root(path: str | Path) -> bool:
    root = Path(path)
    has_package = (root / PHYSICAL_PACKAGE_ROOT).is_dir() or (root / "twc").is_dir()
    return has_package and (
        (root / "pyproject.toml").is_file() or (root / "tests").is_dir()
    )


def find_repo_root(path: str | Path) -> Path:
    """Find the nearest current or legacy TWC repository/release root."""

    requested = Path(path).expanduser().resolve(strict=False)
    start = requested if requested.is_dir() else requested.parent
    for candidate in (start, *start.parents):
        if is_public_repo_root(candidate):
            return candidate
    raise ValueError(f"could not locate TWC repository root from {requested}")


def execution_python_root(
    execution_code_root: str | Path,
    *,
    code_identity_policy_version: int,
) -> Path:
    root = Path(execution_code_root).expanduser().resolve(strict=True)
    if int(code_identity_policy_version) >= 3:
        python_root = root / EXECUTION_IMPORT_ROOT
        if not (python_root / LOGICAL_PACKAGE_ROOT / "__init__.py").is_file():
            raise ValueError("execution release lacks training/twc package root")
        return python_root
    if not (root / LOGICAL_PACKAGE_ROOT / "__init__.py").is_file():
        raise ValueError("legacy execution release lacks twc package root")
    return root


def execution_environment(
    execution_code_root: str | Path,
    *,
    code_identity_policy_version: int,
    base: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    if overrides:
        env.update({str(key): str(value) for key, value in overrides.items()})
    python_root = execution_python_root(
        execution_code_root,
        code_identity_policy_version=code_identity_policy_version,
    )
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{python_root}{os.pathsep}{existing}" if existing else str(python_root)
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env
