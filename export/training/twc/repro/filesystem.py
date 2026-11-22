"""Filesystem observations shared by source-cache budget checks."""

from __future__ import annotations

import os
from pathlib import Path


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for current, _, files in os.walk(path):
        for file in files:
            try:
                total += (Path(current) / file).stat().st_size
            except OSError:
                continue
    return total
