"""Stable local-host identity for process and lease ownership."""

from __future__ import annotations

import socket
import sys

from twc.repro.hashing import stable_json_hash


def host_identity() -> str:
    """Return a stable host binding without exposing the hostname."""

    return stable_json_hash({"hostname": socket.gethostname(), "platform": sys.platform})
