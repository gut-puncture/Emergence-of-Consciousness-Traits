"""PID-reuse-safe local process identity shared by runtime operators."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from twc.repro.hashing import stable_json_hash
from twc.runtime.host import host_identity


def provisional_process_identity(
    pid: int,
    *,
    command: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    """Retain a spawned PID before fallible start-token capture."""

    try:
        process_group_id: int | None = int(os.getpgid(pid))
    except (ProcessLookupError, PermissionError):
        process_group_id = None
    return _identity_payload(
        pid=pid,
        process_start_token_value=None,
        process_group_id=process_group_id,
        command=command,
        attempt_id=attempt_id,
    )


def capture_process_identity(
    pid: int,
    *,
    command: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    start_token = process_start_token(pid)
    if start_token is None:
        raise RuntimeError("could not establish process-start identity")
    return _identity_payload(
        pid=pid,
        process_start_token_value=start_token,
        process_group_id=int(os.getpgid(pid)),
        command=command,
        attempt_id=attempt_id,
    )


def _identity_payload(
    *,
    pid: int,
    process_start_token_value: str | None,
    process_group_id: int | None,
    command: Mapping[str, Any],
    attempt_id: str,
) -> dict[str, Any]:
    return {
        "pid": int(pid),
        "process_start_token": process_start_token_value,
        "host_id": host_identity(),
        "process_group_id": process_group_id,
        "command_hash": stable_json_hash(
            {"argv": command["argv"], "env": command.get("env", {})}
        ),
        "attempt_id": attempt_id,
        "label": command["label"],
        "role": command["role"],
    }


def observe_process(process: Mapping[str, Any]) -> str:
    if process.get("host_id") != host_identity():
        return "uncertain"
    pid = int(process["pid"])
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited"
    except PermissionError:
        return "uncertain"
    if process_is_zombie(pid):
        return "exited"
    observed = process_start_token(pid)
    if observed is None:
        return "uncertain"
    return (
        "confirmed_alive"
        if observed == process.get("process_start_token")
        else "uncertain"
    )


def process_start_token(pid: int) -> str | None:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            fields = proc_stat.read_text(encoding="utf-8").split()
            return stable_json_hash({"source": "proc", "start": fields[21]})
        except (OSError, IndexError):
            return None
    if sys.platform == "darwin":
        token = _darwin_process_start_token(pid)
        if token is not None:
            return token
    try:
        result = subprocess.run(
            ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return (
        stable_json_hash({"source": "ps", "start": value})
        if result.returncode == 0 and value
        else None
    )


def _darwin_process_start_token(pid: int) -> str | None:
    try:
        import ctypes
        import ctypes.util
        import struct

        library_path = ctypes.util.find_library("proc")
        if not library_path:
            return None
        library = ctypes.CDLL(library_path)
        buffer = ctypes.create_string_buffer(136)
        written = int(library.proc_pidinfo(int(pid), 3, 0, buffer, len(buffer)))
        if written < 136:
            return None
        seconds, microseconds = struct.unpack_from("QQ", buffer.raw, 120)
        if seconds <= 0:
            return None
        return stable_json_hash(
            {"source": "proc_pidinfo", "seconds": seconds, "microseconds": microseconds}
        )
    except (OSError, TypeError, ValueError):
        return None


def process_is_zombie(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            value = proc_stat.read_text(encoding="utf-8")
            close = value.rfind(")")
            return close >= 0 and value[close + 2 : close + 3] == "Z"
        except OSError:
            return False
    if sys.platform != "darwin":
        return False
    try:
        import ctypes
        import ctypes.util
        import struct

        library_path = ctypes.util.find_library("proc")
        if not library_path:
            return False
        library = ctypes.CDLL(library_path)
        buffer = ctypes.create_string_buffer(136)
        written = int(library.proc_pidinfo(int(pid), 3, 0, buffer, len(buffer)))
        if written == 0:
            return True
        if written < 8:
            return False
        status = struct.unpack_from("I", buffer.raw, 4)[0]
        return status == 5
    except (OSError, TypeError, ValueError):
        return False
