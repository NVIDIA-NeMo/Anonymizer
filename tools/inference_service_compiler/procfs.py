# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only Linux ``/proc`` observations of managed processes."""

from __future__ import annotations

import secrets
from pathlib import Path

from inference_service_compiler.errors import RuntimeEffectError
from inference_service_compiler.launch_ownership import LAUNCH_TOKEN_ENVIRONMENT_VARIABLE, launch_token_proof
from inference_service_compiler.models import Pid, ProcessGroupId, Sha256Hex

PROC = Path("/proc")
EXITED_STATES = frozenset({"Z", "X"})
"""Zombie and dead: the process holds no memory and cannot be signaled away."""


def read_process_start_marker(pid: Pid) -> str | None:
    """Read Linux process start ticks to disambiguate PID reuse when available."""
    stat = _read_process_stat(pid)
    return stat[1] if stat is not None else None


def read_process_state(pid: Pid) -> str | None:
    """Read the Linux process state so zombies count as stopped."""
    stat = _read_process_stat(pid)
    return stat[0] if stat is not None else None


def parse_process_stat(payload: str) -> tuple[str, str] | None:
    """Return the state and start ticks from one ``/proc/<pid>/stat`` payload."""
    fields = _fields_after_command(payload)
    return (fields[0], fields[19]) if len(fields) > 19 else None


def live_group_members(process_group_id: ProcessGroupId) -> tuple[Pid, ...]:
    """Return the PIDs in the process group that have not exited.

    Raises:
        RuntimeEffectError: ``/proc`` cannot be listed (``process-identity-unavailable``).
    """
    try:
        entries = list(PROC.iterdir())
    except OSError as exc:
        raise RuntimeEffectError.of("process-identity-unavailable", f"cannot list processes: {exc}") from exc
    members: list[Pid] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            payload = (entry / "stat").read_text(encoding="utf-8")
        except OSError:
            continue
        parsed = _parse_group_stat(payload)
        if parsed is not None and parsed[1] == process_group_id and parsed[0] not in EXITED_STATES:
            members.append(Pid(int(entry.name)))
    return tuple(members)


def carries_launch_token(pid: Pid, token_sha256: Sha256Hex) -> bool:
    """Return whether the process inherited the launch token; a process that just exited counts as carrying it."""
    try:
        payload = (PROC / str(pid) / "environ").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return True
    except OSError:
        return False
    if not payload and read_process_state(pid) in {None, *EXITED_STATES}:
        return True
    prefix = f"{LAUNCH_TOKEN_ENVIRONMENT_VARIABLE}=".encode()
    for entry in payload.split(b"\0"):
        if entry.startswith(prefix):
            observed = launch_token_proof(entry.removeprefix(prefix).decode(errors="replace"))
            return secrets.compare_digest(observed, token_sha256)
    return False


def _read_process_stat(pid: Pid) -> tuple[str, str] | None:
    try:
        payload = (PROC / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    return parse_process_stat(payload)


def _parse_group_stat(payload: str) -> tuple[str, int] | None:
    """Return the state and process group ID from one ``/proc/<pid>/stat`` payload."""
    fields = _fields_after_command(payload)
    if len(fields) < 3 or not fields[2].lstrip("-").isdigit():
        return None
    return fields[0], int(fields[2])


def _fields_after_command(payload: str) -> list[str]:
    """Split the fields after the parenthesized command name, which may itself contain spaces."""
    _prefix, separator, suffix = payload.rpartition(")")
    return suffix.strip().split() if separator else []
