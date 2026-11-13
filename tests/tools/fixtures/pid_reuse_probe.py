# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce process-group ID reuse inside a fresh PID namespace and observe ``stop``.

Run as PID 1 of ``unshare --pid --fork --mount-proc`` with the tools directory on
``PYTHONPATH``, passing a scratch directory and a scenario:

- ``stale``: the recorded group exits and its ID is reused before ``status`` and ``stop``.
- ``escalation``: ``stop`` sends SIGTERM, and the ID is reused before it would escalate.

Prints one JSON object describing what the receipt did.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from inference_service_compiler import compiler, launch_ownership, models, runtime

FIXTURE = Path(__file__).with_name("inference_process.py")


def start_group(
    scratch: Path, name: str, *, token: str | None, leader_exits: bool
) -> tuple[subprocess.Popen[bytes], list[int]]:
    ready = scratch / f"{name}.json"
    environment = {k: v for k, v in os.environ.items() if k != launch_ownership.LAUNCH_TOKEN_ENVIRONMENT_VARIABLE}
    if token is not None:
        environment[launch_ownership.LAUNCH_TOKEN_ENVIRONMENT_VARIABLE] = token
    argv = [sys.executable, str(FIXTURE), "group", "--workers", "2", "--ready", str(ready)]
    process = subprocess.Popen(
        [*argv, *(["--leader-exits"] if leader_exits else [])], env=environment, start_new_session=True
    )
    deadline = time.monotonic() + 20
    while not ready.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(name)
        time.sleep(0.05)
    return process, json.loads(ready.read_text())["workers"]


def reap_all() -> None:
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            return


def is_live(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except OSError:
        return False
    return state not in {"Z", "X"}


def reuse_group_id(scratch: Path, owned_pid: int) -> tuple[int, list[int]]:
    """Kill the recorded group and start an unrelated, token-less group under its ID."""
    try:
        os.killpg(owned_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    reap_all()
    Path("/proc/sys/kernel/ns_last_pid").write_text(str(owned_pid - 1))
    foreign, foreign_workers = start_group(scratch, "foreign", token=None, leader_exits=True)
    foreign.wait()
    return foreign.pid, foreign_workers


def main() -> None:
    scratch, scenario = Path(sys.argv[1]), sys.argv[2]
    token = "owned-launch-token"
    owned, _ = start_group(scratch, "owned", token=token, leader_exits=False)
    owned_pid = models.Pid(owned.pid)
    marker = runtime.read_process_start_marker(owned_pid)
    assert marker is not None
    handle = models.LocalProcessHandle(
        pid=owned_pid,
        start_marker=marker,
        launch_token_sha256=launch_ownership.launch_token_proof(token),
        stdout_path="stdout.log",
        stderr_path="stderr.log",
    )
    spec = models.ServiceSpec.model_validate({"backend": {"kind": "vllm", "model": "openai/gpt-oss-20b"}, "port": 8000})
    receipt = models.LaunchReceipt(plan=compiler.compile_spec(spec), launched_at=datetime.now(UTC), handle=handle)

    result: dict[str, object] = {"recorded_pgid": owned_pid}
    foreign_workers: list[int] = []
    wait_group_gone = runtime._wait_group_gone

    def reuse() -> None:
        result["foreign_pgid"], workers = reuse_group_id(scratch, owned_pid)
        foreign_workers.extend(workers)

    def reuse_during_grace_period(process_group_id: models.ProcessGroupId, deadline: float) -> bool:
        if "foreign_pgid" in result:
            return wait_group_gone(process_group_id, deadline)
        reuse()
        return False

    if scenario == "stale":
        reuse()
        reuse_hook: contextlib.AbstractContextManager[object] = contextlib.nullcontext()
    else:
        reuse_hook = mock.patch.object(runtime, "_wait_group_gone", side_effect=reuse_during_grace_period)
    with reuse_hook:
        observe_stop(receipt, scenario, foreign_workers, result)
    print(json.dumps(result))


def observe_stop(
    receipt: models.LaunchReceipt, scenario: str, foreign_workers: list[int], result: dict[str, object]
) -> None:
    try:
        if scenario == "stale":
            result["status"] = runtime.status_run(receipt).state
        try:
            stopped = runtime.stop_run(receipt)
            result["stop"] = {"outcome": stopped.outcome, "cleanup_complete": stopped.cleanup_complete}
        except runtime.RuntimeEffectError as exc:
            result["stop"] = {"code": exc.diagnostic.code, "cleanup_complete": exc.diagnostic.cleanup_complete}
        result["foreign_workers_alive"] = all(is_live(pid) for pid in foreign_workers)
    finally:
        for pid in foreign_workers:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        reap_all()


if __name__ == "__main__":
    main()
