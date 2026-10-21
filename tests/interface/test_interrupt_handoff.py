# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Ctrl-C that lands while the caller is between taking a body and settling it still ends the run.

The caller's thread takes each synchronous body from a queue, runs it and settles the waiting
submission. A KeyboardInterrupt delivered right after the take, or right before the settle, must not
leave that submission waiting forever: the process has to exit on the interrupt, as in 0.4.0.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.compat

_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = r"""
import sys, tempfile, threading
from pathlib import Path

from anonymizer.engine.harness import kernel
from tests.interface.test_thread_placement import _run

point, method = sys.argv[1], sys.argv[2]
fired = []


class InterruptedQueue(kernel.Queue):
    def get(self, *args, **kwargs):
        item = super().get(*args, **kwargs)
        if point == "dequeue" and item[0] is not None and not fired:
            fired.append(True)
            raise KeyboardInterrupt
        return item


class InterruptedFuture(kernel.Future):
    def set_result(self, result):
        on_caller = threading.current_thread() is threading.main_thread()
        if point == "settle" and on_caller and not fired:
            fired.append(True)
            raise KeyboardInterrupt
        return super().set_result(result)


kernel.Queue = InterruptedQueue
kernel.Future = InterruptedFuture
_run(Path(tempfile.mkdtemp()), method=method)
print("finished")
"""


@pytest.mark.parametrize("method", ["redact", "substitute", "rewrite"])
@pytest.mark.parametrize("point", ["dequeue", "settle"])
def test_interrupt_between_take_and_settle_exits(tmp_path: Path, point: str, method: str) -> None:
    script = tmp_path / "interrupted.py"
    script.write_text(_SCRIPT)
    environment = {
        **os.environ,
        "NEMO_TELEMETRY_ENABLED": "false",
        "PYTHONPATH": os.pathsep.join([str(_ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    try:
        child = subprocess.run(
            [sys.executable, str(script), point, method],
            cwd=_ROOT,
            env=environment,
            capture_output=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"run hung after a Ctrl-C at {point}")
    assert child.returncode == -signal.SIGINT, child.stderr.decode()[-2000:]
    assert b"finished" not in child.stdout


def test_interrupt_settles_queued_bodies_once() -> None:
    """A body still queued when an earlier body is interrupted is settled once, as failed."""
    import asyncio
    from types import SimpleNamespace
    from typing import Any

    from anonymizer.engine.graph_sdk._execution_values import LocalFailure
    from anonymizer.engine.harness.kernel import _run_sync_profile

    queued: list[Any] = []

    def interrupted() -> Any:
        raise KeyboardInterrupt

    async def operation(channel: Any, handoff: Any) -> Any:
        first = channel(interrupted)
        second = channel(lambda: "ran")
        queued.append(second)
        await asyncio.gather(asyncio.wrap_future(first), asyncio.wrap_future(second))
        # Settled bodies resolve as failures, so the operation can finish before the stop request
        # arrives; like every production operation it then returns a result carrying its execution.
        return SimpleNamespace(_execution=None)

    with pytest.raises(KeyboardInterrupt):
        _run_sync_profile(operation)
    (second,) = queued
    assert second.result() == LocalFailure(failure="implementation_exception")
