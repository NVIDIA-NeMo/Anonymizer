# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward graph cancellation to a DataDesigner run that executes on a graph worker thread.

In 0.4.0 a Ctrl-C reaches DataDesigner's own wait in the caller thread, which asks its async
scheduler to cancel (cancelling in-flight model requests) and then raises. Under the graph a
worker thread waits there instead, so the interrupt never arrives. This hook keeps DataDesigner's
wait unchanged outside graph workers; inside one, it asks the scheduler to cancel once the graph
invocation owning the worker is cancelled, and lets the cancellation surface as a workflow error.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from importlib.metadata import version
from threading import Lock
from typing import Any

_TICK_S = 0.02
_SUPPORTED_DATA_DESIGNER_VERSION = "0.9.4"
_lock = Lock()
_installed = False


def _worker_cancelled() -> Callable[[], bool] | None:
    from anonymizer.engine.graph_sdk._execution_delegated import _WORKER_BORROW

    borrow = _WORKER_BORROW.get()
    if borrow is None:
        return None
    return lambda: borrow.cancelled


def install() -> None:
    """Install once; a different DataDesigner version keeps its own wait (no graph cancellation)."""
    global _installed
    with _lock:
        if _installed:
            return
        _installed = True
        if version("data-designer") != _SUPPORTED_DATA_DESIGNER_VERSION:
            return
        from data_designer.engine.dataset_builders import dataset_builder

        original = getattr(dataset_builder, "_await_async_scheduler_result", None)
        if original is None:
            return

        def await_scheduler_result(future: concurrent.futures.Future[Any], scheduler: Any) -> None:
            cancelled = _worker_cancelled()
            if cancelled is None:
                return original(future, scheduler)
            try:
                while not future.done():
                    if cancelled():
                        # DataDesigner's own interrupt path: cancel scheduler tasks, then surface the outcome.
                        scheduler.request_cancel()
                        break
                    concurrent.futures.wait((future,), timeout=_TICK_S)
            except KeyboardInterrupt:
                # A caller-thread callback still receives Ctrl-C here; keep DataDesigner's handling of it.
                scheduler.request_cancel()
                try:
                    future.result()
                except Exception:
                    pass
                raise
            future.result()

        setattr(dataset_builder, "_await_async_scheduler_result", await_scheduler_result)
