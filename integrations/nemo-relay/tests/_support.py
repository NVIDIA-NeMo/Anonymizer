# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from nemo_anonymizer_relay.exporter import EventSanitizer, EventSink, ProtectedExporter


class RecordingSanitizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[Any] = []

    async def sanitize_export(self, value: Any) -> Any:
        self.calls.append(value)
        if self.fail:
            raise RuntimeError("detector unavailable")
        return value


class RecordingSink:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[dict[str, Any]] = []
        self.closed = False

    def write(self, events: list[dict[str, Any]]) -> None:
        if self.fail:
            raise OSError("disk full")
        self.events.extend(events)

    def close(self) -> None:
        self.closed = True


class BlockingSink(RecordingSink):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def write(self, events: list[dict[str, Any]]) -> None:
        self.started.set()
        if not self.release.wait(timeout=1):
            raise TimeoutError("test did not release sink")
        super().write(events)


def make_event(sequence: int = 1) -> dict[str, Any]:
    return {
        "uuid": f"event-{sequence}",
        "parent_uuid": "turn-1",
        "kind": "SPAN",
        "name": "tool",
        "category": "tool",
        "data": {"owner": "Marisol Vega"},
        "metadata": {},
    }


def make_exporter(
    sanitizer: EventSanitizer | None = None,
    sink: EventSink | None = None,
    *,
    queue_capacity: int = 2,
    queue_max_bytes: int = 64 * 1024 * 1024,
    max_event_bytes: int = 4096,
    batch_max_events: int = 2,
    batch_max_bytes: int = 4096,
    batch_wait_seconds: float = 0,
    metrics_path: Path | None = None,
) -> ProtectedExporter:
    return ProtectedExporter(
        sanitizer if sanitizer is not None else RecordingSanitizer(),
        sink if sink is not None else RecordingSink(),
        queue_capacity=queue_capacity,
        queue_max_bytes=queue_max_bytes,
        max_event_bytes=max_event_bytes,
        batch_max_events=batch_max_events,
        batch_max_bytes=batch_max_bytes,
        batch_wait_seconds=batch_wait_seconds,
        metrics_path=metrics_path,
    )
