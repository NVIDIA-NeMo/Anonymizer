# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from nemo_anonymizer_relay.exporter import ProtectedExporter


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


def event(sequence: int = 1) -> dict[str, Any]:
    return {
        "uuid": f"event-{sequence}",
        "parent_uuid": "turn-1",
        "kind": "SPAN",
        "name": "tool",
        "category": "tool",
        "data": {"owner": "Marisol Vega"},
        "metadata": {},
    }


@pytest.mark.asyncio
async def test_exporter_batches_in_order_and_preserves_envelope() -> None:
    sanitizer = RecordingSanitizer()
    sink = RecordingSink()
    exporter = ProtectedExporter(
        sanitizer,
        sink,
        queue_capacity=4,
        max_event_bytes=4096,
        batch_max_events=4,
        batch_max_bytes=4096,
        batch_wait_seconds=0,
    )
    exporter.start()

    assert exporter.enqueue(event(1), 100)["status"] == "accepted"
    assert exporter.enqueue(event(2), 100)["status"] == "accepted"
    await exporter.close()

    assert [item["uuid"] for item in sink.events] == ["event-1", "event-2"]
    assert [item["metadata"]["nemo_anonymizer.sequence"] for item in sink.events] == [1, 2]
    assert sink.closed
    stats = exporter.snapshot()
    assert stats["accepted"] == stats["exported"] == 2
    assert stats["batches"] == 1


def test_exporter_rejects_oversize_and_full_queues() -> None:
    exporter = ProtectedExporter(
        RecordingSanitizer(),
        RecordingSink(),
        queue_capacity=1,
        max_event_bytes=100,
        batch_max_events=1,
        batch_max_bytes=100,
        batch_wait_seconds=0,
    )

    assert exporter.enqueue(event(1), 101)["status"] == "oversize"
    assert exporter.enqueue(event(1), 100)["status"] == "accepted"
    assert exporter.enqueue(event(2), 100)["status"] == "full"
    stats = exporter.snapshot()
    assert stats["rejected_oversize"] == 1
    assert stats["rejected_full"] == 1
    assert stats["max_queue_depth"] == 1


def test_exporter_bounds_total_queued_bytes_independently_of_event_count() -> None:
    exporter = ProtectedExporter(
        RecordingSanitizer(),
        RecordingSink(),
        queue_capacity=8,
        queue_max_bytes=150,
        max_event_bytes=100,
        batch_max_events=8,
        batch_max_bytes=150,
        batch_wait_seconds=0,
    )

    assert exporter.enqueue(event(1), 100)["status"] == "accepted"
    assert exporter.enqueue(event(2), 51)["status"] == "full"
    assert exporter.snapshot()["max_queued_bytes"] == 100


@pytest.mark.asyncio
async def test_backend_failure_exports_only_fail_closed_envelope() -> None:
    sink = RecordingSink()
    exporter = ProtectedExporter(
        RecordingSanitizer(fail=True),
        sink,
        queue_capacity=2,
        max_event_bytes=4096,
        batch_max_events=2,
        batch_max_bytes=4096,
        batch_wait_seconds=0,
    )
    exporter.start()

    assert exporter.enqueue(event(), 100)["status"] == "accepted"
    await exporter.close()

    assert sink.events == [
        {
            "uuid": "event-1",
            "parent_uuid": "turn-1",
            "kind": "SPAN",
            "name": "nemo_anonymizer.omitted",
            "category": "unknown",
            "data_schema": None,
            "data": None,
            "category_profile": None,
            "metadata": {
                "nemo_anonymizer.coverage": "omitted",
                "nemo_anonymizer.failure": "RuntimeError",
                "nemo_anonymizer.sequence": 1,
            },
        }
    ]
    assert exporter.snapshot()["backend_failures"] == 1


@pytest.mark.asyncio
async def test_exporter_inspects_caller_selected_envelope_labels() -> None:
    sanitizer = RecordingSanitizer()
    sink = RecordingSink()
    exporter = ProtectedExporter(
        sanitizer,
        sink,
        queue_capacity=2,
        max_event_bytes=4096,
        batch_max_events=2,
        batch_max_bytes=4096,
        batch_wait_seconds=0,
    )
    exporter.start()
    source = event()
    source.update(
        {
            "name": "marisol@example.com",
            "category": "Marisol Vega",
            "data_schema": {"name": "account-078-05-1120", "version": "1"},
        }
    )

    assert exporter.enqueue(source, 200)["status"] == "accepted"
    await exporter.close()

    assert len(sanitizer.calls) == 1
    selected = sanitizer.calls[0][0]
    assert selected["name"] == "marisol@example.com"
    assert selected["category"] == "Marisol Vega"
    assert selected["data_schema"] == {
        "name": "account-078-05-1120",
        "version": "1",
    }
    assert selected["data"] == {"owner": "Marisol Vega"}


@pytest.mark.asyncio
async def test_stream_chunk_payload_is_omitted_without_sending_it_to_anonymizer() -> None:
    sanitizer = RecordingSanitizer()
    sink = RecordingSink()
    exporter = ProtectedExporter(
        sanitizer,
        sink,
        queue_capacity=2,
        max_event_bytes=4096,
        batch_max_events=2,
        batch_max_bytes=4096,
        batch_wait_seconds=0,
    )
    exporter.start()
    chunk = {
        "uuid": "chunk-1",
        "parent_uuid": "llm-1",
        "kind": "mark",
        "name": "llm.chunk",
        "category": "llm",
        "data": {"delta": "marisol@example.com"},
        "category_profile": {"text": "marisol@example.com"},
        "metadata": {"raw": "marisol@example.com"},
    }

    assert exporter.enqueue(chunk, 200)["status"] == "accepted"
    await exporter.close()

    assert sanitizer.calls == [[]]
    assert sink.events == [
        {
            "uuid": "chunk-1",
            "parent_uuid": "llm-1",
            "kind": "mark",
            "name": "llm.chunk",
            "category": "llm",
            "data_schema": None,
            "data": None,
            "category_profile": None,
            "metadata": {
                "nemo_anonymizer.coverage": "stream_chunk_payload_omitted",
                "nemo_anonymizer.provider_body_omitted": True,
                "nemo_anonymizer.sequence": 1,
            },
        }
    ]
    assert exporter.snapshot()["payloads_omitted"] == 1


@pytest.mark.asyncio
async def test_metrics_failure_does_not_reject_or_leak_source(monkeypatch, tmp_path: Path) -> None:
    sink = RecordingSink()
    exporter = ProtectedExporter(
        RecordingSanitizer(),
        sink,
        queue_capacity=2,
        max_event_bytes=4096,
        batch_max_events=2,
        batch_max_bytes=4096,
        batch_wait_seconds=0,
        metrics_path=tmp_path / "stats.json",
    )
    monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    exporter.start()

    assert exporter.enqueue(event(), 100)["status"] == "accepted"
    await asyncio.wait_for(exporter.close(), timeout=1)

    assert [item["uuid"] for item in sink.events] == ["event-1"]
    assert exporter.snapshot()["metrics_failures"] >= 1


@pytest.mark.asyncio
async def test_sink_failure_never_hangs_drain_and_stops_accepting() -> None:
    exporter = ProtectedExporter(
        RecordingSanitizer(),
        RecordingSink(fail=True),
        queue_capacity=2,
        max_event_bytes=4096,
        batch_max_events=2,
        batch_max_bytes=4096,
        batch_wait_seconds=0,
    )
    exporter.start()
    assert exporter.enqueue(event(), 100)["status"] == "accepted"

    with pytest.raises(RuntimeError, match="exporter failed: OSError"):
        await asyncio.wait_for(exporter.flush(), timeout=1)

    stats = exporter.snapshot()
    assert stats["sink_failures"] == 1
    assert not stats["accepting"]
    assert exporter.enqueue(event(2), 100)["status"] == "failed"
    with pytest.raises(RuntimeError, match="exporter failed: OSError"):
        await asyncio.wait_for(exporter.close(), timeout=1)


@pytest.mark.asyncio
async def test_slow_sink_does_not_block_event_admission() -> None:
    sink = BlockingSink()
    exporter = ProtectedExporter(
        RecordingSanitizer(),
        sink,
        queue_capacity=2,
        max_event_bytes=4096,
        batch_max_events=1,
        batch_max_bytes=4096,
        batch_wait_seconds=0,
    )
    exporter.start()
    assert exporter.enqueue(event(1), 100)["status"] == "accepted"

    assert await asyncio.wait_for(asyncio.to_thread(sink.started.wait, 0.5), timeout=0.75)
    assert exporter.enqueue(event(2), 100)["status"] == "accepted"

    sink.release.set()
    await asyncio.wait_for(exporter.close(), timeout=1)
    assert [item["uuid"] for item in sink.events] == ["event-1", "event-2"]
