# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from _support import BlockingSink, RecordingSanitizer, RecordingSink, make_event, make_exporter


async def test_exporter_batches_selected_envelope_fields_in_order() -> None:
    sanitizer = RecordingSanitizer()
    sink = RecordingSink()
    exporter = make_exporter(sanitizer, sink, queue_capacity=4, batch_max_events=4)
    selected = make_event(1)
    selected.update(
        name="marisol@example.com",
        category="Marisol Vega",
        data_schema={"name": "account-078-05-1120", "version": "1"},
    )
    exporter.start()

    assert exporter.enqueue(selected, 100)["status"] == "accepted"
    assert exporter.enqueue(make_event(2), 100)["status"] == "accepted"
    await exporter.close()

    assert [item["uuid"] for item in sink.events] == ["event-1", "event-2"]
    assert [item["metadata"]["nemo_anonymizer.sequence"] for item in sink.events] == [1, 2]
    inspected = sanitizer.calls[0][0]
    assert {key: inspected[key] for key in ("name", "category", "data_schema", "data")} == {
        "name": "marisol@example.com",
        "category": "Marisol Vega",
        "data_schema": {"name": "account-078-05-1120", "version": "1"},
        "data": {"owner": "Marisol Vega"},
    }
    assert sink.closed
    stats = exporter.snapshot()
    assert stats["accepted"] == stats["exported"] == 2
    assert stats["batches"] == 1


def test_exporter_enforces_event_and_queue_limits() -> None:
    exporter = make_exporter(
        queue_capacity=1,
        max_event_bytes=100,
        batch_max_events=1,
        batch_max_bytes=100,
    )

    assert exporter.enqueue(make_event(1), 101)["status"] == "oversize"
    assert exporter.enqueue(make_event(1), 100)["status"] == "accepted"
    assert exporter.enqueue(make_event(2), 100)["status"] == "full"
    stats = exporter.snapshot()
    assert stats["rejected_oversize"] == 1
    assert stats["rejected_full"] == 1
    assert stats["max_queue_depth"] == 1

    byte_bounded = make_exporter(
        queue_capacity=8,
        queue_max_bytes=150,
        max_event_bytes=100,
        batch_max_events=8,
        batch_max_bytes=150,
    )
    assert byte_bounded.enqueue(make_event(1), 100)["status"] == "accepted"
    assert byte_bounded.enqueue(make_event(2), 51)["status"] == "full"
    assert byte_bounded.snapshot()["max_queued_bytes"] == 100


async def test_backend_failure_exports_only_fail_closed_envelope() -> None:
    sink = RecordingSink()
    exporter = make_exporter(RecordingSanitizer(fail=True), sink)
    exporter.start()

    assert exporter.enqueue(make_event(), 100)["status"] == "accepted"
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


async def test_stream_chunk_payload_is_omitted_without_sending_it_to_anonymizer() -> None:
    sanitizer = RecordingSanitizer()
    sink = RecordingSink()
    exporter = make_exporter(sanitizer, sink)
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


async def test_metrics_failure_does_not_reject_export(monkeypatch, tmp_path: Path) -> None:
    sink = RecordingSink()
    exporter = make_exporter(sink=sink, metrics_path=tmp_path / "stats.json")
    monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    exporter.start()

    assert exporter.enqueue(make_event(), 100)["status"] == "accepted"
    await asyncio.wait_for(exporter.close(), timeout=1)

    assert [item["uuid"] for item in sink.events] == ["event-1"]
    assert exporter.snapshot()["metrics_failures"] >= 1


async def test_sink_failure_never_hangs_drain_and_stops_accepting() -> None:
    exporter = make_exporter(sink=RecordingSink(fail=True))
    exporter.start()
    assert exporter.enqueue(make_event(), 100)["status"] == "accepted"

    with pytest.raises(RuntimeError, match="exporter failed: OSError"):
        await asyncio.wait_for(exporter.flush(), timeout=1)

    stats = exporter.snapshot()
    assert stats["sink_failures"] == 1
    assert not stats["accepting"]
    assert exporter.enqueue(make_event(2), 100)["status"] == "failed"
    with pytest.raises(RuntimeError, match="exporter failed: OSError"):
        await asyncio.wait_for(exporter.close(), timeout=1)


async def test_slow_sink_does_not_block_event_admission() -> None:
    sink = BlockingSink()
    exporter = make_exporter(sink=sink, batch_max_events=1)
    exporter.start()
    assert exporter.enqueue(make_event(1), 100)["status"] == "accepted"

    assert await asyncio.wait_for(asyncio.to_thread(sink.started.wait, 0.5), timeout=0.75)
    assert exporter.enqueue(make_event(2), 100)["status"] == "accepted"

    sink.release.set()
    await asyncio.wait_for(exporter.close(), timeout=1)
    assert [item["uuid"] for item in sink.events] == ["event-1", "event-2"]
