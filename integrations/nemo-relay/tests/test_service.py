# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import errno
import os
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from test_exporter import BlockingSink, RecordingSanitizer, RecordingSink, event

from nemo_anonymizer_relay import service
from nemo_anonymizer_relay.exporter import ProtectedExporter
from nemo_anonymizer_relay.service import ExporterServer, _remove_stale_socket
from nemo_anonymizer_relay.transport import exchange


@pytest.fixture
def private_socket_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=f"na-svc-{os.getpid()}-{uuid4().hex[:8]}-", dir="/tmp") as directory:
        yield Path(directory) / "exporter.sock"


@pytest.mark.asyncio
async def test_unix_service_health_event_flush_and_shutdown(private_socket_path: Path) -> None:
    socket = private_socket_path
    endpoint = f"unix://{socket}"
    sink = RecordingSink()
    exporter = ProtectedExporter(
        RecordingSanitizer(),
        sink,
        queue_capacity=4,
        max_event_bytes=4096,
        batch_max_events=4,
        batch_max_bytes=4096,
        batch_wait_seconds=0,
    )
    server = ExporterServer(endpoint, exporter, max_frame_bytes=8192)
    try:
        await server.start()
    except PermissionError as error:
        await server.stop()
        if error.errno == errno.EPERM:
            pytest.skip("execution sandbox blocks Unix-domain socket creation")
        raise

    assert stat.S_IMODE(socket.stat().st_mode) == 0o600
    health = await exchange(endpoint, {"kind": "health"}, timeout_seconds=1)
    accepted = await exchange(
        endpoint,
        {"kind": "event", "event": event()},
        timeout_seconds=1,
    )
    flushed = await exchange(endpoint, {"kind": "flush"}, timeout_seconds=1)
    stopping = await exchange(endpoint, {"kind": "shutdown"}, timeout_seconds=1)
    await server.wait()

    assert health["status"] == "ready"
    assert accepted["status"] == "accepted"
    assert flushed["status"] == "flushed"
    assert flushed["exported"] == 1
    assert stopping["status"] == "stopping"
    assert not socket.exists()
    assert [item["uuid"] for item in sink.events] == ["event-1"]


@pytest.mark.asyncio
async def test_service_rejects_invalid_event_without_stopping(private_socket_path: Path) -> None:
    socket = private_socket_path
    endpoint = f"unix://{socket}"
    server = ExporterServer(
        endpoint,
        ProtectedExporter(
            RecordingSanitizer(),
            RecordingSink(),
            queue_capacity=2,
            max_event_bytes=4096,
            batch_max_events=2,
            batch_max_bytes=4096,
            batch_wait_seconds=0,
        ),
        max_frame_bytes=8192,
    )
    try:
        await server.start()
    except PermissionError as error:
        await server.stop()
        if error.errno == errno.EPERM:
            pytest.skip("execution sandbox blocks Unix-domain socket creation")
        raise
    try:
        response = await exchange(
            endpoint,
            {"kind": "event", "event": "not-an-object"},
            timeout_seconds=1,
        )
        health = await exchange(endpoint, {"kind": "health"}, timeout_seconds=1)
    finally:
        await server.stop()

    assert response == {"status": "error", "error": "TypeError"}
    assert health["status"] == "ready"


def test_stale_socket_cleanup_refuses_regular_file(tmp_path: Path) -> None:
    endpoint = tmp_path / "anonymizer.sock"
    endpoint.write_text("do not delete", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to replace non-socket"):
        _remove_stale_socket(endpoint)

    assert endpoint.read_text(encoding="utf-8") == "do not delete"


@pytest.mark.asyncio
async def test_service_rejects_insecure_parent_before_socket_cleanup(monkeypatch, tmp_path: Path) -> None:
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    cleaned = False

    def record_cleanup(_path: Path) -> None:
        nonlocal cleaned
        cleaned = True

    monkeypatch.setattr(service, "_remove_stale_socket", record_cleanup)
    server = ExporterServer(
        f"unix://{parent / 'exporter.sock'}",
        ProtectedExporter(
            RecordingSanitizer(),
            RecordingSink(),
            queue_capacity=2,
            max_event_bytes=4096,
            batch_max_events=2,
            batch_max_bytes=4096,
            batch_wait_seconds=0,
        ),
        max_frame_bytes=8192,
    )

    with pytest.raises(ValueError, match="must not be accessible"):
        await server.start()

    assert not cleaned


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["busy", "failed"])
async def test_control_command_exits_nonzero_for_unhealthy_status(monkeypatch, status: str) -> None:
    monkeypatch.setattr(service, "exchange", AsyncMock(return_value={"status": status}))
    args = argparse.Namespace(
        command="health",
        endpoint="unix:///private/exporter.sock",
        timeout_seconds=1.0,
    )

    with pytest.raises(SystemExit) as error:
        await service._control(args)

    assert error.value.code == 1


@pytest.mark.asyncio
async def test_shutdown_stops_admission_before_draining() -> None:
    sink = BlockingSink()
    exporter = ProtectedExporter(
        RecordingSanitizer(),
        sink,
        queue_capacity=4,
        max_event_bytes=4096,
        batch_max_events=1,
        batch_max_bytes=4096,
        batch_wait_seconds=0,
    )
    exporter.start()
    assert exporter.enqueue(event(1), 100)["status"] == "accepted"
    assert await asyncio.wait_for(asyncio.to_thread(sink.started.wait, 0.5), timeout=0.75)
    server = ExporterServer("unix:///private/exporter.sock", exporter, max_frame_bytes=8192)

    shutdown = asyncio.create_task(server._dispatch({"kind": "shutdown"}))
    await asyncio.sleep(0)
    rejected = await server._dispatch({"kind": "event", "event": event(2)})
    sink.release.set()
    response = await asyncio.wait_for(shutdown, timeout=1)
    await asyncio.wait_for(server.wait(), timeout=1)

    assert rejected["status"] == "stopping"
    assert response["status"] == "stopping"
    assert [item["uuid"] for item in sink.events] == ["event-1"]
