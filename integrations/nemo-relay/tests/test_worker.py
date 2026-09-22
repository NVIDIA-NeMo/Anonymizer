# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from nemo_relay_plugin import PluginContext

from nemo_anonymizer_relay import worker


class Context:
    def __init__(self) -> None:
        self.registrations: list[tuple[str, Any]] = []

    def register_subscriber(self, name: str, callback: Any) -> None:
        self.registrations.append((name, callback))


def valid_config() -> dict[str, Any]:
    return {
        "endpoint": "unix:///tmp/anonymizer-test.sock",
        "enqueue_timeout_ms": 250,
        "max_event_bytes": 1024,
        "max_frame_bytes": 2048,
    }


def test_normalized_config_rejects_unknown_or_unsafe_limits() -> None:
    with pytest.raises(TypeError, match="endpoint must be"):
        worker.normalized_config({})
    with pytest.raises(ValueError, match="unknown configuration"):
        worker.normalized_config({"surprise": True})
    with pytest.raises(ValueError, match="version must be 1"):
        worker.normalized_config({"version": 2})
    with pytest.raises(TypeError, match="positive integer"):
        worker.normalized_config({"endpoint": "unix:///tmp/exporter.sock", "enqueue_timeout_ms": True})
    with pytest.raises(ValueError, match="must exceed"):
        worker.normalized_config(
            {"endpoint": "unix:///tmp/exporter.sock", "max_event_bytes": 1024, "max_frame_bytes": 1024}
        )
    with pytest.raises(ValueError, match="unix://"):
        worker.normalized_config({"endpoint": "tcp://127.0.0.1:8123"})


@pytest.mark.asyncio
async def test_worker_registers_only_subscriber_after_health_check(monkeypatch) -> None:
    exchange = AsyncMock(return_value={"status": "ready"})
    monkeypatch.setattr(worker, "exchange", exchange)
    context = Context()

    await worker.NemoAnonymizerWorker().register(cast(PluginContext, context), valid_config())

    assert [name for name, _ in context.registrations] == ["protected_export"]
    exchange.assert_awaited_once_with(
        "unix:///tmp/anonymizer-test.sock",
        {"kind": "health"},
        timeout_seconds=0.25,
        max_frame_bytes=2048,
    )


@pytest.mark.asyncio
async def test_worker_subscriber_sends_event_and_requires_acceptance(monkeypatch) -> None:
    exchange = AsyncMock(side_effect=[{"status": "ready"}, {"status": "accepted"}])
    monkeypatch.setattr(worker, "exchange", exchange)
    context = Context()
    await worker.NemoAnonymizerWorker().register(cast(PluginContext, context), valid_config())
    callback = context.registrations[0][1]

    source = {"uuid": "event-1", "data": {"owner": "Marisol Vega"}}
    await callback(source)

    request = exchange.await_args_list[1].args[1]
    assert request == {"kind": "event", "event": source}
    assert source == {"uuid": "event-1", "data": {"owner": "Marisol Vega"}}


@pytest.mark.asyncio
async def test_worker_fails_activation_when_service_is_not_ready(monkeypatch) -> None:
    monkeypatch.setattr(worker, "exchange", AsyncMock(return_value={"status": "starting"}))

    with pytest.raises(RuntimeError, match="not ready"):
        await worker.NemoAnonymizerWorker().register(cast(PluginContext, Context()), valid_config())


@pytest.mark.asyncio
async def test_worker_fails_closed_on_oversize_or_rejected_event(monkeypatch) -> None:
    exchange = AsyncMock(side_effect=[{"status": "ready"}, {"status": "full"}])
    monkeypatch.setattr(worker, "exchange", exchange)
    context = Context()
    await worker.NemoAnonymizerWorker().register(cast(PluginContext, context), valid_config())
    callback = context.registrations[0][1]

    with pytest.raises(RuntimeError, match="rejected event: full"):
        await callback({"data": "short"})
    with pytest.raises(RuntimeError, match="exceeds configured limit"):
        await callback({"data": "x" * 2048})
