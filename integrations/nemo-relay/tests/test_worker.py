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


async def registered_subscriber(monkeypatch, *responses: dict[str, Any]) -> tuple[AsyncMock, Context, Any]:
    exchange = AsyncMock(side_effect=responses)
    monkeypatch.setattr(worker, "exchange", exchange)
    context = Context()
    await worker.NemoAnonymizerWorker().register(cast(PluginContext, context), valid_config())
    return exchange, context, context.registrations[0][1]


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


async def test_worker_registers_only_subscriber_after_health_check(monkeypatch) -> None:
    exchange, context, _ = await registered_subscriber(monkeypatch, {"status": "ready"})

    assert [name for name, _ in context.registrations] == ["protected_export"]
    exchange.assert_awaited_once_with(
        "unix:///tmp/anonymizer-test.sock",
        {"kind": "health"},
        timeout_seconds=0.25,
        max_frame_bytes=2048,
    )


async def test_worker_subscriber_forwards_event_without_mutating(monkeypatch) -> None:
    exchange, _, callback = await registered_subscriber(
        monkeypatch,
        {"status": "ready"},
        {"status": "accepted"},
    )

    source = {"uuid": "event-1", "data": {"owner": "Marisol Vega"}}
    await callback(source)

    request = exchange.await_args_list[1].args[1]
    assert request == {"kind": "event", "event": source}
    assert source == {"uuid": "event-1", "data": {"owner": "Marisol Vega"}}


async def test_worker_fails_activation_when_service_is_not_ready(monkeypatch) -> None:
    monkeypatch.setattr(worker, "exchange", AsyncMock(return_value={"status": "starting"}))

    with pytest.raises(RuntimeError, match="not ready"):
        await worker.NemoAnonymizerWorker().register(cast(PluginContext, Context()), valid_config())


async def test_worker_fails_closed_on_oversize_or_rejected_event(monkeypatch) -> None:
    _, _, callback = await registered_subscriber(monkeypatch, {"status": "ready"}, {"status": "full"})

    with pytest.raises(RuntimeError, match="rejected event: full"):
        await callback({"data": "short"})
    with pytest.raises(RuntimeError, match="exceeds configured limit"):
        await callback({"data": "x" * 2048})
