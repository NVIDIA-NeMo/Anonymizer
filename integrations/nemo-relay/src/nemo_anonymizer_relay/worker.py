# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay worker that forwards copied events to the protected exporter service."""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from copy import deepcopy
from typing import Any

from nemo_relay_plugin import (
    ConfigDiagnostic,
    DiagnosticLevel,
    Json,
    PluginContext,
    WorkerPlugin,
    serve_plugin,
)

from nemo_anonymizer_relay.transport import DEFAULT_MAX_FRAME_BYTES, exchange, parse_endpoint

DEFAULT_CONFIG: dict[str, Json] = {
    "version": 1,
    "endpoint": None,
    "enqueue_timeout_ms": 1000,
    "max_event_bytes": 2 * 1024 * 1024,
    "max_frame_bytes": DEFAULT_MAX_FRAME_BYTES,
}


class NemoAnonymizerWorker(WorkerPlugin):
    """Register one quick subscriber; Anonymizer never runs in this process."""

    plugin_id = "nvidia.nemo_anonymizer"
    allows_multiple_components = False

    def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
        try:
            normalized_config(config)
        except (TypeError, ValueError) as error:
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="nemo_anonymizer.invalid_config",
                    message=str(error),
                )
            ]
        return []

    async def register(self, ctx: PluginContext, config: Json) -> None:
        settings = normalized_config(config)
        endpoint = settings["endpoint"]
        timeout_seconds = settings["enqueue_timeout_ms"] / 1000
        max_event_bytes = settings["max_event_bytes"]
        max_frame_bytes = settings["max_frame_bytes"]
        transport_options: dict[str, Any] = {
            "timeout_seconds": timeout_seconds,
            "max_frame_bytes": max_frame_bytes,
        }
        health = await exchange(
            endpoint,
            {"kind": "health"},
            **transport_options,
        )
        if not isinstance(health, dict) or health.get("status") != "ready":
            raise RuntimeError("NeMo Anonymizer exporter is not ready")

        async def subscriber(event: dict[str, Any]) -> None:
            encoded_bytes = len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            if encoded_bytes > max_event_bytes:
                raise RuntimeError("NeMo Anonymizer exporter event exceeds configured limit")
            response = await exchange(
                endpoint,
                {"kind": "event", "event": event},
                **transport_options,
            )
            if not isinstance(response, dict) or response.get("status") != "accepted":
                status = response.get("status") if isinstance(response, dict) else "invalid_response"
                raise RuntimeError(f"NeMo Anonymizer exporter rejected event: {status}")

        ctx.register_subscriber("protected_export", subscriber)


def normalized_config(raw: Json) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("configuration must be an object")
    unknown = set(raw) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown configuration field(s): {', '.join(sorted(unknown))}")
    config = deepcopy(DEFAULT_CONFIG)
    config.update(raw)
    if not isinstance(config["version"], int) or isinstance(config["version"], bool) or config["version"] != 1:
        raise ValueError("version must be 1")
    endpoint = config.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise TypeError("endpoint must be a non-empty string")
    parse_endpoint(endpoint)
    for field in ("enqueue_timeout_ms", "max_event_bytes", "max_frame_bytes"):
        value = config[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise TypeError(f"{field} must be a positive integer")
    if config["max_frame_bytes"] <= config["max_event_bytes"]:
        raise ValueError("max_frame_bytes must exceed max_event_bytes for the event envelope")
    return config


async def main() -> None:
    if sys.platform == "win32":
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            await serve_plugin(NemoAnonymizerWorker())
        finally:
            signal.signal(signal.SIGINT, previous)
    else:
        await serve_plugin(NemoAnonymizerWorker())


if __name__ == "__main__":
    asyncio.run(main())
