# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from anonymizer.notebooks.local_inference.gliner2 import MODEL_ID, MODEL_REVISION, server


def test_main_serves_an_inherited_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(socket.SOMAXCONN)
        inherited_fd = os.dup(listener.fileno())
        observed_address: list[tuple[str, int]] = []
        runner = Mock()
        runner.run.side_effect = lambda *, sockets: observed_address.append(sockets[0].getsockname())
        monkeypatch.setattr(sys, "argv", ["server", "--fd", str(inherited_fd)])

        with (
            patch.object(server.uvicorn, "Config", return_value=Mock()),
            patch.object(server.uvicorn, "Server", return_value=runner),
        ):
            server.main()

        assert observed_address == [listener.getsockname()]


def test_dispatch_skips_cancelled_future_without_affecting_live_job() -> None:
    async def run_dispatch() -> None:
        detector = server.BatchDetector()
        loop = asyncio.get_running_loop()
        cancelled = loop.create_future()
        cancelled.cancel()
        live = loop.create_future()
        params = server.DetectParams(
            labels=("first_name",),
            threshold=0.3,
            chunk_length=384,
            overlap=128,
            flat_ner=False,
            inference_batch_size=8,
        )
        jobs = [
            server.DetectJob(text="cancelled", params=params, future=cancelled),
            server.DetectJob(text="live", params=params, future=live),
        ]
        expected = [[{"text": "Alice"}], [{"text": "Bob"}]]

        with patch.object(server, "detect_entities_for_texts", return_value=expected):
            await detector._dispatch(jobs)

        assert cancelled.cancelled()
        assert live.result() == expected[1]
        detector._executor.shutdown(wait=True)

    asyncio.run(run_dispatch())


def test_models_requires_token_and_reports_checkpoint_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(server.TOKEN_ENV, "secret")
    client = TestClient(server.app, raise_server_exceptions=False)
    unauthorized = client.get("/v1/models")
    with patch.object(server, "model", object()):
        authorized = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    metadata = authorized.json()["data"][0]
    assert metadata["id"] == MODEL_ID
    assert metadata["revision"] == MODEL_REVISION


def test_chat_completion_returns_anonymizer_entity_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(server.TOKEN_ENV, "secret")
    entities = [{"text": "Alice", "label": "first_name", "start": 0, "end": 5, "score": 0.9}]
    with (
        patch.object(server, "model", object()),
        patch.object(server.detector, "detect", AsyncMock(return_value=entities)),
    ):
        client = TestClient(server.app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Alice"}],
                "labels": ["first_name"],
            },
        )
    assert response.status_code == 200
    content = json.loads(response.json()["choices"][0]["message"]["content"])
    assert content == {"entities": entities}
