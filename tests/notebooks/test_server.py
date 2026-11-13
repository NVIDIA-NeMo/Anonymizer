# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import socket
import sys
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from anonymizer.engine.local_inference.gliner2 import MODEL_ID, MODEL_REVISION, server
from anonymizer.engine.local_inference.gliner2.backend import LoadedModel
from anonymizer.engine.local_inference.gliner2.contract import DetectedEntity, DetectionRequest, Label, Span


def _entity(text: str, label: str = "pii", score: float = 0.9) -> DetectedEntity:
    return DetectedEntity(span=Span(0, len(text)), label=Label(label), text=text, score=score)


def _loaded() -> LoadedModel:
    return LoadedModel(model=cast(Any, object()), device="cpu", dtype="float32", attention="standard")


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
        expected = [[_entity("Alice")], [_entity("Bob")]]

        with patch.object(server, "detect_entities_for_texts", return_value=expected):
            await detector._dispatch(jobs, cast(Any, object()))

        assert cancelled.cancelled()
        assert live.result() == expected[1]
        detector._executor.shutdown(wait=True)

    asyncio.run(run_dispatch())


def test_models_requires_token_and_reports_checkpoint_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(server.TOKEN_ENV, "secret")
    client = TestClient(server.app, raise_server_exceptions=False)
    unauthorized = client.get("/v1/models")
    with patch.object(server, "loaded", _loaded()):
        authorized = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    metadata = authorized.json()["data"][0]
    assert metadata["id"] == MODEL_ID
    assert metadata["revision"] == MODEL_REVISION


@pytest.mark.parametrize(
    ("chunk_options", "expected_window"),
    [({}, (128, 32)), ({"chunk_length": 256, "overlap": 64}, (256, 64))],
)
def test_chat_completion_returns_anonymizer_entity_contract(
    monkeypatch: pytest.MonkeyPatch, chunk_options: dict[str, int], expected_window: tuple[int, int]
) -> None:
    monkeypatch.setenv(server.TOKEN_ENV, "secret")
    entities = [_entity("Alice", label="first_name", score=0.9)]
    with (
        patch.object(server, "loaded", _loaded()),
        patch.object(server.detector, "detect", AsyncMock(return_value=entities)) as detect,
    ):
        client = TestClient(server.app, raise_server_exceptions=False)
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret"},
            json={
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "Alice"}],
                "labels": ["first_name"],
                **chunk_options,
            },
        )
    assert response.status_code == 200
    content = json.loads(response.json()["choices"][0]["message"]["content"])
    assert content == {"entities": [{"text": "Alice", "label": "first_name", "start": 0, "end": 5, "score": 0.9}]}
    assert detect.await_args is not None
    params = detect.await_args.args[1]
    assert (params.chunk_length, params.overlap) == expected_window


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b'{"threshold": "high"}',
        b'{"chunk_length": null}',
        b'{"labels": "name"}',
    ],
)
def test_malformed_detection_request_is_rejected_with_422(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    monkeypatch.setenv(server.TOKEN_ENV, "secret")
    with patch.object(server, "loaded", _loaded()), patch.object(server.detector, "detect", AsyncMock()) as detect:
        client = TestClient(server.app, raise_server_exceptions=False)
        unauthorized = client.post("/v1/chat/completions", content=body)
        response = client.post("/v1/chat/completions", headers={"Authorization": "Bearer secret"}, content=body)
    assert unauthorized.status_code == 401
    assert response.status_code == 422
    detect.assert_not_awaited()


def test_detection_request_reads_text_from_the_last_message_and_ignores_unknown_keys() -> None:
    request = DetectionRequest.model_validate(
        {
            "messages": [
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": [{"type": "text", "text": "Alice "}, {"text": "Smith"}]},
            ],
            "labels": ["name", 3],
            "max_tokens": 5,
        }
    )
    assert request.text() == "Alice Smith"
    assert request.labels == ["name", "3"]
    assert DetectionRequest.model_validate({"messages": "nope"}).text() == ""


def test_default_inference_batch_size_is_32() -> None:
    assert DetectionRequest().batch_size == 32


def test_launch_ownership_constants_match_the_managed_tool_contract() -> None:
    assert server.LAUNCH_TOKEN_ENVIRONMENT_VARIABLE == "ANONYMIZER_INFERENCE_LAUNCH_TOKEN"
    assert server.LAUNCH_OWNERSHIP_HEADER == "X-Anonymizer-Launch-Token"
    assert server.LAUNCH_OWNERSHIP_PATH == "/_anonymizer/launch-ownership"
    assert server.LAUNCH_OWNERSHIP_PROOF_FIELD == "launch_token_sha256"
    tool = None
    for name in ("inference_service_compiler.launch_ownership", "inference_service_compiler.lifecycle"):
        try:
            tool = importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        if hasattr(tool, "LAUNCH_OWNERSHIP_PATH"):
            break
    assert tool is not None, "managed-tool launch-ownership module not found"
    for constant in (
        "LAUNCH_TOKEN_ENVIRONMENT_VARIABLE",
        "LAUNCH_OWNERSHIP_HEADER",
        "LAUNCH_OWNERSHIP_PATH",
        "LAUNCH_OWNERSHIP_PROOF_FIELD",
    ):
        assert getattr(server, constant) == getattr(tool, constant), constant


def test_launch_ownership_proves_the_inherited_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(server.TOKEN_ENV, "api-secret")
    monkeypatch.setenv(server.LAUNCH_TOKEN_ENVIRONMENT_VARIABLE, "launch-secret")
    client = TestClient(server.app, raise_server_exceptions=False)
    path = server.LAUNCH_OWNERSHIP_PATH
    header = server.LAUNCH_OWNERSHIP_HEADER

    right = client.get(path, headers={header: "launch-secret"})
    wrong = client.get(path, headers={header: "other"})
    missing = client.get(path)

    assert right.status_code == 200
    assert right.json() == {"launch_token_sha256": hashlib.sha256(b"launch-secret").hexdigest()}
    assert (wrong.status_code, wrong.json()) == (404, {"detail": "not found"})
    assert (missing.status_code, missing.json()) == (404, {"detail": "not found"})
    assert client.get("/v1/models", headers={header: "launch-secret"}).status_code == 401


def test_launch_ownership_is_not_found_without_a_launch_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(server.LAUNCH_TOKEN_ENVIRONMENT_VARIABLE, raising=False)
    client = TestClient(server.app, raise_server_exceptions=False)
    response = client.get(server.LAUNCH_OWNERSHIP_PATH, headers={server.LAUNCH_OWNERSHIP_HEADER: ""})
    assert (response.status_code, response.json()) == (404, {"detail": "not found"})
