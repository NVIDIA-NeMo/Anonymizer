# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from anonymizer.notebooks.local_inference.gliner2 import MODEL_ID, MODEL_REVISION, server


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
