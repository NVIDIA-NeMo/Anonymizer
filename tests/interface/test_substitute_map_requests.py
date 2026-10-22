# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Substitute replacement-map requests through the public API: concurrency matches 0.4.0.

Only the HTTP transport is scripted (``httpx.MockTransport`` with a fixed latency); DataDesigner and the
replacement-map generation run for real. Runtime: about 3 s.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import yaml
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig, ModelProvider
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop
from data_designer.interface.data_designer import DataDesigner

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Substitute

pytestmark = pytest.mark.compat

_LATENCY_S = 0.25
_RECORDS = 16
_NAMES = [f"Zorblax{chr(97 + index)}" for index in range(_RECORDS)]
_TEXTS = [f"{name} waved." for name in _NAMES]


def _completion(model: str, content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "scripted",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def _fenced(value: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(value) + "\n```"


class _LatencyMock:
    """Answers detection at once and every replacement-map request after a fixed sleep; records the map window."""

    def __init__(self) -> None:
        self.windows: list[tuple[float, float]] = []
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model, messages = body["model"], json.dumps(body["messages"])
        if model == "detector":
            name = next(name for name in _NAMES if f"{name} waved." in messages)
            entity = {"text": name, "label": "first_name", "start": 0, "end": len(name), "score": 0.9}
            return _completion(model, json.dumps({"entities": [entity]}))
        if "Validate entity tags" in messages:
            ids = dict.fromkeys(f"first_name_0_{len(name)}" for name in _NAMES if name in messages)
            return _completion(model, _fenced({"decisions": [{"id": item, "decision": "keep"} for item in ids]}))
        if "Find untagged sensitive entities" in messages:
            return _completion(model, _fenced({"entities": []}))
        if "Generate synthetic replacements" not in messages:
            return _completion(model, "Hello!")
        start = time.monotonic()
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(_LATENCY_S)
        finally:
            self.in_flight -= 1
        self.windows.append((start, time.monotonic()))
        replacements = [
            {"original": name, "label": "first_name", "synthetic": "Maya"} for name in _NAMES if name in messages
        ]
        return _completion(model, _fenced({"replacements": replacements}))


@pytest.fixture
def latency_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[DataDesigner, _LatencyMock]]:
    mock = _LatencyMock()
    clients: list[Any] = []
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[
            ModelProvider(name="scripted", endpoint="http://scripted.invalid/v1", provider_type="openai", api_key="x")
        ],
        auto_configure_logging=False,
    )
    original = designer._create_resource_provider

    def resources(*args: Any, **kwargs: Any) -> Any:
        provider: Any = original(*args, **kwargs)
        factory = provider.model_registry._model_facade_factory

        def facade(*args: Any, **kwargs: Any) -> Any:
            result: Any = factory(*args, **kwargs)
            result._client._inner._transport = httpx.MockTransport(mock)
            clients.append(result._client)
            return result

        provider.model_registry._model_facade_factory = facade
        return provider

    monkeypatch.setattr(designer, "_create_resource_provider", resources)
    try:
        yield designer, mock
    finally:

        async def close() -> None:
            for client in clients:
                await client.aclose()

        asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=30)


def test_substitute_map_requests_run_concurrently_up_to_max_parallel_requests(
    latency_mock: tuple[DataDesigner, _LatencyMock], tmp_path: Path
) -> None:
    designer, mock = latency_mock
    parallel = ChatCompletionInferenceParams(max_parallel_requests=_RECORDS)
    models = [
        ModelConfig(alias=name, model=name, provider="scripted", skip_health_check=name == "detector")
        for name in ("detector", "singleton")
    ] + [ModelConfig(alias="mapper", model="mapper", provider="scripted", inference_parameters=parallel)]
    selected = {
        "detection": {
            "entity_detector": "detector",
            "entity_validator": ["singleton"],
            "entity_augmenter": "singleton",
            "latent_detector": "singleton",
        },
        "replace": {"replacement_generator": "mapper"},
    }
    configs = {
        "model_configs": [model.model_dump(mode="json", exclude_none=True) for model in models],
        "selected_models": selected,
    }
    source = tmp_path / "input.csv"
    pd.DataFrame({"text": _TEXTS}).to_csv(source, index=False)

    result = Anonymizer(data_designer=designer, model_configs=yaml.safe_dump(configs)).run(
        config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source), text_column="text")
    )

    assert not result.failed_records
    assert result.dataframe["text_replaced"].tolist() == ["Maya waved."] * _RECORDS
    assert len(mock.windows) == _RECORDS
    map_stage = max(end for _, end in mock.windows) - min(start for start, _ in mock.windows)
    # 0.4.0 runs the map in one wave (about 1 x latency); one request at a time takes about 16 x latency.
    assert map_stage < 3 * _LATENCY_S, f"map stage {map_stage:.2f}s for {_RECORDS} requests of {_LATENCY_S}s"
    assert mock.peak >= 8
