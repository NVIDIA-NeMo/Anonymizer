# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A DataDesigner release without the private hooks the source-frame binding needs aborts the whole run.

These cases remove a hook from a real DataDesigner and check that ``run()`` and ``preview()`` raise
``AnonymizerWorkflowError`` before any model request, instead of returning a partial result.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import yaml
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop
from data_designer.interface.data_designer import DataDesigner

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite, Substitute
from anonymizer.config.models import EvaluateModelSelection, RewriteModelSelection
from anonymizer.engine.ndd._graph_requests import SUPPORTED_DATA_DESIGNER_RANGE
from anonymizer.interface.errors import AnonymizerWorkflowError
from tests.interface.test_rewrite_withholding import _content

pytestmark = pytest.mark.compat

_RECORDS = ["Alice travels.", "Alice stays."]


def _designer(tmp_path: Path, sent: list[httpx.Request]) -> tuple[DataDesigner, list[Any]]:
    clients: list[Any] = []
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[
            ModelProvider(name="scripted", endpoint="http://scripted.invalid/v1", provider_type="openai", api_key="x")
        ],
        auto_configure_logging=False,
    )
    original = designer._create_resource_provider

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = "\n".join(
            message["content"]
            if isinstance(message["content"], str)
            else "\n".join(part.get("text", "") for part in message["content"])
            for message in body["messages"]
        )
        if body["model"] == "mapper" and "Generate synthetic replacements" not in text:
            message = "Hello!"  # health check
        else:
            content = json.dumps(_content(body["model"], text, "mixed"))
            message = content if body["model"] == "detector" else f"```json\n{content}\n```"
        return httpx.Response(
            200,
            json={
                "id": "scripted",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": message}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return respond(request)

    def resources(*args: Any, **kwargs: Any) -> Any:
        provider: Any = original(*args, **kwargs)
        registry = provider.model_registry
        factory = registry._model_facade_factory

        def facade(*args: Any, **kwargs: Any) -> Any:
            result: Any = factory(*args, **kwargs)
            result._client._inner._transport = httpx.MockTransport(record)
            clients.append(result._client)
            return result

        registry._model_facade_factory = facade
        return provider

    setattr(designer, "_create_resource_provider", resources)
    return designer, clients


def _model_configs() -> str:
    models = [
        ModelConfig(alias=name, model=name, provider="scripted", skip_health_check=name != "mapper")
        for name in ("detector", "singleton", "mapper")
    ]
    selected = {
        "detection": {
            "entity_detector": "detector",
            "entity_validator": ["singleton"],
            "entity_augmenter": "singleton",
            "latent_detector": "singleton",
        },
        "replace": {"replacement_generator": "mapper"},
        "rewrite": dict.fromkeys(RewriteModelSelection.model_fields, "singleton"),
        "evaluate": dict.fromkeys(EvaluateModelSelection.model_fields, "singleton"),
    }
    return yaml.safe_dump(
        {
            "model_configs": [model.model_dump(mode="json", exclude_none=True) for model in models],
            "selected_models": selected,
        }
    )


@pytest.mark.parametrize("fault", ["designer", "builder"])
@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
@pytest.mark.parametrize("mode", ["substitute", "rewrite"])
def test_missing_source_frame_hook_aborts_every_route_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, preview: bool, fault: str
) -> None:
    """Every route runs DD through ``NddAdapter.run_workflow``, which binds the source frame to DD's builder."""
    sent: list[httpx.Request] = []
    designer, clients = _designer(tmp_path, sent)
    if fault == "designer":
        monkeypatch.setattr(designer, "_create_dataset_builder", None)
    else:
        original = designer._create_dataset_builder

        def builder(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            setattr(result, "_initialize_generators_and_graph", None)
            return result

        monkeypatch.setattr(designer, "_create_dataset_builder", builder)
    source = tmp_path / "input.csv"
    pd.DataFrame({"text": _RECORDS}).to_csv(source, index=False)
    anonymizer = Anonymizer(data_designer=designer, model_configs=_model_configs())
    config = (
        AnonymizerConfig(replace=Substitute(), emit_telemetry=False)
        if mode == "substitute"
        else AnonymizerConfig(rewrite=Rewrite(), emit_telemetry=False)
    )
    data = AnonymizerInput(source=str(source), text_column="text")
    result: Any = None
    try:
        with pytest.raises(AnonymizerWorkflowError) as raised:
            if preview:
                result = anonymizer.preview(config=config, data=data, num_records=len(_RECORDS))
            else:
                result = anonymizer.run(config=config, data=data)
    finally:

        async def close() -> None:
            for client in clients:
                await client.aclose()

        asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=30)

    assert result is None
    assert not clients and not sent
    message = str(raised.value)
    attribute = (
        "DataDesigner._create_dataset_builder"
        if fault == "designer"
        else "DatasetBuilder._initialize_generators_and_graph"
    )
    assert f"private `{attribute}`" in message
    assert SUPPORTED_DATA_DESIGNER_RANGE in message
    assert "installed data-designer" in message
    assert "Alice" not in message
