# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real DataDesigner detection and local Redact with scripted HTTP transport."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
import yaml
from data_designer.config import ModelConfig, ModelProvider
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop
from data_designer.interface import DataDesigner

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput
from anonymizer.config.models import DetectionModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow


@pytest.fixture
def scripted_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[DataDesigner, list[ModelConfig], DetectionModelSelection, Counter[str]]]:
    calls: Counter[str] = Counter()
    clients: list[Any] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert "_anonymizer_source_" not in request.content.decode()
        body = json.loads(request.content)
        model = body["model"]
        messages = json.dumps(body["messages"])
        role = (
            "detector"
            if model == "detector"
            else "validator"
            if "Validate entity tags" in messages
            else "augmenter"
            if "Find untagged sensitive entities" in messages
            else "readiness"
        )
        calls[role] += 1
        if role == "detector" and "Alice fails." in messages:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "scripted row failure",
                        "type": "invalid_request_error",
                        "code": "invalid_request_error",
                    }
                },
            )
        if role == "detector":
            content = {"entities": [{"text": "Alice", "label": "first_name", "start": 0, "end": 5, "score": 0.9}]}
        elif role == "validator":
            content = {"decisions": [{"id": "first_name_0_5", "decision": "keep"}]}
        else:
            assert model == "singleton"
            content = {"entities": []}
        return httpx.Response(
            200,
            json={
                "id": "scripted",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content)
                            if role == "detector"
                            else "```json\n" + json.dumps(content) + "\n```",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[
            ModelProvider(
                name="scripted", endpoint="http://scripted.invalid/v1", provider_type="openai", api_key="EMPTY"
            )
        ],
        auto_configure_logging=False,
    )
    original = designer._create_resource_provider

    def resources(*args: Any, **kwargs: Any) -> Any:
        provider: Any = original(*args, **kwargs)
        registry = provider.model_registry
        factory = registry._model_facade_factory

        def facade(*args: Any, **kwargs: Any) -> Any:
            result = factory(*args, **kwargs)
            client = result._client
            assert client._inner._retry_config.max_retries == 0
            assert client._inner._aclient is None and client._inner._client is None
            client._inner._transport = httpx.MockTransport(respond)
            clients.append(client)
            return result

        registry._model_facade_factory = facade
        return provider

    monkeypatch.setattr(designer, "_create_resource_provider", resources)
    models = [
        ModelConfig(alias=name, model=name, provider="scripted", skip_health_check=name == "detector")
        for name in ("detector", "singleton")
    ]
    selected = DetectionModelSelection(
        entity_detector="detector",
        entity_validator=["singleton"],
        entity_augmenter="singleton",
        latent_detector="singleton",
    )
    try:
        yield designer, models, selected, calls
    finally:

        async def close() -> None:
            for client in clients:
                await client.aclose()

        asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=10)


@pytest.mark.parametrize("preview", [False, True])
def test_real_detection_source_correspondence_and_redact(
    scripted_detection: tuple[DataDesigner, list[ModelConfig], DetectionModelSelection, Counter[str]], preview: bool
) -> None:
    designer, models, selected, calls = scripted_detection
    source = pd.DataFrame(
        {COL_TEXT: ["Alice travels.", "Alice stays."], "arbitrary_extra": [17, 29]}, index=pd.Index([4, 4])
    )
    detected = EntityDetectionWorkflow(NddAdapter(designer)).run(
        source,
        model_configs=models,
        selected_models=selected,
        gliner_detection_threshold=0.3,
        tag_latent_entities=False,
        preview_num_records=2 if preview else None,
    )
    replaced = ReplacementWorkflow().run(
        detected.dataframe,
        replace_method=Redact(),
        model_configs=models,
        selected_models=ReplaceModelSelection(replacement_generator="unused"),
    )
    assert detected.failed_records == replaced.failed_records == []
    assert replaced.dataframe[COL_REPLACED_TEXT].tolist() == [
        "[REDACTED_FIRST_NAME] travels.",
        "[REDACTED_FIRST_NAME] stays.",
    ]
    assert replaced.dataframe["arbitrary_extra"].tolist() == [17, 29]
    assert not any(str(name).startswith("_anonymizer_source_") for name in replaced.dataframe.columns)
    assert calls == {"detector": 2, "validator": 2, "augmenter": 2, "readiness": 1}


@pytest.mark.parametrize("preview", [False, True])
def test_public_source_frame(
    scripted_detection: tuple[DataDesigner, list[ModelConfig], DetectionModelSelection, Counter[str]],
    tmp_path: Path,
    preview: bool,
) -> None:
    designer, models, selected, calls = scripted_detection
    path = tmp_path / "source.csv"
    pd.DataFrame({"text": ["Alice travels.", "Alice stays.", "Alice leaves."], "extra": [17, 29, 43]}).to_csv(
        path, index=False
    )
    config_yaml = yaml.safe_dump(
        {
            "model_configs": [model.model_dump(mode="json", exclude_none=True) for model in models],
            "selected_models": {"detection": selected.model_dump()},
        }
    )
    anonymizer = Anonymizer(data_designer=designer, model_configs=config_yaml, artifact_path=tmp_path / "public")
    config = AnonymizerConfig(replace=Redact(), emit_telemetry=False)
    data = AnonymizerInput(source=str(path))
    result = (
        anonymizer.preview(config=config, data=data, num_records=2)
        if preview
        else anonymizer.run(config=config, data=data)
    )
    expected = ["[REDACTED_FIRST_NAME] travels.", "[REDACTED_FIRST_NAME] stays.", "[REDACTED_FIRST_NAME] leaves."][
        : 2 if preview else 3
    ]
    assert result.dataframe["text_replaced"].tolist() == expected
    assert result.trace_dataframe["extra"].tolist() == [17, 29, 43][: len(expected)]
    assert result.failed_records == []
    assert not any(str(name).startswith("_anonymizer_source_") for name in result.trace_dataframe.columns)
    assert calls == {**dict.fromkeys(("detector", "validator", "augmenter"), len(expected)), "readiness": 1}


def test_real_detection_dropped_row_preserves_peer_and_failure(
    scripted_detection: tuple[DataDesigner, list[ModelConfig], DetectionModelSelection, Counter[str]],
) -> None:
    designer, models, selected, calls = scripted_detection
    source = pd.DataFrame(
        {COL_TEXT: ["Alice travels.", "Alice fails."], RECORD_ID_COLUMN: ["keep", "drop"], "extra": [17, 29]}
    )
    result = EntityDetectionWorkflow(NddAdapter(designer)).run(
        source,
        model_configs=models,
        selected_models=selected,
        gliner_detection_threshold=0.3,
        tag_latent_entities=False,
        preview_num_records=2,
    )
    assert result.dataframe[COL_TEXT].tolist() == ["Alice travels."]
    assert result.dataframe["extra"].tolist() == [17]
    assert [(failure.record_id, failure.step, failure.reason) for failure in result.failed_records] == [
        ("drop", "entity-detection", "Record missing from workflow output")
    ]
    assert calls == {"detector": 2, "validator": 1, "augmenter": 1, "readiness": 1}
