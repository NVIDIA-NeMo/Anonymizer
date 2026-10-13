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
from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_TEXT, _jinja
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


@pytest.mark.parametrize("unattributed_loss", [False, True])
def test_real_dd_whole_frame_uses_invocation_readiness_and_source_requests(
    scripted_detection, monkeypatch, unattributed_loss
):
    """One native DD column consumes the new owner; full detection remains unqualified."""
    from contextvars import ContextVar
    from dataclasses import replace

    from data_designer.config import LLMTextColumnConfig

    from anonymizer.engine.graph_sdk.blocking import BlockingPool
    from anonymizer.engine.graph_sdk.executor import LocalCompleted, LocalFailure
    from anonymizer.engine.graph_sdk.requests import (
        AssociationResult,
        ExactUsage,
        ExternalSettlement,
        PortArtifact,
        StopUnknown,
        TextArtifactValue,
        TransportSuccess,
    )
    from anonymizer.engine.ndd._source_frame import current_source_work
    from tests.graph_sdk.delegated_execution_fixtures import with_multiple_sources
    from tests.graph_sdk.test_delegated_execution import _execute
    from tests.graph_sdk.test_delegated_readiness import _ready_fixture
    from tests.graph_sdk.test_foreign_resources import fixture as foreign_fixture

    designer, models, _, calls = scripted_detection
    owner_context = ContextVar("dd_owner")
    exchange = ContextVar("dd_exchange")
    seen = []
    original_resources = designer._create_resource_provider
    base, services, original, _, _, _ = _ready_fixture()
    execution = with_multiple_sources(base)

    class Forward:
        async def dispatch(self, envelope):
            transport, request, returned = exchange.get()
            returned.append(await transport.handle_async_request(request))
            seen.append((current_source_work(), envelope.associations[0].association))
            return TransportSuccess(
                results=(
                    AssociationResult(
                        association=envelope.associations[0].association,
                        outcome="ready" if current_source_work() is None else "ok",
                        outputs=(),
                        consumed_context_ports=frozenset(),
                    ),
                ),
                settlement=ExternalSettlement(
                    request=envelope.request,
                    disposition="completed",
                    remote_stopped=True,
                    usage=ExactUsage(input_units=1, output_units=1),
                ),
            )

        async def cancel(self, request):
            return StopUnknown()

    class Gate(httpx.AsyncBaseTransport):
        def __init__(self, inner, requests):
            self.inner, self.requests = inner, requests

        async def handle_async_request(self, request):
            returned = []
            token = exchange.set((self.inner, request, returned))
            try:
                work = current_source_work()
                if work is None:
                    await self.requests.readiness()
                    await self.requests.probe()
                else:
                    await self.requests.request(original.slots[0], association=work.association)
                assert len(returned) == 1
                return returned[0]
            finally:
                exchange.reset(token)

    def resources(*args, **kwargs):
        provider = original_resources(*args, **kwargs)
        factory = provider.model_registry._model_facade_factory

        def facade(*args, **kwargs):
            result = factory(*args, **kwargs)
            result._client._inner._transport = Gate(result._client._inner._transport, owner_context.get())
            return result

        provider.model_registry._model_facade_factory = facade
        return provider

    monkeypatch.setattr(designer, "_create_resource_provider", resources)
    if unattributed_loss:
        preview = designer.preview

        def lose_output(*args, **kwargs):
            returned = preview(*args, **kwargs)
            returned.dataset = returned.dataset.iloc[:1]
            return returned

        monkeypatch.setattr(designer, "preview", lose_output)

    class Submit:
        execution = "same_loop"

        async def submit(self, inputs, requests):
            source = pd.DataFrame({COL_TEXT: [item.inputs[0].value.text for item in inputs], "extra": [17, 29]})
            frame = requests.bind_frame(source)
            token = owner_context.set(requests)
            try:
                result = await asyncio.to_thread(
                    NddAdapter(designer).run_workflow,
                    source,
                    source_frame=frame,
                    model_configs=[models[0].model_copy(update={"skip_health_check": False})],
                    columns=[LLMTextColumnConfig(name="detected", model_alias="detector", prompt=_jinja(COL_TEXT))],
                    workflow_name="invocation-source-witness",
                    preview_num_records=2,
                )
            except Exception as error:
                from anonymizer.interface.errors import AnonymizerWorkflowError

                assert unattributed_loss and isinstance(error, AnonymizerWorkflowError)
                assert "unattributed source loss" in str(error)
                return LocalFailure(failure="permanent")
            finally:
                owner_context.reset(token)
            assert not unattributed_loss
            assert result.source_positions == (0, 1) and result.dataframe["extra"].tolist() == [17, 29]
            assert not result.failed_records
            return LocalCompleted(
                results=tuple(
                    AssociationResult(
                        association=item.association,
                        outcome="ok",
                        consumed_context_ports=frozenset(),
                        outputs=(
                            PortArtifact(
                                port="context",
                                artifact_type=item.inputs[0].artifact_type,
                                artifact=None,
                                value=TextArtifactValue(text=result.dataframe["detected"].iloc[position]),
                            ),
                        ),
                    )
                    for position, item in enumerate(inputs)
                )
            )

    services = replace(
        services,
        handles=tuple(
            replace(handle, delegated=Submit())
            if handle.delegated
            else replace(handle, transport=Forward())
            if handle.transport
            else handle
            for handle in services.handles
        ),
        limits=replace(services.limits, max_runtime_artifacts=16, max_runtime_artifact_bytes=10000),
    )

    async def run():
        pool = BlockingPool(max_workers=1)
        try:
            admitted, supplied, _, clients, worker, _ = await foreign_fixture(
                ensure_async_engine_loop(), pool, base=(execution, services, original)
            )
            result = await _execute(admitted, supplied)
            assert worker.local_users == pool.occupied == 0
            assert all(client.closes == 1 for client in clients)
            return result
        finally:
            await pool.close()

    result = asyncio.run(run())
    assert result.requests.dispatched_count == 3 and calls == {"detector": 3}
    assert len(result.final_outputs) == (0 if unattributed_loss else 2) and result.invocation_state.complete
    assert sum(work is None for work, _ in seen) == 1
    assert len({work.association for work, _ in seen if work is not None}) == 2
    assert all(work is None or work.association == association for work, association in seen)
