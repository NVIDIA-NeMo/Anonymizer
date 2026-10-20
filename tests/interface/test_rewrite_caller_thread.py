# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rewrite body runs on the caller's thread with a real `NddAdapter`, as in the release.

The released facade ran the Rewrite runner, its `build_graph`, injected detectors and every
DataDesigner call on the thread that called `run()`/`preview()`. Thread-locals and main-thread-only
APIs such as `signal.signal` depend on that. These cases use a real DataDesigner whose model
clients answer from a scripted transport, so the adapter is not opaque.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.interface.test_rewrite_withholding import _content

pytestmark = pytest.mark.compat

_PROBE: contextvars.ContextVar[str] = contextvars.ContextVar("caller_thread_probe", default="unset")


def _designer(tmp_path: Path) -> tuple[Any, list[Any]]:
    import httpx
    from data_designer.config import ModelProvider
    from data_designer.interface import DataDesigner

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = "\n".join(
            message["content"]
            if isinstance(message["content"], str)
            else "\n".join(part.get("text", "") for part in message["content"])
            for message in body["messages"]
        )
        content = json.dumps(_content(body["model"], text, "mixed"))
        return httpx.Response(
            200,
            json={
                "id": "scripted",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content if body["model"] == "detector" else f"```json\n{content}\n```",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    clients: list[Any] = []
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
            result._client._inner._transport = httpx.MockTransport(respond)
            clients.append(result._client)
            return result

        registry._model_facade_factory = facade
        return provider

    setattr(designer, "_create_resource_provider", resources)
    return designer, clients


def _model_configs() -> str:
    import yaml
    from data_designer.config import ModelConfig

    from anonymizer.config.models import EvaluateModelSelection, RewriteModelSelection

    models = [
        ModelConfig(alias=name, model=name, provider="scripted", skip_health_check=True)
        for name in ("detector", "singleton")
    ]
    selected = {
        "detection": {
            "entity_detector": "detector",
            "entity_validator": ["singleton"],
            "entity_augmenter": "singleton",
            "latent_detector": "singleton",
        },
        "replace": {"replacement_generator": "singleton"},
        "rewrite": {key: "singleton" for key in RewriteModelSelection.model_fields},
        "evaluate": dict.fromkeys(EvaluateModelSelection.model_fields, "singleton"),
    }
    return yaml.safe_dump(
        {
            "model_configs": [model.model_dump(mode="json", exclude_none=True) for model in models],
            "selected_models": selected,
        }
    )


def _close(clients: list[Any]) -> None:
    from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

    async def close() -> None:
        for client in clients:
            await client.aclose()

    asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=30)


def _run(tmp_path: Path, *, combined: bool, preview: bool, detector: str) -> tuple[int, list[tuple[str, int, str]]]:
    from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite
    from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
    from anonymizer.engine.ndd.adapter import NddAdapter
    from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow

    seen: list[tuple[str, int, str]] = []

    def probe(label: str) -> None:
        seen.append((label, threading.get_ident(), _PROBE.get()))

    designer, clients = _designer(tmp_path)

    def wrap(name: str) -> None:
        original = getattr(designer, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            probe(f"data_designer.{name}")
            return original(*args, **kwargs)

        setattr(designer, name, wrapped)

    wrap("create")
    wrap("preview")

    class GraphOverride(CombinedRewriteWorkflow):
        def build_graph(self, *args: Any, **kwargs: Any) -> Any:
            probe("build_graph")
            return super().build_graph(*args, **kwargs)

    class StageOverride(EntityDetectionWorkflow):
        def detect_and_validate_entities(self, *args: Any, **kwargs: Any) -> Any:
            probe("detect_and_validate_entities")
            return super().detect_and_validate_entities(*args, **kwargs)

        def identify_latent_entities(self, *args: Any, **kwargs: Any) -> Any:
            probe("identify_latent_entities")
            return super().identify_latent_entities(*args, **kwargs)

    class ProbedAdapter(NddAdapter):
        def run_workflow(self, *args: Any, **kwargs: Any) -> Any:
            probe("detection_adapter.run_workflow")
            return super().run_workflow(*args, **kwargs)

    extra: dict[str, Any] = {}
    if combined:
        extra["combined_rewrite_runner"] = GraphOverride(NddAdapter(designer))
    if detector == "stages":
        extra["detection_workflow"] = StageOverride(NddAdapter(designer))
    elif detector == "adapter":
        extra["detection_workflow"] = EntityDetectionWorkflow(ProbedAdapter(designer))
    anonymizer = Anonymizer(data_designer=designer, model_configs=_model_configs(), **extra)
    path = tmp_path / "input.csv"
    records = ["Alice travels.", "Alice stays."]
    pd.DataFrame({"biography": records}).to_csv(path, index=False)
    config = AnonymizerConfig(rewrite=Rewrite(use_combined_graph=combined), emit_telemetry=False)
    data = AnonymizerInput(source=str(path), text_column="biography")
    token = _PROBE.set("caller")
    try:
        if preview:
            result = anonymizer.preview(config=config, data=data, num_records=len(records))
        else:
            result = anonymizer.run(config=config, data=data)
    finally:
        _PROBE.reset(token)
        _close(clients)
    assert len(result.dataframe) == len(records)
    return threading.get_ident(), seen


@pytest.mark.parametrize("detector", ["builtin", "stages", "adapter"])
@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_native_rewrite_body_runs_on_caller_thread(
    tmp_path: Path, combined: bool, preview: bool, detector: str
) -> None:
    caller, seen = _run(tmp_path, combined=combined, preview=preview, detector=detector)

    labels = {label for label, _, _ in seen}
    assert ("build_graph" in labels) == combined
    assert {"data_designer.preview" if preview else "data_designer.create"} <= labels
    if detector == "stages":
        assert {"detect_and_validate_entities", "identify_latent_entities"} <= labels
    if detector == "adapter":
        assert "detection_adapter.run_workflow" in labels
    assert [(label, ident) for label, ident, _ in seen if ident != caller] == []
    assert {value for _, _, value in seen} == {"caller"}


def test_native_rewrite_runs_from_inside_a_running_event_loop(tmp_path: Path) -> None:
    async def inside() -> tuple[int, list[tuple[str, int, str]]]:
        return _run(tmp_path, combined=True, preview=False, detector="builtin")

    caller, seen = asyncio.run(inside())

    assert seen
    assert all(ident == caller for _, ident, _ in seen)


def test_nested_handoff_from_the_serving_thread_fails_instead_of_deadlocking() -> None:
    from anonymizer.engine.graph_sdk._execution_values import LocalFailure
    from anonymizer.engine.harness.kernel import _run_sync_profile

    async def operation(pool: Any, handoff: Any) -> Any:
        def body() -> Any:
            return handoff(lambda: LocalFailure(failure="implementation_exception"))

        return await asyncio.to_thread(handoff, body)

    with pytest.raises(RuntimeError, match="nested caller-thread handoff"):
        _run_sync_profile(operation, caller_callbacks=True)
