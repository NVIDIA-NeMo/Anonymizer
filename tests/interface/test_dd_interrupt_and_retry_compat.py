# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transient provider errors during Substitute, compared with 0.4.0.

Only the HTTP transport is scripted; DataDesigner and the graph run for real. The expected rows, failed
records and HTTP call counts were recorded by running the same stubs against main (0.4.0 behavior): DataDesigner
retries a 5xx, 429 or timeout on the map request and keeps the record; after three failed tries of one task it
drops only that record. Runtime: under 4 s per case. Ctrl-C is covered in ``test_interrupt.py``.
"""

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
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop
from data_designer.interface.data_designer import DataDesigner

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Substitute

pytestmark = pytest.mark.compat

_NAMES = ("Zorba", "Zorbb", "Zorbc")


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


class _TransientMap:
    """Fails the first ``failures`` map requests for the first record with ``kind``; everything else succeeds."""

    def __init__(self, kind: str, failures: int) -> None:
        self.kind = kind
        self.failures = failures
        self.calls: Counter[str] = Counter()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model, messages = body["model"], json.dumps(body["messages"])
        if model == "detector":
            self.calls["detector"] += 1
            name = next(name for name in _NAMES if f"{name} waved." in messages)
            entity = {"text": name, "label": "first_name", "start": 0, "end": len(name), "score": 0.9}
            return _completion(model, json.dumps({"entities": [entity]}))
        if "Validate entity tags" in messages:
            self.calls["validator"] += 1
            ids = dict.fromkeys(f"first_name_0_{len(name)}" for name in _NAMES if name in messages)
            return _completion(model, _fenced({"decisions": [{"id": item, "decision": "keep"} for item in ids]}))
        if "Find untagged sensitive entities" in messages:
            self.calls["augmenter"] += 1
            return _completion(model, _fenced({"entities": []}))
        if "Generate synthetic replacements" not in messages:
            self.calls["health_check"] += 1
            return _completion(model, "Hello!")
        self.calls["map"] += 1
        if _NAMES[0] in messages and self.calls["map_error"] < self.failures:
            self.calls["map_error"] += 1
            if self.kind == "timeout":
                raise httpx.ReadTimeout("scripted timeout", request=request)
            # A short Retry-After keeps DataDesigner's rate-limit cooldown (2 s by default) fast for the test.
            headers = {"retry-after": "0.01"} if self.kind == "429" else {}
            return httpx.Response(int(self.kind), json={"error": {"message": "scripted"}}, headers=headers)
        replacements = [
            {"original": name, "label": "first_name", "synthetic": "Maya"} for name in _NAMES if name in messages
        ]
        return _completion(model, _fenced({"replacements": replacements}))


@pytest.fixture
def transient_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[DataDesigner, list[_TransientMap]]]:
    scripts: list[_TransientMap] = []
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
            result._client._inner._transport = httpx.MockTransport(lambda request: scripts[0](request))
            clients.append(result._client)
            return result

        provider.model_registry._model_facade_factory = facade
        return provider

    monkeypatch.setattr(designer, "_create_resource_provider", resources)
    try:
        yield designer, scripts
    finally:

        async def close() -> None:
            for client in clients:
                await client.aclose()

        asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=30)


@pytest.mark.parametrize(
    "kind,failures,kept,map_calls",
    [
        ("500", 1, 3, 4),  # one retry, the record is kept
        ("429", 1, 3, 4),
        ("timeout", 1, 3, 4),
        ("500", 3, 2, 5),  # three failed tries of one task: DataDesigner drops only that record
        ("timeout", 3, 2, 5),
        # 429 is never dropped: storms past the admitted leaves (6 parser attempts + 11 retries) keep the record.
        ("429", 17, 3, 20),
        ("429", 25, 3, 28),
        ("429", 60, 3, 63),
    ],
)
def test_substitute_map_retries_transient_errors_like_release(
    transient_map: tuple[DataDesigner, list[_TransientMap]],
    tmp_path: Path,
    kind: str,
    failures: int,
    kept: int,
    map_calls: int,
) -> None:
    designer, scripts = transient_map
    script = _TransientMap(kind, failures)
    scripts.append(script)
    models = [
        ModelConfig(alias=name, model=name, provider="scripted", skip_health_check=name == "detector")
        for name in ("detector", "llm")
    ]
    roles = {
        "entity_detector": "detector",
        "entity_validator": ["llm"],
        "entity_augmenter": "llm",
        "latent_detector": "llm",
    }
    configs = {
        "model_configs": [model.model_dump(mode="json", exclude_none=True) for model in models],
        "selected_models": {"detection": roles, "replace": {"replacement_generator": "llm"}},
    }
    source = tmp_path / "input.csv"
    pd.DataFrame({"text": [f"{name} waved." for name in _NAMES]}).to_csv(source, index=False)

    result = Anonymizer(data_designer=designer, model_configs=yaml.safe_dump(configs)).run(
        config=AnonymizerConfig(replace=Substitute()), data=AnonymizerInput(source=str(source), text_column="text")
    )

    assert result.dataframe["text_replaced"].tolist() == ["Maya waved."] * kept
    assert [(record.step, record.reason) for record in result.failed_records] == [
        ("replace-map-generation", "Record missing from workflow output")
    ] * (3 - kept)
    assert script.calls == {
        "health_check": 2,
        "detector": 3,
        "validator": 3,
        "augmenter": 3,
        "map": map_calls,
        "map_error": failures,
    }
    execution = getattr(result, "_execution", None)
    if execution is not None:
        # DataDesigner's own transport retries the map requests; the graph accounts the map as one opaque call.
        assert not execution.requests.dispatches
        assert execution._execution.context.prepared.declared_request_upper_bound is None
