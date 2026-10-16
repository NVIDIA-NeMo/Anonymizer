# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Released Rewrite output dtypes through real DataDesigner with scripted HTTP responses.

The released Rewrite route returned DataDesigner's frames as-is: `run` reads them
back from parquet as Arrow-backed columns, while `preview` keeps DataDesigner's
in-memory NumPy/object columns. The expected tables below were taken from the
released code path (`main`) under this same scripted transport.
"""

from __future__ import annotations

import asyncio
import json
import re
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

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.models import RewriteModelSelection

pytestmark = pytest.mark.compat

_SCHEMA = re.compile(r"<response_schema>\s*(\{.*?\})\s*</response_schema>", re.S)
_ARROW_RUN = {
    "biography": "string[pyarrow]",
    "biography_rewritten": "string[pyarrow]",
    "utility_score": "double[pyarrow]",
    "leakage_mass": "double[pyarrow]",
    "weighted_leakage_rate": "double[pyarrow]",
    "any_high_leaked": "bool[pyarrow]",
    "needs_human_review": "bool",
}
_PREVIEW = {
    "biography": "object",
    "biography_rewritten": "object",
    "utility_score": "float64",
    "leakage_mass": "float64",
    "weighted_leakage_rate": "float64",
    "any_high_leaked": "bool",
    "needs_human_review": "bool",
}
RELEASED_DTYPES = {
    (False, False): _ARROW_RUN,
    (True, False): {**_ARROW_RUN, "needs_human_review": "bool[pyarrow]"},
    (False, True): _PREVIEW,
    (True, True): _PREVIEW,
}


def _instance(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    """Build a minimal value satisfying one generated JSON schema."""
    if "$ref" in schema:
        return _instance(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:
        return _instance(next(option for option in schema["anyOf"] if option.get("type") != "null"), defs)
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {key: _instance(value, defs) for key, value in schema.get("properties", {}).items()}
    if kind == "array":
        return [_instance(schema.get("items", {}), defs) for _ in range(max(1, schema.get("minItems", 0)))]
    if kind == "string":
        return "scripted text " * max(1, schema.get("minLength", 1))
    return {"number": 0.5, "integer": 1, "boolean": False}.get(kind)


def _content(model: str, text: str) -> dict[str, Any]:
    if model == "detector":
        return {"entities": [{"text": "Alice", "label": "first_name", "start": 0, "end": 5, "score": 0.9}]}
    if "Validate entity tags" in text:
        return {"decisions": [{"id": "first_name_0_5", "decision": "keep"}]}
    schema = json.loads(_SCHEMA.findall(text)[-1])
    content = _instance(schema, schema.get("$defs", {}))
    for key in ("entities", "latent_entities"):
        if key in content:
            content[key] = []
    if "rewritten_text" in content:
        content["rewritten_text"] = "Jane travels."
    if "replacements" in content:
        # The second record gets no replacement, so its rewrite is withheld as null.
        withheld = "Alice</first_name> stays" in text
        content["replacements"] = (
            [] if withheld else [{"original": "Alice", "label": "first_name", "synthetic": "Jane"}]
        )
    for item in content.get("sensitivity_disposition", []):
        item.update(entity_label="first_name", entity_value="Alice")
    return content


@pytest.fixture
def scripted_rewrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Anonymizer]:
    clients: list[Any] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = "\n".join(
            message["content"]
            if isinstance(message["content"], str)
            else "\n".join(part.get("text", "") for part in message["content"])
            for message in body["messages"]
        )
        content = json.dumps(_content(body["model"], text))
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

    monkeypatch.setattr(designer, "_create_resource_provider", resources)
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
    }
    model_configs = {
        "model_configs": [model.model_dump(mode="json", exclude_none=True) for model in models],
        "selected_models": selected,
    }
    try:
        yield Anonymizer(data_designer=designer, model_configs=yaml.safe_dump(model_configs))
    finally:

        async def close() -> None:
            for client in clients:
                await client.aclose()

        asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=10)


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_rewrite_output_dtypes_match_release(scripted_rewrite, tmp_path, combined, preview):
    path = tmp_path / "input.csv"
    pd.DataFrame({"biography": ["Alice travels.", "Alice stays."]}).to_csv(path, index=False)
    options = dict(
        config=AnonymizerConfig(rewrite=Rewrite(use_combined_graph=combined), emit_telemetry=False),
        data=AnonymizerInput(source=str(path), text_column="biography"),
    )
    result = scripted_rewrite.preview(**options, num_records=2) if preview else scripted_rewrite.run(**options)

    frame = result.dataframe
    assert {column: str(dtype) for column, dtype in frame.dtypes.items()} == RELEASED_DTYPES[(combined, preview)]
    assert result.failed_records == []
    assert frame["biography_rewritten"].iloc[0] == "Jane travels."
    # A withheld rewrite stays null and flagged; Arrow-backed run output uses pd.NA like the release.
    assert frame["biography_rewritten"].iloc[1] is (None if preview else pd.NA)
    assert frame["needs_human_review"].tolist() == [False, True]
    for column in frame.columns:
        assert result.trace_dataframe[column].dtype == frame[column].dtype
    assert isinstance(result.trace_dataframe["_domain"].dtype, pd.ArrowDtype) is not preview
