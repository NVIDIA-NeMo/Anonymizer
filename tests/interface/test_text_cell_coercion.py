# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Null and non-string text cells through every route, with real DataDesigner.

The released routes never rejected such a cell: DataDesigner rendered it into
the prompts (`None` -> "None", NaN -> "nan", 31337 -> "31337"), replacement
wrote `str(text)`, the user's text column kept the original cell, and no row
failed. Only the graph's root datum admits exact `str`; it receives the same
rendering (`anonymizer.engine.source_text`).

The expected observations in `tests/fixtures/release_text_cells.json` were
recorded from the released code path (`main`), with this module's own scripted
transport, under each resolvable pandas major; regenerate them with
`PYTHONPATH=<main checkout>/src python tests/interface/test_text_cell_coercion.py`.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
import yaml
from data_designer.config import ModelConfig, ModelProvider
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop
from data_designer.interface import DataDesigner

from anonymizer import Annotate, Anonymizer, AnonymizerConfig, AnonymizerInput, Hash, Redact, Rewrite, Substitute
from anonymizer.config.models import RewriteModelSelection

pytestmark = pytest.mark.compat

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "release_text_cells.json"
PANDAS_MAJOR = pd.__version__.split(".")[0]
_SCHEMA = re.compile(r"<response_schema>\s*(\{.*?\})\s*</response_schema>", re.S)
_RENDERED = ("None", "nan", "<NA>", "31337")

# Each cell kind as a user supplies it: a file whose text column holds that cell.
CELLS: dict[str, tuple[pd.DataFrame, str]] = {
    "none": (pd.DataFrame({"t": pd.Series(["Alice travels.", None, "Alice stays."], dtype=object)}), "parquet"),
    "nan": (pd.DataFrame({"t": ["Alice travels.", "", "Alice stays."]}), "csv"),  # empty CSV field reads as NaN
    "pd_na": (
        pd.DataFrame({"t": pd.array(["Alice travels.", pd.NA, "Alice stays."], dtype="string[pyarrow]")}),
        "parquet",
    ),
    "int": (pd.DataFrame({"t": [31337, 42, 7]}), "csv"),
}
ROUTES: dict[str, AnonymizerConfig] = {
    "redact": AnonymizerConfig(replace=Redact(), emit_telemetry=False),
    "hash": AnonymizerConfig(replace=Hash(), emit_telemetry=False),
    "annotate": AnonymizerConfig(replace=Annotate(), emit_telemetry=False),
    "substitute": AnonymizerConfig(replace=Substitute(), emit_telemetry=False),
    "rewrite_staged": AnonymizerConfig(rewrite=Rewrite(use_combined_graph=False), emit_telemetry=False),
    "rewrite_combined": AnonymizerConfig(rewrite=Rewrite(use_combined_graph=True), emit_telemetry=False),
}
MODES = ("run", "preview")


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


def _name(text: str) -> str | None:
    """The scripted entity: "Alice" in a text row, or the integer cell 31337 rendered as text."""
    return "Alice" if "Alice" in text else "31337" if "31337" in text else None


def _content(model: str, text: str, prompts: list[str]) -> Any:
    if model == "detector":
        prompts.append(text)
        name = _name(text)
        return {"entities": [{"text": name, "label": "first_name", "start": 0, "end": 5, "score": 0.9}] if name else []}
    if "Validate entity tags" in text:
        return {"decisions": [{"id": "first_name_0_5", "decision": "keep"}] if "first_name_0_5" in text else []}
    if not _SCHEMA.findall(text):  # health check
        return {"ok": True}
    schema = json.loads(_SCHEMA.findall(text)[-1])
    content = _instance(schema, schema.get("$defs", {}))
    for key in ("entities", "latent_entities"):
        if key in content:
            content[key] = []
    if "rewritten_text" in content:
        content["rewritten_text"] = "Jane travels."
    name = _name(text)
    if "replacements" in content:
        content["replacements"] = [{"original": name, "label": "first_name", "synthetic": "Jane"}] if name else []
    for item in content.get("sensitivity_disposition", []):
        item.update(entity_label="first_name", entity_value=name or "Alice")
    return content


def _anonymizer(tmp: Path, prompts: list[str], clients: list[Any], patch: pytest.MonkeyPatch) -> Anonymizer:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = "\n".join(
            message["content"]
            if isinstance(message["content"], str)
            else "\n".join(part.get("text", "") for part in message["content"])
            for message in body["messages"]
        )
        content = json.dumps(_content(body["model"], text, prompts))
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
        artifact_path=tmp / "artifacts",
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

    patch.setattr(designer, "_create_resource_provider", resources)
    models = [
        ModelConfig(alias=name, model=name, provider="scripted", skip_health_check=name == "detector")
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
    return Anonymizer(data_designer=designer, model_configs=yaml.safe_dump(model_configs))


def _cell(value: Any) -> Any:
    """Encode a cell with its exact type, so `None`, NaN and `pd.NA` stay distinct."""
    if value is None:
        return "<None>"
    if value is pd.NA:
        return "<pd.NA>"
    if isinstance(value, float) and value != value:
        return "<nan>"
    if isinstance(value, np.generic):
        return [type(value).__name__, value.item()]
    if isinstance(value, (str, bool, int, float)):
        return [type(value).__name__, value]
    if isinstance(value, (list, tuple)):
        return [_cell(item) for item in value]
    if isinstance(value, np.ndarray):
        return ["ndarray", [_cell(item) for item in value.tolist()]]
    if isinstance(value, dict):
        return {str(key): _cell(item) for key, item in value.items()}
    return [type(value).__name__, repr(value)]


def observe(tmp: Path, cell: str, route: str, mode: str) -> dict[str, Any]:
    """Run one route over one cell kind and record the public result."""
    frame, suffix = CELLS[cell]
    path = tmp / f"input.{suffix}"
    if suffix == "parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    prompts: list[str] = []
    clients: list[Any] = []
    config, data = ROUTES[route], AnonymizerInput(source=str(path), text_column="t")
    try:
        with pytest.MonkeyPatch.context() as patch:
            anonymizer = _anonymizer(tmp, prompts, clients, patch)
            if mode == "preview":
                result = anonymizer.preview(config=config, data=data, num_records=len(frame))
            else:
                result = anonymizer.run(config=config, data=data)
    finally:

        async def close() -> None:
            for client in clients:
                await client.aclose()

        asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=30)
    output = result.dataframe
    return {
        "failed": len(result.failed_records),
        "dtypes": {column: str(dtype) for column, dtype in output.dtypes.items()},
        "values": {column: [_cell(value) for value in output[column].tolist()] for column in output.columns},
        "detector_saw": sorted({token for prompt in prompts for token in _RENDERED if token in prompt}),
    }


@pytest.fixture(scope="module")
def released() -> Iterator[dict[str, Any]]:
    recorded = json.loads(FIXTURE.read_text())
    if PANDAS_MAJOR not in recorded:
        pytest.skip(f"no released observations recorded for pandas {pd.__version__}")
    yield recorded[PANDAS_MAJOR]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize("cell", list(CELLS))
def test_text_cell_is_coerced_like_release(released, tmp_path, cell, route, mode):
    expected = released[f"{cell}|{route}|{mode}"]

    observed = observe(tmp_path, cell, route, mode)

    # No row fails, the user's column keeps its original cells, and every output column (text,
    # spans, entities, rewrite metrics, review flag) has the released values and dtypes.
    assert observed == expected
    if cell != "int":
        assert observed["values"]["t"][1] in ("<None>", "<nan>", "<pd.NA>")


def _record(out: Path) -> None:
    """Record this interpreter's observations into the fixture, keyed by pandas major."""
    import tempfile

    recorded = json.loads(out.read_text()) if out.exists() else {}
    observations = {}
    for cell in CELLS:
        for route in ROUTES:
            for mode in MODES:
                with tempfile.TemporaryDirectory() as tmp:
                    observations[f"{cell}|{route}|{mode}"] = observe(Path(tmp), cell, route, mode)
    recorded[PANDAS_MAJOR] = observations
    lines = [
        f"{json.dumps(major)}: {{\n"
        + ",\n".join(f"{json.dumps(key)}: {json.dumps(value, sort_keys=True)}" for key, value in sorted(items.items()))
        + "\n}"
        for major, items in sorted(recorded.items())
    ]
    out.write_text("{\n" + ",\n".join(lines) + "\n}\n")


if __name__ == "__main__":
    _record(Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE)
