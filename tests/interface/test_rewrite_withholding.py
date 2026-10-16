# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Withheld Rewrite records through the evaluate-repair loop, with real DataDesigner.

A rewrite is withheld when its replacement map cannot be applied
(`_prepare_rewrite_tagged_text` leaves the baseline unset and
`_extract_rewritten_text` yields no text). The released staged loop
(`RewriteWorkflow._run_evaluate_repair_loop`) excludes such rows from repair and
nulls their rewritten text before and after the loop, and its regression
`test_replacement_unavailable_remains_failed_closed_through_repair` pins that
the record stays null and flagged for human review. Repair starts from the
rewritten text's sensitive source, so publishing a repair of a withheld record
could reproduce the original entity.

Each scenario runs this module as a script in a subprocess: an all-withheld
batch used to abort the interpreter inside pyarrow, and a regression must fail
the test instead of killing pytest.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.compat

_SCHEMA = re.compile(r"<response_schema>\s*(\{.*?\})\s*</response_schema>", re.S)
_REPAIRED = "Jane journeys."


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
        return "x " * max(1, schema.get("minLength", 1))
    return {"number": 0.37, "integer": 2, "boolean": True}.get(kind)


def _leak(node: Any, leaked: bool) -> None:
    """Script evaluation answers: the first rewrite fails evaluation, and its repair is scored again."""
    if isinstance(node, dict):
        if "evidence" in node:
            node["evidence"] = ["x"] if leaked else []
            if "confidence" in node:
                node["confidence"] = 0.5 if leaked else 1
            if "answer" in node:
                node["answer"] = "no" if leaked else "yes"
        if "score" in node:
            node["score"] = 0.5 if leaked else 1
        for value in node.values():
            _leak(value, leaked)
    elif isinstance(node, list):
        for value in node:
            _leak(value, leaked)


def _content(model: str, text: str, scenario: str) -> dict[str, Any]:
    if model == "detector":
        name = "Alice" if "Alice" in text else "Bob" if "Bob" in text else None
        entities = [{"text": name, "label": "first_name", "start": 0, "end": len(name), "score": 0.9}] if name else []
        return {"entities": entities}
    if "Validate entity tags" in text:
        ids = re.findall(r"first_name_0_\d", text)
        return {"decisions": [{"id": tag, "decision": "keep"} for tag in dict.fromkeys(ids)]}
    schema = json.loads(_SCHEMA.findall(text)[-1])
    content = _instance(schema, schema.get("$defs", {}))
    for key in ("entities", "latent_entities"):
        if key in content:
            content[key] = []
    # "allwithheld" leaves the scripted answers neutral so no row needs repair: the withheld
    # column then reaches the loop's end as an all-null batch. The other scenarios drive repair.
    pressure = scenario != "allwithheld"
    repaired = _REPAIRED in text
    if "rewritten_text" in content:
        content["rewritten_text"] = _REPAIRED if pressure and ("Jane travels" in text or repaired) else "Jane travels."
    if pressure:
        _leak(content, repaired)
    if "replacements" in content:
        # Bob (and, in the all-withheld scenario, everyone) gets no replacement: the rewrite is withheld.
        bob = "Bob</first_name>" in text
        withheld = scenario.startswith("allwithheld") or bob
        content["replacements"] = (
            [] if withheld else [{"original": "Alice", "label": "first_name", "synthetic": "Jane"}]
        )
    for item in content.get("sensitivity_disposition", []):
        item.update(entity_label="first_name", entity_value="Alice")
        for key in ("sensitivity", "combined_risk_level"):
            if key in item and pressure:
                item[key] = "high"
    return content


def _main(mode: str, combined: bool, scenario: str, out: Path) -> None:
    import httpx
    import pandas as pd
    import yaml
    from data_designer.config import ModelConfig, ModelProvider
    from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop
    from data_designer.interface import DataDesigner

    from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite
    from anonymizer.config.models import RewriteModelSelection

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = "\n".join(
            message["content"]
            if isinstance(message["content"], str)
            else "\n".join(part.get("text", "") for part in message["content"])
            for message in body["messages"]
        )
        content = json.dumps(_content(body["model"], text, scenario))
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

    tmp = Path(tempfile.mkdtemp())
    clients: list[Any] = []
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

    setattr(designer, "_create_resource_provider", resources)
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
    anonymizer = Anonymizer(data_designer=designer, model_configs=yaml.safe_dump(model_configs))
    records = (
        ["Alice travels.", "Bob stays.", "Nothing here."] if scenario == "mixed" else ["Alice travels.", "Alice stays."]
    )
    path = tmp / "input.csv"
    pd.DataFrame({"biography": records}).to_csv(path, index=False)
    config = AnonymizerConfig(rewrite=Rewrite(use_combined_graph=combined), emit_telemetry=False)
    data = AnonymizerInput(source=str(path), text_column="biography")
    if mode == "preview":
        result = anonymizer.preview(config=config, data=data, num_records=len(records))
    else:
        result = anonymizer.run(config=config, data=data)

    def cell(value: Any) -> Any:
        if value is pd.NA:
            return "<NA>"
        return value.item() if hasattr(value, "item") else value

    frame, trace = result.dataframe, result.trace_dataframe
    out.write_text(
        json.dumps(
            {
                "rewritten": [cell(value) for value in frame["biography_rewritten"]],
                "rewritten_dtype": str(frame["biography_rewritten"].dtype),
                "needs_human_review": [bool(value) for value in frame["needs_human_review"]],
                "repair_iterations": [int(value) for value in trace["_repair_iterations"]],
                "failed": len(result.failed_records),
            }
        )
    )

    async def close() -> None:
        for client in clients:
            await client.aclose()

    asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=10)


def _scenario(tmp_path: Path, mode: str, combined: bool, scenario: str) -> dict[str, Any]:
    out = tmp_path / "result.json"
    completed = subprocess.run(
        [sys.executable, __file__, mode, str(int(combined)), scenario, str(out)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, f"rc={completed.returncode}\n{completed.stderr[-4000:]}"
    return json.loads(out.read_text())


def _null(preview: bool) -> Any:
    # Run output is Arrow-backed (pd.NA); preview keeps DataDesigner's object column (None), as released.
    return None if preview else "<NA>"


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_staged_withheld_rewrite_stays_null_through_repair(tmp_path: Path, preview: bool) -> None:
    result = _scenario(tmp_path, "preview" if preview else "run", False, "mixed")

    # Released staged loop: Alice is repaired, Bob (withheld) is never sent to repair and stays null.
    assert result["rewritten"] == [_REPAIRED, _null(preview), "Nothing here."]
    assert result["needs_human_review"] == [True, True, False]
    assert result["repair_iterations"][1] == 0
    assert result["failed"] == 0


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_combined_withheld_rewrite_stays_null_through_repair(tmp_path: Path, preview: bool) -> None:
    result = _scenario(tmp_path, "preview" if preview else "run", True, "mixed")

    # Intentional delta from the released combined graph, which publishes the repair of a withheld
    # record. That graph (#240) predates the fail-closed withholding rule (#246) and never gated its
    # repair skip condition on replacement readiness; the staged loop, the withholding rule, and its
    # regression test all require the record to stay null. Both schedules now fail closed.
    assert result["rewritten"] == [_REPAIRED, _null(preview), "Nothing here."]
    assert result["needs_human_review"] == [True, True, False]
    assert result["failed"] == 0


@pytest.mark.parametrize("scenario", ["allwithheld", "allwithheld-repair"], ids=["no-repair", "repair"])
@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_all_withheld_batch_completes_with_null_rewrites(
    tmp_path: Path, combined: bool, preview: bool, scenario: str
) -> None:
    result = _scenario(tmp_path, "preview" if preview else "run", combined, scenario)

    assert result["rewritten"] == [_null(preview), _null(preview)]
    # Same dtype as a partially withheld batch in the release.
    assert result["rewritten_dtype"] == ("object" if preview else "string[pyarrow]")
    assert result["needs_human_review"] == [True, True]
    assert result["failed"] == 0


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2] == "1", sys.argv[3], Path(sys.argv[4]))
