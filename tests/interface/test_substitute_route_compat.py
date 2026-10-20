# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public Substitute routes through a real DataDesigner and scripted HTTP, compared with 0.4.0.

Only the HTTP transport is scripted: ``NddAdapter``, DataDesigner, its health checks and the replacement-map
generation all run for real. The expectations are 0.4.0 behavior and use only public and 0.4.0-era names, so
this module runs unchanged against the release source.
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

from anonymizer import Annotate, Anonymizer, AnonymizerConfig, AnonymizerInput, Hash, Redact, Rewrite, Substitute
from anonymizer.config.models import RewriteModelSelection
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_TAGGED_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.interface.errors import AnonymizerWorkflowError

pytestmark = pytest.mark.compat

_TEXTS = ["Alice waved.", "Bob stayed."]
_PLAIN = ["the sky is blue.", "water is wet."]
_SYNTHETIC = {"Alice": "Maya", "Bob": "Theo"}
_REPLACED = [
    {"text": "Alice waved.", "text_replaced": "Maya waved."},
    {"text": "Bob stayed.", "text_replaced": "Theo stayed."},
]


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


def _respond(calls: Counter[str], request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    model, messages = body["model"], json.dumps(body["messages"])
    if model == "detector":
        calls["detector"] += 1
        text = next(text for text in (*_TEXTS, *_PLAIN) if text in messages)
        entities = [
            {"text": word, "label": "first_name", "start": text.index(word), "end": text.index(word) + len(word)}
            for word in _SYNTHETIC
            if word in text
        ]
        return _completion(model, json.dumps({"entities": [{**item, "score": 0.9} for item in entities]}))
    if "Validate entity tags" in messages:
        calls["validator"] += 1
        ids = [f"first_name_{t.index(w)}_{t.index(w) + len(w)}" for t in _TEXTS for w in _SYNTHETIC if w in t]
        return _completion(model, _fenced({"decisions": [{"id": i, "decision": "keep"} for i in ids if i in messages]}))
    if "Find untagged sensitive entities" in messages:
        calls["augmenter"] += 1
        return _completion(model, _fenced({"entities": []}))
    if "Generate synthetic replacements" in messages:
        calls["map"] += 1
        replacements = [
            {"original": word, "label": "first_name", "synthetic": synthetic}
            for word, synthetic in _SYNTHETIC.items()
            if word in messages
        ]
        return _completion(model, _fenced({"replacements": replacements}))
    assert "Hello!" in messages
    calls["health_check"] += 1
    return _completion(model, "Hello!")


@pytest.fixture
def scripted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[DataDesigner, Counter[str]]]:
    calls: Counter[str] = Counter()
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
            result._client._inner._transport = httpx.MockTransport(lambda request: _respond(calls, request))
            clients.append(result._client)
            return result

        provider.model_registry._model_facade_factory = facade
        return provider

    monkeypatch.setattr(designer, "_create_resource_provider", resources)
    try:
        yield designer, calls
    finally:

        async def close() -> None:
            for client in clients:
                await client.aclose()

        asyncio.run_coroutine_threadsafe(close(), ensure_async_engine_loop()).result(timeout=30)


def _anonymizer(designer: DataDesigner, *, skip_health_check: bool = False, **kwargs: Any) -> Anonymizer:
    models = [
        ModelConfig(
            alias=name, model=name, provider="scripted", skip_health_check=name == "detector" or skip_health_check
        )
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
        "rewrite": dict.fromkeys(RewriteModelSelection.model_fields, "singleton"),
    }
    configs = {
        "model_configs": [m.model_dump(mode="json", exclude_none=True) for m in models],
        "selected_models": selected,
    }
    return Anonymizer(data_designer=designer, model_configs=yaml.safe_dump(configs), **kwargs)


def _frame(
    anonymizer: Anonymizer,
    mode: str,
    tmp_path: Path,
    texts: list[str] = _TEXTS,
    replace: Substitute | Redact | Hash | Annotate | None = None,
) -> pd.DataFrame:
    source = tmp_path / "input.csv"
    pd.DataFrame({"text": texts}).to_csv(source, index=False)
    data = AnonymizerInput(source=str(source), text_column="text")
    config = AnonymizerConfig(replace=replace or Substitute())
    result = (
        anonymizer.run(config=config, data=data)
        if mode == "run"
        else anonymizer.preview(config=config, data=data, num_records=len(texts))
    )
    assert not result.failed_records
    return result.dataframe[["text", "text_replaced"]]


def _execute(anonymizer: Anonymizer, mode: str, tmp_path: Path, **kwargs: Any) -> list[dict[str, str]]:
    return _frame(anonymizer, mode, tmp_path, **kwargs).to_dict("records")


def _custom_frame(dataframe: pd.DataFrame, kwargs: dict[str, Any]) -> pd.DataFrame:
    limit = kwargs.get("preview_num_records")
    output = (dataframe if limit is None else dataframe.head(limit)).copy()
    final, by_value = [], []
    for text in output[COL_TEXT]:
        entities = [
            {
                "id": f"first_name_{text.index(word)}_{text.index(word) + len(word)}",
                "value": word,
                "label": "first_name",
                "start_position": text.index(word),
                "end_position": text.index(word) + len(word),
                "score": 1.0,
                "source": "custom",
            }
            for word in _SYNTHETIC
            if word in text
        ]
        final.append({"entities": entities})
        by_value.append(
            {"entities_by_value": [{"value": item["value"], "labels": ["first_name"]} for item in entities]}
        )
    output[COL_FINAL_ENTITIES] = final
    output[COL_ENTITIES_BY_VALUE] = by_value
    output[COL_TAGGED_TEXT] = output[COL_TEXT]
    return output


class _DuckDetector:
    """A caller-supplied detector that is not an EntityDetectionWorkflow; returns a 0.4.0-shaped result."""

    def __init__(self, rows: list[int] | None = None) -> None:
        self.calls = 0
        self.rows = rows

    def run(self, dataframe: pd.DataFrame, **kwargs: Any) -> EntityDetectionResult:
        self.calls += 1
        output = _custom_frame(dataframe, kwargs)
        return EntityDetectionResult(output if self.rows is None else output.iloc[self.rows], [])


class _SubclassDetector(EntityDetectionWorkflow):
    """A caller-supplied EntityDetectionWorkflow subclass overriding run(); returns a 0.4.0-shaped result."""

    def __init__(self, adapter: NddAdapter, rows: list[int] | None = None) -> None:
        super().__init__(adapter)
        self.calls = 0
        self.rows = rows

    def run(self, dataframe: pd.DataFrame, **kwargs: Any) -> EntityDetectionResult:  # type: ignore[override]
        self.calls += 1
        output = _custom_frame(dataframe, kwargs)
        return EntityDetectionResult(output if self.rows is None else output.iloc[self.rows], [])


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("kind", ["duck_typed", "subclass"])
def test_caller_supplied_detector_with_substitute_matches_release(
    scripted: tuple[DataDesigner, Counter[str]], tmp_path: Path, kind: str, mode: str
) -> None:
    designer, calls = scripted
    detector = _DuckDetector() if kind == "duck_typed" else _SubclassDetector(NddAdapter(designer))
    anonymizer = _anonymizer(designer, detection_workflow=detector)  # type: ignore[arg-type]

    assert _execute(anonymizer, mode, tmp_path) == _REPLACED
    assert detector.calls == 1
    # 0.4.0: the detector's own work makes no request; the map health-checks its model once and asks once per row.
    assert calls == {"health_check": 1, "map": 2}


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("skip_health_check", [True, False])
def test_substitute_with_skipped_health_checks_matches_release(
    scripted: tuple[DataDesigner, Counter[str]], tmp_path: Path, skip_health_check: bool, mode: str
) -> None:
    designer, calls = scripted
    anonymizer = _anonymizer(designer, skip_health_check=skip_health_check)

    assert _execute(anonymizer, mode, tmp_path) == _REPLACED
    # 0.4.0: one health check per DD workflow (detection, then the map) unless the model skips it.
    assert calls == {
        "detector": 2,
        "validator": 2,
        "augmenter": 2,
        "map": 2,
        **({} if skip_health_check else {"health_check": 2}),
    }


def test_substitute_with_environment_skipped_health_checks_matches_release(
    scripted: tuple[DataDesigner, Counter[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    designer, calls = scripted
    monkeypatch.setenv("DATA_DESIGNER_SKIP_MODEL_HEALTH_CHECKS", "1")

    assert _execute(_anonymizer(designer), "run", tmp_path) == _REPLACED
    assert calls == {"detector": 2, "validator": 2, "augmenter": 2, "map": 2}


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("kind", ["builtin", "duck_typed", "subclass"])
def test_substitute_without_detected_entities_matches_release(
    scripted: tuple[DataDesigner, Counter[str]], tmp_path: Path, kind: str, mode: str
) -> None:
    designer, calls = scripted
    detectors: dict[str, Any] = {"duck_typed": _DuckDetector(), "subclass": _SubclassDetector(NddAdapter(designer))}
    anonymizer = _anonymizer(designer, **({"detection_workflow": detectors[kind]} if kind in detectors else {}))

    assert _execute(anonymizer, mode, tmp_path, texts=_PLAIN) == [{"text": t, "text_replaced": t} for t in _PLAIN]
    # 0.4.0: with no entity rows the map makes no request, so its model is never health-checked.
    assert calls == ({"detector": 2, "augmenter": 2, "health_check": 1} if kind == "builtin" else {})


@pytest.mark.parametrize("mode", ["run", "preview"])
def test_substitute_with_some_rows_without_entities_matches_release(
    scripted: tuple[DataDesigner, Counter[str]], tmp_path: Path, mode: str
) -> None:
    designer, calls = scripted
    texts = [_TEXTS[0], _PLAIN[1]]

    assert _execute(_anonymizer(designer), mode, tmp_path, texts=texts) == [
        {"text": "Alice waved.", "text_replaced": "Maya waved."},
        {"text": "water is wet.", "text_replaced": "water is wet."},
    ]
    assert calls == {"detector": 2, "validator": 1, "augmenter": 2, "health_check": 2, "map": 1}


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("strategy", [Redact, Hash, Annotate])
def test_local_strategies_without_detected_entities_match_release(
    scripted: tuple[DataDesigner, Counter[str]], tmp_path: Path, strategy: type[Redact | Hash | Annotate], mode: str
) -> None:
    designer, calls = scripted

    assert _execute(_anonymizer(designer), mode, tmp_path, texts=_PLAIN, replace=strategy()) == [
        {"text": t, "text_replaced": t} for t in _PLAIN
    ]
    assert calls == {"detector": 2, "augmenter": 2, "health_check": 1}


# 0.4.0 transforms whatever rows a caller's detector returns: a dropped row is absent, a duplicate is repeated.
# Substitute renumbers rows and maps each one (preview maps at most the requested count); local strategies keep
# the detector's index.
_DROP, _DUPLICATE = [1], [0, 1, 0]
_ALICE = {"text": "Alice waved.", "text_replaced": "Maya waved."}
_BOB = {"text": "Bob stayed.", "text_replaced": "Theo stayed."}


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("rows", [_DROP, _DUPLICATE], ids=["drop", "duplicate"])
@pytest.mark.parametrize("kind", ["duck_typed", "subclass"])
def test_caller_detector_returning_other_rows_with_substitute_matches_release(
    scripted: tuple[DataDesigner, Counter[str]], tmp_path: Path, kind: str, rows: list[int], mode: str
) -> None:
    designer, calls = scripted
    detector = _DuckDetector(rows) if kind == "duck_typed" else _SubclassDetector(NddAdapter(designer), rows)
    frame = _frame(_anonymizer(designer, detection_workflow=detector), mode, tmp_path)  # type: ignore[arg-type]

    expected = [_BOB] if rows == _DROP else [_ALICE, _BOB, _ALICE] if mode == "run" else [_ALICE, _BOB]
    assert frame.to_dict("records") == expected
    assert frame.index.tolist() == list(range(len(expected)))
    assert calls == {"health_check": 1, "map": len(expected)}


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("rows", [_DROP, _DUPLICATE], ids=["drop", "duplicate"])
def test_caller_detector_returning_other_rows_with_redact_matches_release(
    scripted: tuple[DataDesigner, Counter[str]], tmp_path: Path, rows: list[int], mode: str
) -> None:
    designer, calls = scripted
    detector = _SubclassDetector(NddAdapter(designer), rows)
    frame = _frame(
        _anonymizer(designer, detection_workflow=detector),  # type: ignore[arg-type]
        mode,
        tmp_path,
        replace=Redact(),
    )

    redacted = {"Alice waved.": "[REDACTED_FIRST_NAME] waved.", "Bob stayed.": "[REDACTED_FIRST_NAME] stayed."}
    assert frame.to_dict("records") == [{"text": _TEXTS[r], "text_replaced": redacted[_TEXTS[r]]} for r in rows]
    assert frame.index.tolist() == rows
    assert calls == {}


@pytest.mark.parametrize("mode", ["run", "preview"])
@pytest.mark.parametrize("kind", ["duck_typed", "subclass"])
def test_caller_detector_without_entities_with_rewrite_matches_release(
    scripted: tuple[DataDesigner, Counter[str]], tmp_path: Path, kind: str, mode: str
) -> None:
    designer, calls = scripted
    detector = _DuckDetector() if kind == "duck_typed" else _SubclassDetector(NddAdapter(designer))
    anonymizer = _anonymizer(designer, detection_workflow=detector)  # type: ignore[arg-type]
    source = tmp_path / "input.csv"
    pd.DataFrame({"text": _PLAIN}).to_csv(source, index=False)
    data = AnonymizerInput(source=str(source), text_column="text")
    config = AnonymizerConfig(rewrite=Rewrite())
    result = (
        anonymizer.run(config=config, data=data)
        if mode == "run"
        else anonymizer.preview(config=config, data=data, num_records=len(_PLAIN))
    )

    # 0.4.0: a 0.4.0-shaped result (no source positions) aligns with its rows; no entities, no requests.
    assert not result.failed_records
    assert result.dataframe[["text", "text_rewritten"]].to_dict("records") == [
        {"text": t, "text_rewritten": t} for t in _PLAIN
    ]
    assert detector.calls == 1
    assert calls == {}


def test_substitute_on_zero_rows_fails_like_release(
    scripted: tuple[DataDesigner, Counter[str]], tmp_path: Path
) -> None:
    designer, calls = scripted
    source = tmp_path / "input.parquet"
    pd.DataFrame({"text": pd.Series([], dtype=object)}).to_parquet(source, index=False)
    data = AnonymizerInput(source=str(source), text_column="text")

    # 0.4.0: detection still runs DataDesigner on the empty frame, which health-checks its model and then
    # finds no generated dataset; the facade reports that as its own workflow error.
    with pytest.raises(AnonymizerWorkflowError, match="^Workflow failed: ") as raised:
        _anonymizer(designer).run(config=AnonymizerConfig(replace=Substitute()), data=data)
    assert type(raised.value.__cause__).__name__ == "DataDesignerGenerationError"
    assert calls == {"health_check": 1}
