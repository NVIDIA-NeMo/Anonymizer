# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A malformed model reply through the real DataDesigner run and model facade.

Only the HTTP client is scripted: DataDesigner builds the facades, runs the columns and records failed rows.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.engine.models.clients import factory as client_factory
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.models import DetectionModelSelection, RewriteModelSelection
from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_MEANING_UNITS_SERIALIZED,
    COL_QUALITY_QA,
    COL_TEXT,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import load_default_model_selection
from anonymizer.engine.rewrite.qa_generation import QAGenerationWorkflow

_PROVIDER = ModelProvider(name="scripted", endpoint="http://scripted.invalid/v1", api_key="unused")
_DETECTOR = "scripted-detector"
_VALIDATOR = "scripted-validator"
_AUGMENTER = "scripted-augmenter"
_TEXT = "Card ending in 7312 will be charged."
# A second, clean record keeps the run alive when the first record fails (DataDesigner raises when no record survives).
_OTHER_TEXT = "Nothing to see."
_OTHER_REPLIES = {
    "scripted-detector": '{"entities": []}',
    "scripted-augmenter": '```json\n{"entities": []}\n```',
}

# The 11-14 reply shape: malformed JSON at context_after, which the parser's repair turns into an item without label.
_MALFORMED_AUGMENTER = (
    '```json\n{"entities": [{"value": "7312", "context_before": "ending in ", '
    '"context_after". will be charged", "label": "card_last_four"}]}\n```'
)
_GOOD_AUGMENTER = (
    '```json\n{"entities": [{"value": "7312", "context_before": "ending in ", '
    '"context_after": " will be charged", "label": "card_last_four"}]}\n```'
)
_MALFORMED_QA = (
    '```json\n{"items": [{"id": 1, "aspect": "Charge", "importance": "important", "question": "What?"}]}\n```'
)
_GOOD_QA = (
    '```json\n{"items": [{"id": 1, "aspect": "Charge", "importance": "important", '
    '"question": "What is charged?", "reference_answer": "The card."}]}\n```'
)


_MESSAGE_COUNTS: dict[str, list[int]] = defaultdict(list)


class _ScriptedClient:
    """Replies per alias from a script for the record under test; the last reply repeats once the script runs out.

    Requests for the clean record get a fixed well-formed reply and are not counted.
    """

    def __init__(self, alias: str, script: dict[str, list[str]], calls: dict[str, int]) -> None:
        self._alias = alias
        self._script = script
        self._calls = calls

    def _reply(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if _OTHER_TEXT in json.dumps(request.messages, default=str):
            content = _OTHER_REPLIES.get(self._alias, _GOOD_QA)
            return ChatCompletionResponse(message=AssistantMessage(content=content))
        replies = self._script[self._alias]
        index = self._calls[self._alias]
        self._calls[self._alias] += 1
        _MESSAGE_COUNTS[self._alias].append(len(request.messages))
        return ChatCompletionResponse(message=AssistantMessage(content=replies[min(index, len(replies) - 1)]))

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._reply(request)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._reply(request)

    def supports_chat_completion(self) -> bool:
        return True

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, list[str]], dict[str, int]]:
    script: dict[str, list[str]] = {}
    calls: dict[str, int] = defaultdict(int)
    _MESSAGE_COUNTS.clear()

    def create_model_client(model_config: ModelConfig, *args: Any, **kwargs: Any) -> _ScriptedClient:
        return _ScriptedClient(model_config.alias, script, calls)

    monkeypatch.setattr(client_factory, "create_model_client", create_model_client)
    return script, calls


def _adapter(tmp_path: Path, run_config: RunConfig | None = None) -> NddAdapter:
    data_designer = DataDesigner(artifact_path=tmp_path, model_providers=[_PROVIDER])
    if run_config is not None:
        data_designer.set_run_config(run_config)
    return NddAdapter(data_designer=data_designer)


def _model(alias: str) -> ModelConfig:
    return ModelConfig(alias=alias, model=alias, provider=_PROVIDER.name, skip_health_check=True)


def _detect(tmp_path: Path, run_config: RunConfig | None = None):
    selection = DetectionModelSelection(
        entity_detector=_DETECTOR,
        entity_validator=[_VALIDATOR],
        entity_augmenter=_AUGMENTER,
        latent_detector=_AUGMENTER,
    )
    return EntityDetectionWorkflow(adapter=_adapter(tmp_path, run_config)).detect_and_validate_entities(
        pd.DataFrame({COL_TEXT: [_TEXT, _OTHER_TEXT]}),
        model_configs=[_model(_DETECTOR), _model(_VALIDATOR), _model(_AUGMENTER)],
        selected_models=selection,
        gliner_detection_threshold=0.3,
    )


def test_augmenter_retries_a_malformed_reply_instead_of_dropping_the_record(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DETECTOR] = [json.dumps({"entities": []})]
    script[_AUGMENTER] = [_MALFORMED_AUGMENTER, _GOOD_AUGMENTER]

    result = _detect(tmp_path)

    assert result.failed_records == []
    assert calls[_AUGMENTER] == 2
    assert len(result.dataframe) == 2
    entities = result.dataframe[COL_DETECTED_ENTITIES].iloc[0]["entities"]
    assert [(e["value"], e["label"]) for e in entities] == [("7312", "card_last_four")]
    assert COL_AUGMENTED_ENTITIES in result.dataframe.columns


def test_augmenter_reply_malformed_on_every_attempt_still_fails_the_record(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DETECTOR] = [json.dumps({"entities": []})]
    script[_AUGMENTER] = [_MALFORMED_AUGMENTER]

    result = _detect(tmp_path)

    assert len(result.dataframe) == 1
    assert [record.step for record in result.failed_records] == ["entity-detection"]
    # One grounding attempt, given the run config's restarts (DataDesigner default 5).
    assert calls[_AUGMENTER] == 6


def test_augmenter_uses_the_run_config_restarts_and_corrections(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script[_DETECTOR] = [json.dumps({"entities": []})]
    script[_AUGMENTER] = [_MALFORMED_AUGMENTER]

    result = _detect(tmp_path, RunConfig(max_conversation_restarts=1, max_conversation_correction_steps=2))

    assert [record.step for record in result.failed_records] == ["entity-detection"]
    # Two conversations (1 restart), each a first request plus 2 corrections that grow the conversation.
    first = _MESSAGE_COUNTS[_AUGMENTER][0]
    assert _MESSAGE_COUNTS[_AUGMENTER] == [first, first + 2, first + 4] * 2
    assert calls[_AUGMENTER] == 6


def _quality_qa(tmp_path: Path):
    alias = "scripted-qa"
    selection = RewriteModelSelection.model_validate(
        {**load_default_model_selection().rewrite.model_dump(), "qa_generator": alias}
    )
    (column,) = [
        c
        for c in QAGenerationWorkflow().columns(selected_models=selection)
        if isinstance(c, LLMStructuredColumnConfig) and c.name == COL_QUALITY_QA
    ]
    return _adapter(tmp_path).run_workflow(
        pd.DataFrame(
            {COL_TEXT: [_TEXT, _OTHER_TEXT], COL_MEANING_UNITS_SERIALIZED: ["1. A card is charged.", _OTHER_TEXT]}
        ),
        model_configs=[_model(alias)],
        columns=[column],
        workflow_name="rewrite-pipeline",
    ), alias


def test_quality_qa_already_retries_a_malformed_reply(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script["scripted-qa"] = [_MALFORMED_QA, _GOOD_QA]

    result, alias = _quality_qa(tmp_path)

    assert result.failed_records == []
    assert len(result.dataframe) == 2
    assert calls[alias] == 2


def test_quality_qa_reply_malformed_on_every_attempt_still_fails_the_record(
    tmp_path: Path, scripted: tuple[dict[str, list[str]], dict[str, int]]
) -> None:
    script, calls = scripted
    script["scripted-qa"] = [_MALFORMED_QA]

    result, alias = _quality_qa(tmp_path)

    assert len(result.dataframe) == 1
    assert [record.step for record in result.failed_records] == ["rewrite-pipeline"]
    assert calls[alias] == 6
