# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse
from data_designer.engine.models.errors import ModelGenerationValidationFailureError
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.testing.utils import assert_valid_plugin

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.evaluation.judge_base import _BaseJudgeWorkflow
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.evaluation.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.evaluation.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow
from anonymizer.engine.workflow_columns.evaluation.judge.config import JudgeColumnConfig, JudgeKind
from anonymizer.engine.workflow_columns.evaluation.judge.impl import _JUDGE_SCHEMAS, JudgeColumnGenerator
from anonymizer.engine.workflow_columns.evaluation.judge.plugins import judge_column_plugin

_MODEL_ALIAS = "judge-model"
_MODEL_CONFIG = ModelConfig(alias=_MODEL_ALIAS, model="test-model", provider="test-provider")
_MODEL_PROVIDER = ModelProvider(name="test-provider", endpoint="https://example.invalid/v1")


class _ScriptedClient:
    def __init__(
        self,
        *,
        sync_responses: list[str] | None = None,
        async_responses: list[str] | None = None,
    ) -> None:
        self.sync_responses = list(sync_responses or [])
        self.async_responses = list(async_responses or [])
        self.sync_requests: list[ChatCompletionRequest] = []
        self.async_requests: list[ChatCompletionRequest] = []

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.sync_requests.append(request)
        return ChatCompletionResponse(message=AssistantMessage(content=self.sync_responses.pop(0)))

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.async_requests.append(request)
        return ChatCompletionResponse(message=AssistantMessage(content=self.async_responses.pop(0)))


class _ModelRegistry:
    def __init__(self, facade: ModelFacade) -> None:
        self._facade = facade

    def get_model(self, *, model_alias: str) -> ModelFacade:
        if model_alias != _MODEL_ALIAS:
            raise KeyError(model_alias)
        return self._facade

    def get_model_config(self, *, model_alias: str) -> ModelConfig:
        if model_alias != _MODEL_ALIAS:
            raise KeyError(model_alias)
        return _MODEL_CONFIG

    def get_model_provider(self, *, model_alias: str) -> ModelProvider:
        if model_alias != _MODEL_ALIAS:
            raise KeyError(model_alias)
        return _MODEL_PROVIDER


def _fenced(payload: dict[str, Any]) -> str:
    return f"```json\n{json.dumps(payload)}\n```"


def _make_generator(
    *,
    judge_kind: JudgeKind = JudgeKind.DETECTION,
    sync_responses: list[str] | None = None,
    async_responses: list[str] | None = None,
    correction_steps: int = 0,
    restarts: int = 0,
) -> tuple[JudgeColumnGenerator, _ScriptedClient, ModelFacade]:
    client = _ScriptedClient(sync_responses=sync_responses, async_responses=async_responses)
    facade = ModelFacade(
        _MODEL_CONFIG,
        ModelProviderRegistry(providers=[_MODEL_PROVIDER]),
        client=cast(ModelClient, client),
    )
    resource_provider = SimpleNamespace(
        model_registry=_ModelRegistry(facade),
        run_config=RunConfig(
            max_conversation_correction_steps=correction_steps,
            max_conversation_restarts=restarts,
        ),
    )
    config = JudgeColumnConfig(
        name="_judgment",
        prompt="Judge this record.",
        model_alias=_MODEL_ALIAS,
        judge_kind=judge_kind,
    )
    return JudgeColumnGenerator(config, cast(ResourceProvider, resource_provider)), client, facade


def test_judge_plugin_satisfies_data_designer_contract() -> None:
    assert_valid_plugin(judge_column_plugin)


def test_judge_config_round_trips_and_is_discoverable_in_fresh_process(tmp_path: Path) -> None:
    builder = DataDesignerConfigBuilder(model_configs=[_MODEL_CONFIG]).add_column(
        JudgeColumnConfig(
            name="_judgment",
            prompt="Judge {{ text }}.",
            model_alias=_MODEL_ALIAS,
            judge_kind=JudgeKind.DETECTION,
        )
    )
    config_path = tmp_path / "judge-config.json"
    output_path = tmp_path / "restored-column.json"
    builder.get_builder_config().to_json(config_path)

    restored = DataDesignerConfigBuilder.from_config(config_path).get_column_configs()
    assert len(restored) == 1
    assert isinstance(restored[0], JudgeColumnConfig)
    assert restored[0].judge_kind == JudgeKind.DETECTION.value

    script = """
import json
import sys
from pathlib import Path

from data_designer.config.config_builder import DataDesignerConfigBuilder

column = DataDesignerConfigBuilder.from_config(Path(sys.argv[1])).get_column_configs()[0]
Path(sys.argv[2]).write_text(
    json.dumps({"class_name": type(column).__name__, "column_type": column.column_type})
)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(config_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output_path.read_text()) == {
        "class_name": "JudgeColumnConfig",
        "column_type": "anonymizer-evaluation-judge",
    }


@pytest.mark.parametrize(
    ("judge_kind", "payload", "detail_field"),
    [
        (JudgeKind.DETECTION, {"all_valid": True, "invalid_entities": None}, "invalid_entities"),
        (JudgeKind.TYPE_FIDELITY, {"all_valid": True, "invalid_replacements": None}, "invalid_replacements"),
        (JudgeKind.ATTRIBUTE_FIDELITY, {"all_valid": True, "entities": None}, "entities"),
        (JudgeKind.RELATIONAL_CONSISTENCY, {"all_consistent": True, "relations": None}, "relations"),
    ],
)
def test_parser_normalizes_positive_null_details(
    judge_kind: JudgeKind,
    payload: dict[str, Any],
    detail_field: str,
) -> None:
    generator, _, _ = _make_generator(judge_kind=judge_kind)
    parsed = generator.response_recipe.parse(_fenced(payload))
    assert parsed.model_dump(mode="python")[detail_field] == []


@pytest.mark.parametrize("invalid_verdict", ["true", 1])
def test_parser_rejects_coerced_boolean_verdicts(invalid_verdict: str | int) -> None:
    generator, _, _ = _make_generator()
    with pytest.raises(ParserException, match="response_schema"):
        generator.response_recipe.parse(
            _fenced({"all_valid": invalid_verdict, "invalid_entities": []}),
        )


_FAILING_ATTRIBUTE = {
    "original": "40",
    "label": "age",
    "synthetic": "12",
    "attributes_checked": ["age_bucket"],
    "passes": False,
    "reasoning": "Adult bucket changed to child.",
}
_PASSING_ATTRIBUTE = {**_FAILING_ATTRIBUTE, "passes": True, "reasoning": "Same bucket."}
_FAILING_RELATION = {
    "description": "city <-> state",
    "entities": ["Austin (city) -> Portland", "TX (state) -> CA"],
    "passes": False,
    "reasoning": "Portland is not in California.",
}
_PASSING_RELATION = {**_FAILING_RELATION, "passes": True, "reasoning": "Portland is in Oregon."}
_INVALID_ENTITY = {"value": "Alice", "label": "first_name", "reasoning": "contradiction"}
_INVALID_REPLACEMENT = {"original": "Alice", "label": "first_name", "synthetic": "[X]", "reasoning": "placeholder"}


@pytest.mark.parametrize(
    ("judge_kind", "payload"),
    [
        # Contradictory verdict/detail pairs, in both directions.
        (JudgeKind.DETECTION, {"all_valid": True, "invalid_entities": [_INVALID_ENTITY]}),
        (JudgeKind.DETECTION, {"all_valid": False, "invalid_entities": []}),
        (JudgeKind.TYPE_FIDELITY, {"all_valid": True, "invalid_replacements": [_INVALID_REPLACEMENT]}),
        (JudgeKind.TYPE_FIDELITY, {"all_valid": False, "invalid_replacements": []}),
        (JudgeKind.ATTRIBUTE_FIDELITY, {"all_valid": True, "entities": [_FAILING_ATTRIBUTE]}),
        (JudgeKind.ATTRIBUTE_FIDELITY, {"all_valid": False, "entities": [_PASSING_ATTRIBUTE]}),
        (JudgeKind.ATTRIBUTE_FIDELITY, {"all_valid": False, "entities": []}),
        (JudgeKind.RELATIONAL_CONSISTENCY, {"all_consistent": True, "relations": [_FAILING_RELATION]}),
        (JudgeKind.RELATIONAL_CONSISTENCY, {"all_consistent": False, "relations": [_PASSING_RELATION]}),
        (JudgeKind.RELATIONAL_CONSISTENCY, {"all_consistent": False, "relations": []}),
        # Nested nulls and non-boolean verdicts.
        (JudgeKind.DETECTION, {"all_valid": False, "invalid_entities": [None]}),
        (JudgeKind.TYPE_FIDELITY, {"all_valid": False, "invalid_replacements": [None]}),
        (JudgeKind.ATTRIBUTE_FIDELITY, {"all_valid": True, "entities": [None]}),
        (JudgeKind.RELATIONAL_CONSISTENCY, {"all_consistent": True, "relations": [None]}),
        (JudgeKind.ATTRIBUTE_FIDELITY, {"all_valid": 1, "entities": []}),
        (JudgeKind.RELATIONAL_CONSISTENCY, {"all_consistent": "true", "relations": []}),
    ],
)
def test_parser_rejects_contradictory_and_malformed_responses(judge_kind: JudgeKind, payload: dict[str, Any]) -> None:
    generator, _, _ = _make_generator(judge_kind=judge_kind)
    with pytest.raises(ParserException, match="response_schema"):
        generator.response_recipe.parse(_fenced(payload))


@pytest.mark.parametrize(
    ("judge_kind", "payload"),
    [
        (JudgeKind.ATTRIBUTE_FIDELITY, {"all_valid": True, "entities": [_PASSING_ATTRIBUTE]}),
        (JudgeKind.RELATIONAL_CONSISTENCY, {"all_consistent": True, "relations": [_PASSING_RELATION]}),
        (JudgeKind.ATTRIBUTE_FIDELITY, {"all_valid": False, "entities": [_PASSING_ATTRIBUTE, _FAILING_ATTRIBUTE]}),
    ],
)
def test_parser_preserves_nonempty_details_that_agree_with_verdict(
    judge_kind: JudgeKind, payload: dict[str, Any]
) -> None:
    generator, _, _ = _make_generator(judge_kind=judge_kind)
    assert generator.response_recipe.parse(_fenced(payload)).model_dump(mode="python") == payload


@pytest.mark.parametrize(
    "workflow_cls",
    [
        DetectionJudgeWorkflow,
        TypeFidelityJudgeWorkflow,
        AttributeFidelityJudgeWorkflow,
        RelationalConsistencyJudgeWorkflow,
    ],
)
def test_workflow_column_config_uses_judge_column_with_matching_schema(
    workflow_cls: type[_BaseJudgeWorkflow],
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    column = workflow_cls(adapter=cast(Any, None)).column_config(stub_evaluate_model_selection)

    assert isinstance(column, JudgeColumnConfig)
    assert column.name == workflow_cls.RAW_COL
    assert _JUDGE_SCHEMAS[JudgeKind(column.judge_kind)] is workflow_cls.SCHEMA
    assert "wrapped in a single ```json Markdown code fence" in column.prompt
    assert "Do NOT wrap your output" not in column.prompt


def test_generator_prompt_includes_response_schema_instructions() -> None:
    generator, client, _ = _make_generator(
        sync_responses=[_fenced({"all_valid": True, "invalid_entities": []})],
    )

    generator.generate({})

    prompt_text = json.dumps(client.sync_requests[0].messages)
    assert "all_valid" in prompt_text


def test_sync_generator_corrects_invalid_response_and_returns_structured_output() -> None:
    generator, client, facade = _make_generator(
        sync_responses=[
            _fenced({"all_valid": "true", "invalid_entities": []}),
            _fenced({"all_valid": True, "invalid_entities": None}),
        ],
        correction_steps=1,
    )

    result = generator.generate({})

    assert result["_judgment"] == {"all_valid": True, "invalid_entities": []}
    assert len(client.sync_requests) == 2
    assert len(client.async_requests) == 0
    assert facade.usage_stats.request_usage.total_requests == 2


def test_async_generator_corrects_invalid_response_and_tracks_requests_separately() -> None:
    generator, client, facade = _make_generator(
        async_responses=[
            _fenced({"all_valid": False, "invalid_entities": []}),
            _fenced(
                {
                    "all_valid": False,
                    "invalid_entities": [
                        {"value": "Alice", "label": "first_name", "reasoning": "Not identifying in context."}
                    ],
                }
            ),
        ],
        correction_steps=1,
    )

    result = asyncio.run(generator.agenerate({}))

    assert result["_judgment"]["all_valid"] is False
    assert len(result["_judgment"]["invalid_entities"]) == 1
    assert len(client.sync_requests) == 0
    assert len(client.async_requests) == 2
    assert facade.usage_stats.request_usage.total_requests == 2


def test_generator_preserves_correction_and_restart_budgets() -> None:
    invalid = _fenced({"all_valid": False, "invalid_entities": []})
    generator, client, _ = _make_generator(
        sync_responses=[
            invalid,
            invalid,
            invalid,
            _fenced({"all_valid": True, "invalid_entities": None}),
        ],
        correction_steps=1,
        restarts=1,
    )

    result = generator.generate({})

    assert result["_judgment"] == {"all_valid": True, "invalid_entities": []}
    assert len(client.sync_requests) == 4


def test_exhausted_corrections_leave_judgment_unavailable() -> None:
    invalid = _fenced({"all_valid": False, "invalid_entities": []})
    generator, client, _ = _make_generator(
        sync_responses=[invalid, invalid],
        correction_steps=1,
    )

    with pytest.raises(ModelGenerationValidationFailureError, match="could not be parsed"):
        generator.generate({})

    assert len(client.sync_requests) == 2
    assert DetectionJudgeWorkflow._flatten_judgment(None) == (None, [])
