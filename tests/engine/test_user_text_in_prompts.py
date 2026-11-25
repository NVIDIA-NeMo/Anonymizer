# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User text (``Rewrite.instructions``) reaches the prompts that take it as literal text, and no other prompt.

The end-to-end cases run the real DataDesigner render with a scripted model client, so the Jinja environment and
delimiters are DataDesigner's own; only the HTTP client is replaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.engine.models.clients import factory as client_factory
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_FULL_REWRITE,
    COL_LEAKAGE_MASS,
    COL_LEAKED_PRIVACY_ITEMS,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITE_BASELINE_TEXT,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_REWRITE_TAGGED_TEXT,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TAG_NOTATION,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.prompt_utils import jinja_literal
from anonymizer.engine.rewrite import rewrite_workflow
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.repair import RepairParams, _render_repair_prompt
from anonymizer.engine.rewrite.rewrite_generation import RewriteGenerationWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.engine.workflow_columns.response.config import ResponseColumnConfig
from anonymizer.interface.anonymizer import Anonymizer

# Template syntax a user might type, including a block end that would close a plain raw block early.
USER_TEXT = "Keep {{ x }} and {% if y %}z{% endif %} {# note #} {% endraw %}{{ 7 * 7 }} as written."
_PROVIDER = ModelProvider(name="scripted", endpoint="http://scripted.invalid/v1", api_key="unused")
_GOAL = PrivacyGoal(protect="Names of people", preserve="The general meaning of the text")


class _RecordingClient:
    """Records each request's message text and replies with ``reply``."""

    def __init__(self, prompts: list[str], reply: str) -> None:
        self._prompts = prompts
        self._reply = reply

    def _response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self._prompts.append(json.dumps(request.messages, default=str, ensure_ascii=False))
        return ChatCompletionResponse(message=AssistantMessage(content=self._reply))

    def completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._response(request)

    async def acompletion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return self._response(request)

    def supports_chat_completion(self) -> bool:
        return True

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


@pytest.fixture
def prompts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sent: list[str] = []

    def create_model_client(model_config: ModelConfig, *args: Any, **kwargs: Any) -> _RecordingClient:
        return _RecordingClient(sent, '{"rewritten_text": "Someone works here."}')

    monkeypatch.setattr(client_factory, "create_model_client", create_model_client)
    return sent


def _model(alias: str) -> ModelConfig:
    return ModelConfig(alias=alias, model=alias, provider=_PROVIDER.name, skip_health_check=True)


def _rendered(text: str) -> str:
    """``text`` as it appears inside the JSON-encoded messages a client receives."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


def test_jinja_literal_survives_its_own_block_end() -> None:
    from data_designer.engine.processing.ginja.environment import (
        NativeJinjaSandboxEnvironment,
        UserTemplateSandboxEnvironment,
    )

    for environment in (UserTemplateSandboxEnvironment, NativeJinjaSandboxEnvironment):
        env = environment(allowed_references=["col"])
        template = "A " + jinja_literal(USER_TEXT) + " {{ col }}"
        env.validate_template(template)
        assert env.render_template(template, {"col": "v"}) == f"A {USER_TEXT} v"


def test_rewrite_instructions_reach_the_rewrite_prompt_verbatim_through_datadesigner(
    tmp_path: Path, stub_rewrite_model_selection: RewriteModelSelection, prompts: list[str]
) -> None:
    columns = RewriteGenerationWorkflow().columns(
        selected_models=stub_rewrite_model_selection, privacy_goal=_GOAL, instructions=USER_TEXT
    )
    rewrite = next(column for column in columns if column.name == COL_FULL_REWRITE)
    assert isinstance(rewrite, ResponseColumnConfig)
    seeded = pd.DataFrame(
        {
            COL_TEXT: ["Alice works here."],
            COL_TAG_NOTATION: ["xml"],
            COL_REWRITE_TAGGED_TEXT: ["<first_name>Alice</first_name> works here."],
            COL_REWRITE_DISPOSITION_BLOCK: [[{"entity_label": "first_name", "entity_value": "Alice",
                                              "sensitivity": "high", "protection_method_suggestion": "generalize"}]],
            COL_REPLACEMENT_MAP_FOR_PROMPT: [{"replacements": []}],
        }
    )  # fmt: skip
    adapter = NddAdapter(data_designer=DataDesigner(artifact_path=tmp_path, model_providers=[_PROVIDER]))

    adapter.run_workflow(
        seeded,
        model_configs=[_model(rewrite.model_alias)],
        columns=[rewrite],
        workflow_name="rewrite-instructions",
    )

    assert len(prompts) == 1
    assert "<user_instructions>" in prompts[0]
    assert _rendered(USER_TEXT) in prompts[0]


def test_rewrite_prompt_without_instructions_is_unchanged(stub_rewrite_model_selection: RewriteModelSelection) -> None:
    def prompt(**kwargs: Any) -> str:
        columns = RewriteGenerationWorkflow().columns(
            selected_models=stub_rewrite_model_selection, privacy_goal=_GOAL, **kwargs
        )
        column = next(column for column in columns if column.name == COL_FULL_REWRITE)
        assert isinstance(column, ResponseColumnConfig)
        return column.prompt

    assert prompt(instructions=None) == prompt(instructions="  ") == prompt()
    assert "user_instructions" not in prompt()


def test_repair_prompt_takes_instructions_as_plain_text() -> None:
    row = {
        COL_TEXT: "Alice works here.",
        COL_REWRITE_BASELINE_TEXT: "Alice works here.",
        COL_REWRITTEN_TEXT: "Alice works here.",
        COL_LEAKAGE_MASS: 1.0,
        COL_ANY_HIGH_LEAKED: True,
        COL_LEAKED_PRIVACY_ITEMS: "- name",
        COL_UTILITY_SCORE: 0.9,
    }
    goal = _GOAL.to_prompt_string()

    with_text = _render_repair_prompt(
        row, RepairParams(privacy_goal_str=goal, max_privacy_leak=1.0, instructions=USER_TEXT)
    )
    without = _render_repair_prompt(row, RepairParams(privacy_goal_str=goal, max_privacy_leak=1.0))

    assert f"\n{USER_TEXT}\n</user_instructions>" in with_text  # not rendered as Jinja, so not wrapped
    assert "{% raw %}" not in with_text
    assert "user_instructions" not in without


def _payload(column: Any) -> str:
    """Everything a column hands its model: its prompt, system prompt and generator parameters."""
    params = getattr(column, "generator_params", None)
    return " ".join(
        str(part)
        for part in (
            getattr(column, "prompt", None),
            getattr(column, "system_prompt", None),
            params.model_dump_json() if params is not None else None,
        )
        if part is not None
    )


def test_combined_graph_gives_instructions_only_to_rewrite_and_repair(
    stub_rewrite_model_selection: RewriteModelSelection, stub_replace_model_selection: ReplaceModelSelection
) -> None:
    graph = CombinedRewriteWorkflow(adapter=Mock()).build_graph(
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=2),
        data_summary="Support tickets.",
        instructions="MARKER-7731",
    )

    carrying = {column.name for column in graph.columns if "MARKER-7731" in _payload(column)}

    assert carrying == {COL_FULL_REWRITE, *(state.rewritten_text for state in graph.repair_states)}


def test_staged_workflow_gives_instructions_only_to_rewrite_and_repair(
    monkeypatch: pytest.MonkeyPatch, stub_rewrite_model_selection: RewriteModelSelection
) -> None:
    workflow = RewriteWorkflow(adapter=Mock())
    built: list[Any] = []

    def recording(build: Any) -> Any:
        def columns(**kwargs: Any) -> list[Any]:
            result = build(**kwargs)
            built.extend(result)
            return result

        return columns

    def run_columns(adapter: Any, frame: Any, columns: list[Any], **_: Any) -> Any:
        built.extend(columns)
        raise _Stop

    monkeypatch.setattr(rewrite_workflow, "run_columns", run_columns)
    monkeypatch.setattr(workflow._evaluate_wf, "columns", recording(workflow._evaluate_wf.columns))
    monkeypatch.setattr(workflow._repair_wf, "columns", recording(workflow._repair_wf.columns))
    rows = rewrite_workflow._Rows(pd.DataFrame({COL_TEXT: ["Alice works here."]}))

    with pytest.raises(_Stop):
        workflow._pipeline(
            rows,
            model_configs=[],
            selected_models=stub_rewrite_model_selection,
            privacy_goal=_GOAL,
            data_summary="Support tickets.",
            preview_num_records=None,
            strict_entity_protection=False,
            instructions="MARKER-7731",
        )
    workflow._evaluate_repair_stages(
        rows,
        model_configs=[],
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=1),
        preview_num_records=None,
        instructions="MARKER-7731",
    )

    assert {column.name for column in built if "MARKER-7731" in _payload(column)} == {
        COL_FULL_REWRITE,
        COL_REWRITTEN_TEXT_NEXT,
    }


class _Stop(Exception):
    pass


class _RecordingRunner(RewriteWorkflow):
    """A caller's runner: ``Anonymizer`` calls its ``run`` with the rewrite options after detection."""

    options: dict[str, Any] = {}

    def run(self, dataframe: pd.DataFrame, **options: Any) -> Any:
        type(self).options = options
        raise _Stop


def test_anonymizer_passes_rewrite_instructions_to_the_runner(tmp_path: Path) -> None:
    detector = EntityDetectionWorkflow(adapter=Mock())
    detector.run = Mock(
        side_effect=lambda frame, **_: EntityDetectionResult(
            dataframe=frame, failed_records=[], source_positions=tuple(range(len(frame)))
        )
    )
    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works here."]}).to_csv(csv_path, index=False)
    anonymizer = Anonymizer(detection_workflow=detector, rewrite_runner=_RecordingRunner(adapter=Mock()))

    with pytest.raises(Exception):
        anonymizer.run(
            config=AnonymizerConfig(rewrite=Rewrite(privacy_goal=_GOAL, instructions=USER_TEXT), emit_telemetry=False),
            data=AnonymizerInput(source=str(csv_path), text_column="text"),
        )

    assert _RecordingRunner.options["instructions"] == USER_TEXT
