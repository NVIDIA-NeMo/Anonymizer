# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One replace judge's unparseable response through the real DataDesigner run and model facade.

Only the HTTP client is scripted. The failing judge must not blank the other judges' scores for that row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.engine.models.clients import factory as client_factory
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE,
    COL_FINAL_ENTITIES,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_TEXT,
    COL_TYPE_FIDELITY_VALID,
)
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.evaluation.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.evaluation.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow

_PROVIDER = ModelProvider(name="scripted", endpoint="http://scripted.invalid/v1", api_key="unused")
_ROLES = {
    "entity_coverage_judge": "s-coverage",
    "detection_validity_judge": "s-detection",
    "replace_type_fidelity_judge": "s-type",
    "replace_relational_consistency_judge": "s-relational",
    "replace_attribute_fidelity_judge": "s-attribute",
    "rewrite_judge": "s-rewrite",
}
_GOOD = {
    "s-coverage": {"candidate_entities": []},
    "s-type": {"all_valid": True, "invalid_replacements": []},
    "s-relational": {"all_consistent": True, "relations": []},
    "s-attribute": {"all_valid": True, "entities": []},
}
_BAD_ROW_MARK = "Northwind"  # only row 0 contains it


class _ScriptedClient:
    """Well-formed judge responses, except prose from the relational judge for row 0 on every attempt."""

    def __init__(self, alias: str) -> None:
        self._alias = alias

    def _response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if self._alias == "s-relational" and _BAD_ROW_MARK in json.dumps(request.messages, default=str):
            content = "I think the relations look fine overall."
        else:
            content = "```json\n" + json.dumps(_GOOD[self._alias]) + "\n```"
        return ChatCompletionResponse(message=AssistantMessage(content=content))

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


def _row(person: str, org: str, sub_person: str, sub_org: str) -> dict[str, Any]:
    return {
        COL_TEXT: f"{person} works at {org}.",
        COL_FINAL_ENTITIES: {"entities": []},
        COL_REPLACED_TEXT: f"{sub_person} works at {sub_org}.",
        COL_REPLACEMENT_MAP: {
            "replacements": [
                {"original": person, "label": "first_name", "synthetic": sub_person},
                {"original": org, "label": "organization", "synthetic": sub_org},
            ]
        },
        COL_ENTITIES_BY_VALUE: {
            "entities_by_value": [
                {"value": person, "labels": ["first_name"]},
                {"value": org, "labels": ["organization"]},
            ]
        },
    }


def test_one_unparseable_judge_keeps_the_other_judges_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client_factory, "create_model_client", lambda model_config, *a, **k: _ScriptedClient(model_config.alias)
    )
    adapter = NddAdapter(data_designer=DataDesigner(artifact_path=tmp_path, model_providers=[_PROVIDER]))
    runner = ReplacementWorkflow(
        type_fidelity_judge=TypeFidelityJudgeWorkflow(adapter=adapter),
        relational_consistency_judge=RelationalConsistencyJudgeWorkflow(adapter=adapter),
        attribute_fidelity_judge=AttributeFidelityJudgeWorkflow(adapter=adapter),
        adapter=adapter,
    )
    trace = pd.DataFrame([_row("Alice", "Northwind", "Maya", "NovaCorp"), _row("Bob", "Contoso", "Liam", "Globex")])

    result = runner.evaluate(
        trace,
        replace_method=Substitute(),
        model_configs=[
            ModelConfig(alias=a, model=a, provider=_PROVIDER.name, skip_health_check=True) for a in _ROLES.values()
        ],
        selected_models=EvaluateModelSelection(**_ROLES),
    )

    row0, row1 = result.dataframe.iloc[0], result.dataframe.iloc[1]
    assert row0[COL_ENTITY_COVERAGE] == 1.0
    assert bool(row0[COL_TYPE_FIDELITY_VALID]) is True
    assert bool(row0[COL_ATTRIBUTE_FIDELITY_VALID]) is True
    assert row0[COL_RELATIONAL_CONSISTENCY_VALID] is None
    for column in (COL_TYPE_FIDELITY_VALID, COL_RELATIONAL_CONSISTENCY_VALID, COL_ATTRIBUTE_FIDELITY_VALID):
        assert bool(row1[column]) is True
    assert row1[COL_ENTITY_COVERAGE] == 1.0
    assert len(result.failed_records) == 1
    assert "_relational_consistency_judge" in result.failed_records[0].reason
