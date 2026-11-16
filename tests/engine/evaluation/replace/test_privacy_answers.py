# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import EvaluateConfig
from anonymizer.engine.constants import (
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_ERROR,
    COL_PRIVACY_QA_REANSWER,
    COL_REPLACED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.evaluation.replace.privacy_answers import (
    CompletePrivacyAnswersSchema,
    PrivacyAnswerParams,
    ReplacePrivacyAnswerWorkflow,
    _make_complete_answer_parser,
    _make_privacy_answer_column,
)
from anonymizer.engine.schemas import PrivacyAnswersSchema


def _answer(item_id: int = 1) -> dict[str, object]:
    return {
        "id": item_id,
        "answer": "no",
        "confidence": 0.95,
        "reason": "Only Maya is mentioned.",
        "evidence": ["Maya"],
    }


def _row() -> dict[str, Any]:
    return {
        COL_TEXT: "ORIGINAL CONTEXT MUST NOT BE INCLUDED",
        COL_REPLACED_TEXT: "Maya arrived.",
        COL_PRIVACY_QA: {
            "items": [
                {
                    "id": 1,
                    "question": "Can the first_name 'Alice' be deduced from the replaced text?",
                    "entity_value": "Alice",
                    "entity_label": "first_name",
                    "category": "direct_identifier",
                    "sensitivity": "high",
                }
            ]
        },
    }


@pytest.mark.parametrize("answers", [[], [_answer(2)], [_answer(), _answer()]])
def test_parser_rejects_missing_extra_and_duplicate_answers(answers: list[dict[str, object]]) -> None:
    recipe = PydanticResponseRecipe(data_type=CompletePrivacyAnswersSchema)
    parser = _make_complete_answer_parser(recipe, [1])
    with pytest.raises(ParserException):
        parser(json.dumps({"answers": answers}))


@pytest.mark.parametrize("field", ["id", "answer", "confidence", "reason", "evidence"])
@pytest.mark.parametrize("null", [False, True])
def test_parser_rejects_missing_or_null_fields(field: str, null: bool) -> None:
    recipe = PydanticResponseRecipe(data_type=CompletePrivacyAnswersSchema)
    parser = _make_complete_answer_parser(recipe, [1])
    answer = _answer()
    if null:
        answer[field] = None
    else:
        del answer[field]
    with pytest.raises(ParserException):
        parser(json.dumps({"answers": [answer]}))


def test_parser_keeps_complete_answers_without_manufacturing_leaks() -> None:
    recipe = PydanticResponseRecipe(data_type=CompletePrivacyAnswersSchema)
    parser = _make_complete_answer_parser(recipe, [1])
    result = parser(json.dumps({"answers": [_answer()]}))
    assert result.model_dump(mode="json") == {"answers": [_answer()]}


class _CorrectingModel:
    """Exercise the supplied parser as the model correction loop would."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0
        self.prompt = ""

    async def agenerate(
        self,
        *,
        prompt: str,
        parser: Callable[[str], PrivacyAnswersSchema],
        max_correction_steps: int,
    ) -> tuple[PrivacyAnswersSchema, None]:
        self.prompt = prompt
        for attempt in range(max_correction_steps + 1):
            self.calls += 1
            try:
                return parser(self.responses[min(attempt, len(self.responses) - 1)]), None
            except ParserException:
                if attempt == max_correction_steps:
                    raise
        raise RuntimeError("Unreachable correction state")


def test_incomplete_answer_can_be_corrected_and_prompt_uses_only_replaced_evidence() -> None:
    model = _CorrectingModel(['{"answers": []}', json.dumps({"answers": [_answer()]})])
    generator = _make_privacy_answer_column("judge")
    result = asyncio.run(generator(_row(), None, {"judge": model}))
    assert model.calls == 2
    assert "Maya arrived." in model.prompt
    assert "Alice" in model.prompt
    assert "ORIGINAL CONTEXT MUST NOT BE INCLUDED" not in model.prompt
    assert result[COL_PRIVACY_QA_REANSWER] == {"answers": [_answer()]}
    assert result[COL_PRIVACY_QA_ERROR] is None


def test_exhausted_corrections_return_unavailable_without_losing_record() -> None:
    model = _CorrectingModel(['{"answers": []}'])
    row = _row()
    row[COL_PRIVACY_QA_REANSWER] = {"answers": [_answer()]}
    result = asyncio.run(_make_privacy_answer_column("judge")(row, None, {"judge": model}))
    assert model.calls == 4
    assert result is row
    assert result[COL_PRIVACY_QA_REANSWER] is None
    assert "unavailable" in result[COL_PRIVACY_QA_ERROR]
    assert result[COL_REPLACED_TEXT] == "Maya arrived."


def test_model_failure_does_not_expose_exception_content(caplog: pytest.LogCaptureFixture) -> None:
    model = Mock()
    model.agenerate = AsyncMock(side_effect=RuntimeError("SECRET ORIGINAL DATA"))
    result = asyncio.run(_make_privacy_answer_column("judge")(_row(), None, {"judge": model}))
    assert result[COL_PRIVACY_QA_REANSWER] is None
    assert "SECRET ORIGINAL DATA" not in str(result[COL_PRIVACY_QA_ERROR])
    assert "SECRET ORIGINAL DATA" not in caplog.text


def test_empty_questions_skip_model_and_clear_previous_error() -> None:
    row = {COL_PRIVACY_QA: {"items": []}, COL_PRIVACY_QA_ERROR: "previous failure"}
    result = asyncio.run(_make_privacy_answer_column("judge")(row, None, {}))
    assert result[COL_PRIVACY_QA_REANSWER] == {"answers": []}
    assert result[COL_PRIVACY_QA_ERROR] is None


def test_workflow_declares_replaced_evidence_and_error_output() -> None:
    columns = ReplacePrivacyAnswerWorkflow().columns(evaluator_alias="judge")
    assert len(columns) == 1
    column = columns[0]
    assert isinstance(column, CustomColumnConfig)
    assert column.name == COL_PRIVACY_QA_REANSWER
    assert set(column.required_columns) == {COL_PRIVACY_QA, COL_REPLACED_TEXT}
    assert column.side_effect_columns == [COL_PRIVACY_QA_ERROR]


@pytest.mark.parametrize(
    "batch_size, expected_batches",
    [
        (None, [[11, 22, 33, 44, 55]]),
        (2, [[11, 22], [33, 44], [55]]),
        (1, [[11], [22], [33], [44], [55]]),
        (10, [[11, 22, 33, 44, 55]]),
    ],
)
def test_batching_preserves_ids_full_text_and_question_order(
    batch_size: int | None, expected_batches: list[list[int]]
) -> None:
    row = _row()
    question = row[COL_PRIVACY_QA]["items"][0]
    row[COL_PRIVACY_QA] = {"items": [{**question, "id": item_id} for item_id in [11, 22, 33, 44, 55]]}
    model = Mock()
    model.agenerate = AsyncMock(
        side_effect=[
            (PrivacyAnswersSchema.model_validate({"answers": [_answer(item_id) for item_id in reversed(batch)]}), None)
            for batch in expected_batches
        ]
    )
    result = asyncio.run(
        _make_privacy_answer_column("judge")(row, PrivacyAnswerParams(batch_size=batch_size), {"judge": model})
    )
    assert [answer["id"] for answer in result[COL_PRIVACY_QA_REANSWER]["answers"]] == [11, 22, 33, 44, 55]
    assert model.agenerate.call_count == len(expected_batches)
    for call, ids in zip(model.agenerate.call_args_list, expected_batches, strict=True):
        prompt = call.kwargs["prompt"]
        assert "Maya arrived." in prompt
        assert "ORIGINAL CONTEXT MUST NOT BE INCLUDED" not in prompt
        template = json.loads(prompt.split("<answer_template>")[1].split("</answer_template>")[0])
        assert [item["id"] for item in template["answers"]] == ids
        assert all(item["confidence"] is None for item in template["answers"])
        assert "text that has undergone entity replacement or redaction" in prompt
        assert call.kwargs["max_correction_steps"] == 3
        assert call.kwargs["parser"](json.dumps({"answers": [_answer(item_id) for item_id in ids]})).answers
        with pytest.raises(ParserException):
            call.kwargs["parser"]('{"answers": []}')


def test_later_batch_failure_discards_partial_answers() -> None:
    row = _row()
    question = row[COL_PRIVACY_QA]["items"][0]
    row[COL_PRIVACY_QA] = {"items": [{**question, "id": item_id} for item_id in [1, 2, 3]]}
    row[COL_PRIVACY_QA_REANSWER] = {"answers": [_answer(99)]}
    model = Mock()
    model.agenerate = AsyncMock(
        side_effect=[
            (PrivacyAnswersSchema.model_validate({"answers": [_answer(1)]}), None),
            RuntimeError("Synthetic second-batch failure"),
        ]
    )
    result = asyncio.run(_make_privacy_answer_column("judge")(row, PrivacyAnswerParams(batch_size=1), {"judge": model}))
    assert model.agenerate.call_count == 2
    assert result[COL_PRIVACY_QA_REANSWER] is None
    assert "unavailable" in result[COL_PRIVACY_QA_ERROR]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "5"])
def test_batch_size_requires_a_positive_integer(batch_size: object) -> None:
    with pytest.raises(ValidationError):
        EvaluateConfig.model_validate({"privacy_qa_batch_size": batch_size})


def test_replace_uses_dedicated_prompt_and_unfilled_confidence() -> None:
    model = _CorrectingModel([json.dumps({"answers": [_answer()]})])
    asyncio.run(_make_privacy_answer_column("judge")(_row(), None, {"judge": model}))
    assert "text that has undergone entity replacement or redaction" in model.prompt
    assert "It does not reveal the hidden value." in model.prompt
    assert "A matching string used in a clearly different role is not sufficient." in model.prompt
    assert '"North America" alone does not establish US.' in model.prompt
    assert "citizenship, or nationality. Those require evidence" in model.prompt
    template = json.loads(model.prompt.split("<answer_template>")[1].split("</answer_template>")[0])
    assert template["answers"][0]["confidence"] is None
    assert template["answers"][0]["id"] == 1


def test_rewrite_still_uses_original_prompt_and_template() -> None:
    from anonymizer.engine.constants import COL_REWRITTEN_TEXT
    from anonymizer.engine.rewrite.evaluate import _make_privacy_reanswer_column

    row = _row()
    row[COL_REWRITTEN_TEXT] = "Maya arrived."
    model = _CorrectingModel([json.dumps({"answers": [_answer()]})])
    asyncio.run(_make_privacy_reanswer_column("judge")(row, None, {"judge": model}))
    assert 'You are a privacy auditor. Read the text and answer each question with "yes" or "no".' in model.prompt
    assert "Weak or indirect signals still count" in model.prompt
    assert "undergo" not in model.prompt
    template = json.loads(model.prompt.split("<answer_template>")[1].split("</answer_template>")[0])
    assert template["answers"][0]["confidence"] == 1.0
    assert "ORIGINAL CONTEXT MUST NOT BE INCLUDED" not in model.prompt
