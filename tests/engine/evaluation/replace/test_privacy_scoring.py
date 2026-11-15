# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from data_designer.config.column_configs import CustomColumnConfig
from pydantic import ValidationError

from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_LEAKAGE_MASS,
    COL_PRIVACY_LEAKAGE_CONTRIBUTIONS,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_ERROR,
    COL_PRIVACY_QA_REANSWER,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.evaluation.replace.privacy_scoring import (
    ReplacePrivacyScoringWorkflow,
    _compute_privacy_score_column,
    compute_privacy_leakage_score,
)
from anonymizer.engine.schemas import PrivacyAnswer, PrivacyAnswersSchema, PrivacyQAPairsSchema


def _questions() -> PrivacyQAPairsSchema:
    return PrivacyQAPairsSchema.model_validate(
        {
            "items": [
                {
                    "id": item_id,
                    "question": f"Can target {item_id} be deduced from the replaced text?",
                    "entity_value": value,
                    "entity_label": label,
                    "sensitivity": sensitivity,
                    "category": category,
                }
                for item_id, value, label, sensitivity, category in [
                    (1, "Alice", "first_name", "high", "direct_identifier"),
                    (2, "Denver", "city", "medium", "quasi_identifier"),
                    (3, "USA", "country", "low", "quasi_identifier"),
                    (4, "Alice", "full_name", "high", "direct_identifier"),
                ]
            ]
        }
    )


def _answers() -> PrivacyAnswersSchema:
    return PrivacyAnswersSchema.model_validate(
        {
            "answers": [
                {
                    "id": item_id,
                    "answer": answer,
                    "confidence": confidence,
                    "reason": f"Reason for target {item_id}",
                    "evidence": [f"Evidence for target {item_id}"],
                }
                for item_id, answer, confidence in [(4, "no", 1.0), (3, "yes", 0.4), (1, "yes", 0.8), (2, "yes", 0.5)]
            ]
        }
    )


def test_score_reconciles_all_sensitivities_and_joins_reordered_answers() -> None:
    qa, answers = _questions(), _answers()
    score = compute_privacy_leakage_score(qa, answers)
    assert score.leakage_mass == pytest.approx(1.22)
    assert [entity.id for entity in score.entities] == [1, 2, 3, 4]
    assert [entity.leakage_contribution for entity in score.entities] == pytest.approx([0.8, 0.3, 0.12, 0.0])
    assert sum(entity.leakage_contribution for entity in score.entities) == pytest.approx(score.leakage_mass)
    answers_by_id = {answer.id: answer for answer in answers.answers}
    for entity, question in zip(score.entities, qa.items, strict=True):
        assert entity.entity_value == question.entity_value
        assert entity.entity_label == question.entity_label
        assert entity.sensitivity == question.sensitivity
        assert entity.category == question.category
        assert entity.reason == answers_by_id[entity.id].reason
        assert entity.evidence == answers_by_id[entity.id].evidence


def test_empty_evaluation_has_zero_mass_and_no_contributions() -> None:
    score = compute_privacy_leakage_score(PrivacyQAPairsSchema(items=[]), PrivacyAnswersSchema(answers=[]))
    assert score.leakage_mass == 0.0
    assert score.weighted_leakage_rate == 0.0
    assert score.entities == []


def test_all_negative_answers_have_zero_mass() -> None:
    answers = _answers()
    for answer in answers.answers:
        answer.answer = PrivacyAnswer.no
    score = compute_privacy_leakage_score(_questions(), answers)
    assert score.leakage_mass == 0.0
    assert score.weighted_leakage_rate == 0.0
    assert all(entity.leakage_contribution == 0.0 for entity in score.entities)


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate"])
def test_scoring_rejects_incomplete_or_ambiguous_coverage(kind: str) -> None:
    answers = _answers().model_dump(mode="json")["answers"]
    if kind == "missing":
        answers.pop()
    elif kind == "extra":
        answers.append({**answers[0], "id": 99})
    else:
        answers.append(answers[0])
    with pytest.raises(ValidationError):
        compute_privacy_leakage_score(_questions(), PrivacyAnswersSchema.model_validate({"answers": answers}))


def test_scoring_rejects_duplicate_question_ids() -> None:
    qa = _questions()
    qa.items[1].id = qa.items[0].id
    with pytest.raises(ValueError, match="unique IDs"):
        compute_privacy_leakage_score(qa, _answers())


@pytest.mark.parametrize("serialized", [False, True])
def test_score_column_handles_payloads_without_adding_other_metrics(serialized: bool) -> None:
    qa = _questions().model_dump(mode="json")
    answers = _answers().model_dump(mode="json")
    row = {
        COL_PRIVACY_QA: json.dumps(qa) if serialized else qa,
        COL_PRIVACY_QA_REANSWER: json.dumps(answers) if serialized else answers,
        COL_PRIVACY_QA_ERROR: None,
    }
    result = _compute_privacy_score_column(row)
    assert result[COL_LEAKAGE_MASS] == pytest.approx(1.22)
    assert len(result[COL_PRIVACY_LEAKAGE_CONTRIBUTIONS]) == 4
    assert result[COL_WEIGHTED_LEAKAGE_RATE] == pytest.approx(1.22 / 2.9)
    assert COL_ANY_HIGH_LEAKED not in result


def test_unavailable_answers_clear_old_scores_and_preserve_error() -> None:
    row = {
        COL_PRIVACY_QA: _questions().model_dump(),
        COL_PRIVACY_QA_REANSWER: None,
        COL_PRIVACY_QA_ERROR: "Privacy QA evaluation failed; answers unavailable.",
        COL_LEAKAGE_MASS: 0.0,
        COL_PRIVACY_LEAKAGE_CONTRIBUTIONS: [],
    }
    result = _compute_privacy_score_column(row)
    assert result[COL_LEAKAGE_MASS] is None
    assert result[COL_WEIGHTED_LEAKAGE_RATE] is None
    assert result[COL_PRIVACY_LEAKAGE_CONTRIBUTIONS] is None
    assert result[COL_PRIVACY_QA_ERROR] == row[COL_PRIVACY_QA_ERROR]


@pytest.mark.parametrize("answers", [None, {"answers": []}, "invalid JSON", {"answers": [{"id": 1, "answer": None}]}])
def test_invalid_answers_never_become_zero_leakage(answers: object) -> None:
    row = {
        COL_PRIVACY_QA: _questions().model_dump(),
        COL_PRIVACY_QA_REANSWER: answers,
        COL_PRIVACY_QA_ERROR: None,
    }
    result = _compute_privacy_score_column(row)
    assert result[COL_LEAKAGE_MASS] is None
    assert result[COL_WEIGHTED_LEAKAGE_RATE] is None
    assert result[COL_PRIVACY_LEAKAGE_CONTRIBUTIONS] is None


def test_workflow_declares_mass_and_contributions_only() -> None:
    columns = ReplacePrivacyScoringWorkflow().columns()
    assert len(columns) == 1
    column = columns[0]
    assert isinstance(column, CustomColumnConfig)
    assert column.name == COL_LEAKAGE_MASS
    assert column.side_effect_columns == [COL_PRIVACY_LEAKAGE_CONTRIBUTIONS, COL_WEIGHTED_LEAKAGE_RATE]
