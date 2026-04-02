# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pytest
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.models import ModelConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import SENSITIVITY_WEIGHTS, EvaluationCriteria
from anonymizer.engine.constants import COL_UTILITY_SCORE
from anonymizer.engine.rewrite.evaluate import (
    EvaluateWorkflow,
    _normalize_answer_items,
    compute_any_high_leaked,
    compute_leakage_mass,
    compute_utility_score,
    determine_repair_needs,
)
from anonymizer.engine.schemas.rewrite import (
    PrivacyAnswerItemSchema,
    PrivacyQAPairsSchema,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def privacy_qa_high_and_low() -> dict:
    return {
        "items": [
            {
                "id": 1,
                "question": "Can the first_name 'Alice' be deduced?",
                "sensitivity": "high",
                "entity_label": "first_name",
                "entity_value": "Alice",
                "category": "direct_identifier",
            },
            {
                "id": 2,
                "question": "Can the city 'Seattle' be deduced?",
                "sensitivity": "low",
                "entity_label": "city",
                "entity_value": "Seattle",
                "category": "quasi_identifier",
            },
        ]
    }


def _privacy_answer(
    item_id: int, answer: str, confidence: float = 1.0, reason: str = "evidence in text"
) -> PrivacyAnswerItemSchema:
    return PrivacyAnswerItemSchema(id=item_id, answer=answer, confidence=confidence, reason=reason)


# ---------------------------------------------------------------------------
# compute_leakage_mass
# ---------------------------------------------------------------------------


class TestComputeLeakageMass:
    def test_no_leaks(self, privacy_qa_high_and_low: dict) -> None:
        answers = [_privacy_answer(1, "no"), _privacy_answer(2, "no")]
        qa = PrivacyQAPairsSchema.model_validate(privacy_qa_high_and_low)
        assert compute_leakage_mass(answers, qa, dict(SENSITIVITY_WEIGHTS)) == 0.0

    def test_high_leak_only(self, privacy_qa_high_and_low: dict) -> None:
        answers = [_privacy_answer(1, "yes"), _privacy_answer(2, "no")]
        qa = PrivacyQAPairsSchema.model_validate(privacy_qa_high_and_low)
        assert compute_leakage_mass(answers, qa, dict(SENSITIVITY_WEIGHTS)) == 1.0

    def test_low_leak_only(self, privacy_qa_high_and_low: dict) -> None:
        answers = [_privacy_answer(1, "no"), _privacy_answer(2, "yes")]
        qa = PrivacyQAPairsSchema.model_validate(privacy_qa_high_and_low)
        assert compute_leakage_mass(answers, qa, dict(SENSITIVITY_WEIGHTS)) == 0.3

    def test_both_leaked(self, privacy_qa_high_and_low: dict) -> None:
        answers = [_privacy_answer(1, "yes"), _privacy_answer(2, "yes")]
        qa = PrivacyQAPairsSchema.model_validate(privacy_qa_high_and_low)
        assert compute_leakage_mass(answers, qa, dict(SENSITIVITY_WEIGHTS)) == pytest.approx(1.3)

    def test_custom_weights(self, privacy_qa_high_and_low: dict) -> None:
        answers = [_privacy_answer(1, "yes"), _privacy_answer(2, "yes")]
        qa = PrivacyQAPairsSchema.model_validate(privacy_qa_high_and_low)
        assert compute_leakage_mass(answers, qa, {"high": 2.0, "medium": 1.0, "low": 0.5}) == pytest.approx(2.5)

    def test_empty_answers(self, privacy_qa_high_and_low: dict) -> None:
        qa = PrivacyQAPairsSchema.model_validate(privacy_qa_high_and_low)
        assert compute_leakage_mass([], qa, dict(SENSITIVITY_WEIGHTS)) == 0.0

    def test_confidence_weights_leakage_mass(self, privacy_qa_high_and_low: dict) -> None:
        answers = [_privacy_answer(1, "yes", confidence=0.4), _privacy_answer(2, "yes", confidence=0.5)]
        qa = PrivacyQAPairsSchema.model_validate(privacy_qa_high_and_low)
        assert compute_leakage_mass(answers, qa, dict(SENSITIVITY_WEIGHTS)) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# compute_any_high_leaked
# ---------------------------------------------------------------------------


class TestComputeAnyHighLeaked:
    def test_high_leaked(self, privacy_qa_high_and_low: dict) -> None:
        answers = [_privacy_answer(1, "yes"), _privacy_answer(2, "no")]
        qa = PrivacyQAPairsSchema.model_validate(privacy_qa_high_and_low)
        assert compute_any_high_leaked(answers, qa) is True

    def test_only_low_leaked(self, privacy_qa_high_and_low: dict) -> None:
        answers = [_privacy_answer(1, "no"), _privacy_answer(2, "yes")]
        qa = PrivacyQAPairsSchema.model_validate(privacy_qa_high_and_low)
        assert compute_any_high_leaked(answers, qa) is False

    def test_no_leaks(self, privacy_qa_high_and_low: dict) -> None:
        answers = [_privacy_answer(1, "no"), _privacy_answer(2, "no")]
        qa = PrivacyQAPairsSchema.model_validate(privacy_qa_high_and_low)
        assert compute_any_high_leaked(answers, qa) is False


# ---------------------------------------------------------------------------
# compute_utility_score
# ---------------------------------------------------------------------------


class TestComputeUtilityScore:
    def test_perfect_scores(self) -> None:
        assert compute_utility_score([1.0, 1.0, 1.0]) == 1.0

    def test_mixed_scores(self) -> None:
        assert compute_utility_score([0.8, 0.6]) == pytest.approx(0.7)

    def test_empty(self) -> None:
        assert compute_utility_score([]) == 0.0

    def test_single_score(self) -> None:
        assert compute_utility_score([0.5]) == 0.5


# ---------------------------------------------------------------------------
# determine_repair_needs
# ---------------------------------------------------------------------------


class TestDetermineRepairNeeds:
    def test_high_leak_triggers_repair(self) -> None:
        assert determine_repair_needs(
            any_high_leaked=True,
            leakage_mass=0.0,
            auto_repair_privacy=True,
            repair_any_high_leak=True,
            effective_threshold=1.0,
        )

    def test_leakage_mass_above_threshold(self) -> None:
        assert determine_repair_needs(
            any_high_leaked=False,
            leakage_mass=1.5,
            auto_repair_privacy=True,
            repair_any_high_leak=True,
            effective_threshold=1.0,
        )

    def test_below_threshold_no_repair(self) -> None:
        assert not determine_repair_needs(
            any_high_leaked=False,
            leakage_mass=0.3,
            auto_repair_privacy=True,
            repair_any_high_leak=True,
            effective_threshold=1.0,
        )

    def test_auto_repair_disabled(self) -> None:
        assert not determine_repair_needs(
            any_high_leaked=True,
            leakage_mass=5.0,
            auto_repair_privacy=False,
            repair_any_high_leak=True,
            effective_threshold=1.0,
        )

    def test_repair_any_high_leak_false(self) -> None:
        assert not determine_repair_needs(
            any_high_leaked=True,
            leakage_mass=0.3,
            auto_repair_privacy=True,
            repair_any_high_leak=False,
            effective_threshold=1.0,
        )

    def test_domain_adjusts_threshold(self) -> None:
        evaluation = EvaluationCriteria(auto_adjust_by_domain=True, repair_any_high_leak=False)
        assert determine_repair_needs(
            any_high_leaked=False,
            leakage_mass=0.8,
            auto_repair_privacy=True,
            repair_any_high_leak=False,
            effective_threshold=evaluation.get_effective_threshold("medical"),
        )
        assert not determine_repair_needs(
            any_high_leaked=False,
            leakage_mass=0.8,
            auto_repair_privacy=True,
            repair_any_high_leak=False,
            effective_threshold=evaluation.get_effective_threshold("social_media"),
        )


# ---------------------------------------------------------------------------
# EvaluateWorkflow.columns()
# ---------------------------------------------------------------------------


def test_evaluate_columns_pipeline(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    wf = EvaluateWorkflow(adapter=Mock())
    cols = wf.columns(
        selected_models=stub_rewrite_model_selection,
        evaluation=EvaluationCriteria(),
    )

    assert len(cols) == 5
    for col in cols:
        assert isinstance(col, CustomColumnConfig)
    assert cols[3].name == COL_UTILITY_SCORE


def test_normalize_answer_items_fills_missing_privacy_answers_conservatively() -> None:
    normalized = _normalize_answer_items(
        [{"id": 1, "answer": "no", "confidence": 0.0, "reason": "not inferable"}],
        expected_ids=[1, 2],
        label="privacy answer",
        default_item_factory=lambda item_id: {
            "id": item_id,
            "answer": "yes",
            "confidence": 1.0,
            "reason": "Model omitted this item; defaulted to highest-confidence leak.",
        },
    )
    assert normalized == [
        {"id": 1, "answer": "no", "confidence": 0.0, "reason": "not inferable"},
        {
            "id": 2,
            "answer": "yes",
            "confidence": 1.0,
            "reason": "Model omitted this item; defaulted to highest-confidence leak.",
        },
    ]


def test_normalize_answer_items_drops_extra_and_duplicate_ids() -> None:
    normalized = _normalize_answer_items(
        [
            {"id": 1, "answer": "first"},
            {"id": 1, "answer": "duplicate"},
            {"id": 99, "answer": "extra"},
        ],
        expected_ids=[1, 2],
        label="quality answer",
        default_item_factory=lambda item_id: {"id": item_id, "answer": "unknown"},
    )
    assert normalized == [{"id": 1, "answer": "first"}, {"id": 2, "answer": "unknown"}]
