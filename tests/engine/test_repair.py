# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.models import ModelConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_LEAKAGE_MASS,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.rewrite.repair import (
    RepairParams,
    RepairWorkflow,
    _leaked_items_text,
    _render_repair_prompt,
)
from anonymizer.engine.schemas.rewrite import (
    PrivacyAnswerItemSchema,
    PrivacyQAPairsSchema,
)

_STUB_PRIVACY_GOAL = PrivacyGoal(
    protect="Direct identifiers, quasi-identifier combinations, and latent inferences",
    preserve="General utility, content quality, and semantic meaning of the original text",
)

_STUB_PRIVACY_QA = {
    "items": [
        {
            "id": 1,
            "question": "Can the first_name 'Alice' be deduced?",
            "sensitivity": "high",
            "entity_label": "first_name",
            "entity_value": "Alice",
            "category": "direct_identifier",
        },
    ]
}


# ---------------------------------------------------------------------------
# _leaked_items_text
# ---------------------------------------------------------------------------


class TestLeakedItemsText:
    def test_leaked_item_formatted(self) -> None:
        answers = [PrivacyAnswerItemSchema(id=1, answer="yes", confidence=0.9, reason="Name remains explicit")]
        qa = PrivacyQAPairsSchema.model_validate(_STUB_PRIVACY_QA)
        result = _leaked_items_text(answers, qa)
        assert "[HIGH]" in result
        assert "first_name" in result
        assert "Alice" in result
        assert "confidence_leakage_occurred: 0.90" in result
        assert "reason: Name remains explicit" in result

    def test_no_leaks_returns_empty(self) -> None:
        answers = [PrivacyAnswerItemSchema(id=1, answer="no", confidence=0.0, reason="Not inferable")]
        qa = PrivacyQAPairsSchema.model_validate(_STUB_PRIVACY_QA)
        assert _leaked_items_text(answers, qa) == ""


# ---------------------------------------------------------------------------
# _render_repair_prompt
# ---------------------------------------------------------------------------


class TestRenderRepairPrompt:
    def test_contains_key_sections(self) -> None:
        row = {
            COL_TEXT: "Alice lives in Seattle.",
            COL_REWRITTEN_TEXT: "Bob lives in Portland.",
            COL_LEAKAGE_MASS: 1.0,
            COL_ANY_HIGH_LEAKED: True,
            COL_UTILITY_SCORE: 0.85,
            "_leaked_privacy_items": '- [HIGH] first_name: "Alice"',
        }
        params = RepairParams(privacy_goal_str=_STUB_PRIVACY_GOAL.to_prompt_string(), max_privacy_leak=1.0)
        result = _render_repair_prompt(row, params)
        assert "<privacy_goal>" in result
        assert "<adversarial_goal>" in result
        assert "Alice" in result
        assert "WARNING" in result
        assert "0.85" in result

    def test_no_high_warn_when_no_high_leak(self) -> None:
        row = {
            COL_TEXT: "",
            COL_REWRITTEN_TEXT: "",
            COL_LEAKAGE_MASS: 0.5,
            COL_ANY_HIGH_LEAKED: False,
            COL_UTILITY_SCORE: 0.9,
            "_leaked_privacy_items": "",
        }
        params = RepairParams(privacy_goal_str=_STUB_PRIVACY_GOAL.to_prompt_string(), max_privacy_leak=1.0)
        result = _render_repair_prompt(row, params)
        assert "WARNING" not in result


# ---------------------------------------------------------------------------
# RepairWorkflow.columns()
# ---------------------------------------------------------------------------


def test_repair_columns_pipeline(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    wf = RepairWorkflow(adapter=Mock())
    cols = wf.columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
        effective_threshold=1.0,
    )

    assert len(cols) == 2
    assert isinstance(cols[0], CustomColumnConfig)
    assert isinstance(cols[1], CustomColumnConfig)
    assert cols[1].name == COL_REWRITTEN_TEXT_NEXT
