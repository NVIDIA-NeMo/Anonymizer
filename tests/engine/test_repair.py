# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.models import ModelConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_LEAKAGE_MASS,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.rewrite.repair import (
    RepairParams,
    RepairWorkflow,
    _format_protection_block,
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

_STUB_DISPOSITION = {
    "sensitivity_disposition": [
        {
            "id": 1,
            "source": "tagged",
            "category": "direct_identifier",
            "sensitivity": "high",
            "entity_label": "first_name",
            "entity_value": "Alice",
            "needs_protection": True,
            "protection_reason": "Direct identifier that enables re-identification",
            "protection_method_suggestion": "replace",
            "combined_risk_level": "high",
        },
        {
            "id": 2,
            "source": "tagged",
            "category": "quasi_identifier",
            "sensitivity": "low",
            "entity_label": "city",
            "entity_value": "Seattle",
            "needs_protection": False,
            "protection_method_suggestion": "leave_as_is",
            "protection_reason": "Low risk, not identifying alone",
            "combined_risk_level": "low",
        },
    ]
}

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
# _format_protection_block
# ---------------------------------------------------------------------------


class TestFormatProtectionBlock:
    def test_includes_protected_entities(self) -> None:
        row = {COL_SENSITIVITY_DISPOSITION: _STUB_DISPOSITION}
        result = _format_protection_block(row)
        assert "Alice" in result
        assert "replace" in result
        assert "Seattle" not in result

    def test_empty_when_none_protected(self) -> None:
        row = {
            COL_SENSITIVITY_DISPOSITION: {
                "sensitivity_disposition": [
                    {
                        "id": 1,
                        "source": "tagged",
                        "category": "quasi_identifier",
                        "sensitivity": "low",
                        "entity_label": "city",
                        "entity_value": "Portland",
                        "needs_protection": False,
                        "protection_method_suggestion": "leave_as_is",
                        "protection_reason": "Low risk location",
                        "combined_risk_level": "low",
                    },
                ]
            }
        }
        assert _format_protection_block(row) == ""


# ---------------------------------------------------------------------------
# _render_repair_prompt
# ---------------------------------------------------------------------------


class TestRenderRepairPrompt:
    def test_contains_key_sections(self) -> None:
        row = {
            COL_SENSITIVITY_DISPOSITION: _STUB_DISPOSITION,
            COL_TEXT: "Alice lives in Seattle.",
            COL_REWRITTEN_TEXT: "Bob lives in Portland.",
            COL_REPLACEMENT_MAP_FOR_PROMPT: {
                "replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Bob"}]
            },
            COL_LEAKAGE_MASS: 1.0,
            COL_ANY_HIGH_LEAKED: True,
            COL_UTILITY_SCORE: 0.85,
            COL_PRIVACY_QA_REANSWER: {
                "answers": [{"id": 1, "answer": "yes", "confidence": 0.9, "reason": "Name remains explicit"}]
            },
            COL_PRIVACY_QA: _STUB_PRIVACY_QA,
            "_leaked_privacy_items": '- [HIGH] first_name: "Alice"',
        }
        params = RepairParams(privacy_goal_str=_STUB_PRIVACY_GOAL.to_prompt_string(), max_privacy_leak=1.0)
        result = _render_repair_prompt(row, params)
        assert "<privacy_goal>" in result
        assert "<protection_decisions>" in result
        assert "Alice" in result
        assert "WARNING" in result
        assert "0.85" in result

    def test_no_high_warn_when_no_high_leak(self) -> None:
        row = {
            COL_SENSITIVITY_DISPOSITION: _STUB_DISPOSITION,
            COL_TEXT: "",
            COL_REWRITTEN_TEXT: "",
            COL_REPLACEMENT_MAP_FOR_PROMPT: "",
            COL_LEAKAGE_MASS: 0.5,
            COL_ANY_HIGH_LEAKED: False,
            COL_UTILITY_SCORE: 0.9,
            "_leaked_privacy_items": "",
        }
        params = RepairParams(privacy_goal_str=_STUB_PRIVACY_GOAL.to_prompt_string(), max_privacy_leak=1.0)
        result = _render_repair_prompt(row, params)
        assert "WARNING" not in result

    def test_accepts_numpy_array_replacement_map_payload(self) -> None:
        row = {
            COL_SENSITIVITY_DISPOSITION: _STUB_DISPOSITION,
            COL_TEXT: "Alice lives in Seattle.",
            COL_REWRITTEN_TEXT: "A person lives in a city.",
            COL_REPLACEMENT_MAP_FOR_PROMPT: {"replacements": np.array([], dtype=object)},
            COL_LEAKAGE_MASS: 1.0,
            COL_ANY_HIGH_LEAKED: False,
            COL_UTILITY_SCORE: 0.85,
            "_leaked_privacy_items": "",
        }
        params = RepairParams(privacy_goal_str=_STUB_PRIVACY_GOAL.to_prompt_string(), max_privacy_leak=1.0)
        result = _render_repair_prompt(row, params)
        assert "<replacement_map>" in result


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
