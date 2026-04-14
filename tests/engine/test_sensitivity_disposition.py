# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import LLMStructuredColumnConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DOMAIN_SUPPLEMENT_PRIVACY,
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAGGED_TEXT,
    _jinja,
)
from anonymizer.engine.rewrite.sensitivity_disposition import (
    SensitivityDispositionWorkflow,
    _get_sensitivity_disposition_prompt,
)

_STUB_PRIVACY_GOAL = PrivacyGoal(
    protect="Protect direct identifiers and quasi-identifier combinations from re-identification.",
    preserve="General utility and semantic meaning of the original text.",
)


def test_columns_uses_disposition_analyzer_alias(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    cols = SensitivityDispositionWorkflow().columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
    )
    assert len(cols) == 1
    assert isinstance(cols[0], LLMStructuredColumnConfig)
    assert cols[0].model_alias == stub_rewrite_model_selection.disposition_analyzer
    assert cols[0].name == COL_SENSITIVITY_DISPOSITION


def test_privacy_goal_interpolated_into_prompt() -> None:
    prompt = _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL)
    assert "Protect direct identifiers and quasi-identifier combinations" in prompt
    assert "General utility and semantic meaning" in prompt


def test_prompt_includes_data_summary_when_provided() -> None:
    prompt = _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL, data_summary="Medical visit notes")
    assert "Medical visit notes" in prompt
    assert "Dataset description:" in prompt


def test_prompt_omits_data_summary_when_none() -> None:
    prompt = _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL, data_summary=None)
    assert "Dataset description:" not in prompt
    assert "<data_context>" not in prompt


def test_prompt_references_required_columns() -> None:
    prompt = _get_sensitivity_disposition_prompt(_STUB_PRIVACY_GOAL)
    assert _jinja(COL_TAGGED_TEXT) in prompt
    assert _jinja(COL_ENTITIES_BY_VALUE) in prompt
    assert _jinja(COL_LATENT_ENTITIES) in prompt
    assert _jinja(COL_DOMAIN_SUPPLEMENT_PRIVACY) in prompt
    assert _jinja("_domain", key="domain") in prompt
