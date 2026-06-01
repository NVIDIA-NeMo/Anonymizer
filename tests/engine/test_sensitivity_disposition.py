# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DOMAIN_SUPPLEMENT_PRIVACY,
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_SENSITIVITY_DISPOSITION,
    COL_SIMPLE_DISPOSITION,
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


def test_columns_emits_two_step_pipeline(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    """Post-#130: the disposition workflow emits a 2-step pipeline — a
    ``SimpleDispositionResult`` LLM column (loose wire, dropped from preview)
    + a deterministic CustomColumnConfig that reconstructs the strict
    ``SensitivityDispositionSchema`` server-side."""
    cols = SensitivityDispositionWorkflow().columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
    )
    assert len(cols) == 2

    llm_col, recon_col = cols
    assert isinstance(llm_col, LLMStructuredColumnConfig)
    assert llm_col.name == COL_SIMPLE_DISPOSITION
    assert llm_col.model_alias == stub_rewrite_model_selection.disposition_analyzer
    # DataDesigner serializes ``output_format`` to its JSON schema at
    # construction time, so we assert on the schema's ``$defs`` rather than
    # identity. Looking for ``SimpleDispositionItem`` (the loose wire item)
    # is a tighter check than just "some schema is set" — it confirms we
    # are passing the loose wrapper and not the strict
    # ``SensitivityDispositionSchema``.
    assert isinstance(llm_col.output_format, dict)
    assert "SimpleDispositionItem" in llm_col.output_format.get("$defs", {})
    assert llm_col.drop is True

    assert isinstance(recon_col, CustomColumnConfig)
    assert recon_col.name == COL_SENSITIVITY_DISPOSITION


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
