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
    _reconstruct_full_disposition_column,
)

_STUB_PRIVACY_GOAL = PrivacyGoal(
    protect="Protect direct identifiers and quasi-identifier combinations from re-identification.",
    preserve="General utility and semantic meaning of the original text.",
)


def test_columns_uses_disposition_analyzer_alias(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    """Workflow now returns two columns: an LLM column with the loose
    wire-contract schema writing COL_SIMPLE_DISPOSITION, and a pure-python
    reconstruction column writing COL_SENSITIVITY_DISPOSITION."""
    cols = SensitivityDispositionWorkflow().columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
    )
    assert len(cols) == 2
    # Step 1: LLM call with the simple (loose) output format.
    assert isinstance(cols[0], LLMStructuredColumnConfig)
    assert cols[0].model_alias == stub_rewrite_model_selection.disposition_analyzer
    assert cols[0].name == COL_SIMPLE_DISPOSITION
    # Step 2: deterministic server-side reconstruction into the strict schema.
    assert isinstance(cols[1], CustomColumnConfig)
    assert cols[1].name == COL_SENSITIVITY_DISPOSITION


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


def test_reconstruct_column_handles_empty_simple_result(caplog) -> None:
    """When the simple wire payload is empty (no items at all), the
    reconstructor would produce an empty list which trips
    SensitivityDispositionSchema.min_length=1 and raises ValidationError.
    The column generator must short-circuit and emit an empty disposition
    instead of letting the exception propagate (which would drop the row
    — exactly the failure mode this PR is meant to prevent)."""
    row = {
        COL_SIMPLE_DISPOSITION: {"sensitivity_disposition": []},
        COL_ENTITIES_BY_VALUE: [{"value": "Alice", "labels": ["first_name"]}],
        COL_LATENT_ENTITIES: [],
    }
    with caplog.at_level("WARNING"):
        result = _reconstruct_full_disposition_column(row)
    assert result is row
    assert result[COL_SENSITIVITY_DISPOSITION] == {"sensitivity_disposition": []}
    assert any("empty" in r.message.lower() for r in caplog.records)


def test_reconstruct_column_handles_orphans_only(caplog) -> None:
    """When every SimpleDispositionItem id is out-of-range AND the model
    omitted entity_label/value echoes (orphans-only response), the
    reconstructor skips them all -> empty result -> would raise
    ValidationError without the guard. Must emit empty disposition."""
    row = {
        # ids 100-101 with no echoes; context has only 1 entity (id=1).
        COL_SIMPLE_DISPOSITION: {
            "sensitivity_disposition": [
                {
                    "id": 100,
                    "category": "direct_identifier",
                    "sensitivity": "high",
                    "protection_method_suggestion": "replace",
                    "protection_reason": "Ten+ chars.",
                },
                {
                    "id": 101,
                    "category": "quasi_identifier",
                    "sensitivity": "low",
                    "protection_method_suggestion": "leave_as_is",
                    "protection_reason": "Ten+ chars.",
                },
            ]
        },
        COL_ENTITIES_BY_VALUE: [{"value": "Alice", "labels": ["first_name"]}],
        COL_LATENT_ENTITIES: [],
    }
    with caplog.at_level("WARNING"):
        result = _reconstruct_full_disposition_column(row)
    assert result[COL_SENSITIVITY_DISPOSITION] == {"sensitivity_disposition": []}
