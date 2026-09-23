# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.ndd.adapter import WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import (
    _INTERNAL_COLUMNS,
    LlmReplaceWorkflow,
    _filter_replacement_map_to_input_entities,
)
from anonymizer.engine.schemas.detection import EntitiesByValueSchema


def test_generate_map_only_returns_replacement_map(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    """Workflow operates on plain DataFrames; pipeline metadata is the orchestrator's job."""
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                "text": ["Alice works at Acme"],
                COL_REPLACEMENT_MAP: [{"replacements": []}],
                "_anonymizer_row_order": [0],
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            "text": ["Alice works at Acme"],
            "_entities_by_value": [[{"value": "Alice", "labels": ["first_name"]}]],
            "tagged_text": ["<<SENSITIVE:first_name>>Alice<</SENSITIVE:first_name>> works at Acme"],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert COL_REPLACEMENT_MAP in result.dataframe.columns


def test_generate_map_only_skips_llm_for_rows_without_entities(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = Mock()
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["CT guided biopsy revealed a fatty mass which was diagnosed as lipoma."],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}],
            "tagged_text": ["CT guided biopsy revealed a fatty mass which was diagnosed as lipoma."],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    adapter.run_workflow.assert_not_called()
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == {"replacements": []}


def test_generate_map_only_filters_hallucinated_replacements(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
                "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
                "_anonymizer_row_order": [0],
                COL_REPLACEMENT_MAP: [
                    {
                        "replacements": [
                            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
                            {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
                        ]
                    }
                ],
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
            "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == {
        "replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]
    }


def test_generate_map_only_preserves_original_anonymizer_row_order_with_mixed_rows(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
                "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
                "_anonymizer_row_order": [1],
                COL_REPLACEMENT_MAP: [
                    {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]}
                ],
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["No entities here", "Alice works at Acme", "Still no entities"],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": []},
                {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
                {"entities_by_value": []},
            ],
            "tagged_text": [
                "No entities here",
                "<<PII:first_name>>Alice<</PII:first_name>> works at Acme",
                "Still no entities",
            ],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert result.dataframe[COL_TEXT].tolist() == ["No entities here", "Alice works at Acme", "Still no entities"]
    assert result.dataframe[COL_REPLACEMENT_MAP].tolist() == [
        {"replacements": []},
        {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]},
        {"replacements": []},
    ]


def test_generate_map_only_strips_internal_prompt_columns(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    """Workflow-internal prompt-construction columns must be stripped from the
    result. They carry pyarrow-backed pandas extension dtypes that would break
    a downstream `trace_dataframe.to_parquet` round-trip, and nothing
    downstream of this workflow consumes them.
    """
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
                "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
                "_anonymizer_row_order": [0],
                COL_REPLACEMENT_MAP: [
                    {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]}
                ],
                # NDD passes input columns through to its output; simulate that so the
                # drop in the entity-rows return path actually has work to do.
                **{col: [""] for col in _INTERNAL_COLUMNS},
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
            "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    for col in _INTERNAL_COLUMNS:
        assert col not in result.dataframe.columns, f"workflow-internal column {col!r} leaked into result"


def test_generate_map_only_strips_internal_prompt_columns_when_no_entities(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    """Same guarantee as above for the early-return path (no entities → no NDD call)."""
    adapter = Mock()
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["No entities here"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}],
            "tagged_text": ["No entities here"],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    adapter.run_workflow.assert_not_called()
    for col in _INTERNAL_COLUMNS:
        assert col not in result.dataframe.columns, f"workflow-internal column {col!r} leaked into result"


# ---------------------------------------------------------------------------
# PII-free logging regression tests for _filter_replacement_map_to_input_entities
#
# The replacement map filter runs after the LLM proposes substitutions and
# emits a DEBUG summary plus WARNINGs for collision repair, omission fills,
# and (still-empty) maps. All log paths must report counts and labels only,
# never raw entity values.
# ---------------------------------------------------------------------------

_ORIGINAL_PII = ("Jane Doe", "jane.doe@example.com", "+1-555-867-5309")
_SYNTHETIC_PII = ("Maya Chen", "maya.chen@example.com", "+1-555-000-1234")


def _assert_no_pii_in_logs(caplog: pytest.LogCaptureFixture, extra_secrets: tuple[str, ...] = ()) -> None:
    for secret in (*_ORIGINAL_PII, *_SYNTHETIC_PII, *extra_secrets):
        assert secret not in caplog.text, f"PII leak in logs: {secret!r} appeared in:\n{caplog.text}"


def test_filter_replacement_map_debug_log_does_not_leak_pii(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The DEBUG summary on the happy path must not emit raw entity values."""
    parsed_entities = EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [
                {"value": "Jane Doe", "labels": ["first_name"]},
                {"value": "jane.doe@example.com", "labels": ["email"]},
                {"value": "+1-555-867-5309", "labels": ["phone_number"]},
            ]
        }
    )
    raw_map = {
        "replacements": [
            {"original": "Jane Doe", "label": "first_name", "synthetic": "Maya Chen"},
            {"original": "jane.doe@example.com", "label": "email", "synthetic": "maya.chen@example.com"},
            {"original": "+1-555-867-5309", "label": "phone_number", "synthetic": "+1-555-000-1234"},
        ]
    }

    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        result = _filter_replacement_map_to_input_entities(
            raw_map=raw_map, parsed_entities=parsed_entities, record_id="row-abc123"
        )

    assert "Replacement map record" in caplog.text
    assert "requested=3" in caplog.text
    _assert_no_pii_in_logs(caplog)
    assert len(result["replacements"]) == 3


def test_filter_replacement_map_anomaly_summaries_do_not_leak_pii(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unrequested-by-label / unfilled-by-label extras must name the LABEL, not the value."""
    parsed_entities = EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [
                {"value": "Jane Doe", "labels": ["first_name"]},
                {"value": "+1-555-867-5309", "labels": ["phone_number"]},
            ]
        }
    )
    raw_map = {
        "replacements": [
            {"original": "Jane Doe", "label": "first_name", "synthetic": "Maya Chen"},
            {"original": "Acme Corp", "label": "organization_name", "synthetic": "NovaCorp"},
        ]
    }

    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        _filter_replacement_map_to_input_entities(raw_map=raw_map, parsed_entities=parsed_entities, record_id="row-xyz")

    assert "unrequested_by_label" in caplog.text
    assert "unfilled_by_label" in caplog.text
    assert "organization_name" in caplog.text
    assert "phone_number" in caplog.text

    _assert_no_pii_in_logs(caplog, extra_secrets=("Acme Corp", "NovaCorp"))


def test_filter_replacement_map_repairs_synthetic_original_collisions_without_pii(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Synthetic values must not reuse another protected original from the same row."""
    parsed_entities = EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [
                {"value": "1979-01-01", "labels": ["date"]},
                {"value": "1980-02-02", "labels": ["date"]},
            ]
        }
    )
    raw_map = {
        "replacements": [
            {"original": "1979-01-01", "label": "date", "synthetic": "1980-02-02"},
            {"original": "1980-02-02", "label": "date", "synthetic": "1991-03-04"},
        ]
    }

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = _filter_replacement_map_to_input_entities(
            raw_map=raw_map, parsed_entities=parsed_entities, record_id="row-collision"
        )

    assert result == {
        "replacements": [
            {"original": "1979-01-01", "label": "date", "synthetic": "[SUBSTITUTE_DATE_1]"},
            {"original": "1980-02-02", "label": "date", "synthetic": "1991-03-04"},
        ]
    }
    assert "synthetic-original collision" in caplog.text
    assert "date" in caplog.text
    _assert_no_pii_in_logs(caplog, extra_secrets=("1979-01-01", "1980-02-02", "1991-03-04"))


def test_filter_replacement_map_fills_all_omitted_pairs_without_pii(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the LLM returns no matching entries, fill every requested pair with a placeholder."""
    parsed_entities = EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [
                {"value": "Jane Doe", "labels": ["first_name"]},
            ]
        }
    )
    raw_map = {
        "replacements": [
            {"original": "Acme Corp", "label": "organization_name", "synthetic": "NovaCorp"},
        ]
    }

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = _filter_replacement_map_to_input_entities(
            raw_map=raw_map, parsed_entities=parsed_entities, record_id="row-empty"
        )

    assert "Replacement map filled omitted entries" in caplog.text
    assert "filled_by_label" in caplog.text
    assert "first_name" in caplog.text
    assert "Replacement map empty after filtering" not in caplog.text
    _assert_no_pii_in_logs(caplog, extra_secrets=("Acme Corp", "NovaCorp", "Jane Doe"))
    assert result == {
        "replacements": [
            {"original": "Jane Doe", "label": "first_name", "synthetic": "[SUBSTITUTE_FIRST_NAME_1]"},
        ]
    }


def test_filter_replacement_map_fills_partial_omissions_without_pii(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A partial LLM map must be completed with collision-safe placeholders for omitted pairs."""
    parsed_entities = EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [
                {"value": "Jane Doe", "labels": ["first_name"]},
                {"value": "jane.doe@example.com", "labels": ["email"]},
                {"value": "12 Oak Street", "labels": ["street_address"]},
            ]
        }
    )
    raw_map = {
        "replacements": [
            {"original": "Jane Doe", "label": "first_name", "synthetic": "Maya Chen"},
            {"original": "jane.doe@example.com", "label": "email", "synthetic": "maya.chen@example.com"},
        ]
    }

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = _filter_replacement_map_to_input_entities(
            raw_map=raw_map, parsed_entities=parsed_entities, record_id="row-partial"
        )

    assert len(result["replacements"]) == 3
    by_key = {(entry["original"], entry["label"]): entry["synthetic"] for entry in result["replacements"]}
    assert by_key[("Jane Doe", "first_name")] == "Maya Chen"
    assert by_key[("jane.doe@example.com", "email")] == "maya.chen@example.com"
    assert by_key[("12 Oak Street", "street_address")] == "[SUBSTITUTE_STREET_ADDRESS_1]"
    assert "Replacement map filled omitted entries" in caplog.text
    assert "street_address" in caplog.text
    assert "filled=1" in caplog.text
    _assert_no_pii_in_logs(caplog, extra_secrets=("12 Oak Street",))


def test_filter_replacement_map_omission_fill_continues_collision_indices(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Omission fills must share the per-label index counter with collision repairs."""
    parsed_entities = EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [
                {"value": "1979-01-01", "labels": ["date"]},
                {"value": "1980-02-02", "labels": ["date"]},
            ]
        }
    )
    raw_map = {
        "replacements": [
            {"original": "1979-01-01", "label": "date", "synthetic": "1980-02-02"},
        ]
    }

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = _filter_replacement_map_to_input_entities(
            raw_map=raw_map, parsed_entities=parsed_entities, record_id="row-index"
        )

    by_key = {(entry["original"], entry["label"]): entry["synthetic"] for entry in result["replacements"]}
    assert by_key[("1979-01-01", "date")] == "[SUBSTITUTE_DATE_1]"
    assert by_key[("1980-02-02", "date")] == "[SUBSTITUTE_DATE_2]"
    assert "synthetic-original collision" in caplog.text
    assert "filled omitted entries" in caplog.text
    _assert_no_pii_in_logs(caplog, extra_secrets=("1979-01-01", "1980-02-02"))


def test_filter_replacement_map_complete_map_skips_omission_fill_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A complete exact-match map must not emit the omission-fill WARNING."""
    parsed_entities = EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [
                {"value": "Jane Doe", "labels": ["first_name"]},
                {"value": "jane.doe@example.com", "labels": ["email"]},
            ]
        }
    )
    raw_map = {
        "replacements": [
            {"original": "Jane Doe", "label": "first_name", "synthetic": "Maya Chen"},
            {"original": "jane.doe@example.com", "label": "email", "synthetic": "maya.chen@example.com"},
        ]
    }

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = _filter_replacement_map_to_input_entities(
            raw_map=raw_map, parsed_entities=parsed_entities, record_id="row-complete"
        )

    assert len(result["replacements"]) == 2
    assert "filled omitted entries" not in caplog.text
    _assert_no_pii_in_logs(caplog)
