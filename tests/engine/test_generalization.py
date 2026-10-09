# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.models import RewriteModelSelection
from anonymizer.engine.constants import (
    COL_DISPOSITION_COVERAGE,
    COL_DISPOSITION_LATENT_ENTITIES,
    COL_GENERALIZATION_NEEDS_REVIEW,
    COL_GENERALIZATION_SUGGESTIONS,
    COL_GENERALIZATION_TARGETS,
    COL_RAW_GENERALIZATION_SUGGESTIONS,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITE_ACTIONS,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_SENSITIVITY_DISPOSITION,
)
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.rewrite.generalization import (
    GeneralizationWorkflow,
    build_generalization_targets,
    validate_generalization_suggestions,
)
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.rewrite.rewrite_generation import (
    _build_rewrite_actions,
    _filter_replacement_map_for_prompt,
    _format_rewrite_disposition_block,
)
from anonymizer.engine.schemas.generalization import GeneralizationSuggestion


def _entity(entity_id: int, value: str, method: str) -> dict[str, Any]:
    return {
        "id": entity_id,
        "source": "tagged",
        "category": "direct_identifier" if method == "replace" else "quasi_identifier",
        "entity_label": "name" if method == "replace" else "occupation",
        "entity_value": value,
        "sensitivity": "high" if method == "replace" else "medium",
        "protection_method_suggestion": method,
        "protection_reason": "Protect identifying information.",
    }


def _suggestion(entity_id: int = 2, status: str = "ready") -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "suggested_value": None if status == "no_effective_generalization" else "a professional",
        "status": status,
        "privacy_reason": "Conceals the exact occupation.",
    }


def _row() -> dict[str, Any]:
    row = {
        COL_SENSITIVITY_DISPOSITION: {
            "sensitivity_disposition": [_entity(1, "Alice", "replace"), _entity(2, "patent attorney", "generalize")]
        }
    }
    build_generalization_targets(row)
    row[COL_RAW_GENERALIZATION_SUGGESTIONS] = {
        "generalization_suggestions": [{"entity_id": 2, "suggested_value": "a professional"}]
    }
    return row


@pytest.mark.parametrize("value", [None, "   ", "patent attorney", " Patent Attorney ", "a professional"])
def test_candidates_are_validated_directly(value: str | None) -> None:
    row = _row()
    raw = row[COL_RAW_GENERALIZATION_SUGGESTIONS]["generalization_suggestions"][0]
    raw["suggested_value"] = value
    result = validate_generalization_suggestions(row)
    suggestion = result[COL_GENERALIZATION_SUGGESTIONS]["generalization_suggestions"][0]
    omission = value != "a professional"
    assert suggestion["suggested_value"] == (None if omission else value)
    assert suggestion["status"] == ("no_effective_generalization" if omission else "ready")
    assert result[COL_GENERALIZATION_NEEDS_REVIEW] is omission
    assert raw["suggested_value"] == value
    GeneralizationSuggestion.model_validate(suggestion)
    result[COL_DISPOSITION_LATENT_ENTITIES] = ""
    actions = _build_rewrite_actions(result)[COL_REWRITE_ACTIONS]
    assert len(actions["remove"]) == int(omission)
    assert len(actions["generalize"]) == int(not omission)


@pytest.mark.parametrize("ids", [[], [1], [2, 2], [2, 3]])
def test_rejects_missing_extra_or_duplicate_targets(ids: list[int]) -> None:
    row = _row()
    row[COL_RAW_GENERALIZATION_SUGGESTIONS] = {
        "generalization_suggestions": [{"entity_id": i, "suggested_value": "a professional"} for i in ids]
    }
    with pytest.raises(ValueError, match="IDs must match"):
        validate_generalization_suggestions(row)


def test_workflow_has_only_generator_and_local_validation(stub_rewrite_model_selection: RewriteModelSelection) -> None:
    columns = GeneralizationWorkflow().columns(selected_models=stub_rewrite_model_selection)
    assert [column.name for column in columns] == [
        COL_GENERALIZATION_TARGETS,
        COL_RAW_GENERALIZATION_SUGGESTIONS,
        COL_GENERALIZATION_SUGGESTIONS,
    ]
    assert isinstance(columns[2], CustomColumnConfig)
    assert columns[2].generator_function is validate_generalization_suggestions


@custom_column_generator(required_columns=[COL_GENERALIZATION_TARGETS, COL_REPLACEMENT_MAP_FOR_PROMPT])
def _generate_suggestions(row: dict[str, Any]) -> dict[str, Any]:
    assert normalize_payload(row[COL_GENERALIZATION_TARGETS]), "Empty targets must skip generation"
    replacements = normalize_payload(row[COL_REPLACEMENT_MAP_FOR_PROMPT])
    assert "patent attorney" not in str(replacements)
    row[COL_RAW_GENERALIZATION_SUGGESTIONS] = {
        "generalization_suggestions": [{"entity_id": 2, "suggested_value": "a professional"}]
    }
    return row


def test_scheduler_filters_map_and_skips_empty_targets(
    tmp_path: Path,
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    columns = GeneralizationWorkflow().columns(
        selected_models=stub_rewrite_model_selection,
    )
    columns[1] = CustomColumnConfig(
        name=COL_RAW_GENERALIZATION_SUGGESTIONS,
        generator_function=_generate_suggestions,
        skip=columns[1].skip,
    )
    columns.insert(
        0,
        CustomColumnConfig(
            name=COL_REPLACEMENT_MAP_FOR_PROMPT,
            generator_function=_filter_replacement_map_for_prompt,
        ),
    )
    columns.insert(
        0, CustomColumnConfig(name=COL_REWRITE_DISPOSITION_BLOCK, generator_function=_format_rewrite_disposition_block)
    )
    dispositions = [_row()[COL_SENSITIVITY_DISPOSITION], {"sensitivity_disposition": [_entity(1, "Alice", "replace")]}]
    adapter = NddAdapter(DataDesigner(artifact_path=tmp_path / "artifacts", auto_configure_logging=False))
    result = adapter.run_workflow(
        pd.DataFrame(
            {
                COL_SENSITIVITY_DISPOSITION: dispositions,
                COL_DISPOSITION_COVERAGE: [True, True],
                COL_REPLACEMENT_MAP: [
                    {
                        "replacements": [
                            {"original": "Alice", "label": "name", "synthetic": "Maria"},
                            {"original": "patent attorney", "label": "occupation", "synthetic": "teacher"},
                        ]
                    }
                ]
                * 2,
            }
        ),
        model_configs=[],
        columns=columns,
        workflow_name="generalization-test",
        preview_num_records=2,
    )
    assert not result.failed_records
    rows = result.dataframe.to_dict("records")
    for row in rows:
        targets = normalize_payload(row[COL_GENERALIZATION_TARGETS])
        suggestions = normalize_payload(row[COL_GENERALIZATION_SUGGESTIONS])["generalization_suggestions"]
        assert len(suggestions) == len(targets)
        assert not row[COL_GENERALIZATION_NEEDS_REVIEW]
        if targets:
            assert suggestions[0]["status"] == "ready"
            assert suggestions[0]["suggested_value"] == "a professional"
            assert normalize_payload(row[COL_RAW_GENERALIZATION_SUGGESTIONS])["generalization_suggestions"] == [
                {"entity_id": 2, "suggested_value": "a professional"}
            ]
        filtered = normalize_payload(row[COL_REPLACEMENT_MAP_FOR_PROMPT])
        assert "Maria" in str(filtered)
        assert "teacher" not in str(filtered)


def test_generator_output_contract() -> None:
    from anonymizer.engine.rewrite.generalization import _get_generalization_prompt
    from anonymizer.engine.schemas.generalization import GeneralizationCandidate

    assert set(GeneralizationCandidate.model_fields) == {"entity_id", "suggested_value"}
    assert "privacy_goal" not in _get_generalization_prompt()
