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
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DISPOSITION_COVERAGE,
    COL_DISPOSITION_LATENT_ENTITIES,
    COL_GENERALIZATION_NEEDS_REVIEW,
    COL_GENERALIZATION_REVIEW_INPUT,
    COL_GENERALIZATION_SUGGESTIONS,
    COL_GENERALIZATION_TARGETS,
    COL_RAW_GENERALIZATION_SUGGESTIONS,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REVIEWED_GENERALIZATION_SUGGESTIONS,
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
        "rewrite_instruction": "" if status == "ready" else "Omit identifying supporting details.",
        "related_entity_ids": [],
    }


def _row() -> dict[str, Any]:
    row = {
        COL_SENSITIVITY_DISPOSITION: {
            "sensitivity_disposition": [_entity(1, "Alice", "replace"), _entity(2, "patent attorney", "generalize")]
        }
    }
    build_generalization_targets(row)
    row[COL_REVIEWED_GENERALIZATION_SUGGESTIONS] = {"defects": [], "generalization_suggestions": [_suggestion()]}
    return row


@pytest.mark.parametrize("status", ["ready", "needs_context_change", "no_effective_generalization"])
def test_status_controls_review(status: str) -> None:
    row = _row()
    row[COL_REVIEWED_GENERALIZATION_SUGGESTIONS]["generalization_suggestions"][0] = _suggestion(status=status)
    result = validate_generalization_suggestions(row)
    assert result[COL_GENERALIZATION_NEEDS_REVIEW] is (status != "ready")


@pytest.mark.parametrize("ids", [[], [1], [2, 2], [2, 3]])
def test_rejects_missing_extra_or_duplicate_targets(ids: list[int]) -> None:
    row = _row()
    row[COL_REVIEWED_GENERALIZATION_SUGGESTIONS] = {"generalization_suggestions": [_suggestion(i) for i in ids]}
    with pytest.raises(ValueError, match="IDs must match"):
        validate_generalization_suggestions(row)


@pytest.mark.parametrize("status", ["ready", "needs_context_change"])
@pytest.mark.parametrize("value", ["patent attorney", " Patent Attorney "])
def test_unchanged_value_becomes_removal(value: str, status: str) -> None:
    row = _row()
    row[COL_REVIEWED_GENERALIZATION_SUGGESTIONS]["generalization_suggestions"][0] = _suggestion(status=status)
    suggestion = row[COL_REVIEWED_GENERALIZATION_SUGGESTIONS]["generalization_suggestions"][0]
    suggestion["suggested_value"] = value
    result = validate_generalization_suggestions(row)
    canonical = result[COL_GENERALIZATION_SUGGESTIONS]["generalization_suggestions"][0]
    assert canonical["status"] == "no_effective_generalization"
    assert canonical["suggested_value"] is None
    GeneralizationSuggestion.model_validate(canonical)
    assert result[COL_GENERALIZATION_NEEDS_REVIEW] is True
    assert suggestion["suggested_value"] == value
    assert suggestion["status"] == status
    result[COL_DISPOSITION_LATENT_ENTITIES] = ""
    actions = _build_rewrite_actions(result)[COL_REWRITE_ACTIONS]
    assert actions["generalize"] == []
    assert [action["entity_id"] for action in actions["remove"]] == [2]
    assert "Do not substitute" in actions["remove"][0]["rewrite_instruction"]


@pytest.mark.parametrize("value", ["a professional", "patent attorney"])
def test_rejects_unknown_dependencies(value: str) -> None:
    row = _row()
    suggestion = row[COL_REVIEWED_GENERALIZATION_SUGGESTIONS]["generalization_suggestions"][0]
    suggestion["suggested_value"] = value
    suggestion["related_entity_ids"] = [99]
    with pytest.raises(ValueError, match="unknown entity"):
        validate_generalization_suggestions(row)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_context_change_without_wording_becomes_removal(value: str | None) -> None:
    row = _row()
    suggestion = {**_suggestion(status="needs_context_change"), "suggested_value": value, "rewrite_instruction": ""}
    row[COL_REVIEWED_GENERALIZATION_SUGGESTIONS]["generalization_suggestions"] = [suggestion]
    result = validate_generalization_suggestions(row)
    canonical = result[COL_GENERALIZATION_SUGGESTIONS]["generalization_suggestions"][0]
    assert canonical["status"] == "no_effective_generalization"
    assert canonical["suggested_value"] is None
    GeneralizationSuggestion.model_validate(canonical)
    assert result[COL_GENERALIZATION_NEEDS_REVIEW] is True
    assert suggestion["status"] == "needs_context_change"
    assert suggestion["suggested_value"] == value
    result[COL_DISPOSITION_LATENT_ENTITIES] = ""
    actions = _build_rewrite_actions(result)[COL_REWRITE_ACTIONS]
    assert actions["generalize"] == []
    assert [action["entity_id"] for action in actions["remove"]] == [2]
    suggestion["related_entity_ids"] = [99]
    with pytest.raises(ValueError, match="unknown entity"):
        validate_generalization_suggestions(row)


def test_status_schema_requires_actionable_wording_or_limitation() -> None:
    with pytest.raises(ValueError):
        GeneralizationSuggestion.model_validate({**_suggestion(), "suggested_value": None})
    with pytest.raises(ValueError):
        GeneralizationSuggestion.model_validate(
            {**_suggestion(status="no_effective_generalization"), "rewrite_instruction": ""}
        )


@custom_column_generator(required_columns=[COL_GENERALIZATION_TARGETS, COL_REPLACEMENT_MAP_FOR_PROMPT])
def _generate_suggestions(row: dict[str, Any]) -> dict[str, Any]:
    assert normalize_payload(row[COL_GENERALIZATION_TARGETS]), "Empty targets must skip generation"
    replacements = normalize_payload(row[COL_REPLACEMENT_MAP_FOR_PROMPT])
    assert "patent attorney" not in str(replacements)
    row[COL_RAW_GENERALIZATION_SUGGESTIONS] = {"defects": [], "generalization_suggestions": [_suggestion()]}
    return row


@custom_column_generator(required_columns=[COL_GENERALIZATION_REVIEW_INPUT, COL_GENERALIZATION_TARGETS])
def _review_suggestions(row: dict[str, Any]) -> dict[str, Any]:
    assert normalize_payload(row[COL_GENERALIZATION_TARGETS]), "Empty targets must skip review"
    candidates = normalize_payload(row[COL_GENERALIZATION_REVIEW_INPUT])
    assert candidates == [{"entity_id": 2, "suggested_value": "a professional"}]
    row[COL_REVIEWED_GENERALIZATION_SUGGESTIONS] = {
        "defects": [
            {
                "entity_id": 2,
                "evidence": "a professional",
                "problem": "Insufficient contextual protection.",
                "conflicting_entity_ids": [],
            }
        ],
        "generalization_suggestions": [_suggestion(status="no_effective_generalization")],
    }
    return row


def test_scheduler_filters_map_and_skips_empty_targets(
    tmp_path: Path,
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    columns = GeneralizationWorkflow().columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=PrivacyGoal(protect="Protect personal identity", preserve="Preserve document meaning"),
    )
    columns[1] = CustomColumnConfig(
        name=COL_RAW_GENERALIZATION_SUGGESTIONS,
        generator_function=_generate_suggestions,
        skip=columns[1].skip,
    )
    columns[3] = CustomColumnConfig(
        name=COL_REVIEWED_GENERALIZATION_SUGGESTIONS,
        generator_function=_review_suggestions,
        skip=columns[3].skip,
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
        assert bool(row[COL_GENERALIZATION_NEEDS_REVIEW]) == bool(targets)
        if targets:
            assert suggestions[0]["status"] == "no_effective_generalization"
            assert suggestions[0]["suggested_value"] is None
            assert (
                normalize_payload(row[COL_RAW_GENERALIZATION_SUGGESTIONS])["generalization_suggestions"][0]["status"]
                == "ready"
            )
        filtered = normalize_payload(row[COL_REPLACEMENT_MAP_FOR_PROMPT])
        assert "Maria" in str(filtered)
        assert "teacher" not in str(filtered)


@pytest.mark.parametrize("target,conflicts", [(99, []), (2, [99])])
def test_rejects_defects_with_unknown_ids(target: int, conflicts: list[int]) -> None:
    row = _row()
    row[COL_REVIEWED_GENERALIZATION_SUGGESTIONS]["defects"] = [
        {
            "entity_id": target,
            "evidence": "a professional",
            "problem": "Retains identifying evidence.",
            "conflicting_entity_ids": conflicts,
        }
    ]
    with pytest.raises(ValueError, match="defect"):
        validate_generalization_suggestions(row)
