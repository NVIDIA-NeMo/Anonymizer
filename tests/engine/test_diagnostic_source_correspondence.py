# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic drops must not transfer a surviving source's judgments to its sibling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.replace_strategies import Substitute
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_JUDGE,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTION_JUDGE,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE,
    COL_ENTITY_COVERAGE_JUDGE,
    COL_JUDGE_EVALUATION,
    COL_RELATIONAL_CONSISTENCY_JUDGE,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
    COL_TYPE_FIDELITY_JUDGE,
    COL_TYPE_FIDELITY_VALID,
)
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.evaluation.entity_coverage_judge import EntityCoverageWorkflow
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.evaluation.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.evaluation.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.engine.row_partitioning import restore_diagnostic_columns
from anonymizer.interface.errors import AnonymizerWorkflowError


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("consumer", ["rewrite", "replacement", "coverage", "detection"])
def test_actual_dd_diagnostic_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_evaluate_model_selection, preview: bool, consumer: str
) -> None:
    adapter = NddAdapter(DataDesigner(artifact_path=tmp_path, auto_configure_logging=False))
    run = adapter.run_workflow
    receipts = []
    payloads = {
        COL_DETECTION_JUDGE: {"all_valid": True, "invalid_entities": []},
        COL_ENTITY_COVERAGE_JUDGE: {"candidate_entities": []},
        COL_TYPE_FIDELITY_JUDGE: {"all_valid": True, "invalid_replacements": []},
        COL_RELATIONAL_CONSISTENCY_JUDGE: {"all_consistent": True, "relations": []},
        COL_ATTRIBUTE_FIDELITY_JUDGE: {"all_valid": True, "entities": []},
        COL_JUDGE_EVALUATION: {"literal": "judgment-for-Alice"},
    }

    def scripted(frame, *, columns, model_configs, **kwargs):
        names = [column.name for column in columns]

        @custom_column_generator(required_columns=list(frame.columns), side_effect_columns=names[1:])
        def response(row):
            if row[COL_TEXT] == "drop":
                raise ValueError("literal first-source diagnostic drop")
            row.update({name: payloads[name] for name in names})
            return row

        result = run(
            frame,
            model_configs=[],
            columns=[CustomColumnConfig(name=names[0], generator_function=response)],
            **kwargs,
        )
        receipts.append(result)
        return result

    monkeypatch.setattr(adapter, "run_workflow", scripted)
    source = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["same"] * 3,
            COL_TEXT: ["drop", "Alice", "ordinary"],
            COL_REPLACED_TEXT: ["hidden", "Beth", "ordinary"],
            COL_REWRITTEN_TEXT: ["hidden", "Beth", "ordinary"],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": "drop", "labels": ["first_name"]}]},
                {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
                {"entities_by_value": []},
            ],
            COL_REPLACEMENT_MAP: [
                {
                    "replacements": [
                        {"original": "drop", "label": "first_name", "synthetic": "hidden"},
                        {"original": "Acme", "label": "organization", "synthetic": "Nova"},
                    ]
                },
                {
                    "replacements": [
                        {"original": "Alice", "label": "first_name", "synthetic": "Beth"},
                        {"original": "Acme", "label": "organization", "synthetic": "Nova"},
                    ]
                },
                {"replacements": []},
            ],
            "marker": pd.array([17, 29, 41], dtype="Int64"),
        },
        index=pd.Index([9, 9, 2]),
    )
    kwargs: dict[str, Any] = dict(
        model_configs=[], selected_models=stub_evaluate_model_selection, preview_num_records=3 if preview else None
    )
    detection = DetectionJudgeWorkflow(adapter)
    if consumer == "rewrite":
        result = RewriteWorkflow(adapter).evaluate(
            source,
            privacy_goal=PrivacyGoal(
                protect="Protect all personal identities", preserve="Preserve all original meaning"
            ),
            compute_detection_validity=True,
            **kwargs,
        )
        assert result.dataframe[COL_JUDGE_EVALUATION].tolist() == [None, {"literal": "judgment-for-Alice"}, None]
        assert result.dataframe[COL_DETECTION_VALID].iloc[0] is None or pd.isna(
            result.dataframe[COL_DETECTION_VALID].iloc[0]
        )
        assert result.dataframe[COL_DETECTION_VALID].iloc[1] == 1.0
    elif consumer == "replacement":
        result = ReplacementWorkflow(
            adapter=adapter,
            detection_judge=detection,
            type_fidelity_judge=TypeFidelityJudgeWorkflow(adapter),
            relational_consistency_judge=RelationalConsistencyJudgeWorkflow(adapter),
            attribute_fidelity_judge=AttributeFidelityJudgeWorkflow(adapter),
        ).evaluate(source, replace_method=Substitute(), compute_detection_validity=True, **kwargs)
        for column in (
            COL_DETECTION_VALID,
            COL_TYPE_FIDELITY_VALID,
            COL_RELATIONAL_CONSISTENCY_VALID,
            COL_ATTRIBUTE_FIDELITY_VALID,
        ):
            assert result.dataframe[column].tolist() == [None, True, True]
        assert pd.isna(result.dataframe[COL_ENTITY_COVERAGE].iloc[0])
        assert result.dataframe[COL_ENTITY_COVERAGE].iloc[1:].tolist() == [1.0, 1.0]
    elif consumer == "coverage":
        frame, failures = EntityCoverageWorkflow(adapter).run_non_critical(source, **kwargs)
        assert frame[COL_ENTITY_COVERAGE].tolist() == [None, 1.0, 1.0]
        assert frame["marker"].tolist() == [17, 29, 41]
        assert str(frame["marker"].dtype) == "Int64"
        assert len(failures) == 1
        return
    else:
        result = detection.evaluate(source, **kwargs)
        assert result.source_positions == (1, 2)
        assert result.dataframe["marker"].tolist() == [29, 41]
        assert result.dataframe[COL_DETECTION_VALID].tolist() == [True, True]
        assert len(result.failed_records) == 1
        return
    assert result.dataframe["marker"].tolist() == [17, 29, 41]
    assert str(result.dataframe["marker"].dtype) == "Int64"
    assert all(receipt.source_positions[0] == 1 for receipt in receipts)
    assert all(failure.record_id == "same" for failure in result.failed_records)
    assert len(result.failed_records) == (2 if consumer == "rewrite" else 1)


def test_diagnostic_reordering_retains_dtype_and_input_index() -> None:
    source = pd.DataFrame({"marker": pd.array([17, 29], dtype="Int64")}, index=pd.Index([7, 7]))
    returned = pd.DataFrame({"verdict": pd.array([False, True], dtype="boolean")})
    actual = restore_diagnostic_columns(source, returned, {"verdict": None}, source_positions=(1, 0))
    assert actual["marker"].tolist() == [17, 29]
    assert actual["verdict"].tolist() == [True, False]
    assert actual.index.tolist() == [7, 7]
    assert str(actual["verdict"].dtype) == "boolean"


@pytest.mark.parametrize("positions", [(0, 0), (True,), (2,)])
def test_diagnostic_invalid_native_positions_do_not_fall_back_to_matching_ids(positions) -> None:
    source = pd.DataFrame({RECORD_ID_COLUMN: ["first", "second"]})
    returned = source if len(positions) == 2 else source.iloc[[1]]
    with pytest.raises(AnonymizerWorkflowError, match="invalid source correspondence"):
        restore_diagnostic_columns(source, returned, {"verdict": None}, source_positions=positions)
