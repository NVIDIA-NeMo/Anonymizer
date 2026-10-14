# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal source/drop controls using real DD execution and local generators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_DETECTED_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA,
    COL_QUALITY_QA,
    COL_REPAIR_ITERATIONS,
    COL_REPLACEMENT_MAP,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter, WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow, _join_new_columns
from anonymizer.engine.row_partitioning import resolve_source_positions
from anonymizer.interface.errors import AnonymizerWorkflowError


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("id_mode", ["unique", "duplicate", "generated"])
def test_actual_dd_duplicate_id_drop(tmp_path: Path, preview: bool, id_mode: str) -> None:
    @custom_column_generator(required_columns=[COL_TEXT])
    def candidate(row):
        if row[COL_TEXT] == "drop":
            raise ValueError("intentional source drop")
        row[COL_REWRITTEN_TEXT] = "kept candidate"
        return row

    source = pd.DataFrame(
        {RECORD_ID_COLUMN: ["same", "same" if id_mode == "duplicate" else "other"], COL_TEXT: ["drop", "keep"]}
    )
    if id_mode == "generated":
        source = source.drop(columns=RECORD_ID_COLUMN)
    adapter = NddAdapter(DataDesigner(artifact_path=tmp_path, auto_configure_logging=False))
    result = adapter.run_workflow(
        source,
        model_configs=[],
        columns=[CustomColumnConfig(name=COL_REWRITTEN_TEXT, generator_function=candidate)],
        workflow_name="literal-drop",
        preview_num_records=2 if preview else None,
    )
    assert result.source_positions == (1,)
    assert [failure.step for failure in result.failed_records] == ["literal-drop"]
    if id_mode != "generated":
        assert result.failed_records[0].record_id == "same"
    else:
        assert len(result.failed_records[0].record_id) == 32
    joined = _join_new_columns(source, result.dataframe, source_positions=result.source_positions)
    assert joined[COL_TEXT].tolist() == ["keep"]
    assert joined[COL_REWRITTEN_TEXT].tolist() == ["kept candidate"]


@pytest.mark.parametrize("positions", [(), (0, 0), (2,), (-1,), (True,), [1], ("1",)])
def test_malformed_explicit_positions_fail_closed(positions) -> None:
    source = pd.DataFrame({COL_TEXT: ["first", "second"]})
    returned = source if positions == (0, 0) else source.iloc[[1]]
    with pytest.raises(AnonymizerWorkflowError, match="invalid source correspondence"):
        resolve_source_positions(source, returned, positions)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("preview", [False, True])
def test_actual_dd_rewrite_schedule_drop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_rewrite_model_selection,
    stub_replace_model_selection,
    combined: bool,
    preview: bool,
) -> None:
    """Keep actual DD source projection while replacing model responses with literals."""
    adapter = NddAdapter(DataDesigner(artifact_path=tmp_path, auto_configure_logging=False))
    run = adapter.run_workflow

    def scripted(frame, *, columns, model_configs, workflow_name, **kwargs):
        output = (
            COL_REPLACEMENT_MAP
            if workflow_name == "replace-map-generation"
            else COL_UTILITY_SCORE
            if workflow_name.startswith("rewrite-evaluate-")
            else COL_REWRITTEN_TEXT
        )
        side_effects = [
            COL_QUALITY_QA,
            COL_PRIVACY_QA,
            COL_UTILITY_SCORE,
            COL_LEAKAGE_MASS,
            COL_ANY_HIGH_LEAKED,
            COL_NEEDS_REPAIR,
            COL_NEEDS_HUMAN_REVIEW,
            COL_REPAIR_ITERATIONS,
        ]

        side_effects = [name for name in side_effects if name not in frame.columns and name != output]

        @custom_column_generator(required_columns=list(frame.columns), side_effect_columns=side_effects)
        def response(row):
            text = row.get(COL_TEXT, row.get(COL_REWRITTEN_TEXT))
            if text == "drop" and workflow_name in {"rewrite-pipeline", "rewrite-combined"}:
                raise ValueError("intentional rewrite source drop")
            row[output] = {"replacements": []} if output == COL_REPLACEMENT_MAP else text
            row.update(
                {
                    COL_QUALITY_QA: {"questions": []},
                    COL_PRIVACY_QA: {"questions": []},
                    COL_UTILITY_SCORE: 1.0,
                    COL_LEAKAGE_MASS: 0.0,
                    COL_ANY_HIGH_LEAKED: False,
                    COL_NEEDS_REPAIR: False,
                    COL_NEEDS_HUMAN_REVIEW: False,
                    COL_REPAIR_ITERATIONS: 0,
                }
            )
            return row

        return run(
            frame,
            model_configs=[],
            columns=[CustomColumnConfig(name=output, generator_function=response)],
            workflow_name=workflow_name,
            **kwargs,
        )

    monkeypatch.setattr(adapter, "run_workflow", scripted)
    source = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["same"] * 3,
            COL_TEXT: ["drop", "keep", "plain"],
            "marker": [17, 29, 43],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "name", "labels": ["person"]}]}] * 2
            + [{"entities_by_value": []}],
        },
        index=pd.Index([8, 8, 8]),
    )
    workflow = CombinedRewriteWorkflow(adapter) if combined else RewriteWorkflow(adapter)
    result = workflow.run(
        source,
        model_configs=[],
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=PrivacyGoal(protect="Protect all names", preserve="Preserve original meaning"),
        evaluation=EvaluationCriteria(max_repair_iterations=0),
        preview_num_records=3 if preview else None,
    )
    assert result.dataframe["marker"].tolist() == [29, 43]
    assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == ["keep", "plain"]
    assert [failure.record_id for failure in result.failed_records] == ["same"]


def test_actual_dd_map_and_latent_positions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_replace_model_selection,
    stub_detection_model_selection,
) -> None:
    adapter = NddAdapter(DataDesigner(artifact_path=tmp_path, auto_configure_logging=False))
    run = adapter.run_workflow

    def scripted(frame, *, columns, model_configs, workflow_name, **kwargs):
        output = (
            COL_REPLACEMENT_MAP
            if workflow_name == "replace-map-generation"
            else COL_DETECTED_ENTITIES
            if workflow_name == "entity-detection"
            else COL_LATENT_ENTITIES
        )

        @custom_column_generator(required_columns=[COL_TEXT])
        def response(row):
            dropped = "detector-drop" if workflow_name == "entity-detection" else "drop"
            if row[COL_TEXT] == dropped:
                raise ValueError("intentional stage drop")
            key = (
                "replacements"
                if output == COL_REPLACEMENT_MAP
                else "entities"
                if output == COL_DETECTED_ENTITIES
                else "latent_entities"
            )
            row[output] = {key: []}
            return row

        return run(
            frame,
            model_configs=[],
            columns=[CustomColumnConfig(name=output, generator_function=response)],
            workflow_name=workflow_name,
            **kwargs,
        )

    monkeypatch.setattr(adapter, "run_workflow", scripted)
    source = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["same"] * 3,
            COL_TEXT: ["plain", "drop", "keep"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}]
            + [{"entities_by_value": [{"value": "name", "labels": ["person"]}]}] * 2,
        }
    )
    mapped = LlmReplaceWorkflow(adapter).generate_map_only(
        source, model_configs=[], selected_models=stub_replace_model_selection
    )
    assert mapped.source_positions == (0, 2)
    assert mapped.dataframe[COL_TEXT].tolist() == ["plain", "keep"]
    assert [failure.record_id for failure in mapped.failed_records] == ["same"]
    latent = EntityDetectionWorkflow(adapter).identify_latent_entities(
        source,
        model_configs=[],
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.3,
        privacy_goal=PrivacyGoal(protect="Protect all names", preserve="Preserve original meaning"),
    )
    assert latent.source_positions == (0, 2)
    assert latent.dataframe[COL_TEXT].tolist() == ["plain", "keep"]

    initial = pd.DataFrame({RECORD_ID_COLUMN: ["same"] * 4, COL_TEXT: ["detector-drop", "drop", "keep", "plain"]})
    detected = EntityDetectionWorkflow(adapter).run(
        initial,
        model_configs=[],
        selected_models=stub_detection_model_selection,
        gliner_detection_threshold=0.3,
        tag_latent_entities=True,
        privacy_goal=PrivacyGoal(protect="Protect all names", preserve="Preserve original meaning"),
        source_frame=WorkflowSourceFrame.capture(initial),
    )
    assert detected.source_positions == (2, 3)
    assert detected.dataframe[COL_TEXT].tolist() == ["keep", "plain"]
    assert [(failure.record_id, failure.step) for failure in detected.failed_records] == [
        ("same", "entity-detection"),
        ("same", "latent-entity-detection"),
    ]


def test_actual_dd_repair_subset_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_rewrite_model_selection, stub_replace_model_selection
) -> None:
    adapter = NddAdapter(DataDesigner(artifact_path=tmp_path, auto_configure_logging=False))
    run = adapter.run_workflow

    def scripted(frame, *, columns, model_configs, workflow_name, **kwargs):
        output = (
            COL_REWRITTEN_TEXT
            if workflow_name == "rewrite-pipeline"
            else COL_REWRITTEN_TEXT_NEXT
            if workflow_name.startswith("rewrite-repair-")
            else COL_UTILITY_SCORE
        )
        side_effects = (
            []
            if output in {COL_REWRITTEN_TEXT_NEXT, COL_REWRITTEN_TEXT}
            else [COL_LEAKAGE_MASS, COL_ANY_HIGH_LEAKED, COL_NEEDS_REPAIR]
        )

        @custom_column_generator(required_columns=list(frame.columns), side_effect_columns=side_effects)
        def response(row):
            if output == COL_REWRITTEN_TEXT:
                row[output] = row[COL_TEXT]
            elif output == COL_REWRITTEN_TEXT_NEXT:
                if row[COL_REWRITTEN_TEXT] == "drop":
                    raise ValueError("intentional repair subset drop")
                row[output] = "repaired"
            else:
                row.update(
                    {
                        COL_UTILITY_SCORE: 1.0,
                        COL_LEAKAGE_MASS: 0.0,
                        COL_ANY_HIGH_LEAKED: False,
                        COL_NEEDS_REPAIR: row[COL_REWRITTEN_TEXT] in {"drop", "repair"},
                    }
                )
            return row

        return run(
            frame,
            model_configs=[],
            columns=[CustomColumnConfig(name=output, generator_function=response)],
            workflow_name=workflow_name,
            **kwargs,
        )

    monkeypatch.setattr(adapter, "run_workflow", scripted)
    source = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["same"] * 3,
            COL_TEXT: ["pass", "drop", "repair"],
            COL_REWRITTEN_TEXT: ["pass", "drop", "repair"],
            "marker": [17, 29, 43],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "name", "labels": ["person"]}]}] * 3,
        }
    )
    monkeypatch.setattr(
        LlmReplaceWorkflow,
        "generate_map_only",
        lambda self, rows, **kwargs: WorkflowRunResult(rows.copy(), [], tuple(range(len(rows)))),
    )
    rewritten = RewriteWorkflow(adapter).run(
        source,
        model_configs=[],
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=PrivacyGoal(protect="Protect all names", preserve="Preserve original meaning"),
        evaluation=EvaluationCriteria(max_repair_iterations=1),
        preview_num_records=None,
    )
    result, failures = rewritten.dataframe, rewritten.failed_records
    assert result["marker"].tolist() == [17, 43]
    assert result[COL_REWRITTEN_TEXT].tolist() == ["pass", "repaired"]
    assert result[COL_REPAIR_ITERATIONS].tolist() == [0, 1]
    assert [(failure.record_id, failure.step) for failure in failures] == [("same", "rewrite-repair-0")]


def test_bound_detection_rejects_omitted_positions(
    monkeypatch: pytest.MonkeyPatch, stub_detection_model_selection
) -> None:
    source = pd.DataFrame({COL_TEXT: ["keep"]})
    adapter = NddAdapter(DataDesigner(auto_configure_logging=False))
    monkeypatch.setattr(adapter, "run_workflow", lambda *args, **kwargs: WorkflowRunResult(source, []))
    with pytest.raises(AnonymizerWorkflowError, match="missing source correspondence"):
        EntityDetectionWorkflow(adapter).detect_and_validate_entities(
            source,
            model_configs=[],
            selected_models=stub_detection_model_selection,
            gliner_detection_threshold=0.3,
            source_frame=WorkflowSourceFrame.capture(source),
        )


def test_actual_dd_preview_excludes_unrequested_failures(tmp_path: Path) -> None:
    @custom_column_generator(required_columns=[COL_TEXT])
    def candidate(row):
        if row[COL_TEXT] == "unrequested":
            raise AssertionError("preview must not execute an unrequested row")
        row[COL_REWRITTEN_TEXT] = "kept"
        return row

    source = pd.DataFrame({RECORD_ID_COLUMN: ["same", "same"], COL_TEXT: ["keep", "unrequested"]})
    result = NddAdapter(DataDesigner(artifact_path=tmp_path, auto_configure_logging=False)).run_workflow(
        source,
        model_configs=[],
        columns=[CustomColumnConfig(name=COL_REWRITTEN_TEXT, generator_function=candidate)],
        workflow_name="preview-prefix",
        preview_num_records=1,
    )
    assert result.source_positions == (0,)
    assert result.failed_records == []
    assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == ["kept"]
