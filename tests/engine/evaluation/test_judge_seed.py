# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judges hand DataDesigner a seed that Parquet can store, and return the full trace."""

from __future__ import annotations

import io
import logging
from typing import Any, cast

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DETECTION_JUDGE,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
    COL_TEXT,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.evaluation.entity_coverage_judge import EntityCoverageWorkflow
from anonymizer.engine.evaluation.judge_base import _BaseJudgeWorkflow
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.evaluation.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.evaluation.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow

_PAYLOADS = {
    COL_DETECTION_JUDGE: {"all_valid": True, "invalid_entities": []},
    TypeFidelityJudgeWorkflow.RAW_COL: {"all_valid": True, "invalid_replacements": []},
    RelationalConsistencyJudgeWorkflow.RAW_COL: {"all_consistent": True, "relations": []},
    AttributeFidelityJudgeWorkflow.RAW_COL: {"all_valid": True, "entities": []},
    EntityCoverageWorkflow.RAW_COL: {"candidate_entities": []},
}


class _ParquetAdapter:
    """Stores each seed as Parquet and reads it back, as DataDesigner does; can drop the first row."""

    def __init__(self, *, drop_first: bool = False) -> None:
        self.drop_first = drop_first
        self.seeds: list[pd.DataFrame] = []

    def run_workflow(
        self,
        df: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        columns: list[Any],
        workflow_name: str,
        preview_num_records: int | None = None,
    ) -> WorkflowRunResult:
        self.seeds.append(df)
        buffer = io.BytesIO()
        df.to_parquet(buffer)
        out = pd.read_parquet(io.BytesIO(buffer.getvalue())).reset_index(drop=True)
        for column in columns:
            out[column.name] = [_PAYLOADS[column.name] for _ in range(len(out))]
        positions = tuple(range(len(out)))
        if self.drop_first:
            out, positions = out.iloc[1:].reset_index(drop=True), positions[1:]
        return WorkflowRunResult(out, [], source_positions=positions)


def _telemetry(counts: dict[str, int]) -> dict[str, Any]:
    skipped = sum(counts.values())
    return {
        "targeted_span_count": 2,
        "applied_span_count": 2 - skipped,
        "skipped_span_count": skipped,
        "skipped_span_label_counts": counts,
    }


def _trace(telemetry: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_TEXT: ["Alice lives in Paris", "Bob met Carol in Rome"],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
                {
                    "entities_by_value": [
                        {"value": "Bob", "labels": ["first_name"]},
                        {"value": "Rome", "labels": ["city"]},
                    ]
                },
            ],
            COL_REPLACEMENT_MAP: [
                {
                    "replacements": [
                        {"original": "Alice", "label": "first_name", "synthetic": "Maria"},
                        {"original": "Paris", "label": "city", "synthetic": "Lyon"},
                    ]
                },
                {
                    "replacements": [
                        {"original": "Bob", "label": "first_name", "synthetic": "Tom"},
                        {"original": "Rome", "label": "city", "synthetic": "Milan"},
                    ]
                },
            ],
            COL_REPLACED_TEXT: ["Maria lives in Lyon", "Tom met Carol in Milan"],
            COL_REPLACEMENT_APPLICATION: telemetry,
        }
    )


_ALL_EMPTY = [_telemetry({}), _telemetry({})]
_MIXED = [_telemetry({}), _telemetry({"city": 1})]


@pytest.mark.parametrize("drop_first", [False, True], ids=["all-rows", "one-row-dropped"])
def test_shared_judge_path_seed_survives_parquet(
    drop_first: bool, stub_evaluate_model_selection: EvaluateModelSelection
) -> None:
    adapter = _ParquetAdapter(drop_first=drop_first)
    trace = _trace(_ALL_EMPTY)

    result = DetectionJudgeWorkflow(adapter).evaluate(
        trace, model_configs=[], selected_models=stub_evaluate_model_selection
    )

    assert COL_REPLACEMENT_APPLICATION not in adapter.seeds[0].columns
    kept = [1] if drop_first else [0, 1]
    assert result.source_positions == tuple(kept)
    assert list(result.dataframe[COL_TEXT]) == [trace[COL_TEXT][i] for i in kept]
    assert list(result.dataframe[COL_REPLACEMENT_APPLICATION]) == [_ALL_EMPTY[i] for i in kept]
    assert list(result.dataframe[COL_DETECTION_VALID]) == [True] * len(kept)


@pytest.mark.parametrize(
    "judge_cls",
    [
        DetectionJudgeWorkflow,
        TypeFidelityJudgeWorkflow,
        RelationalConsistencyJudgeWorkflow,
        AttributeFidelityJudgeWorkflow,
        EntityCoverageWorkflow,
    ],
    ids=lambda cls: cls.__name__,
)
@pytest.mark.parametrize("telemetry", [_ALL_EMPTY, _MIXED], ids=["all-empty", "mixed"])
def test_every_judge_returns_native_trace_cells(
    judge_cls: type[_BaseJudgeWorkflow],
    telemetry: list[dict[str, Any]],
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    trace = _trace(telemetry)

    adapter = _ParquetAdapter()
    result = judge_cls(adapter).evaluate(trace, model_configs=[], selected_models=stub_evaluate_model_selection)

    assert adapter.seeds, "the judge must run, not pass every row through"

    assert result.dataframe[judge_cls.VALID_COL].notna().all()
    assert list(result.dataframe[COL_REPLACEMENT_APPLICATION]) == telemetry
    assert list(result.dataframe[COL_ENTITIES_BY_VALUE]) == list(trace[COL_ENTITIES_BY_VALUE])
    assert RECORD_ID_COLUMN not in result.dataframe.columns


@pytest.mark.parametrize("drop_first", [False, True], ids=["all-rows", "one-row-dropped"])
def test_rewrite_evaluate_scores_detection_validity_with_real_judge(
    drop_first: bool, stub_evaluate_model_selection: EvaluateModelSelection
) -> None:
    workflow = RewriteWorkflow(adapter=cast(NddAdapter, _ParquetAdapter(drop_first=drop_first)))
    workflow._detection_judge_wf = DetectionJudgeWorkflow(_ParquetAdapter(drop_first=drop_first))

    result = workflow.evaluate(
        _trace(_ALL_EMPTY),
        model_configs=[ModelConfig(alias="stub-model", model="stub-model", provider="stub")],
        selected_models=stub_evaluate_model_selection,
        privacy_goal=PrivacyGoal(protect="All direct identifiers", preserve="General meaning of the text"),
        compute_detection_validity=True,
    )

    expected = [None, 1.0] if drop_first else [1.0, 1.0]
    assert [None if pd.isna(v) else v for v in result.dataframe[COL_DETECTION_VALID]] == expected
    assert list(result.dataframe[COL_REPLACEMENT_APPLICATION]) == _ALL_EMPTY


def test_rewrite_evaluate_logs_detection_judge_failure_without_exception_text(
    stub_evaluate_model_selection: EvaluateModelSelection, caplog: pytest.LogCaptureFixture
) -> None:
    class _FailingJudge:
        def evaluate(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("secret Alice")

    workflow = RewriteWorkflow(adapter=cast(NddAdapter, _ParquetAdapter()))
    workflow._detection_judge_wf = cast(DetectionJudgeWorkflow, _FailingJudge())

    with caplog.at_level(logging.DEBUG, logger="anonymizer"):
        result = workflow.evaluate(
            _trace(_ALL_EMPTY),
            model_configs=[ModelConfig(alias="stub-model", model="stub-model", provider="stub")],
            selected_models=stub_evaluate_model_selection,
            privacy_goal=PrivacyGoal(protect="All direct identifiers", preserve="General meaning of the text"),
            compute_detection_validity=True,
        )

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "Detection validity judge failed (RuntimeError); detection_valid is unavailable for 2 row(s)." in warnings
    assert "secret Alice" not in caplog.text
    assert result.dataframe[COL_DETECTION_VALID].isna().all()


class _PermutingAdapter:
    """Returns the surviving rows in reverse order, optionally dropping one; each row's payload names its text."""

    def __init__(self, *, drop: int | None = None) -> None:
        self.drop = drop

    def run_workflow(
        self,
        df: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        columns: list[Any],
        workflow_name: str,
        preview_num_records: int | None = None,
    ) -> WorkflowRunResult:
        positions = tuple(i for i in reversed(range(len(df))) if i != self.drop)
        out = df.iloc[list(positions)].reset_index(drop=True)
        for column in columns:
            out[column.name] = [{"row_text": text} for text in out[COL_TEXT]]
        return WorkflowRunResult(out, [], source_positions=positions)


def _texts_trace() -> pd.DataFrame:
    texts = ["Alice lives in Paris", "Bob met Carol in Rome", "Dana works in Oslo"]
    return pd.DataFrame(
        {
            COL_TEXT: texts,
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": text.split()[0], "labels": ["first_name"]}]} for text in texts
            ],
        }
    )


@pytest.mark.parametrize("judge_cls", [DetectionJudgeWorkflow, EntityCoverageWorkflow], ids=lambda c: c.__name__)
def test_each_judge_result_lands_on_its_own_row(
    judge_cls: type[_BaseJudgeWorkflow], stub_evaluate_model_selection: EvaluateModelSelection
) -> None:
    trace = _texts_trace()

    result = judge_cls(_PermutingAdapter(drop=1)).evaluate(
        trace, model_configs=[], selected_models=stub_evaluate_model_selection
    )

    assert result.source_positions == (0, 2)
    assert list(result.dataframe[COL_TEXT]) == [trace[COL_TEXT][0], trace[COL_TEXT][2]]
    assert [raw["row_text"] for raw in result.dataframe[judge_cls.RAW_COL]] == list(result.dataframe[COL_TEXT])


def test_entity_coverage_run_non_critical_keeps_scores_on_their_rows_when_rows_return_reversed(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    trace = _texts_trace()
    adapter = cast(Any, _PermutingAdapter())
    adapter._attach_record_ids = lambda df: df.assign(**{RECORD_ID_COLUMN: [f"r{i}" for i in range(len(df))]})

    out, _ = EntityCoverageWorkflow(adapter).run_non_critical(
        trace, model_configs=[], selected_models=stub_evaluate_model_selection
    )

    assert list(out[COL_TEXT]) == list(trace[COL_TEXT])
    assert [raw["row_text"] for raw in out[EntityCoverageWorkflow.RAW_COL]] == list(trace[COL_TEXT])
