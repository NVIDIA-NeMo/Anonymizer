# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.run_config import RunConfig
from data_designer.config.utils.storage_schema import JSON_STORAGE_RUN_KEY
from data_designer.engine.models.facade import ModelFacade
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT,
    COL_DOMAIN_SUPPLEMENT_PRIVACY,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_CLASSIFICATION,
    COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS,
    COL_ENTITY_SENSITIVITY,
    COL_FINAL_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_PRIVACY_LEAKAGE_CONTRIBUTIONS,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_ERROR,
    COL_PRIVACY_QA_REANSWER,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_TEXT,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.evaluation.replace.privacy_leakage import (
    PrivacyLeakageEvaluationResult,
    ReplacePrivacyLeakageWorkflow,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.rewrite.entity_classification import EntityClassificationWorkflow
from anonymizer.engine.rewrite.parsers import normalize_payload


def _input() -> pd.DataFrame:
    entities = {
        "entities_by_value": [{"value": "Alice", "labels": ["first_name"]}, {"value": "USA", "labels": ["country"]}]
    }
    return pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["a", "empty", "b"],
            COL_TEXT: ["Alice lives in USA.", "No entities.", "Alice lives in USA."],
            COL_REPLACED_TEXT: ["Maya lives in USA.", "No entities.", "Maya lives in North America."],
            COL_ENTITIES_BY_VALUE: [entities, {"entities_by_value": []}, entities],
        },
        index=pd.Index([9, 4, 2]),
    )


def _adapter() -> Mock:
    adapter = Mock(spec=NddAdapter)
    adapter._attach_record_ids.side_effect = lambda frame: frame.copy()
    return adapter


@pytest.mark.parametrize("strategy", [Substitute(), Redact(), Hash(), Annotate()])
@pytest.mark.parametrize("enabled", [False, True])
def test_replace_runner_dispatches_privacy_only_when_enabled(
    stub_evaluate_model_selection: EvaluateModelSelection,
    strategy: Substitute | Redact | Hash | Annotate,
    enabled: bool,
) -> None:
    source = _input()
    source[COL_REPLACEMENT_MAP] = [{"replacements": []}] * len(source)
    runner = ReplacementWorkflow(adapter=_adapter())
    failure = FailedRecord(record_id="a", step="replace-privacy", reason="Synthetic failure")
    with (
        patch.object(runner, "_run_merged_judges", return_value=source),
        patch("anonymizer.engine.replace.replace_runner.ReplacePrivacyLeakageWorkflow") as privacy,
    ):
        privacy.return_value.run.return_value = PrivacyLeakageEvaluationResult(
            dataframe=source, failed_records=[failure]
        )
        result = runner.evaluate(
            source,
            replace_method=strategy,
            model_configs=[],
            selected_models=stub_evaluate_model_selection,
            compute_privacy_inference=enabled,
            privacy_qa_batch_size=5,
        )
    if enabled:
        privacy.return_value.run.assert_called_once()
        assert privacy.return_value.run.call_args.kwargs["selected_models"] is stub_evaluate_model_selection
        assert privacy.return_value.run.call_args.kwargs["privacy_qa_batch_size"] == 5
        assert result.failed_records == [failure]
    else:
        privacy.assert_not_called()
        assert result.failed_records == []


def test_missing_rows_keep_unavailable_scores_and_order(stub_evaluate_model_selection: EvaluateModelSelection) -> None:
    source = _input()
    source[COL_LEAKAGE_MASS] = 99.0
    source[COL_PRIVACY_QA] = "old questions"
    adapter = _adapter()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame({RECORD_ID_COLUMN: ["b"], COL_LEAKAGE_MASS: [0.0], COL_PRIVACY_QA: [{"items": []}]}),
        failed_records=[FailedRecord(record_id="a", step="replace-privacy", reason="Synthetic failure")],
    )
    result = ReplacePrivacyLeakageWorkflow(adapter).run(
        source, model_configs=[], selected_models=stub_evaluate_model_selection
    )
    frame = result.dataframe
    assert frame.index.tolist() == source.index.tolist()
    assert frame[RECORD_ID_COLUMN].tolist() == ["a", "empty", "b"]
    assert frame[COL_LEAKAGE_MASS].tolist() == [None, 0.0, 0.0]
    assert frame[COL_PRIVACY_QA].tolist() == [None, {"items": []}, {"items": []}]
    assert frame[COL_PRIVACY_QA_ERROR].iloc[0]
    assert len(result.failed_records) == 1
    seed = adapter.run_workflow.call_args.args[0]
    assert seed[RECORD_ID_COLUMN].tolist() == ["a", "b"]
    assert COL_LEAKAGE_MASS not in seed
    assert COL_PRIVACY_QA not in seed
    assert source[COL_LEAKAGE_MASS].tolist() == [99.0] * 3


def test_whole_workflow_failure_preserves_all_records(stub_evaluate_model_selection: EvaluateModelSelection) -> None:
    adapter = _adapter()
    adapter.run_workflow.side_effect = RuntimeError("Synthetic failure")
    result = ReplacePrivacyLeakageWorkflow(adapter).run(
        _input(), model_configs=[], selected_models=stub_evaluate_model_selection
    )
    assert result.dataframe[COL_LEAKAGE_MASS].tolist() == [None, 0.0, None]
    assert {failure.record_id for failure in result.failed_records} == {"a", "b"}


def test_empty_entities_make_no_model_workflow_call(stub_evaluate_model_selection: EvaluateModelSelection) -> None:
    adapter = _adapter()
    source = _input().iloc[[1]]
    result = ReplacePrivacyLeakageWorkflow(adapter).run(
        source, model_configs=[], selected_models=stub_evaluate_model_selection
    )
    adapter.run_workflow.assert_not_called()
    assert result.dataframe[COL_LEAKAGE_MASS].tolist() == [0.0]
    assert result.dataframe[COL_PRIVACY_LEAKAGE_CONTRIBUTIONS].tolist() == [[]]


def test_malformed_entity_row_does_not_discard_other_rows(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    adapter = _adapter()
    source = _input().iloc[:2].copy()
    source.at[9, COL_ENTITIES_BY_VALUE] = "invalid JSON"
    result = ReplacePrivacyLeakageWorkflow(adapter).run(
        source, model_configs=[], selected_models=stub_evaluate_model_selection
    )
    adapter.run_workflow.assert_not_called()
    assert result.dataframe[COL_LEAKAGE_MASS].tolist() == [None, 0.0]
    assert result.failed_records[0].record_id == "a"


@pytest.mark.parametrize("fails", [False, True])
def test_custom_categories_are_classified_once_upfront_for_the_entire_dataset(
    stub_evaluate_model_selection: EvaluateModelSelection, fails: bool
) -> None:
    source = _input()
    source.at[9, COL_ENTITIES_BY_VALUE] = {
        "entities_by_value": [
            {"value": "Alice", "labels": ["first_name"]},
            {"value": "A001", "labels": ["custom_badge"]},
        ]
    }
    source.at[2, COL_ENTITIES_BY_VALUE] = {
        "entities_by_value": [
            {"value": "A002", "labels": ["custom_badge"]},
            {"value": "Engineering", "labels": ["custom_team"]},
        ]
    }
    categories = {
        "first_name": "direct_identifier",
        "custom_badge": "direct_identifier",
        "custom_team": "quasi_identifier",
    }
    adapter = _adapter()
    classification = WorkflowRunResult(
        dataframe=pd.DataFrame({COL_ENTITY_CLASSIFICATION: [categories]}),
        failed_records=[
            FailedRecord(record_id="taxonomy", step="replace-label-classification", reason="Synthetic failure")
        ]
        if fails
        else [],
    )
    privacy = WorkflowRunResult(
        dataframe=pd.DataFrame({RECORD_ID_COLUMN: ["a", "b"], COL_LEAKAGE_MASS: [0.0, 0.0]}), failed_records=[]
    )
    adapter.run_workflow.side_effect = [classification, privacy]
    result = ReplacePrivacyLeakageWorkflow(adapter).run(
        source, model_configs=[], selected_models=stub_evaluate_model_selection
    )
    first = adapter.run_workflow.call_args_list[0]
    assert first.kwargs["workflow_name"] == "replace-label-classification"
    assert len(first.args[0]) == 1
    assert first.args[0].iloc[0][COL_ENTITY_CLASSIFICATION_UNMAPPED_LABELS] == ["custom_badge", "custom_team"]
    if fails:
        assert adapter.run_workflow.call_count == 1
        assert result.dataframe[COL_LEAKAGE_MASS].tolist() == [None, 0.0, None]
    else:
        assert adapter.run_workflow.call_count == 2
        second = adapter.run_workflow.call_args_list[1]
        assert second.kwargs["workflow_name"] == "replace-privacy"
        assert second.kwargs["columns"][0].generator_params.categories == categories


@pytest.mark.parametrize("override", [None, "separate-classifier"])
def test_classifier_model_defaults_to_sensitivity_analyzer_with_optional_override(
    stub_evaluate_model_selection: EvaluateModelSelection, override: str | None
) -> None:
    selection = stub_evaluate_model_selection.model_copy(
        update={"privacy_entity_classifier": override, "privacy_sensitivity_analyzer": "sensitivity-model"}
    )
    column = EntityClassificationWorkflow().columns(selected_models=selection)[1]
    assert isinstance(column, CustomColumnConfig)
    assert column.model_aliases == [override or "sensitivity-model"]


async def _stub_privacy_generation(
    self: ModelFacade, *, parser: Callable[[str], Any], prompt: str, purpose: str = "", **kwargs: Any
) -> tuple[Any, list[Any]]:
    if "Assess the re-identification sensitivity" in prompt:
        payload = {
            "entities": [
                {
                    "id": idx,
                    "sensitivity": sensitivity,
                    "reason": "Contextual sensitivity assessment.",
                }
                for idx, category, sensitivity in [(1, "direct_identifier", "high"), (2, "quasi_identifier", "low")]
            ]
        }
    elif "You are a privacy auditor." in prompt:
        if "<text>\nMODEL_FAILURE\n</text>" in prompt:
            raise RuntimeError("Synthetic privacy answer failure")
        leaked = "<text>\nMaya lives in USA.\n</text>" in prompt
        payload = {
            "answers": [
                {"id": 1, "answer": "no", "confidence": 1.0, "reason": "Original name absent.", "evidence": []},
                {
                    "id": 2,
                    "answer": "yes" if leaked else "no",
                    "confidence": 1.0,
                    "reason": "Country assessment.",
                    "evidence": ["USA"] if leaked else [],
                },
            ]
        }
        template = json.loads(prompt.split("<answer_template>")[1].split("</answer_template>")[0])
        ids = {item["id"] for item in template["answers"]}
        payload["answers"] = [answer for answer in payload["answers"] if answer["id"] in ids]
    else:
        raise AssertionError("Unexpected model call in Replace privacy evaluation")
    return parser(json.dumps(payload)), []


@pytest.mark.parametrize("fail_answers", [False, True])
@pytest.mark.parametrize("batch_size", [None, 1])
def test_pipeline_through_real_adapter_and_repeated_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_evaluate_model_selection: EvaluateModelSelection,
    fail_answers: bool,
    batch_size: int | None,
) -> None:
    monkeypatch.setattr(ModelFacade, "agenerate", _stub_privacy_generation)
    monkeypatch.setattr("data_designer.engine.dataset_builders.dataset_builder.run_readiness_check", Mock())
    designer = DataDesigner(
        artifact_path=tmp_path,
        model_providers=[ModelProvider(name="synthetic", endpoint="http://127.0.0.1:9/v1")],
        auto_configure_logging=False,
    )
    designer.set_run_config(RunConfig(buffer_size=1))
    workflow = ReplacePrivacyLeakageWorkflow(NddAdapter(designer))
    selection = stub_evaluate_model_selection.model_copy(
        update={
            "privacy_sensitivity_analyzer": "synthetic",
            "privacy_qa_judge": "synthetic",
        }
    )
    models = [ModelConfig(alias="synthetic", model="synthetic", provider="synthetic")]
    source = _input()
    if fail_answers:
        source.at[2, COL_REPLACED_TEXT] = "MODEL_FAILURE"
    for column in (COL_DOMAIN, COL_DOMAIN_SUPPLEMENT, COL_DOMAIN_SUPPLEMENT_PRIVACY):
        source[column] = "stale domain output"
    source[COL_FINAL_ENTITIES] = [{"entities": []}] * len(source)
    source.attrs[JSON_STORAGE_RUN_KEY] = {
        "version": 1,
        "columns": {COL_FINAL_ENTITIES: [[]], COL_ENTITIES_BY_VALUE: [[]]},
    }
    first = workflow.run(source, model_configs=models, selected_models=selection, privacy_qa_batch_size=batch_size)
    assert first.failed_records == []
    assert not {COL_DOMAIN, COL_DOMAIN_SUPPLEMENT, COL_DOMAIN_SUPPLEMENT_PRIVACY} & set(first.dataframe.columns)
    assert first.dataframe[COL_LEAKAGE_MASS].tolist() == [0.3, 0.0, None if fail_answers else 0.0]
    assert len(normalize_payload(first.dataframe[COL_PRIVACY_QA].iloc[0])["items"]) == 2
    assert normalize_payload(first.dataframe[COL_ENTITY_SENSITIVITY].iloc[0])["entities"][1]["sensitivity"] == "low"
    assert first.dataframe[COL_PRIVACY_QA_ERROR].iloc[:2].tolist() == [None, None]
    if fail_answers:
        assert "unavailable" in first.dataframe[COL_PRIVACY_QA_ERROR].iloc[2]
    else:
        assert first.dataframe[COL_PRIVACY_QA_ERROR].iloc[2] is None
    changed = first.dataframe.copy()
    changed.at[9, COL_REPLACED_TEXT] = "Maya lives in North America."
    second = workflow.run(changed, model_configs=models, selected_models=selection, privacy_qa_batch_size=batch_size)
    assert second.failed_records == []
    assert second.dataframe[COL_LEAKAGE_MASS].tolist() == [0.0, 0.0, None if fail_answers else 0.0]
    assert normalize_payload(second.dataframe[COL_PRIVACY_QA_REANSWER].iloc[0])["answers"][1]["answer"] == "no"
    assert first.dataframe[COL_LEAKAGE_MASS].iloc[0] == 0.3
