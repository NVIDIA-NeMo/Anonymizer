# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded witnesses for the production facade and its DataDesigner boundary."""

from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from data_designer.config.column_configs import LLMTextColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.models import ModelConfig
from data_designer.config.seed_source import LocalFileSeedSource
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.clients.types import AssistantMessage, ChatCompletionRequest, ChatCompletionResponse
from data_designer.engine.models.request_admission.controller import AdaptiveRequestAdmissionController
from data_designer.interface.data_designer import DataDesigner

from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Redact
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE_JUDGE,
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.interface.errors import AnonymizerWorkflowError
from anonymizer.interface.results import AnonymizerResult


@pytest.fixture
def local_route(
    tmp_path: Path, stub_dataframe_with_entities: pd.DataFrame
) -> tuple[Anonymizer, AnonymizerConfig, AnonymizerInput, Mock]:
    """Reuse canonical synthetic entities; leave replacement and projections real."""
    detected = pd.concat(
        [
            stub_dataframe_with_entities,
            pd.DataFrame({COL_TEXT: ["Nothing private."], COL_FINAL_ENTITIES: [{"entities": []}]}),
        ],
        ignore_index=True,
    )
    detected[COL_ENTITIES_BY_VALUE] = [
        {"entities": [{"value": "Alice", "label": "first_name"}, {"value": "Acme", "label": "organization"}]},
        {"entities": []},
    ]
    detected[COL_TAGGED_TEXT] = [
        "<first_name>Alice</first_name> works at <organization>Acme</organization>",
        "Nothing private.",
    ]
    detected[RECORD_ID_COLUMN] = ["kept-1", "kept-2"]
    detector = Mock(spec=EntityDetectionWorkflow)
    detector.run.return_value = EntityDetectionResult(
        detected, [FailedRecord(record_id="dropped-3", step="detection", reason="synthetic timeout")]
    )
    source = tmp_path / "input.csv"
    pd.DataFrame({"biography": ["Alice works at Acme", "Nothing private.", "Dropped synthetic row."]}).to_csv(
        source, index=False
    )
    return (
        Anonymizer(data_designer=Mock(spec=DataDesigner), detection_workflow=detector),
        AnonymizerConfig(replace=Redact(format_template="<{label}>", normalize_label=False)),
        AnonymizerInput(source=str(source), text_column="biography", data_summary="Synthetic biographies"),
        detector,
    )


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_local_redact_preserves_projection_passthrough_and_failure(
    local_route: tuple[Anonymizer, AnonymizerConfig, AnonymizerInput, Mock], preview: bool
) -> None:
    anonymizer, config, data, detector = local_route
    result = (
        anonymizer.preview(config=config, data=data, num_records=3)
        if preview
        else anonymizer.run(config=config, data=data)
    )

    assert list(result.dataframe.columns) == [
        "biography",
        "final_entities",
        "biography_with_spans",
        "biography_replaced",
    ]
    assert result.dataframe["biography_replaced"].tolist() == [
        "<first_name> works at <organization>",
        "Nothing private.",
    ]
    assert result.trace_dataframe[RECORD_ID_COLUMN].tolist() == ["kept-1", "kept-2"]
    assert result.trace_dataframe[COL_REPLACEMENT_MAP].tolist() == [
        {
            "replacements": [
                {"original": "Alice", "label": "first_name", "synthetic": "<first_name>"},
                {"original": "Acme", "label": "organization", "synthetic": "<organization>"},
            ]
        },
        {"replacements": []},
    ]
    assert [(f.record_id, f.step, f.reason) for f in result.failed_records] == [
        ("dropped-3", "detection", "synthetic timeout")
    ]
    assert result.resolved_text_column == "biography"
    assert result.data_summary == "Synthetic biographies"
    assert result.replace_method == Redact(format_template="<{label}>", normalize_label=False)
    assert detector.run.call_args.kwargs["preview_num_records"] == (3 if preview else None)
    cast(Mock, anonymizer._data_designer).create.assert_not_called()
    cast(Mock, anonymizer._data_designer).preview.assert_not_called()


def test_saved_result_evaluation_uses_trace_and_preserves_unavailable_rows(
    local_route: tuple[Anonymizer, AnonymizerConfig, AnonymizerInput, Mock], tmp_path: Path
) -> None:
    anonymizer, config, data, _ = local_route
    original = anonymizer.run(config=config, data=data)
    saved = tmp_path / "result.pkl"
    saved.write_bytes(pickle.dumps(original))
    loaded = pickle.loads(saved.read_bytes())
    evaluator = Anonymizer(data_designer=Mock(spec=DataDesigner))
    with patch.object(evaluator._adapter, "run_workflow", side_effect=_return_partial_coverage) as workflow:
        evaluated = evaluator.evaluate(loaded)

    assert isinstance(evaluated, AnonymizerResult)
    assert evaluated.dataframe["biography"].tolist() == ["Alice works at Acme", "Nothing private."]
    assert evaluated.dataframe["biography_replaced"].tolist() == [
        "<first_name> works at <organization>",
        "Nothing private.",
    ]
    assert evaluated.dataframe["entity_coverage"].iloc[0] == 1.0
    assert pd.isna(evaluated.dataframe["entity_coverage"].iloc[1])
    assert evaluated.dataframe["missed_entities"].tolist() == [[], []]
    assert "detection_valid" not in evaluated.dataframe
    assert [(f.record_id, f.step, f.reason) for f in evaluated.failed_records] == [
        ("kept-2", "replace-judges", "synthetic drop")
    ]
    assert evaluated.resolved_text_column == "biography"
    assert evaluated.replace_method == original.replace_method
    assert evaluated.data_summary == "Synthetic biographies"
    pd.testing.assert_frame_equal(loaded.dataframe, original.dataframe)
    pd.testing.assert_frame_equal(loaded.trace_dataframe, original.trace_dataframe)
    workflow.assert_called_once()
    assert workflow.call_args.kwargs["workflow_name"] == "replace-judges"
    cast(Mock, evaluator._data_designer).create.assert_not_called()


def _return_partial_coverage(dataframe: pd.DataFrame, **_kwargs: object) -> WorkflowRunResult:
    assert dataframe[COL_TEXT].tolist() == ["Alice works at Acme", "Nothing private."]
    assert dataframe[COL_REPLACED_TEXT].tolist() == ["<first_name> works at <organization>", "Nothing private."]
    output = dataframe.iloc[:1].copy()
    output[COL_ENTITY_COVERAGE_JUDGE] = [
        {
            "candidate_entities": [
                {"value": "Alice", "label": "first_name", "reasoning": "Synthetic name"},
                {"value": "Acme", "label": "organization", "reasoning": "Synthetic organization"},
            ]
        }
    ]
    return WorkflowRunResult(output, [FailedRecord(record_id="kept-2", step="replace-judges", reason="synthetic drop")])


@pytest.mark.parametrize("exhausted", [False, True], ids=["retry-success", "retry-exhausted"])
def test_one_adapter_workflow_can_contain_two_backend_attempts(
    stub_model_configs: list[ModelConfig], exhausted: bool
) -> None:
    """Real DD retry policy; mocked provider and DD orchestration, no network."""
    frame = pd.DataFrame({COL_TEXT: ["synthetic input"], RECORD_ID_COLUMN: ["kept-1"]})
    backend, provider, admission, failure = _make_retry_backend(exhausted=exhausted)
    adapter = NddAdapter(data_designer=backend)
    with (
        patch.object(admission, "acquire_sync", wraps=admission.acquire_sync) as acquire,
        patch.object(admission, "release", wraps=admission.release) as release,
    ):
        columns: list[ColumnConfigT] = [
            LLMTextColumnConfig(name="output", prompt="Synthetic", model_alias="stub-model")
        ]
        run = partial(
            adapter.run_workflow,
            frame,
            model_configs=stub_model_configs,
            columns=columns,
            workflow_name="m0-seam",
            preview_num_records=1,
        )
        if exhausted:
            with pytest.raises(AnonymizerWorkflowError, match="synthetic connection failure") as error:
                run()
            assert error.value.__cause__ is failure
        else:
            result = run()
            assert result.failed_records == []
            pd.testing.assert_frame_equal(result.dataframe, frame)
    backend.preview.assert_called_once()
    assert provider.completion.call_count == acquire.call_count == release.call_count == 2
    attempts = [call.args[0].event_context.request_attempt_id for call in acquire.call_args_list]
    assert len(set(attempts)) == 2
    assert [call.args[1].kind for call in release.call_args_list] == [
        "provider_failure",
        "provider_failure" if exhausted else "success",
    ]
    provider.close.assert_not_called()


def _make_retry_backend(*, exhausted: bool) -> tuple[Mock, Mock, AdaptiveRequestAdmissionController, ProviderError]:
    provider = Mock(spec=ModelClient)
    failure = ProviderError(kind=ProviderErrorKind.API_CONNECTION, message="synthetic connection failure")
    response = ChatCompletionResponse(message=AssistantMessage(content="synthetic output"))
    provider.completion.side_effect = [failure, failure if exhausted else response]
    admission = AdaptiveRequestAdmissionController()
    admission.register(provider_name="stub", model_id="stub-model", alias="stub-model", max_parallel_requests=1)
    executor = ModelRequestExecutor(
        inner=provider,
        request_admission=admission,
        provider_name="stub",
        model_id="stub-model",
        retry_config=RetryConfig(max_retries=1, backoff_factor=0),
    )
    backend = Mock(spec=DataDesigner)

    def preview(_builder: DataDesignerConfigBuilder, *, num_records: int) -> SimpleNamespace:
        executor.completion(ChatCompletionRequest(model="stub-model", messages=[]))
        seed = _builder.get_seed_config()
        assert seed is not None and isinstance(seed.source, LocalFileSeedSource)
        return SimpleNamespace(dataset=pd.read_parquet(seed.source.path).iloc[:num_records].copy())

    backend.preview.side_effect = preview
    return backend, provider, admission, failure
