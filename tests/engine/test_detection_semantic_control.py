# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config import ModelConfig

from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.graph_sdk.requests import TextArtifactValue
from anonymizer.engine.harness.frames import PublishedFrame
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult
from anonymizer.graph import EffectRejected
from anonymizer.interface.errors import AnonymizerWorkflowError


def _arguments(models: list[ModelConfig], selected: DetectionModelSelection, *, latent: bool = False) -> dict[str, Any]:
    return dict(
        model_configs=models,
        selected_models=selected,
        gliner_detection_threshold=0.3,
        tag_latent_entities=latent,
        privacy_goal=PrivacyGoal(protect="Protect all identifying details", preserve="Preserve the useful meaning")
        if latent
        else None,
    )


def test_latent_drop_composes_original_positions_and_failures(
    stub_detector_model_configs: list[ModelConfig], stub_detection_model_selection: DetectionModelSelection
) -> None:
    source = pd.DataFrame({COL_TEXT: ["Alice.", "Bob.", "Carol."]}, index=pd.Index([7, 7, 7]))
    core = source.iloc[[2, 0]].copy()
    core[COL_DETECTED_ENTITIES] = [{"entities": []}, {"entities": []}]
    latent = core.iloc[[1]].copy()
    first = FailedRecord(record_id="core", step="entity-detection", reason="dropped")
    second = FailedRecord(record_id="latent", step="latent-entity-detection", reason="dropped")
    adapter = Mock()
    adapter.run_workflow.side_effect = [
        WorkflowRunResult(core, [first], (2, 0)),
        WorkflowRunResult(latent, [second], (1,)),
    ]
    result = EntityDetectionWorkflow(adapter).run(
        source, **_arguments(stub_detector_model_configs, stub_detection_model_selection, latent=True)
    )
    assert result.source_positions == (0,)
    assert result.dataframe[COL_TEXT].tolist() == ["Alice."]
    assert result.failed_records == [first, second]
    assert result.dataframe[COL_ENTITIES_BY_VALUE].iloc[0] == {"entities_by_value": []}
    assert adapter.run_workflow.call_args_list[0].args[0] is source


def test_core_mutation_rejected_before_latent_dispatch(
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    source = pd.DataFrame({COL_TEXT: ["Alice."]})
    adapter = Mock()

    def core(frame: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        del kwargs
        output = frame.copy()
        output[COL_DETECTED_ENTITIES] = [{"entities": []}]
        output[COL_TEXT] = ["Mallory."]
        return WorkflowRunResult(output, [], (0,))

    adapter.run_workflow.side_effect = core
    with pytest.raises((EffectRejected, AnonymizerWorkflowError)):
        EntityDetectionWorkflow(adapter).run(
            source, **_arguments(stub_detector_model_configs, stub_detection_model_selection, latent=True)
        )
    assert adapter.run_workflow.call_count == 1


def test_direct_final_entities_project_from_canonical_artifacts(
    stub_detector_model_configs: list[ModelConfig], stub_detection_model_selection: DetectionModelSelection
) -> None:
    source = pd.DataFrame({COL_TEXT: ["Alice."]})
    detected = source.copy()
    detected[COL_DETECTED_ENTITIES] = [
        {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
    ]
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(detected, [], (0,))
    result = EntityDetectionWorkflow(adapter).run(
        source, **_arguments(stub_detector_model_configs, stub_detection_model_selection)
    )
    execution = result._execution
    assert execution is not None
    publication = execution.frames[0]
    assert isinstance(publication, PublishedFrame)
    assert COL_FINAL_ENTITIES not in publication._content
    assert COL_ENTITIES_BY_VALUE not in publication._content
    result.dataframe[COL_FINAL_ENTITIES].iloc[0]["entities"].clear()
    projected = publication.project(execution)
    assert projected[COL_FINAL_ENTITIES].iloc[0]["entities"][0]["value"] == "Alice"
    assert projected[COL_ENTITIES_BY_VALUE].iloc[0]["entities_by_value"][0]["value"] == "Alice"
    references = dict(publication._text)[COL_FINAL_ENTITIES]
    artifact = dict(execution.artifacts)[references[0]]
    assert isinstance(artifact, TextArtifactValue)


def test_empty_direct_preserves_adapter_call_and_output_columns(
    stub_detector_model_configs: list[ModelConfig], stub_detection_model_selection: DetectionModelSelection
) -> None:
    source = pd.DataFrame({COL_TEXT: pd.Series(dtype=str)})
    detected = source.copy()
    detected[COL_DETECTED_ENTITIES] = pd.Series(dtype=object)
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(detected, [], ())
    result = EntityDetectionWorkflow(adapter).run(
        source, **_arguments(stub_detector_model_configs, stub_detection_model_selection)
    )
    assert result.source_positions == ()
    assert result.dataframe.empty
    assert COL_FINAL_ENTITIES in result.dataframe
    assert COL_ENTITIES_BY_VALUE in result.dataframe
    assert adapter.run_workflow.call_args.args[0] is source


def test_supplied_source_frame_authenticated_before_effects(
    stub_detector_model_configs: list[ModelConfig], stub_detection_model_selection: DetectionModelSelection
) -> None:
    from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame

    adapter = Mock()
    with pytest.raises(AnonymizerWorkflowError, match="source frame does not match"):
        EntityDetectionWorkflow(adapter).run(
            pd.DataFrame({COL_TEXT: ["Alice."]}),
            source_frame=WorkflowSourceFrame.capture(pd.DataFrame({COL_TEXT: ["Mallory."]})),
            **_arguments(stub_detector_model_configs, stub_detection_model_selection),
        )
    adapter.run_workflow.assert_not_called()


def test_empty_latent_invocation_preserves_callback_objects_and_has_no_source_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    stub_detector_model_configs: list[ModelConfig],
    stub_detection_model_selection: DetectionModelSelection,
) -> None:
    source = pd.DataFrame({COL_TEXT: pd.Series(dtype=str)})
    detected = source.copy()
    detected[COL_DETECTED_ENTITIES] = pd.Series(dtype=object)
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(detected, [], ())
    arguments = _arguments(stub_detector_model_configs, stub_detection_model_selection, latent=True)
    receiver = EntityDetectionWorkflow(adapter)
    identify = receiver.identify_latent_entities
    callback_arguments = []

    def latent(frame: pd.DataFrame, **kwargs: Any) -> EntityDetectionResult:
        callback_arguments.append(kwargs)
        return identify(frame, **kwargs)

    monkeypatch.setattr(receiver, "identify_latent_entities", latent)
    result = receiver.run(source, **arguments)
    assert result.dataframe.empty and result.source_positions == ()
    assert COL_FINAL_ENTITIES in result.dataframe and COL_ENTITIES_BY_VALUE in result.dataframe
    assert adapter.run_workflow.call_args_list[0].args[0] is source
    assert callback_arguments[0]["model_configs"] is stub_detector_model_configs
    assert callback_arguments[0]["privacy_goal"] is arguments["privacy_goal"]
    execution = result._execution
    assert execution is not None and execution.states == ()
    assert execution.invocation_state is not None and execution.invocation_state.complete
    assert not execution.artifacts and not execution.final_outputs
    assert {node.operation.name for node in execution._execution.context.prepared.workflow.operations} == {
        "detection-entry",
        "detection-core",
        "detection-latent",
        "detection-final",
        "empty-detection-lowering",
    }
