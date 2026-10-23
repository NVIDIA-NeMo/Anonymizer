# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pandas as pd
from data_designer.config import ModelConfig

from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult


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
