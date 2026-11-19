# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from unittest.mock import Mock

import pandas as pd
import pytest

from anonymizer.config import options as config_options
from anonymizer.config.models import ModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.detection import detection_workflow
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd.model_loader import load_default_model_selection
from anonymizer.interface import workflows
from anonymizer.interface.workflows import DetectionOptions, DetectionWorkflow, EntityDetectionResult, RunContext


class _Conforming:
    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        options: DetectionOptions,
        context: RunContext,
        latent_goal: PrivacyGoal | None = None,
    ) -> EntityDetectionResult:
        return EntityDetectionResult(dataframe=dataframe, failed_records=[])


class _KeywordListDetector:
    def run(self, dataframe: pd.DataFrame, *, gliner_detection_threshold: float) -> EntityDetectionResult:
        return EntityDetectionResult(dataframe=dataframe, failed_records=[])


class _NoLatentDefault:
    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        options: DetectionOptions,
        context: RunContext,
        latent_goal: PrivacyGoal | None,
    ) -> EntityDetectionResult:
        return EntityDetectionResult(dataframe=dataframe, failed_records=[])


class _Swallowing:
    def run(self, dataframe: pd.DataFrame, **_: object) -> EntityDetectionResult:
        return EntityDetectionResult(dataframe=dataframe, failed_records=[])


# Static pins, checked by the type checker: the built-in workflow, a conforming class, a `**_` class and a Mock satisfy
# the Protocol; a keyword-list workflow and one without the `latent_goal` default do not.
_builtin: DetectionWorkflow = EntityDetectionWorkflow(adapter=Mock())
_conforming: DetectionWorkflow = _Conforming()
_swallowing: DetectionWorkflow = _Swallowing()
_mock: DetectionWorkflow = Mock()
_legacy: DetectionWorkflow = _KeywordListDetector()  # ty: ignore[invalid-assignment]
_no_default: DetectionWorkflow = _NoLatentDefault()  # ty: ignore[invalid-assignment]


def _context() -> RunContext:
    return RunContext(model_configs=(), models=load_default_model_selection())


def test_run_context_has_exactly_four_fields() -> None:
    assert [field.name for field in dataclasses.fields(RunContext)] == [
        "model_configs",
        "models",
        "data_summary",
        "preview_num_records",
    ]


def test_run_context_is_frozen() -> None:
    context = _context()
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.data_summary = "changed"  # ty: ignore[invalid-assignment]
    assert isinstance(context.models, ModelSelection)


def test_workflows_module_reexports_the_types_a_workflow_needs() -> None:
    assert workflows.DetectionOptions is config_options.DetectionOptions
    assert workflows.RunContext is config_options.RunContext
    assert workflows.EntityDetectionResult is detection_workflow.EntityDetectionResult
