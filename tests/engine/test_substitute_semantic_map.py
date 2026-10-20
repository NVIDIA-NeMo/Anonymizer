# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replacement map control, opaque callbacks and accepted publication boundaries."""

from __future__ import annotations

from threading import current_thread, main_thread
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_TEXT,
)
from anonymizer.engine.graph_sdk._execution_delegated import source_worker_borrow
from anonymizer.engine.harness.frames import FramePublication, InvocationFrame, PublishedFrame
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult
from anonymizer.engine.replace._semantic_map import MapBinding
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceResult, LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from tests.engine import test_detection_source_frame as detection_tests

scripted_detection = detection_tests.scripted_detection

pytestmark = pytest.mark.compat


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
            COL_FINAL_ENTITIES: [
                {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
            ],
            "number": pd.Series([pd.NA], dtype="Int64"),
            "category": pd.Categorical(["kept"]),
        }
    )


def _raw() -> dict[str, Any]:
    return {
        "replacements": [
            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
            {"original": "Unrequested", "label": "first_name", "synthetic": "Nora"},
        ]
    }


@pytest.mark.parametrize(
    "raw", [pd.NA, np.int64(1), {"replacements": np.array(_raw()["replacements"], dtype=object), "ignored": object()}]
)
def test_maps_preserve_permissive_payloads_and_extension_dtypes(raw: Any) -> None:
    source = _source()
    adapter = Mock()

    def returned(rows: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        output = rows.copy()
        output[COL_REPLACEMENT_MAP] = [raw]
        return WorkflowRunResult(output, [], (0,))

    adapter.run_workflow.side_effect = returned
    result = LlmReplaceWorkflow(adapter).generate_map_only(
        source, model_configs=[], selected_models=ReplaceModelSelection(replacement_generator="test")
    )
    assert result.dataframe.number.dtype == source.number.dtype
    assert result.dataframe.category.dtype == source.category.dtype
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == (
        {"replacements": [_raw()["replacements"][0]]} if isinstance(raw, dict) else {"replacements": []}
    )
    assert result._execution is not None and isinstance(result._execution.frames[0], PublishedFrame)


def test_direct_delegating_map_runs_opaque_and_its_builtin_standalone() -> None:
    source = _source()
    calls = []
    owners = []
    adapter = Mock()

    def returned(rows: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        owners.append(source_worker_borrow().owner)
        output = rows.copy()
        output[COL_REPLACEMENT_MAP] = [_raw()]
        return WorkflowRunResult(output, [], (0,))

    adapter.run_workflow.side_effect = returned

    class DelegatingMap(LlmReplaceWorkflow):
        def generate_map_only(self, dataframe: pd.DataFrame, **kwargs: Any) -> LlmReplaceResult:
            rows = dataframe
            calls.append((rows, current_thread()))
            return super().generate_map_only(rows, **kwargs)

    result = ReplacementWorkflow(llm_workflow=DelegatingMap(adapter)).run(
        source,
        replace_method=Substitute(),
        model_configs=[],
        selected_models=ReplaceModelSelection(replacement_generator="test"),
    )
    assert len(calls) == 1 and calls[0][0] is source and calls[0][1] is main_thread()
    assert len(owners) == 1
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Maya"]


def test_direct_opaque_map_keeps_callback_frame_failures_and_thread() -> None:
    source = _source()
    calls = []
    failure = FailedRecord("record", "custom-map", "reason")

    class OpaqueMap(LlmReplaceWorkflow):
        def generate_map_only(self, dataframe: pd.DataFrame, **kwargs: Any) -> LlmReplaceResult:
            rows = dataframe
            calls.append((rows, current_thread(), kwargs))
            output = rows.copy()
            output[COL_REPLACEMENT_MAP] = [_raw()]
            return LlmReplaceResult(output, [failure])

    models = []
    selected = ReplaceModelSelection(replacement_generator="test")
    result = ReplacementWorkflow(llm_workflow=OpaqueMap(Mock())).run(
        source,
        replace_method=Substitute(instructions="keep"),
        model_configs=models,
        selected_models=selected,
        preview_num_records=4,
    )
    assert len(calls) == 1 and calls[0][0] is source and calls[0][1] is main_thread()
    assert calls[0][2]["model_configs"] is models and calls[0][2]["selected_models"] is selected
    assert calls[0][2]["instructions"] == "keep" and calls[0][2]["preview_num_records"] == 4
    assert result.failed_records == [failure] and result.dataframe[COL_REPLACED_TEXT].tolist() == ["Maya"]
    assert result._execution is not None and isinstance(result._execution.frames[0], InvocationFrame)


def test_raw_generation_cannot_be_substituted_for_accepted_map_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    generated = []
    publish = MapBinding.publish
    capture = FramePublication.capture.__func__

    def observed(self: MapBinding, item: Any, **kwargs: Any) -> Any:
        result = publish(self, item, **kwargs)
        if kwargs.get("raw_map") is not None:
            generated.append((item.association, kwargs["raw_map"]))
        return result

    def changed(cls: Any, *args: Any, **kwargs: Any) -> Any:
        publication = capture(cls, *args, **kwargs)
        if generated:
            association, value = generated[0]
            object.__setattr__(publication, "associations", (association,))
            object.__setattr__(publication, "_text", ((COL_REPLACEMENT_MAP, "raw_map", (value,)),))
        return publication

    adapter = Mock()

    def returned(rows: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        output = rows.copy()
        output[COL_REPLACEMENT_MAP] = [_raw()]
        return WorkflowRunResult(output, [], (0,))

    adapter.run_workflow.side_effect = returned
    monkeypatch.setattr(MapBinding, "publish", observed)
    monkeypatch.setattr(FramePublication, "capture", classmethod(changed))
    with pytest.raises(RuntimeError, match="semantic publication"):
        LlmReplaceWorkflow(adapter).generate_map_only(
            _source(), model_configs=[], selected_models=ReplaceModelSelection(replacement_generator="test")
        )
