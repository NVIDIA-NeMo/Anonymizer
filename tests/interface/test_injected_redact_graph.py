# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opaque callback execution preserves compatibility without invented provenance."""

from __future__ import annotations

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.replace.replace_runner import ReplacementResult
from tests.interface.test_anonymizer_interface import _make_anonymizer


def _input(tmp_path, rows=2):
    path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["same"] * rows}).to_csv(path, index=False)
    return AnonymizerInput(source=str(path))


@pytest.mark.parametrize(
    "strategy,preview",
    [(Redact(), False), (Redact(), True), (Hash(), False), (Annotate(), True), (Substitute(), False)],
)
def test_injected_redact_keeps_frame_and_calls(tmp_path, strategy, preview):
    detected = pd.DataFrame(
        {COL_TEXT: ["same", "same"], COL_FINAL_ENTITIES: [{"entities": []}] * 2}, index=pd.Index([7, 7])
    )
    cell = object()
    replaced = pd.DataFrame(
        {COL_TEXT: ["same", "same"], COL_REPLACED_TEXT: ["second", "first"], "extra": [cell, cell]},
        index=pd.Index([7, 7]),
    )
    anonymizer, detector, replacement, _ = _make_anonymizer(
        detection_return=EntityDetectionResult(detected, []),
        replace_return=ReplacementResult(replaced, []),
    )
    config = AnonymizerConfig(replace=strategy)
    result = (
        anonymizer.preview(config=config, data=_input(tmp_path), num_records=2)
        if preview
        else anonymizer.run(config=config, data=_input(tmp_path))
    )
    assert detector.run.call_count == replacement.run.call_count == 1
    assert not hasattr(detector, "_adapter")
    assert set(detector.run.call_args.kwargs) == {
        "model_configs",
        "selected_models",
        "gliner_detection_threshold",
        "validation_max_entities_per_call",
        "validation_excerpt_window_chars",
        "entity_labels",
        "excluded_entity_labels",
        "entity_label_examples",
        "privacy_goal",
        "data_summary",
        "tag_latent_entities",
        "compute_grouped_entities",
        "preview_num_records",
    }
    assert replacement.run.call_args.args[0] is detected
    assert replacement.run.call_args.kwargs["replace_method"] is config.replace
    assert result.trace_dataframe["text_replaced"].tolist() == ["second", "first"]
    assert result.trace_dataframe.index.tolist() == [7, 7]
    assert result.trace_dataframe["extra"].iloc[0] is cell


@pytest.mark.parametrize("stage", ["detection", "replacement"])
def test_injected_exception_identity_survives(tmp_path, stage):
    anonymizer, detector, replacement, _ = _make_anonymizer()
    error = ValueError("callback-owned failure")
    (detector if stage == "detection" else replacement).run.side_effect = error
    with pytest.raises(ValueError) as caught:
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=_input(tmp_path, 1))
    assert caught.value is error
    assert detector.run.call_count == 1
    assert replacement.run.call_count == (stage == "replacement")


def test_empty_injected_frame_still_calls_both_callbacks(tmp_path):
    detected = pd.DataFrame({COL_TEXT: []})
    replaced = pd.DataFrame({COL_TEXT: [], COL_REPLACED_TEXT: []})
    anonymizer, detector, replacement, _ = _make_anonymizer(
        detection_return=EntityDetectionResult(detected, []), replace_return=ReplacementResult(replaced, [])
    )
    result = anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=_input(tmp_path, 0))
    assert detector.run.call_count == replacement.run.call_count == 1
    assert result.dataframe.empty


def test_injected_callbacks_keep_caller_thread_context_and_running_loop(tmp_path):
    import asyncio
    import sqlite3
    from contextvars import ContextVar
    from threading import get_ident

    owner = get_ident()
    context: ContextVar[str | None] = ContextVar("caller_marker", default=None)
    connection = sqlite3.connect(":memory:")
    anonymizer, detector, replacement, _ = _make_anonymizer()
    calls = []
    for runner in (detector, replacement):
        returned = runner.run.return_value

        def callback(*args, _returned=returned, **kwargs):
            assert get_ident() == owner
            assert context.get() == "caller"
            assert asyncio.get_running_loop() is loop
            assert connection.execute("select 37").fetchone() == (37,)
            calls.append(_returned)
            context.set("caller")
            return _returned

        runner.run.side_effect = callback

    async def run():
        nonlocal loop
        loop = asyncio.get_running_loop()
        context.set("caller")
        return anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=_input(tmp_path, 1))

    loop = None
    try:
        result = asyncio.run(run())
        assert len(calls) == 2 and result.dataframe is not None
    finally:
        connection.close()
