# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual replacement consumers share the existing graph invocation owner."""

from __future__ import annotations

import json
import pickle

import httpx
import pandas as pd
import pytest

from anonymizer import Annotate, AnonymizerConfig, AnonymizerInput, Hash, Substitute
from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.detection import _graph_profile
from anonymizer.engine.detection._frame_publication import InvocationFrame, PublishedFrame
from anonymizer.engine.replace._graph_redact import local_strategy
from anonymizer.interface.errors import AnonymizerWorkflowError
from tests.engine import test_detection_source_frame as detection_tests
from tests.engine.test_detection_graph_publication import _modify_dd_responses

scripted_detection = detection_tests.scripted_detection


@pytest.mark.parametrize(
    "strategy,expected",
    [
        (Hash(), "<HASH_FIRST_NAME_3bc51062973c> travels."),
        (Annotate(format_template="[{label}]::{text}"), "[first_name]::Alice travels."),
        (Hash(format_template="{digest:.0}"), "Alice travels."),
        (Annotate(format_template="{text:.0}{label:.0}"), "Alice travels."),
    ],
)
def test_public_local_replacement_keeps_authenticated_frame(
    scripted_detection, monkeypatch, tmp_path, strategy, expected
):
    original = strategy.model_copy(deep=True)
    start = _graph_profile.Thread.start

    def started(thread):
        strategy.format_template = "CHANGED"
        start(thread)

    monkeypatch.setattr(_graph_profile.Thread, "start", started)
    path = tmp_path / "local.csv"
    pd.DataFrame({"text": ["Alice travels."], "extra": [17]}).to_csv(path, index=False)
    facade = detection_tests.public_anonymizer(scripted_detection)
    monkeypatch.setattr(facade._replace_runner, "run", lambda *a, **k: pytest.fail("displaced local dispatch"))
    result = facade.run(
        config=AnonymizerConfig(replace=strategy, emit_telemetry=False), data=AnonymizerInput(source=str(path))
    )
    assert strategy.format_template == "CHANGED"
    configured = [cap.configuration for cap in result._execution._execution.capabilities if cap.configuration.fields]
    assert len(configured) == 1 and local_strategy(configured[0]) == original
    assert result.dataframe.text_replaced.tolist() == [expected]
    assert result.trace_dataframe.extra.tolist() == [17]
    assert isinstance(result._execution.frames[0], PublishedFrame)
    assert result._execution._execution.context.prepared.configuration.purpose == "execution_only"
    assert result._execution._execution.context.prepared.declared_request_upper_bound is None
    assert not result._execution.requests.dispatches
    assert pickle.loads(pickle.dumps(result))._execution is None


@pytest.mark.parametrize("mode", ["create", "preview", "drop", "all_failed", "no_entities"])
def test_public_substitute_runs_actual_dd_and_preserves_map_failures(scripted_detection, monkeypatch, tmp_path, mode):
    maps = []

    def response(request, answer):
        body = json.loads(request.content)
        messages = json.dumps(body["messages"])
        if "Generate synthetic replacements" not in messages:
            return answer
        maps.append(messages)
        assert "keep the row-local instruction" in messages
        if mode == "all_failed" or mode == "drop" and "travels" in messages:
            return httpx.Response(
                400, json={"error": {"message": "scripted map failure", "type": "invalid_request_error"}}
            )
        payload = answer.json()
        payload["choices"][0]["message"]["content"] = (
            '```json\n{"replacements":[{"original":"Alice","label":"first_name","synthetic":"Maya"}]}\n```'
        )
        return httpx.Response(200, json=payload)

    _modify_dd_responses(scripted_detection, monkeypatch, response)
    path = tmp_path / "substitute.csv"
    texts = ["Alice travels."] if mode == "all_failed" else ["No entities here.", "Alice travels.", "Alice stays."]
    if mode == "no_entities":
        texts = ["No entities here."]
    if mode == "drop":
        texts.insert(0, "Alice fails.")
    pd.DataFrame({"text": texts, "extra": list(range(len(texts)))}).to_csv(path, index=False)
    facade = detection_tests.public_anonymizer(scripted_detection)
    facade._selected_models.replace = ReplaceModelSelection(replacement_generator="singleton")
    options = dict(
        config=AnonymizerConfig(
            replace=Substitute(instructions="keep the row-local instruction"), emit_telemetry=False
        ),
        data=AnonymizerInput(source=str(path)),
    )
    if mode == "all_failed":
        with pytest.raises(AnonymizerWorkflowError, match="scripted map failure") as error:
            facade.run(**options)
        execution = getattr(error.value, "_anonymizer_execution")
        assert not execution.frames and not execution.final_outputs
        assert len(maps) == 1
        return
    result = facade.preview(**options, num_records=3) if mode == "preview" else facade.run(**options)
    assert result.dataframe.text_replaced.tolist() == (
        ["No entities here."]
        if mode == "no_entities"
        else ["No entities here.", "Maya stays."]
        if mode == "drop"
        else ["No entities here.", "Maya travels.", "Maya stays."]
    )
    assert result.trace_dataframe.extra.tolist() == (
        [0] if mode == "no_entities" else [1, 3] if mode == "drop" else [0, 1, 2]
    )
    assert len(maps) == (0 if mode == "no_entities" else 2)
    assert [failure.step for failure in result.failed_records] == (
        ["entity-detection", "replace-map-generation"] if mode == "drop" else []
    )
    assert isinstance(result._execution.frames[0], InvocationFrame)
    assert not result._execution.final_outputs and not result._execution.requests.dispatches
    assert result._execution._execution.context.prepared.declared_request_upper_bound is None
    restored = pickle.loads(pickle.dumps(result))
    pd.testing.assert_frame_equal(restored.trace_dataframe, result.trace_dataframe)
    assert restored._execution is None
    artifacts = list((tmp_path / "artifacts").rglob("*.parquet"))
    if mode != "preview":
        assert artifacts
    for artifact in artifacts:
        assert not any(str(column).startswith("_anonymizer_source_") for column in pd.read_parquet(artifact).columns), (
            artifact
        )
