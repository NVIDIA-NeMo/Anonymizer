# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Existing Redact choices bind both real DD execution and independent checking."""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection import _graph_invocation, _graph_profile
from anonymizer.engine.graph_sdk.executor import ExecutionLimits
from anonymizer.engine.replace._graph_redact import redact_strategy
from anonymizer.engine.replace.strategies import apply_local_replace_strategy
from anonymizer.interface.errors import AnonymizerWorkflowError
from tests.engine import test_detection_source_frame as detection_tests
from tests.engine.test_detection_source_frame import run_graph_profile

scripted_detection = detection_tests.scripted_detection


def _source():
    return pd.DataFrame({COL_TEXT: ["Alice travels.", "Alice stays."], "extra": [17, 29]}, index=pd.Index([4, 4]))


def _options():
    return {"gliner_detection_threshold": 0.3, "tag_latent_entities": False, "preview_num_records": 2}


@pytest.mark.parametrize(
    "template,normalize,replacement",
    [
        ("[REDACTED_{label}]", True, "[REDACTED_FIRST_NAME]"),
        ("[REDACTED_{label}]", False, "[REDACTED_first_name]"),
        ("<{label}>", True, "<FIRST_NAME>"),
        ("<{label}>", False, "<first_name>"),
        ("{{removed}}", True, "{removed}"),
    ],
)
def test_existing_redact_configuration_matches_legacy_and_literal_output(
    scripted_detection, template, normalize, replacement
):
    designer, models, selected, calls = scripted_detection
    strategy = Redact(format_template=template, normalize_label=normalize)
    detected = detection_tests.EntityDetectionWorkflow(detection_tests.NddAdapter(designer)).run(
        _source(), model_configs=models, selected_models=selected, **_options()
    )
    expected = apply_local_replace_strategy(detected.dataframe, strategy=strategy)
    literal = [f"{replacement} travels.", f"{replacement} stays."]
    assert expected[COL_REPLACED_TEXT].tolist() == literal
    calls.clear()
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options(), replace_method=strategy))
    projected = _graph_profile.project_detection_redact(result)
    pd.testing.assert_frame_equal(projected.dataframe, expected)
    assert projected.dataframe[COL_REPLACED_TEXT].tolist() == literal
    assert len(_graph_profile.qualify_detection_redact(result).qualified) == 2
    assert calls == {"detector": 2, "validator": 2, "augmenter": 2, "readiness": 1}
    configured = [cap.configuration for cap in result._execution.capabilities if cap.configuration.fields]
    assert len(configured) == 2
    assert all(redact_strategy(configuration) == strategy for configuration in configured)


@pytest.mark.parametrize("template", ["", "{label:.0}"])
def test_empty_render_preserves_legacy_candidate_but_cannot_claim_protection(scripted_detection, template):
    strategy = Redact(format_template=template)
    designer, models, selected, _ = scripted_detection
    detected = detection_tests.EntityDetectionWorkflow(detection_tests.NddAdapter(designer)).run(
        _source(), model_configs=models, selected_models=selected, **_options()
    )
    legacy = apply_local_replace_strategy(detected.dataframe, strategy=strategy)
    assert legacy[COL_REPLACED_TEXT].tolist() == ["Alice travels.", "Alice stays."]
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options(), replace_method=strategy))
    pd.testing.assert_frame_equal(result.frames[0].project(result), legacy)
    assert [fact.finding.status for fact in result.assessments] == ["unsatisfied", "unsatisfied"]
    assert not _graph_profile.qualify_detection_redact(result).qualified
    with pytest.raises(AnonymizerWorkflowError):
        _graph_profile.project_detection_redact(result)


@pytest.mark.parametrize(
    "wrong", [Redact(format_template="WRONG", normalize_label=False), Redact(format_template="<{label}>")]
)
def test_wrong_producer_configuration_cannot_satisfy_selected_redact(scripted_detection, monkeypatch, wrong):
    def changed(frame, *, strategy):
        return apply_local_replace_strategy(frame, strategy=wrong)

    monkeypatch.setattr(_graph_invocation, "apply_local_replace_strategy", changed)
    result = asyncio.run(
        run_graph_profile(
            scripted_detection,
            _source(),
            _options(),
            replace_method=Redact(format_template="<{label}>", normalize_label=False),
        )
    )
    assert [fact.finding.status for fact in result.assessments] == ["unsatisfied", "unsatisfied"]
    assert not _graph_profile.qualify_detection_redact(result).qualified


def test_async_configuration_snapshot_precedes_resource_acquisition(scripted_detection, monkeypatch):
    strategy = Redact(format_template="<{label}>", normalize_label=False)
    initialize = _graph_profile.NddPhysicalTransport.__init__

    def initialized(self):
        strategy.format_template = "WRONG"
        strategy.normalize_label = True
        initialize(self)

    monkeypatch.setattr(_graph_profile.NddPhysicalTransport, "__init__", initialized)
    result = asyncio.run(run_graph_profile(scripted_detection, _source(), _options(), replace_method=strategy))
    assert strategy.format_template == "WRONG"
    assert _graph_profile.project_detection_redact(result).dataframe[COL_REPLACED_TEXT].tolist() == [
        "<first_name> travels.",
        "<first_name> stays.",
    ]


def test_sync_configuration_snapshot_precedes_host_thread(scripted_detection, monkeypatch):
    strategy = Redact(format_template="<{label}>", normalize_label=False)
    start = _graph_profile.Thread.start

    def started(self):
        strategy.format_template = "WRONG"
        strategy.normalize_label = True
        start(self)

    monkeypatch.setattr(_graph_profile.Thread, "start", started)
    designer, models, selected, calls = scripted_detection
    projected = _graph_profile.run_detection_redact(
        _source(),
        detection=detection_tests.EntityDetectionWorkflow(detection_tests.NddAdapter(designer)),
        model_configs=models,
        selected_models=selected,
        detection_options=_options(),
        replace_method=strategy,
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1,
            max_runtime_artifacts=16,
            max_runtime_artifact_bytes=10000,
            max_collection_items=16,
        ),
    )
    assert strategy.format_template == "WRONG"
    assert projected.dataframe[COL_REPLACED_TEXT].tolist() == ["<first_name> travels.", "<first_name> stays."]
    assert calls == {"detector": 2, "validator": 2, "augmenter": 2, "readiness": 1}
