# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The actual DD consumer derives count capacity without implicit output-byte limits."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pandas as pd
import pytest

from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection import _graph_profile
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.graph_sdk.executor import ExecutionLimits
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.graph._values import ContractViolation
from anonymizer.interface.errors import AnonymizerWorkflowError
from tests.engine import test_detection_source_frame as detection_tests
from tests.engine.test_detection_graph_publication import _modify_dd_responses

scripted_detection = detection_tests.scripted_detection


def test_default_profile_retains_large_valid_labels_and_actual_byte_observations(scripted_detection, monkeypatch):
    designer, models, selected, calls = scripted_detection
    label = "L" * 7000

    def modify(request, answer):
        if json.loads(request.content)["model"] == "detector":
            payload = answer.json()
            content = json.loads(payload["choices"][0]["message"]["content"])
            for entity in content["entities"]:
                entity["label"] = label
            payload["choices"][0]["message"]["content"] = json.dumps(content)
            return httpx.Response(200, json=payload)
        return answer

    _modify_dd_responses(scripted_detection, monkeypatch, modify)
    arguments: dict[str, Any] = dict(
        detection=EntityDetectionWorkflow(NddAdapter(designer)),
        model_configs=models,
        selected_models=selected,
        detection_options={"gliner_detection_threshold": 0.3, "tag_latent_entities": False, "preview_num_records": 1},
    )
    source = pd.DataFrame({COL_TEXT: ["Alice travels."], "extra": [17]})
    projected = _graph_profile.run_detection_redact(source, **arguments)
    assert projected.dataframe[COL_REPLACED_TEXT].tolist() == [f"[REDACTED_{label}] travels."]
    assert projected.dataframe["extra"].tolist() == [17]
    result = projected._execution
    assert result is not None and len(result.artifacts) == 5
    assert sum(value.shape.content_bytes for _, value in result.artifacts) > 21000
    assert result._execution.ingress_limits.max_text_bytes is None
    assert result._execution.ingress_limits.max_metadata_bytes is None
    assert calls == {"detector": 1, "validator": 1, "augmenter": 1, "readiness": 1}
    with pytest.raises(AnonymizerWorkflowError, match="did not qualify"):
        _graph_profile.run_detection_redact(
            source,
            **arguments,
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=8,
                max_runtime_artifacts=5,
                max_runtime_artifact_bytes=10000,
                max_collection_items=16,
            ),
        )


def test_profile_association_capacity_comes_from_selected_sources(scripted_detection, monkeypatch):
    designer, models, selected, calls = scripted_detection

    class Captured(Exception):
        pass

    def admission(**kwargs):
        limits = kwargs["ingress_limits"]
        assert limits.max_associations == 1025
        assert limits.max_collection_items == 1025 * (len("Alice travels.") + 1)
        assert limits.max_text_bytes is limits.max_metadata_bytes is None
        raise Captured

    monkeypatch.setattr(_graph_profile, "admit_execution_plan", admission)
    with pytest.raises(Captured):
        _graph_profile.run_detection_redact(
            pd.DataFrame({COL_TEXT: ["Alice travels."] * 1025}),
            detection=EntityDetectionWorkflow(NddAdapter(designer)),
            model_configs=models,
            selected_models=selected,
            detection_options={"gliner_detection_threshold": 0.3, "tag_latent_entities": False},
        )
    assert not calls


@pytest.mark.parametrize("rows", [0, 2])
def test_default_count_policy_preserves_empty_failure_and_complete_inventory(scripted_detection, rows):
    designer, models, selected, _ = scripted_detection
    source = pd.DataFrame({COL_TEXT: ["Alice travels."] * rows})
    arguments: dict[str, Any] = dict(
        detection=EntityDetectionWorkflow(NddAdapter(designer)),
        model_configs=models,
        selected_models=selected,
        detection_options={
            "gliner_detection_threshold": 0.3,
            "tag_latent_entities": False,
            "preview_num_records": rows,
        },
    )
    if not rows:
        with pytest.raises(AnonymizerWorkflowError, match="Dataset is empty"):
            _graph_profile.run_detection_redact(source, **arguments)
    else:
        projected = _graph_profile.run_detection_redact(source, **arguments)
        assert len(projected.dataframe) == rows
        assert projected._execution is not None
        assert len(projected._execution.artifacts) == 5 * rows


def test_default_policy_preserves_canonical_unicode_rejection_before_effects(scripted_detection):
    designer, models, selected, calls = scripted_detection
    with pytest.raises(ContractViolation):
        _graph_profile.run_detection_redact(
            pd.DataFrame({COL_TEXT: ["\ud800"]}),
            detection=EntityDetectionWorkflow(NddAdapter(designer)),
            model_configs=models,
            selected_models=selected,
            detection_options={"gliner_detection_threshold": 0.3, "tag_latent_entities": False},
        )
    assert not calls
