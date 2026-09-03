# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_PHASE6_AUGMENTED,
    COL_PHASE6_CONTEXT,
    COL_PHASE6_VALIDATION,
    COL_RAW_DETECTED,
    COL_TEXT,
    _jinja,
)
from anonymizer.engine.execution.graph import _DatumId
from anonymizer.engine.execution.mention_admission import (
    _CandidateToken,
    _MentionProvenance,
    _MentionTarget,
    _MentionTargetToken,
    _ProvisionalCandidate,
)
from anonymizer.engine.execution.phase6_ndd_backend import (
    _Phase6NddBackend,
    _Phase6NddStageError,
)
from anonymizer.engine.execution.phase6_runtime import (
    _Phase6AugmentationWork,
    _Phase6CandidateWork,
    _Phase6ValidationWork,
)
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from tests.streaming.structured_trace_prototype import build_synthetic_anonymizer


@dataclass
class _ScriptedAdapter:
    detector_payload: object
    calls: list[tuple[str, tuple[str, ...]]]
    prompts: list[str] = field(default_factory=list)
    augmentation_payload: object = field(default_factory=lambda: {"entities": []})
    validation_payload: object = field(
        default_factory=lambda: {"decisions": [{"ordinal": 0, "decision": "keep", "proposed_label": None}]}
    )

    def run_workflow(
        self,
        dataframe: pd.DataFrame,
        *,
        workflow_name: str,
        columns: list[Any],
        **_: Any,
    ) -> WorkflowRunResult:
        self.calls.append((workflow_name, tuple(dataframe.columns)))
        self.prompts.extend(str(column.prompt) for column in columns)
        output = dataframe.copy()
        if workflow_name == "phase6-detect":
            output[COL_RAW_DETECTED] = [self.detector_payload]
        elif workflow_name == "phase6-augment":
            output[COL_PHASE6_AUGMENTED] = [self.augmentation_payload]
        elif workflow_name == "phase6-validate":
            output[COL_PHASE6_VALIDATION] = [self.validation_payload]
        else:  # pragma: no cover - challenge guard
            raise AssertionError(workflow_name)
        return WorkflowRunResult(output, [])


def _backend(adapter: _ScriptedAdapter) -> _Phase6NddBackend:
    anonymizer = build_synthetic_anonymizer({})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    return _Phase6NddBackend(cast(NddAdapter, adapter), plan.invocation)


def test_phase6_ndd_detector_uses_only_target_text_and_preserves_exact_spans() -> None:
    payload = json.dumps({"entities": [{"text": "Alice", "label": "name", "start": 0, "end": 5, "score": 0.9}]})
    adapter = _ScriptedAdapter(payload, [])
    backend = _backend(adapter)
    target = _MentionTarget(_MentionTargetToken(), _DatumId("target"), "Alice met Bob")

    proposals = backend.detect(_Phase6CandidateWork(target))

    assert tuple((item.start, item.end, item.source_slice, item.detector_label) for item in proposals) == (
        (0, 5, "Alice", "name"),
    )
    assert adapter.calls == [("phase6-detect", (COL_TEXT,))]


@pytest.mark.parametrize(
    "payload",
    [
        {"entities": [{"text": "Alice", "label": "name", "start": 0}]},
        {"entities": "not-a-list"},
        {"entities": [], "unexpected": True},
    ],
)
def test_phase6_ndd_detector_rejects_malformed_payload_without_partial_candidates(payload: object) -> None:
    backend = _backend(_ScriptedAdapter(payload, []))
    target = _MentionTarget(_MentionTargetToken(), _DatumId("target"), "Alice")

    with pytest.raises(_Phase6NddStageError):
        backend.detect(_Phase6CandidateWork(target))


def test_phase6_ndd_augmentation_uses_valid_jinja_for_target_and_context() -> None:
    adapter = _ScriptedAdapter({"entities": []}, [])
    backend = _backend(adapter)
    target = _MentionTarget(_MentionTargetToken(), _DatumId("target"), "Alice")

    assert backend.augment(_Phase6AugmentationWork(target, ())) == ()

    prompt = adapter.prompts[-1]
    assert "{{ {{" not in prompt
    assert _jinja(COL_TEXT) in prompt
    assert _jinja(COL_PHASE6_CONTEXT) in prompt


@pytest.mark.parametrize(
    "field_value",
    [
        pytest.param("0", id="string"),
        pytest.param(0.0, id="float"),
        pytest.param(False, id="boolean"),
    ],
)
@pytest.mark.parametrize("field_name", ["start", "end"])
def test_phase6_ndd_augmentation_rejects_non_integer_offsets(field_name: str, field_value: object) -> None:
    entity: dict[str, object] = {
        "start": 0,
        "end": 5,
        "source_slice": "Alice",
        "detector_label": "name",
    }
    entity[field_name] = field_value
    payload = {"entities": [entity]}
    backend = _backend(_ScriptedAdapter({"entities": []}, [], augmentation_payload=payload))
    target = _MentionTarget(_MentionTargetToken(), _DatumId("target"), "Alice")

    with pytest.raises(_Phase6NddStageError):
        backend.augment(_Phase6AugmentationWork(target, ()))


@pytest.mark.parametrize("field_name", ["source_slice", "detector_label"])
def test_phase6_ndd_augmentation_rejects_coerced_string_fields(field_name: str) -> None:
    entity: dict[str, object] = {
        "start": 0,
        "end": 5,
        "source_slice": "Alice",
        "detector_label": "name",
    }
    entity[field_name] = b"coerced"
    payload = {"entities": [entity]}
    backend = _backend(_ScriptedAdapter({"entities": []}, [], augmentation_payload=payload))
    target = _MentionTarget(_MentionTargetToken(), _DatumId("target"), "Alice")

    with pytest.raises(_Phase6NddStageError):
        backend.augment(_Phase6AugmentationWork(target, ()))


def test_phase6_ndd_validator_rejects_a_missing_candidate_decision() -> None:
    backend = _backend(_ScriptedAdapter({"entities": []}, []))
    target = _MentionTarget(_MentionTargetToken(), _DatumId("target"), "Alice Bob")
    candidates = tuple(
        _ProvisionalCandidate(
            _CandidateToken(),
            target.token,
            start,
            end,
            target.text[start:end],
            "name",
            _MentionProvenance.SPAN_DETECTOR,
        )
        for start, end in ((0, 5), (6, 9))
    )

    with pytest.raises(_Phase6NddStageError):
        backend.validate(_Phase6ValidationWork(target, candidates))


@pytest.mark.parametrize(
    "ordinal",
    [
        pytest.param("0", id="string"),
        pytest.param(0.0, id="float"),
        pytest.param(False, id="boolean"),
    ],
)
def test_phase6_ndd_validator_rejects_non_integer_ordinals(ordinal: object) -> None:
    payload = {"decisions": [{"ordinal": ordinal, "decision": "drop", "proposed_label": None}]}
    backend = _backend(_ScriptedAdapter({"entities": []}, [], validation_payload=payload))
    target = _MentionTarget(_MentionTargetToken(), _DatumId("target"), "Alice")
    candidate = _ProvisionalCandidate(
        _CandidateToken(),
        target.token,
        0,
        5,
        "Alice",
        "name",
        _MentionProvenance.SPAN_DETECTOR,
    )

    with pytest.raises(_Phase6NddStageError):
        backend.validate(_Phase6ValidationWork(target, (candidate,)))


def test_phase6_ndd_validator_rejects_a_coerced_proposed_label() -> None:
    payload = {"decisions": [{"ordinal": 0, "decision": "reclass", "proposed_label": b"name"}]}
    backend = _backend(_ScriptedAdapter({"entities": []}, [], validation_payload=payload))
    target = _MentionTarget(_MentionTargetToken(), _DatumId("target"), "Alice")
    candidate = _ProvisionalCandidate(
        _CandidateToken(),
        target.token,
        0,
        5,
        "Alice",
        "name",
        _MentionProvenance.SPAN_DETECTOR,
    )

    with pytest.raises(_Phase6NddStageError):
        backend.validate(_Phase6ValidationWork(target, (candidate,)))
