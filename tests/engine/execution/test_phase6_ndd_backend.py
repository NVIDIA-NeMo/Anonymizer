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
            output[COL_PHASE6_AUGMENTED] = [{"entities": []}]
        elif workflow_name == "phase6-validate":
            output[COL_PHASE6_VALIDATION] = [
                {"decisions": [{"ordinal": 0, "decision": "keep", "proposed_label": None}]}
            ]
        else:  # pragma: no cover - challenge guard
            raise AssertionError(workflow_name)
        return WorkflowRunResult(output, [])


def _backend(adapter: _ScriptedAdapter) -> _Phase6NddBackend:
    anonymizer = build_synthetic_anonymizer({})
    plan = anonymizer._compile_protection_plan(AnonymizerConfig(replace=Redact(), emit_telemetry=False))
    return _Phase6NddBackend(cast(NddAdapter, adapter), plan.invocation)


def test_phase6_ndd_detector_uses_only_target_text_and_preserves_exact_spans() -> None:
    payload = json.dumps(
        {"entities": [{"text": "Alice", "label": "name", "start": 0, "end": 5, "score": 0.9}]}
    )
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
