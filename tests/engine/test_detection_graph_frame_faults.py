# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent callback/frame acceptance controls; literal owner/publication expectations."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace

import pandas as pd
import pytest

from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection._frame_publication import FramePublication, FrameStages
from anonymizer.engine.graph_sdk.executor import LocalCompleted
from anonymizer.engine.graph_sdk.requests import AssociationResult, PortArtifact, TextArtifactValue
from tests.graph_sdk.test_delegated_execution import _execute
from tests.graph_sdk.test_delegated_readiness import _ready_fixture

pytestmark = pytest.mark.compat


async def run_case(fault="valid", limited=False):
    e, s, original, *_ = _ready_fixture(usage="exact")
    saved = {}

    class Submit:
        execution = "same_loop"

        async def submit(self, inputs, requests):
            await requests.readiness()
            await requests.probe()
            await requests.request(original.slots[0])
            row = inputs[0]
            value = row.inputs[0]
            candidate = "candidate:" + value.value.text
            out = AssociationResult(
                association=row.association,
                outcome="ok",
                consumed_context_ports=frozenset(),
                outputs=(
                    PortArtifact(
                        port="context",
                        artifact_type=value.artifact_type,
                        artifact=None,
                        value=TextArtifactValue(text=candidate),
                    ),
                ),
            )
            source = pd.DataFrame({COL_TEXT: [value.value.text], "opaque": [{"original": []}]})
            requests.bind_frame(source)
            saved["original"] = source["opaque"].iloc[0]
            content = source.assign(**{COL_REPLACED_TEXT: [candidate], "trace": [{"accepted": [1, 2]}]})
            if fault == "rebind":
                requests.bind_frame(content)
            frame = FramePublication.capture(
                content,
                associations=(row.association,),
                text_ports=((COL_REPLACED_TEXT, "context"),),
                failed_records=(),
                opaque_source=content if fault == "rebind" else source,
                stages=FrameStages(content, (), (), 0.0, 0.0),
            )
            if fault == "stage_missing":
                object.__delattr__(frame.stages, "detected_dataframe")
            elif fault == "stage_nan":
                object.__setattr__(frame.stages, "detection_elapsed", float("nan"))
            elif fault == "empty_text":
                frame = FramePublication(content, content.columns, (row.association,), (), (), ())
            elif fault == "nan_text":
                frame = replace(frame, _text=((COL_REPLACED_TEXT, "context", (float("nan"),)),))
            elif fault == "content_type":
                frame = replace(frame, _content="bad")
            elif fault.startswith("missing:"):
                missing = fault.split(":", 1)[1]
                raw = object.__new__(FramePublication)
                for field in fields(frame):
                    if field.name != missing:
                        object.__setattr__(raw, field.name, getattr(frame, field.name))
                frame = raw
            saved["callback_frame"] = frame
            saved["content"] = content
            return LocalCompleted(results=(out,), frame=frame)

    s = replace(
        s,
        handles=tuple(replace(h, delegated=Submit()) if h.delegated else h for h in s.handles),
        limits=replace(s.limits, max_runtime_artifacts=1 if limited else 8, max_runtime_artifact_bytes=1000),
    )
    result = await _execute(e, s)
    return e, result, saved


@pytest.mark.parametrize("limited", [False, True])
def test_valid_frame_requires_successful_semantic_publication(limited):
    e, r, s = asyncio.run(run_case(limited=limited))
    assert len(r.frames) == (0 if limited else 1)
    if not limited:
        projected = r.frames[0].project(r)
        assert projected[COL_REPLACED_TEXT].tolist() == ["candidate:target-0"]
        assert COL_REPLACED_TEXT not in r.frames[0]._content
        projected["trace"].iloc[0]["accepted"].clear()
        s["content"]["trace"].iloc[0]["accepted"].clear()
        s["callback_frame"].stages.detected_dataframe["trace"].iloc[0]["accepted"].clear()
        assert r.frames[0].stages.detected_dataframe["trace"].iloc[0]["accepted"] == [1, 2]
        assert r.frames[0].stages.detected_dataframe["opaque"].iloc[0] is s["original"]
        assert r.frames[0].project(r)["trace"].iloc[0]["accepted"] == [1, 2]


@pytest.mark.parametrize("limited", [False, True])
def test_empty_text_frame_is_rejected_even_with_valid_semantic_callback(limited):
    _, r, _ = asyncio.run(run_case("empty_text", limited=limited))
    assert not r.frames, "empty text mapping bypassed required artifact/port publication"


@pytest.mark.parametrize(
    "fault",
    [
        "content_type",
        "nan_text",
        "missing:_content",
        "missing:_columns",
        "missing:failed_records",
        "missing:_opaque",
        "stage_missing",
        "stage_nan",
    ],
)
def test_raw_frame_shape_rejects_before_finalization(fault):
    _, r, _ = asyncio.run(run_case(fault))
    assert not r.frames, "malformed callback frame was retained"


def test_callback_frame_cannot_mutate_accepted_execution_content():
    _, r, s = asyncio.run(run_case())
    s["callback_frame"]._content["trace"].iloc[0]["accepted"].clear()
    assert r.frames[0].project(r)["trace"].iloc[0]["accepted"] == [1, 2], (
        "accepted trace still aliases callback-owned frame"
    )


@pytest.mark.parametrize("bad", ["empty_text"])  # keeps the retained case id
def test_retained_frame_shape_rejects_with_value_error(bad):
    from tests.graph_sdk.child_qualification_fixtures import result_with

    _, result, _ = asyncio.run(run_case())
    frame = replace(result.frames[0], **{"_text" if bad == "empty_text" else bad: ()})
    with pytest.raises(ValueError):
        frame.project(result_with(result, frames=(frame,)))


@pytest.mark.parametrize("rebind", [False, True])
def test_source_rebinding_cannot_exempt_generated_content_from_snapshot(rebind):
    _, result, saved = asyncio.run(run_case("rebind" if rebind else "valid"))
    assert result.frames[0].project(result)["opaque"].iloc[0] is saved["original"]
    saved["content"]["trace"].iloc[0]["accepted"].clear()
    saved["callback_frame"]._content["trace"].iloc[0]["accepted"].clear()
    assert result.frames[0].project(result)["trace"].iloc[0]["accepted"] == [1, 2]


@pytest.mark.parametrize("replacement", ["root", "assessment"])
def test_retained_frame_reference_must_belong_to_bound_output_port(replacement):
    from tests.graph_sdk.child_qualification_fixtures import result_with

    _, result, _ = asyncio.run(run_case())
    frame = result.frames[0]
    actual = frame._text[0][1][0]
    candidates = [ref for ref, value in result.artifacts if ref != actual and type(value) is TextArtifactValue]
    reference = candidates[0] if replacement == "root" else candidates[-1]
    frame = replace(frame, _text=((COL_REPLACED_TEXT, (reference,)),))
    with pytest.raises(ValueError):
        frame.project(result_with(result, frames=(frame,)))
