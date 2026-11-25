# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mentions an operation produces are checked when accepted: bad mentions fail their target with a typed reason,
limits fail the whole publication, and the Replace route turns rejected spans into failed records."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import (
    COL_AUGMENTATION_DETECTION_RESULT,
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_SEED_ENTITIES_JSON,
    COL_TEXT,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.augmentation import augment_row
from anonymizer.engine.detection.detect_step import DetectState, detect_step
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.graph_core.authoring import (
    DATUM,
    MENTIONS,
    GraphScope,
    Limits,
    Mention,
    Operation,
    ProtectionGraph,
    Workflow,
)
from anonymizer.engine.graph_core.execution import ProtectionResult, prepare, protect
from anonymizer.engine.replace.route import run_replace_pipeline
from anonymizer.engine.schemas import AugmentedEntitiesSchema, DetectionResult

TEXTS = ("Alice met Bob.", "Carol left.")


def _ids(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with the record id column the ``Anonymizer`` seam attaches before the detect step."""
    return frame.assign(**{RECORD_ID_COLUMN: [f"r{i}" for i in range(len(frame))]})


def _run(produce: Any, *, per_invocation: bool = False, texts: tuple[str, ...] = TEXTS, **prepare_kwargs: Any):
    """``produce(target, position, targets)`` returns the mentions published for a target; a consumer reads them."""
    scope = GraphScope()
    targets = tuple(scope.datum(t) for t in texts)
    position = {d: i for i, d in enumerate(targets)}
    consumed: list[int] = []
    if per_invocation:
        producer = Operation.per_invocation(
            "produce",
            lambda target: [produce(d, position[d], targets) for d in target],
            {"target": DATUM},
            {"mentions": MENTIONS},
        )
    else:
        producer = Operation(
            "produce",
            lambda target: produce(target, position[target], targets),
            {"target": DATUM},
            {"mentions": MENTIONS},
        )
    consumer = Operation(
        "consume", lambda text, mentions: consumed.append(len(mentions)) or text, {"text": "text", "mentions": MENTIONS}
    )
    g = ProtectionGraph(scope, targets)
    r = protect(prepare(g, Workflow("w", (producer, consumer)), **prepare_kwargs))
    return r, targets, consumed


def _outcomes(r: ProtectionResult, targets: tuple) -> list[tuple[str, str]]:
    return [(o.terminal, o.reason) for o in map(r.outcome, targets)]


def _redact(texts: list[str], entities: list[list[dict]]) -> Any:
    """The Replace route with Redact over ``texts``, the detector returning ``entities`` per row."""

    def detect(df: pd.DataFrame) -> EntityDetectionResult:
        found = df.assign(**{COL_FINAL_ENTITIES: [{"entities": e} for e in entities]})
        return EntityDetectionResult(found, [], tuple(range(len(df))))

    frame = _ids(pd.DataFrame({COL_TEXT: texts}))
    return run_replace_pipeline(frame, detect=detect, replace_method=Redact(), llm_workflow=None, map_arguments={})


def test_slice_mismatch_fails_that_target_only() -> None:
    r, ts, consumed = _run(lambda d, i, ts: (Mention(d, 10, 14, "name", "Bob"),) if i == 0 else ())
    assert _outcomes(r, ts) == [("failed", "mention_slice_mismatch"), ("success", "")]
    assert consumed == [0]


@pytest.mark.parametrize(
    ("reason", "make"),
    [
        ("mention_out_of_range", lambda d, ts: (Mention(d, 10, 99, "name"),)),
        ("overlapping_output_ownership", lambda d, ts: (Mention(d, 0, 5, "a"), Mention(d, 3, 8, "b"))),
        ("mention_not_on_target", lambda d, ts: (Mention(ts[1], 0, 5, "name"),)),
        ("invalid_output", lambda d, ts: ("Alice",)),
    ],
)
def test_bad_mention_fails_with_its_own_reason(reason: str, make: Any) -> None:
    r, ts, _ = _run(lambda d, i, ts: make(d, ts) if i == 0 else (), per_invocation=True)
    assert _outcomes(r, ts) == [("failed", reason), ("success", "")]


def test_over_mention_limit_fails_every_target_in_either_order() -> None:
    def three(d: Any, i: int, ts: tuple) -> tuple:
        return (Mention(d, 0, 5, "a"), Mention(d, 10, 13, "b")) if d.text == TEXTS[0] else (Mention(d, 0, 5, "c"),)

    forward, fts, _ = _run(three, per_invocation=True, limits=Limits(max_mentions=2))
    backward, bts, _ = _run(three, per_invocation=True, texts=TEXTS[::-1], limits=Limits(max_mentions=2))
    assert _outcomes(forward, fts) == [("failed", "over_limit")] * 2
    assert _outcomes(backward, bts)[::-1] == [("failed", "over_limit")] * 2


def test_detector_nested_spans_resolve_and_partial_overlap_fails() -> None:
    text = "Alice Smith met Bob at noon."
    frame = _ids(pd.DataFrame({COL_TEXT: [text, text, "plain"]}))
    nested = [
        {"value": text[0:4], "label": "a", "start_position": 0, "end_position": 4},
        {"value": text[0:22], "label": "b", "start_position": 0, "end_position": 22},
    ]
    partial = [nested[1], {"value": text[16:27], "label": "c", "start_position": 16, "end_position": 27}]

    def detect(df: pd.DataFrame) -> EntityDetectionResult:
        rows = [{"entities": nested}, {"entities": partial}, {"entities": []}]
        return EntityDetectionResult(df.assign(**{COL_FINAL_ENTITIES: rows}), [], (0, 1, 2))

    scope = GraphScope()
    targets = tuple(scope.datum(t) for t in frame[COL_TEXT])
    step = detect_step(frame, detect, DetectState())
    r = protect(prepare(ProtectionGraph(scope, targets), Workflow("w", (step,))))
    assert _outcomes(r, targets) == [("success", ""), ("failed", "overlapping_output_ownership"), ("success", "")]
    assert [(m.start, m.end, m.label) for m in r.value(targets[0], "mentions") or ()] == [(0, 22, "b")]


def test_augmenter_value_not_in_source_is_rejected() -> None:
    text = "Alice met Bob."
    reply = AugmentedEntitiesSchema.model_validate(
        {
            "entities": [
                {"value": "Alice", "label": "first_name", "context_before": "", "context_after": " met"},
                {"value": "BOB", "label": "first_name", "context_before": "met ", "context_after": "."},
            ]
        }
    )

    class _Augmenter:
        replies = [reply, AugmentedEntitiesSchema(entities=[])]  # the repair request gets no correction

        def generate(self, **_kwargs: Any) -> tuple[AugmentedEntitiesSchema, list[Any]]:
            return self.replies.pop(0), []

    row = augment_row(
        row={RECORD_ID_COLUMN: "r", COL_TEXT: text, COL_SEED_ENTITIES_JSON: "[]"},
        prompt_template="{{ " + COL_TEXT + " }}",
        model_alias="augmenter",
        facade=_Augmenter(),
        allowed_entity_labels=["first_name"],
        excluded_entity_labels=[],
    )
    found = DetectionResult.model_validate(row[COL_AUGMENTATION_DETECTION_RESULT]).observations
    spans = [(o.value, o.start_position, o.end_position) for o in found]
    r = _redact(
        [text], [[{"value": v, "label": "first_name", "start_position": a, "end_position": b} for v, a, b in spans]]
    )
    assert spans == [("Alice", 0, 5)]
    assert all(text[a:b] == v for v, a, b in spans)
    assert "Alice" not in list(r.dataframe[COL_REPLACED_TEXT])[0]
    assert not r.failed_records


def test_rejected_detected_span_makes_a_failed_record() -> None:
    bad = [{"value": "Bob", "label": "first_name", "start_position": 10, "end_position": 14}]
    r = _redact(["Alice met Bob.", "Carol left."], [bad, []])
    assert list(r.dataframe[COL_TEXT]) == ["Carol left."]
    assert [(f.step, f.reason) for f in r.failed_records] == [("detect", "mention_slice_mismatch")]
