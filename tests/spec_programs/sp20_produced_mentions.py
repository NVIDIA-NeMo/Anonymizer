# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SP-20 mentions an operation produces are checked when accepted: typed target failures, limits (E20.1-E20.4)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection.detect_step import DetectState, detect_step
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.detection.postprocess import EntitySpan, apply_augmented_entities
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
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.replace.route import run_replace_pipeline
from tests.spec_programs._builder import Checks

TEXTS = ("Alice met Bob.", "Carol left.")


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


def _redact(texts: list[str], entities: list[list[dict]]):
    """The Replace route with Redact over ``texts``, the detector returning ``entities`` per row."""

    def detect(df: pd.DataFrame) -> EntityDetectionResult:
        found = df.assign(**{COL_FINAL_ENTITIES: [{"entities": e} for e in entities]})
        return EntityDetectionResult(found, [], tuple(range(len(df))))

    frame = pd.DataFrame({COL_TEXT: texts})
    return run_replace_pipeline(
        frame,
        detect=detect,
        replace_method=Redact(),
        llm_workflow=None,
        map_arguments={},
        reroute=lambda df: ReplacementResult(df, []),
    )


def program() -> list[dict[str, str]]:
    c = Checks("SP-20")

    # E20.1 a claimed text that is not the source slice fails that target only; its consumer never runs.
    r, ts, consumed = _run(lambda d, i, ts: (Mention(d, 10, 14, "name", "Bob"),) if i == 0 else ())
    out = _outcomes(r, ts)
    c.check(
        "E20.1",
        out == [("failed", "mention_slice_mismatch"), ("success", "")] and consumed == [0],
        f"outcomes={out}, consumer calls={consumed}",
    )

    # E20.2 out of range, overlapping, on another target and non-mention items each fail with their own reason.
    variants = {
        "mention_out_of_range": lambda d, ts: (Mention(d, 10, 99, "name"),),
        "overlapping_output_ownership": lambda d, ts: (Mention(d, 0, 5, "a"), Mention(d, 3, 8, "b")),
        "mention_not_on_target": lambda d, ts: (Mention(ts[1], 0, 5, "name"),),
        "invalid_output": lambda d, ts: ("Alice",),
    }
    seen = {}
    for reason, make in variants.items():
        r, ts, _ = _run(lambda d, i, ts, make=make: make(d, ts) if i == 0 else (), per_invocation=True)
        seen[reason] = _outcomes(r, ts)
    c.check(
        "E20.2",
        all(out == [("failed", reason), ("success", "")] for reason, out in seen.items()),
        f"outcomes={seen}",
    )

    # E20.3 three produced mentions over max_mentions=2: every target of that publication fails, in either order.
    def three(d: Any, i: int, ts: tuple) -> tuple:
        return (Mention(d, 0, 5, "a"), Mention(d, 10, 13, "b")) if d.text == TEXTS[0] else (Mention(d, 0, 5, "c"),)

    forward, fts, _ = _run(three, per_invocation=True, limits=Limits(max_mentions=2))
    backward, bts, _ = _run(three, per_invocation=True, texts=TEXTS[::-1], limits=Limits(max_mentions=2))
    f_out, b_out = _outcomes(forward, fts), _outcomes(backward, bts)[::-1]
    c.check(
        "E20.3",
        f_out == b_out == [("failed", "over_limit")] * 2,
        f"forward={f_out}, reversed={b_out}",
    )

    # E20.4 a detector returning nested spans on one row fails that target; it is not resolved by picking one.
    text = "Alice Smith met Bob at noon."
    frame = pd.DataFrame({COL_TEXT: [text, "plain"]})
    spans = [{"value": text[0:4], "label": "a", "start_position": 0, "end_position": 4}]
    spans.append({"value": text[0:22], "label": "b", "start_position": 0, "end_position": 22})

    def detect(df: pd.DataFrame) -> EntityDetectionResult:
        return EntityDetectionResult(
            df.assign(**{COL_FINAL_ENTITIES: [{"entities": spans}, {"entities": []}]}), [], (0, 1)
        )

    scope = GraphScope()
    targets = tuple(scope.datum(t) for t in frame[COL_TEXT])
    step = detect_step(frame, detect, DetectState())
    r4 = protect(prepare(ProtectionGraph(scope, targets), Workflow("w", (step,))))
    out4 = _outcomes(r4, targets)
    c.check("E20.4", out4 == [("failed", "overlapping_output_ownership"), ("success", "")], f"outcomes={out4}")

    # E20.5 built-in name splitting and augmentation match case-insensitively; every row is still redacted.
    john = "John Smith wrote from john@x.org today."
    alice = "Alice met alice."
    full = EntitySpan("full_name_0_10", "John Smith", "full_name", 0, 10, 0.9, "detector")
    built = [
        apply_augmented_entities(john, [full], {"entities": []}),
        apply_augmented_entities(alice, [], {"entities": [{"value": "Alice", "label": "first_name"}]}),
    ]
    r5 = _redact([john, alice], [[e.as_dict() for e in row] for row in built])
    shown5 = list(r5.dataframe[COL_REPLACED_TEXT])
    c.check(
        "E20.5",
        len(shown5) == 2 and "john@" not in shown5[0] and "alice" not in shown5[1].lower() and not r5.failed_records,
        f"rows={shown5}, failed={r5.failed_records}",
    )

    # E20.6 a detected span that fails acceptance makes its row a failed record naming the reason; others proceed.
    bad = [{"value": "Bob", "label": "first_name", "start_position": 10, "end_position": 14}]
    r6 = _redact(["Alice met Bob.", "Carol left."], [bad, []])
    reasons = [(f.step, f.reason) for f in r6.failed_records]
    c.check(
        "E20.6",
        list(r6.dataframe[COL_TEXT]) == ["Carol left."] and reasons == [("detect", "mention_slice_mismatch")],
        f"rows={list(r6.dataframe[COL_TEXT])}, failed={reasons}",
    )
    return c.lines
