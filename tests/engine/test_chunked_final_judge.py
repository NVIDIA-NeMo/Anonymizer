# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the windowed (parallel) final judge.

Pure helpers (slicing, window planning, score aggregation) are tested directly; the
per-window dispatch is tested via a fake facade that replays canned judge scores
through the recipe parser, including a failing window (which must be skipped).
"""

from __future__ import annotations

import json
from typing import Any, Callable

from anonymizer.engine.constants import COL_JUDGE_EVALUATION, COL_REWRITTEN_TEXT, COL_TEXT
from anonymizer.engine.rewrite.chunked_final_judge import (
    JudgeScoresSchema,
    WindowedJudgeParams,
    aggregate_judge,
    judge_row,
    plan_judge_windows,
    slice_evenly,
)

_TEMPLATE = "ORIG:{{ original_text }} REW:{{ rewritten_text }}"


def _scores(privacy: int, quality: int, naturalness: int, note: str = "r") -> JudgeScoresSchema:
    return JudgeScoresSchema.model_validate(
        {
            "privacy": {"score": privacy, "reasoning": note},
            "quality": {"score": quality, "reasoning": note},
            "naturalness": {"score": naturalness, "reasoning": note},
        }
    )


class FakeJudgeFacade:
    """Replays canned JudgeScoresSchema responses through the recipe parser."""

    def __init__(self, response: dict | Callable[[str], dict]) -> None:
        self._response = response
        self.calls: list[str] = []

    def generate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        self.calls.append(purpose)
        resp = self._response(prompt) if callable(self._response) else self._response
        return parser(f"```json\n{json.dumps(resp)}\n```"), []


def _params(max_render_chars: int, **kw: Any) -> WindowedJudgeParams:
    return WindowedJudgeParams(alias="judge", prompt_template=_TEMPLATE, max_render_chars=max_render_chars, **kw)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_slice_evenly_single_window():
    assert slice_evenly("abcdef", 1) == ["abcdef"]


def test_slice_evenly_splits_and_pads():
    assert slice_evenly("abcdef", 3) == ["ab", "cd", "ef"]
    assert slice_evenly("ab", 3) == ["a", "b", ""]  # padded to exactly n


def test_plan_judge_windows_single_when_small():
    prompts = plan_judge_windows(original="o", rewritten="r", template=_TEMPLATE, cap=10_000, safety_margin_chars=0)
    assert len(prompts) == 1
    assert "ORIG:o REW:r" in prompts[0]


def test_plan_judge_windows_splits_when_large():
    original, rewritten = "o" * 9000, "r" * 9000
    # tiny budget -> multiple windows; both texts sliced into the same count
    prompts = plan_judge_windows(original=original, rewritten=rewritten, template=_TEMPLATE, cap=4200, safety_margin_chars=0)
    assert len(prompts) > 1


def test_aggregate_judge_takes_min_per_dimension():
    agg = aggregate_judge([_scores(8, 9, 7, "good"), _scores(3, 6, 9, "leak")])
    assert agg["privacy"]["score"] == 3
    assert agg["quality"]["score"] == 6
    assert agg["naturalness"]["score"] == 7
    assert "min over 2 window(s)" in agg["privacy"]["reasoning"]
    assert "leak" in agg["privacy"]["reasoning"]  # reasoning from the worst window


# ---------------------------------------------------------------------------
# judge_row
# ---------------------------------------------------------------------------


def test_judge_row_fast_path_single_call():
    facade = FakeJudgeFacade(_scores(8, 8, 8).model_dump())
    row = {COL_TEXT: "original", COL_REWRITTEN_TEXT: "rewritten"}
    judge_row(row, _params(max_render_chars=10_000), {"judge": facade})
    assert len(facade.calls) == 1
    ev = row[COL_JUDGE_EVALUATION]
    assert set(ev) == {"privacy", "quality", "naturalness"}
    assert ev["privacy"]["score"] == 8


def test_judge_row_windowed_aggregates_min():
    # distinct score per window so we can verify min aggregation
    import itertools

    counter = itertools.count()

    def resp(_prompt):
        i = next(counter)
        return _scores(5 + i, 9, 9).model_dump()

    facade = FakeJudgeFacade(resp)
    row = {COL_TEXT: "o" * 9000, COL_REWRITTEN_TEXT: "r" * 9000}
    judge_row(row, _params(max_render_chars=4200, safety_margin_chars=0), {"judge": facade})
    assert len(facade.calls) > 1
    # min privacy across windows is the first window's (5 + 0)
    assert row[COL_JUDGE_EVALUATION]["privacy"]["score"] == 5


def test_judge_row_skips_failing_window_not_fatal():
    import itertools

    counter = itertools.count()

    def resp(_prompt):
        i = next(counter)
        if i == 0:
            raise ValueError("simulated judge parse failure")
        return _scores(7, 7, 7).model_dump()

    facade = FakeJudgeFacade(resp)
    row = {COL_TEXT: "o" * 9000, COL_REWRITTEN_TEXT: "r" * 9000}
    judge_row(row, _params(max_render_chars=4200, safety_margin_chars=0), {"judge": facade})
    # one window failed but the rest produced a valid evaluation
    assert row[COL_JUDGE_EVALUATION] is not None
    assert row[COL_JUDGE_EVALUATION]["privacy"]["score"] == 7


def test_judge_row_all_windows_fail_yields_none():
    def resp(_prompt):
        raise RuntimeError("judge down")

    facade = FakeJudgeFacade(resp)
    row = {COL_TEXT: "original", COL_REWRITTEN_TEXT: "rewritten"}
    judge_row(row, _params(max_render_chars=10_000), {"judge": facade})
    assert row[COL_JUDGE_EVALUATION] is None


def test_judge_row_no_rewrite_yields_none():
    facade = FakeJudgeFacade(_scores(8, 8, 8).model_dump())
    row = {COL_TEXT: "original", COL_REWRITTEN_TEXT: None}
    judge_row(row, _params(max_render_chars=10_000), {"judge": facade})
    assert row[COL_JUDGE_EVALUATION] is None
    assert facade.calls == []  # no judge call when there's nothing to judge
