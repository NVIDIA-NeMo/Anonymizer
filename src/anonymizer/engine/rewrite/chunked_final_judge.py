# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Windowed final judge for long-context rewrites.

The final judge scores a rewrite on privacy / quality / naturalness. Normally it
runs as a single DataDesigner ``LLMJudgeColumnConfig`` whose prompt embeds the full
original + rewritten text, which blows past NDD's ginja render cap (512000 chars) on
long documents.

This module instead tiles the original and rewritten text into matching, *independent*
windows, judges each window in parallel (like detection — no cross-window state), and
aggregates the per-window scores by taking the **minimum** per dimension (the worst
section drives the score; conservative for privacy). Windows are paired positionally:
both texts are split into the same number of near-equal slices.

Output column (``COL_JUDGE_EVALUATION``) keeps the judge's dict shape so the existing
display/extract code is unchanged: ``{rubric_name: {"score": int, "reasoning": str}}``.

Public entry point: :func:`make_windowed_judge_generator`.
"""

from __future__ import annotations

import functools
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from jinja2 import BaseLoader, Environment, StrictUndefined
from pydantic import BaseModel, Field

from anonymizer.engine.constants import COL_JUDGE_EVALUATION, COL_REWRITTEN_TEXT, COL_TEXT

logger = logging.getLogger("anonymizer.rewrite.chunked_judge")

# Floor on the per-window text budget so a tiny cap still makes progress.
_MIN_BUDGET_CHARS = 4000

# Upper bound on judge windows dispatched concurrently for one record. The per-alias
# rate limit on the facade still caps real in-flight calls; this bounds thread creation.
_MAX_PARALLEL_WINDOWS = 8

_DIMENSIONS = ("privacy", "quality", "naturalness")

_PROMPT_ENV = Environment(loader=BaseLoader(), autoescape=False, undefined=StrictUndefined, keep_trailing_newline=True)


@functools.lru_cache(maxsize=4)
def _compile_template(template: str) -> Any:
    return _PROMPT_ENV.from_string(template)


# ---------------------------------------------------------------------------
# Structured output schema (one window's scores)
# ---------------------------------------------------------------------------


class _DimensionScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reasoning: str = ""


class JudgeScoresSchema(BaseModel):
    """Per-window judge scores; mirrors the three final-judge rubrics."""

    privacy: _DimensionScore
    quality: _DimensionScore
    naturalness: _DimensionScore


class WindowedJudgeParams(BaseModel):
    """Params for the windowed final judge (via DD ``generator_params``)."""

    alias: str = Field(min_length=1)
    prompt_template: str = Field(repr=False)
    max_render_chars: int = Field(gt=0)
    safety_margin_chars: int = Field(default=8000, ge=0)
    system_prompt: str | None = Field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def slice_evenly(text: str, n: int) -> list[str]:
    """Split ``text`` into ``n`` contiguous, near-equal character slices."""
    if n <= 1:
        return [text]
    size = max(1, math.ceil(len(text) / n))
    slices = [text[i * size : (i + 1) * size] for i in range(n)]
    # Guarantee exactly n entries (pad with "" if text shorter than n chars).
    while len(slices) < n:
        slices.append("")
    return slices[:n]


def render_judge_prompt(*, template: str, original: str, rewritten: str) -> str:
    """Render the judge prompt for one window."""
    return _compile_template(template).render(original_text=original, rewritten_text=rewritten)


def plan_judge_windows(
    *, original: str, rewritten: str, template: str, cap: int, safety_margin_chars: int
) -> list[str]:
    """Return one rendered prompt per window (positionally paired slices).

    The window count is chosen so each window's combined text fits the render
    budget; both texts are split into that many near-equal slices. No LLM calls.
    """
    overhead = len(render_judge_prompt(template=template, original="", rewritten=""))
    budget = max(_MIN_BUDGET_CHARS, cap - safety_margin_chars - overhead)
    total = len(original) + len(rewritten)
    n = max(1, math.ceil(total / budget))
    o_slices = slice_evenly(original, n)
    r_slices = slice_evenly(rewritten, n)
    return [render_judge_prompt(template=template, original=o, rewritten=r) for o, r in zip(o_slices, r_slices)]


def aggregate_judge(results: list[JudgeScoresSchema]) -> dict[str, dict[str, Any]]:
    """Aggregate per-window scores into the judge-column dict shape.

    Per dimension, takes the minimum score across windows (worst section) and the
    reasoning from that worst window. Shape matches LLMJudgeColumnConfig output:
    ``{dim: {"score": int, "reasoning": str}}``.
    """
    out: dict[str, dict[str, Any]] = {}
    for dim in _DIMENSIONS:
        dims = [getattr(r, dim) for r in results]
        worst = min(dims, key=lambda d: d.score)
        note = worst.reasoning
        if len(results) > 1:
            note = f"{note} [min over {len(results)} window(s)]".strip()
        out[dim] = {"score": worst.score, "reasoning": note}
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _judge_window(*, facade: Any, prompt: str, system_prompt: str | None, idx: int) -> JudgeScoresSchema | None:
    """Judge one window; return its scores, or ``None`` if the call fails (logged, skipped)."""
    recipe = PydanticResponseRecipe(data_type=JudgeScoresSchema)
    try:
        output, _messages = facade.generate(
            prompt=recipe.apply_recipe_to_user_prompt(prompt),
            parser=recipe.parse,
            system_prompt=recipe.apply_recipe_to_system_prompt(system_prompt),
            purpose=f"final-judge-window-{idx}",
        )
    except Exception as exc:  # noqa: BLE001 — one window must not sink the (non-critical) judge
        logger.warning("final-judge window %d failed (%s: %s); skipping it", idx, type(exc).__name__, exc)
        return None
    return output


def judge_row(row: dict[str, Any], params: WindowedJudgeParams, models: dict[str, Any]) -> dict[str, Any]:
    """Run the (possibly windowed) final judge for one row, writing ``COL_JUDGE_EVALUATION``."""
    if params.alias not in models:
        raise KeyError(
            f"Judge alias {params.alias!r} not present in models dict. Ensure "
            "make_windowed_judge_generator was invoked with the same alias."
        )
    facade = models[params.alias]

    original = str(row.get(COL_TEXT, ""))
    rewritten = row.get(COL_REWRITTEN_TEXT)
    if not rewritten:
        # No rewrite to judge (dropped/empty). Leave the evaluation unset.
        row[COL_JUDGE_EVALUATION] = None
        return row
    rewritten = str(rewritten)

    prompts = plan_judge_windows(
        original=original,
        rewritten=rewritten,
        template=params.prompt_template,
        cap=params.max_render_chars,
        safety_margin_chars=params.safety_margin_chars,
    )
    max_workers = min(len(prompts), _MAX_PARALLEL_WINDOWS)
    if len(prompts) > 1:
        logger.info("final-judge: judging %d window(s) in parallel (max_workers=%d)", len(prompts), max_workers)

    results: list[JudgeScoresSchema] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_judge_window, facade=facade, prompt=prompt, system_prompt=params.system_prompt, idx=idx)
            for idx, prompt in enumerate(prompts)
        ]
        for future in futures:
            res = future.result()  # _judge_window swallows errors -> never raises
            if res is not None:
                results.append(res)

    if not results:
        logger.warning("final-judge: all %d window(s) failed; leaving evaluation unset", len(prompts))
        row[COL_JUDGE_EVALUATION] = None
        return row

    row[COL_JUDGE_EVALUATION] = aggregate_judge(results)
    return row


# ---------------------------------------------------------------------------
# DataDesigner wiring factory.
# ---------------------------------------------------------------------------


def make_windowed_judge_generator(alias: str) -> Any:
    """Build a ``@custom_column_generator``-decorated final judge bound to ``alias``."""
    if not alias:
        raise ValueError("Cannot build windowed judge generator: alias is empty.")

    @custom_column_generator(
        required_columns=[COL_TEXT, COL_REWRITTEN_TEXT],
        model_aliases=[alias],
    )
    def windowed_judge(
        row: dict[str, Any],
        generator_params: WindowedJudgeParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return judge_row(row, generator_params, models)

    return windowed_judge
