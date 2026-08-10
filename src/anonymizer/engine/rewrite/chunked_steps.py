# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic boundary-windowed LLM step for rewrite-pipeline metadata columns.

Several rewrite steps (domain classification, sensitivity disposition, QA /
meaning-unit extraction, evaluate, final judge) embed the full document into a
single LLM call. For long documents this exceeds the render cap. This module
runs such a step over boundary-aligned windows and merges the per-window outputs
with a step-specific ``merge_fn``:

- domain          -> classify the first window only (coarse, doc-level label)
- disposition     -> union per-entity protection decisions
- meaning units   -> concatenate per-window units
- evaluate        -> aggregate metrics
- final judge     -> OR across windows

Rendering here (instead of via an ``LLMStructuredColumnConfig``) also sidesteps
NDD's ginja per-render length cap, like the chunked detection steps.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from jinja2 import BaseLoader, Environment, StrictUndefined
from pydantic import BaseModel, Field

from anonymizer.engine.windowing import DEFAULT_DELIMITER, iter_boundary_windows

logger = logging.getLogger("anonymizer.rewrite.chunked_steps")

_MIN_WINDOW_CHARS = 4000

_PROMPT_ENV = Environment(loader=BaseLoader(), autoescape=False, undefined=StrictUndefined, keep_trailing_newline=True)


@functools.lru_cache(maxsize=16)
def _compile_template(template: str) -> Any:
    return _PROMPT_ENV.from_string(template)


# A merge function takes the list of per-window parsed schema objects and returns
# a JSON-serializable value to store in the output column.
MergeFn = Callable[[list[Any]], Any]


class WindowedStepParams(BaseModel):
    """Params for a generic windowed rewrite metadata step (via DD ``generator_params``)."""

    alias: str = Field(min_length=1)
    prompt_template: str = Field(repr=False)
    output_column: str = Field(min_length=1)
    text_column: str = Field(min_length=1)
    max_render_chars: int = Field(gt=0)
    safety_margin_chars: int = Field(default=8000, ge=0)
    delimiter: str = Field(default=DEFAULT_DELIMITER)
    first_only: bool = Field(default=False)
    system_prompt: str | None = Field(default=None, repr=False)


def run_windowed_step(
    row: dict[str, Any],
    params: WindowedStepParams,
    models: dict[str, Any],
    *,
    schema: type[BaseModel],
    merge_fn: MergeFn,
    purpose_prefix: str,
) -> dict[str, Any]:
    """Run ``params.prompt_template`` over boundary windows of ``params.text_column``; store ``merge_fn`` result."""
    if params.alias not in models:
        raise KeyError(f"Alias {params.alias!r} not present in models dict for step {purpose_prefix!r}.")
    facade = models[params.alias]
    recipe = PydanticResponseRecipe(data_type=schema)
    cap = params.max_render_chars
    initial_window = max(_MIN_WINDOW_CHARS, cap - params.safety_margin_chars)

    def _call(prompt: str, purpose: str) -> Any:
        output, _ = facade.generate(
            prompt=recipe.apply_recipe_to_user_prompt(prompt),
            parser=recipe.parse,
            system_prompt=recipe.apply_recipe_to_system_prompt(params.system_prompt),
            purpose=purpose,
        )
        return output

    full_rendered = _compile_template(params.prompt_template).render(**row)
    if len(full_rendered) <= cap:
        row[params.output_column] = merge_fn([_call(full_rendered, purpose_prefix)])
        return row

    text = str(row.get(params.text_column, ""))
    windows = iter_boundary_windows(text, initial_window, delimiter=params.delimiter)
    if params.first_only:
        windows = windows[:1]
    # Run each window independently: a single transient model error (or a chunk
    # the model cannot parse into the schema) should drop only that window, not
    # the whole record. Failures are logged and skipped; the all-failed case is
    # handled explicitly below so we never merge an empty output set silently.
    outputs = []
    failed = 0
    for start, end in windows:
        rendered = _compile_template(params.prompt_template).render(**{**row, params.text_column: text[start:end]})
        try:
            outputs.append(_call(rendered, f"{purpose_prefix}-{start}"))
        except Exception:
            failed += 1
            logger.warning(
                "windowed step %s: window [%d, %d) failed and was skipped",
                purpose_prefix,
                start,
                end,
                exc_info=True,
            )
    if not outputs:
        raise RuntimeError(
            f"windowed step {purpose_prefix!r}: all {len(windows)} window(s) failed; no output produced "
            f"for column {params.output_column!r}."
        )
    if failed:
        logger.warning(
            "windowed step %s: %d of %d window(s) failed and were skipped", purpose_prefix, failed, len(windows)
        )
    logger.debug("windowed step %s: %d window(s) over %d chars", purpose_prefix, len(windows), len(text))
    row[params.output_column] = merge_fn(outputs)
    return row


def make_windowed_metadata_generator(
    *,
    alias: str,
    required_columns: list[str],
    schema: type[BaseModel],
    merge_fn: MergeFn,
    purpose_prefix: str,
) -> Any:
    """Build a ``@custom_column_generator`` running a windowed metadata step.

    ``schema``/``merge_fn`` are bound here (not in params) since they are not
    serializable; window sizing + prompt come from ``WindowedStepParams``.
    """
    if not alias:
        raise ValueError(f"Cannot build windowed step generator for {purpose_prefix}: alias is empty.")

    @custom_column_generator(required_columns=list(required_columns), model_aliases=[alias])
    def windowed_step(
        row: dict[str, Any],
        generator_params: WindowedStepParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return run_windowed_step(
            row, generator_params, models, schema=schema, merge_fn=merge_fn, purpose_prefix=purpose_prefix
        )

    return windowed_step
