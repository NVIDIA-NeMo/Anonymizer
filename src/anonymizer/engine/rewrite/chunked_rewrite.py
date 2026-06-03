# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-context (chunked) rewrite generation.

The rewrite-generation step normally rewrites the whole tagged document in a
single LLM call. For documents whose rendered prompt would exceed the render
cap, this module rewrites the document in boundary-aligned chunks and stitches
the results, carrying a rolling summary of what has already been rewritten so the
narrative stays coherent across chunks. The (already global, consistency-checked)
replacement map and the protection-disposition block are passed per chunk,
filtered to the entities that occur in that chunk.

Rendering the prompt here (instead of via an ``LLMStructuredColumnConfig``) also
sidesteps NDD's ginja per-render length cap, like the chunked detection steps.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe, TextResponseRecipe
from jinja2 import BaseLoader, Environment, StrictUndefined
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_FULL_REWRITE,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.postprocess import TagNotation
from anonymizer.engine.schemas import RewriteOutputSchema
from anonymizer.engine.windowing import DEFAULT_DELIMITER, iter_boundary_windows

logger = logging.getLogger("anonymizer.rewrite.chunked")

_MIN_WINDOW_CHARS = 4000

# Max characters of free-form text (e.g. a rolling summary) to emit in a single
# debug line, so logs stay readable even when the underlying value is large.
_LOG_CLIP_CHARS = 800


def _clip(text: str, limit: int = _LOG_CLIP_CHARS) -> str:
    """Single-line, length-bounded rendering of ``text`` for debug logs."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}… (+{len(flat) - limit} chars)"


_PROMPT_ENV = Environment(loader=BaseLoader(), autoescape=False, undefined=StrictUndefined, keep_trailing_newline=True)


@functools.lru_cache(maxsize=8)
def _compile_template(template: str) -> Any:
    return _PROMPT_ENV.from_string(template)


# Preamble injected ahead of the per-chunk rewrite prompt to carry continuity.
_CONTINUITY_PREAMBLE = """<continuity>
You are rewriting ONE section of a longer document. Below is a summary of how the
EARLIER sections have already been rewritten. Keep this section consistent with it
(same pseudonyms, tone, and narrative); do NOT repeat earlier content. Output only
the rewritten text for THIS section.

Summary so far:
{{ summary }}
</continuity>

"""

_SUMMARY_PROMPT = """You maintain a concise running summary of a long document being rewritten for privacy.
Update the summary to incorporate the newly rewritten section, keeping only what is needed to keep
later sections consistent (narrative state, established pseudonyms/relationships). Be terse; hard
limit {{ summary_max_chars }} characters. Return only the updated summary.

Previous summary:
{{ prev_summary }}

Newly rewritten section:
{{ rewritten_chunk }}
"""


class WindowedRewriteParams(BaseModel):
    """Params for chunked rewrite generation (via DD ``generator_params``)."""

    alias: str = Field(min_length=1)
    single_call_prompt_template: str = Field(repr=False)
    max_render_chars: int = Field(gt=0)
    safety_margin_chars: int = Field(default=8000, ge=0)
    summary_max_chars: int = Field(default=2000, gt=0)
    delimiter: str = Field(default=DEFAULT_DELIMITER)
    system_prompt: str | None = Field(default=None, repr=False)


def _filter_disposition_to_chunk(block: list[dict[str, Any]], chunk_raw: str) -> list[dict[str, Any]]:
    """Keep disposition entries whose entity value appears in this chunk's raw text."""
    if not isinstance(block, list):
        return []
    return [
        e
        for e in block
        if isinstance(e, dict) and str(e.get("entity_value", "")) and str(e["entity_value"]) in chunk_raw
    ]


def _render_chunk_prompt(*, template: str, chunk_row: dict[str, Any], summary: str) -> str:
    """Render the rewrite prompt for one chunk, prepended with the continuity preamble."""
    preamble = _compile_template(_CONTINUITY_PREAMBLE).render(summary=summary or "(this is the first section)")
    body = _compile_template(template).render(**chunk_row)
    return preamble + body


def _rewrite_chunk(*, facade: Any, prompt: str, system_prompt: str | None, purpose: str) -> str:
    recipe = PydanticResponseRecipe(data_type=RewriteOutputSchema)
    output, _ = facade.generate(
        prompt=recipe.apply_recipe_to_user_prompt(prompt),
        parser=recipe.parse,
        system_prompt=recipe.apply_recipe_to_system_prompt(system_prompt),
        purpose=purpose,
    )
    text = ""
    if output is not None:
        dumped = output.model_dump(mode="python") if hasattr(output, "model_dump") else output
        text = str(dumped.get("rewritten_text", "")) if isinstance(dumped, dict) else ""
    return text


def _update_summary(
    *,
    facade: Any,
    prev_summary: str,
    rewritten_chunk: str,
    summary_max_chars: int,
    system_prompt: str | None,
    purpose: str,
) -> str:
    recipe = TextResponseRecipe()
    rendered = _compile_template(_SUMMARY_PROMPT).render(
        prev_summary=prev_summary or "(none yet)", rewritten_chunk=rewritten_chunk, summary_max_chars=summary_max_chars
    )
    output, _ = facade.generate(
        prompt=recipe.apply_recipe_to_user_prompt(rendered),
        parser=recipe.parse,
        system_prompt=recipe.apply_recipe_to_system_prompt(system_prompt),
        purpose=purpose,
    )
    return str(output)[:summary_max_chars]


def generate_rewrite_row(
    row: dict[str, Any],
    params: WindowedRewriteParams,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Produce ``COL_FULL_REWRITE`` for one row, chunking long documents with rolling-summary continuity."""
    if params.alias not in models:
        raise KeyError(
            f"Rewriter alias {params.alias!r} not present in models dict. Ensure "
            "make_windowed_rewrite_generator was invoked with the same alias."
        )
    facade = models[params.alias]
    cap = params.max_render_chars
    initial_window = max(_MIN_WINDOW_CHARS, cap - params.safety_margin_chars)

    # Fast path: the full single-call rewrite prompt fits under the cap.
    single_rendered = _render_chunk_prompt(template=params.single_call_prompt_template, chunk_row=row, summary="")
    if len(single_rendered) <= cap:
        logger.debug("rewrite: single-call fast path (rendered=%d chars <= cap=%d)", len(single_rendered), cap)
        text = _rewrite_chunk(
            facade=facade,
            prompt=_compile_template(params.single_call_prompt_template).render(**row),
            system_prompt=params.system_prompt,
            purpose="rewrite-generation",
        )
        row[COL_FULL_REWRITE] = RewriteOutputSchema(rewritten_text=text).model_dump(mode="json")
        return row

    # Chunked path: rewrite each boundary window with continuity carry-over, then stitch.
    tagged = str(row.get(COL_TAGGED_TEXT, ""))
    notation = TagNotation(str(row.get(COL_TAG_NOTATION) or TagNotation.sentinel.value))
    disposition_block = row.get(COL_REWRITE_DISPOSITION_BLOCK, [])
    replacement_map = row.get(COL_REPLACEMENT_MAP_FOR_PROMPT, {"replacements": []})

    # Chunk on the already-tagged text so tag boundaries stay intact; the delimiter
    # default ("\n") keeps cuts on line breaks.
    windows = iter_boundary_windows(tagged, initial_window, delimiter=params.delimiter)
    logger.info(
        "rewrite: rendered prompt %d chars > cap %d; chunking %d-char tagged document into %d boundary "
        "window(s) (initial_window=%d, delimiter=%r, summary_max=%d)",
        len(single_rendered),
        cap,
        len(tagged),
        len(windows),
        initial_window,
        params.delimiter,
        params.summary_max_chars,
    )
    rewritten_parts: list[str] = []
    summary = ""
    for i, (start, end) in enumerate(windows):
        chunk_tagged = tagged[start:end]
        chunk_disposition = _filter_disposition_to_chunk(disposition_block, chunk_tagged)
        chunk_row = {
            **row,
            COL_TAGGED_TEXT: chunk_tagged,
            COL_TAG_NOTATION: notation.value,
            COL_REWRITE_DISPOSITION_BLOCK: chunk_disposition,
            COL_REPLACEMENT_MAP_FOR_PROMPT: replacement_map,
        }
        prompt = _render_chunk_prompt(template=params.single_call_prompt_template, chunk_row=chunk_row, summary=summary)
        logger.debug(
            "rewrite window %d/%d: chars [%d, %d) size=%d, %d in-chunk disposition entr(y/ies), prompt=%d chars",
            i + 1,
            len(windows),
            start,
            end,
            end - start,
            len(chunk_disposition),
            len(prompt),
        )
        rewritten_chunk = _rewrite_chunk(
            facade=facade,
            prompt=prompt,
            system_prompt=params.system_prompt,
            purpose=f"rewrite-generation-chunk-{start}",
        )
        logger.debug(
            "rewrite window %d/%d: produced %d chars of rewritten text",
            i + 1,
            len(windows),
            len(rewritten_chunk),
        )
        rewritten_parts.append(rewritten_chunk)
        if i < len(windows) - 1:
            summary = _update_summary(
                facade=facade,
                prev_summary=summary,
                rewritten_chunk=rewritten_chunk,
                summary_max_chars=params.summary_max_chars,
                system_prompt=params.system_prompt,
                purpose=f"rewrite-summary-{start}",
            )
            logger.debug(
                "rewrite window %d/%d: continuity summary updated -> %d chars: %s",
                i + 1,
                len(windows),
                len(summary),
                _clip(summary),
            )

    stitched = "\n".join(part for part in rewritten_parts if part)
    logger.info(
        "rewrite: %d window(s) over %d chars -> %d chars stitched output",
        len(windows),
        len(tagged),
        len(stitched),
    )
    row[COL_FULL_REWRITE] = RewriteOutputSchema(rewritten_text=stitched).model_dump(mode="json")
    return row


def make_windowed_rewrite_generator(alias: str) -> Any:
    """Build a ``@custom_column_generator`` for chunked rewrite generation bound to ``alias``."""
    if not alias:
        raise ValueError("Cannot build windowed rewrite generator: alias is empty.")

    @custom_column_generator(
        required_columns=[
            COL_TEXT,
            COL_TAGGED_TEXT,
            COL_TAG_NOTATION,
            COL_REWRITE_DISPOSITION_BLOCK,
            COL_REPLACEMENT_MAP_FOR_PROMPT,
        ],
        model_aliases=[alias],
    )
    def windowed_rewrite(
        row: dict[str, Any],
        generator_params: WindowedRewriteParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return generate_rewrite_row(row, generator_params, models)

    return windowed_rewrite
