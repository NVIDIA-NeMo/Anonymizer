# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Windowed latent-entity detection (rewrite-mode only) for long documents.

Latent entities are *inferred*, non-explicit sensitive attributes; they carry no
offsets, so per-window results merge by union (dedupe by ``(label, value)``).
Like windowed augmentation, rendering the prompt here (instead of via an
``LLMStructuredColumnConfig``) bypasses NDD's ginja per-render length cap.

Caveat: windowing necessarily limits cross-window inference. A latent fact only
deducible by combining distant parts of a very long document may be missed. This
is a pragmatic trade so long documents do not fail outright; for documents that
fit in one window behaviour is unchanged (fast path).

Public entry point: :func:`make_windowed_latent_generator`.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from jinja2 import BaseLoader, Environment, StrictUndefined
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.chunked_augmentation import build_window_inputs, iter_windows
from anonymizer.engine.detection.postprocess import EntitySpan, TagNotation
from anonymizer.engine.schemas import EntitiesSchema, LatentEntitiesSchema

logger = logging.getLogger("anonymizer.detection.chunked_latent")

# Floor on window size so a pathologically entity-dense slice still progresses.
_MIN_WINDOW_CHARS = 4000

_PROMPT_ENV = Environment(
    loader=BaseLoader(),
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


@functools.lru_cache(maxsize=4)
def _compile_template(template: str) -> Any:
    return _PROMPT_ENV.from_string(template)


class WindowedLatentParams(BaseModel):
    """Parameters supplied to :func:`latent_row` via DD's ``generator_params``.

    Mirrors ``WindowedAugmentationParams`` but for the latent prompt, which reads
    ``_tagged_text`` (the finalized tagged document) and ``_tag_notation``.
    """

    alias: str = Field(min_length=1)
    prompt_template: str = Field(repr=False)
    max_render_chars: int = Field(gt=0)
    safety_margin_chars: int = Field(default=8000, ge=0)
    overlap_chars: int = Field(default=1000, ge=0)
    system_prompt: str | None = Field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def parse_detected_spans(raw_payload: object) -> list[EntitySpan]:
    """Parse ``COL_DETECTED_ENTITIES`` (EntitiesSchema) into ``EntitySpan``s."""
    parsed = EntitiesSchema.from_raw(raw_payload)
    return [
        EntitySpan(
            entity_id=e.id,
            value=e.value,
            label=e.label,
            start_position=e.start_position,
            end_position=e.end_position,
            score=e.score,
            source=e.source,
        )
        for e in parsed.entities
    ]


def render_latent_prompt(*, template: str, tagged_text: str, notation: TagNotation) -> str:
    """Render the latent prompt for a single window via Jinja2."""
    compiled = _compile_template(template)
    return compiled.render(**{COL_TAGGED_TEXT: tagged_text, COL_TAG_NOTATION: notation.value})


def merge_latent(results: list[LatentEntitiesSchema]) -> LatentEntitiesSchema:
    """Union per-window latent entities, deduping by ``(label, normalized value)``."""
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for result in results:
        for entity in result.latent_entities:
            value = entity.value.strip()
            if not value:
                continue
            key = (entity.label.casefold(), value.casefold())
            if key in seen:
                continue
            seen.add(key)
            merged.append(entity.model_dump(mode="json"))
    return LatentEntitiesSchema.model_validate({"latent_entities": merged})


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _call_latent(*, facade: Any, prompt: str, system_prompt: str | None, purpose: str) -> LatentEntitiesSchema:
    recipe = PydanticResponseRecipe(data_type=LatentEntitiesSchema)
    final_prompt = recipe.apply_recipe_to_user_prompt(prompt)
    final_system = recipe.apply_recipe_to_system_prompt(system_prompt)
    output, _messages = facade.generate(
        prompt=final_prompt,
        parser=recipe.parse,
        system_prompt=final_system,
        purpose=purpose,
    )
    return output


def latent_row(
    row: dict[str, Any],
    params: WindowedLatentParams,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Run (possibly windowed) latent detection for a single row, writing ``COL_LATENT_ENTITIES``."""
    if params.alias not in models:
        raise KeyError(
            f"Latent alias {params.alias!r} not present in models dict. Ensure "
            "make_windowed_latent_generator was invoked with the same alias passed "
            "in WindowedLatentParams.alias."
        )
    facade = models[params.alias]

    text = str(row.get(COL_TEXT, ""))
    notation = TagNotation(str(row.get(COL_TAG_NOTATION) or TagNotation.sentinel.value))
    # ``cap`` is the hard ceiling a rendered prompt may not exceed. ``initial_window``
    # sizes the first raw window below it by ``safety_margin_chars`` so per-window
    # render overhead (scaffolding + tags) usually fits without shrinking.
    cap = params.max_render_chars
    initial_window = max(_MIN_WINDOW_CHARS, cap - params.safety_margin_chars)

    # Fast path: the full finalized tagged document fits under the cap.
    full_tagged = str(row.get(COL_TAGGED_TEXT, ""))
    full_rendered = render_latent_prompt(template=params.prompt_template, tagged_text=full_tagged, notation=notation)
    if len(full_rendered) <= cap:
        logger.debug("latent: single-call fast path (rendered=%d chars <= cap=%d)", len(full_rendered), cap)
        output = _call_latent(
            facade=facade,
            prompt=full_rendered,
            system_prompt=params.system_prompt,
            purpose="latent-detection",
        )
        row[COL_LATENT_ENTITIES] = output.model_dump(mode="json")
        return row

    # Windowed path.
    spans = parse_detected_spans(row.get(COL_DETECTED_ENTITIES, {}))
    results: list[LatentEntitiesSchema] = []
    window = initial_window
    pos = 0
    text_len = len(text)
    logger.info(
        "latent: rendered prompt %d chars > cap %d; tiling %d-char document into "
        "overlapping windows (initial_window=%d, overlap=%d, min_window=%d)",
        len(full_rendered), cap, text_len, initial_window, params.overlap_chars, _MIN_WINDOW_CHARS,
    )
    window_index = 0
    while pos < text_len:
        end = min(text_len, pos + window)
        # Reuse augmentation's span-rebasing + tagging; latent ignores the seed JSON.
        tagged, _seed_json = build_window_inputs(
            text=text, all_spans=spans, start=pos, end=end, notation=notation
        )
        rendered = render_latent_prompt(template=params.prompt_template, tagged_text=tagged, notation=notation)
        if len(rendered) > cap and (end - pos) > _MIN_WINDOW_CHARS:
            # Shrink proportionally to the measured overage so the next try lands
            # just under the cap (0.95 = small safety margin), instead of halving
            # and overshooting. Strictly decreasing -> converges, then the floor stops it.
            shrunk = max(_MIN_WINDOW_CHARS, int(window * (cap / len(rendered)) * 0.95))
            logger.debug(
                "latent window %d @pos=%d: render %d > cap %d; shrinking window %d -> %d chars and retrying",
                window_index, pos, len(rendered), cap, window, shrunk,
            )
            window = shrunk
            continue
        logger.debug(
            "latent window %d: chars [%d, %d) size=%d, rendered=%d/%d chars",
            window_index, pos, end, end - pos, len(rendered), cap,
        )
        result = _call_latent(
            facade=facade,
            prompt=rendered,
            system_prompt=params.system_prompt,
            purpose=f"latent-detection-window-{pos}",
        )
        logger.debug("latent window %d: detector proposed %d latent entities", window_index, len(result.latent_entities))
        results.append(result)
        if end >= text_len:
            break
        next_pos = max(pos + 1, end - params.overlap_chars)
        logger.debug(
            "latent window %d: advancing pos %d -> %d (overlap back %d chars)",
            window_index, pos, next_pos, end - next_pos,
        )
        pos = next_pos
        window_index += 1

    merged = merge_latent(results)
    logger.info(
        "latent: %d window(s) over %d chars -> %d unique latent entities after dedupe (cap=%d, overlap=%d)",
        len(results), text_len, len(merged.latent_entities), cap, params.overlap_chars,
    )
    row[COL_LATENT_ENTITIES] = merged.model_dump(mode="json")
    return row


# ---------------------------------------------------------------------------
# DataDesigner wiring factory.
# ---------------------------------------------------------------------------


def make_windowed_latent_generator(alias: str) -> Any:
    """Build a ``@custom_column_generator``-decorated function bound to ``alias``."""
    if not alias:
        raise ValueError("Cannot build windowed latent generator: alias is empty.")

    @custom_column_generator(
        required_columns=[
            COL_TEXT,
            COL_TAGGED_TEXT,
            COL_DETECTED_ENTITIES,
            COL_TAG_NOTATION,
        ],
        model_aliases=[alias],
    )
    def windowed_latent(
        row: dict[str, Any],
        generator_params: WindowedLatentParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return latent_row(row, generator_params, models)

    return windowed_latent


# Re-exported for symmetry/tests; latent windowing reuses the augmentation tiler.
__all__ = [
    "WindowedLatentParams",
    "iter_windows",
    "latent_row",
    "make_windowed_latent_generator",
    "merge_latent",
    "parse_detected_spans",
    "render_latent_prompt",
]
