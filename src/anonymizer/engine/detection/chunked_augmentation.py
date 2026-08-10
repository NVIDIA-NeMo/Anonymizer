# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Windowed LLM augmentation for the entity detection pipeline.

Augmentation finds entities the detector missed by showing the LLM the full
(tagged) document. For very long documents that prompt can blow past model
context / render budgets, so this module tiles the document into overlapping
windows, runs the augmenter per window, and unions the proposed entities.

Augmented entities carry **no offsets** -- positions are assigned later by
``apply_augmented_entities``, which locates each value in the full text. So the
per-window merge is a simple union/dedupe by ``(value, label)`` and produces the
same final result as a single-pass augmentation.

Rendering the prompt here (instead of via an ``LLMStructuredColumnConfig``) also
sidesteps NDD's ginja per-render length cap, exactly like chunked validation.

Public entry point: :func:`make_windowed_augmentation_generator`. The helpers
below are exposed for unit testing.
"""

from __future__ import annotations

import functools
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from jinja2 import BaseLoader, Environment, StrictUndefined
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_AUGMENTATION_FAILED_WINDOWS,
    COL_AUGMENTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TEXT,
    COL_VALIDATED_SEED_ENTITIES,
)
from anonymizer.engine.detection.postprocess import EntitySpan, TagNotation, build_tagged_text
from anonymizer.engine.schemas import AugmentedEntitiesSchema, EntitiesSchema

logger = logging.getLogger("anonymizer.detection.chunked_augmentation")

# Floor on window size so a pathologically entity-dense slice still makes
# progress instead of shrinking toward zero.
_MIN_WINDOW_CHARS = 4000

# Upper bound on augmentation windows dispatched concurrently for one record. The
# per-alias rate limit (``max_parallel_requests`` on the facade) still caps the
# real in-flight count; this just bounds thread creation on very long inputs.
_MAX_PARALLEL_WINDOWS = 8

# Jinja2 environment used to render the per-window augmentation prompt. Mirrors
# chunked_validation: same template, same placeholders, per-window values.
_PROMPT_ENV = Environment(
    loader=BaseLoader(),
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


@functools.lru_cache(maxsize=4)
def _compile_template(template: str) -> Any:
    return _PROMPT_ENV.from_string(template)


class WindowedAugmentationParams(BaseModel):
    """Parameters supplied to :func:`augment_row` via DD's ``generator_params``.

    Attributes:
        alias: Augmenter model alias (must also be in the decorator's
            ``model_aliases`` so DataDesigner materialises the facade).
        prompt_template: Jinja2 source for the augmentation prompt (with
            ``_initial_tagged_text``, ``_seed_entities_json``, ``_tag_notation``
            placeholders). Typically produced by ``_get_augment_prompt``.
        max_render_chars: Upper bound on a single rendered prompt's length;
            windows are sized so each render stays under
            ``max_render_chars - safety_margin_chars``.
        safety_margin_chars: Headroom subtracted from ``max_render_chars``.
        overlap_chars: Overlap between adjacent windows so an entity straddling
            a boundary is fully visible in at least one window.
        system_prompt: Optional system prompt forwarded to each call.

    ``prompt_template``/``system_prompt`` are ``repr=False`` because DD logs this
    model and the prompt is multi-kB.
    """

    alias: str = Field(min_length=1)
    prompt_template: str = Field(repr=False)
    max_render_chars: int = Field(gt=0)
    safety_margin_chars: int = Field(default=8000, ge=0)
    overlap_chars: int = Field(default=1000, ge=0)
    system_prompt: str | None = Field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Pure helpers (no DataDesigner, no LLM). Tested directly.
# ---------------------------------------------------------------------------


def parse_validated_seed_spans(raw_payload: object) -> list[EntitySpan]:
    """Parse ``COL_VALIDATED_SEED_ENTITIES`` (EntitiesSchema) into ``EntitySpan``s."""
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


def build_window_inputs(
    *,
    text: str,
    all_spans: list[EntitySpan],
    start: int,
    end: int,
    notation: TagNotation,
) -> tuple[str, str]:
    """Build the per-window tagged text and seed-entities JSON.

    Only seed spans fully contained in ``[start, end)`` are tagged (with offsets
    re-based to the window), so the window's tagged view matches the full-doc
    view. The seed JSON lists those same in-window seeds so the augmenter does
    not re-propose already-detected entities; positions are window-local but the
    augmenter only reads value/label.
    """
    window_raw = text[start:end]
    in_window = [
        EntitySpan(
            entity_id=span.entity_id,
            value=span.value,
            label=span.label,
            start_position=span.start_position - start,
            end_position=span.end_position - start,
            score=span.score,
            source=span.source,
        )
        for span in all_spans
        if span.start_position >= start and span.end_position <= end
    ]
    tagged = build_tagged_text(window_raw, in_window, notation=notation)
    seed_json = json.dumps([span.as_dict() for span in in_window])
    return tagged, seed_json


def render_augment_prompt(
    *,
    template: str,
    tagged_text: str,
    seed_entities_json: str,
    notation: TagNotation,
) -> str:
    """Render the augmentation prompt for a single window via Jinja2."""
    compiled = _compile_template(template)
    return compiled.render(
        **{
            COL_INITIAL_TAGGED_TEXT: tagged_text,
            COL_SEED_ENTITIES_JSON: seed_entities_json,
            COL_TAG_NOTATION: notation.value,
        }
    )


def merge_augmented(results: list[AugmentedEntitiesSchema]) -> AugmentedEntitiesSchema:
    """Union per-window augmented entities, deduping by (normalized value, label)."""
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for result in results:
        for entity in result.entities:
            value = entity.value.strip()
            if not value:
                continue
            key = (value.casefold(), entity.label)
            if key in seen:
                continue
            seen.add(key)
            merged.append({"value": entity.value, "label": entity.label, "reason": entity.reason})
    return AugmentedEntitiesSchema.model_validate({"entities": merged})


def iter_windows(text_len: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Tile ``[0, text_len)`` into ``[start, end)`` windows of size ``window`` with ``overlap``."""
    if text_len <= 0:
        return []
    window = max(window, _MIN_WINDOW_CHARS)
    step = max(1, window - overlap)
    bounds: list[tuple[int, int]] = []
    pos = 0
    while pos < text_len:
        end = min(text_len, pos + window)
        bounds.append((pos, end))
        if end >= text_len:
            break
        pos += step
    return bounds


# ---------------------------------------------------------------------------
# Dispatch. Testable by passing a fake facade.
# ---------------------------------------------------------------------------


def _call_augmenter(*, facade: Any, prompt: str, system_prompt: str | None, purpose: str) -> AugmentedEntitiesSchema:
    """Call the augmenter facade with structured output and return the parsed schema."""
    recipe = PydanticResponseRecipe(data_type=AugmentedEntitiesSchema)
    final_prompt = recipe.apply_recipe_to_user_prompt(prompt)
    final_system = recipe.apply_recipe_to_system_prompt(system_prompt)
    output, _messages = facade.generate(
        prompt=final_prompt,
        parser=recipe.parse,
        system_prompt=final_system,
        purpose=purpose,
    )
    return output


def plan_augmentation_windows(
    *,
    text: str,
    all_spans: list[EntitySpan],
    notation: TagNotation,
    params: WindowedAugmentationParams,
    cap: int,
    initial_window: int,
) -> list[tuple[int, int, str]]:
    """Walk the document applying the shrink rule, returning ``(start, end, rendered_prompt)``.

    No LLM calls — only Jinja renders + length checks — so this is cheap and runs
    serially to fix the (data-dependent) window boundaries before the parallel LLM
    pass. The window shrinks when a render exceeds ``cap`` and the shrunk size
    carries forward, exactly as the previous in-line loop did.
    """
    text_len = len(text)
    windows: list[tuple[int, int, str]] = []
    window = initial_window
    pos = 0
    while pos < text_len:
        end = min(text_len, pos + window)
        tagged, seed_json = build_window_inputs(text=text, all_spans=all_spans, start=pos, end=end, notation=notation)
        rendered = render_augment_prompt(
            template=params.prompt_template, tagged_text=tagged, seed_entities_json=seed_json, notation=notation
        )
        if len(rendered) > cap and (end - pos) > _MIN_WINDOW_CHARS:
            # Shrink proportionally to the measured overage so the next try lands just
            # under the cap (0.95 = small safety margin). Strictly decreasing -> converges.
            shrunk = max(_MIN_WINDOW_CHARS, int(window * (cap / len(rendered)) * 0.95))
            logger.debug(
                "augmentation @pos=%d: render %d > cap %d; shrinking window %d -> %d chars and retrying",
                pos,
                len(rendered),
                cap,
                window,
                shrunk,
            )
            window = shrunk
            continue
        logger.debug(
            "augmentation window @[%d, %d) size=%d, rendered=%d/%d chars", pos, end, end - pos, len(rendered), cap
        )
        windows.append((pos, end, rendered))
        if end >= text_len:
            break
        pos = max(pos + 1, end - params.overlap_chars)
    return windows


def _augment_window(
    *, facade: Any, rendered: str, system_prompt: str | None, start: int
) -> AugmentedEntitiesSchema | None:
    """Run one augmentation window; return its result, or ``None`` if the call fails.

    A single window's failure (e.g. an unparseable model response, or a timeout)
    must not drop the whole record: augmentation only *adds* entities GLiNER missed,
    so a skipped window degrades gracefully. The error is logged at WARNING with its
    type and message so it stays fully visible. Safe to run per window in a pool.
    """
    try:
        result = _call_augmenter(
            facade=facade, prompt=rendered, system_prompt=system_prompt, purpose=f"entity-augmentation-window-{start}"
        )
    except Exception as exc:  # noqa: BLE001 — one window must not sink the record
        logger.warning(
            "augmentation window @%d failed (%s: %s); skipping this window's entities", start, type(exc).__name__, exc
        )
        return None
    logger.debug("augmentation window @%d: augmenter proposed %d entities", start, len(result.entities))
    return result


def augment_row(
    row: dict[str, Any],
    params: WindowedAugmentationParams,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Run (possibly windowed) augmentation for a single row, writing ``COL_AUGMENTED_ENTITIES``.

    Call directly in tests with a fake ``models`` dict; the DataDesigner-decorated
    wrapper from :func:`make_windowed_augmentation_generator` just forwards here.
    """
    if params.alias not in models:
        raise KeyError(
            f"Augmenter alias {params.alias!r} not present in models dict. Ensure "
            "make_windowed_augmentation_generator was invoked with the same alias "
            "passed in WindowedAugmentationParams.alias."
        )
    facade = models[params.alias]

    text = str(row.get(COL_TEXT, ""))
    notation = TagNotation(str(row.get(COL_TAG_NOTATION) or TagNotation.sentinel.value))
    # ``cap`` is the hard ceiling a rendered prompt may not exceed. ``initial_window``
    # sizes the first raw window below it by ``safety_margin_chars`` so the per-window
    # render overhead (scaffolding + tags + seed JSON) usually fits without shrinking.
    cap = params.max_render_chars
    initial_window = max(_MIN_WINDOW_CHARS, cap - params.safety_margin_chars)

    # Fast path: the full tagged document fits under the cap, so behave exactly like
    # the pre-windowing single-call augmentation.
    full_tagged = str(row.get(COL_INITIAL_TAGGED_TEXT, ""))
    full_seed_json = str(row.get(COL_SEED_ENTITIES_JSON) or "[]")
    full_rendered = render_augment_prompt(
        template=params.prompt_template,
        tagged_text=full_tagged,
        seed_entities_json=full_seed_json,
        notation=notation,
    )
    if len(full_rendered) <= cap:
        logger.debug("augmentation: single-call fast path (rendered=%d chars <= cap=%d)", len(full_rendered), cap)
        output = _call_augmenter(
            facade=facade,
            prompt=full_rendered,
            system_prompt=params.system_prompt,
            purpose="entity-augmentation",
        )
        row[COL_AUGMENTED_ENTITIES] = output.model_dump(mode="json")
        row[COL_AUGMENTATION_FAILED_WINDOWS] = 0
        return row

    # Windowed path. First plan the windows serially (cheap renders + the
    # data-dependent shrink), then run the LLM calls in parallel. A single
    # window's failure is logged and skipped rather than dropping the record.
    all_spans = parse_validated_seed_spans(row.get(COL_VALIDATED_SEED_ENTITIES, {}))
    text_len = len(text)
    windows = plan_augmentation_windows(
        text=text, all_spans=all_spans, notation=notation, params=params, cap=cap, initial_window=initial_window
    )
    max_workers = min(len(windows), _MAX_PARALLEL_WINDOWS)
    logger.info(
        "augmentation: rendered prompt %d chars > cap %d; tiling %d-char document into %d overlapping "
        "window(s) (initial_window=%d, overlap=%d, min_window=%d, max_workers=%d)",
        len(full_rendered),
        cap,
        text_len,
        len(windows),
        initial_window,
        params.overlap_chars,
        _MIN_WINDOW_CHARS,
        max_workers,
    )

    results: list[AugmentedEntitiesSchema] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _augment_window, facade=facade, rendered=rendered, system_prompt=params.system_prompt, start=start
            )
            for start, _end, rendered in windows
        ]
        for future in futures:
            result = future.result()  # _augment_window swallows errors -> never raises
            if result is not None:
                results.append(result)

    failed = len(windows) - len(results)
    if failed:
        logger.warning("augmentation: %d of %d window(s) failed and were skipped", failed, len(windows))
    merged = merge_augmented(results)
    logger.info(
        "augmentation: %d window(s) over %d chars -> %d unique entities after dedupe (cap=%d, overlap=%d, %d failed)",
        len(windows),
        text_len,
        len(merged.entities),
        cap,
        params.overlap_chars,
        failed,
    )
    row[COL_AUGMENTED_ENTITIES] = merged.model_dump(mode="json")
    row[COL_AUGMENTATION_FAILED_WINDOWS] = failed
    return row


# ---------------------------------------------------------------------------
# DataDesigner wiring factory.
# ---------------------------------------------------------------------------


def make_windowed_augmentation_generator(alias: str) -> Any:
    """Build a ``@custom_column_generator``-decorated function bound to ``alias``.

    ``model_aliases`` must be declared statically so DataDesigner materialises the
    augmenter facade. ``required_columns`` are exhaustive for DD's DAG ordering:
    the generator reads the raw text, the full tagged text + seed JSON (fast
    path), the validated seed spans (to rebuild per-window tagged text), and the
    tag notation.
    """
    if not alias:
        raise ValueError("Cannot build windowed augmentation generator: alias is empty.")

    @custom_column_generator(
        required_columns=[
            COL_TEXT,
            COL_INITIAL_TAGGED_TEXT,
            COL_SEED_ENTITIES_JSON,
            COL_VALIDATED_SEED_ENTITIES,
            COL_TAG_NOTATION,
        ],
        side_effect_columns=[COL_AUGMENTATION_FAILED_WINDOWS],
        model_aliases=[alias],
    )
    def windowed_augment(
        row: dict[str, Any],
        generator_params: WindowedAugmentationParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return augment_row(row, generator_params, models)

    return windowed_augment
