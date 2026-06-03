# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Windowed GLiNER (seed) detection for long documents.

The seed detector normally runs on the whole document via a single DataDesigner
``LLMTextColumnConfig`` whose prompt *is* the raw text. DataDesigner renders that
prompt through its ginja engine, which enforces a hard ``MAX_RENDERED_LEN`` cap
(512000 chars) on every template — so a document larger than that fails outright
before any of the later (already windowed) stages run.

This module tiles the raw text into overlapping windows, runs the detector per
window, rebases each window's character offsets back to the full document, and
merges the results — bypassing the ginja cap exactly like the chunked
augmentation/validation steps. The detector returns char offsets relative to the
submitted text, so rebasing is ``+window_start``.

Boundary handling (an entity cut by a window edge)
--------------------------------------------------
Windows overlap by ``overlap_chars`` so any entity straddling a cut is fully
*interior* to at least one adjacent window (this holds as long as the overlap is
larger than the longest entity — true for PII spans at the 1000-char default).
On top of that:

* Spans that touch an **artificial** window edge (a left edge where
  ``window_start > 0``, or a right edge where ``window_end < len(text)``) are
  dropped: they are the truncated half of a straddling entity, and the full span
  is recovered from the neighbouring window where it sits interior.
* After merging, :func:`resolve_overlaps` keeps the longest span among any that
  overlap, which both de-duplicates the identical copies produced in the overlap
  region and, as a backstop, discards any partial that still overlaps a full span.

The output column (``COL_RAW_DETECTED``) is re-emitted in the detector's own JSON
shape (``{"entities": [{"text", "label", "start", "end", "score"}]}``) with global
offsets, so the downstream ``parse_detected_entities`` step is unchanged.

Public entry point: :func:`make_windowed_detection_generator`.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import TextResponseRecipe
from pydantic import BaseModel, Field

from anonymizer.engine.constants import COL_RAW_DETECTED, COL_TEXT
from anonymizer.engine.detection.chunked_augmentation import iter_windows
from anonymizer.engine.detection.postprocess import EntitySpan, parse_raw_entities, resolve_overlaps

logger = logging.getLogger("anonymizer.detection.chunked_detection")

# Floor on window size so a pathologically small cap still makes progress.
_MIN_WINDOW_CHARS = 4000

# Upper bound on detector windows dispatched concurrently for one record. Windows
# are independent LLM calls, so they run in parallel; the per-alias rate limit
# (``max_parallel_requests`` on the facade's ThrottledModelClient) still caps the
# real in-flight count, so this just bounds thread creation on very long inputs.
_MAX_PARALLEL_WINDOWS = 16


class WindowedDetectionParams(BaseModel):
    """Parameters supplied to :func:`detect_row` via DD's ``generator_params``.

    Attributes:
        alias: Detector model alias (must also be in the decorator's
            ``model_aliases`` so DataDesigner materialises the facade).
        max_render_chars: Upper bound on the text submitted in one detector call;
            windows are sized to ``max_render_chars - safety_margin_chars``.
        safety_margin_chars: Headroom subtracted from ``max_render_chars``.
        overlap_chars: Overlap between adjacent windows. Must exceed the longest
            expected entity so an entity straddling a window boundary is fully
            visible (interior) in at least one window.
        system_prompt: Optional system prompt; the hosted detector takes none.
    """

    alias: str = Field(min_length=1)
    max_render_chars: int = Field(gt=0)
    safety_margin_chars: int = Field(default=8000, ge=0)
    overlap_chars: int = Field(default=1000, ge=0)
    system_prompt: str | None = Field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Pure helpers (no DataDesigner, no LLM). Tested directly.
# ---------------------------------------------------------------------------


def rebase_spans(spans: list[EntitySpan], offset: int) -> list[EntitySpan]:
    """Shift window-local spans to full-document coordinates by ``offset``."""
    if offset == 0:
        return list(spans)
    return [
        EntitySpan(
            entity_id=s.entity_id,
            value=s.value,
            label=s.label,
            start_position=s.start_position + offset,
            end_position=s.end_position + offset,
            score=s.score,
            source=s.source,
        )
        for s in spans
    ]


def drop_boundary_spans(
    spans: list[EntitySpan], *, window_start: int, window_end: int, text_len: int
) -> list[EntitySpan]:
    """Drop spans touching an *artificial* window edge (a likely-truncated entity).

    ``spans`` are in full-document coordinates. A span touching the left edge is
    dropped only when this window does not start at the document start
    (``window_start > 0``); likewise the right edge only when it is not the
    document end (``window_end < text_len``). The dropped span's full form is
    recovered from the overlapping neighbour window, where it sits interior.
    """
    left_artificial = window_start > 0
    right_artificial = window_end < text_len
    kept: list[EntitySpan] = []
    for s in spans:
        if left_artificial and s.start_position <= window_start:
            continue
        if right_artificial and s.end_position >= window_end:
            continue
        kept.append(s)
    return kept


def spans_to_detector_payload(spans: list[EntitySpan]) -> str:
    """Serialize merged spans back into the detector's raw JSON shape."""
    return json.dumps(
        {
            "entities": [
                {
                    "text": s.value,
                    "label": s.label,
                    "start": s.start_position,
                    "end": s.end_position,
                    "score": s.score,
                }
                for s in spans
            ]
        }
    )


# ---------------------------------------------------------------------------
# Dispatch. Testable by passing a fake facade.
# ---------------------------------------------------------------------------


def _call_detector(*, facade: Any, prompt: str, system_prompt: str | None, purpose: str) -> str:
    """Call the detector facade with the text as a plain prompt; return raw text.

    Uses ``TextResponseRecipe`` (a pass-through that adds no task instructions),
    so the submitted prompt is exactly the window text — identical to the
    ``LLMTextColumnConfig(prompt=_jinja(COL_TEXT))`` this replaces.
    """
    recipe = TextResponseRecipe()
    final_prompt = recipe.apply_recipe_to_user_prompt(prompt)
    final_system = recipe.apply_recipe_to_system_prompt(system_prompt)
    output, _messages = facade.generate(
        prompt=final_prompt,
        parser=recipe.parse,
        system_prompt=final_system,
        purpose=purpose,
    )
    return str(output)


def _detect_window(
    *, facade: Any, text: str, start: int, end: int, text_len: int, system_prompt: str | None
) -> list[EntitySpan]:
    """Detect entities in one window and return them in full-document coordinates.

    Runs the detector on ``text[start:end]`` (window-local offsets), rebases to
    global offsets, then drops spans touching an artificial window edge (truncated
    halves of straddling entities — recovered interior to the overlapping
    neighbour). Pure w.r.t. shared state, so it is safe to run per window in a
    thread pool.
    """
    window_text = text[start:end]
    raw = _call_detector(
        facade=facade,
        prompt=window_text,
        system_prompt=system_prompt,
        purpose=f"entity-detection-window-{start}",
    )
    local_spans = parse_raw_entities(raw_response=raw, text=window_text)
    global_spans = rebase_spans(local_spans, start)
    kept = drop_boundary_spans(global_spans, window_start=start, window_end=end, text_len=text_len)
    logger.debug(
        "detection window [%d, %d) size=%d -> %d span(s), %d after edge-drop",
        start, end, end - start, len(local_spans), len(kept),
    )
    return kept


def detect_row(
    row: dict[str, Any],
    params: WindowedDetectionParams,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Run (possibly windowed) seed detection for one row, writing ``COL_RAW_DETECTED``."""
    if params.alias not in models:
        raise KeyError(
            f"Detector alias {params.alias!r} not present in models dict. Ensure "
            "make_windowed_detection_generator was invoked with the same alias "
            "passed in WindowedDetectionParams.alias."
        )
    facade = models[params.alias]

    text = str(row.get(COL_TEXT, ""))
    cap = params.max_render_chars

    # Fast path: the whole document fits in one call. Identical to the
    # pre-windowing LLMTextColumnConfig behaviour (raw detector JSON passed through).
    if len(text) <= cap:
        logger.debug("detection: single-call fast path (text=%d chars <= cap=%d)", len(text), cap)
        row[COL_RAW_DETECTED] = _call_detector(
            facade=facade, prompt=text, system_prompt=params.system_prompt, purpose="entity-detection"
        )
        return row

    # Windowed path: overlapping windows, rebase offsets, drop truncated edge
    # spans, then resolve_overlaps to dedupe overlap-region copies.
    text_len = len(text)
    window = max(_MIN_WINDOW_CHARS, cap - params.safety_margin_chars)
    windows = iter_windows(text_len, window, params.overlap_chars)
    max_workers = min(len(windows), _MAX_PARALLEL_WINDOWS)
    logger.info(
        "detection: text %d chars > cap %d; tiling into %d overlapping window(s) "
        "(window=%d, overlap=%d, min_window=%d, max_workers=%d)",
        text_len, cap, len(windows), window, params.overlap_chars, _MIN_WINDOW_CHARS, max_workers,
    )

    # Windows are independent LLM calls -> dispatch concurrently. ``future.result()``
    # re-raises the first failing window (same fail-the-row semantics as a serial
    # loop); the merge is order-independent because resolve_overlaps sorts.
    all_spans: list[EntitySpan] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _detect_window,
                facade=facade,
                text=text,
                start=start,
                end=end,
                text_len=text_len,
                system_prompt=params.system_prompt,
            )
            for start, end in windows
        ]
        for future in futures:
            all_spans.extend(future.result())

    merged = resolve_overlaps(all_spans)
    logger.info(
        "detection: %d window(s) over %d chars -> %d unique span(s) after merge",
        len(windows), text_len, len(merged),
    )
    row[COL_RAW_DETECTED] = spans_to_detector_payload(merged)
    return row


# ---------------------------------------------------------------------------
# DataDesigner wiring factory.
# ---------------------------------------------------------------------------


def make_windowed_detection_generator(alias: str) -> Any:
    """Build a ``@custom_column_generator``-decorated function bound to ``alias``.

    ``model_aliases`` must be declared statically so DataDesigner materialises the
    detector facade. The only required column is the raw text.
    """
    if not alias:
        raise ValueError("Cannot build windowed detection generator: alias is empty.")

    @custom_column_generator(
        required_columns=[COL_TEXT],
        model_aliases=[alias],
    )
    def windowed_detect(
        row: dict[str, Any],
        generator_params: WindowedDetectionParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return detect_row(row, generator_params, models)

    return windowed_detect
