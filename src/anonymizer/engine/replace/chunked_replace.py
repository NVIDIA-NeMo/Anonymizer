# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Long-context (chunked) replacement-map generation for Substitute.

Substitute normally builds the per-record replacement map in a single LLM call
that embeds the whole tagged document. For documents whose rendered prompt would
exceed the render cap, this module instead processes the document in
boundary-aligned chunks, carrying context forward so replacements stay consistent
across chunks:

For chunk *k* the model is given
  (1) a rolling LLM **summary** of chunks 1..k-1,
  (2) the **already-generated replacement map** (to reuse, never re-map), and
  (3) the **new entities detected within chunk k**,
and asked to produce replacements only for the new entities. After the chunk, the
rolling summary is refreshed with chunk *k* via a second LLM call.

Rendering the prompts here (rather than via an ``LLMStructuredColumnConfig``)
also sidesteps NDD's ginja per-render length cap, like chunked detection.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe, TextResponseRecipe
from jinja2 import BaseLoader, Environment, StrictUndefined
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_ENTITIES_FOR_REPLACE,
    COL_ENTITY_EXAMPLES,
    COL_FINAL_ENTITIES,
    COL_REPLACEMENT_MAP,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.postprocess import EntitySpan, TagNotation, build_tagged_text
from anonymizer.engine.schemas import EntitiesSchema, EntityReplacementMapSchema
from anonymizer.engine.windowing import DEFAULT_DELIMITER, iter_boundary_windows

logger = logging.getLogger("anonymizer.replace.chunked")

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


# --- prompts owned by the chunked path (the single-call prompt is passed in) ---

_CHUNK_MAP_PROMPT = """Generate synthetic replacements for sensitive entities in ONE section of a longer document.
Output ONE replacement per NEW entity listed below. Replacements must:
- prevent re-identification, stay plausible in context, match grammatical role and class label
- NOT be a synonym/near-synonym of the original; shift to a distinct but plausible value
- keep related entities mutually consistent (geographic, personal name/email, org/domain, temporal, contact)
- preserve format/patterns and wildcards (* % ?); never return original unchanged

Summary of earlier sections (for context and consistency):
{{ summary }}

Already-generated replacements from earlier sections — REUSE these EXACTLY if the same entity
recurs here, and keep new replacements consistent with them (do NOT restate them in your output):
{{ existing_map }}

Current section (entities tagged inline):
{{ chunk_tagged_text }}

NEW entities to replace in this section:
{%- for entity in chunk_entities %}
- "{{ entity.value }}" ({{ entity.labels_str }})
{%- endfor %}

Examples: {{ examples }}

Return replacements ONLY for the NEW entities listed above.
"""

_SUMMARY_PROMPT = """You maintain a concise running summary of a long document that is being anonymized.
Update the summary to incorporate the NEW section, keeping only facts useful for keeping entity
replacements consistent across sections (who/what/where, relationships between entities, ongoing
context). Be terse; hard limit {{ summary_max_chars }} characters. Return only the updated summary.

Previous summary:
{{ prev_summary }}

New section:
{{ chunk_text }}
"""


class WindowedReplaceParams(BaseModel):
    """Params for chunked Substitute map generation (via DD ``generator_params``)."""

    alias: str = Field(min_length=1)
    single_call_prompt_template: str = Field(repr=False)
    max_render_chars: int = Field(gt=0)
    safety_margin_chars: int = Field(default=8000, ge=0)
    summary_max_chars: int = Field(default=2000, gt=0)
    delimiter: str = Field(default=DEFAULT_DELIMITER)
    system_prompt: str | None = Field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _parse_spans(raw: object) -> list[EntitySpan]:
    parsed = EntitiesSchema.from_raw(raw)
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


def chunk_tagged_text(text: str, spans: list[EntitySpan], start: int, end: int, notation: TagNotation) -> str:
    """Build the inline-tagged text for the window ``[start, end)`` (spans re-based)."""
    window_raw = text[start:end]
    in_window = [
        EntitySpan(
            entity_id=s.entity_id,
            value=s.value,
            label=s.label,
            start_position=s.start_position - start,
            end_position=s.end_position - start,
            score=s.score,
            source=s.source,
        )
        for s in spans
        if s.start_position >= start and s.end_position <= end
    ]
    return build_tagged_text(window_raw, in_window, notation=notation)


def new_chunk_entities(
    spans: list[EntitySpan],
    start: int,
    end: int,
    already_mapped: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Distinct (value, label) entities whose span starts in ``[start, end)`` and aren't mapped yet.

    Returned shape matches the prompt's ``chunk_entities`` loop (value, labels, labels_str).
    """
    by_value: dict[str, set[str]] = {}
    for span in spans:
        if not (start <= span.start_position < end):
            continue
        if not span.value or not span.label:
            continue
        if (span.value, span.label) in already_mapped:
            continue
        by_value.setdefault(span.value, set()).add(span.label)
    out: list[dict[str, Any]] = []
    for value in sorted(by_value):
        labels = sorted(by_value[value])
        out.append({"value": value, "labels": labels, "labels_str": ", ".join(labels)})
    return out


def merge_replacements(existing: list[dict[str, str]], new: EntityReplacementMapSchema) -> list[dict[str, str]]:
    """Append new replacements to ``existing``, deduping by (original, label); earlier wins."""
    seen = {(r["original"], r["label"]) for r in existing}
    merged = list(existing)
    for r in new.replacements:
        key = (r.original, r.label)
        if key in seen:
            continue
        seen.add(key)
        merged.append({"original": r.original, "label": r.label, "synthetic": r.synthetic})
    return merged


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _generate_chunk_map(
    *,
    facade: Any,
    chunk_tagged: str,
    chunk_entities: list[dict[str, Any]],
    existing_map: list[dict[str, str]],
    summary: str,
    examples: str,
    system_prompt: str | None,
    purpose: str,
) -> EntityReplacementMapSchema:
    recipe = PydanticResponseRecipe(data_type=EntityReplacementMapSchema)
    rendered = _compile_template(_CHUNK_MAP_PROMPT).render(
        summary=summary or "(none yet)",
        existing_map=json.dumps(existing_map) if existing_map else "(none yet)",
        chunk_tagged_text=chunk_tagged,
        chunk_entities=chunk_entities,
        examples=examples or "{}",
    )
    output, _ = facade.generate(
        prompt=recipe.apply_recipe_to_user_prompt(rendered),
        parser=recipe.parse,
        system_prompt=recipe.apply_recipe_to_system_prompt(system_prompt),
        purpose=purpose,
    )
    return output


def _update_summary(
    *,
    facade: Any,
    prev_summary: str,
    chunk_text: str,
    summary_max_chars: int,
    system_prompt: str | None,
    purpose: str,
) -> str:
    recipe = TextResponseRecipe()
    rendered = _compile_template(_SUMMARY_PROMPT).render(
        prev_summary=prev_summary or "(none yet)",
        chunk_text=chunk_text,
        summary_max_chars=summary_max_chars,
    )
    output, _ = facade.generate(
        prompt=recipe.apply_recipe_to_user_prompt(rendered),
        parser=recipe.parse,
        system_prompt=recipe.apply_recipe_to_system_prompt(system_prompt),
        purpose=purpose,
    )
    return str(output)[:summary_max_chars]


def generate_replacement_map_row(
    row: dict[str, Any],
    params: WindowedReplaceParams,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Build ``COL_REPLACEMENT_MAP`` for one row, chunking long documents with context carry-over."""
    if params.alias not in models:
        raise KeyError(
            f"Replacement alias {params.alias!r} not present in models dict. Ensure "
            "make_windowed_replace_generator was invoked with the same alias."
        )
    facade = models[params.alias]
    cap = params.max_render_chars
    initial_window = max(_MIN_WINDOW_CHARS, cap - params.safety_margin_chars)

    # Fast path: the single-call prompt (full tagged doc + all entities) fits under the cap.
    single_rendered = _compile_template(params.single_call_prompt_template).render(**row)
    if len(single_rendered) <= cap:
        logger.debug("replace-map: single-call fast path (rendered=%d chars <= cap=%d)", len(single_rendered), cap)
        recipe = PydanticResponseRecipe(data_type=EntityReplacementMapSchema)
        output, _ = facade.generate(
            prompt=recipe.apply_recipe_to_user_prompt(single_rendered),
            parser=recipe.parse,
            system_prompt=recipe.apply_recipe_to_system_prompt(params.system_prompt),
            purpose="replace-map-generation",
        )
        row[COL_REPLACEMENT_MAP] = output.model_dump(mode="json")
        return row

    # Chunked path with rolling summary + carried map.
    text = str(row.get(COL_TEXT, ""))
    notation = TagNotation(str(row.get(COL_TAG_NOTATION) or TagNotation.sentinel.value))
    examples = str(row.get(COL_ENTITY_EXAMPLES) or "{}")
    spans = _parse_spans(row.get(COL_FINAL_ENTITIES, {}))

    windows = iter_boundary_windows(text, initial_window, delimiter=params.delimiter)
    logger.info(
        "replace-map: rendered prompt %d chars > cap %d; chunking %d-char document into %d boundary "
        "window(s) (initial_window=%d, delimiter=%r, summary_max=%d)",
        len(single_rendered),
        cap,
        len(text),
        len(windows),
        initial_window,
        params.delimiter,
        params.summary_max_chars,
    )
    accumulated: list[dict[str, str]] = []
    summary = ""
    for i, (start, end) in enumerate(windows):
        already = {(r["original"], r["label"]) for r in accumulated}
        chunk_entities = new_chunk_entities(spans, start, end, already)
        logger.debug(
            "replace-map window %d/%d: chars [%d, %d) size=%d, %d new entit(y/ies), %d already mapped",
            i + 1,
            len(windows),
            start,
            end,
            end - start,
            len(chunk_entities),
            len(already),
        )
        if chunk_entities:
            tagged = chunk_tagged_text(text, spans, start, end, notation)
            chunk_map = _generate_chunk_map(
                facade=facade,
                chunk_tagged=tagged,
                chunk_entities=chunk_entities,
                existing_map=accumulated,
                summary=summary,
                examples=examples,
                system_prompt=params.system_prompt,
                purpose=f"replace-map-chunk-{start}",
            )
            accumulated = merge_replacements(accumulated, chunk_map)
            logger.debug(
                "replace-map window %d/%d: chunk produced %d replacement(s); accumulated map now %d entries",
                i + 1,
                len(windows),
                len(chunk_map.replacements),
                len(accumulated),
            )
        else:
            logger.debug("replace-map window %d/%d: no new entities, skipping map call", i + 1, len(windows))
        # Refresh the rolling summary for subsequent chunks (skip after the last chunk).
        if i < len(windows) - 1:
            summary = _update_summary(
                facade=facade,
                prev_summary=summary,
                chunk_text=text[start:end],
                summary_max_chars=params.summary_max_chars,
                system_prompt=params.system_prompt,
                purpose=f"replace-summary-{start}",
            )
            logger.debug(
                "replace-map window %d/%d: rolling summary updated -> %d chars: %s",
                i + 1,
                len(windows),
                len(summary),
                _clip(summary),
            )

    logger.info(
        "replace-map: %d window(s) over %d chars -> %d total replacement(s)",
        len(windows),
        len(text),
        len(accumulated),
    )
    row[COL_REPLACEMENT_MAP] = EntityReplacementMapSchema.model_validate({"replacements": accumulated}).model_dump(
        mode="json"
    )
    return row


def make_windowed_replace_generator(alias: str) -> Any:
    """Build a ``@custom_column_generator`` for chunked replacement-map generation bound to ``alias``."""
    if not alias:
        raise ValueError("Cannot build windowed replace generator: alias is empty.")

    @custom_column_generator(
        required_columns=[
            COL_TEXT,
            COL_TAGGED_TEXT,
            COL_FINAL_ENTITIES,
            COL_TAG_NOTATION,
            COL_ENTITY_EXAMPLES,
            COL_ENTITIES_FOR_REPLACE,
        ],
        model_aliases=[alias],
    )
    def windowed_replace(
        row: dict[str, Any],
        generator_params: WindowedReplaceParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return generate_replacement_map_row(row, generator_params, models)

    return windowed_replace
