# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunked LLM augmentation for the entity detection pipeline.

Split the row's text at line boundaries into chunks of at most ``chunk_tokens``
cl100k tokens, render the augmenter prompt per chunk (the chunk text becomes
``<<TAGGED_TEXT>>`` with the chunk's overlapping seed entities re-tagged into
its local coordinate system), dispatch each chunk to the configured augmenter
alias, then concatenate the per-chunk ``AugmentedEntitiesSchema`` payloads.
Entity offsets are not produced here — the downstream
``apply_augmented_entities`` post-processor locates each value in the full
original text by string search, which makes per-chunk position-tracking
unnecessary for merging.

Public entry point: :func:`make_chunked_augmentation_generator`, which
produces a ``@custom_column_generator``-decorated function bound to a single
augmenter alias. The helpers below are exposed for unit testing.

Failure contract. A chunk dispatch failure re-raises out of the generator,
DataDesigner drops the row, and ``NddAdapter._detect_missing_records``
surfaces it as a ``FailedRecord``. The single-alias contract here is
intentional: augmentation is one logical step per row; we parallelise across
the row's chunks, not across alternative aliases.

Concurrency. Chunks dispatch through a ``ThreadPoolExecutor``. Per-alias
concurrency is enforced downstream by the facade's ``ThrottledModelClient``
(AIMD on 429), so there is no row-level cap here; the pool exists purely to
overlap one row's chunks.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import tiktoken
from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from jinja2 import BaseLoader, Environment, StrictUndefined
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TEXT,
)
from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    TagNotation,
    build_tagged_text,
)
from anonymizer.engine.schemas import AugmentedEntitiesSchema, EntitiesSchema

logger = logging.getLogger("anonymizer.detection.chunked_augmentation")


_PROMPT_ENV = Environment(
    loader=BaseLoader(),
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


@functools.lru_cache(maxsize=4)
def _compile_template(template: str) -> Any:
    """Return a compiled Jinja2 template, cached by source string."""
    return _PROMPT_ENV.from_string(template)


@functools.lru_cache(maxsize=1)
def _get_encoding() -> Any:
    """cl100k_base encoder, cached across all rows of a run."""
    return tiktoken.get_encoding("cl100k_base")


class ChunkedAugmentationParams(BaseModel):
    """Parameters supplied to :func:`chunked_augment_row` via DD's ``generator_params``.

    Attributes:
        alias: The augmenter model alias. Must be present in the decorator's
            ``model_aliases`` so DataDesigner materialises the facade.
        chunk_tokens: Upper bound on cl100k tokens per chunk. The text is split
            at line boundaries (newline characters); a single line longer than
            ``chunk_tokens`` becomes its own chunk regardless.
        prompt_template: Jinja2 source for the augmenter prompt. Must contain
            ``{{_initial_tagged_text}}``, ``{{_seed_entities_json}}``, and
            ``{{_tag_notation}}`` placeholders (the same ones the single-shot
            ``LLMStructuredColumnConfig`` path consumes). Typically produced by
            ``_get_augment_prompt`` in ``detection_workflow``.
        system_prompt: Optional system prompt forwarded to each chunk call.

    ``prompt_template`` and ``system_prompt`` are ``repr=False`` to keep
    DataDesigner's setup logs readable; ``model_dump()`` still carries them.
    """

    alias: str = Field(min_length=1)
    chunk_tokens: int = Field(gt=0)
    prompt_template: str = Field(repr=False)
    system_prompt: str | None = Field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Pure helpers (no DataDesigner, no LLM). Tested directly.
# ---------------------------------------------------------------------------


def chunk_text_at_line_boundaries(text: str, max_tokens: int) -> list[tuple[int, str]]:
    """Split ``text`` into chunks of at most ``max_tokens`` cl100k tokens at line boundaries.

    Returns a list of ``(start_char_offset, chunk_text)`` pairs. ``start_char_offset``
    is the index of the chunk's first character in the original text — used to
    re-anchor overlapping seed entities into the chunk's local coordinate system
    for tagging.

    Edge cases:
      - Empty text → one chunk ``(0, "")``.
      - Total tokens already ≤ ``max_tokens`` → one chunk ``(0, text)`` (no work).
      - A single line longer than ``max_tokens`` → emitted as its own chunk
        without in-line splitting. The augmenter still gets that chunk; chunking
        is best-effort, not a hard cap on prompt size.
    """
    if not text:
        return [(0, "")]
    enc = _get_encoding()
    if len(enc.encode(text)) <= max_tokens:
        return [(0, text)]

    lines = text.splitlines(keepends=True)
    chunks: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_tokens = 0
    current_start = 0
    char_pos = 0
    for line in lines:
        line_tokens = len(enc.encode(line))
        if current_lines and current_tokens + line_tokens > max_tokens:
            chunks.append((current_start, "".join(current_lines)))
            current_lines = []
            current_tokens = 0
            current_start = char_pos
        current_lines.append(line)
        current_tokens += line_tokens
        char_pos += len(line)
    if current_lines:
        chunks.append((current_start, "".join(current_lines)))
    return chunks


def build_chunk_tagged_text(
    *,
    chunk_text: str,
    chunk_offset: int,
    all_seeds: Iterable[EntitySpan],
    notation: TagNotation,
) -> str:
    """Re-tag the chunk with the seed entities that fall inside it.

    Each seed whose ``[start_position, end_position)`` lies fully inside
    ``[chunk_offset, chunk_offset + len(chunk_text))`` is re-anchored to the
    chunk's local coordinate system (subtracting ``chunk_offset``) and rendered
    via ``build_tagged_text``. Seeds outside the chunk are dropped — they don't
    appear inline in this chunk's text, so tagging them would be a no-op /
    confusing to the augmenter.
    """
    chunk_end = chunk_offset + len(chunk_text)
    local_seeds = [
        EntitySpan(
            entity_id=s.entity_id,
            value=s.value,
            label=s.label,
            start_position=s.start_position - chunk_offset,
            end_position=s.end_position - chunk_offset,
            score=s.score,
            source=s.source,
        )
        for s in all_seeds
        if s.start_position >= chunk_offset and s.end_position <= chunk_end
    ]
    return build_tagged_text(chunk_text, local_seeds, notation=notation)


def render_chunk_prompt(
    *,
    template: str,
    chunk_tagged_text: str,
    seed_entities_json: dict[str, Any],
    notation: TagNotation,
) -> str:
    """Render the augmenter prompt for a single chunk via Jinja2.

    The template (from ``_get_augment_prompt``) references the same column-name
    locals the single-shot path consumes: ``_initial_tagged_text``,
    ``_seed_entities_json``, ``_tag_notation``. We provide them per chunk here.
    """
    compiled = _compile_template(template)
    return compiled.render(
        **{
            COL_INITIAL_TAGGED_TEXT: chunk_tagged_text,
            COL_SEED_ENTITIES_JSON: seed_entities_json,
            COL_TAG_NOTATION: notation.value,
        }
    )


def merge_chunk_outputs(chunk_outputs: list[AugmentedEntitiesSchema]) -> dict[str, Any]:
    """Concatenate per-chunk augmenter outputs into a single ``AugmentedEntitiesSchema`` payload.

    Dedupes by case-insensitive ``(value, label)`` to avoid identical suggestions
    from overlapping chunks. The downstream ``apply_augmented_entities`` step
    re-locates each value in the full original text by string search, so
    cross-chunk duplicates are largely idempotent there too — this dedupe is a
    cheap optimisation that keeps the augmenter output small and the merge
    log readable.
    """
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []
    for output in chunk_outputs:
        for entity in output.entities:
            key = (entity.value.strip().lower(), entity.label.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(entity.model_dump())
    return AugmentedEntitiesSchema.model_validate({"entities": merged}).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Chunk dispatch. Testable by passing fake ``models``.
# ---------------------------------------------------------------------------


def _dispatch_chunk(
    *,
    facade: Any,
    alias: str,
    prompt: str,
    system_prompt: str | None,
    chunk_index: int,
) -> AugmentedEntitiesSchema:
    """Dispatch a single augmenter chunk to ``facade``.

    Augmentation is a single-alias step (no validator-style failover pool):
    one facade per row, retries handled by the facade's own ``RetryConfig``
    and AIMD throttling on 429. A terminal exception escapes here so
    DataDesigner records the row as a ``FailedRecord``.
    """
    recipe = PydanticResponseRecipe(data_type=AugmentedEntitiesSchema)
    final_prompt = recipe.apply_recipe_to_user_prompt(prompt)
    final_system = recipe.apply_recipe_to_system_prompt(system_prompt)
    try:
        output, _messages = facade.generate(
            prompt=final_prompt,
            parser=recipe.parse,
            system_prompt=final_system,
            purpose=f"entity-augmentation-chunk-{chunk_index}",
        )
        return output
    except Exception as exc:
        logger.error(
            "augmenter chunk %d: alias=%s raised %s (%s); row will be dropped",
            chunk_index,
            alias,
            type(exc).__name__,
            exc,
        )
        raise


def chunked_augment_row(
    row: dict[str, Any],
    params: ChunkedAugmentationParams,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Run chunked augmentation for a single row and write ``COL_AUGMENTED_ENTITIES``.

    Direct testable entry point. The DataDesigner-decorated wrapper produced by
    :func:`make_chunked_augmentation_generator` just forwards to it.
    """
    if params.alias not in models:
        raise KeyError(
            f"Augmenter alias {params.alias!r} not present in models dict. "
            f"Ensure make_chunked_augmentation_generator was invoked with the same alias "
            f"passed in ChunkedAugmentationParams.alias."
        )

    text = str(row.get(COL_TEXT, ""))
    seed_entities_schema = EntitiesSchema.from_raw(row.get(COL_SEED_ENTITIES, {}))
    seed_entities_json = row.get(COL_SEED_ENTITIES_JSON, {})
    notation_raw = row.get(COL_TAG_NOTATION) or TagNotation.sentinel.value
    notation = TagNotation(str(notation_raw))

    all_seeds = [
        EntitySpan(
            entity_id=e.id,
            value=e.value,
            label=e.label,
            start_position=e.start_position,
            end_position=e.end_position,
            score=e.score,
            source=e.source,
        )
        for e in seed_entities_schema.entities
    ]

    chunks = chunk_text_at_line_boundaries(text, params.chunk_tokens)

    if len(chunks) == 1:
        logger.debug(
            "chunked augmentation: 1 chunk (text fits in chunk_tokens=%d)",
            params.chunk_tokens,
        )
    else:
        logger.debug(
            "chunked augmentation: %d chunks (chunk_tokens=%d), alias=%s",
            len(chunks),
            params.chunk_tokens,
            params.alias,
        )

    facade = models[params.alias]

    dispatch_kwargs_per_chunk: list[dict[str, Any]] = []
    for chunk_index, (chunk_offset, chunk_text) in enumerate(chunks):
        tagged = build_chunk_tagged_text(
            chunk_text=chunk_text,
            chunk_offset=chunk_offset,
            all_seeds=all_seeds,
            notation=notation,
        )
        prompt = render_chunk_prompt(
            template=params.prompt_template,
            chunk_tagged_text=tagged,
            seed_entities_json=seed_entities_json,
            notation=notation,
        )
        dispatch_kwargs_per_chunk.append(
            {
                "facade": facade,
                "alias": params.alias,
                "prompt": prompt,
                "system_prompt": params.system_prompt,
                "chunk_index": chunk_index,
            }
        )

    if not chunks:
        chunk_outputs: list[AugmentedEntitiesSchema] = []
    else:
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(_dispatch_chunk, **kw) for kw in dispatch_kwargs_per_chunk]
            chunk_outputs = [f.result() for f in futures]

    row[COL_AUGMENTED_ENTITIES] = merge_chunk_outputs(chunk_outputs)
    return row


# ---------------------------------------------------------------------------
# DataDesigner wiring factory.
# ---------------------------------------------------------------------------


def make_chunked_augmentation_generator(alias: str) -> Any:
    """Build a ``@custom_column_generator``-decorated function bound to ``alias``.

    Mirrors :func:`make_chunked_validation_generator`. Required columns are
    exhaustive for DataDesigner's DAG ordering: the generator reads the raw text
    (to chunk), seed entities (to re-anchor for per-chunk tagging), the seed
    entities JSON (passed verbatim to each chunk's prompt), and the tag notation.
    """
    if not alias:
        raise ValueError("Cannot build chunked augmentation generator: alias is empty.")

    @custom_column_generator(
        required_columns=[
            COL_TEXT,
            COL_SEED_ENTITIES,
            COL_SEED_ENTITIES_JSON,
            COL_TAG_NOTATION,
        ],
        model_aliases=[alias],
    )
    def chunked_augment(
        row: dict[str, Any],
        generator_params: ChunkedAugmentationParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return chunked_augment_row(row, generator_params, models)

    return chunked_augment
