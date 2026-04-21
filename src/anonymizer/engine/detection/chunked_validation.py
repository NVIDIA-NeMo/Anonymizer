# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunked LLM validation for the entity detection pipeline.

Validation is the step that asks an LLM to keep/reclass/drop each candidate
entity produced by the detector. Historically we ran one LLM call per row over
the full tagged text. For long documents with many candidates this hits
context-window and TPM/RPM limits. This module replaces that single call with
a fan-out: partition the row's candidates into chunks, build a small tagged
excerpt of text around each chunk, render the existing validation prompt per
chunk, and dispatch each chunk to an alias selected round-robin from a
configured validator pool.

The per-chunk decisions are merged into a ``ValidationDecisionsSchema``-shaped
payload so the downstream ``enrich_validation_decisions`` column keeps
working unchanged.

The public entry point for DataDesigner wiring is
:func:`make_chunked_validation_generator`, which produces a generator function
decorated with ``@custom_column_generator`` and bound to a concrete pool.
The pure helpers below are exposed for unit testing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from jinja2 import BaseLoader, Environment, StrictUndefined
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_MERGED_TAGGED_TEXT,
    COL_SEED_ENTITIES,
    COL_TAG_NOTATION,
    COL_TEXT,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_SKELETON,
)
from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    TagNotation,
    build_tagged_text,
)
from anonymizer.engine.schemas import (
    EntitiesSchema,
    RawValidationDecisionsSchema,
    ValidationCandidatesSchema,
    ValidationDecisionsSchema,
    ValidationSkeletonDecisionSchema,
    ValidationSkeletonSchema,
)

logger = logging.getLogger("anonymizer.detection.chunked_validation")

# Jinja2 environment used to render the per-chunk validation prompt.
# The template mirrors the production prompt exactly: we substitute the same
# placeholders (``_merged_tagged_text``, ``_validation_skeleton``,
# ``_tag_notation``) but with per-chunk values.
_PROMPT_ENV = Environment(
    loader=BaseLoader(),
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


class ChunkedValidationParams(BaseModel):
    """Parameters supplied to :func:`chunked_validate_row` via DD's ``generator_params``.

    Attributes:
        pool: Ordered list of validator model aliases. Chunk ``i`` is dispatched
            to ``pool[i % len(pool)]``. Must be non-empty and every alias must
            also be present in the decorator's ``model_aliases`` so DataDesigner
            materialises the facade.
        max_entities_per_call: Upper bound on candidates per chunk.
        excerpt_window_chars: Chars of surrounding raw text included in each
            chunk's excerpt on either side of the chunk span.
        prompt_template: Jinja2 source for the validation prompt (with
            ``_merged_tagged_text``, ``_validation_skeleton``, ``_tag_notation``
            placeholders). Typically produced by ``_get_validation_prompt``.
        system_prompt: Optional system prompt forwarded to each chunk call.
    """

    pool: list[str] = Field(min_length=1)
    max_entities_per_call: int = Field(gt=0)
    excerpt_window_chars: int = Field(gt=0)
    prompt_template: str
    system_prompt: str | None = None


# ---------------------------------------------------------------------------
# Pure helpers (no DataDesigner, no LLM). Tested directly.
# ---------------------------------------------------------------------------


def order_candidates_by_position(
    candidates: ValidationCandidatesSchema,
    seed_entities: list[EntitySpan],
) -> list[tuple[Any, EntitySpan]]:
    """Pair each candidate with its matching seed entity and sort by text position.

    Every candidate id must resolve to a seed entity. A missing id indicates an
    upstream bug in ``merge_and_build_candidates`` or ``prepare_validation_inputs``
    (both produce candidates whose ids come from ``EntitySpan.entity_id``). We
    raise early with the offending id so the failure is easy to triage.
    """
    seed_by_id = {span.entity_id: span for span in seed_entities}
    paired: list[tuple[Any, EntitySpan]] = []
    for candidate in candidates.candidates:
        seed = seed_by_id.get(candidate.id)
        if seed is None:
            raise ValueError(
                f"Validation candidate id {candidate.id!r} has no matching seed entity. "
                "Every candidate produced by merge_and_build_candidates or "
                "prepare_validation_inputs must correspond to a seed entity with "
                "start_position and end_position populated; this inconsistency "
                "indicates a bug in one of those upstream generators."
            )
        paired.append((candidate, seed))
    paired.sort(key=lambda pair: (pair[1].start_position, pair[1].end_position, pair[1].entity_id))
    return paired


def chunk_candidates(
    ordered: list[tuple[Any, EntitySpan]],
    max_entities_per_call: int,
) -> list[list[tuple[Any, EntitySpan]]]:
    """Partition the ordered (candidate, seed) pairs into chunks of at most ``max_entities_per_call``."""
    if max_entities_per_call <= 0:
        raise ValueError(f"max_entities_per_call must be > 0, got {max_entities_per_call}.")
    return [ordered[i : i + max_entities_per_call] for i in range(0, len(ordered), max_entities_per_call)]


def build_chunk_excerpt(
    *,
    text: str,
    chunk_spans: list[EntitySpan],
    all_spans: list[EntitySpan],
    window_chars: int,
    notation: TagNotation,
) -> str:
    """Build a tagged text excerpt wide enough to give the LLM context around ``chunk_spans``.

    The excerpt spans ``[min(chunk.start) - window, max(chunk.end) + window]``
    clamped to the text bounds. Any entity from ``all_spans`` fully contained
    in that window is re-tagged inside the excerpt so the surrounding context
    matches the full-document view. The forced ``notation`` keeps tags stable
    across chunks of the same row even when a local slice would otherwise
    pick a different heuristic.
    """
    if not chunk_spans:
        return ""
    chunk_start = min(span.start_position for span in chunk_spans)
    chunk_end = max(span.end_position for span in chunk_spans)
    excerpt_start = max(0, chunk_start - window_chars)
    excerpt_end = min(len(text), chunk_end + window_chars)
    excerpt_raw = text[excerpt_start:excerpt_end]
    in_window = [
        EntitySpan(
            entity_id=span.entity_id,
            value=span.value,
            label=span.label,
            start_position=span.start_position - excerpt_start,
            end_position=span.end_position - excerpt_start,
            score=span.score,
            source=span.source,
        )
        for span in all_spans
        if span.start_position >= excerpt_start and span.end_position <= excerpt_end
    ]
    return build_tagged_text(excerpt_raw, in_window, notation=notation)


def build_chunk_skeleton(chunk_candidates_: list[Any]) -> dict[str, Any]:
    """Build the validation skeleton (``ValidationSkeletonSchema``) for a chunk."""
    skeleton = ValidationSkeletonSchema(
        decisions=[ValidationSkeletonDecisionSchema(id=c.id, value=c.value, label=c.label) for c in chunk_candidates_]
    )
    return skeleton.model_dump(mode="json")


def render_chunk_prompt(
    *,
    template: str,
    excerpt: str,
    skeleton: dict[str, Any],
    notation: TagNotation,
) -> str:
    """Render the validation prompt for a single chunk via Jinja2.

    The template and context match the production ``LLMStructuredColumnConfig``
    call: dicts are rendered with Python ``str()`` (Jinja2 default), which is
    how the existing prompt has always served ``{{ _validation_skeleton }}``.
    """
    compiled = _PROMPT_ENV.from_string(template)
    return compiled.render(
        **{
            COL_MERGED_TAGGED_TEXT: excerpt,
            COL_VALIDATION_SKELETON: skeleton,
            COL_TAG_NOTATION: notation.value,
        }
    )


def merge_chunk_decisions(
    chunk_results: list[RawValidationDecisionsSchema],
    candidates: ValidationCandidatesSchema,
) -> dict[str, Any]:
    """Flatten chunk decisions into a single ``ValidationDecisionsSchema`` payload.

    Mirrors the single-call contract:
    - Only decisions whose ids match a known candidate are retained. This is
      consistent with ``enrich_validation_decisions``, which also filters to
      valid ids; doing it here too keeps COL_VALIDATION_DECISIONS minimal.
    - Duplicate ids across chunks keep the first occurrence. Duplicates
      shouldn't happen because candidates partition cleanly, but deduping is
      cheap defence-in-depth.
    - Candidates with no decision at all flow through as absent; downstream
      ``apply_validation_decisions`` treats them as ``keep`` with the original
      label, which matches prior behaviour for a partially-answered response.
    """
    candidate_lookup = {c.id: c for c in candidates.candidates}
    valid_ids = set(candidate_lookup)
    seen: set[str] = set()
    merged_decisions: list[dict[str, Any]] = []
    for result in chunk_results:
        for decision in result.decisions:
            if decision.id not in valid_ids or decision.id in seen:
                continue
            cand = candidate_lookup[decision.id]
            merged_decisions.append(
                {
                    "id": decision.id,
                    "value": cand.value,
                    "label": cand.label,
                    "decision": decision.decision.value if decision.decision is not None else None,
                    "proposed_label": decision.proposed_label or "",
                    "reason": decision.reason,
                }
            )
            seen.add(decision.id)
    # Validate the final shape so malformed output fails loud here rather than in a
    # downstream parser. ``ValidationDecisionsSchema.decision`` is non-optional, so
    # we drop rows where the LLM didn't supply a decision; the downstream
    # "no decision -> keep" semantics rely on candidate-not-in-output, not on a
    # null-decision entry.
    filtered = [d for d in merged_decisions if d["decision"] is not None]
    return ValidationDecisionsSchema.model_validate({"decisions": filtered}).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Async dispatch. Testable by passing fake ``models``.
# ---------------------------------------------------------------------------


async def _dispatch_chunk(
    *,
    facade: Any,
    prompt: str,
    system_prompt: str | None,
    chunk_index: int,
) -> RawValidationDecisionsSchema:
    """Dispatch a single chunk via the configured ModelFacade.

    We use ``PydanticResponseRecipe`` so the facade appends JSON task
    instructions and parses the response into ``RawValidationDecisionsSchema``.
    Parser failures and transient LLM errors are handled by
    ``ModelFacade.agenerate`` (throttle, retries); any error that escapes is
    terminal for the chunk. We let it propagate so the row fails as a whole
    and DataDesigner records it as a ``FailedRecord``.
    """
    recipe = PydanticResponseRecipe(data_type=RawValidationDecisionsSchema)
    final_prompt = recipe.apply_recipe_to_user_prompt(prompt)
    final_system = recipe.apply_recipe_to_system_prompt(system_prompt)
    output, _messages = await facade.agenerate(
        prompt=final_prompt,
        parser=recipe.parse,
        system_prompt=final_system,
        purpose=f"entity-validation-chunk-{chunk_index}",
    )
    return output


async def chunked_validate_row(
    row: dict[str, Any],
    params: ChunkedValidationParams,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Run chunked validation for a single row and write ``COL_VALIDATION_DECISIONS``.

    This is the async workhorse. Call it directly in tests with fake ``models``;
    the DataDesigner-decorated wrapper produced by
    :func:`make_chunked_validation_generator` just forwards to it.
    """
    missing_aliases = [alias for alias in params.pool if alias not in models]
    if missing_aliases:
        raise KeyError(
            f"Validator pool aliases {missing_aliases} not present in models dict. "
            f"Ensure make_chunked_validation_generator was invoked with the same pool "
            f"passed in ChunkedValidationParams.pool."
        )

    text = str(row.get(COL_TEXT, ""))
    candidates = ValidationCandidatesSchema.from_raw(row.get(COL_VALIDATION_CANDIDATES, {}))
    seed_entities_schema = EntitiesSchema.from_raw(row.get(COL_SEED_ENTITIES, {}))
    notation_raw = row.get(COL_TAG_NOTATION) or TagNotation.sentinel.value
    notation = TagNotation(str(notation_raw))

    # Short-circuit: a row with no candidates has no decisions to make.
    if not candidates.candidates:
        row[COL_VALIDATION_DECISIONS] = ValidationDecisionsSchema().model_dump(mode="json")
        return row

    all_spans = [
        EntitySpan(
            entity_id=entity.id,
            value=entity.value,
            label=entity.label,
            start_position=entity.start_position,
            end_position=entity.end_position,
            score=entity.score,
            source=entity.source,
        )
        for entity in seed_entities_schema.entities
    ]

    ordered = order_candidates_by_position(candidates, all_spans)
    chunks = chunk_candidates(ordered, params.max_entities_per_call)

    tasks: list[Any] = []
    for chunk_index, chunk in enumerate(chunks):
        chunk_candidates_ = [pair[0] for pair in chunk]
        chunk_spans = [pair[1] for pair in chunk]
        excerpt = build_chunk_excerpt(
            text=text,
            chunk_spans=chunk_spans,
            all_spans=all_spans,
            window_chars=params.excerpt_window_chars,
            notation=notation,
        )
        skeleton = build_chunk_skeleton(chunk_candidates_)
        prompt = render_chunk_prompt(
            template=params.prompt_template,
            excerpt=excerpt,
            skeleton=skeleton,
            notation=notation,
        )
        # Round-robin across the validator pool. ``ChunkedValidationParams``
        # guarantees ``pool`` is non-empty; ``chunk_index`` comes from
        # ``enumerate`` so it's non-negative by construction.
        alias = params.pool[chunk_index % len(params.pool)]
        facade = models[alias]
        tasks.append(
            _dispatch_chunk(
                facade=facade,
                prompt=prompt,
                system_prompt=params.system_prompt,
                chunk_index=chunk_index,
            )
        )

    # gather() propagates the first exception, cancelling siblings. That's the
    # all-or-nothing row contract: a single terminal chunk failure fails the row.
    chunk_results = await asyncio.gather(*tasks)

    row[COL_VALIDATION_DECISIONS] = merge_chunk_decisions(chunk_results, candidates)
    return row


# ---------------------------------------------------------------------------
# DataDesigner wiring factory.
# ---------------------------------------------------------------------------


def make_chunked_validation_generator(pool: list[str]) -> Any:
    """Build a ``@custom_column_generator``-decorated async function bound to ``pool``.

    ``model_aliases`` must be declared statically on the decorator so
    DataDesigner knows which facades to materialise for the generator. Since
    the pool is config-driven (per-run), we generate the function dynamically.
    The required_columns are exhaustive for DataDesigner's DAG ordering: the
    generator reads the raw text, seed entities (for positions), the candidate
    list (what to decide), and the tag notation (for excerpt tagging).
    """
    if not pool:
        raise ValueError("Cannot build chunked validation generator: pool is empty.")

    @custom_column_generator(
        required_columns=[
            COL_TEXT,
            COL_SEED_ENTITIES,
            COL_VALIDATION_CANDIDATES,
            COL_TAG_NOTATION,
        ],
        model_aliases=list(pool),
    )
    async def chunked_validate(
        row: dict[str, Any],
        generator_params: ChunkedValidationParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return await chunked_validate_row(row, generator_params, models)

    return chunked_validate
