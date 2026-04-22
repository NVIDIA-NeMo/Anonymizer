# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunked LLM validation for the entity detection pipeline.

Validation is the step that asks an LLM to keep/reclass/drop each candidate
entity produced by the detector. Historically we ran one LLM call per row over
the full tagged text. For long documents with many candidates this hits
context-window and provider rate limits. This module replaces that single
call with a fan-out: partition the row's candidates into chunks, build a
small tagged excerpt of text around each chunk, render the existing
validation prompt per chunk, and dispatch each chunk to an alias selected
round-robin from a configured validator pool.

Failure contract. Each chunk attempts its round-robin-assigned alias first
and, on any terminal exception from that alias (5xx after transport retries,
connection errors, parser errors), fails over sequentially to the rest of
the pool. A chunk only fails when every pool member has raised. The first
failing chunk re-raises out of the custom column generator, at which point
DataDesigner drops the row from the output DataFrame and
``NddAdapter._detect_missing_records`` surfaces it as a ``FailedRecord``.
This is the "best effort on first pass, identify and reprocess misses"
contract: raw text never silently leaks through as unscrubbed output.

The per-chunk decisions are merged into a ``ValidationDecisionsSchema``-shaped
payload so the downstream ``enrich_validation_decisions`` column keeps
working unchanged.

The public entry point for DataDesigner wiring is
:func:`make_chunked_validation_generator`, which produces a generator function
decorated with ``@custom_column_generator`` and bound to a concrete pool.
The pure helpers below are exposed for unit testing.

Per-instance validation is intentional. Every candidate goes to the LLM in
its own excerpt, even when the same ``(value, label)`` pair appears many
times in a row. Deduping by ``(value, label)`` and broadcasting one
canonical decision to every duplicate was considered (it is the single
largest cost lever on this path) and rejected.

The concern is that the same surface form + label can carry different
correctness across a document. Short names that collide with common
English words are the canonical case: the detector tags ``"Ira"`` as
``first_name`` in both "Ira said hello" (true positive) and "he grew
irate" (false positive from a substring span), producing identical
``(value, label)`` keys. Same for ``Mark`` / ``mark``, ``Bill`` /
``bill``, ``Hope`` / ``hope``, ``Art`` / ``art``. The validator is
precisely the component whose job is to resolve these; deduping erases
the question before the validator can answer it, and we have no signal
when it happens. See the "Validator chunking" section of
``docs/concepts/detection.md`` and the C5 review thread in PR #126 for
the full reasoning.

If cost becomes pressing before we have data on how often duplicates
are genuinely semantically equivalent, prefer orthogonal levers:
tighter excerpt windows, larger ``max_entities_per_call`` for cheap
validators, or token-budget-aware chunking (tracked as C7 in the same
review thread).

Concurrency model. ``chunked_validate_row`` dispatches its chunks through a
``ThreadPoolExecutor``. Per-alias concurrency is already enforced downstream
by each facade's ``ThrottledModelClient`` (AIMD on 429), so we intentionally
do not impose a row-level cap here; the pool exists purely to overlap the
chunks for this single row. Under DataDesigner's opt-in async engine the
sync ``facade.generate()`` calls we make are transparently bridged to
``agenerate`` by the DD runtime, so this code path is agnostic to which
engine is active.

TODO(async-native): once DataDesigner's async engine becomes the default
(tracking: DATA_DESIGNER_ASYNC_ENGINE opt-in flag flips off), drop the
``ThreadPoolExecutor`` + sync ``facade.generate()`` pattern here in favour
of ``async def`` functions calling ``facade.agenerate()`` directly, with
``asyncio.gather`` replacing the executor. That removes one thread hop per
chunk and lets DD's scheduler see per-chunk dispatch as first-class async
work. See the PR #126 review thread (step 2 of Andre's async
simplification suggestion) for context.

Eager prompt construction. ``chunked_validate_row`` builds the excerpt,
skeleton, and rendered prompt for every chunk before submitting any worker.
Peak prompt memory is ``len(chunks) * (2 * excerpt_window_chars + skeleton
+ ~3KB template overhead)``, i.e. low-MB even at 1000 entities with
``max_entities_per_call`` in the single digits. At workloads where this
becomes observable (very small ``max_entities_per_call`` on multi-thousand-
entity rows, or large ``excerpt_window_chars``), the fix is to move prompt
construction inside each worker and pair it with a row-level concurrency
cap (otherwise all workers race the construction phase in parallel and the
bound is unchanged). Deferred because we have no evidence of memory
pressure at realistic scale; tracked as C6 in the PR #126 review thread.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from jinja2 import BaseLoader, Environment, StrictUndefined
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_SEED_ENTITIES,
    COL_SEED_TAGGED_TEXT,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TEXT,
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
# placeholders (``_seed_tagged_text``, ``_validation_skeleton``,
# ``_tag_notation``) but with per-chunk values.
_PROMPT_ENV = Environment(
    loader=BaseLoader(),
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


@functools.lru_cache(maxsize=4)
def _compile_template(template: str) -> Any:
    """Return a compiled Jinja2 template, cached by source string.

    A row with ``N`` chunks would otherwise re-parse the same template ``N``
    times, which is wasteful for the exact workload this module targets
    (high-entity-count rows). ``maxsize=4`` is deliberately tiny: in practice
    there is one prompt string per ``EntityDetectionWorkflow`` instance, but
    tests may instantiate a handful of distinct templates in a single
    process so we keep a small LRU rather than an unbounded cache.
    """
    return _PROMPT_ENV.from_string(template)


class ChunkedValidationParams(BaseModel):
    """Parameters supplied to :func:`chunked_validate_row` via DD's ``generator_params``.

    Attributes:
        pool: Ordered list of validator model aliases. Chunk ``i`` is dispatched
            to ``pool[i % len(pool)]`` as its primary; on any terminal exception
            from that alias the chunk fails over through the rest of the pool
            (starting from the next position, wrapping around). Must be
            non-empty and every alias must also be present in the decorator's
            ``model_aliases`` so DataDesigner materialises the facade.
        max_entities_per_call: Upper bound on candidates per chunk.
        excerpt_window_chars: Chars of surrounding raw text included in each
            chunk's excerpt on either side of the chunk span.
        prompt_template: Jinja2 source for the validation prompt (with
            ``_seed_tagged_text``, ``_validation_skeleton``, ``_tag_notation``
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
    ordered: Sequence[tuple[Any, EntitySpan]],
    max_entities_per_call: int,
) -> list[list[tuple[Any, EntitySpan]]]:
    """Partition the ordered (candidate, seed) pairs into chunks of at most ``max_entities_per_call``.

    Assumes ``max_entities_per_call > 0``; positivity is enforced upstream at
    ``ChunkedValidationParams.max_entities_per_call`` and
    ``AnonymizerDetectConfig.validation_max_entities_per_call`` (both
    ``Field(gt=0)``).
    """
    return [list(ordered[i : i + max_entities_per_call]) for i in range(0, len(ordered), max_entities_per_call)]


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
    compiled = _compile_template(template)
    return compiled.render(
        **{
            COL_SEED_TAGGED_TEXT: excerpt,
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
# Chunk dispatch. Testable by passing fake ``models``.
# ---------------------------------------------------------------------------


def _dispatch_chunk(
    *,
    facades: list[tuple[str, Any]],
    prompt: str,
    system_prompt: str | None,
    chunk_index: int,
) -> RawValidationDecisionsSchema:
    """Dispatch a single chunk with cross-alias failover across the pool.

    ``facades`` is an ordered list of ``(alias, facade)`` pairs. The first
    entry is the chunk's round-robin-assigned primary; subsequent entries
    are the rest of the pool, tried in order on any terminal exception from
    the primary. Each facade carries its own transport-level retry policy
    (``RetryConfig.max_retries`` + exponential backoff on 5xx and connection
    errors) and its own AIMD throttling on 429, so by the time an exception
    escapes the facade call we consider that alias exhausted for this chunk.

    We use ``PydanticResponseRecipe`` so the facade appends JSON task
    instructions and parses the response into ``RawValidationDecisionsSchema``.

    Single-alias pools run the loop exactly once and re-raise the original
    exception (no alternate alias to try). Multi-alias pools get
    ``len(pool)`` total attempts. If every pool member raises, the *last*
    exception propagates so DataDesigner records the row as a
    ``FailedRecord`` via ``NddAdapter._detect_missing_records``.

    Each failover attempt is logged at WARNING so operators can correlate
    degraded pool members with run-level failure-rate spikes.
    """
    recipe = PydanticResponseRecipe(data_type=RawValidationDecisionsSchema)
    final_prompt = recipe.apply_recipe_to_user_prompt(prompt)
    final_system = recipe.apply_recipe_to_system_prompt(system_prompt)

    last_exc: BaseException | None = None
    for attempt_index, (alias, facade) in enumerate(facades):
        try:
            output, _messages = facade.generate(
                prompt=final_prompt,
                parser=recipe.parse,
                system_prompt=final_system,
                purpose=f"entity-validation-chunk-{chunk_index}-attempt-{attempt_index}",
            )
            if attempt_index > 0:
                logger.info(
                    "validator chunk %d: recovered on failover alias=%s (attempt %d of %d)",
                    chunk_index,
                    alias,
                    attempt_index + 1,
                    len(facades),
                )
            return output
        except Exception as exc:  # noqa: BLE001 — we classify by failover position, not type
            last_exc = exc
            remaining = len(facades) - attempt_index - 1
            if remaining > 0:
                logger.warning(
                    "validator chunk %d: alias=%s raised %s (%s); failing over to next pool member (%d remaining)",
                    chunk_index,
                    alias,
                    type(exc).__name__,
                    exc,
                    remaining,
                )
            else:
                logger.error(
                    "validator chunk %d: alias=%s raised %s (%s); pool exhausted — row will be dropped",
                    chunk_index,
                    alias,
                    type(exc).__name__,
                    exc,
                )

    # ``facades`` is non-empty by caller contract: ``chunked_validate_row``
    # builds it from the configured pool, and the config validator requires
    # a non-empty pool. After the loop, ``last_exc`` is therefore set and
    # we re-raise it. The ``None`` branch exists only to give a loud,
    # named error if that precondition is ever violated (rather than
    # ``raise None``, which would surface as ``TypeError: exceptions must
    # derive from BaseException``) and to keep the guard live under
    # ``python -O``, which strips ``assert``.
    if last_exc is None:
        raise RuntimeError(
            "_dispatch_chunk was called with an empty facades list; "
            "this violates the caller contract (a non-empty validator pool)."
        )
    raise last_exc


def chunked_validate_row(
    row: dict[str, Any],
    params: ChunkedValidationParams,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Run chunked validation for a single row and write ``COL_VALIDATION_DECISIONS``.

    This is the workhorse. Call it directly in tests with fake ``models``;
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
    candidates = ValidationCandidatesSchema.from_raw(row.get(COL_SEED_VALIDATION_CANDIDATES, {}))
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

    dispatch_kwargs_per_chunk: list[dict[str, Any]] = []
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
        # ``enumerate`` so it's non-negative by construction. The rotated
        # order (primary first, then the rest of the pool) is what
        # ``_dispatch_chunk`` walks on cross-alias failover.
        start = chunk_index % len(params.pool)
        rotated_aliases = [params.pool[(start + offset) % len(params.pool)] for offset in range(len(params.pool))]
        chunk_facades = [(alias, models[alias]) for alias in rotated_aliases]
        dispatch_kwargs_per_chunk.append(
            {
                "facades": chunk_facades,
                "prompt": prompt,
                "system_prompt": params.system_prompt,
                "chunk_index": chunk_index,
            }
        )

    # Dispatch all chunks concurrently via a ThreadPoolExecutor. Per-alias
    # concurrency is still capped downstream by each facade's
    # ``ThrottledModelClient`` (AIMD on 429), so the pool's only job here is
    # to overlap one row's chunks. ``f.result()`` re-raises the first chunk
    # exception, which is what we want: a single terminal chunk failure
    # fails the row. Pending workers finish naturally as the ``with`` block
    # exits — we just stop observing their results once we re-raise.
    if not chunks:
        chunk_results: list[RawValidationDecisionsSchema] = []
    else:
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(_dispatch_chunk, **kwargs) for kwargs in dispatch_kwargs_per_chunk]
            chunk_results = [f.result() for f in futures]

    row[COL_VALIDATION_DECISIONS] = merge_chunk_decisions(chunk_results, candidates)
    return row


# ---------------------------------------------------------------------------
# DataDesigner wiring factory.
# ---------------------------------------------------------------------------


def make_chunked_validation_generator(pool: list[str]) -> Any:
    """Build a ``@custom_column_generator``-decorated function bound to ``pool``.

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
            COL_SEED_VALIDATION_CANDIDATES,
            COL_TAG_NOTATION,
        ],
        model_aliases=list(pool),
    )
    def chunked_validate(
        row: dict[str, Any],
        generator_params: ChunkedValidationParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return chunked_validate_row(row, generator_params, models)

    return chunked_validate
