# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunked LLM validation for the entity detection pipeline.

Partition a row's validation candidates into chunks, build a small tagged
excerpt around each chunk, render the validation prompt per chunk, and
dispatch each chunk to an alias selected round-robin from a configured
validator pool. Each response is bound to its request inventory and written to
``COL_VALIDATION_INVENTORIES`` and ``COL_VALIDATION_DECISIONS`` for admission.

Public entry point: :func:`make_chunked_validation_generator`, which
produces a ``@custom_column_generator``-decorated function bound to a
concrete pool. Plugin columns call :func:`chunked_validate_row_async`
directly. The helpers below are exposed for unit testing.

Failure contract. Each chunk attempts its round-robin primary first and
fails over sequentially to the rest of the pool; a chunk only fails when
every pool member has raised. The first failing chunk re-raises out of
the generator, DataDesigner drops the row, and
``NddAdapter._detect_missing_records`` surfaces it as a ``FailedRecord``.
Raw text never silently leaks through as unscrubbed output.

Concurrency. Async plugin execution dispatches chunks with ``asyncio.gather``
and ``facade.agenerate()``. The plugin's synchronous execution path uses a
``ThreadPoolExecutor`` and ``facade.generate()``. Per-alias concurrency is
enforced downstream by DataDesigner's request-admission layer, so the optional
row-level cap only bounds local fan-out pressure.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from jinja2 import BaseLoader, Environment, StrictUndefined
from pydantic import BaseModel, Field

from anonymizer.engine.constants import (
    COL_SEED_DETECTION_RESULT,
    COL_SEED_TAGGED_TEXT,
    COL_TAG_NOTATION,
    COL_TEXT,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_INVENTORIES,
    COL_VALIDATION_SKELETON,
)
from anonymizer.engine.detection.admission import find_duplicate_assigned_aliases
from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    TagNotation,
    build_tagged_text,
    group_overlapping_spans,
)
from anonymizer.engine.schemas import (
    AdmissionRoute,
    DetectionResult,
    EffectiveLabelPolicy,
    Observation,
    RawValidationDecisionsSchema,
    SourceView,
    ValidationInventoryEntry,
    ValidationRequestDecisionSet,
    ValidationRequestDecisionSets,
    ValidationRequestInventory,
)

logger = logging.getLogger("anonymizer.detection.chunked_validation")


class _SemanticValidationResponseError(RuntimeError):
    """A parsed response violates its dispatched request inventory."""


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
    """Return a compiled Jinja2 template, cached by source string."""
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
        max_entities_per_call: Target upper bound on candidates per chunk.
            An overlapping conflict group larger than this is sent alone,
            intact, with a warning.
        excerpt_window_chars: Chars of surrounding raw text included in each
            chunk's excerpt on either side of the chunk span.
        max_parallel_chunks: Optional row-local cap on concurrently dispatched
            chunks. Plugin generators derive a default cap from validator model
            capacity.
        single_chunk_full_text: If True, a row with one validation chunk sees
            the full tagged document. If False, even a single chunk uses the
            excerpt window. The default preserves production parity with the
            pre-chunking validation path; benchmarks may disable it to probe
            compact validation prompts.
        prompt_template: Jinja2 source for the validation prompt (with
            ``_seed_tagged_text``, ``_validation_skeleton``, ``_tag_notation``
            placeholders). Typically produced by ``_get_validation_prompt``.
        system_prompt: Optional system prompt forwarded to each chunk call.

    ``prompt_template`` and ``system_prompt`` are marked ``repr=False`` because
    DataDesigner's pre-generation logger f-strings this model
    (``generator_params: {params}``) and our validation prompt is multi-kB of
    entity rules; a non-trivial system prompt would compound that. Hiding
    them from ``__str__``/``__repr__`` keeps setup logs readable without
    touching serialization — ``model_dump()`` still carries both, so the
    generator receives them unchanged.
    """

    pool: list[str] = Field(min_length=1)
    max_entities_per_call: int = Field(gt=0)
    excerpt_window_chars: int = Field(gt=0)
    max_parallel_chunks: int | None = Field(default=None, gt=0)
    single_chunk_full_text: bool = True
    prompt_template: str = Field(repr=False)
    system_prompt: str | None = Field(default=None, repr=False)
    allowed_entity_labels: list[str] | None = None
    excluded_entity_labels: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _PreparedValidationRequest:
    inventory: ValidationRequestInventory
    prompt: str


# ---------------------------------------------------------------------------
# Pure helpers (no DataDesigner, no LLM). Tested directly.
# ---------------------------------------------------------------------------


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


def build_validation_conflict_groups(
    observations: Sequence[Observation],
) -> list[tuple[Observation, ...]]:
    """Build source-ordered connected components of overlapping observations."""
    ordered = sorted(
        (observation for observation in observations if observation.admission_route is AdmissionRoute.llm_validation),
        key=lambda item: (
            item.start_position,
            item.end_position,
            item.label.strip().casefold(),
            item.detection_result_id,
            item.observation_id,
        ),
    )
    return group_overlapping_spans(
        ordered,
        start_position=lambda observation: observation.start_position,
        end_position=lambda observation: observation.end_position,
    )


def chunk_validation_conflict_groups(
    groups: Sequence[tuple[Observation, ...]],
    max_entities_per_call: int,
) -> list[list[tuple[Observation, ...]]]:
    """Pack whole conflict groups, sending an oversized group alone with a warning."""
    chunks: list[list[tuple[Observation, ...]]] = []
    current: list[tuple[Observation, ...]] = []
    current_size = 0
    for group in groups:
        group_size = len(group)
        if group_size > max_entities_per_call:
            if current:
                chunks.append(current)
                current = []
                current_size = 0
            logger.warning(
                "Validation conflict group has %d candidates, exceeding the configured target %d; "
                "sending the group alone to keep overlapping alternatives together",
                group_size,
                max_entities_per_call,
            )
            chunks.append([group])
            continue
        if current and current_size + group_size > max_entities_per_call:
            chunks.append(current)
            current = []
            current_size = 0
        current.append(group)
        current_size += group_size
    if current:
        chunks.append(current)
    return chunks


def prepare_validation_requests(
    *,
    detection_result: DetectionResult,
    canonical_text: str,
    max_entities_per_call: int,
    excerpt_window_chars: int,
    single_chunk_full_text: bool,
    notation: TagNotation,
    prompt_template: str,
    effective_label_policy: EffectiveLabelPolicy,
) -> list[_PreparedValidationRequest]:
    """Create immutable request inventories and conflict-aware prompts.

    Only observations assigned to the request are tagged. Neighboring
    observations in its source excerpt remain plain text without label hints.
    Prompt offsets are relative to the untagged excerpt; inventory offsets
    remain canonical source positions.
    """
    groups = build_validation_conflict_groups(detection_result.observations)
    chunks = chunk_validation_conflict_groups(groups, max_entities_per_call)
    prepared: list[_PreparedValidationRequest] = []
    for request_index, chunk in enumerate(chunks, start=1):
        observations = [observation for group in chunk for observation in group]
        if len(chunks) == 1 and single_chunk_full_text:
            context_start = detection_result.source_view.start_position
            context_end = detection_result.source_view.end_position
        else:
            context_start = max(
                detection_result.source_view.start_position,
                min(item.start_position for item in observations) - excerpt_window_chars,
            )
            context_end = min(
                detection_result.source_view.end_position,
                max(item.end_position for item in observations) + excerpt_window_chars,
            )

        entries: list[ValidationInventoryEntry] = []
        prompt_markers: list[EntitySpan] = []
        next_alias = 1
        next_group_alias = 1
        for group in chunk:
            conflict_group_id = None
            if len(group) > 1:
                conflict_group_id = f"g{next_group_alias}"
                next_group_alias += 1
                group_start = min(item.start_position for item in group)
                group_end = max(item.end_position for item in group)
                prompt_markers.append(
                    EntitySpan(
                        entity_id=conflict_group_id,
                        value=canonical_text[group_start:group_end],
                        label=f"NAGROUP_{conflict_group_id}",
                        start_position=group_start - context_start,
                        end_position=group_end - context_start,
                        score=0.0,
                        source="validation_prompt",
                    )
                )
            else:
                observation = group[0]
                prompt_markers.append(
                    EntitySpan(
                        entity_id=observation.observation_id,
                        value=observation.value,
                        label=observation.label,
                        start_position=observation.start_position - context_start,
                        end_position=observation.end_position - context_start,
                        score=observation.native_score or 0.0,
                        source="validation_prompt",
                    )
                )
            for observation in group:
                entries.append(
                    ValidationInventoryEntry(
                        alias=f"c{next_alias}",
                        conflict_group_id=conflict_group_id,
                        observation_ref=observation.ref,
                        value=observation.value,
                        label=observation.label,
                        start_position=observation.start_position,
                        end_position=observation.end_position,
                    )
                )
                next_alias += 1

        context_view = SourceView(
            source_text_ref=detection_result.source_view.source_text_ref,
            start_position=context_start,
            end_position=context_end,
        )
        source_context = canonical_text[context_start:context_end]
        inventory = ValidationRequestInventory(
            request_id=f"validation-request-{request_index}",
            source_text_ref=detection_result.source_view.source_text_ref,
            context_view=context_view,
            source_context=source_context,
            effective_label_policy=effective_label_policy,
            entries=tuple(entries),
        )
        tagged_context = build_tagged_text(
            source_context,
            prompt_markers,
            notation=notation,
        )
        skeleton = {
            "decisions": [
                {
                    "id": entry.alias,
                    "conflict_group_id": entry.conflict_group_id,
                    "value": entry.value,
                    "label": entry.label,
                    "start_position": entry.start_position - context_start,
                    "end_position": entry.end_position - context_start,
                    "decision": None,
                    "proposed_label": None,
                    "reason": None,
                }
                for entry in inventory.entries
            ]
        }
        prepared.append(
            _PreparedValidationRequest(
                inventory=inventory,
                prompt=render_chunk_prompt(
                    template=prompt_template,
                    excerpt=tagged_context,
                    skeleton=skeleton,
                    notation=notation,
                ),
            )
        )
    return prepared


def serialize_inventory_decisions(
    chunk_results: list[RawValidationDecisionsSchema],
    requests: list[_PreparedValidationRequest],
) -> dict[str, Any]:
    """Bind every parsed response to the immutable request that authorized it."""
    return ValidationRequestDecisionSets(
        responses=tuple(
            ValidationRequestDecisionSet(
                request_id=request.inventory.request_id,
                decisions=tuple(result.decisions),
            )
            for result, request in zip(chunk_results, requests, strict=True)
        )
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Chunk dispatch. Testable by passing fake ``models``.
# ---------------------------------------------------------------------------


def _dispatch_chunk(
    *,
    facades: list[tuple[str, Any]],
    prompt: str,
    system_prompt: str | None,
    chunk_index: int,
    assigned_aliases: tuple[str, ...],
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

    Transport failures try each pool member once. A duplicate assigned alias
    is a semantic response failure: multi-alias pools advance to the next
    alias, while a single-alias pool receives one bounded same-request retry.
    If all permitted attempts fail, the last exception propagates so
    DataDesigner records the row as a ``FailedRecord``.

    Each failover attempt is logged at WARNING so operators can correlate
    degraded pool members with run-level failure-rate spikes.
    Logs include exception types, not messages, because messages may contain
    source text or model output.
    """
    recipe = PydanticResponseRecipe(data_type=RawValidationDecisionsSchema)
    final_prompt = recipe.apply_recipe_to_user_prompt(prompt)
    final_system = recipe.apply_recipe_to_system_prompt(system_prompt)

    last_exc: BaseException | None = None
    attempt_index = 0
    for facade_index, (alias, facade) in enumerate(facades):
        semantic_attempts = 2 if len(facades) == 1 else 1
        for semantic_attempt in range(semantic_attempts):
            try:
                output, _messages = facade.generate(
                    prompt=final_prompt,
                    parser=recipe.parse,
                    system_prompt=final_system,
                    purpose=f"entity-validation-chunk-{chunk_index}-attempt-{attempt_index}",
                )
            except Exception as exc:  # noqa: BLE001 — failover is based on pool position
                last_exc = exc
                break
            duplicate_aliases = find_duplicate_assigned_aliases(
                decisions=output.decisions,
                assigned_aliases=assigned_aliases,
            )
            if duplicate_aliases:
                last_exc = _SemanticValidationResponseError(
                    f"validator response contains duplicate assigned aliases {duplicate_aliases}"
                )
                has_same_alias_retry = semantic_attempt + 1 < semantic_attempts
                if has_same_alias_retry:
                    logger.warning(
                        "validator chunk %d: alias=%s returned duplicate assigned aliases; retrying same request",
                        chunk_index,
                        alias,
                    )
                    attempt_index += 1
                    continue
                break
            if attempt_index > 0:
                logger.info(
                    "validator chunk %d: recovered on alias=%s after %d prior attempt(s)",
                    chunk_index,
                    alias,
                    attempt_index,
                )
            return output
        remaining = len(facades) - facade_index - 1
        if remaining > 0:
            logger.warning(
                "validator chunk %d: alias=%s exhausted (%s); failing over (%d remaining)",
                chunk_index,
                alias,
                type(last_exc).__name__ if last_exc is not None else "unknown error",
                remaining,
            )
        else:
            logger.error(
                "validator chunk %d: alias=%s exhausted (%s); pool exhausted — row will be dropped",
                chunk_index,
                alias,
                type(last_exc).__name__ if last_exc is not None else "unknown error",
            )
        attempt_index += 1

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


async def _dispatch_chunk_async(
    *,
    facades: list[tuple[str, Any]],
    prompt: str,
    system_prompt: str | None,
    chunk_index: int,
    assigned_aliases: tuple[str, ...],
) -> RawValidationDecisionsSchema:
    """Async equivalent of :func:`_dispatch_chunk`, preserving failover order."""
    recipe = PydanticResponseRecipe(data_type=RawValidationDecisionsSchema)
    final_prompt = recipe.apply_recipe_to_user_prompt(prompt)
    final_system = recipe.apply_recipe_to_system_prompt(system_prompt)

    last_exc: BaseException | None = None
    attempt_index = 0
    for facade_index, (alias, facade) in enumerate(facades):
        semantic_attempts = 2 if len(facades) == 1 else 1
        for semantic_attempt in range(semantic_attempts):
            try:
                output, _messages = await facade.agenerate(
                    prompt=final_prompt,
                    parser=recipe.parse,
                    system_prompt=final_system,
                    purpose=f"entity-validation-chunk-{chunk_index}-attempt-{attempt_index}",
                )
            except Exception as exc:  # noqa: BLE001 — failover is based on pool position
                last_exc = exc
                break
            duplicate_aliases = find_duplicate_assigned_aliases(
                decisions=output.decisions,
                assigned_aliases=assigned_aliases,
            )
            if duplicate_aliases:
                last_exc = _SemanticValidationResponseError(
                    f"validator response contains duplicate assigned aliases {duplicate_aliases}"
                )
                has_same_alias_retry = semantic_attempt + 1 < semantic_attempts
                if has_same_alias_retry:
                    logger.warning(
                        "validator chunk %d: alias=%s returned duplicate assigned aliases; retrying same request",
                        chunk_index,
                        alias,
                    )
                    attempt_index += 1
                    continue
                break
            if attempt_index > 0:
                logger.info(
                    "validator chunk %d: recovered on alias=%s after %d prior attempt(s)",
                    chunk_index,
                    alias,
                    attempt_index,
                )
            return output
        remaining = len(facades) - facade_index - 1
        if remaining > 0:
            logger.warning(
                "validator chunk %d: alias=%s exhausted (%s); failing over (%d remaining)",
                chunk_index,
                alias,
                type(last_exc).__name__ if last_exc is not None else "unknown error",
                remaining,
            )
        else:
            logger.error(
                "validator chunk %d: alias=%s exhausted (%s); pool exhausted — row will be dropped",
                chunk_index,
                alias,
                type(last_exc).__name__ if last_exc is not None else "unknown error",
            )
        attempt_index += 1

    if last_exc is None:
        raise RuntimeError(
            "_dispatch_chunk_async was called with an empty facades list; "
            "this violates the caller contract (a non-empty validator pool)."
        )
    raise last_exc


def _build_lifecycle_dispatch_kwargs(
    row: dict[str, Any],
    params: ChunkedValidationParams,
    models: dict[str, Any],
) -> tuple[list[_PreparedValidationRequest], list[dict[str, Any]]]:
    missing_aliases = [alias for alias in params.pool if alias not in models]
    if missing_aliases:
        raise KeyError(
            f"Validator pool aliases {missing_aliases} not present in models dict. "
            "Ensure the configured validation pool matches the available models."
        )

    text = str(row.get(COL_TEXT, ""))
    detection_result = DetectionResult.model_validate(row[COL_SEED_DETECTION_RESULT])
    notation_raw = row.get(COL_TAG_NOTATION) or TagNotation.sentinel.value
    notation = TagNotation(str(notation_raw))
    requests = prepare_validation_requests(
        detection_result=detection_result,
        canonical_text=text,
        max_entities_per_call=params.max_entities_per_call,
        excerpt_window_chars=params.excerpt_window_chars,
        single_chunk_full_text=params.single_chunk_full_text,
        notation=notation,
        prompt_template=params.prompt_template,
        effective_label_policy=EffectiveLabelPolicy.from_labels(
            allowed_labels=params.allowed_entity_labels,
            excluded_labels=params.excluded_entity_labels,
        ),
    )

    dispatch_kwargs_per_chunk: list[dict[str, Any]] = []
    for chunk_index, request in enumerate(requests):
        start = chunk_index % len(params.pool)
        rotated_aliases = [params.pool[(start + offset) % len(params.pool)] for offset in range(len(params.pool))]
        dispatch_kwargs_per_chunk.append(
            {
                "facades": [(alias, models[alias]) for alias in rotated_aliases],
                "prompt": request.prompt,
                "system_prompt": params.system_prompt,
                "chunk_index": chunk_index,
                "assigned_aliases": tuple(entry.alias for entry in request.inventory.entries),
            }
        )
    return requests, dispatch_kwargs_per_chunk


def _chunk_worker_count(params: ChunkedValidationParams, chunk_count: int) -> int:
    if chunk_count == 0:
        return 0
    if params.max_parallel_chunks is None:
        return chunk_count
    return min(chunk_count, params.max_parallel_chunks)


async def _bounded_dispatch_chunk_async(
    semaphore: asyncio.Semaphore,
    kwargs: dict[str, Any],
) -> RawValidationDecisionsSchema:
    async with semaphore:
        return await _dispatch_chunk_async(**kwargs)


def chunked_validate_row(
    row: dict[str, Any],
    params: ChunkedValidationParams,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Run chunked validation for a single row and write ``COL_VALIDATION_DECISIONS``.

    The plugin's synchronous generator uses this path. Its asynchronous generator
    calls :func:`chunked_validate_row_async`.
    """
    requests, dispatch_kwargs_per_chunk = _build_lifecycle_dispatch_kwargs(row, params, models)

    # Dispatch all chunks concurrently via a ThreadPoolExecutor. Per-alias
    # concurrency is still capped downstream by each facade's
    # request-admission layer. ``f.result()`` re-raises the first chunk
    # exception; a single terminal chunk failure fails the row.
    if not dispatch_kwargs_per_chunk:
        chunk_results: list[RawValidationDecisionsSchema] = []
    else:
        with ThreadPoolExecutor(max_workers=_chunk_worker_count(params, len(dispatch_kwargs_per_chunk))) as executor:
            futures = [executor.submit(_dispatch_chunk, **kwargs) for kwargs in dispatch_kwargs_per_chunk]
            chunk_results = [f.result() for f in futures]

    row[COL_VALIDATION_INVENTORIES] = [request.inventory.model_dump(mode="json") for request in requests]
    row[COL_VALIDATION_DECISIONS] = serialize_inventory_decisions(chunk_results, requests)
    return row


async def chunked_validate_row_async(
    row: dict[str, Any],
    params: ChunkedValidationParams,
    models: dict[str, Any],
) -> dict[str, Any]:
    """Async-native chunked validation for plugin column generators."""
    requests, dispatch_kwargs_per_chunk = _build_lifecycle_dispatch_kwargs(row, params, models)
    if not dispatch_kwargs_per_chunk:
        chunk_results: list[RawValidationDecisionsSchema] = []
    else:
        semaphore = asyncio.Semaphore(_chunk_worker_count(params, len(dispatch_kwargs_per_chunk)))
        chunk_results = await asyncio.gather(
            *[_bounded_dispatch_chunk_async(semaphore, kwargs) for kwargs in dispatch_kwargs_per_chunk]
        )

    row[COL_VALIDATION_INVENTORIES] = [request.inventory.model_dump(mode="json") for request in requests]
    row[COL_VALIDATION_DECISIONS] = serialize_inventory_decisions(chunk_results, requests)
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
    generator reads the raw text, source-bound detector observations, and the
    tag notation used to render request excerpts.
    """
    if not pool:
        raise ValueError("Cannot build chunked validation generator: pool is empty.")

    @custom_column_generator(
        required_columns=[
            COL_TEXT,
            COL_SEED_DETECTION_RESULT,
            COL_TAG_NOTATION,
        ],
        side_effect_columns=[COL_VALIDATION_INVENTORIES],
        model_aliases=list(pool),
    )
    def chunked_validate(
        row: dict[str, Any],
        generator_params: ChunkedValidationParams,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return chunked_validate_row(row, generator_params, models)

    return chunked_validate
