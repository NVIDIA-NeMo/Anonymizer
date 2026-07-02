# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunked LLM generalization suggestions for the rewrite pipeline.

For each row, extract entities whose protection_method_suggestion is "generalize",
partition them into chunks of at most ``chunk_size``, and dispatch each chunk to a
model alias selected round-robin from the configured pool. Chunks run concurrently
via ``ThreadPoolExecutor``. The per-chunk results are merged into a flat
``dict[entity_value → generalization_suggestion]`` stored in
``COL_GENERALIZATION_SUGGESTIONS``.

Public entry point: :class:`GeneralizationSuggestionsWorkflow` whose ``columns()``
method returns the column configs to insert after sensitivity disposition in the
pipeline. ``chunk_size`` is the only tuning knob exposed at call time (default 10).

Failure contract mirrors chunked validation: a chunk failing on every pool member
re-raises out of the generator, DataDesigner drops the row, and
``NddAdapter._detect_missing_records`` surfaces it as a ``FailedRecord``.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe

from anonymizer.config.models import RewriteModelSelection
from anonymizer.engine.constants import (
    COL_GENERALIZATION_SUGGESTIONS,
    COL_SENSITIVITY_DISPOSITION,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.schemas import (
    GeneralizationSuggestionsSchema,
    ProtectionMethod,
    SensitivityDispositionSchema,
)

logger = logging.getLogger("anonymizer.rewrite.generalization_suggestions")

_DEFAULT_CHUNK_SIZE = 10


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _get_generalization_prompt(entities: list[dict[str, str]]) -> str:
    """Build a prompt asking the LLM to generalize a chunk of entities."""
    entities_json = json.dumps(entities, indent=2)
    return f"""You are generating generalization suggestions for privacy-preserving text rewriting.

For each entity below, produce a concise phrase that generalizes it — specific enough to preserve semantic meaning but broad enough to prevent re-identification.

<examples>
- entity_value: "Portland", entity_label: "city" → "a city in the Pacific Northwest"
- entity_value: "MIT", entity_label: "university" → "a prestigious technical university"
- entity_value: "34", entity_label: "age" → "mid-thirties"
- entity_value: "Google", entity_label: "employer" → "a major technology company"
- entity_value: "March 15, 1987", entity_label: "date_of_birth" → "the late 1980s"
- entity_value: "software engineer", entity_label: "occupation" → "a technical professional"
</examples>

<entities>
{entities_json}
</entities>

Return exactly one suggestion per entity. Use the exact entity_value and entity_label from the input in your response."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _extract_generalize_entities(disposition_raw: Any) -> list[dict[str, str]]:
    """Extract entities with protection_method_suggestion == 'generalize' from disposition."""
    disposition_raw = normalize_payload(disposition_raw)
    if isinstance(disposition_raw, SensitivityDispositionSchema):
        disposition = disposition_raw
    elif isinstance(disposition_raw, dict):
        try:
            disposition = SensitivityDispositionSchema.model_validate(disposition_raw)
        except Exception:
            logger.warning("Failed to parse sensitivity disposition for generalization extraction")
            return []
    else:
        return []
    return [
        {"entity_value": e.entity_value, "entity_label": e.entity_label}
        for e in disposition.sensitivity_disposition
        if e.protection_method_suggestion == ProtectionMethod.generalize
    ]


def _chunk_entities(
    entities: list[dict[str, str]], chunk_size: int
) -> list[list[dict[str, str]]]:
    return [entities[i : i + chunk_size] for i in range(0, len(entities), chunk_size)]


def _merge_chunk_results(
    chunk_results: list[GeneralizationSuggestionsSchema],
) -> dict[str, str]:
    """Flatten chunk results into entity_value → generalization_suggestion."""
    merged: dict[str, str] = {}
    for result in chunk_results:
        for item in result.suggestions:
            if item.entity_value not in merged:
                merged[item.entity_value] = item.generalization_suggestion
    return merged


# ---------------------------------------------------------------------------
# Chunk dispatch
# ---------------------------------------------------------------------------


def _dispatch_generalization_chunk(
    *,
    facades: list[tuple[str, Any]],
    prompt: str,
    chunk_index: int,
) -> GeneralizationSuggestionsSchema:
    """Dispatch one chunk with cross-alias failover."""
    recipe = PydanticResponseRecipe(data_type=GeneralizationSuggestionsSchema)
    final_prompt = recipe.apply_recipe_to_user_prompt(prompt)
    final_system = recipe.apply_recipe_to_system_prompt(None)

    last_exc: BaseException | None = None
    for attempt_index, (alias, facade) in enumerate(facades):
        try:
            output, _messages = facade.generate(
                prompt=final_prompt,
                parser=recipe.parse,
                system_prompt=final_system,
                purpose=f"generalization-chunk-{chunk_index}-attempt-{attempt_index}",
            )
            if attempt_index > 0:
                logger.info(
                    "generalization chunk %d: recovered on failover alias=%s (attempt %d of %d)",
                    chunk_index,
                    alias,
                    attempt_index + 1,
                    len(facades),
                )
            return output
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            remaining = len(facades) - attempt_index - 1
            if remaining > 0:
                logger.warning(
                    "generalization chunk %d: alias=%s raised %s (%s); failing over (%d remaining)",
                    chunk_index,
                    alias,
                    type(exc).__name__,
                    exc,
                    remaining,
                )
            else:
                logger.error(
                    "generalization chunk %d: alias=%s raised %s (%s); pool exhausted — row will be dropped",
                    chunk_index,
                    alias,
                    type(exc).__name__,
                    exc,
                )

    if last_exc is None:
        raise RuntimeError("_dispatch_generalization_chunk called with empty facades list.")
    raise last_exc


# ---------------------------------------------------------------------------
# Row-level worker (testable without DataDesigner)
# ---------------------------------------------------------------------------


def generalize_row(
    row: dict[str, Any],
    pool: list[str],
    models: dict[str, Any],
    chunk_size: int,
) -> dict[str, Any]:
    """Run chunked generalization suggestion for a single row."""
    entities = _extract_generalize_entities(row.get(COL_SENSITIVITY_DISPOSITION, {}))

    if not entities:
        row[COL_GENERALIZATION_SUGGESTIONS] = {}
        return row

    chunks = _chunk_entities(entities, chunk_size)
    logger.debug(
        "generalization suggestions: %d entity(ies) in %d chunk(s) (size=%d), pool=%s",
        len(entities),
        len(chunks),
        chunk_size,
        pool,
    )

    dispatch_kwargs: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(chunks):
        prompt = _get_generalization_prompt(chunk)
        start = chunk_index % len(pool)
        rotated_aliases = [pool[(start + offset) % len(pool)] for offset in range(len(pool))]
        facades = [(alias, models[alias]) for alias in rotated_aliases]
        dispatch_kwargs.append({"facades": facades, "prompt": prompt, "chunk_index": chunk_index})

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [executor.submit(_dispatch_generalization_chunk, **kwargs) for kwargs in dispatch_kwargs]
        chunk_results = [f.result() for f in futures]

    row[COL_GENERALIZATION_SUGGESTIONS] = _merge_chunk_results(chunk_results)
    return row


# ---------------------------------------------------------------------------
# DataDesigner wiring factory
# ---------------------------------------------------------------------------


def make_generalization_suggestions_generator(
    pool: list[str], chunk_size: int = _DEFAULT_CHUNK_SIZE
) -> Any:
    """Build a ``@custom_column_generator``-decorated function bound to ``pool`` and ``chunk_size``."""
    if not pool:
        raise ValueError("Cannot build generalization suggestions generator: pool is empty.")

    @custom_column_generator(
        required_columns=[COL_SENSITIVITY_DISPOSITION],
        model_aliases=list(pool),
    )
    def _generate_generalization_suggestions(
        row: dict[str, Any],
        generator_params: None,
        models: dict[str, Any],
    ) -> dict[str, Any]:
        return generalize_row(row, pool, models, chunk_size)

    return _generate_generalization_suggestions


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class GeneralizationSuggestionsWorkflow:
    def columns(
        self,
        *,
        selected_models: RewriteModelSelection,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> list[CustomColumnConfig]:
        alias = resolve_model_alias("disposition_analyzer", selected_models)
        generator = make_generalization_suggestions_generator(pool=[alias], chunk_size=chunk_size)
        return [
            CustomColumnConfig(
                name=COL_GENERALIZATION_SUGGESTIONS,
                generator_function=generator,
            )
        ]
