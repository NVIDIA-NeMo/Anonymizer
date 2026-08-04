# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owns coercion of trace-dataframe cell payloads (JSON strings, Pydantic models, numpy/parquet round-trips) into plain dicts and counts for display."""

from __future__ import annotations

import json

import pandas as pd

from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_JUDGE,
    COL_ENTITIES_BY_VALUE,
    COL_RELATIONAL_CONSISTENCY_JUDGE,
    COL_REPLACEMENT_MAP,
)
from anonymizer.engine.schemas import (
    EntitiesByValueSchema,
    EntityReplacementMapSchema,
)

__all__ = [
    "normalize_replacement_map",
    "extract_judge_scores",
    "extract_all_attribute_entries",
    "normalize_attribute_entries",
    "extract_all_relations",
    "normalize_relations",
    "count_detected_entity_label_pairs",
    "count_replacement_triples",
    "normalize_invalid_entities",
    "normalize_disposition",
]


def normalize_replacement_map(raw: str | dict | object) -> list[dict[str, str]]:
    """Coerce ``_replacement_map`` cell values into a list of ``{original, label, synthetic}`` dicts.

    Cells can arrive as JSON strings, Pydantic models, plain dicts, or after a
    parquet round-trip with ``replacements`` wrapped as a ``numpy.ndarray``.
    Run the value through ``EntityReplacementMapSchema.model_validate`` so
    Pydantic's coercion absorbs numpy shapes, then fall back to a permissive
    hand-walk if validation rejects the payload.
    """
    if raw is None:
        return []
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="python")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(raw, dict):
        return []
    try:
        parsed = EntityReplacementMapSchema.model_validate(raw)
        return [r.model_dump() for r in parsed.replacements]
    except Exception:
        pass
    replacements = raw.get("replacements", [])
    if hasattr(replacements, "tolist"):
        replacements = replacements.tolist()
    if not isinstance(replacements, list):
        return []
    result: list[dict[str, str]] = []
    for r in replacements:
        if hasattr(r, "model_dump"):
            r = r.model_dump()
        if isinstance(r, dict):
            result.append(r)
    return result


def extract_judge_scores(raw: object) -> list[tuple[str, int | str]]:
    """Extract (name, score) pairs from the judge evaluation column.

    LLMJudgeColumnConfig output is a plain dict keyed by rubric name, each
    value carrying ``{"score": <int|str>, "reasoning": "..."}``. Scores are
    returned as-is — callers must not assume int (rewrite mode uses strings).
    """
    if not isinstance(raw, dict):
        return []
    result: list[tuple[str, int | str]] = []
    for name, value in raw.items():
        if not isinstance(value, dict) or "score" not in value:
            continue
        score = value["score"]
        if score is None:
            continue
        result.append((str(name), score))
    return result


def extract_all_attribute_entries(row: pd.Series) -> list[dict[str, object]]:
    """Read the full entities list (passes + fails) from the raw attribute-fidelity column."""
    raw = row.get(COL_ATTRIBUTE_FIDELITY_JUDGE) if COL_ATTRIBUTE_FIDELITY_JUDGE in row.index else None
    if raw is None:
        return []
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="python")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(raw, dict):
        return []
    entities = raw.get("entities", [])
    if hasattr(entities, "tolist"):
        entities = entities.tolist()
    if not isinstance(entities, list):
        return []
    out: list[dict[str, object]] = []
    for entry in entities:
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        if isinstance(entry, dict):
            out.append(entry)
    return out


def normalize_attribute_entries(raw: object) -> list[dict[str, object]]:
    """Coerce the invalid-entities column into a list of plain dicts."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        return []
    out: list[dict[str, object]] = []
    for entry in raw:
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        if isinstance(entry, dict):
            out.append(entry)
    return out


def extract_all_relations(row: pd.Series) -> list[dict[str, object]]:
    """Read the full relations list (passes + fails) from the raw judge column."""
    raw = row.get(COL_RELATIONAL_CONSISTENCY_JUDGE) if COL_RELATIONAL_CONSISTENCY_JUDGE in row.index else None
    if raw is None:
        return []
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="python")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(raw, dict):
        return []
    relations = raw.get("relations", [])
    if hasattr(relations, "tolist"):
        relations = relations.tolist()
    if not isinstance(relations, list):
        return []
    out: list[dict[str, object]] = []
    for entry in relations:
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        if isinstance(entry, dict):
            out.append(entry)
    return out


def normalize_relations(raw: object) -> list[dict[str, object]]:
    """Coerce the invalid-relations column into a list of plain dicts."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        return []
    out: list[dict[str, object]] = []
    for entry in raw:
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        if isinstance(entry, dict):
            out.append(entry)
    return out


def count_detected_entity_label_pairs(row: pd.Series) -> int:
    """Count (value, label) pairs the judge had a chance to evaluate.

    The judge schema flags entities at the (value, label) granularity, so the
    denominator for the LLM alignment score is the total number of such pairs in the
    deduped entity payload, not the number of unique values.
    """
    raw = row.get(COL_ENTITIES_BY_VALUE) if COL_ENTITIES_BY_VALUE in row.index else None
    if raw is None:
        return 0
    try:
        parsed = EntitiesByValueSchema.from_raw(raw)
    except Exception:
        return 0
    return sum(len(entity.labels) for entity in parsed.entities_by_value)


def count_replacement_triples(row: pd.Series, *, fallback: list[dict[str, str]]) -> int:
    """Count replacement entries the type-fidelity judge had a chance to evaluate.

    ``normalize_replacement_map`` (used to render the table) rejects shapes
    like ``{"replacements": numpy.ndarray(...)}`` from parquet round-trips,
    which would silently zero out the success-rate denominator here. Validate
    via Pydantic instead so the count matches what the judge actually saw.
    """
    raw = row.get(COL_REPLACEMENT_MAP) if COL_REPLACEMENT_MAP in row.index else None
    if raw is not None:
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(mode="python")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                raw = None
        if isinstance(raw, dict):
            try:
                return len(EntityReplacementMapSchema.model_validate(raw).replacements)
            except Exception:
                pass
    return len(fallback)


def normalize_invalid_entities(raw: object) -> list[dict[str, str]]:
    """Coerce the invalid-entities column into a list of plain dicts."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for entry in raw:
        if hasattr(entry, "model_dump"):
            entry = entry.model_dump()
        if isinstance(entry, dict):
            out.append(entry)
    return out


def normalize_disposition(raw: object) -> list[dict[str, str]]:
    """Extract disposition entries from the raw column value.

    LLMStructuredColumnConfig output lands as a plain dict keyed by
    ``sensitivity_disposition``, each entry being an EntityDispositionSchema dict.
    """
    if not isinstance(raw, dict):
        return []
    entries = raw.get("sensitivity_disposition", [])
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]
