# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from anonymizer.config.replace_strategies import LocalReplaceMethod
from anonymizer.engine.constants import (
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
    COL_TEXT,
)
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.logging import ProgressTracker

logger = logging.getLogger("anonymizer")


@dataclass(frozen=True)
class ReplacementEntry:
    original: str
    label: str
    synthetic: str


@dataclass(frozen=True)
class ReplacementApplication:
    """Sanitized facts about an offset replacement attempt."""

    targeted_span_count: int
    applied_span_count: int
    skipped_span_count: int
    skipped_span_label_counts: dict[str, int]

    def to_metrics(self) -> dict[str, int | dict[str, int]]:
        return {
            "targeted_span_count": self.targeted_span_count,
            "applied_span_count": self.applied_span_count,
            "skipped_span_count": self.skipped_span_count,
            "skipped_span_label_counts": self.skipped_span_label_counts,
        }


def apply_local_replace_strategy(
    dataframe: pd.DataFrame,
    *,
    strategy: LocalReplaceMethod,
    text_column: str = COL_TEXT,
    entities_column: str = COL_FINAL_ENTITIES,
) -> pd.DataFrame:
    """Apply deterministic local replace strategy on detected entities."""
    output_df = dataframe.copy()
    tracker = ProgressTracker(total=len(output_df), label="Replacement")

    _debug = logger.isEnabledFor(logging.DEBUG)
    total_label_counts: Counter[str] = Counter()
    replacement_maps = []
    for idx, (_, row) in enumerate(output_df.iterrows()):
        rmap = _build_local_replacement_map(
            entities=EntitiesSchema.from_raw(row.get(entities_column, {})),
            strategy=strategy,
        )
        replacement_maps.append(rmap)
        if _debug:
            row_counts: Counter[str] = Counter()
            for r in rmap.get("replacements", []):
                row_counts[r.get("label", "unknown")] += 1
            total_label_counts += row_counts
            n = sum(row_counts.values())
            summary = ", ".join(f"{l}={c}" for l, c in row_counts.most_common()) if row_counts else "(none)"
            logger.debug("  record %d: %d replacements — %s", idx, n, summary)
        tracker.record_success()
    output_df[COL_REPLACEMENT_MAP] = replacement_maps

    if _debug:
        total = sum(total_label_counts.values())
        summary = ", ".join(f"{label}={count}" for label, count in total_label_counts.most_common())
        logger.debug("replacement stats: %d unique entities replaced (%s)", total, summary)

    replaced_texts = []
    applications = []
    for _, row in output_df.iterrows():
        replaced_text, application = apply_replacements_to_spans(
            text=str(row.get(text_column, "")),
            entities=EntitiesSchema.from_raw(row.get(entities_column, {})),
            replacements=_parse_replacements(row[COL_REPLACEMENT_MAP]),
            allow_value_fallback=True,
        )
        replaced_texts.append(replaced_text)
        applications.append(application.to_metrics())
    output_df[COL_REPLACED_TEXT] = replaced_texts
    output_df[COL_REPLACEMENT_APPLICATION] = applications

    tracker.log_final()
    return output_df


def apply_replacement_map(
    dataframe: pd.DataFrame,
    *,
    text_column: str = COL_TEXT,
    entities_column: str = COL_FINAL_ENTITIES,
    replacement_map_column: str = COL_REPLACEMENT_MAP,
) -> pd.DataFrame:
    """Apply pre-generated replacement map to text."""
    output_df = dataframe.copy()
    replaced_texts = []
    applications = []
    for _, row in output_df.iterrows():
        replaced_text, application = apply_replacements_to_spans(
            text=str(row.get(text_column, "")),
            entities=EntitiesSchema.from_raw(row.get(entities_column, {})),
            replacements=_parse_replacements(row.get(replacement_map_column, {"replacements": []})),
            allow_value_fallback=True,
        )
        replaced_texts.append(replaced_text)
        applications.append(application.to_metrics())
    output_df[COL_REPLACED_TEXT] = replaced_texts
    output_df[COL_REPLACEMENT_APPLICATION] = applications
    return output_df


def _build_local_replacement_map(
    entities: EntitiesSchema, strategy: LocalReplaceMethod
) -> dict[str, list[dict[str, str]]]:
    if not entities.entities:
        return {"replacements": []}
    replacements: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities.entities:
        if not entity.value or not entity.label:
            continue
        key = (entity.value, entity.label)
        if key in seen:
            continue
        seen.add(key)
        synthetic = strategy.replace(text=entity.value, label=entity.label)
        replacements.append({"original": entity.value, "label": entity.label, "synthetic": synthetic})
    return {"replacements": replacements}


def _apply_replacement_map_to_text(text: str, entities: EntitiesSchema, replacements: list[ReplacementEntry]) -> str:
    """Compatibility wrapper around the canonical offset replacement primitive."""
    return apply_replacements_to_spans(text, entities, replacements, allow_value_fallback=True)[0]


def apply_replacements_to_spans(
    text: str,
    entities: EntitiesSchema,
    replacements: list[ReplacementEntry],
    *,
    allow_value_fallback: bool,
) -> tuple[str, ReplacementApplication]:
    """Apply replacements once per admitted source span, without cascade matching.

    Every admitted span must be in range, non-overlapping, and exactly match the
    entity value.  Tuple matches are label-aware.  The legacy value-only fallback
    is available only when the value has one unambiguous synthetic value.
    """
    targeted = len(entities.entities)
    if not entities.entities:
        return text, ReplacementApplication(0, 0, 0, {})

    by_value_label, by_value = _index_replacements(replacements)

    spans = sorted(
        ((entity.start_position, entity.end_position, entity.value, entity.label) for entity in entities.entities),
        key=lambda item: item[0],
    )
    replaced, applied, skipped = _apply_indexed_replacements(
        text,
        spans,
        by_value_label=by_value_label,
        by_value=by_value,
        allow_value_fallback=allow_value_fallback,
    )
    return replaced, ReplacementApplication(
        targeted_span_count=targeted,
        applied_span_count=applied,
        skipped_span_count=sum(skipped.values()),
        skipped_span_label_counts=dict(sorted(skipped.items())),
    )


def _apply_indexed_replacements(
    text: str,
    spans: list[tuple[int, int, str, str]],
    *,
    by_value_label: dict[tuple[str, str], set[str]],
    by_value: dict[str, set[str]],
    allow_value_fallback: bool,
) -> tuple[str, int, Counter[str]]:

    parts: list[str] = []
    cursor = 0
    applied = 0
    skipped: Counter[str] = Counter()
    for start, end, value, label in spans:
        if start < cursor or end <= start or end > len(text) or text[start:end] != value:
            skipped[label or "unknown"] += 1
            continue
        synthetic = _resolve_synthetic(
            value,
            label,
            by_value_label=by_value_label,
            by_value=by_value,
            allow_value_fallback=allow_value_fallback,
        )
        if synthetic is None:
            skipped[label or "unknown"] += 1
            continue
        parts.append(text[cursor:start])
        parts.append(synthetic)
        cursor = end
        applied += 1
    parts.append(text[cursor:])
    return "".join(parts), applied, skipped


def _index_replacements(
    replacements: list[ReplacementEntry],
) -> tuple[dict[tuple[str, str], set[str]], dict[str, set[str]]]:
    by_value_label: dict[tuple[str, str], set[str]] = {}
    by_value: dict[str, set[str]] = {}
    for replacement in replacements:
        by_value_label.setdefault((replacement.original, replacement.label), set()).add(replacement.synthetic)
        by_value.setdefault(replacement.original, set()).add(replacement.synthetic)
    return by_value_label, by_value


def _resolve_synthetic(
    value: str,
    label: str,
    *,
    by_value_label: dict[tuple[str, str], set[str]],
    by_value: dict[str, set[str]],
    allow_value_fallback: bool,
) -> str | None:
    tuple_synthetics = by_value_label.get((value, label), set())
    if len(tuple_synthetics) == 1:
        return next(iter(tuple_synthetics))
    value_synthetics = by_value.get(value, set())
    if allow_value_fallback and len(value_synthetics) == 1:
        return next(iter(value_synthetics))
    return None


def _parse_replacements(raw: str | dict | object) -> list[ReplacementEntry]:
    """Parse raw replacement map (JSON string or dict) into typed entries."""
    parsed = raw
    model_dump = getattr(raw, "model_dump", None)
    if callable(model_dump):
        parsed = model_dump(mode="python")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, dict):
        return []
    replacements = parsed.get("replacements", [])
    if not isinstance(replacements, list):
        return []
    normalized: list[ReplacementEntry] = []
    for replacement in replacements:
        if not isinstance(replacement, dict):
            continue
        original = str(replacement.get("original", ""))
        label = str(replacement.get("label", ""))
        synthetic = str(replacement.get("synthetic", ""))
        if not original or not label or not synthetic:
            continue
        normalized.append(ReplacementEntry(original=original, label=label, synthetic=synthetic))
    return normalized
