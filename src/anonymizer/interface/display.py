# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import json
import re

import pandas as pd

from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_REPLACEMENT_MAP
from anonymizer.engine.schemas import EntitiesSchema, EntitySchema

ENTITY_COLORS: list[str] = [
    "#dbeafe",  # blue
    "#dcfce7",  # green
    "#fef3c7",  # amber
    "#fce7f3",  # pink
    "#e0e7ff",  # indigo
    "#d1fae5",  # emerald
    "#ffedd5",  # orange
    "#ede9fe",  # violet
    "#cffafe",  # cyan
    "#fef9c3",  # yellow
    "#f3e8ff",  # purple
    "#fee2e2",  # red
]

LABEL_BORDER_COLORS: list[str] = [
    "#3b82f6",
    "#22c55e",
    "#f59e0b",
    "#ec4899",
    "#6366f1",
    "#10b981",
    "#f97316",
    "#8b5cf6",
    "#06b6d4",
    "#eab308",
    "#a855f7",
    "#ef4444",
]


def _color_for_label(label: str) -> tuple[str, str]:
    idx = hash(label) % len(ENTITY_COLORS)
    return ENTITY_COLORS[idx], LABEL_BORDER_COLORS[idx]


def render_record_html(row: pd.Series, record_index: int | None = None, original_text_column: str | None = None) -> str:
    """Render a single anonymizer result record as self-contained HTML.

    Returns an HTML string with three sections:
    1. Original text with entity highlights
    2. Replaced text with entity highlights
    3. Replacement map table
    """
    text_col = original_text_column or "text"
    text = str(row.get(text_col, ""))
    replaced_text = str(row.get(f"{text_col}_replaced", ""))
    entities = EntitiesSchema.from_raw(row.get(COL_DETECTED_ENTITIES, {})).entities
    replacement_map = _normalize_replacement_map(row.get(COL_REPLACEMENT_MAP, {}))

    if not entities and replacement_map:
        entities = _build_original_entities_from_map(replacement_map, text)

    original_html = _render_highlighted_text(text, entities)

    replaced_entities = _build_replaced_entities(entities, replacement_map, text)
    if not replaced_entities and replacement_map:
        replaced_entities = _build_replaced_entities_from_map(replacement_map, replaced_text)
    replaced_html = _render_highlighted_text(replaced_text, replaced_entities)
    table_html = _render_replacement_table(replacement_map)

    index_label = f" (record {record_index})" if record_index is not None else ""

    return _TEMPLATE.format(
        index_label=html.escape(index_label),
        original_html=original_html,
        replaced_html=replaced_html,
        table_html=table_html,
    )


def _render_highlighted_text(text: str, entities: list[EntitySchema]) -> str:
    """Render text with inline entity highlights using positioned spans."""
    if not entities:
        return f"<span>{html.escape(text)}</span>"

    sorted_entities = sorted(entities, key=lambda e: (e.start_position, e.end_position))

    parts: list[str] = []
    cursor = 0
    for entity in sorted_entities:
        if (
            entity.start_position < cursor
            or entity.end_position <= entity.start_position
            or entity.end_position > len(text)
        ):
            continue

        _, border_color = _color_for_label(entity.label)
        parts.append(html.escape(text[cursor : entity.start_position]))
        entity_value = html.escape(text[entity.start_position : entity.end_position])
        parts.append(
            f'<span style="border:1.5px solid {border_color};'
            f'padding:1px 4px;border-radius:3px">'
            f"{entity_value}"
            f'<span style="color:{border_color};font-size:0.75em;font-weight:600;'
            f'margin-left:3px;opacity:0.85">| {html.escape(entity.label)}</span></span>'
        )
        cursor = entity.end_position

    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def _build_replaced_entities(
    original_entities: list[EntitySchema],
    replacement_map: list[dict[str, str]],
    original_text: str,
) -> list[EntitySchema]:
    """Compute entity positions in the replaced text by replaying the replacement splicing."""
    by_value_label: dict[tuple[str, str], str] = {}
    by_value: dict[str, str] = {}
    for entry in replacement_map:
        orig = entry.get("original", "")
        label = entry.get("label", "")
        synth = entry.get("synthetic", "")
        by_value_label[(orig, label)] = synth
        by_value[orig] = synth

    sorted_entities = sorted(original_entities, key=lambda e: (e.start_position, e.end_position))

    replaced_entities: list[EntitySchema] = []
    original_cursor = 0
    replaced_cursor = 0

    for entity in sorted_entities:
        if (
            entity.start_position < original_cursor
            or entity.end_position <= entity.start_position
            or entity.end_position > len(original_text)
        ):
            continue

        gap = entity.start_position - original_cursor
        replaced_cursor += gap

        synthetic = by_value_label.get((entity.value, entity.label), by_value.get(entity.value, entity.value))
        replaced_entities.append(
            EntitySchema(
                value=synthetic,
                label=entity.label,
                start_position=replaced_cursor,
                end_position=replaced_cursor + len(synthetic),
            )
        )

        replaced_cursor += len(synthetic)
        original_cursor = entity.end_position

    return replaced_entities


def _build_original_entities_from_map(
    replacement_map: list[dict[str, str]],
    original_text: str,
) -> list[EntitySchema]:
    """Build entity positions by finding original values in original text.

    Fallback when _detected_entities is empty but replacement_map exists.
    Uses case-insensitive matching to align with how the detection engine
    finds entities (e.g. "The Lantern" matching "the Lantern" in text).
    """
    result: list[EntitySchema] = []
    for entry in replacement_map:
        original = str(entry.get("original", ""))
        label = str(entry.get("label", ""))
        if not original or not label:
            continue
        for match in re.finditer(re.escape(original), original_text, flags=re.IGNORECASE):
            result.append(
                EntitySchema(
                    value=original_text[match.start() : match.end()],
                    label=label,
                    start_position=match.start(),
                    end_position=match.end(),
                )
            )
    return sorted(result, key=lambda e: (e.start_position, e.end_position))


def _build_replaced_entities_from_map(
    replacement_map: list[dict[str, str]],
    replaced_text: str,
) -> list[EntitySchema]:
    """Build entity positions by finding synthetic values in replaced text.

    Fallback when _detected_entities is empty but replacement_map exists (e.g. LLM
    replace path where entity format differs).
    """
    result: list[EntitySchema] = []
    for entry in replacement_map:
        synthetic = str(entry.get("synthetic", ""))
        label = str(entry.get("label", ""))
        if not synthetic or not label:
            continue
        start = 0
        while True:
            pos = replaced_text.find(synthetic, start)
            if pos < 0:
                break
            result.append(
                EntitySchema(
                    value=synthetic,
                    label=label,
                    start_position=pos,
                    end_position=pos + len(synthetic),
                )
            )
            start = pos + len(synthetic)
    return sorted(result, key=lambda e: (e.start_position, e.end_position))


def _render_replacement_table(replacement_map: list[dict[str, str]]) -> str:
    if not replacement_map:
        return "<p style='opacity:0.5;font-style:italic'>No replacement map available.</p>"

    rows: list[str] = []
    for entry in replacement_map:
        original = html.escape(str(entry.get("original", "")))
        label = str(entry.get("label", ""))
        synthetic = html.escape(str(entry.get("synthetic", "")))
        _, border_color = _color_for_label(label)
        rows.append(
            f"<tr>"
            f"<td style='padding:6px 12px'>{original}</td>"
            f"<td style='padding:6px 12px'>"
            f"<span style='border:1.5px solid {border_color};"
            f"padding:1px 6px;border-radius:3px;font-size:0.85em'>{html.escape(label)}</span></td>"
            f"<td style='padding:6px 12px'>{synthetic}</td>"
            f"</tr>"
        )

    return (
        "<table style='border-collapse:collapse;width:100%;font-size:0.9em'>"
        "<thead><tr style='border-bottom:2px solid currentColor;text-align:left;opacity:0.7'>"
        "<th style='padding:6px 12px'>Original</th>"
        "<th style='padding:6px 12px'>Label</th>"
        "<th style='padding:6px 12px'>Replacement</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _normalize_replacement_map(raw: str | dict | object) -> list[dict[str, str]]:
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
    elif hasattr(raw, "model_dump"):
        data = raw.model_dump()
    else:
        data = raw
    if not isinstance(data, dict):
        return []
    replacements = data.get("replacements", [])
    if not isinstance(replacements, list):
        return []
    result: list[dict[str, str]] = []
    for r in replacements:
        if isinstance(r, dict):
            result.append(r)
        elif hasattr(r, "model_dump"):
            result.append(r.model_dump())
    return result


_TEMPLATE = """\
<div style="font-family:system-ui,-apple-system,sans-serif;max-width:960px;margin:12px 0">
  <div style="border:1px solid currentColor;border-radius:8px;overflow:hidden;opacity:0.95">
    <div style="padding:10px 16px;border-bottom:1px solid currentColor;opacity:0.7">
      <strong>Anonymizer Preview{index_label}</strong>
    </div>
    <div style="padding:16px">
      <div style="margin-bottom:16px">
        <div style="font-size:0.8em;font-weight:600;text-transform:uppercase;\
letter-spacing:0.05em;margin-bottom:6px;opacity:0.5">Original</div>
        <div style="font-family:'SF Mono',Menlo,Consolas,monospace;font-size:0.85em;\
line-height:1.6;white-space:pre-wrap;padding:12px;border:1px solid currentColor;\
border-radius:6px;opacity:0.85">{original_html}</div>
      </div>
      <div style="margin-bottom:16px">
        <div style="font-size:0.8em;font-weight:600;text-transform:uppercase;\
letter-spacing:0.05em;margin-bottom:6px;opacity:0.5">Replaced</div>
        <div style="font-family:'SF Mono',Menlo,Consolas,monospace;font-size:0.85em;\
line-height:1.6;white-space:pre-wrap;padding:12px;border:1px solid currentColor;\
border-radius:6px;opacity:0.85">{replaced_html}</div>
      </div>
      <div>
        <div style="font-size:0.8em;font-weight:600;text-transform:uppercase;\
letter-spacing:0.05em;margin-bottom:6px;opacity:0.5">Replacement Map</div>
        {table_html}
      </div>
    </div>
  </div>
</div>"""
