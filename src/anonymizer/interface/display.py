# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd

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
    entities = _normalize_entity_list(row.get("_detected_entities", []))
    replacement_map = _normalize_replacement_map(row.get("_replacement_map", {}))

    original_html = _render_highlighted_text(text, entities)
    replaced_entities = _build_replaced_entities(entities, replacement_map, text, replaced_text)
    replaced_html = _render_highlighted_text(replaced_text, replaced_entities)
    table_html = _render_replacement_table(replacement_map)

    index_label = f" (record {record_index})" if record_index is not None else ""

    return _TEMPLATE.format(
        index_label=html.escape(index_label),
        original_html=original_html,
        replaced_html=replaced_html,
        table_html=table_html,
    )


def _render_highlighted_text(text: str, entities: list[dict[str, Any]]) -> str:
    """Render text with inline entity highlights using positioned spans."""
    if not entities:
        return f"<span>{html.escape(text)}</span>"

    sorted_entities = sorted(entities, key=lambda e: (int(e.get("start_position", 0)), int(e.get("end_position", 0))))

    parts: list[str] = []
    cursor = 0
    for entity in sorted_entities:
        start = int(entity.get("start_position", 0))
        end = int(entity.get("end_position", 0))
        label = str(entity.get("label", ""))

        if start < cursor or end <= start or end > len(text):
            continue

        _, border_color = _color_for_label(label)
        parts.append(html.escape(text[cursor:start]))
        entity_value = html.escape(text[start:end])
        parts.append(
            f'<span style="border:1.5px solid {border_color};'
            f'padding:1px 4px;border-radius:3px">'
            f"{entity_value}"
            f'<span style="color:{border_color};font-size:0.75em;font-weight:600;'
            f'margin-left:3px;opacity:0.85">| {html.escape(label)}</span></span>'
        )
        cursor = end

    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def _build_replaced_entities(
    original_entities: list[dict[str, Any]],
    replacement_map: list[dict[str, str]],
    original_text: str,
    replaced_text: str,
) -> list[dict[str, Any]]:
    """Compute entity positions in the replaced text by replaying the replacement splicing."""
    by_value_label: dict[tuple[str, str], str] = {}
    for entry in replacement_map:
        by_value_label[(entry.get("original", ""), entry.get("label", ""))] = entry.get("synthetic", "")

    sorted_entities = sorted(
        original_entities,
        key=lambda e: (int(e.get("start_position", 0)), int(e.get("end_position", 0))),
    )

    replaced_entities: list[dict[str, Any]] = []
    original_cursor = 0
    replaced_cursor = 0

    for entity in sorted_entities:
        start = int(entity.get("start_position", 0))
        end = int(entity.get("end_position", 0))
        value = str(entity.get("value", ""))
        label = str(entity.get("label", ""))

        if start < original_cursor or end <= start or end > len(original_text):
            continue

        gap = start - original_cursor
        replaced_cursor += gap

        synthetic = by_value_label.get((value, label), value)
        replaced_entities.append(
            {
                "value": synthetic,
                "label": label,
                "start_position": replaced_cursor,
                "end_position": replaced_cursor + len(synthetic),
            }
        )

        replaced_cursor += len(synthetic)
        original_cursor = end

    return replaced_entities


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


def _normalize_entity_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    return []


def _normalize_replacement_map(raw: Any) -> list[dict[str, str]]:
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(parsed, dict):
        replacements = parsed.get("replacements", [])
        if isinstance(replacements, list):
            return [r for r in replacements if isinstance(r, dict)]
    return []


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
