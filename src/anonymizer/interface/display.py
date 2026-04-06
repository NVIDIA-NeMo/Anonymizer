# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass

import pandas as pd

from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_FINAL_ENTITIES,
    COL_JUDGE_EVALUATION,
    COL_REPLACEMENT_MAP,
    COL_SENSITIVITY_DISPOSITION,
)
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


@dataclass(frozen=True)
class _SyntheticLookupMaps:
    by_value_label: dict[tuple[str, str], str]
    by_value: dict[str, str]
    by_value_label_ci: dict[tuple[str, str], str]
    by_value_ci: dict[str, str]


def _color_for_label(label: str) -> tuple[str, str]:
    idx = hash(label) % len(ENTITY_COLORS)
    return ENTITY_COLORS[idx], LABEL_BORDER_COLORS[idx]


def render_record_html(row: pd.Series, record_index: int | None = None, original_text_column: str | None = None) -> str:
    """Render a single anonymizer result record as self-contained HTML.

    Dispatches to rewrite-mode or replace-mode layout based on which output
    columns are present.
    """
    text_col = original_text_column or "text"

    if f"{text_col}_rewritten" in row.index:
        return _render_rewrite_html(row, text_col=text_col, record_index=record_index)

    return _render_replace_html(row, text_col=text_col, record_index=record_index)


def _render_replace_html(row: pd.Series, *, text_col: str, record_index: int | None) -> str:
    """Replace-mode layout: Original, Replaced, Replacement Map."""
    text = str(row.get(text_col, ""))
    replaced_text = str(row.get(f"{text_col}_replaced", ""))
    entities = _resolve_display_entities(row)
    replacement_map = _normalize_replacement_map(row.get(COL_REPLACEMENT_MAP, {}))

    if not entities and replacement_map:
        entities = _build_original_entities_from_map(replacement_map, text)

    original_html = _render_highlighted_text(text, entities)

    replaced_entities = _build_replaced_entities(entities, replacement_map, text, replaced_text)
    if not replaced_entities and replacement_map:
        replaced_entities = _build_replaced_entities_from_map(replacement_map, replaced_text)
    replaced_html = _render_highlighted_text(replaced_text, replaced_entities)
    table_html = _render_replacement_table(replacement_map)

    index_label = f" (record {record_index})" if record_index is not None else ""

    return _REPLACE_TEMPLATE.format(
        index_label=html.escape(index_label),
        original_html=original_html,
        replaced_html=replaced_html,
        table_html=table_html,
    )


def _render_rewrite_html(row: pd.Series, *, text_col: str, record_index: int | None) -> str:
    """Rewrite-mode layout: Original (highlighted), Rewritten, Scores, Entity Disposition."""
    text = str(row.get(text_col, ""))
    rewritten_text = str(row.get(f"{text_col}_rewritten", ""))
    entities = _resolve_display_entities(row)
    original_html = _render_highlighted_text(text, entities)
    rewritten_html = f"<span>{html.escape(rewritten_text)}</span>"
    scores_html = _render_scores_section(row)
    disposition_html = _render_disposition_table(row)
    index_label = f" (record {record_index})" if record_index is not None else ""

    return _REWRITE_TEMPLATE.format(
        index_label=html.escape(index_label),
        original_html=original_html,
        rewritten_html=rewritten_html,
        scores_html=scores_html,
        disposition_html=disposition_html,
    )


def _resolve_display_entities(row: pd.Series) -> list[EntitySchema]:
    """Prefer the final entity set so preview matches replacement behavior."""
    if COL_FINAL_ENTITIES in row:
        return EntitiesSchema.from_raw(row.get(COL_FINAL_ENTITIES, {})).entities
    return EntitiesSchema.from_raw(row.get(COL_DETECTED_ENTITIES, {})).entities


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
    replaced_text: str,
) -> list[EntitySchema]:
    """Compute entity positions in the replaced text by locating synthetic values directly.

    Instead of replaying cursor arithmetic (which drifts when entity values
    differ in case from the replacement map keys), we resolve each entity's
    synthetic value and find it in the replaced text by scanning forward.
    """
    by_value_label: dict[tuple[str, str], str] = {}
    by_value: dict[str, str] = {}
    by_value_label_ci: dict[tuple[str, str], str] = {}
    by_value_ci: dict[str, str] = {}
    for entry in replacement_map:
        orig = entry.get("original", "")
        label = entry.get("label", "")
        synth = entry.get("synthetic", "")
        by_value_label[(orig, label)] = synth
        by_value[orig] = synth
        by_value_label_ci[(orig.lower(), label)] = synth
        by_value_ci[orig.lower()] = synth
    lookups = _SyntheticLookupMaps(
        by_value_label=by_value_label,
        by_value=by_value,
        by_value_label_ci=by_value_label_ci,
        by_value_ci=by_value_ci,
    )

    sorted_entities = sorted(original_entities, key=lambda e: (e.start_position, e.end_position))
    replaced_entities: list[EntitySchema] = []
    original_cursor = 0
    search_from = 0

    for entity in sorted_entities:
        start = entity.start_position
        end = entity.end_position
        value = entity.value
        label = entity.label
        if start < original_cursor or end <= start or end > len(original_text):
            continue

        synthetic = _resolve_synthetic(value, label, lookups)
        original_span = original_text[start:end]

        pos = replaced_text.find(synthetic, search_from) if synthetic else -1
        if pos < 0 and synthetic != original_span:
            pos = replaced_text.find(original_span, search_from)
            if pos >= 0:
                synthetic = original_span

        if pos < 0:
            original_cursor = end
            continue

        replaced_entities.append(
            EntitySchema(
                value=replaced_text[pos : pos + len(synthetic)],
                label=label,
                start_position=pos,
                end_position=pos + len(synthetic),
            )
        )
        search_from = pos + len(synthetic)
        original_cursor = end

    return replaced_entities


def _resolve_synthetic(
    value: str,
    label: str,
    lookups: _SyntheticLookupMaps,
) -> str:
    """Look up synthetic value with exact-match first, then case-insensitive fallback."""
    result = lookups.by_value_label.get((value, label))
    if result is not None:
        return result
    result = lookups.by_value.get(value)
    if result is not None:
        return result
    result = lookups.by_value_label_ci.get((value.lower(), label))
    if result is not None:
        return result
    result = lookups.by_value_ci.get(value.lower())
    if result is not None:
        return result
    return value


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


def _render_scores_section(row: pd.Series) -> str:
    """Render utility/leakage metrics and optional judge scores."""
    parts: list[str] = []

    utility = row.get("utility_score")
    leakage = row.get("leakage_mass")
    weighted_leakage_rate = row.get("weighted_leakage_rate")
    needs_review = row.get("needs_human_review")

    if utility is not None:
        parts.append(f"<span style='margin-right:16px'><strong>Utility:</strong> {float(utility):.2f}</span>")
    if leakage is not None:
        parts.append(f"<span style='margin-right:16px'><strong>Leakage:</strong> {float(leakage):.2f}</span>")
    if weighted_leakage_rate is not None:
        parts.append(
            "<span style='margin-right:16px'><strong>Weighted Leakage Rate:</strong> "
            f"{float(weighted_leakage_rate):.2f}</span>"
        )
    if needs_review is not None:
        badge_color = "#ef4444" if needs_review else "#22c55e"
        badge_text = "Yes" if needs_review else "No"
        parts.append(
            f"<span style='margin-right:16px'><strong>Needs Review:</strong> "
            f"<span style='color:{badge_color};font-weight:600'>{badge_text}</span></span>"
        )

    judge_raw = row.get(COL_JUDGE_EVALUATION)
    judge_scores = _extract_judge_scores(judge_raw)
    if judge_scores:
        score_strs = [f"{name}: {score}/10" for name, score in judge_scores]
        parts.append(f"<span><strong>Judge:</strong> {html.escape(', '.join(score_strs))}</span>")

    if not parts:
        return "<p style='opacity:0.5;font-style:italic'>No scores available.</p>"
    return "<div style='font-size:0.9em;line-height:1.8'>" + "".join(parts) + "</div>"


def _extract_judge_scores(raw: object) -> list[tuple[str, int]]:
    """Extract (name, score) pairs from the judge evaluation column.

    LLMJudgeColumnConfig output is keyed by rubric name, each value carrying
    ``{"score": <enum_value>, "reasoning": "..."}``.
    """
    if raw is None:
        return []
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(raw, dict):
        return []
    result: list[tuple[str, int]] = []
    for name, value in raw.items():
        if not isinstance(value, dict) or "score" not in value:
            continue
        try:
            result.append((str(name), int(value["score"])))
        except (ValueError, TypeError):
            continue
    return result


def _render_disposition_table(row: pd.Series) -> str:
    """Render entity disposition table from _sensitivity_disposition column."""
    raw = row.get(COL_SENSITIVITY_DISPOSITION)
    entries = _normalize_disposition(raw)
    if not entries:
        return ""

    rows_html: list[str] = []
    for entry in entries:
        value = html.escape(str(entry.get("entity_value", "")))
        label = str(entry.get("entity_label", ""))
        sensitivity = html.escape(str(entry.get("sensitivity", "")))
        method = html.escape(str(entry.get("protection_method_suggestion", "")))
        _, border_color = _color_for_label(label)
        rows_html.append(
            f"<tr>"
            f"<td style='padding:6px 12px'>{value}</td>"
            f"<td style='padding:6px 12px'>"
            f"<span style='border:1.5px solid {border_color};"
            f"padding:1px 6px;border-radius:3px;font-size:0.85em'>{html.escape(label)}</span></td>"
            f"<td style='padding:6px 12px'>{sensitivity}</td>"
            f"<td style='padding:6px 12px'>{method}</td>"
            f"</tr>"
        )

    return (
        "<table style='border-collapse:collapse;width:100%;font-size:0.9em'>"
        "<thead><tr style='border-bottom:2px solid currentColor;text-align:left;opacity:0.7'>"
        "<th style='padding:6px 12px'>Entity</th>"
        "<th style='padding:6px 12px'>Label</th>"
        "<th style='padding:6px 12px'>Sensitivity</th>"
        "<th style='padding:6px 12px'>Protection</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
    )


def _normalize_disposition(raw: object) -> list[dict[str, str]]:
    """Extract disposition entries from the raw column value."""
    if raw is None:
        return []
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(raw, dict):
        entries = raw.get("sensitivity_disposition", [])
        if isinstance(entries, list):
            return [e if isinstance(e, dict) else getattr(e, "model_dump", dict)() for e in entries]
    return []


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_REPLACE_TEMPLATE = """\
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

_REWRITE_TEMPLATE = """\
<div style="font-family:system-ui,-apple-system,sans-serif;max-width:960px;margin:12px 0">
  <div style="border:1px solid currentColor;border-radius:8px;overflow:hidden;opacity:0.95">
    <div style="padding:10px 16px;border-bottom:1px solid currentColor;opacity:0.7">
      <strong>Anonymizer Rewrite Preview{index_label}</strong>
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
letter-spacing:0.05em;margin-bottom:6px;opacity:0.5">Rewritten</div>
        <div style="font-family:'SF Mono',Menlo,Consolas,monospace;font-size:0.85em;\
line-height:1.6;white-space:pre-wrap;padding:12px;border:1px solid currentColor;\
border-radius:6px;opacity:0.85">{rewritten_html}</div>
      </div>
      <div style="margin-bottom:16px">
        <div style="font-size:0.8em;font-weight:600;text-transform:uppercase;\
letter-spacing:0.05em;margin-bottom:6px;opacity:0.5">Scores</div>
        {scores_html}
      </div>
      <div>
        <div style="font-size:0.8em;font-weight:600;text-transform:uppercase;\
letter-spacing:0.05em;margin-bottom:6px;opacity:0.5">Entity Disposition</div>
        {disposition_html}
      </div>
    </div>
  </div>
</div>"""
