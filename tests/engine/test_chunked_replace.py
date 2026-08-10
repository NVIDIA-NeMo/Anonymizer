# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for chunked (long-context) Substitute replacement-map generation."""

from __future__ import annotations

import json
import re

import pytest

from anonymizer.engine.constants import (
    COL_ENTITIES_FOR_REPLACE,
    COL_ENTITY_EXAMPLES,
    COL_FINAL_ENTITIES,
    COL_REPLACEMENT_MAP,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.postprocess import EntitySpan, TagNotation
from anonymizer.engine.replace.chunked_replace import (
    WindowedReplaceParams,
    chunk_tagged_text,
    generate_replacement_map_row,
    merge_replacements,
    new_chunk_entities,
)
from anonymizer.engine.schemas import EntitiesSchema, EntityReplacementMapSchema

# Single-call prompt stand-in (the real one is large); references the columns the
# fast path renders with the row.
_SINGLE_PROMPT = (
    "MAP {{ tagged_text }} || {% for e in _entities_for_replace %}{{ e.value }};{% endfor %} || {{ _entity_examples }}"
)


def _span(value: str, label: str, start: int, end: int) -> EntitySpan:
    return EntitySpan(
        entity_id=f"{label}_{start}",
        value=value,
        label=label,
        start_position=start,
        end_position=end,
        score=1.0,
        source="d",
    )


class TestNewChunkEntities:
    def test_selects_in_window_excludes_mapped(self) -> None:
        spans = [_span("Alice", "name", 0, 5), _span("Bob", "name", 50, 53), _span("Alice", "name", 60, 65)]
        # window [0,55): Alice(0) + Bob(50); Alice already mapped -> excluded
        got = new_chunk_entities(spans, 0, 55, already_mapped={("Alice", "name")})
        assert [(e["value"], e["labels_str"]) for e in got] == [("Bob", "name")]

    def test_groups_labels_per_value(self) -> None:
        spans = [_span("Wash", "city", 0, 4), _span("Wash", "last_name", 10, 14)]
        got = new_chunk_entities(spans, 0, 20, already_mapped=set())
        assert got == [{"value": "Wash", "labels": ["city", "last_name"], "labels_str": "city, last_name"}]


class TestMergeReplacements:
    def test_dedupes_by_original_label_earlier_wins(self) -> None:
        existing = [{"original": "Alice", "label": "name", "synthetic": "Jane"}]
        new = EntityReplacementMapSchema.model_validate(
            {
                "replacements": [
                    {"original": "Alice", "label": "name", "synthetic": "DIFFERENT"},  # dup -> ignored
                    {"original": "Bob", "label": "name", "synthetic": "Mike"},
                ]
            }
        )
        merged = merge_replacements(existing, new)
        assert merged == [
            {"original": "Alice", "label": "name", "synthetic": "Jane"},
            {"original": "Bob", "label": "name", "synthetic": "Mike"},
        ]


def test_chunk_tagged_text_rebases_and_tags_in_window() -> None:
    text = "Alice met Bob in Paris"  # Alice 0-5, Bob 10-13
    spans = [_span("Alice", "first_name", 0, 5), _span("Bob", "first_name", 10, 13), _span("Paris", "city", 17, 22)]
    tagged = chunk_tagged_text(text, spans, 0, 14, TagNotation.xml)
    assert "<first_name>Alice</first_name>" in tagged and "<first_name>Bob</first_name>" in tagged
    assert "Paris" not in tagged


class _Fake:
    def __init__(self) -> None:
        self.map_calls = 0
        self.summary_calls = 0

    def generate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        if "summary" in (purpose or ""):
            self.summary_calls += 1
            return parser("rolling summary"), []
        self.map_calls += 1
        pairs = re.findall(r'- "([^"]+)" \(([^)]+)\)', prompt)
        reps = {"replacements": [{"original": v, "label": lab, "synthetic": v + "_S"} for v, lab in pairs]}
        return parser("```json\n" + json.dumps(reps) + "\n```"), []


def _line(s: str) -> str:
    return s + "x" * (100 - len(s) - 1) + "\n"


class TestGenerateReplacementMapRow:
    def test_fast_path_single_call(self) -> None:
        facade = _Fake()
        row = {
            COL_TEXT: "Alice here",
            COL_TAGGED_TEXT: "Alice here",
            COL_TAG_NOTATION: "xml",
            COL_ENTITY_EXAMPLES: "{}",
            COL_ENTITIES_FOR_REPLACE: [{"value": "Alice", "labels": ["name"], "labels_str": "name"}],
            COL_FINAL_ENTITIES: EntitiesSchema(entities=[]).model_dump(mode="json"),
        }
        # NB: fast path renders the single prompt; our stand-in lists entities so the fake maps them.
        params = WindowedReplaceParams(
            alias="r", single_call_prompt_template='MAP - "Alice" (name)', max_render_chars=1_000_000
        )
        out = generate_replacement_map_row(row, params, {"r": facade})
        assert facade.map_calls == 1 and facade.summary_calls == 0
        m = EntityReplacementMapSchema.model_validate(out[COL_REPLACEMENT_MAP])
        assert [(r.original, r.synthetic) for r in m.replacements] == [("Alice", "Alice_S")]

    def test_chunked_with_rolling_summary_and_dedupe(self) -> None:
        lines = [_line("") for _ in range(120)]
        lines[0] = _line("Alice")
        lines[50] = _line("Alice")  # recurs in window 2 -> must NOT be re-mapped
        lines[60] = _line("Bob")
        lines[119] = _line("Carol")
        text = "".join(lines)
        spans = []
        for val in ["Alice", "Bob", "Carol"]:
            for mobj in re.finditer(re.escape(val), text):
                spans.append(
                    {
                        "id": f"name_{mobj.start()}",
                        "value": val,
                        "label": "name",
                        "start_position": mobj.start(),
                        "end_position": mobj.end(),
                        "score": 1.0,
                        "source": "d",
                    }
                )
        final = EntitiesSchema.model_validate({"entities": spans}).model_dump(mode="json")

        facade = _Fake()
        row = {
            COL_TEXT: text,
            COL_TAGGED_TEXT: text,
            COL_TAG_NOTATION: "xml",
            COL_ENTITY_EXAMPLES: "{}",
            COL_ENTITIES_FOR_REPLACE: [
                {"value": v, "labels": ["name"], "labels_str": "name"} for v in ["Alice", "Bob", "Carol"]
            ],
            COL_FINAL_ENTITIES: final,
        }
        params = WindowedReplaceParams(
            alias="r", single_call_prompt_template=_SINGLE_PROMPT, max_render_chars=4000, safety_margin_chars=0
        )
        out = generate_replacement_map_row(row, params, {"r": facade})
        result = EntityReplacementMapSchema.model_validate(out[COL_REPLACEMENT_MAP])
        assert facade.map_calls == 3  # one per window with new entities
        assert facade.summary_calls == 2  # after windows 1 and 2, not the last
        assert sorted(r.original for r in result.replacements) == ["Alice", "Bob", "Carol"]
        assert sum(r.original == "Alice" for r in result.replacements) == 1  # deduped across chunks

    def test_missing_alias_raises(self) -> None:
        with pytest.raises(KeyError, match="not present in models"):
            generate_replacement_map_row(
                {COL_TEXT: "x"},
                WindowedReplaceParams(alias="r", single_call_prompt_template="x", max_render_chars=10),
                {},
            )
