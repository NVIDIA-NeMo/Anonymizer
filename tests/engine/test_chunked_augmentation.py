# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for windowed LLM augmentation.

Pure helpers (windowing, per-window inputs, merge) are tested directly; the
per-window dispatch is tested via a fake ``ModelFacade`` that records calls and
replays canned responses (mirrors the chunked-validation tests).
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

import pytest

from anonymizer.engine.constants import (
    COL_AUGMENTATION_FAILED_WINDOWS,
    COL_AUGMENTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TEXT,
    COL_VALIDATED_SEED_ENTITIES,
)
from anonymizer.engine.detection.chunked_augmentation import (
    WindowedAugmentationParams,
    augment_row,
    build_window_inputs,
    iter_windows,
    merge_augmented,
    render_augment_prompt,
)
from anonymizer.engine.detection.postprocess import EntitySpan, TagNotation
from anonymizer.engine.schemas import AugmentedEntitiesSchema, EntitiesSchema

# Small stand-in for the real (multi-kB) augment prompt so window sizes are
# controllable in tests. References the same placeholders the real prompt uses.
_TEMPLATE = "AUG[{{ _tag_notation }}] {{ _initial_tagged_text }} || SEEDS: {{ _seed_entities_json }}"


class FakeFacade:
    """Records invocations and replays canned responses through the recipe parser."""

    def __init__(self, response: dict | str | Callable[[str], dict | str]) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def generate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "purpose": purpose})
        response = self._response
        if callable(response):
            response = response(prompt)
        raw = response if isinstance(response, str) else f"```json\n{json.dumps(response)}\n```"
        return parser(raw), []


def _span(value: str, label: str, start: int, end: int) -> EntitySpan:
    return EntitySpan(
        entity_id=f"{label}_{start}_{end}",
        value=value,
        label=label,
        start_position=start,
        end_position=end,
        score=1.0,
        source="detector",
    )


def _aug(*pairs: tuple[str, str]) -> AugmentedEntitiesSchema:
    return AugmentedEntitiesSchema.model_validate({"entities": [{"value": v, "label": lab} for v, lab in pairs]})


def _params(max_render_chars: int, **kw: Any) -> WindowedAugmentationParams:
    return WindowedAugmentationParams(alias="aug", prompt_template=_TEMPLATE, max_render_chars=max_render_chars, **kw)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestIterWindows:
    def test_tiles_with_overlap(self) -> None:
        assert iter_windows(10000, window=4000, overlap=1000) == [(0, 4000), (3000, 7000), (6000, 10000)]

    def test_single_window_when_text_fits(self) -> None:
        assert iter_windows(500, window=4000, overlap=1000) == [(0, 500)]

    def test_empty(self) -> None:
        assert iter_windows(0, window=4000, overlap=1000) == []


class TestMergeAugmented:
    def test_dedupes_by_value_and_label_case_insensitively(self) -> None:
        first = _aug(("Alice", "first_name"), ("bob", "first_name"))
        second = AugmentedEntitiesSchema.model_validate(
            {
                "entities": [
                    {"value": "alice", "label": "first_name"},  # case-dup of Alice -> dropped
                    {"value": "Alice", "label": "city"},  # same value, different label -> kept
                    {"value": "   ", "label": "first_name"},  # blank after strip -> dropped
                ]
            }
        )
        merged = merge_augmented([first, second])
        pairs = {(e.value, e.label) for e in merged.entities}
        assert pairs == {("Alice", "first_name"), ("bob", "first_name"), ("Alice", "city")}


class TestBuildWindowInputs:
    def test_tags_only_in_window_spans_rebased(self) -> None:
        text = "Alice met Bob in Paris"  # Alice 0-5, Bob 10-13, Paris 17-22
        spans = [
            _span("Alice", "first_name", 0, 5),
            _span("Bob", "first_name", 10, 13),
            _span("Paris", "city", 17, 22),
        ]
        tagged, seed_json = build_window_inputs(text=text, all_spans=spans, start=0, end=14, notation=TagNotation.xml)
        assert "<first_name>Alice</first_name>" in tagged
        assert "<first_name>Bob</first_name>" in tagged
        assert "Paris" not in tagged  # out-of-window span excluded
        seeds = json.loads(seed_json)
        assert {s["value"] for s in seeds} == {"Alice", "Bob"}
        assert all(0 <= s["start_position"] < s["end_position"] <= 14 for s in seeds)  # window-local offsets


def test_render_includes_notation_and_inputs() -> None:
    rendered = render_augment_prompt(
        template=_TEMPLATE, tagged_text="TAGGED", seed_entities_json="[]", notation=TagNotation.bracket
    )
    assert "TAGGED" in rendered
    assert "bracket" in rendered


# ---------------------------------------------------------------------------
# augment_row
# ---------------------------------------------------------------------------


class TestAugmentRowFastPath:
    def test_single_call_when_under_budget(self) -> None:
        facade = FakeFacade({"entities": [{"value": "Alice", "label": "first_name"}]})
        row = {
            COL_TEXT: "Alice in Paris",
            COL_INITIAL_TAGGED_TEXT: "<first_name>Alice</first_name> in Paris",
            COL_SEED_ENTITIES_JSON: "[]",
            COL_TAG_NOTATION: "xml",
        }
        out = augment_row(row, _params(max_render_chars=1_000_000), {"aug": facade})
        assert len(facade.calls) == 1
        result = AugmentedEntitiesSchema.model_validate(out[COL_AUGMENTED_ENTITIES])
        assert [(e.value, e.label) for e in result.entities] == [("Alice", "first_name")]


class TestAugmentRowWindowed:
    def test_multiple_windows_unioned(self) -> None:
        text = ("A" * 5000) + ("B" * 5000)  # 10k chars -> forces several windows
        counter = itertools.count()
        # Each call returns a distinct entity so the union size == number of windows.
        facade = FakeFacade(lambda _prompt: {"entities": [{"value": f"v{next(counter)}", "label": "name"}]})
        row = {
            COL_TEXT: text,
            COL_INITIAL_TAGGED_TEXT: text,  # full render exceeds budget -> windowed path
            COL_SEED_ENTITIES_JSON: "[]",
            COL_VALIDATED_SEED_ENTITIES: EntitiesSchema(entities=[]).model_dump(mode="json"),
            COL_TAG_NOTATION: "xml",
        }
        out = augment_row(
            row, _params(max_render_chars=4000, safety_margin_chars=0, overlap_chars=1000), {"aug": facade}
        )
        assert len(facade.calls) > 1
        result = AugmentedEntitiesSchema.model_validate(out[COL_AUGMENTED_ENTITIES])
        assert len(result.entities) == len(facade.calls)

    def test_one_failing_window_is_skipped_not_fatal(self) -> None:
        # A single window's LLM failure must not drop the whole record: the row
        # still completes and the other windows' entities survive.
        text = ("A" * 5000) + ("B" * 5000)  # forces multiple windows
        counter = itertools.count()

        def resp(_prompt):
            i = next(counter)
            if i == 1:  # exactly one window raises (order is nondeterministic under the pool)
                raise ValueError("simulated unparseable model output")
            return {"entities": [{"value": f"v{i}", "label": "name"}]}

        facade = FakeFacade(resp)
        row = {
            COL_TEXT: text,
            COL_INITIAL_TAGGED_TEXT: text,
            COL_SEED_ENTITIES_JSON: "[]",
            COL_VALIDATED_SEED_ENTITIES: EntitiesSchema(entities=[]).model_dump(mode="json"),
            COL_TAG_NOTATION: "xml",
        }
        # Should NOT raise despite one window failing.
        out = augment_row(
            row, _params(max_render_chars=4000, safety_margin_chars=0, overlap_chars=1000), {"aug": facade}
        )
        result = AugmentedEntitiesSchema.model_validate(out[COL_AUGMENTED_ENTITIES])
        assert len(facade.calls) > 1  # multiple windows attempted
        assert len(result.entities) == len(facade.calls) - 1  # exactly one window dropped
        assert out[COL_AUGMENTATION_FAILED_WINDOWS] == 1  # the skip is recorded for degraded-flagging

    def test_missing_alias_raises(self) -> None:
        with pytest.raises(KeyError, match="not present in models"):
            augment_row({COL_TEXT: "x"}, _params(max_render_chars=10), {})
