# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for windowed latent-entity detection."""

from __future__ import annotations

import itertools
import json
from typing import Any, Callable

import pytest

from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.detection.chunked_latent import (
    WindowedLatentParams,
    latent_row,
    merge_latent,
    render_latent_prompt,
)
from anonymizer.engine.detection.postprocess import TagNotation
from anonymizer.engine.schemas import EntitiesSchema, LatentEntitiesSchema

# Build from the real column constants so the template can't drift from them.
_TEMPLATE = "LATENT[{{ " + COL_TAG_NOTATION + " }}] {{ " + COL_TAGGED_TEXT + " }}"


def _latent(label: str, value: str) -> dict[str, Any]:
    return {
        "category": "latent_identifier",
        "label": label,
        "value": value,
        "confidence": "high",
        "evidence": [f"context mentioning {value}"],
        "rationale": "The surrounding text strongly implies this attribute about the subject.",
    }


def _latent_schema(*pairs: tuple[str, str]) -> LatentEntitiesSchema:
    return LatentEntitiesSchema.model_validate({"latent_entities": [_latent(lab, val) for lab, val in pairs]})


class FakeFacade:
    def __init__(self, response: dict | str | Callable[[str], dict | str]) -> None:
        self._response = response
        self.calls: list[str | None] = []

    def generate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        self.calls.append(purpose)
        response = self._response
        if callable(response):
            response = response(prompt)
        raw = response if isinstance(response, str) else f"```json\n{json.dumps(response)}\n```"
        return parser(raw), []


def _params(max_render_chars: int, **kw: Any) -> WindowedLatentParams:
    return WindowedLatentParams(alias="lat", prompt_template=_TEMPLATE, max_render_chars=max_render_chars, **kw)


class TestMergeLatent:
    def test_dedupes_by_label_and_value(self) -> None:
        first = _latent_schema(("employer", "Acme"), ("home_location", "Boston"))
        second = _latent_schema(("employer", "acme"), ("employer", "Globex"))  # acme dup, Globex new
        merged = merge_latent([first, second])
        pairs = {(e.label, e.value) for e in merged.latent_entities}
        assert pairs == {("employer", "Acme"), ("home_location", "Boston"), ("employer", "Globex")}


def test_render_includes_notation_and_text() -> None:
    rendered = render_latent_prompt(template=_TEMPLATE, tagged_text="TAGGED", notation=TagNotation.xml)
    assert "TAGGED" in rendered
    assert "xml" in rendered


class TestLatentRowFastPath:
    def test_single_call_when_under_budget(self) -> None:
        facade = FakeFacade({"latent_entities": [_latent("employer", "Acme")]})
        row = {
            COL_TEXT: "She works there",
            COL_TAGGED_TEXT: "She works there",
            COL_TAG_NOTATION: "xml",
        }
        out = latent_row(row, _params(max_render_chars=1_000_000), {"lat": facade})
        assert len(facade.calls) == 1
        result = LatentEntitiesSchema.model_validate(out[COL_LATENT_ENTITIES])
        assert [(e.label, e.value) for e in result.latent_entities] == [("employer", "Acme")]


class TestLatentRowWindowed:
    def test_multiple_windows_unioned(self) -> None:
        text = ("A" * 5000) + ("B" * 5000)
        counter = itertools.count()
        facade = FakeFacade(lambda _p: {"latent_entities": [_latent("employer", f"Org{next(counter)}")]})
        row = {
            COL_TEXT: text,
            COL_TAGGED_TEXT: text,  # full render exceeds budget -> windowed
            COL_DETECTED_ENTITIES: EntitiesSchema(entities=[]).model_dump(mode="json"),
            COL_TAG_NOTATION: "xml",
        }
        out = latent_row(
            row, _params(max_render_chars=4000, safety_margin_chars=0, overlap_chars=1000), {"lat": facade}
        )
        assert len(facade.calls) > 1
        result = LatentEntitiesSchema.model_validate(out[COL_LATENT_ENTITIES])
        assert len(result.latent_entities) == len(facade.calls)

    def test_missing_alias_raises(self) -> None:
        with pytest.raises(KeyError, match="not present in models"):
            latent_row({COL_TEXT: "x"}, _params(max_render_chars=10), {})
