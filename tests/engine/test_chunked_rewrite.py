# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for chunked (long-context) rewrite generation."""

from __future__ import annotations

import pytest

from anonymizer.engine.constants import (
    COL_FULL_REWRITE,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.rewrite.chunked_rewrite import (
    WindowedRewriteParams,
    _filter_disposition_to_chunk,
    generate_rewrite_row,
)
from anonymizer.engine.schemas import RewriteOutputSchema

_TEMPLATE = "REWRITE[" + COL_TAG_NOTATION + "] {{ " + COL_TAGGED_TEXT + " }}"


class _Fake:
    def __init__(self) -> None:
        self.rewrite_calls = 0
        self.summary_calls = 0

    def generate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        if "summary" in (purpose or ""):
            self.summary_calls += 1
            return parser("running summary"), []
        self.rewrite_calls += 1
        return parser('```json\n{"rewritten_text":"OUT%d"}\n```' % self.rewrite_calls), []


def _row(tagged: str) -> dict:
    return {
        COL_TEXT: tagged,
        COL_TAGGED_TEXT: tagged,
        COL_TAG_NOTATION: "xml",
        COL_REWRITE_DISPOSITION_BLOCK: [],
        COL_REPLACEMENT_MAP_FOR_PROMPT: {"replacements": []},
    }


def test_filter_disposition_to_chunk() -> None:
    block = [{"entity_value": "Alice"}, {"entity_value": "Bob"}, {"entity_value": "Carol"}]
    assert _filter_disposition_to_chunk(block, "... Alice and Carol ...") == [
        {"entity_value": "Alice"},
        {"entity_value": "Carol"},
    ]


class TestGenerateRewriteRow:
    def test_fast_path_single_call(self) -> None:
        facade = _Fake()
        params = WindowedRewriteParams(alias="w", single_call_prompt_template=_TEMPLATE, max_render_chars=1_000_000)
        out = generate_rewrite_row(_row("short tagged text"), params, {"w": facade})
        assert facade.rewrite_calls == 1 and facade.summary_calls == 0
        assert RewriteOutputSchema.model_validate(out[COL_FULL_REWRITE]).rewritten_text == "OUT1"

    def test_chunked_stitches_with_rolling_summary(self) -> None:
        tagged = ("X" * 4000 + "\n") * 3  # ~12k chars -> several windows
        facade = _Fake()
        params = WindowedRewriteParams(
            alias="w", single_call_prompt_template=_TEMPLATE, max_render_chars=4000, safety_margin_chars=0
        )
        out = generate_rewrite_row(_row(tagged), params, {"w": facade})
        text = RewriteOutputSchema.model_validate(out[COL_FULL_REWRITE]).rewritten_text
        assert facade.rewrite_calls > 1
        assert facade.summary_calls == facade.rewrite_calls - 1  # summary after each chunk except the last
        # stitched in order
        assert text.split("\n") == [f"OUT{i}" for i in range(1, facade.rewrite_calls + 1)]

    def test_missing_alias_raises(self) -> None:
        with pytest.raises(KeyError, match="not present in models"):
            generate_rewrite_row(_row("x"), WindowedRewriteParams(alias="w", single_call_prompt_template="x", max_render_chars=10), {})
