# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for windowed (chunked) GLiNER seed detection.

Pure helpers (rebasing, boundary-span dropping, payload serialization) are tested
directly. The per-window dispatch is tested via a fake detector facade that
simulates a NER endpoint over the *submitted window text*: it emits every full
occurrence of a target string plus a truncated prefix/suffix when an entity is
cut by the window edge — exercising the offset-rebasing and boundary handling.
"""

from __future__ import annotations

import json
from typing import Any

from anonymizer.engine.constants import COL_RAW_DETECTED, COL_TEXT
from anonymizer.engine.detection.chunked_detection import (
    WindowedDetectionParams,
    detect_row,
    drop_boundary_spans,
    rebase_spans,
    spans_to_detector_payload,
)
from anonymizer.engine.detection.postprocess import EntitySpan


def _span(value: str, start: int, end: int, label: str = "person") -> EntitySpan:
    return EntitySpan(
        entity_id=f"{label}_{start}_{end}",
        value=value,
        label=label,
        start_position=start,
        end_position=end,
        score=1.0,
        source="detector",
    )


# ---------------------------------------------------------------------------
# Fake detector facade
# ---------------------------------------------------------------------------


class FakeDetectorFacade:
    """Simulates a NER endpoint over the submitted window text.

    Emits window-local offsets for every full occurrence of each target, plus a
    truncated prefix at the window's right edge / suffix at the left edge to mimic
    an entity straddling a window boundary.
    """

    def __init__(self, targets: list[str], label: str = "person") -> None:
        self.targets = targets
        self.label = label
        self.calls: list[dict[str, Any]] = []

    def generate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        self.calls.append({"prompt": prompt, "purpose": purpose})
        wt = prompt
        ents: list[dict[str, Any]] = []
        for t in self.targets:
            i = wt.find(t)
            while i != -1:
                ents.append({"text": t, "label": self.label, "start": i, "end": i + len(t), "score": 0.99})
                i = wt.find(t, i + 1)
            for k in range(len(t) - 1, 2, -1):  # truncated prefix at right edge
                if wt.endswith(t[:k]):
                    ents.append(
                        {"text": t[:k], "label": self.label, "start": len(wt) - k, "end": len(wt), "score": 0.5}
                    )
                    break
            for k in range(len(t) - 1, 2, -1):  # truncated suffix at left edge
                if wt.startswith(t[-k:]):
                    ents.append({"text": t[-k:], "label": self.label, "start": 0, "end": k, "score": 0.5})
                    break
        raw = json.dumps({"entities": ents})
        return parser(raw), []


def _place(buf: list[str], s: str, pos: int) -> None:
    for j, ch in enumerate(s):
        buf[pos + j] = ch


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_rebase_spans_shifts_offsets():
    out = rebase_spans([_span("X", 5, 7)], 100)
    assert (out[0].start_position, out[0].end_position) == (105, 107)
    assert out[0].value == "X"


def test_rebase_spans_zero_offset_is_noop():
    spans = [_span("X", 5, 7)]
    assert rebase_spans(spans, 0) == spans


def test_drop_boundary_spans_drops_artificial_edges_only():
    text_len = 1000
    spans = [
        _span("left-edge", 200, 260),  # touches left edge (window_start=200)
        _span("right-edge", 540, 600),  # touches right edge (window_end=600)
        _span("interior", 300, 320),  # safe
    ]
    kept = drop_boundary_spans(spans, window_start=200, window_end=600, text_len=text_len)
    assert [s.value for s in kept] == ["interior"]


def test_drop_boundary_spans_keeps_true_document_edges():
    # window_start == 0 (true doc start) and window_end == text_len (true doc end):
    # spans touching those are real, not truncated, so they are kept.
    spans = [_span("doc-start", 0, 40), _span("doc-end", 460, 500)]
    kept = drop_boundary_spans(spans, window_start=0, window_end=500, text_len=500)
    assert [s.value for s in kept] == ["doc-start", "doc-end"]


def test_spans_to_detector_payload_roundtrips_detector_shape():
    payload = json.loads(spans_to_detector_payload([_span("Alice", 3, 8, label="first_name")]))
    assert payload == {"entities": [{"text": "Alice", "label": "first_name", "start": 3, "end": 8, "score": 1.0}]}


# ---------------------------------------------------------------------------
# detect_row: fast path
# ---------------------------------------------------------------------------


def test_fast_path_single_call_passes_raw_through():
    facade = FakeDetectorFacade(["Alice"])
    text = "hello Alice world"
    row = {COL_TEXT: text}
    params = WindowedDetectionParams(alias="det", max_render_chars=10_000)
    detect_row(row, params, {"det": facade})

    assert len(facade.calls) == 1  # one call, whole document
    assert facade.calls[0]["prompt"] == text
    payload = json.loads(row[COL_RAW_DETECTED])
    assert payload["entities"][0]["text"] == "Alice"
    assert text[payload["entities"][0]["start"] : payload["entities"][0]["end"]] == "Alice"


# ---------------------------------------------------------------------------
# detect_row: windowed path (the boundary-cut scenario)
# ---------------------------------------------------------------------------


def test_windowed_recovers_straddling_entity_and_dedupes_overlap():
    targets = ["Maria Garcia", "Bob Smith", "Acme Corporation"]
    buf = ["a"] * 12_000
    _place(buf, "Maria Garcia", 4_995)  # straddles the window-A boundary at 5000
    _place(buf, "Bob Smith", 4_200)  # inside overlap region [4000, 5000] -> seen twice
    _place(buf, "Acme Corporation", 9_100)  # only inside window C
    text = "".join(buf)

    facade = FakeDetectorFacade(targets)
    row = {COL_TEXT: text}
    # cap=5000, margin=0 -> window=5000; overlap=1000 -> windows [0,5000)[4000,9000)[8000,12000)
    params = WindowedDetectionParams(alias="det", max_render_chars=5_000, safety_margin_chars=0, overlap_chars=1_000)
    detect_row(row, params, {"det": facade})

    ents = json.loads(row[COL_RAW_DETECTED])["entities"]
    spans = {(e["start"], e["end"], e["text"]) for e in ents}

    # straddling entity recovered with correct GLOBAL offsets
    assert (4_995, 5_007, "Maria Garcia") in spans
    # overlap-region entity present exactly once (deduped across windows A and B)
    assert (4_200, 4_209, "Bob Smith") in spans
    assert sum(1 for e in ents if e["text"] == "Bob Smith") == 1
    # window-C-only entity present
    assert (9_100, 9_116, "Acme Corporation") in spans
    # every emitted span maps to a real, full target — no truncated partial leaked
    assert all(text[e["start"] : e["end"]] in targets for e in ents)
    assert len(ents) == 3


def test_windowed_emits_empty_entities_when_nothing_detected():
    facade = FakeDetectorFacade(["Nonexistent Name"])
    text = "b" * 12_000
    row = {COL_TEXT: text}
    params = WindowedDetectionParams(alias="det", max_render_chars=5_000, safety_margin_chars=0, overlap_chars=1_000)
    detect_row(row, params, {"det": facade})
    assert json.loads(row[COL_RAW_DETECTED]) == {"entities": []}
    assert len(facade.calls) >= 2  # actually windowed


def test_missing_alias_raises():
    params = WindowedDetectionParams(alias="det", max_render_chars=5_000)
    try:
        detect_row({COL_TEXT: "x"}, params, {})
    except KeyError as exc:
        assert "det" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError for missing alias")
