# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from anonymizer.engine.detection.postprocess import parse_raw_entities
from anonymizer.engine.local_inference.gliner2 import backend
from anonymizer.engine.local_inference.gliner2.backend import (
    _best_disjoint,
    _keep_outermost,
    _maximum_score_non_overlapping,
    _resolve_document_overlaps,
    detect_entities_for_texts,
    flatten_result,
    overlap_policy,
)
from anonymizer.engine.local_inference.gliner2.contract import DetectedEntity, Device, Label, OverlapPolicy, Span


def _split_words(text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
    for match in re.finditer(r"\S+", text):
        yield match.group().lower() if lower else match.group(), match.start(), match.end()


def _wire(entity: DetectedEntity) -> dict[str, Any]:
    return {
        "text": entity.text,
        "label": entity.label,
        "start": entity.span.start,
        "end": entity.span.end,
        "score": entity.score,
    }


class _FakeModel:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.processor = SimpleNamespace(word_splitter=_split_words)
        self.calls: list[tuple[list[str], list[str], dict[str, Any]]] = []

    def batch_extract_entities(
        self,
        texts: list[str],
        labels: list[str],
        **kwargs: object,
    ) -> list[dict[str, Any]]:
        self.calls.append((texts, labels, kwargs))
        return self.results


def test_flatten_result_converts_label_keyed_entities() -> None:
    result = flatten_result(
        {
            "entities": {
                "first_name": [
                    {"text": "Alice", "confidence": 0.91, "start": 0, "end": 5},
                ]
            }
        }
    )
    assert [_wire(entity) for entity in result] == [
        {"text": "Alice", "label": "first_name", "start": 0, "end": 5, "score": 0.91}
    ]


def test_flat_ner_maps_to_disallow_overlap_policy() -> None:
    assert overlap_policy(True) == "disallow"
    assert overlap_policy(False) == "longest"


def _entity(text: str, start: int, end: int, score: float = 0.8, label: str = "pii") -> DetectedEntity:
    return DetectedEntity(span=Span(start, end), label=Label(label), text=text, score=score)


def test_longest_overlap_contract_handles_disjoint_nested_crossing_and_same_span() -> None:
    entities = [
        _entity("outer", 0, 10, 0.7),
        _entity("nested", 2, 5, 0.99),
        _entity("crossing", 8, 14, 0.6),
        _entity("disjoint", 20, 25, 0.5),
        _entity("same-span-lower", 20, 25, 0.4),
    ]
    assert _resolve_document_overlaps(entities, "longest") == [
        entities[0],
        entities[2],
        entities[3],
    ]


def test_disallow_overlap_contract_uses_maximum_total_score() -> None:
    outer = _entity("outer", 0, 10, 0.9)
    left = _entity("left", 0, 5, 0.6)
    right = _entity("right", 5, 10, 0.6)
    assert _resolve_document_overlaps([outer, left, right], "disallow") == [left, right]


@pytest.mark.parametrize("policy", ["longest", "disallow"])
def test_overlap_contract_retains_different_labels_for_the_same_span(policy: OverlapPolicy) -> None:
    name = _entity("Alice", 0, 5, 0.9)
    identifier = _entity("Alice", 0, 5, 0.8, label="identifier")

    assert _resolve_document_overlaps([name, identifier], policy) == [identifier, name]


def test_disallow_equal_score_tie_uses_confidence_start_end_ranking() -> None:
    outer = _entity("outer", 2, 10, 0.8)
    inner = _entity("inner", 5, 8, 0.8)

    assert _resolve_document_overlaps([inner, outer], "disallow") == [outer]


def _brute_force_best(candidates: list[DetectedEntity]) -> set[DetectedEntity]:
    """Maximum total score, then most picks, then smallest sorted rank: the documented preference order."""
    best: tuple[float, int, tuple[tuple[float, int, int], ...]] | None = None
    best_picks: set[DetectedEntity] = set()
    for mask in range(1 << len(candidates)):
        picks = [candidate for index, candidate in enumerate(candidates) if mask >> index & 1]
        spans = sorted(pick.span for pick in picks)
        if any(left.end > right.start for left, right in zip(spans, spans[1:], strict=False)):
            continue
        key = (-sum(pick.score for pick in picks), -len(picks), tuple(sorted(pick.rank for pick in picks)))
        if best is None or key < best:
            best, best_picks = key, set(picks)
    return best_picks


@pytest.mark.parametrize("seed", range(40))
def test_maximum_score_selection_matches_brute_force_including_ties(seed: int) -> None:
    rng = random.Random(seed)
    boundaries = {
        (start, start + rng.randint(1, 6)) for start in (rng.randint(0, 14) for _ in range(rng.randint(0, 9)))
    }
    # Scores are multiples of 1/8 so float sums are exact and ties are common.
    candidates = [
        DetectedEntity(Span(start, end), Label("pii"), "x" * (end - start), rng.choice([0.25, 0.5, 0.5, 0.75, 1.0]))
        for start, end in sorted(boundaries)
    ]
    assert set(_maximum_score_non_overlapping(candidates)) == _brute_force_best(candidates)


def test_keep_outermost_and_best_disjoint_are_independent_policies() -> None:
    outer = _entity("outer", 0, 10, 0.5)
    left = _entity("left", 0, 5, 0.4)
    right = _entity("right", 5, 10, 0.4)
    assert _keep_outermost([outer, left, right]) == [outer]
    assert _best_disjoint([outer, left, right]) == [left, right]


def test_detect_entities_remaps_and_deduplicates_overlapping_word_chunks() -> None:
    model = _FakeModel(
        [
            {"entities": {"name": [{"text": "Alice", "confidence": 0.8, "start": 5, "end": 10}]}},
            {"entities": {"name": [{"text": "Alice", "confidence": 0.9, "start": 0, "end": 5}]}},
        ]
    )
    result = detect_entities_for_texts(
        model,
        ["xxxx Alice yyyy"],
        ["name"],
        threshold=0.3,
        chunk_length=2,
        overlap=1,
        flat_ner=False,
        inference_batch_size=2,
    )
    assert [[_wire(e) for e in entities] for entities in result] == [
        [{"text": "Alice", "label": "name", "start": 5, "end": 10, "score": 0.9}]
    ]
    assert model.calls[0][0] == ["xxxx Alice", "Alice yyyy"]
    assert model.calls[0][2]["include_spans"] is True
    assert model.calls[0][2]["include_confidence"] is True
    assert model.calls[0][2]["overlap_policy"] == "longest"


@pytest.mark.parametrize(
    ("chunk_length", "overlap", "message"),
    [(0, 0, "chunk_length"), (10, -1, "overlap"), (10, 10, "less than")],
)
def test_detect_entities_rejects_invalid_chunk_settings(
    chunk_length: int,
    overlap: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        detect_entities_for_texts(
            _FakeModel([]),
            ["text"],
            ["name"],
            threshold=0.3,
            chunk_length=chunk_length,
            overlap=overlap,
            flat_ner=False,
            inference_batch_size=1,
        )


@pytest.mark.parametrize(
    ("text", "value", "start", "label"),
    [("Ask May", "May", 4, "first_name"), ("Visit https://example.org", "https://example.org", 6, "url")],
)
def test_synthetic_suffix_is_removed_without_losing_the_source_entity(
    text: str, value: str, start: int, label: str
) -> None:
    model = _FakeModel(
        [{"entities": {label: [{"text": value + ".", "confidence": 0.9, "start": start, "end": len(text) + 1}]}}]
    )
    result = detect_entities_for_texts(
        model,
        [text],
        [label],
        threshold=0.3,
        chunk_length=384,
        overlap=128,
        flat_ner=False,
        inference_batch_size=1,
    )[0]
    wire = [_wire(entity) for entity in result]
    assert wire == [{"text": value, "label": label, "start": start, "end": len(text), "score": 0.9}]
    parsed = parse_raw_entities(raw_response=json.dumps({"entities": wire}), text=text)
    assert [(entity.value, entity.start_position, entity.end_position) for entity in parsed] == [
        (value, start, len(text))
    ]


def test_boundary_punctuation_does_not_consume_the_next_source_character() -> None:
    text = "Ask May today"
    model = _FakeModel(
        [
            {"entities": {"first_name": [{"text": "May.", "confidence": 0.8, "start": 4, "end": 8}]}},
            {"entities": {"first_name": [{"text": "May", "confidence": 0.9, "start": 0, "end": 3}]}},
        ]
    )
    result = detect_entities_for_texts(
        model,
        [text],
        ["first_name"],
        threshold=0.3,
        chunk_length=2,
        overlap=1,
        flat_ner=False,
        inference_batch_size=2,
    )[0]
    assert model.calls[0][0] == ["Ask May", "May today"]
    assert [_wire(entity) for entity in result] == [
        {"text": "May", "label": "first_name", "start": 4, "end": 7, "score": 0.9}
    ]
    assert text[result[0].span.start : result[0].span.end] == result[0].text


def test_synthetic_only_prediction_does_not_remove_valid_entities() -> None:
    model = _FakeModel(
        [
            {
                "entities": {
                    "first_name": [
                        {"text": "May", "confidence": 0.9, "start": 4, "end": 7},
                        {"text": ".", "confidence": 0.8, "start": 7, "end": 8},
                    ]
                }
            }
        ]
    )
    result = detect_entities_for_texts(
        model,
        ["Ask May"],
        ["first_name"],
        threshold=0.3,
        chunk_length=384,
        overlap=128,
        flat_ner=False,
        inference_batch_size=1,
    )[0]
    assert [entity.text for entity in result] == ["May"]


def test_word_windows_preserve_unicode_whitespace_and_document_coordinates() -> None:
    text = "  İpek\t\nvisits José  today "
    model = _FakeModel(
        [
            {"entities": {"first_name": [{"text": "İpek", "confidence": 0.9, "start": 0, "end": 4}]}},
            {"entities": {"first_name": [{"text": "José", "confidence": 0.9, "start": 7, "end": 11}]}},
            {"entities": {"first_name": [{"text": "José", "confidence": 0.8, "start": 0, "end": 4}]}},
        ]
    )
    result = detect_entities_for_texts(
        model,
        [text],
        ["first_name"],
        threshold=0.3,
        chunk_length=2,
        overlap=1,
        flat_ner=False,
        inference_batch_size=3,
    )[0]
    assert model.calls[0][0] == ["İpek\t\nvisits", "visits José", "José  today"]
    assert [(e.text, *e.span) for e in result] == [("İpek", 2, 6), ("José", 15, 19)]
    assert all(text[e.span.start : e.span.end] == e.text for e in result)


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_empty_or_whitespace_source_skips_inference(text: str) -> None:
    model = _FakeModel([])
    assert detect_entities_for_texts(
        model,
        [text],
        ["name"],
        threshold=0.3,
        chunk_length=384,
        overlap=128,
        flat_ner=False,
        inference_batch_size=1,
    ) == [[]]
    assert model.calls == []


@pytest.mark.parametrize(
    ("source", "value", "start", "end"),
    [
        ("Ask May", "June", 4, 7),
        ("Ask May", "MayX", 4, 8),
        ("Ask May", "may", 4, 7),
        ("Ask May", " May", 4, 7),
        ("Ask May", "May..", 4, 9),
        ("Ask May", ".", 9, 10),
        ("Ask May.", "May..", 4, 9),
        ("Ask May", "May", -1, 3),
        ("Ask May", "May", 4, 4),
    ],
)
def test_unexplained_span_errors_still_fail_without_logging_values(
    source: str,
    value: str,
    start: int,
    end: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _FakeModel(
        [
            {
                "entities": {
                    "first_name": [
                        {"text": value, "confidence": 0.9, "start": start, "end": end},
                    ]
                }
            }
        ]
    )
    with pytest.raises(ValueError, match="inconsistent offsets") as error:
        detect_entities_for_texts(
            model,
            [source],
            ["first_name"],
            threshold=0.3,
            chunk_length=384,
            overlap=128,
            flat_ner=False,
            inference_batch_size=1,
        )
    assert value not in str(error.value)
    assert source not in str(error.value)
    assert value not in caplog.text


@pytest.mark.parametrize("value", ["", " \t\n "])
def test_blank_entity_text_is_rejected_even_for_a_whitespace_source_span(value: str) -> None:
    model = _FakeModel([{"entities": {"first_name": [{"text": value, "confidence": 0.9, "start": 3, "end": 6}]}}])
    with pytest.raises(ValueError, match="inconsistent offsets"):
        detect_entities_for_texts(
            model,
            ["Ask \t May"],
            ["first_name"],
            threshold=0.3,
            chunk_length=384,
            overlap=128,
            flat_ner=False,
            inference_batch_size=1,
        )


def test_batch_word_windows_keep_document_offsets_separate() -> None:
    model = _FakeModel(
        [
            {"entities": {"first_name": [{"text": "May.", "confidence": 0.9, "start": 4, "end": 8}]}},
            {"entities": {"first_name": [{"text": "June", "confidence": 0.9, "start": 5, "end": 9}]}},
        ]
    )
    result = detect_entities_for_texts(
        model,
        ["Ask May", "Call June"],
        ["first_name"],
        threshold=0.3,
        chunk_length=384,
        overlap=128,
        flat_ner=False,
        inference_batch_size=2,
    )
    assert [(row[0].text, *row[0].span) for row in result] == [("May", 4, 7), ("June", 5, 9)]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("auto", "float32"), ("FP32", "float32"), ("float32", "float32"), ("bf16", "bfloat16"), ("bfloat16", "bfloat16")],
)
def test_resolve_dtype_is_float32_unless_bfloat16_is_requested(requested: str, expected: str) -> None:
    assert backend.resolve_dtype(requested) == expected


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("auto", "standard"),
        ("standard", "standard"),
        ("disentangled_flash", "disentangled_flash"),
        ("Disentangled_Flash", "disentangled_flash"),
    ],
)
def test_resolve_attention_is_standard_unless_flash_is_requested(requested: str, expected: str) -> None:
    assert backend.resolve_attention(requested) == expected


@pytest.mark.parametrize("call", [lambda: backend.resolve_dtype("fp16"), lambda: backend.resolve_attention("sdpa")])
def test_unsupported_precision_or_attention_is_rejected(call: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported GLiNER2"):
        call()


class _LoadableModel:
    def __init__(self, *, flash_error: Exception | None) -> None:
        self.flash_error = flash_error
        self.flash_enabled = False

    def enable_disentangled_flash(self) -> None:
        if self.flash_error is not None:
            raise self.flash_error
        self.flash_enabled = True

    def batch_extract_entities(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"entities": {}}]


def _patch_loading(monkeypatch: pytest.MonkeyPatch, flash_error: Exception | None) -> list[str]:
    import huggingface_hub

    loads: list[str] = []

    def load_weights(snapshot: Path, device: str, dtype: str) -> _LoadableModel:
        loads.append(dtype)
        return _LoadableModel(flash_error=flash_error)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **_: "/snapshot")
    monkeypatch.setattr(backend, "_load_weights", load_weights)
    return loads


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_default_loads_float32_with_standard_attention(monkeypatch: pytest.MonkeyPatch, device: Device) -> None:
    loads = _patch_loading(monkeypatch, flash_error=None)
    loaded = backend.load_model(device)
    assert (loaded.dtype, loaded.attention) == ("float32", "standard")
    assert not cast(_LoadableModel, loaded.model).flash_enabled
    assert loads == ["float32"]


def test_flash_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_loading(monkeypatch, flash_error=None)
    loaded = backend.load_model("cuda", attention="disentangled_flash")
    assert loaded.attention == "disentangled_flash"
    assert cast(_LoadableModel, loaded.model).flash_enabled


def test_bfloat16_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _patch_loading(monkeypatch, flash_error=None)
    loaded = backend.load_model("cuda", dtype="bfloat16")
    assert loaded.dtype == "bfloat16"
    assert loads == ["bfloat16"]


def test_flash_that_cannot_load_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_loading(monkeypatch, flash_error=ImportError("no disentangled_flash"))
    with pytest.raises(RuntimeError, match=backend.ATTENTION_ENV):
        backend.load_model("cuda", attention="disentangled_flash")
