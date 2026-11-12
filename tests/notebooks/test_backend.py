# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anonymizer.engine.detection.postprocess import parse_raw_entities
from anonymizer.engine.local_inference.gliner2 import backend
from anonymizer.engine.local_inference.gliner2.backend import (
    _resolve_document_overlaps,
    detect_entities_for_texts,
    flatten_result,
    overlap_policy,
)


def _split_words(text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]:
    for match in re.finditer(r"\S+", text):
        yield match.group().lower() if lower else match.group(), match.start(), match.end()


class _FakeModel:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.processor = SimpleNamespace(word_splitter=_split_words)
        self.calls: list[tuple[list[str], list[str], dict[str, Any]]] = []

    def batch_extract_entities(
        self,
        texts: list[str],
        labels: list[str],
        **kwargs: Any,
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
    assert result == [{"text": "Alice", "label": "first_name", "start": 0, "end": 5, "score": 0.91}]


def test_overlap_policy_preserves_existing_flat_flag_contract() -> None:
    assert overlap_policy(True) == "disallow"
    assert overlap_policy(False) == "longest"


def _entity(text: str, start: int, end: int, score: float = 0.8) -> dict[str, Any]:
    return {"text": text, "label": "pii", "start": start, "end": end, "score": score}


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
def test_overlap_contract_retains_different_labels_for_the_same_span(policy: str) -> None:
    name = _entity("Alice", 0, 5, 0.9)
    identifier = {**_entity("Alice", 0, 5, 0.8), "label": "identifier"}

    assert _resolve_document_overlaps([name, identifier], policy) == [identifier, name]


def test_disallow_equal_score_tie_uses_confidence_start_end_ranking() -> None:
    outer = _entity("outer", 2, 10, 0.8)
    inner = _entity("inner", 5, 8, 0.8)

    assert _resolve_document_overlaps([inner, outer], "disallow") == [outer]


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
    assert result == [[{"text": "Alice", "label": "name", "start": 5, "end": 10, "score": 0.9}]]
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
    assert result == [{"text": value, "label": label, "start": start, "end": len(text), "score": 0.9}]
    parsed = parse_raw_entities(raw_response=json.dumps({"entities": result}), text=text)
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
    assert result == [{"text": "May", "label": "first_name", "start": 4, "end": 7, "score": 0.9}]
    assert text[result[0]["start"] : result[0]["end"]] == result[0]["text"]


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
    assert [entity["text"] for entity in result] == ["May"]


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
    assert [(e["text"], e["start"], e["end"]) for e in result] == [("İpek", 2, 6), ("José", 15, 19)]
    assert all(text[e["start"] : e["end"]] == e["text"] for e in result)


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
    assert [(row[0]["text"], row[0]["start"], row[0]["end"]) for row in result] == [("May", 4, 7), ("June", 5, 9)]


@pytest.mark.parametrize(
    ("requested", "device", "expected"),
    [
        ("auto", "cuda", "bfloat16"),
        ("auto", "mps", "float32"),
        ("auto", "cpu", "float32"),
        ("FP32", "cuda", "float32"),
        ("bf16", "cpu", "bfloat16"),
    ],
)
def test_resolve_dtype_defaults_to_bf16_on_cuda_only(requested: str, device: str, expected: str) -> None:
    assert backend.resolve_dtype(requested, device) == expected


@pytest.mark.parametrize(
    ("requested", "device", "expected"),
    [
        ("auto", "cuda", ("disentangled_flash", False)),
        ("auto", "mps", ("standard", False)),
        ("auto", "cpu", ("standard", False)),
        ("standard", "cuda", ("standard", True)),
        ("Disentangled_Flash", "cpu", ("disentangled_flash", True)),
    ],
)
def test_resolve_attention_defaults_to_flash_on_cuda_only(
    requested: str, device: str, expected: tuple[str, bool]
) -> None:
    assert backend.resolve_attention(requested, device) == expected


@pytest.mark.parametrize(
    "call", [lambda: backend.resolve_dtype("fp16", "cuda"), lambda: backend.resolve_attention("sdpa", "cuda")]
)
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


def test_cuda_default_loads_bf16_with_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _patch_loading(monkeypatch, flash_error=None)
    loaded = backend.load_model("cuda")
    assert (loaded.dtype, loaded.attention) == ("bfloat16", "disentangled_flash")
    assert loaded.model.flash_enabled
    assert loads == ["bfloat16"]


def test_cuda_default_falls_back_to_standard_attention_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    loads = _patch_loading(monkeypatch, flash_error=RuntimeError("triton unavailable"))
    loaded = backend.load_model("cuda")
    assert (loaded.dtype, loaded.attention) == ("bfloat16", "standard")
    assert loads == ["bfloat16", "bfloat16"]
    assert "falling back to standard attention" in caplog.text


def test_explicit_flash_that_cannot_load_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_loading(monkeypatch, flash_error=ImportError("no disentangled_flash"))
    with pytest.raises(RuntimeError, match=backend.ATTENTION_ENV):
        backend.load_model("cuda", attention="disentangled_flash")


def test_cpu_default_is_float32_standard_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = _patch_loading(monkeypatch, flash_error=None)
    loaded = backend.load_model("cpu")
    assert (loaded.dtype, loaded.attention) == ("float32", "standard")
    assert not loaded.model.flash_enabled
    assert loads == ["float32"]
