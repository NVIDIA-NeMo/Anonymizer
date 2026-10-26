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
from anonymizer.notebooks.local_inference.gliner2.backend import (
    _prepare_transformers_v4_snapshot,
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


def test_pinned_snapshot_adapts_v5_tokenizer_field_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    original = {"extra_special_tokens": ["[P]", "[E]"], "tokenizer_class": "DebertaV2Tokenizer"}
    (source / "tokenizer_config.json").write_text(json.dumps(original))
    (source / "model.safetensors").write_bytes(b"weights")

    prepared = _prepare_transformers_v4_snapshot(source, cache_root=tmp_path / "cache")
    adapted = json.loads((prepared / "tokenizer_config.json").read_text())

    assert adapted["additional_special_tokens"] == ["[P]", "[E]"]
    assert "extra_special_tokens" not in adapted
    assert json.loads((source / "tokenizer_config.json").read_text()) == original


def test_pinned_snapshot_dereferences_hugging_face_cache_symlinks(tmp_path: Path) -> None:
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "tokenizer").write_text(json.dumps({"tokenizer_class": "DebertaV2Tokenizer"}))
    (blobs / "weights").write_bytes(b"weights")
    source = tmp_path / "snapshots" / "revision"
    source.mkdir(parents=True)
    (source / "tokenizer_config.json").symlink_to(Path("../../blobs/tokenizer"))
    (source / "model.safetensors").symlink_to(Path("../../blobs/weights"))

    prepared = _prepare_transformers_v4_snapshot(source, cache_root=tmp_path / "cache")

    assert not (prepared / "tokenizer_config.json").is_symlink()
    assert not (prepared / "model.safetensors").is_symlink()
    assert (prepared / "model.safetensors").read_bytes() == b"weights"


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


@pytest.mark.parametrize("suffix", [".", ".."])
def test_synthetic_suffix_is_removed_without_losing_the_source_entity(suffix: str) -> None:
    text = "Ask May"
    model = _FakeModel(
        [
            {
                "entities": {
                    "first_name": [{"text": "May" + suffix, "confidence": 0.9, "start": 4, "end": 7 + len(suffix)}]
                }
            }
        ]
    )
    result = detect_entities_for_texts(
        model,
        [text],
        ["first_name"],
        threshold=0.3,
        chunk_length=384,
        overlap=128,
        flat_ner=False,
        inference_batch_size=1,
    )[0]
    assert result == [{"text": "May", "label": "first_name", "start": 4, "end": 7, "score": 0.9}]
    parsed = parse_raw_entities(raw_response=json.dumps({"entities": result}), text=text)
    assert [(entity.value, entity.start_position, entity.end_position) for entity in parsed] == [("May", 4, 7)]


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
