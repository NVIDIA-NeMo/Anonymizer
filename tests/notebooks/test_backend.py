# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anonymizer.notebooks.local_inference.gliner2.backend import (
    _prepare_transformers_v4_snapshot,
    _resolve_document_overlaps,
    detect_entities_for_texts,
    flatten_result,
    overlap_policy,
)


class _FakeModel:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
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


def test_detect_entities_remaps_and_deduplicates_overlapping_character_chunks() -> None:
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
        chunk_length=10,
        overlap=5,
        flat_ner=False,
        inference_batch_size=2,
    )
    assert result == [[{"text": "Alice", "label": "name", "start": 5, "end": 10, "score": 0.9}]]
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
