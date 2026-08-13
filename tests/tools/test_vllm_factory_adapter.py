# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the vLLM Factory detector protocol adapter."""

from __future__ import annotations

from pathlib import Path

from inference_service_compiler import vllm_factory_adapter as adapter

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"


def load_adapter():
    """Compatibility alias for the directly imported production module."""
    return adapter


def test_parse_detection_request_preserves_anonymizer_options() -> None:
    """The adapter accepts the detector extras emitted by DataDesigner."""
    request = adapter.parse_detection_request(
        {
            "model": "nvidia/gliner-pii",
            "messages": [{"role": "user", "content": "Ada Lovelace"}],
            "labels": ["person", "email"],
            "threshold": 0.42,
            "chunk_length": 256,
            "overlap": 64,
            "flat_ner": True,
        }
    )

    assert request.text == "Ada Lovelace"
    assert request.labels == ("person", "email")
    assert request.threshold == 0.42
    assert request.chunk_length == 256
    assert request.overlap == 64
    assert request.flat_ner is True


def test_parse_detection_request_accepts_label_free_health_check() -> None:
    """DataDesigner's generic model health check receives a valid empty result."""

    request = adapter.parse_detection_request(
        {
            "model": "nvidia/gliner-pii",
            "messages": [{"role": "user", "content": "health check"}],
        }
    )

    assert request.labels == ()


def test_merge_gliner_entities_restores_offsets_and_deduplicates_overlap() -> None:
    """Chunk-relative factory spans become stable document offsets."""
    adapter = load_adapter()
    chunks = [adapter.TextChunk("Alice met Bob", 0), adapter.TextChunk("Bob at NVIDIA", 10)]
    results = [
        [
            {"text": "Alice", "label": "person", "start": 0, "end": 5, "score": 0.9},
            {"text": "Bob", "label": "person", "start": 10, "end": 13, "score": 0.8},
        ],
        [
            {"text": "Bob", "label": "person", "start": 0, "end": 3, "score": 0.95},
            {"text": "NVIDIA", "label": "company", "start": 7, "end": 13, "score": 0.88},
        ],
    ]

    entities = adapter.merge_entities(
        plugin="deberta_gliner",
        chunks=chunks,
        results=results,
        flat_ner=True,
    )

    assert [entity.as_dict() for entity in entities] == [
        {"text": "Alice", "label": "person", "start": 0, "end": 5, "score": 0.9},
        {"text": "Bob", "label": "person", "start": 10, "end": 13, "score": 0.95},
        {"text": "NVIDIA", "label": "company", "start": 17, "end": 23, "score": 0.88},
    ]


def test_merge_gliner2_entities_normalizes_confidence_and_spans() -> None:
    """GLiNER2's schema result becomes the detector's flat entity list."""
    adapter = load_adapter()

    entities = adapter.merge_entities(
        plugin="deberta_gliner2",
        chunks=[adapter.TextChunk("Email alice@example.com", 0)],
        results=[
            {
                "entities": {
                    "email": [
                        {
                            "text": "alice@example.com",
                            "start": 6,
                            "end": 23,
                            "confidence": 0.97,
                        }
                    ]
                }
            }
        ],
        flat_ner=False,
    )

    assert [entity.as_dict() for entity in entities] == [
        {
            "text": "alice@example.com",
            "label": "email",
            "start": 6,
            "end": 23,
            "score": 0.97,
        }
    ]


def test_split_text_matches_native_character_overlap_contract() -> None:
    """The adapter keeps the characterized character-based chunk semantics."""
    adapter = load_adapter()

    assert adapter.split_text("abcdefghij", chunk_length=6, overlap=2) == [
        adapter.TextChunk("abcdef", 0),
        adapter.TextChunk("efghij", 4),
    ]
