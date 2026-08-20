# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the vLLM Factory detector protocol adapter."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from inference_service_compiler import vllm_factory_adapter as adapter


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
            "messages": [{"role": "user", "content": "x" * 257}],
            "chunk_length": 1,
            "overlap": 0,
        }
    )

    assert request.labels == ()


def test_parse_detection_request_rejects_excessive_chunk_fanout() -> None:
    """One detector request cannot enqueue unbounded pooling work."""

    with pytest.raises(ValueError, match="at most 256 chunks"):
        adapter.parse_detection_request(
            {
                "model": "nvidia/gliner-pii",
                "messages": [{"role": "user", "content": "x" * 257}],
                "labels": ["person"],
                "chunk_length": 1,
                "overlap": 0,
            }
        )


def test_parse_detection_request_accepts_chunk_budget_boundary() -> None:
    """The admission cap includes requests with exactly 256 chunks."""

    request = adapter.parse_detection_request(
        {
            "model": "nvidia/gliner-pii",
            "messages": [{"role": "user", "content": "x" * 256}],
            "labels": ["person"],
            "chunk_length": 1,
            "overlap": 0,
        }
    )

    assert request.text == "x" * 256


def test_chat_compatibility_bounds_pooling_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepted chunks use a bounded worker frontier and preserve result order."""

    async def exercise() -> None:
        expected_peak = 8
        active = 0
        max_active = 0
        total_calls = 0
        reached_limit = asyncio.Event()
        release = asyncio.Event()

        async def fake_invoke_pooling(**kwargs: object) -> object:
            nonlocal active, max_active, total_calls
            active += 1
            total_calls += 1
            max_active = max(max_active, active)
            if active == expected_peak:
                reached_limit.set()
            try:
                await release.wait()
                text = kwargs["text"]
                assert isinstance(text, str)
                return [{"text": text, "label": "token", "start": 0, "end": 1, "score": 0.9}]
            finally:
                active -= 1

        async def request_json() -> dict[str, object]:
            return {
                "model": "nvidia/gliner-pii",
                "messages": [{"role": "user", "content": "abcdefghijklmnopq"}],
                "labels": ["token"],
                "chunk_length": 1,
                "overlap": 0,
            }

        async def call_next(_: object) -> None:
            raise AssertionError("detector requests must not reach the next middleware")

        monkeypatch.setattr(adapter, "invoke_pooling", fake_invoke_pooling)
        monkeypatch.setenv("ANONYMIZER_VLLM_FACTORY_PLUGIN", "deberta_gliner")
        request = SimpleNamespace(
            url=SimpleNamespace(path="/v1/chat/completions"),
            app=SimpleNamespace(state=SimpleNamespace(serving_pooling=object())),
            json=request_json,
        )

        operation = asyncio.create_task(adapter.anonymizer_chat_compatibility(request, call_next))
        await reached_limit.wait()
        await asyncio.sleep(0)
        observed_peak = max_active
        release.set()
        response = await operation

        assert response.status_code == 200
        assert observed_peak <= expected_peak
        assert total_calls == 17
        payload = json.loads(response.body)
        content = json.loads(payload["choices"][0]["message"]["content"])
        assert [entity["start"] for entity in content["entities"]] == list(range(17))

    asyncio.run(exercise())


def test_merge_gliner_entities_restores_offsets_and_deduplicates_overlap() -> None:
    """Chunk-relative factory spans become stable document offsets."""
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
    assert adapter.split_text("abcdefghij", chunk_length=6, overlap=2) == [
        adapter.TextChunk("abcdef", 0),
        adapter.TextChunk("efghij", 4),
    ]
