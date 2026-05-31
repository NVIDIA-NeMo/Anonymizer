# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for chunked LLM augmentation.

Pure helpers (chunker, tag re-anchoring, prompt rendering, merge dedupe) are
tested directly; the chunk dispatch is tested via a fake ``ModelFacade`` that
records calls and returns preconfigured ``AugmentedEntitiesSchema`` responses.
"""

from __future__ import annotations

from typing import Any

import pytest

from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TEXT,
)
from anonymizer.engine.detection.chunked_augmentation import (
    ChunkedAugmentationParams,
    build_chunk_tagged_text,
    chunk_text_at_line_boundaries,
    chunked_augment_row,
    make_chunked_augmentation_generator,
    merge_chunk_outputs,
    render_chunk_prompt,
)
from anonymizer.engine.detection.postprocess import EntitySpan, TagNotation
from anonymizer.engine.schemas import AugmentedEntitiesSchema


# ---------------------------------------------------------------------------
# chunk_text_at_line_boundaries
# ---------------------------------------------------------------------------


def test_chunk_empty_text():
    assert chunk_text_at_line_boundaries("", max_tokens=100) == [(0, "")]


def test_chunk_text_fits_in_one_chunk():
    text = "hello world"
    assert chunk_text_at_line_boundaries(text, max_tokens=100) == [(0, text)]


def test_chunk_splits_at_line_boundaries():
    # Each "line N" is small; use a tight token budget to force multiple chunks.
    text = "\n".join([f"line {i}" for i in range(20)]) + "\n"
    chunks = chunk_text_at_line_boundaries(text, max_tokens=10)
    assert len(chunks) > 1
    # Chunks must concatenate back to original text exactly.
    assert "".join(c for _, c in chunks) == text
    # Each chunk_offset must equal the cumulative length of preceding chunks.
    running = 0
    for off, c in chunks:
        assert off == running
        running += len(c)


def test_chunk_single_long_line_emits_as_one_chunk():
    # No newline anywhere → cannot split below 1 chunk even if it's over budget.
    text = "x" * 5000
    chunks = chunk_text_at_line_boundaries(text, max_tokens=10)
    assert len(chunks) == 1
    assert chunks[0] == (0, text)


# ---------------------------------------------------------------------------
# build_chunk_tagged_text — verifies seed re-anchoring to local coords
# ---------------------------------------------------------------------------


def test_build_chunk_tagged_text_anchors_seeds_locally():
    full = "alice met bob at noon\nlater she emailed carol\n"
    # "bob" at chars [10, 13) in full; chunk starts at offset 0 (first line only)
    chunk_offset = 0
    chunk_text = "alice met bob at noon\n"
    seeds = [
        EntitySpan(
            entity_id="bob-1", value="bob", label="first_name",
            start_position=10, end_position=13, score=1.0, source="gliner",
        ),
        # "carol" is in the second line — outside the chunk, should be dropped
        EntitySpan(
            entity_id="carol-1", value="carol", label="first_name",
            start_position=full.find("carol"), end_position=full.find("carol") + 5,
            score=1.0, source="gliner",
        ),
    ]
    tagged = build_chunk_tagged_text(
        chunk_text=chunk_text, chunk_offset=chunk_offset,
        all_seeds=seeds, notation=TagNotation.sentinel,
    )
    # "bob" should be wrapped in sentinel tags; "carol" must not appear
    # (it's not even in the chunk text)
    assert "bob" in tagged
    assert "first_name" in tagged
    assert "carol" not in tagged


def test_build_chunk_tagged_text_skips_seeds_outside_chunk():
    text = "filler-prefix\ntarget\nfiller-suffix\n"
    chunk_offset = text.index("target")
    chunk_text = "target\n"
    seeds = [
        # Seed BEFORE the chunk
        EntitySpan(entity_id="a", value="filler-prefix", label="x",
                   start_position=0, end_position=13, score=1.0, source="gliner"),
        # Seed AFTER the chunk
        EntitySpan(entity_id="b", value="filler-suffix", label="x",
                   start_position=text.index("filler-suffix"),
                   end_position=text.index("filler-suffix") + len("filler-suffix"),
                   score=1.0, source="gliner"),
    ]
    tagged = build_chunk_tagged_text(
        chunk_text=chunk_text, chunk_offset=chunk_offset,
        all_seeds=seeds, notation=TagNotation.sentinel,
    )
    # Neither out-of-chunk seed should leak into the tagged chunk text
    assert "filler-prefix" not in tagged
    assert "filler-suffix" not in tagged
    # The chunk text body remains
    assert "target" in tagged


# ---------------------------------------------------------------------------
# render_chunk_prompt — Jinja substitution sanity
# ---------------------------------------------------------------------------


def test_render_chunk_prompt_substitutes_all_placeholders():
    template = (
        "TEXT={{ _initial_tagged_text }}\n"
        "SEEDS={{ _seed_entities_json }}\n"
        "NOTATION={{ _tag_notation }}\n"
    )
    out = render_chunk_prompt(
        template=template,
        chunk_tagged_text="hello",
        seed_entities_json={"entities": []},
        notation=TagNotation.sentinel,
    )
    assert "TEXT=hello" in out
    assert "SEEDS={'entities': []}" in out
    assert f"NOTATION={TagNotation.sentinel.value}" in out


# ---------------------------------------------------------------------------
# merge_chunk_outputs — case-insensitive (value, label) dedupe
# ---------------------------------------------------------------------------


def _aug(entities: list[dict]) -> AugmentedEntitiesSchema:
    return AugmentedEntitiesSchema.model_validate({"entities": entities})


def test_merge_dedupes_case_insensitively():
    o1 = _aug([{"value": "Alice", "label": "first_name"}])
    o2 = _aug([{"value": "alice", "label": "first_name"},
               {"value": "Bob", "label": "first_name"}])
    merged = merge_chunk_outputs([o1, o2])
    values = sorted(e["value"] for e in merged["entities"])
    assert values == ["Alice", "Bob"]  # Alice wins (first-seen)


def test_merge_preserves_label_distinction():
    # Same value, different label → both retained
    o1 = _aug([{"value": "10", "label": "age"}])
    o2 = _aug([{"value": "10", "label": "count"}])
    merged = merge_chunk_outputs([o1, o2])
    pairs = sorted((e["value"], e["label"]) for e in merged["entities"])
    assert pairs == [("10", "age"), ("10", "count")]


def test_merge_empty_inputs():
    assert merge_chunk_outputs([]) == {"entities": []}
    assert merge_chunk_outputs([_aug([])]) == {"entities": []}


# ---------------------------------------------------------------------------
# chunked_augment_row — full dispatch with a fake facade
# ---------------------------------------------------------------------------


class _FakeFacade:
    """Records each generate() call; returns a preconfigured AugmentedEntitiesSchema per call."""

    def __init__(self, responses: list[AugmentedEntitiesSchema]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(self, *, prompt, parser, system_prompt, purpose):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "purpose": purpose})
        if not self._responses:
            raise RuntimeError("FakeFacade ran out of responses")
        return self._responses.pop(0), []  # (output, messages)


@pytest.fixture
def basic_row() -> dict[str, Any]:
    text = "\n".join([f"alice talks line {i}" for i in range(20)]) + "\n"
    return {
        COL_TEXT: text,
        COL_SEED_ENTITIES: {"entities": []},
        COL_SEED_ENTITIES_JSON: {"entities": []},
        COL_TAG_NOTATION: TagNotation.sentinel.value,
    }


def test_chunked_augment_row_writes_merged_output(basic_row):
    # Two-chunk row produced by tight max_tokens — fake facade returns one entity per chunk
    fake = _FakeFacade(
        responses=[
            _aug([{"value": "alice", "label": "first_name"}]),
            _aug([{"value": "alice", "label": "first_name"}]),  # duplicate
        ]
    )
    params = ChunkedAugmentationParams(
        alias="aug",
        chunk_tokens=10,  # forces multi-chunk on the 20-line text
        prompt_template="dummy",
    )
    # First produce the dispatcher's expected number of chunks; reuse chunker
    expected_chunks = chunk_text_at_line_boundaries(basic_row[COL_TEXT], max_tokens=10)
    # Re-init fake with one response per expected chunk
    fake = _FakeFacade(
        responses=[
            _aug([{"value": f"alice", "label": "first_name"}])
            for _ in expected_chunks
        ]
    )

    out = chunked_augment_row(dict(basic_row), params, {"aug": fake})

    # All chunks dispatched
    assert len(fake.calls) == len(expected_chunks)
    # All facade calls used a purpose tagged with the chunk index
    purposes = [c["purpose"] for c in fake.calls]
    assert all(p.startswith("entity-augmentation-chunk-") for p in purposes)
    # Output contains exactly one deduped 'alice' entity
    payload = out[COL_AUGMENTED_ENTITIES]
    assert payload["entities"] == [{"value": "alice", "label": "first_name", "reason": None}]


def test_chunked_augment_row_single_chunk_when_text_fits(basic_row):
    # Text is small enough that it fits in one chunk
    short_row = dict(basic_row, **{COL_TEXT: "alice and bob\n"})
    fake = _FakeFacade(responses=[_aug([{"value": "alice", "label": "first_name"}])])
    params = ChunkedAugmentationParams(
        alias="aug", chunk_tokens=1000, prompt_template="dummy",
    )
    out = chunked_augment_row(short_row, params, {"aug": fake})
    assert len(fake.calls) == 1
    assert out[COL_AUGMENTED_ENTITIES]["entities"] == [
        {"value": "alice", "label": "first_name", "reason": None}
    ]


def test_chunked_augment_row_raises_when_alias_missing(basic_row):
    params = ChunkedAugmentationParams(
        alias="missing_alias", chunk_tokens=1000, prompt_template="dummy",
    )
    with pytest.raises(KeyError, match="missing_alias"):
        chunked_augment_row(basic_row, params, {"other_alias": _FakeFacade([])})


# ---------------------------------------------------------------------------
# make_chunked_augmentation_generator — factory contract
# ---------------------------------------------------------------------------


def test_factory_rejects_empty_alias():
    with pytest.raises(ValueError, match="alias is empty"):
        make_chunked_augmentation_generator("")


def test_factory_returns_callable():
    gen = make_chunked_augmentation_generator("aug")
    assert callable(gen)
