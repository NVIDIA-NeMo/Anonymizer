# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for chunked LLM validation.

The module is layered: pure helpers (ordering, chunking, excerpts, prompt
rendering, merging) are tested directly; the async dispatch is tested via a
fake ``ModelFacade`` that records calls and returns preconfigured responses.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import pytest

from anonymizer.engine.constants import (
    COL_MERGED_TAGGED_TEXT,
    COL_SEED_ENTITIES,
    COL_TAG_NOTATION,
    COL_TEXT,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_SKELETON,
)
from anonymizer.engine.detection.chunked_validation import (
    ChunkedValidationParams,
    build_chunk_excerpt,
    build_chunk_skeleton,
    chunk_candidates,
    chunked_validate_row,
    make_chunked_validation_generator,
    merge_chunk_decisions,
    order_candidates_by_position,
    render_chunk_prompt,
)
from anonymizer.engine.detection.postprocess import EntitySpan, TagNotation
from anonymizer.engine.schemas import (
    EntitiesSchema,
    RawValidationDecisionsSchema,
    ValidationCandidateSchema,
    ValidationCandidatesSchema,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class FakeFacade:
    """Test double for ``ModelFacade`` recording invocations and replaying canned responses.

    A canned response can be a ``dict`` (auto-wrapped in a ```json fence), a
    raw string, or a callable that receives the prompt and returns either.
    Set ``raise_on_call=True`` to simulate a terminal LLM failure.
    """

    def __init__(
        self,
        alias: str,
        response: dict | str | Callable[[str], dict | str] | None = None,
        *,
        raise_on_call: bool = False,
    ) -> None:
        self.alias = alias
        self._response = response
        self._raise = raise_on_call
        self.calls: list[dict[str, Any]] = []

    async def agenerate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "purpose": purpose,
                "kwargs": kwargs,
            }
        )
        if self._raise:
            raise RuntimeError(f"forced failure from {self.alias}")
        response = self._response
        if callable(response):
            response = response(prompt)
        raw = response if isinstance(response, str) else f"```json\n{json.dumps(response)}\n```"
        return parser(raw), []


def _entity_span(entity_id: str, value: str, label: str, start: int, end: int) -> EntitySpan:
    return EntitySpan(
        entity_id=entity_id,
        value=value,
        label=label,
        start_position=start,
        end_position=end,
        score=1.0,
        source="detector",
    )


def _candidates_schema(*candidates: tuple[str, str, str]) -> ValidationCandidatesSchema:
    return ValidationCandidatesSchema(
        candidates=[ValidationCandidateSchema(id=cid, value=val, label=lab) for cid, val, lab in candidates]
    )


# ---------------------------------------------------------------------------
# Pure helpers: ordering / chunking / excerpt / skeleton / prompt / merge
# ---------------------------------------------------------------------------


class TestOrderCandidatesByPosition:
    def test_orders_by_start_then_end_then_id(self) -> None:
        candidates = _candidates_schema(
            ("a_10_13", "foo", "first_name"),
            ("b_0_5", "bar", "email"),
            ("c_10_12", "baz", "city"),
        )
        spans = [
            _entity_span("a_10_13", "foo", "first_name", 10, 13),
            _entity_span("b_0_5", "bar", "email", 0, 5),
            _entity_span("c_10_12", "baz", "city", 10, 12),
        ]
        ordered = order_candidates_by_position(candidates, spans)
        assert [pair[0].id for pair in ordered] == ["b_0_5", "c_10_12", "a_10_13"]

    def test_missing_seed_raises_with_triage_hint(self) -> None:
        candidates = _candidates_schema(("missing", "x", "y"))
        with pytest.raises(ValueError, match="merge_and_build_candidates or prepare_validation_inputs"):
            order_candidates_by_position(candidates, [])


class TestChunkCandidates:
    def test_splits_into_chunks_of_at_most_n(self) -> None:
        ordered = [(i, None) for i in range(5)]  # type: ignore[misc]
        chunks = chunk_candidates(ordered, max_entities_per_call=2)
        assert [len(c) for c in chunks] == [2, 2, 1]

    def test_empty_input_yields_no_chunks(self) -> None:
        assert chunk_candidates([], max_entities_per_call=10) == []

    def test_non_positive_limit_raises(self) -> None:
        with pytest.raises(ValueError):
            chunk_candidates([(1, None)], max_entities_per_call=0)


class TestBuildChunkExcerpt:
    def test_includes_fully_contained_neighbors_and_drops_outside_spans(self) -> None:
        text = "Alice met Bob at Acme HQ in Seattle yesterday."
        spans = [
            _entity_span("alice_0_5", "Alice", "first_name", 0, 5),
            _entity_span("bob_10_13", "Bob", "first_name", 10, 13),
            _entity_span("acme_17_21", "Acme", "organization", 17, 21),
            _entity_span("seattle_28_35", "Seattle", "city", 28, 35),
        ]
        # Window 8 around Bob (10..13) => excerpt [2, 21]; contains Acme fully (17..21)
        # but truncates 'Alice' (0..5) and excludes 'Seattle' (28..35).
        excerpt = build_chunk_excerpt(
            text=text,
            chunk_spans=[spans[1]],
            all_spans=spans,
            window_chars=8,
            notation=TagNotation.xml,
        )
        assert "<first_name>Bob</first_name>" in excerpt
        assert "<organization>Acme</organization>" in excerpt
        # Alice is partially in the window (end=5 inside, but start=0 before) → excluded
        assert "first_name>Alice" not in excerpt
        assert "Seattle" not in excerpt

    def test_partially_contained_neighbor_is_excluded(self) -> None:
        """A neighbor that only partially overlaps the excerpt window must not be re-tagged.

        Tagging a truncated span would emit text that doesn't match the
        entity's actual value, which is worse than omitting the tag entirely.
        """
        text = "PREFIX Alice SUFFIX"  # Alice at 7..12
        spans = [_entity_span("a", "Alice", "first_name", 7, 12)]
        # Excerpt window [8, 12] cuts off 'A' from Alice → partial, must be excluded.
        excerpt = build_chunk_excerpt(
            text=text,
            chunk_spans=spans,
            all_spans=spans,
            window_chars=0,
            notation=TagNotation.xml,
        )
        # chunk_spans IS Alice (7..12), so its own span is within. This test
        # instead builds the partial case via a distinct chunk entity:
        bob = _entity_span("b", "lice", "first_name", 8, 12)
        chunk = [bob]
        excerpt2 = build_chunk_excerpt(
            text=text,
            chunk_spans=chunk,
            all_spans=[spans[0], bob],
            window_chars=0,
            notation=TagNotation.xml,
        )
        # 'Alice' at 7..12 starts before excerpt_start=8 → excluded.
        # 'bob' (the chunk entity) has positions 8..12 which are within the window → included.
        assert "<first_name>lice</first_name>" in excerpt2
        assert "Alice" not in excerpt2
        _ = excerpt  # suppress "unused" lint

    def test_forces_requested_notation_over_heuristic(self) -> None:
        text = "Alice met Bob at HQ"
        spans = [_entity_span("alice_0_5", "Alice", "first_name", 0, 5)]
        excerpt = build_chunk_excerpt(
            text=text,
            chunk_spans=spans,
            all_spans=spans,
            window_chars=100,
            notation=TagNotation.paren,
        )
        assert "((SENSITIVE:first_name|Alice))" in excerpt
        assert "<first_name>" not in excerpt

    def test_empty_chunk_returns_empty_string(self) -> None:
        assert build_chunk_excerpt(
            text="x", chunk_spans=[], all_spans=[], window_chars=5, notation=TagNotation.xml
        ) == ""


class TestBuildChunkSkeleton:
    def test_skeleton_matches_chunk_only(self) -> None:
        candidates = _candidates_schema(("a", "Alice", "first_name"), ("b", "Bob", "first_name"))
        skeleton = build_chunk_skeleton([candidates.candidates[0]])
        assert skeleton == {
            "decisions": [
                {"id": "a", "value": "Alice", "label": "first_name", "decision": None, "proposed_label": None, "reason": None}
            ]
        }


class TestRenderChunkPrompt:
    def test_substitutes_excerpt_skeleton_and_notation(self) -> None:
        template = (
            "Input: {{ _merged_tagged_text }}\n"
            "Skeleton: {{ _validation_skeleton }}\n"
            '{%- if _tag_notation == "xml" -%}notation-is-xml{%- endif -%}'
        )
        rendered = render_chunk_prompt(
            template=template,
            excerpt="hello <first_name>Alice</first_name>",
            skeleton={"decisions": [{"id": "a"}]},
            notation=TagNotation.xml,
        )
        assert "Input: hello <first_name>Alice</first_name>" in rendered
        assert "notation-is-xml" in rendered
        # Dict rendered via Python str(); this matches the existing production prompt path.
        assert "'id': 'a'" in rendered


class TestMergeChunkDecisions:
    def test_filters_unknown_ids_and_deduplicates(self) -> None:
        candidates = _candidates_schema(("a", "Alice", "first_name"), ("b", "Bob", "first_name"))
        chunk_one = RawValidationDecisionsSchema.model_validate(
            {"decisions": [{"id": "a", "decision": "keep"}, {"id": "ghost", "decision": "drop"}]}
        )
        chunk_two = RawValidationDecisionsSchema.model_validate(
            {"decisions": [{"id": "b", "decision": "drop"}, {"id": "a", "decision": "reclass"}]}
        )
        merged = merge_chunk_decisions([chunk_one, chunk_two], candidates)
        ids = [d["id"] for d in merged["decisions"]]
        assert ids == ["a", "b"]  # 'ghost' dropped; duplicate 'a' from chunk_two ignored
        by_id = {d["id"]: d for d in merged["decisions"]}
        assert by_id["a"]["decision"] == "keep"
        assert by_id["b"]["decision"] == "drop"
        assert by_id["a"]["value"] == "Alice"  # enriched from candidate
        assert by_id["a"]["label"] == "first_name"

    def test_drops_decisions_without_verdict(self) -> None:
        """A decision with ``decision=None`` is equivalent to 'no answer' and must not leak through.

        Downstream ``apply_validation_decisions`` interprets missing-id as
        'keep unchanged'. Emitting a null-decision entry would break that.
        """
        candidates = _candidates_schema(("a", "Alice", "first_name"))
        chunk = RawValidationDecisionsSchema.model_validate({"decisions": [{"id": "a", "decision": None}]})
        merged = merge_chunk_decisions([chunk], candidates)
        assert merged == {"decisions": []}


# ---------------------------------------------------------------------------
# Async dispatch: chunked_validate_row end-to-end with fake facades
# ---------------------------------------------------------------------------


def _build_row(
    *,
    text: str,
    seed_entities: list[EntitySpan],
    candidates: ValidationCandidatesSchema,
    notation: TagNotation = TagNotation.xml,
) -> dict[str, Any]:
    return {
        COL_TEXT: text,
        COL_SEED_ENTITIES: EntitiesSchema(
            entities=[
                {
                    "id": span.entity_id,
                    "value": span.value,
                    "label": span.label,
                    "start_position": span.start_position,
                    "end_position": span.end_position,
                    "score": span.score,
                    "source": span.source,
                }
                for span in seed_entities
            ]
        ).model_dump(mode="json"),
        COL_VALIDATION_CANDIDATES: candidates.model_dump(mode="json"),
        COL_TAG_NOTATION: notation.value,
    }


_MINIMAL_TEMPLATE = (
    "TAGGED:{{ _merged_tagged_text }}|"
    "SKELETON:{{ _validation_skeleton }}|"
    "NOTATION:{{ _tag_notation }}"
)


class TestChunkedValidateRowPoolOfOne:
    def test_single_chunk_single_alias_dispatches_once_and_merges(self) -> None:
        text = "Alice and Bob met."
        spans = [
            _entity_span("a", "Alice", "first_name", 0, 5),
            _entity_span("b", "Bob", "first_name", 10, 13),
        ]
        candidates = _candidates_schema(("a", "Alice", "first_name"), ("b", "Bob", "first_name"))
        row = _build_row(text=text, seed_entities=spans, candidates=candidates)

        facade = FakeFacade(
            "v0",
            response={
                "decisions": [
                    {"id": "a", "decision": "keep", "proposed_label": "", "reason": "real"},
                    {"id": "b", "decision": "drop", "proposed_label": "", "reason": "placeholder"},
                ]
            },
        )
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=10,
            excerpt_window_chars=100,
            prompt_template=_MINIMAL_TEMPLATE,
        )

        out = asyncio.run(chunked_validate_row(row, params, {"v0": facade}))

        assert len(facade.calls) == 1
        decisions = out[COL_VALIDATION_DECISIONS]["decisions"]
        assert {d["id"]: d["decision"] for d in decisions} == {"a": "keep", "b": "drop"}

    def test_empty_candidates_short_circuits_without_calls(self) -> None:
        row = _build_row(text="hello", seed_entities=[], candidates=_candidates_schema())
        facade = FakeFacade("v0", response={"decisions": []})
        params = ChunkedValidationParams(
            pool=["v0"], max_entities_per_call=10, excerpt_window_chars=50, prompt_template=_MINIMAL_TEMPLATE
        )

        out = asyncio.run(chunked_validate_row(row, params, {"v0": facade}))

        assert facade.calls == []
        assert out[COL_VALIDATION_DECISIONS] == {"decisions": []}


class TestChunkedValidateRowPoolOfTwoRoundRobin:
    def test_chunks_assigned_round_robin_across_pool(self) -> None:
        text = "A B C D E F" + " " * 50  # pad so excerpts don't overlap
        spans = [_entity_span(f"e{i}", chr(ord("A") + i), "first_name", i * 2, i * 2 + 1) for i in range(6)]
        candidates = _candidates_schema(*[(f"e{i}", chr(ord("A") + i), "first_name") for i in range(6)])
        row = _build_row(text=text, seed_entities=spans, candidates=candidates)

        def make_response(chunk_size: int) -> Callable[[str], dict]:
            def _respond(prompt: str) -> dict:
                # Parse out which entity ids are in this chunk from the skeleton.
                # We can't easily, so just return empty decisions — the dispatch
                # order assertion is about which alias was called, not contents.
                return {"decisions": []}

            return _respond

        v0 = FakeFacade("v0", response=make_response(2))
        v1 = FakeFacade("v1", response=make_response(2))
        params = ChunkedValidationParams(
            pool=["v0", "v1"],
            max_entities_per_call=2,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        asyncio.run(chunked_validate_row(row, params, {"v0": v0, "v1": v1}))
        # 6 candidates / 2 per chunk = 3 chunks; round-robin → v0,v1,v0
        assert len(v0.calls) == 2
        assert len(v1.calls) == 1


class TestChunkedValidateRowMultiChunkReassembly:
    def test_decisions_merged_across_chunks(self) -> None:
        text = "one two three four five"
        spans = [
            _entity_span("c1", "one", "first_name", 0, 3),
            _entity_span("c2", "two", "first_name", 4, 7),
            _entity_span("c3", "three", "first_name", 8, 13),
            _entity_span("c4", "four", "first_name", 14, 18),
        ]
        candidates = _candidates_schema(
            ("c1", "one", "first_name"),
            ("c2", "two", "first_name"),
            ("c3", "three", "first_name"),
            ("c4", "four", "first_name"),
        )
        row = _build_row(text=text, seed_entities=spans, candidates=candidates)

        def responder_for(alias: str) -> Callable[[str], dict]:
            def respond(prompt: str) -> dict:
                # Return a decision for every id mentioned in the skeleton portion of this prompt.
                # Use alias-encoded decisions so we can verify which chunk decided which id.
                ids_here = [cid for cid in ("c1", "c2", "c3", "c4") if f"'id': '{cid}'" in prompt]
                return {
                    "decisions": [
                        {"id": cid, "decision": "keep", "proposed_label": "", "reason": alias} for cid in ids_here
                    ]
                }

            return respond

        v0 = FakeFacade("v0", response=responder_for("v0"))
        v1 = FakeFacade("v1", response=responder_for("v1"))
        params = ChunkedValidationParams(
            pool=["v0", "v1"],
            max_entities_per_call=2,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        out = asyncio.run(chunked_validate_row(row, params, {"v0": v0, "v1": v1}))
        decisions = {d["id"]: d for d in out[COL_VALIDATION_DECISIONS]["decisions"]}
        assert set(decisions) == {"c1", "c2", "c3", "c4"}
        # Chunk 0 (c1,c2) → v0; chunk 1 (c3,c4) → v1.
        assert decisions["c1"]["reason"] == "v0"
        assert decisions["c2"]["reason"] == "v0"
        assert decisions["c3"]["reason"] == "v1"
        assert decisions["c4"]["reason"] == "v1"


class TestChunkedValidateRowFailurePropagation:
    def test_one_chunk_terminal_error_fails_whole_row(self) -> None:
        """A terminal LLM error in any chunk bubbles up; downstream DD reporting turns that into a FailedRecord."""
        spans = [
            _entity_span("a", "Alice", "first_name", 0, 5),
            _entity_span("b", "Bob", "first_name", 10, 13),
        ]
        candidates = _candidates_schema(("a", "Alice", "first_name"), ("b", "Bob", "first_name"))
        row = _build_row(text="Alice and Bob", seed_entities=spans, candidates=candidates)

        v0 = FakeFacade("v0", response={"decisions": [{"id": "a", "decision": "keep"}]})
        v1 = FakeFacade("v1", raise_on_call=True)
        params = ChunkedValidationParams(
            pool=["v0", "v1"],
            max_entities_per_call=1,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        with pytest.raises(RuntimeError, match="forced failure from v1"):
            asyncio.run(chunked_validate_row(row, params, {"v0": v0, "v1": v1}))


class TestChunkedValidateRowMissingIdMirrorsSingleCall:
    def test_decision_for_non_candidate_id_is_dropped(self) -> None:
        """Matches single-call contract: ``enrich_validation_decisions`` filters to candidate ids."""
        spans = [_entity_span("a", "Alice", "first_name", 0, 5)]
        candidates = _candidates_schema(("a", "Alice", "first_name"))
        row = _build_row(text="Alice", seed_entities=spans, candidates=candidates)
        facade = FakeFacade(
            "v0",
            response={
                "decisions": [
                    {"id": "a", "decision": "keep"},
                    {"id": "unknown", "decision": "drop"},
                ]
            },
        )
        params = ChunkedValidationParams(
            pool=["v0"], max_entities_per_call=5, excerpt_window_chars=20, prompt_template=_MINIMAL_TEMPLATE
        )
        out = asyncio.run(chunked_validate_row(row, params, {"v0": facade}))
        ids = [d["id"] for d in out[COL_VALIDATION_DECISIONS]["decisions"]]
        assert ids == ["a"]


class TestChunkedValidateRowGuardsBadConfig:
    def test_pool_alias_missing_from_models_raises_helpful_error(self) -> None:
        spans = [_entity_span("a", "Alice", "first_name", 0, 5)]
        candidates = _candidates_schema(("a", "Alice", "first_name"))
        row = _build_row(text="Alice", seed_entities=spans, candidates=candidates)
        params = ChunkedValidationParams(
            pool=["missing_alias"],
            max_entities_per_call=5,
            excerpt_window_chars=20,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        with pytest.raises(KeyError, match="missing_alias"):
            asyncio.run(chunked_validate_row(row, params, {"v0": FakeFacade("v0", response={"decisions": []})}))


# ---------------------------------------------------------------------------
# Factory: make_chunked_validation_generator
# ---------------------------------------------------------------------------


class TestMakeChunkedValidationGenerator:
    def test_decorator_metadata_encodes_pool_and_required_columns(self) -> None:
        fn = make_chunked_validation_generator(["v0", "v1"])
        meta = fn.custom_column_metadata
        assert meta["model_aliases"] == ["v0", "v1"]
        assert set(meta["required_columns"]) == {
            COL_TEXT,
            COL_SEED_ENTITIES,
            COL_VALIDATION_CANDIDATES,
            COL_TAG_NOTATION,
        }
        # Must not declare columns we deliberately don't read; an over-broad
        # required_columns would distort DAG ordering elsewhere.
        assert COL_VALIDATION_SKELETON not in meta["required_columns"]
        assert COL_MERGED_TAGGED_TEXT not in meta["required_columns"]

    def test_factory_rejects_empty_pool(self) -> None:
        with pytest.raises(ValueError, match="pool is empty"):
            make_chunked_validation_generator([])

    def test_generator_forwards_to_chunked_validate_row(self) -> None:
        spans = [_entity_span("a", "Alice", "first_name", 0, 5)]
        candidates = _candidates_schema(("a", "Alice", "first_name"))
        row = _build_row(text="Alice", seed_entities=spans, candidates=candidates)
        fn = make_chunked_validation_generator(["v0"])
        facade = FakeFacade("v0", response={"decisions": [{"id": "a", "decision": "keep"}]})
        params = ChunkedValidationParams(
            pool=["v0"], max_entities_per_call=5, excerpt_window_chars=20, prompt_template=_MINIMAL_TEMPLATE
        )
        out = asyncio.run(fn(row, params, {"v0": facade}))
        assert out[COL_VALIDATION_DECISIONS]["decisions"][0]["id"] == "a"
