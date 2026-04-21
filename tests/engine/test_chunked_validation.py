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
import re
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
from anonymizer.engine.detection.postprocess import EntitySpan, TagNotation, apply_validation_decisions
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

    Exposes a *sync* ``generate()`` method because ``_dispatch_chunk`` now
    calls ``asyncio.to_thread(facade.generate, ...)`` (see the module
    docstring for why the sync primitive is correct under both DD engines).
    Tests still drive the pipeline through ``asyncio.run(chunked_validate_row(...))``,
    which awaits the ``to_thread`` task that invokes this sync method in a
    worker thread.

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

    def generate(self, *, prompt, parser, system_prompt=None, purpose=None, **kwargs):
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

    def test_tuple_input_returns_list_of_lists(self) -> None:
        # Input-type tolerance: a sequence (not only a list) should work, and
        # the declared ``list[list[...]]`` return contract must hold even
        # when the caller hands in a tuple.
        ordered: tuple[tuple[int, None], ...] = tuple((i, None) for i in range(3))  # type: ignore[misc]
        chunks = chunk_candidates(ordered, max_entities_per_call=2)
        assert chunks == [[(0, None), (1, None)], [(2, None)]]
        assert all(isinstance(c, list) for c in chunks)


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
        assert (
            build_chunk_excerpt(text="x", chunk_spans=[], all_spans=[], window_chars=5, notation=TagNotation.xml) == ""
        )


class TestBuildChunkSkeleton:
    def test_skeleton_matches_chunk_only(self) -> None:
        candidates = _candidates_schema(("a", "Alice", "first_name"), ("b", "Bob", "first_name"))
        skeleton = build_chunk_skeleton([candidates.candidates[0]])
        assert skeleton == {
            "decisions": [
                {
                    "id": "a",
                    "value": "Alice",
                    "label": "first_name",
                    "decision": None,
                    "proposed_label": None,
                    "reason": None,
                }
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


_MINIMAL_TEMPLATE = "TAGGED:{{ _merged_tagged_text }}|SKELETON:{{ _validation_skeleton }}|NOTATION:{{ _tag_notation }}"


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

    def test_system_prompt_is_forwarded_to_facade(self) -> None:
        # ``ChunkedValidationParams.system_prompt`` must reach ``facade.generate``.
        # The recipe appends JSON task instructions before dispatch, so we assert
        # substring containment with a distinctive sentinel rather than equality.
        text = "Alice spoke."
        spans = [_entity_span("a", "Alice", "first_name", 0, 5)]
        candidates = _candidates_schema(("a", "Alice", "first_name"))
        row = _build_row(text=text, seed_entities=spans, candidates=candidates)

        facade = FakeFacade(
            "v0",
            response={"decisions": [{"id": "a", "decision": "keep", "proposed_label": "", "reason": "x"}]},
        )
        sentinel = "SYSPROMPT_SENTINEL_CHUNKED_VALIDATION_TEST"
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=10,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
            system_prompt=f"You are a validator. {sentinel}",
        )

        asyncio.run(chunked_validate_row(row, params, {"v0": facade}))

        assert len(facade.calls) == 1
        forwarded = facade.calls[0]["system_prompt"]
        assert forwarded is not None
        assert sentinel in forwarded

    def test_system_prompt_default_none_is_forwarded_untouched(self) -> None:
        # When the caller leaves ``system_prompt`` unset, nothing upstream of
        # the recipe should synthesize one. Whatever the recipe produces from
        # ``None`` is what the facade sees (today: ``None``). This test pins
        # that we don't accidentally inject a placeholder along the way.
        text = "Alice spoke."
        spans = [_entity_span("a", "Alice", "first_name", 0, 5)]
        candidates = _candidates_schema(("a", "Alice", "first_name"))
        row = _build_row(text=text, seed_entities=spans, candidates=candidates)

        facade = FakeFacade(
            "v0",
            response={"decisions": [{"id": "a", "decision": "keep", "proposed_label": "", "reason": "x"}]},
        )
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=10,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
            # system_prompt intentionally omitted (default None)
        )

        asyncio.run(chunked_validate_row(row, params, {"v0": facade}))

        assert len(facade.calls) == 1
        # The recipe maps ``None`` input to ``None`` output, so the facade
        # should receive no system prompt.
        assert facade.calls[0]["system_prompt"] is None


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
    def test_row_fails_only_when_every_pool_member_raises_for_a_chunk(self) -> None:
        """With failover enabled, a chunk only fails when *every* pool member raises.

        Downstream DD reporting then turns that row into a FailedRecord via
        ``NddAdapter._detect_missing_records`` — no unscrubbed passthrough.
        """
        spans = [
            _entity_span("a", "Alice", "first_name", 0, 5),
            _entity_span("b", "Bob", "first_name", 10, 13),
        ]
        candidates = _candidates_schema(("a", "Alice", "first_name"), ("b", "Bob", "first_name"))
        row = _build_row(text="Alice and Bob", seed_entities=spans, candidates=candidates)

        v0 = FakeFacade("v0", raise_on_call=True)
        v1 = FakeFacade("v1", raise_on_call=True)
        params = ChunkedValidationParams(
            pool=["v0", "v1"],
            max_entities_per_call=1,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        # The last alias tried propagates; order is round-robin + wrap, so
        # chunk 0 starts at v0 → fails → v1 (fails, last) → raises "from v1".
        with pytest.raises(RuntimeError, match="forced failure from v1"):
            asyncio.run(chunked_validate_row(row, params, {"v0": v0, "v1": v1}))


class TestChunkedValidateRowCrossAliasFailover:
    def test_primary_alias_failure_falls_over_to_secondary(self) -> None:
        """Primary raises, secondary in pool returns a valid response → chunk succeeds, row does not fail."""
        spans = [_entity_span("a", "Alice", "first_name", 0, 5)]
        candidates = _candidates_schema(("a", "Alice", "first_name"))
        row = _build_row(text="Alice", seed_entities=spans, candidates=candidates)

        v0 = FakeFacade("v0", raise_on_call=True)
        v1 = FakeFacade("v1", response={"decisions": [{"id": "a", "decision": "keep"}]})
        params = ChunkedValidationParams(
            pool=["v0", "v1"],
            max_entities_per_call=5,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        out = asyncio.run(chunked_validate_row(row, params, {"v0": v0, "v1": v1}))
        decisions = out[COL_VALIDATION_DECISIONS]["decisions"]
        assert len(decisions) == 1
        assert decisions[0]["id"] == "a"
        assert decisions[0]["decision"] == "keep"
        # Each alias was tried exactly once for this single-chunk row.
        assert len(v0.calls) == 1
        assert len(v1.calls) == 1

    def test_single_alias_pool_does_not_failover(self) -> None:
        """A one-alias pool has no fallback — one attempt, propagate immediately.

        This keeps the behavioural guarantee that pools of size 1 behave
        exactly as the pre-failover dispatch did: no hidden extra attempts.
        """
        spans = [_entity_span("a", "Alice", "first_name", 0, 5)]
        candidates = _candidates_schema(("a", "Alice", "first_name"))
        row = _build_row(text="Alice", seed_entities=spans, candidates=candidates)

        v0 = FakeFacade("v0", raise_on_call=True)
        params = ChunkedValidationParams(
            pool=["v0"],
            max_entities_per_call=5,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        with pytest.raises(RuntimeError, match="forced failure from v0"):
            asyncio.run(chunked_validate_row(row, params, {"v0": v0}))
        assert len(v0.calls) == 1

    def test_failover_wraps_starting_from_round_robin_primary(self) -> None:
        """With pool [v0,v1] and 3 chunks, chunk 1's primary is v1.

        If v1 fails, failover wraps to v0 (next position mod len(pool)).
        We verify by ensuring chunk 1 succeeds on v0's response, while
        chunk 0 and chunk 2 succeed on v0 (their primary) directly.
        """
        spans = [
            _entity_span("a", "Alice", "first_name", 0, 5),
            _entity_span("b", "Bob", "first_name", 10, 13),
            _entity_span("c", "Carol", "first_name", 20, 25),
        ]
        candidates = _candidates_schema(
            ("a", "Alice", "first_name"),
            ("b", "Bob", "first_name"),
            ("c", "Carol", "first_name"),
        )
        row = _build_row(text="Alice and Bob and Carol", seed_entities=spans, candidates=candidates)

        def v0_response(prompt: str) -> dict:
            # The chunk's skeleton is serialized into the prompt; pick the id
            # from there. We can't use the raw text excerpt to distinguish
            # chunks because the text is short enough that every chunk's
            # excerpt window covers the whole string.
            for candidate_id in ("a", "b", "c"):
                if f"'id': '{candidate_id}'" in prompt:
                    return {"decisions": [{"id": candidate_id, "decision": "keep"}]}
            raise AssertionError(f"no known candidate id found in prompt: {prompt!r}")

        v0 = FakeFacade("v0", response=v0_response)
        v1 = FakeFacade("v1", raise_on_call=True)
        params = ChunkedValidationParams(
            pool=["v0", "v1"],
            max_entities_per_call=1,
            excerpt_window_chars=50,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        out = asyncio.run(chunked_validate_row(row, params, {"v0": v0, "v1": v1}))
        decisions = {d["id"]: d["decision"] for d in out[COL_VALIDATION_DECISIONS]["decisions"]}
        assert decisions == {"a": "keep", "b": "keep", "c": "keep"}
        # v0 serviced all three chunks: chunk 0 + chunk 2 directly, chunk 1 via failover.
        assert len(v0.calls) == 3
        # v1 saw exactly one call — chunk 1's primary attempt that raised.
        assert len(v1.calls) == 1


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
        """The DD-exposed wrapper is *sync* (bridges DD's thread-pool fan-out
        to our async row logic via asyncio.run), so this test calls it
        directly and asserts it returns a dict, not a coroutine.
        """
        spans = [_entity_span("a", "Alice", "first_name", 0, 5)]
        candidates = _candidates_schema(("a", "Alice", "first_name"))
        row = _build_row(text="Alice", seed_entities=spans, candidates=candidates)
        fn = make_chunked_validation_generator(["v0"])
        facade = FakeFacade("v0", response={"decisions": [{"id": "a", "decision": "keep"}]})
        params = ChunkedValidationParams(
            pool=["v0"], max_entities_per_call=5, excerpt_window_chars=20, prompt_template=_MINIMAL_TEMPLATE
        )
        out = fn(row, params, {"v0": facade})
        assert isinstance(out, dict)
        assert out[COL_VALIDATION_DECISIONS]["decisions"][0]["id"] == "a"

    def test_generator_is_sync_callable_returning_dict(self) -> None:
        """Regression: DD's default thread-pool engine calls the wrapper
        synchronously and rejects coroutine returns with "must return a dict,
        got coroutine". Guard against accidentally re-introducing an async
        outer wrapper."""
        import asyncio as _asyncio
        import inspect as _inspect

        fn = make_chunked_validation_generator(["v0"])
        # The decorator wraps the function; unwrap and check the inner target.
        inner = _inspect.unwrap(fn)
        assert not _asyncio.iscoroutinefunction(inner), (
            "DD-exposed generator must be sync; an async outer wrapper breaks the default thread-pool engine."
        )


# ---------------------------------------------------------------------------
# Behavioral regression: single-chunk vs multi-chunk parity (pool-of-one).
# ---------------------------------------------------------------------------


def _selective_facade(alias: str, decisions_by_id: dict[str, dict[str, Any]]) -> FakeFacade:
    """Return a facade whose response depends on the ids embedded in each chunk's prompt.

    The prompt renders ``_validation_skeleton`` as Python ``str(dict)``; we
    parse ``'id': 'X'`` tokens back out to select which decisions to return.
    This makes the LLM behaviour a pure function of the *candidate ids in
    the chunk*, not of chunk shape or sequencing -- which is exactly the
    assumption a real deterministic validator would satisfy when re-run.
    """

    def respond(prompt: str) -> dict[str, Any]:
        ids = re.findall(r"'id':\s*'([^']+)'", prompt)
        return {"decisions": [decisions_by_id[i] for i in ids if i in decisions_by_id]}

    return FakeFacade(alias, response=respond)


def _summarize(validated: list[EntitySpan]) -> list[tuple[str, str]]:
    return [(e.entity_id, e.label) for e in validated]


def _tally(before: list[EntitySpan], after: list[EntitySpan], decisions: dict) -> dict[str, int]:
    """Count keep/reclass/drop/untouched relative to the decisions the LLM returned."""
    before_labels = {e.entity_id: e.label for e in before}
    after_ids = {e.entity_id for e in after}
    decided_ids: dict[str, dict[str, str]] = {d["id"]: d for d in decisions["decisions"]}
    counts = {"keep": 0, "reclass": 0, "drop": 0, "untouched": 0}
    for entity_id, original_label in before_labels.items():
        decision = decided_ids.get(entity_id)
        if decision is None:
            counts["untouched"] += 1
            continue
        verdict = decision.get("decision")
        if verdict == "drop":
            assert entity_id not in after_ids, f"drop decision for {entity_id} did not remove it"
            counts["drop"] += 1
        elif verdict == "reclass":
            counts["reclass"] += 1
        elif verdict == "keep":
            counts["keep"] += 1
    return counts


def _normalize_decisions(doc: dict) -> list[tuple[str, str, str]]:
    return sorted((d["id"], d.get("decision") or "", d.get("proposed_label") or "") for d in doc["decisions"])


class TestChunkedValidationRegression:
    """Partitioning must not change outcomes when the LLM is deterministic per candidate.

    Guards the most important property we promised when we switched from a
    single ``LLMStructuredColumnConfig`` to chunked ``CustomColumnConfig``
    dispatch: given the same set of per-id decisions, chunk sizing is an
    implementation detail of *how* we talk to the validator, not *what*
    entities survive validation.
    """

    SCENARIO_TEXT = (
        # Positions referenced below are into this exact string.
        "Alice met Bob in Chicago at Acme HQ; Doe introduced Eve to the team later."
        # 0    5    10   15  20      28    34       43       53  56
    )

    @pytest.fixture
    def scenario(self) -> tuple[list[EntitySpan], ValidationCandidatesSchema, dict[str, dict[str, Any]]]:
        spans = [
            _entity_span("a", "Alice", "first_name", 0, 5),
            _entity_span("b", "Bob", "first_name", 10, 13),
            _entity_span("c", "Chicago", "city", 17, 24),
            _entity_span("d", "Acme", "organization", 28, 32),
            _entity_span("e", "Doe", "last_name", 37, 40),
            _entity_span("f", "Eve", "first_name", 54, 57),
        ]
        candidates = _candidates_schema(
            ("a", "Alice", "first_name"),
            ("b", "Bob", "first_name"),
            ("c", "Chicago", "city"),
            ("d", "Acme", "organization"),
            ("e", "Doe", "last_name"),
            ("f", "Eve", "first_name"),
        )
        # Deterministic per-id decisions covering all branches.
        # ``f`` intentionally has no decision: downstream must keep it as-is,
        # regardless of whether it lands in its own chunk or shares one.
        decisions_by_id: dict[str, dict[str, Any]] = {
            "a": {"id": "a", "decision": "keep"},
            "b": {"id": "b", "decision": "drop"},
            "c": {"id": "c", "decision": "reclass", "proposed_label": "location"},
            "d": {"id": "d", "decision": "keep"},
            "e": {"id": "e", "decision": "reclass", "proposed_label": "surname"},
        }
        return spans, candidates, decisions_by_id

    def _run(
        self,
        *,
        spans: list[EntitySpan],
        candidates: ValidationCandidatesSchema,
        decisions_by_id: dict[str, dict[str, Any]],
        max_per_call: int,
    ) -> tuple[dict, list[EntitySpan], int]:
        row = _build_row(text=self.SCENARIO_TEXT, seed_entities=spans, candidates=candidates)
        facade = _selective_facade("solo", decisions_by_id)
        params = ChunkedValidationParams(
            pool=["solo"],
            max_entities_per_call=max_per_call,
            excerpt_window_chars=200,
            prompt_template=_MINIMAL_TEMPLATE,
        )
        out = asyncio.run(chunked_validate_row(row, params, {"solo": facade}))
        decisions_doc = out[COL_VALIDATION_DECISIONS]
        validated = apply_validation_decisions(spans, decisions_doc)
        return decisions_doc, validated, len(facade.calls)

    def test_multi_chunk_matches_single_chunk(self, scenario) -> None:
        spans, candidates, decisions_by_id = scenario

        # 1 chunk -- stands in for the "legacy single-call" path: all
        # candidates fit into one validator call, no partitioning.
        single_doc, single_validated, single_calls = self._run(
            spans=spans, candidates=candidates, decisions_by_id=decisions_by_id, max_per_call=10
        )
        # 3 chunks of size 2 -- pool-of-one with real partitioning.
        multi_doc, multi_validated, multi_calls = self._run(
            spans=spans, candidates=candidates, decisions_by_id=decisions_by_id, max_per_call=2
        )

        # Sanity: the two configurations actually differ in *how* they call
        # the validator. Without this the test is trivially satisfiable.
        assert single_calls == 1
        assert multi_calls == 3

        # Decisions merged back together are identical (order-insensitive).
        assert _normalize_decisions(single_doc) == _normalize_decisions(multi_doc)

        # Final per-entity outcomes (surviving ids + their post-validation
        # labels) are identical. This is the ``COL_DETECTED_ENTITIES`` parity
        # claim: downstream stages cannot tell the two runs apart.
        assert _summarize(single_validated) == _summarize(multi_validated)

        # Keep/reclass/drop/untouched tallies match the fixed decision set.
        expected_tally = {"keep": 2, "reclass": 2, "drop": 1, "untouched": 1}
        assert _tally(spans, single_validated, single_doc) == expected_tally
        assert _tally(spans, multi_validated, multi_doc) == expected_tally

        # Concrete post-validation outcome, pinned so a regression in
        # ``apply_validation_decisions`` or chunk merging is caught
        # precisely, not just "something changed".
        assert _summarize(multi_validated) == [
            ("a", "first_name"),  # keep
            ("c", "location"),  # reclass
            ("d", "organization"),  # keep
            ("e", "surname"),  # reclass
            ("f", "first_name"),  # untouched (no decision)
            # "b" dropped
        ]
